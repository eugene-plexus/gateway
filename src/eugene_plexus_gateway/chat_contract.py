"""Validate public chat settings before generated models can discard them."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, RootModel, ValidationError

from ._generated.models import ChatCompletionRequest
from .images import ImageRefusal, validate_messages


class Refusal(Exception):
    def __init__(self, field: str, reason: str = "is not supported") -> None:
        self.field = field
        self.message = f"{field}: {reason}."
        super().__init__(self.message)


def _field_name(name: str) -> str:
    # Unknown property names are caller input too. Never reflect arbitrary prose,
    # URLs, control characters or long strings in an error response.
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,63}", name) else "unknown_field"


def _check_objects(raw: Any, parsed: Any, path: str = "") -> None:
    """Check typed objects, leaving arbitrary tool/response JSON Schemas intact."""
    if isinstance(parsed, RootModel):
        _check_objects(raw, parsed.root, path)
    elif isinstance(parsed, BaseModel) and isinstance(raw, dict):
        fields = {field.alias or name: name for name, field in type(parsed).model_fields.items()}
        for key, value in raw.items():
            where = f"{path}.{_field_name(key)}" if path else _field_name(key)
            if key not in fields:
                raise Refusal(where)
            _check_objects(value, getattr(parsed, fields[key]), where)
    elif isinstance(raw, list) and isinstance(parsed, list):
        for index, (value, item) in enumerate(zip(raw, parsed, strict=True)):
            _check_objects(value, item, f"{path}[{index}]")


def parse_request(raw: Any) -> ChatCompletionRequest:
    if not isinstance(raw, dict):
        raise Refusal("body", "must be a JSON object")
    body = dict(raw)
    for field in ("max_tokens", "max_completion_tokens"):
        value = body.get(field)
        if value is not None and (type(value) is not int or value < 1):
            raise Refusal(field, "must be a positive JSON integer or null")
    old, new = body.get("max_tokens"), body.pop("max_completion_tokens", None)
    if old is not None and new is not None and old != new:
        raise Refusal(
            "max_completion_tokens", "conflicts with max_tokens; use one limit or equal values"
        )
    if new is not None:
        body["max_tokens"] = new

    # Opaque client annotations do not control inference or establish identity.
    # They are deliberately neither stored nor forwarded as provider metadata.
    for field in ("metadata", "safety_identifier"):
        value = body.pop(field, None)
        if value is not None:
            if field == "metadata":
                if not isinstance(value, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in value.items()
                ):
                    raise Refusal(field, "must be an object of strings or null")
            elif not isinstance(value, str):
                raise Refusal(field, "must be a string or null")

    # Common SDK defaults with exactly the behavior this endpoint provides.
    for field, neutral in {"n": 1, "logprobs": False, "store": False}.items():
        value = body.pop(field, None)
        if value is not None and (type(value) is not type(neutral) or value != neutral):
            raise Refusal(field, f"only {str(neutral).lower()} is supported")

    if isinstance(body.get("stop"), str):
        body["stop"] = [body["stop"]]
    options = body.get("stream_options")
    if isinstance(options, dict):
        if "include_usage" in options and type(options["include_usage"]) is not bool:
            raise Refusal("stream_options.include_usage", "must be a boolean")
        if body.get("stream") is not True:
            raise Refusal("stream_options", "requires stream true")
    try:
        parsed = ChatCompletionRequest.model_validate(body)
    except ValidationError as exc:
        # Do not return Pydantic's input/ctx or interpolate the invalid value.
        parts = exc.errors(include_input=False, include_context=False)[0]["loc"]
        where = ".".join(_field_name(p) if isinstance(p, str) else str(p) for p in parts)
        raise Refusal(where or "body", "has an invalid or missing value") from None
    _check_objects(body, parsed)
    if parsed.response_format is not None:
        fmt = parsed.response_format
        if fmt.type.value == "json_schema" and fmt.json_schema is None:
            raise Refusal("response_format.json_schema", "is required for json_schema output")
        if fmt.type.value != "json_schema" and fmt.json_schema is not None:
            raise Refusal("response_format.json_schema", "requires type json_schema")
    try:
        validate_messages(parsed.messages)
    except ImageRefusal as exc:
        raise Refusal(exc.field, exc.reason) from None
    return parsed


def request_body_schema() -> dict[str, Any]:
    """Keep FastAPI's documentation while parsing raw JSON for safe errors."""
    schema = ChatCompletionRequest.model_json_schema()
    definitions = schema.pop("$defs", {})

    def inline(value: Any) -> Any:
        if isinstance(value, dict):
            if "$ref" in value:
                return inline(definitions[value["$ref"].rsplit("/", 1)[-1]])
            return {key: inline(item) for key, item in value.items()}
        if isinstance(value, list):
            return [inline(item) for item in value]
        return value

    return {
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": inline(schema)}},
        }
    }
