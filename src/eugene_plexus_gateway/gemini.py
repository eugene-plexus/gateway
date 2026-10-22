"""Gemini's native wire translated onto the shared chat path (row 58)."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
import uuid
from collections.abc import Callable
from typing import Any
from urllib.parse import parse_qsl, urlencode

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError
from starlette.types import ASGIApp, Receive, Scope, Send

from . import chat_contract, door_paths
from ._generated.models import ChatCompletionRequest, GeminiGenerateContentRequest, GeminiModel

METHODS = ("generateContent", "streamGenerateContent", "countTokens")
BODY_PATHS = frozenset(
    f"/{version}/models/{{model:path}}:{method}"
    for version in ("v1", "v1beta")
    for method in METHODS
)
PATHS = BODY_PATHS | {"/v1beta/models", "/v1beta/models/{model:path}"}
_STATUS = {
    400: "INVALID_ARGUMENT",
    401: "UNAUTHENTICATED",
    403: "PERMISSION_DENIED",
    404: "NOT_FOUND",
    413: "INVALID_ARGUMENT",
    429: "RESOURCE_EXHAUSTED",
    499: "CANCELLED",
    500: "INTERNAL",
    502: "INTERNAL",
    503: "UNAVAILABLE",
    504: "DEADLINE_EXCEEDED",
}


def is_request(scope: Scope) -> bool:
    return (
        scope["path"].startswith("/v1beta/models")
        or door_paths.matches(scope["path"], BODY_PATHS)
        or (scope["path"].startswith("/v1/models") and has_key(scope))
    )


def has_key(scope: Scope) -> bool:
    return "gemini_query_key" in scope.get("state", {}) or any(
        k.lower() == b"x-goog-api-key" for k, _ in scope.get("headers", [])
    )


def error_body(status: int, message: str) -> dict[str, Any]:
    return {
        "error": {"code": status, "message": message, "status": _STATUS.get(status, "INTERNAL")}
    }


def error_response(status: int, message: str) -> JSONResponse:
    return JSONResponse(error_body(status, message), status_code=status)


class Refusal(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class _QueryKeyFilter(logging.Filter):
    """Also cover client HTTP logs which hold their own copy of the URL."""

    def filter(self, record: logging.LogRecord) -> bool:
        rendered = record.getMessage()
        record.msg = re.sub(r"([?&])key=[^&\s\"']*&?", r"\1", rendered, flags=re.I)
        record.msg = re.sub(r"\?(?=[\s\"']|$)", "", record.msg)
        record.args = ()
        return True


class QueryKeyRedaction:
    """Remove query credentials before access logs or downstream code see the URL."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        for name in ("uvicorn.access", "httpx", "httpcore"):
            logger = logging.getLogger(name)
            if not any(isinstance(f, _QueryKeyFilter) for f in logger.filters):
                logger.addFilter(_QueryKeyFilter())

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"].startswith(("/v1/models", "/v1beta/models")):
            pairs = parse_qsl(
                scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True
            )
            keys = [value for name, value in pairs if name.lower() == "key"]
            if keys:
                scope.setdefault("state", {})["gemini_query_key"] = keys[0]
                scope["query_string"] = urlencode(
                    [(k, v) for k, v in pairs if k.lower() != "key"]
                ).encode()
        await self.app(scope, receive, send)


async def authorize(request: Request) -> JSONResponse | None:
    from .dependencies import require_authorized

    key = request.headers.get("x-goog-api-key") or getattr(request.state, "gemini_query_key", None)
    header = request.headers.get("authorization", "")
    scheme, _, bearer = header.partition(" ")
    token = key or (bearer if scheme.lower() == "bearer" else None)
    try:
        await require_authorized(
            request,
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=token) if token else None,
        )
    except HTTPException as exc:
        if not is_request(request.scope):
            raise
        detail = exc.detail
        return error_response(
            exc.status_code,
            detail.get("detail", str(detail)) if isinstance(detail, dict) else str(detail),
        )
    return None


def resolve(model: str, table: Any, store: Any) -> str:
    model = model.removeprefix("models/")
    if table is not None and table.resolve(model).has_backends():
        return model
    for mapping in (store.get("geminiModelNames") or []) if store is not None else []:
        if mapping["model"] == model:
            return str(mapping["targets"][0])
    return "role:chat"


def _alias(obj: dict[str, Any], camel: str, snake: str) -> Any:
    if camel in obj and snake in obj and obj[camel] != obj[snake]:
        raise Refusal(f"{snake}: conflicts with {camel}.")
    return obj.get(camel, obj.get(snake))


def _schema(value: Any) -> Any:
    """Gemini schema type enums use uppercase; preserve arbitrary JSON schema keys."""
    if isinstance(value, list):
        return [_schema(v) for v in value]
    if isinstance(value, dict):
        return {
            k: v.lower() if k == "type" and isinstance(v, str) else _schema(v)
            for k, v in value.items()
        }
    return value


def translate(
    raw: Any, model: str, *, stream: bool, max_images: int
) -> tuple[ChatCompletionRequest, list[str], bool]:
    if not isinstance(raw, dict):
        raise Refusal("body: must be a JSON object.")
    try:
        parsed = GeminiGenerateContentRequest.model_validate(raw)
        chat_contract._check_objects(raw, parsed)
    except ValidationError as exc:
        loc = ".".join(str(x) for x in exc.errors()[0]["loc"])
        raise Refusal(f"{loc}: {exc.errors()[0]['msg']}.") from exc
    except chat_contract.Refusal as exc:
        raise Refusal(exc.message) from exc
    raw = parsed.model_dump(mode="json", exclude_none=True, exclude_unset=True)
    messages: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    ignored: list[str] = []
    if raw.get("safetySettings") is not None:
        ignored.append("safetySettings")
    if system := raw.get("systemInstruction"):
        if any(set(p) - {"text"} for p in system["parts"]):
            raise Refusal("systemInstruction.parts: only text is supported.")
        messages.append(
            {"role": "system", "content": "\n".join(p.get("text", "") for p in system["parts"])}
        )
    for turn in raw["contents"]:
        role = "assistant" if turn.get("role") == "model" else "user"
        parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning: list[str] = []
        for part in turn["parts"]:
            kinds = [
                k
                for k in ("text", "inlineData", "functionCall", "functionResponse")
                if part.get(k) is not None
            ]
            if len(kinds) != 1:
                raise Refusal("contents.parts: each part must carry exactly one content field.")
            if "thoughtSignature" in part:
                ignored.append("contents.parts.thoughtSignature")
            if "text" in part:
                if part.get("thought"):
                    reasoning.append(part["text"])
                else:
                    parts.append({"type": "text", "text": part["text"]})
            elif "inlineData" in part:
                inline = part["inlineData"]
                mime, data = inline["mimeType"], inline["data"]
                if mime.startswith("image/"):
                    parts.append(
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
                    )
                elif mime.startswith("audio/"):
                    fmt = {
                        "audio/wav": "wav",
                        "audio/x-wav": "wav",
                        "audio/mpeg": "mp3",
                        "audio/mp3": "mp3",
                        "audio/ogg": "ogg",
                        "audio/flac": "flac",
                    }.get(mime)
                    if fmt is None:
                        raise Refusal(f"inlineData.mimeType: unsupported audio type {mime!r}.")
                    parts.append(
                        {"type": "input_audio", "input_audio": {"format": fmt, "data": data}}
                    )
                elif mime == "application/pdf":
                    parts.append(
                        {
                            "type": "file",
                            "file": {
                                "file_data": f"data:{mime};base64,{data}",
                                "filename": "attachment.pdf",
                            },
                        }
                    )
                else:
                    raise Refusal(f"inlineData.mimeType: unsupported attachment type {mime!r}.")
            elif "functionCall" in part:
                if role != "assistant":
                    raise Refusal("functionCall: requires role model.")
                call = part["functionCall"]
                entry = {
                    "id": call.get("id") or f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
                }
                tool_calls.append(entry)
                calls.append(entry)
            else:
                if parts or tool_calls:
                    messages.append(
                        {
                            "role": role,
                            "content": parts or None,
                            **({"tool_calls": tool_calls} if tool_calls else {}),
                        }
                    )
                    parts, tool_calls = [], []
                result = part["functionResponse"]
                matches = [
                    c
                    for c in calls
                    if c["function"]["name"] == result["name"]
                    and (not result.get("id") or result["id"] == c["id"])
                ]
                if not matches or (result.get("id") and len(matches) != 1):
                    raise Refusal(
                        "functionResponse: no unique preceding functionCall; supply its id."
                    )
                matched = matches[0]
                calls.remove(matched)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": matched["id"],
                        "content": json.dumps(result["response"]),
                    }
                )
        if parts or tool_calls or reasoning:
            content: Any = parts
            if parts and all(p["type"] == "text" for p in parts):
                content = "\n".join(p["text"] for p in parts)
            messages.append(
                {
                    "role": role,
                    "content": content or None,
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                    **({"reasoning_content": "".join(reasoning)} if reasoning else {}),
                }
            )
    body: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    declarations: list[dict[str, Any]] = []
    for tool in raw.get("tools") or []:
        if tool.get("googleSearch") is not None:
            if tool["googleSearch"]:
                raise Refusal("tools.googleSearch: options are not supported.")
            body["web_search_options"] = {}
        for declaration in tool.get("functionDeclarations") or []:
            parameters = _alias(declaration, "parametersJsonSchema", "parameters_json_schema")
            if declaration.get("parameters") is not None and parameters is not None:
                raise Refusal("parametersJsonSchema: conflicts with parameters.")
            declarations.append(
                {
                    "type": "function",
                    "function": {
                        "name": declaration["name"],
                        "description": declaration.get("description"),
                        "parameters": _schema(
                            parameters
                            if parameters is not None
                            else declaration.get("parameters", {"type": "object"})
                        ),
                    },
                }
            )
    config = (raw.get("toolConfig") or {}).get("functionCallingConfig") or {}
    allowed = config.get("allowedFunctionNames")
    if allowed is not None:
        known = {t["function"]["name"] for t in declarations}
        if not allowed or set(allowed) - known:
            raise Refusal("allowedFunctionNames: must name declared functions.")
        declarations = [t for t in declarations if t["function"]["name"] in allowed]
    if declarations:
        body["tools"] = declarations
    mode = config.get("mode")
    if mode:
        body["tool_choice"] = {"AUTO": "auto", "ANY": "required", "NONE": "none"}[mode]
        if mode == "ANY" and allowed and len(allowed) == 1:
            body["tool_choice"] = {"type": "function", "function": {"name": allowed[0]}}
    gen = raw.get("generationConfig") or {}
    for native, shared in {
        "temperature": "temperature",
        "topP": "top_p",
        "topK": "top_k",
        "maxOutputTokens": "max_tokens",
        "stopSequences": "stop",
        "seed": "seed",
    }.items():
        if gen.get(native) is not None:
            body[shared] = gen[native]
    mime = gen.get("responseMimeType")
    if mime not in (None, "text/plain", "application/json"):
        raise Refusal("responseMimeType: only text/plain and application/json are supported.")
    schema = gen.get("responseJsonSchema", gen.get("responseSchema"))
    if gen.get("responseJsonSchema") is not None and gen.get("responseSchema") is not None:
        raise Refusal("responseJsonSchema: conflicts with responseSchema.")
    if schema is not None and mime != "application/json":
        raise Refusal("responseSchema: requires responseMimeType application/json.")
    if mime == "application/json":
        body["response_format"] = (
            {
                "type": "json_schema",
                "json_schema": {"name": "gemini_response", "schema": _schema(schema)},
            }
            if schema is not None
            else {"type": "json_object"}
        )
    thinking = gen.get("thinkingConfig") or {}
    level = _alias(thinking, "thinkingLevel", "thinking_level")
    budget = _alias(thinking, "thinkingBudget", "thinking_budget")
    thoughts = bool(_alias(thinking, "includeThoughts", "include_thoughts"))
    if level is not None and budget is not None:
        raise Refusal("thinkingConfig: thinkingLevel conflicts with thinkingBudget.")
    if level is not None:
        if level.lower() not in ("low", "medium", "high"):
            raise Refusal("thinkingLevel: must be LOW, MEDIUM or HIGH.")
        body["reasoning_effort"] = level.lower()
    if budget is not None and budget != -1:
        body["reasoning_effort"] = (
            "none"
            if budget == 0
            else "low"
            if budget <= 1024
            else "medium"
            if budget <= 8192
            else "high"
        )
        if budget > 0:
            ignored.append("thinkingConfig.thinkingBudget")
    try:
        return chat_contract.parse_request(body, max_images=max_images), ignored, thoughts
    except chat_contract.Refusal as exc:
        raise Refusal(exc.message) from exc


def parts_of(response: Any, thoughts: bool, *, text: bool = True) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if thoughts and response.reasoning and text:
        parts.append({"text": response.reasoning, "thought": True})
    if response.content and text:
        parts.append({"text": response.content})
    for call in response.toolCalls or []:
        try:
            args = json.loads(call.function.arguments)
        except (ValueError, TypeError) as exc:
            raise Refusal(
                f"functionCall.args: backend returned invalid JSON for {call.function.name}."
            ) from exc
        if not isinstance(args, dict):
            raise Refusal("functionCall.args: backend returned a value that is not an object.")
        parts.append({"functionCall": {"id": call.id, "name": call.function.name, "args": args}})
    return parts


def response_of(response: Any, model: str, thoughts: bool, *, text: bool = True) -> dict[str, Any]:
    finish = {
        "stop": "STOP",
        "tool_calls": "STOP",
        "length": "MAX_TOKENS",
        "content_filter": "SAFETY",
    }.get(response.finishReason.value)
    if finish is None:
        raise Refusal(f"finishReason: backend ended with {response.finishReason.value}.")
    candidate: dict[str, Any] = {"index": 0, "finishReason": finish}
    parts = parts_of(response, thoughts, text=text)
    if parts:
        candidate["content"] = {"role": "model", "parts": parts}
    result: dict[str, Any] = {
        "candidates": [candidate],
        "modelVersion": response.modelId or model,
        "usageMetadata": {},
    }
    usage = response.usage
    if usage:
        mapped = {
            "promptTokenCount": usage.promptTokens,
            "totalTokenCount": usage.totalTokens,
            "thoughtsTokenCount": usage.reasoningTokens,
            "cachedContentTokenCount": usage.cachedPromptTokens,
        }
        if usage.completionTokens is not None:
            mapped["candidatesTokenCount"] = max(
                0, usage.completionTokens - (usage.reasoningTokens or 0)
            )
        result["usageMetadata"] = {k: v for k, v in mapped.items() if v is not None}
    return result


class StreamResponse(StreamingResponse):
    """Resolve streaming headers at the first output, after failover has committed."""

    def __init__(self, content: Any, headers: Callable[[], dict[str, str]]) -> None:
        super().__init__(content, media_type="text/event-stream")
        self._served_headers = headers

    async def stream_response(self, send: Send) -> None:
        iterator = aiter(self.body_iterator)
        try:
            first = await anext(iterator, None)
            self.headers.update(self._served_headers())
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                }
            )
            if first is not None:
                await send(
                    {
                        "type": "http.response.body",
                        "body": first.encode() if isinstance(first, str) else first,
                        "more_body": True,
                    }
                )
            async for chunk in iterator:
                await send(
                    {
                        "type": "http.response.body",
                        "body": chunk.encode() if isinstance(chunk, str) else chunk,
                        "more_body": True,
                    }
                )
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()


def project(model: Any, *, name: str | None = None) -> GeminiModel:
    info = model.x_eugene_plexus
    result = GeminiModel(
        name=f"models/{name or model.id}",
        displayName=name or model.id,
        supportedGenerationMethods=list(METHODS)
        if info is None or not info.surfaces or "chat" in info.surfaces
        else [],
    )
    if info:
        result.inputTokenLimit = info.context_length
    return result


def page(request: Request, models: list[Any]) -> tuple[list[Any], str | None]:
    raw_size = request.query_params.get("pageSize", "1000")
    if not raw_size.isascii() or not raw_size.isdigit() or not 1 <= int(raw_size) <= 1000:
        raise Refusal("pageSize: must be an integer from 1 through 1000.")
    size = int(raw_size)
    token = request.query_params.get("pageToken")
    offset = 0
    if token:
        try:
            decoded = base64.b64decode(
                token + "=" * (-len(token) % 4), altchars=b"-_", validate=True
            ).decode("ascii")
            prefix, raw = decoded.split(":", 1)
            if prefix != "row58" or not raw.isdigit():
                raise ValueError
            offset = int(raw)
            if offset >= len(models):
                raise ValueError
        except (ValueError, UnicodeError, binascii.Error) as exc:
            raise Refusal("pageToken: invalid or outside the current model listing.") from exc
    end = offset + size
    next_token = (
        base64.urlsafe_b64encode(f"row58:{end}".encode()).decode().rstrip("=")
        if end < len(models)
        else None
    )
    return models[offset:end], next_token
