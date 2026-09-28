"""Validate public chat settings before generated models can discard them."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, RootModel, ValidationError

from ._generated.models import ChatCompletionRequest, SpeechRequest
from .images import DEFAULT_MAX_IMAGES, ImageRefusal, validate_messages


class Refusal(Exception):
    def __init__(self, field: str, reason: str = "is not supported") -> None:
        self.field = field
        self.message = f"{field}: {reason}."
        super().__init__(self.message)


def _field_name(name: str) -> str:
    # Unknown property names are caller input too. Never reflect arbitrary prose,
    # URLs, control characters or long strings in an error response.
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,63}", name) else "unknown_field"


def _most_specific(errors: list[Any]) -> tuple[Any, ...]:
    """The location of the error the caller meant, in their own field names.

    A content part is a union, and Pydantic reports one error per branch:
    the first is `content.str` whatever was wrong, which named no field at
    all when an `input_audio` carried a bad `format` (2026-09-28). The
    branch whose `type` literal matched is the one the caller wrote, so
    the others are set aside and the deepest remaining error is named,
    without the generated class names Pydantic threads through the path.
    """
    mismatched = [
        e["loc"][:-1] for e in errors if e["type"] == "literal_error" and e["loc"][-1] == "type"
    ]
    meant = [e for e in errors if not any(e["loc"][: len(b)] == b for b in mismatched)]
    loc = max(meant or errors, key=lambda e: len(e["loc"]))["loc"]
    return tuple(
        p for p in loc if not (isinstance(p, str) and (re.fullmatch(r"[A-Z]\w*", p) or "[" in p))
    )


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


def parse_request(raw: Any, *, max_images: int = DEFAULT_MAX_IMAGES) -> ChatCompletionRequest:
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
    # `metadata` is deliberately neither stored nor forwarded as provider
    # metadata. `safety_identifier` was popped here too until P2c
    # (2026-09-28); it is a hint carried to OpenAI's own API now, validated
    # by the schema like any other field.
    value = body.pop("metadata", None)
    if value is not None and (
        not isinstance(value, dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items())
    ):
        raise Refusal("metadata", "must be an object of strings or null")

    # Common SDK defaults with exactly the behavior this endpoint provides.
    for field, neutral in {"n": 1, "store": False}.items():
        value = body.pop(field, None)
        if value is not None and (type(value) is not type(neutral) or value != neutral):
            raise Refusal(field, f"only {str(neutral).lower()} is supported")

    if isinstance(body.get("stop"), str):
        body["stop"] = [body["stop"]]
    options = body.get("stream_options")
    if isinstance(options, dict):
        for flag in ("include_usage", "include_progress"):
            if flag in options and type(options[flag]) is not bool:
                raise Refusal(f"stream_options.{flag}", "must be a boolean")
        if body.get("stream") is not True:
            raise Refusal("stream_options", "requires stream true")
    _translate_functions(body)
    # Before the schema, so `{"id": ...}` -- the shape OpenAI takes here --
    # is told why rather than which of the response's fields it lacks.
    for index, message in enumerate(body.get("messages") or []):
        if isinstance(message, dict) and "audio" in message:
            raise Refusal(
                f"messages[{index}].audio",
                "names stored audio, and this install keeps none; send the "
                "transcript as content instead",
            )
    try:
        parsed = ChatCompletionRequest.model_validate(body)
    except ValidationError as exc:
        # Do not return Pydantic's input/ctx or interpolate the invalid value.
        parts = _most_specific(exc.errors(include_input=False, include_context=False))
        where = ".".join(_field_name(p) if isinstance(p, str) else str(p) for p in parts)
        raise Refusal(where or "body", "has an invalid or missing value") from None
    _check_objects(body, parsed)
    if parsed.response_format is not None:
        fmt = parsed.response_format
        if fmt.type.value == "json_schema" and fmt.json_schema is None:
            raise Refusal("response_format.json_schema", "is required for json_schema output")
        if fmt.type.value != "json_schema" and fmt.json_schema is not None:
            raise Refusal("response_format.json_schema", "requires type json_schema")
    # Only a model's own turn has reasoning to hand back. Anywhere else it
    # has nowhere to go, and dropping text that changes the prompt is the
    # silent discard A2 exists to refuse.
    for index, message in enumerate(parsed.messages):
        if message.reasoning_content is not None and message.role.value != "assistant":
            raise Refusal(
                f"messages[{index}].reasoning_content", "is accepted only on assistant messages"
            )
    try:
        validate_messages(parsed.messages, max_images)
    except ImageRefusal as exc:
        raise Refusal(exc.field, exc.reason) from None
    _check_audio_output(parsed)
    if parsed.top_logprobs is not None and parsed.logprobs is not True:
        raise Refusal("top_logprobs", "requires logprobs true")
    return parsed


def parse_speech(raw: Any) -> SpeechRequest:
    """The speech door's body (P3a), refused naming the field, as chat is."""
    if not isinstance(raw, dict):
        raise Refusal("body", "must be a JSON object")
    try:
        parsed = SpeechRequest.model_validate(raw)
    except ValidationError as exc:
        parts = _most_specific(exc.errors(include_input=False, include_context=False))
        where = ".".join(_field_name(p) if isinstance(p, str) else str(p) for p in parts)
        raise Refusal(where or "body", "has an invalid or missing value") from None
    if parsed.stream_format is not None and parsed.stream_format.value == "sse":
        raise Refusal(
            "stream_format", 'only "audio" is served: the audio itself is streamed as raw bytes'
        )
    return parsed


#: OpenAI's upload limit (P3b). The body limit is a little larger, for the
#: form's other fields; this is the file itself.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

#: Every field of OpenAI's transcription form, and what this door does with
#: it. A field not here is refused naming it, as an unknown JSON field is.
_CARRIED = {
    "file",
    "model",
    "language",
    "prompt",
    "response_format",
    "temperature",
    "timestamp_granularities[]",
    "stream",
}
_NOT_CARRIED = {
    "chunking_strategy": "is not carried: the backend decides how to split the audio",
    "include[]": "is not carried: no backend here returns transcription logprobs",
    "known_speaker_names[]": "is not carried: no backend here labels speakers",
    "known_speaker_references[]": "is not carried: no backend here labels speakers",
}

#: The OpenAI SDK's translation form is five fields (P3-4). Three of the
#: transcription form's others are named when sent, since each has a reason.
_TRANSLATION_CARRIED = {"file", "model", "prompt", "response_format", "temperature"}
_NOT_TRANSLATED = {
    "language": "is not taken by a translation: the text is always English",
    "timestamp_granularities[]": "is not taken by a translation, as OpenAI's is not",
    "stream": "is not served: the answer is one JSON document",
}


class TooLarge(Refusal):
    """A refusal that is a 413, not a 400."""


@dataclass(frozen=True)
class TranscriptionAsk:
    """The transcription form, read (P3b)."""

    model: str
    audio: bytes
    filename: str
    media_type: str | None
    response_format: str
    language: str | None
    prompt: str | None
    temperature: float | None
    granularities: list[str]
    #: `/v1/audio/translations` (P3-4): the text in English.
    translate: bool = False


async def read_transcription(form: Any, *, translate: bool = False) -> TranscriptionAsk:
    """OpenAI's multipart form, refused naming the field, as chat is. With
    `translate`, the translation form: the same minus the language and
    timestamps, which a translation does not take."""
    for key in form:
        if translate:
            if key in _NOT_TRANSLATED:
                raise Refusal(key.rstrip("[]"), _NOT_TRANSLATED[key])
            if key not in _TRANSLATION_CARRIED:
                raise Refusal(_field_name(key), "is not a field of this form")
            continue
        if key in _NOT_CARRIED:
            raise Refusal(key.rstrip("[]"), _NOT_CARRIED[key])
        if key not in _CARRIED:
            raise Refusal(_field_name(key), "is not a field of this form")

    def text(key: str, *, limit: int) -> str | None:
        value = form.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise Refusal(key, "must be a text field")
        if len(value) > limit:
            raise Refusal(key, f"is longer than {limit} characters")
        return value

    upload = form.get("file")
    if upload is None or isinstance(upload, str):
        raise Refusal("file", "is required: the audio, as a multipart file")
    model = text("model", limit=512)
    if not model:
        raise Refusal("model", "is required")
    fmt = text("response_format", limit=32) or "json"
    if fmt in ("srt", "vtt"):
        raise Refusal(
            "response_format", "srt and vtt are not made here; ask for json, text or verbose_json"
        )
    if fmt not in ("json", "text", "verbose_json"):
        raise Refusal("response_format", "must be json, text or verbose_json")
    stream = text("stream", limit=8)
    if stream is not None and stream.lower() not in ("false", "0"):
        raise Refusal("stream", "is not served: the answer is one JSON document")
    temperature: float | None = None
    raw = text("temperature", limit=32)
    if raw is not None:
        try:
            temperature = float(raw)
        except ValueError:
            raise Refusal("temperature", "must be a number") from None
        if not 0 <= temperature <= 1:
            raise Refusal("temperature", "must be between 0 and 1")
    granularities = [str(g) for g in form.getlist("timestamp_granularities[]")]
    if any(g not in ("word", "segment") for g in granularities):
        raise Refusal("timestamp_granularities", "must be word or segment")
    if granularities and fmt != "verbose_json":
        raise Refusal(
            "timestamp_granularities", "needs response_format verbose_json, as OpenAI's does"
        )
    audio = await upload.read()
    if len(audio) > MAX_UPLOAD_BYTES:
        raise TooLarge("file", f"is {len(audio)} bytes; the limit is 25 MiB, as OpenAI's is")
    if not audio:
        raise Refusal("file", "is empty")
    return TranscriptionAsk(
        model=model,
        audio=audio,
        filename=(upload.filename or "audio")[:255],
        media_type=upload.content_type or None,
        response_format=fmt,
        language=text("language", limit=16) or None,
        prompt=text("prompt", limit=8192) or None,
        temperature=temperature,
        granularities=granularities,
        translate=translate,
    )


def uses_functions(parsed: ChatCompletionRequest) -> bool:
    """Whether the caller used the deprecated `functions`, and so reads the
    answer in the deprecated shape (P2c)."""
    return parsed.functions is not None


def _translate_functions(body: dict[str, Any]) -> None:
    """The deprecated `functions`, `function_call` and `function` role, as
    tools, in the raw body before the schema reads it (P2c).

    OpenAI deprecated them for tools, which are the same thing with every
    tool a function; clients built on the old shape still send it. So they
    are carried rather than refused, and `functions` stays on the request
    as the mark that the answer goes back in the old shape. The old API
    gave one call per turn; if a model asks for more, the first is given.

    **`parallel_tool_calls` is not sent for it.** A first version set it
    false, which made it an explicit setting, and A2 then routed the
    request only to models listing `parallel_tool_calls` -- 12 of 455 on
    OpenRouter (measured). A default of ours must not narrow routing.
    """
    functions = body.get("functions")
    if body.get("function_call") is not None and functions is None:
        raise Refusal("function_call", "requires functions")
    if functions is not None:
        if body.get("tools") is not None:
            raise Refusal("functions", "cannot be sent with tools; send one or the other")
        if body.get("tool_choice") is not None:
            raise Refusal("tool_choice", "cannot be sent with functions; use function_call")
        if isinstance(functions, list):
            body["tools"] = [{"type": "function", "function": f} for f in functions]
        choice = body.get("function_call")
        if isinstance(choice, str):
            body["tool_choice"] = choice
        elif isinstance(choice, dict):
            body["tool_choice"] = {"type": "function", "function": {"name": choice.get("name")}}
    # The history, which may be in the old shape whichever the request is:
    # each `function` result answers the assistant `function_call` before
    # it, so they are paired in order and given one id.
    pending: list[str] = []
    for index, message in enumerate(body.get("messages") or []):
        if not isinstance(message, dict):
            continue
        call = message.get("function_call")
        if call is not None:
            if message.get("role") != "assistant":
                raise Refusal(
                    f"messages[{index}].function_call", "is accepted only on assistant messages"
                )
            if message.get("tool_calls"):
                raise Refusal(f"messages[{index}].function_call", "cannot be sent with tool_calls")
            if isinstance(call, dict):
                call_id = f"call_function_{index}"
                message["tool_calls"] = [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": call.get("name"), "arguments": call.get("arguments")},
                    }
                ]
                del message["function_call"]
                pending.append(call_id)
        if message.get("role") == "function":
            if not pending:
                raise Refusal(
                    f"messages[{index}]",
                    "a function result must follow the assistant function_call it answers",
                )
            message["role"] = "tool"
            message["tool_call_id"] = pending.pop(0)
            message.pop("name", None)


def wants_audio(parsed: ChatCompletionRequest) -> bool:
    """Whether the caller asked for a spoken answer (P2b)."""
    return parsed.modalities is not None and any(m.value == "audio" for m in parsed.modalities)


def _check_audio_output(parsed: ChatCompletionRequest) -> None:
    """`modalities` and `audio` agree, and the format is one we can serve.

    Every audio-output model behind an account answers audio only on a
    stream and only as `pcm16` (measured 2026-09-28). A non-streamed answer
    is that stream assembled, so it can be `pcm16` or, with a header,
    `wav`; a streamed one is `pcm16`. The rest would need a transcoder and
    are refused, naming what works (P2-1), rather than answered in a
    format the caller did not ask for.
    """
    asked = wants_audio(parsed)
    if parsed.audio is not None and not asked:
        raise Refusal("audio", 'is set but modalities does not include "audio"')
    if not asked:
        return
    if parsed.audio is None:
        raise Refusal(
            "audio", 'is required when modalities includes "audio": send voice and format'
        )
    fmt = parsed.audio.format.value
    if parsed.stream and fmt != "pcm16":
        raise Refusal(
            "audio.format",
            f"cannot be {fmt} on a stream: backends stream audio as pcm16 only, so ask for pcm16",
        )
    if not parsed.stream and fmt not in ("wav", "pcm16"):
        raise Refusal(
            "audio.format",
            f"cannot be {fmt}: backends answer audio as a pcm16 stream, which this gateway "
            "returns as wav or pcm16; ask for one of those",
        )


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
