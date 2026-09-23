"""OpenAI's Responses wire, translated at the edge.

Slice 4 of the API-parity work (2026-09-23). Codex CLI 0.130 refuses
`wire_api = "chat"`, so without this door it cannot be pointed at this
install at all. Like `..anthropic`, this module is **a protocol translation
and nothing else**: the body becomes the `ChatCompletionRequest` every door
serves, and what comes back becomes the Responses shapes. Routing, the
cascade, the wake, the profile, admission, images and the recording are one
layer up and shared.

**Every decision here came from a capture of a real Codex CLI**, not from
the documentation: `specs/docs/acceptance/responses-measurement.md`. The
comments say which measurement decided what, because the failure this door
is most likely to have is passing its own tests while refusing, or quietly
mistreating, the one client it exists for.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError

from . import chat_contract, images
from ._generated.models import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    Function,
    FunctionCall,
    FunctionDefinition,
    ImageContentPart,
    ImageUrl,
    NamedToolChoice,
    ResponseFormat,
    ResponseJsonSchema,
    Role1,
    TextContentPart,
    Tool,
    ToolCall,
    ToolChoice,
)
from .anthropic import decode_signature, encode_signature

log = logging.getLogger(__name__)

#: How often an open stream says it is still working while nothing else is
#: being sent. Codex drops a stream silent for `stream_idle_timeout_ms`
#: (default five minutes), before the first event too, and retries it from
#: the start; an SSE comment does not reset that timer and a
#: `response.in_progress` event does (measured, record §5).
KEEPALIVE_SECONDS = 10.0

# --------------------------------------------------------------------------- #
# Errors: OpenAI's envelope, statuses chosen for what Codex does with them
# --------------------------------------------------------------------------- #


def error_body(
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "param": param, "code": code}}


def error_response(
    status: int,
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=error_body(message, error_type=error_type, param=param, code=code),
    )


class Refusal(Exception):
    """A request this door will not carry, with the field named."""

    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        status: int = 400,
        error_type: str = "invalid_request_error",
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.param = param
        self.status = status
        self.error_type = error_type
        self.code = code

    def response(self) -> JSONResponse:
        return error_response(
            self.status,
            self.message,
            error_type=self.error_type,
            param=self.param,
            code=self.code,
        )


def status_for(status: int) -> int:
    """Our internal failure status, as this door reports it.

    **A 404 becomes a 400**, on a measurement: Codex retries a 404 five
    times before it shows the message, and shows a 400 at once (record
    §3). Everything else keeps its status -- a 5xx is retried, which is
    what "nothing is ready yet" wants, and a 401 is bounded and shown.
    """
    return 400 if status == 404 else status


# The backend's words for a prompt its context cannot hold. llama.cpp says
# `exceed_context_size_error` / "exceeds the available context size", vLLM
# "maximum context length", OpenAI `context_length_exceeded`.
_CONTEXT_WORDS = (
    "exceed_context_size",
    "context size",
    "context length",
    "context_length_exceeded",
    "maximum context",
)


def is_context_overflow(message: str) -> bool:
    lowered = message.lower()
    return any(word in lowered for word in _CONTEXT_WORDS)


def error_code(status: int, message: str, error_type: str) -> str | None:
    """The OpenAI `code` for a refusal rendered before the stream opened."""
    if is_context_overflow(message):
        return "context_length_exceeded"
    if status == 404 or error_type == "model_not_found":
        return "model_not_found"
    return None


def stream_error_code(status: int, message: str, error_type: str) -> str:
    """The `response.failed` code for a failure after the stream opened.

    **The code decides what Codex does next** (record §4):
    `context_length_exceeded` stops and says the context is full;
    `invalid_prompt` stops and shows our message; `server_error` is retried
    five times. So a failure another attempt can fix -- a backend that
    died, nothing ready yet -- is `server_error`, and one it cannot is
    `invalid_prompt`: a backend that refused the request (the next attempt
    sends the same request), a deadline that fired while the engine was
    still computing (the next attempt computes the same prompt for as
    long, R2.5's finding), a driver refusing the gateway's own credential.
    """
    if is_context_overflow(message):
        return "context_length_exceeded"
    if status in (502, 503) and error_type != "upstream_auth_error":
        return "server_error"
    return "invalid_prompt"


# --------------------------------------------------------------------------- #
# Authentication: the chat door's rules, in this door's envelope
# --------------------------------------------------------------------------- #


async def authorize(request: Request) -> None:
    """Accept what `/v1/chat/completions` accepts, refused in OpenAI's shape.

    The same dependency, so the same audiences, client-key guard and
    statuses -- a rejected key stays a **401**, which Codex retries five
    times and then shows (record §3). Only the envelope differs: the
    shared dependency raises a `problem+json` wrapped in FastAPI's
    `detail`, which Codex would print raw.
    """
    from .dependencies import require_authorized

    header = request.headers.get("authorization") or ""
    scheme, _, value = header.partition(" ")
    creds = (
        HTTPAuthorizationCredentials(scheme="Bearer", credentials=value.strip())
        if scheme.lower() == "bearer" and value.strip()
        else None
    )
    try:
        await require_authorized(request, creds)
    except HTTPException as e:
        detail = e.detail if isinstance(e.detail, Mapping) else {}
        message = str(detail.get("detail") or detail.get("title") or e.detail)
        raise Refusal(
            message,
            status=e.status_code,
            error_type="invalid_request_error" if e.status_code < 500 else "service_unavailable",
            code="invalid_api_key" if e.status_code == 401 else None,
        ) from e


# --------------------------------------------------------------------------- #
# Request translation
# --------------------------------------------------------------------------- #

#: Every top-level field this door reads, accepts or refuses by name. Any
#: other is refused: a field that changes the answer must never be dropped.
_TOP_LEVEL = frozenset(
    {
        "model",
        "input",
        "instructions",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream",
        "temperature",
        "top_p",
        "max_output_tokens",
        "text",
        "reasoning",
        "include",
        "store",
        "truncation",
        "metadata",
        "client_metadata",
        "user",
        "safety_identifier",
        "prompt_cache_key",
        "prompt_cache_retention",
        "service_tier",
        "stream_options",
        "max_tool_calls",
        "top_logprobs",
        "background",
        "previous_response_id",
        "conversation",
        "prompt",
    }
)

#: State that lives only on OpenAI's servers. This gateway keeps none, so
#: each is refused by name rather than silently answered without it.
_STATEFUL = {
    "previous_response_id": "names a stored response",
    "conversation": "names a stored conversation",
    "prompt": "names a stored prompt template",
}

#: `include` values asking for the output of server-side tools. Nothing here
#: runs one, so there is never such an item to add them to.
_SERVER_TOOL_INCLUDES = frozenset(
    {
        "web_search_call.action.sources",
        "web_search_call.results",
        "file_search_call.results",
        "code_interpreter_call.outputs",
        "computer_call_output.output.image_url",
        "message.input_image.image_url",
    }
)

_ENCRYPTED_REASONING = "reasoning.encrypted_content"


@dataclass
class Translated:
    """A Responses body, as the shared path's request plus what the response
    needs to say about it."""

    request: ChatCompletionRequest
    #: Accepted settings this gateway does not honour, for the header.
    ignored: list[str] = field(default_factory=list)
    #: Whether reasoning items carry `encrypted_content`.
    include_reasoning: bool = False
    #: The request's settings, echoed on the response object as OpenAI does.
    echo: dict[str, Any] = field(default_factory=dict)

    def headers(self) -> dict[str, str]:
        if not self.ignored:
            return {}
        return {"x-eugene-plexus-ignored-settings": ", ".join(dict.fromkeys(self.ignored))}


def _name(value: Any) -> str:
    return chat_contract._field_name(str(value))


def _mapping(raw: Mapping[str, Any], name: str, allowed: frozenset[str]) -> Mapping[str, Any]:
    value = raw.get(name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise Refusal(f"{name}: must be an object or null.", param=name)
    for key in value:
        if key not in allowed:
            raise Refusal(f"{name}.{_name(key)}: unsupported setting.", param=name)
    return value


def _response_format(text: Mapping[str, Any]) -> ResponseFormat | None:
    fmt = text.get("format")
    if fmt is None:
        return None
    if not isinstance(fmt, Mapping):
        raise Refusal("text.format: must be an object.", param="text")
    kind = fmt.get("type")
    if kind == "text":
        return None
    if kind == "json_object":
        return ResponseFormat.model_validate({"type": "json_object"})
    if kind == "json_schema":
        if not isinstance(fmt.get("schema"), Mapping):
            raise Refusal("text.format.schema: required for json_schema output.", param="text")
        # The Responses API flattens what the chat wire nests under
        # `json_schema`; the same four fields either way.
        return ResponseFormat.model_validate(
            {
                "type": "json_schema",
                "json_schema": ResponseJsonSchema.model_validate(
                    {
                        "name": str(fmt.get("name") or "response"),
                        "description": fmt.get("description"),
                        "schema": dict(fmt["schema"]),
                        "strict": fmt.get("strict"),
                    }
                ),
            }
        )
    raise Refusal(f"text.format.type: {_name(kind)!s} is not supported.", param="text")


def _tools(definitions: Any, ignored: list[str]) -> list[Tool] | None:
    """Function tools as the chat shape; `web_search` removed and named.

    `web_search` rides on **every** Codex request (record §1). Refusing it
    would refuse the first request of every session -- `/v1/messages` did
    exactly that with `output_config` -- so it is taken off and named on
    the ignored-settings header: there is no search here to run, and the
    model is never told there is. Every other server-side tool is refused,
    because the model would be offered something that cannot happen.
    """
    if definitions is None:
        return None
    if not isinstance(definitions, list):
        raise Refusal("tools: must be a list.", param="tools")
    out: list[Tool] = []
    for index, definition in enumerate(definitions):
        if not isinstance(definition, Mapping):
            raise Refusal(f"tools[{index}]: must be an object.", param="tools")
        kind = definition.get("type")
        if isinstance(kind, str) and kind.startswith("web_search"):
            ignored.append("tools.web_search")
            continue
        if kind != "function":
            raise Refusal(
                f"tools[{index}]: {_name(kind)} is a tool this gateway cannot run. It routes to "
                "local engines and hands function calls back to you; only `function` tools "
                "(and `web_search`, which is removed) are accepted.",
                param="tools",
            )
        name = definition.get("name")
        if not isinstance(name, str) or not name:
            raise Refusal(f"tools[{index}].name: required.", param="tools")
        out.append(
            Tool(
                type="function",
                function=FunctionDefinition(
                    name=name,
                    description=definition.get("description"),
                    parameters=definition.get("parameters"),
                    strict=definition.get("strict"),
                ),
            )
        )
    return out or None


def _tool_choice(choice: Any) -> Any:
    if choice is None:
        return None
    if isinstance(choice, str):
        try:
            return ToolChoice(choice)
        except ValueError:
            raise Refusal(
                f"tool_choice: {_name(choice)} is not supported.", param="tool_choice"
            ) from None
    if isinstance(choice, Mapping) and choice.get("type") == "function":
        name = choice.get("name")
        if not isinstance(name, str) or not name:
            raise Refusal("tool_choice.name: required for a function.", param="tool_choice")
        return NamedToolChoice(type="function", function=Function(name=name))
    kind = choice.get("type") if isinstance(choice, Mapping) else choice
    raise Refusal(
        f"tool_choice: forcing {_name(kind)} is not supported; this gateway runs no server-side "
        "tools and carries `auto`, `none`, `required` or one named function.",
        param="tool_choice",
    )


def _split_data_url(url: Any, where: str) -> tuple[str, str]:
    if not isinstance(url, str) or not url.startswith("data:"):
        raise Refusal(
            f"{where}.image_url: send the image inline as a data URL. URLs are not fetched and "
            "file ids name a store that exists only at OpenAI.",
            param="input",
        )
    header, separator, data = url.partition(",")
    if not separator or not header.endswith(";base64"):
        raise Refusal(f"{where}.image_url: the data URL must be base64.", param="input")
    return header[len("data:") : -len(";base64")], data


def _image_part(
    part: Mapping[str, Any], where: str, budget: images.ImageBudget, ignored: list[str]
) -> ImageContentPart:
    """An `input_image` as the part every door carries.

    `detail` is **accepted whatever it says**: Codex sends `"high"` on every
    image, on a `-i` attachment and in `view_image`'s result alike (record
    §2), and the chat door's refusal of anything but `auto` would 400 each
    of them. A local projector sizes the picture itself, so a value other
    than `auto` is named on the ignored-settings header.
    """
    if part.get("file_id"):
        raise Refusal(
            f"{where}.file_id: file ids name a store that exists only at OpenAI; send the image "
            "inline as a data URL.",
            param="input",
        )
    media_type, data = _split_data_url(part.get("image_url"), where)
    if part.get("detail") not in (None, "auto"):
        ignored.append("input_image.detail")
    try:
        url = images.data_url_from_base64(media_type, data, f"{where}.image_url")
        budget.admit(url, f"{where}.image_url")
    except images.ImageRefusal as e:
        raise Refusal(str(e), param="input") from None
    return ImageContentPart(type="image_url", image_url=ImageUrl(url=url))


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise Refusal(f"{where}.text: must be a string.", param="input")
    return value


def _content_parts(
    content: Any,
    where: str,
    *,
    role: str,
    budget: images.ImageBudget,
    ignored: list[str],
) -> list[TextContentPart | ImageContentPart]:
    """A message's content as ordered text and image parts."""
    if content is None:
        return []
    if isinstance(content, str):
        return [TextContentPart(type="text", text=content)] if content else []
    if not isinstance(content, list):
        raise Refusal(f"{where}.content: must be a string or a list.", param="input")
    parts: list[TextContentPart | ImageContentPart] = []
    for index, part in enumerate(content):
        at = f"{where}.content[{index}]"
        if not isinstance(part, Mapping):
            raise Refusal(f"{at}: must be an object.", param="input")
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            text = _text(part.get("text"), at)
            if text:
                parts.append(TextContentPart(type="text", text=text))
        elif kind == "refusal":
            # What the model said when it declined: still its own words.
            refusal = part.get("refusal")
            if isinstance(refusal, str) and refusal:
                parts.append(TextContentPart(type="text", text=refusal))
        elif kind == "input_image":
            if role != "user":
                raise Refusal(f"{at}: images are accepted on user messages only.", param="input")
            parts.append(_image_part(part, at, budget, ignored))
        elif kind in ("input_file", "input_audio"):
            raise Refusal(
                f"{at}: this gateway cannot carry an {kind} part. It routes to local engines, "
                "and answering without it would answer a different question than the one asked.",
                param="input",
            )
        else:
            raise Refusal(f"{at}: part type {_name(kind)} is not supported.", param="input")
    return parts


def _joined(parts: list[TextContentPart | ImageContentPart]) -> Any:
    """A string when the parts are all text, ordered parts when a picture is
    among them. Adjacent text joins with a blank line, as on `/v1/messages`."""
    merged: list[TextContentPart | ImageContentPart] = []
    for part in parts:
        previous = merged[-1] if merged else None
        if isinstance(part, TextContentPart) and isinstance(previous, TextContentPart):
            merged[-1] = TextContentPart(type="text", text=f"{previous.text}\n\n{part.text}")
        else:
            merged.append(part)
    if not merged:
        return None
    if len(merged) == 1 and isinstance(merged[0], TextContentPart):
        return merged[0].text
    return merged


def _as_parts(content: Any) -> list[TextContentPart | ImageContentPart]:
    """A built message's content back as parts. Pydantic wraps it -- the
    content and each part are root models once validated -- so unwrap both."""
    content = getattr(content, "root", content)
    if content is None:
        return []
    if isinstance(content, str):
        return [TextContentPart(type="text", text=content)]
    return [getattr(part, "root", part) for part in content]


def reasoning_text(item: Mapping[str, Any]) -> str | None:
    """A reasoning item's text: its `reasoning_text` content, else ours.

    Codex sends reasoning items back with `content` and `encrypted_content`
    as we sent them (record §2), so either carries the model's earlier
    thinking back. An `encrypted_content` this gateway did not write --
    OpenAI's, from an earlier session -- is not readable here and is
    ignored, as is a `summary`, which is someone else's condensation.
    """
    content = item.get("content")
    if isinstance(content, list):
        texts = [
            str(part.get("text"))
            for part in content
            if isinstance(part, Mapping)
            and part.get("type") == "reasoning_text"
            and part.get("text")
        ]
        if texts:
            return "\n\n".join(texts)
    return decode_signature(item.get("encrypted_content"))


class _Conversation:
    """Items in, chat messages out, in order.

    Two rules do the work. **Consecutive assistant-side items are one
    turn**: a reasoning item, a message and its function calls arrive as
    separate items and are one assistant message on a chat wire. And
    **adjacent user or system messages are joined**: Codex sends two user
    messages in a row on every request (record §1), and a chat template
    that requires alternating roles refuses them.
    """

    def __init__(self, budget: images.ImageBudget, ignored: list[str]) -> None:
        self.budget = budget
        self.ignored = ignored
        self.messages: list[ChatCompletionMessage] = []
        self._parts: list[TextContentPart | ImageContentPart] = []
        self._calls: list[ToolCall] = []
        self._reasoning: list[str] = []
        self._assistant = False
        #: Pictures a tool returned, carried on the next user message.
        self._hoisted: list[TextContentPart | ImageContentPart] = []

    def _flush_assistant(self) -> None:
        if not self._assistant:
            return
        content = _joined(self._parts)
        if content is not None or self._calls or self._reasoning:
            self.messages.append(
                ChatCompletionMessage(
                    role=Role1.assistant,
                    content=content,
                    tool_calls=self._calls or None,
                    reasoning_content="\n\n".join(self._reasoning) if self._reasoning else None,
                )
            )
        self._parts, self._calls, self._reasoning, self._assistant = [], [], [], False

    def _flush_hoisted(self) -> None:
        if self._hoisted:
            pictures, self._hoisted = self._hoisted, []
            self._append(Role1.user, pictures)

    def _append(self, role: Role1, parts: list[TextContentPart | ImageContentPart]) -> None:
        if not parts:
            return
        previous = self.messages[-1] if self.messages else None
        if (
            previous is not None
            and previous.role == role
            and role in (Role1.user, Role1.system)
            and not previous.tool_calls
        ):
            # Rebuilt rather than assigned: assignment is not validated, and
            # a raw list left on `content` is not the shape every reader
            # downstream (the image check first) expects.
            self.messages[-1] = ChatCompletionMessage(
                role=role, content=_joined(_as_parts(previous.content) + parts)
            )
            return
        self.messages.append(ChatCompletionMessage(role=role, content=_joined(parts)))

    def system(self, text: str) -> None:
        self._append(Role1.system, [TextContentPart(type="text", text=text)])

    def message(self, item: Mapping[str, Any], where: str) -> None:
        role = item.get("role")
        if role == "assistant":
            self._flush_hoisted()
            self._assistant = True
            self._parts.extend(
                _content_parts(
                    item.get("content"),
                    where,
                    role="assistant",
                    budget=self.budget,
                    ignored=self.ignored,
                )
            )
            return
        if role not in ("user", "system", "developer"):
            raise Refusal(f"{where}.role: {_name(role)} is not supported.", param="input")
        self._flush_assistant()
        parts = _content_parts(
            item.get("content"), where, role=role, budget=self.budget, ignored=self.ignored
        )
        if role == "user":
            parts, self._hoisted = self._hoisted + parts, []
            self._append(Role1.user, parts)
        else:
            # `developer` is OpenAI's newer name for the instruction role; a
            # local chat template knows only `system`. Carried where it
            # stands, never hoisted: the client put it there.
            self._flush_hoisted()
            self._append(Role1.system, parts)

    def reasoning(self, item: Mapping[str, Any]) -> None:
        self._flush_hoisted()
        self._assistant = True
        text = reasoning_text(item)
        if text:
            self._reasoning.append(text)

    def function_call(self, item: Mapping[str, Any], where: str) -> None:
        self._flush_hoisted()
        self._assistant = True
        call_id = item.get("call_id")
        name = item.get("name")
        if not isinstance(call_id, str) or not call_id:
            raise Refusal(f"{where}.call_id: required.", param="input")
        if not isinstance(name, str) or not name:
            raise Refusal(f"{where}.name: required.", param="input")
        arguments = item.get("arguments")
        if isinstance(arguments, Mapping):
            arguments = json.dumps(arguments)
        if arguments is None:
            arguments = "{}"
        if not isinstance(arguments, str):
            raise Refusal(f"{where}.arguments: must be a JSON string.", param="input")
        self._calls.append(
            ToolCall(
                id=call_id, type="function", function=FunctionCall(name=name, arguments=arguments)
            )
        )

    def function_call_output(self, item: Mapping[str, Any], where: str) -> None:
        """A tool's result, as the `tool` message answering its call.

        Codex's `view_image` returns a list holding an `input_image` (record
        §2). A `tool` message carries text only, so the picture moves to the
        next user message, labelled with the call id -- the tool message
        says where it went, because a model reading an empty result and
        then an unexplained image cannot tell which call made it.
        """
        self._flush_assistant()
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise Refusal(f"{where}.call_id: required.", param="input")
        output = item.get("output")
        texts: list[str] = []
        pictures: list[ImageContentPart] = []
        if isinstance(output, str):
            texts.append(output)
        elif isinstance(output, list):
            for index, part in enumerate(output):
                at = f"{where}.output[{index}]"
                if not isinstance(part, Mapping):
                    raise Refusal(f"{at}: must be an object.", param="input")
                kind = part.get("type")
                if kind in ("input_text", "output_text", "text"):
                    texts.append(_text(part.get("text"), at))
                elif kind == "input_image":
                    pictures.append(_image_part(part, at, self.budget, self.ignored))
                else:
                    raise Refusal(f"{at}: part type {_name(kind)} is not supported.", param="input")
        elif output is not None:
            raise Refusal(f"{where}.output: must be a string or a list.", param="input")
        text = "\n".join(t for t in texts if t)
        if pictures:
            many = len(pictures) > 1
            note = (
                f"[The tool returned {len(pictures)} images; they are attached to the next user "
                "message.]"
                if many
                else "[The tool returned an image; it is attached to the next user message.]"
            )
            text = f"{text}\n{note}" if text else note
            label = "The images" if many else "The image"
            self._hoisted.append(
                TextContentPart(type="text", text=f"{label} returned by tool call {call_id}:")
            )
            self._hoisted.extend(pictures)
        self.messages.append(
            ChatCompletionMessage(role=Role1.tool, content=text, tool_call_id=call_id)
        )

    def finish(self) -> list[ChatCompletionMessage]:
        self._flush_assistant()
        self._flush_hoisted()
        return self.messages


def _messages(
    raw: Mapping[str, Any], budget: images.ImageBudget, ignored: list[str]
) -> list[ChatCompletionMessage]:
    conversation = _Conversation(budget, ignored)
    instructions = raw.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise Refusal("instructions: must be a string or null.", param="instructions")
    if instructions:
        conversation.system(instructions)
    items = raw.get("input")
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    if not isinstance(items, list):
        raise Refusal("input: must be a string or a list of items.", param="input")
    for index, item in enumerate(items):
        where = f"input[{index}]"
        if not isinstance(item, Mapping):
            raise Refusal(f"{where}: must be an object.", param="input")
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind == "message":
            conversation.message(item, where)
        elif kind == "function_call":
            conversation.function_call(item, where)
        elif kind == "function_call_output":
            conversation.function_call_output(item, where)
        elif kind == "reasoning":
            conversation.reasoning(item)
        elif kind == "item_reference":
            raise Refusal(
                f"{where}: an item_reference names a stored item, and this gateway keeps no "
                "store. Send the item itself.",
                param="input",
            )
        else:
            raise Refusal(f"{where}: item type {_name(kind)} is not supported.", param="input")
    messages = conversation.finish()
    if not messages:
        raise Refusal("input: nothing to send -- every item was empty.", param="input")
    return messages


def _number(raw: Mapping[str, Any], name: str, low: float, high: float) -> float | None:
    value = raw.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise Refusal(f"{name}: must be a number from {low:g} to {high:g}.", param=name)
    return float(value)


def translate_request(
    raw: Mapping[str, Any], *, max_images: int = images.DEFAULT_MAX_IMAGES
) -> Translated:
    """A Responses body as the request the shared path serves."""
    for name in raw:
        if name not in _TOP_LEVEL:
            raise Refusal(f"{_name(name)}: unsupported setting.", param=_name(name))
    model = raw.get("model")
    if not isinstance(model, str) or not model:
        raise Refusal("model: required.", param="model")
    for name, why in _STATEFUL.items():
        if raw.get(name) is not None:
            raise Refusal(
                f"{name}: this gateway keeps no response store, so it cannot resolve what this "
                f"{why}. Send the whole conversation in `input` with `store: false`, which is "
                "what Codex CLI does.",
                param=name,
            )
    if raw.get("background"):
        raise Refusal(
            "background: a background response is polled from a store this gateway does not keep.",
            param="background",
        )
    top_logprobs = raw.get("top_logprobs")
    if top_logprobs not in (None, 0):
        raise Refusal("top_logprobs: log probabilities are not carried.", param="top_logprobs")

    ignored: list[str] = []
    include_reasoning = False
    include = raw.get("include")
    if include is not None:
        if not isinstance(include, list):
            raise Refusal("include: must be a list.", param="include")
        for value in include:
            if value == _ENCRYPTED_REASONING:
                include_reasoning = True
            elif value == "message.output_text.logprobs":
                raise Refusal("include: log probabilities are not carried.", param="include")
            elif value not in _SERVER_TOOL_INCLUDES:
                raise Refusal(f"include: {_name(value)} is not supported.", param="include")

    if raw.get("store") is True:
        ignored.append("store")
    reasoning = _mapping(raw, "reasoning", frozenset({"effort", "summary", "generate_summary"}))
    if reasoning.get("effort") not in (None, "none"):
        ignored.append("reasoning.effort")
    if reasoning.get("summary") not in (None, "none") or reasoning.get("generate_summary") not in (
        None,
        "none",
    ):
        ignored.append("reasoning.summary")
    text = _mapping(raw, "text", frozenset({"format", "verbosity"}))
    if text.get("verbosity") is not None:
        ignored.append("text.verbosity")
    truncation = raw.get("truncation")
    if truncation == "auto":
        # A permission to drop input so it fits, which this gateway never
        # does: the engine refuses an over-long prompt instead.
        ignored.append("truncation")
    elif truncation not in (None, "disabled"):
        raise Refusal("truncation: must be auto or disabled.", param="truncation")
    options = _mapping(raw, "stream_options", frozenset({"include_obfuscation"}))
    if options.get("include_obfuscation") is True:
        ignored.append("stream_options.include_obfuscation")
    for name in ("prompt_cache_key", "prompt_cache_retention"):
        if raw.get(name) is not None:
            ignored.append(name)
    if raw.get("service_tier") not in (None, "auto", "default"):
        ignored.append("service_tier")
    for name in ("parallel_tool_calls", "stream"):
        if raw.get(name) is not None and not isinstance(raw.get(name), bool):
            raise Refusal(f"{name}: must be a boolean.", param=name)
    limit = raw.get("max_output_tokens")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise Refusal(
            "max_output_tokens: must be a positive JSON integer.", param="max_output_tokens"
        )

    budget = images.ImageBudget(max_images)
    messages = _messages(raw, budget, ignored)
    try:
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            max_tokens=limit,
            temperature=_number(raw, "temperature", 0, 2),
            top_p=_number(raw, "top_p", 0, 1),
            stream=bool(raw.get("stream")),
            tools=_tools(raw.get("tools"), ignored),
            tool_choice=_tool_choice(raw.get("tool_choice")),
            parallel_tool_calls=raw.get("parallel_tool_calls"),
            response_format=_response_format(text),
        )
    except ValidationError as e:
        first = e.errors(include_input=False, include_context=False)[0]
        where = ".".join(_name(p) if isinstance(p, str) else str(p) for p in first.get("loc", ()))
        raise Refusal(f"{where or 'body'}: has an invalid value.", param=where or None) from None

    return Translated(
        request=request,
        ignored=ignored,
        include_reasoning=include_reasoning,
        echo={
            "instructions": raw.get("instructions"),
            "tools": raw.get("tools") or [],
            "tool_choice": raw.get("tool_choice") or "auto",
            "parallel_tool_calls": raw.get("parallel_tool_calls", True),
            "temperature": raw.get("temperature"),
            "top_p": raw.get("top_p"),
            "max_output_tokens": limit,
            "reasoning": raw.get("reasoning"),
            "text": raw.get("text") or {"format": {"type": "text"}},
            "truncation": truncation or "disabled",
            "metadata": raw.get("metadata") or {},
            # Nothing is ever kept, whatever was asked for.
            "store": False,
            "previous_response_id": None,
        },
    )


# --------------------------------------------------------------------------- #
# Response translation
# --------------------------------------------------------------------------- #


def new_id(request_id: uuid.UUID | None = None) -> str:
    """`resp_` and the admission request id, so a mid-stream refusal written
    by the middleware names the same response."""
    return f"resp_{(request_id or uuid.uuid4()).hex}"


def _item_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def status_of(finish: Any) -> tuple[str, dict[str, Any] | None]:
    """The driver's finish reason as a status and `incomplete_details`.

    An answer cut at the output cap is `incomplete` with reason
    `max_output_tokens`, and a content filter's stop is `incomplete` with
    reason `content_filter` -- OpenAI's own values. A backend that broke is
    `completed`: there is no Responses status for it, and the truncation is
    reported where it can be acted on, the log and the metrics row.
    """
    reason = getattr(finish, "value", finish)
    if reason == "length":
        return "incomplete", {"reason": "max_output_tokens"}
    if reason == "content_filter":
        return "incomplete", {"reason": "content_filter"}
    return "completed", None


def usage_block(usage: Any) -> dict[str, Any] | None:
    """Token counts in OpenAI's names.

    **`input_tokens` includes cached input**, unlike Anthropic's. The two
    detail objects appear only when the backend reported them, as on the
    chat door: a zero would claim the backend counted and found none.
    """
    if usage is None:
        return None
    prompt = getattr(usage, "promptTokens", 0) or 0
    completion = getattr(usage, "completionTokens", 0) or 0
    out: dict[str, Any] = {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": getattr(usage, "totalTokens", None) or prompt + completion,
    }
    cached = getattr(usage, "cachedPromptTokens", None)
    if cached is not None:
        out["input_tokens_details"] = {"cached_tokens": cached}
    reasoning = getattr(usage, "reasoningTokens", None)
    if reasoning is not None:
        out["output_tokens_details"] = {"reasoning_tokens": reasoning}
    return out


def reasoning_item(text: str, *, include_encrypted: bool, item_id: str | None = None) -> dict:
    """The model's reasoning in the slot the Responses API gives a model's
    own reasoning -- `content` of `reasoning_text` -- with `summary` empty,
    because nothing here condenses it. `encrypted_content`, when asked for,
    is the Anthropic door's carrier: readable only here."""
    item: dict[str, Any] = {
        "id": item_id or _item_id("rs"),
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
    }
    if include_encrypted:
        item["encrypted_content"] = encode_signature(text)
    return item


def message_item(text: str, *, status: str = "completed", item_id: str | None = None) -> dict:
    return {
        "id": item_id or _item_id("msg"),
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
    }


def call_item(
    call_id: str,
    name: str,
    arguments: str,
    *,
    status: str = "completed",
    item_id: str | None = None,
) -> dict:
    return {
        "id": item_id or _item_id("fc"),
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def output_items(
    *,
    content: str | None,
    tool_calls: Any,
    reasoning: str | None,
    include_reasoning: bool,
    status: str = "completed",
) -> list[dict[str, Any]]:
    """Reasoning first, then the answer, then one item per tool call. An
    item in a response cut short is `incomplete` too, as on the stream."""
    item_status = "incomplete" if status == "incomplete" else "completed"
    items: list[dict[str, Any]] = []
    if reasoning:
        items.append(reasoning_item(reasoning, include_encrypted=include_reasoning))
    if content:
        items.append(message_item(content, status=item_status))
    for call in tool_calls or []:
        items.append(
            call_item(
                call.id, call.function.name, call.function.arguments or "{}", status=item_status
            )
        )
    return items


def response_object(
    *,
    response_id: str,
    created_at: int,
    model: str,
    status: str,
    output: list[dict[str, Any]],
    echo: Mapping[str, Any],
    usage: Any = None,
    incomplete_details: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": error,
        "incomplete_details": incomplete_details,
        "model": model,
        "output": output,
        "usage": usage_block(usage),
        **echo,
    }


# --------------------------------------------------------------------------- #
# The stream
# --------------------------------------------------------------------------- #


async def with_keepalive(
    events: AsyncIterator[Any], interval: float = KEEPALIVE_SECONDS
) -> AsyncIterator[Any]:
    """The driver's events, with a `None` whenever `interval` passes quietly.

    **The driver's stream runs in one task of its own** and hands events
    across a queue. Resuming it from a fresh task on every event would move
    httpx's connection -- and anyio's cancel scopes inside it -- between
    tasks, and R2.5 has already met a cancel scope entered in one task and
    touched from another. On the way out the task is cancelled and the
    stream closed, so the backend call ends with the client's interest.
    """
    queue: asyncio.Queue[tuple[bool, Any]] = asyncio.Queue(maxsize=1)

    async def pump() -> None:
        try:
            async for event in events:
                await queue.put((True, event))
        except Exception as e:  # handed to the consumer, which classifies it
            await queue.put((False, e))
            return
        finally:
            closer = getattr(events, "aclose", None)
            if closer is not None:
                await closer()
        await queue.put((False, None))

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                ok, value = await asyncio.wait_for(queue.get(), interval)
            except TimeoutError:
                yield None
                continue
            if ok:
                yield value
            elif value is None:
                return
            else:
                raise value
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class StreamTranslator:
    """Our internal stream as the Responses event stream.

    **Output items are numbered statefully and our stream is not** -- the
    same work `anthropic.StreamTranslator` does. Reasoning and text deltas
    carry no index, tool fragments carry a per-call index; so this holds the
    one open item and a map from call index to item, and closes whatever is
    open before the next kind of item starts.
    """

    def __init__(
        self,
        response_id: str,
        model: str,
        *,
        echo: Mapping[str, Any],
        include_reasoning: bool = False,
    ) -> None:
        self.response_id = response_id
        self.model = model
        self.echo = echo
        self.include_reasoning = include_reasoning
        self.created_at = int(time.time())
        self.output: list[dict[str, Any]] = []
        self._sequence = 0
        #: The open reasoning or message item: kind, id, output_index, text.
        self._open: dict[str, Any] | None = None
        #: Open function calls by the driver's call index.
        self._calls: dict[int, dict[str, Any]] = {}

    # -- framing ----------------------------------------------------------

    def frame(self, name: str, data: dict[str, Any]) -> str:
        """One SSE frame; `event:` and `data.type` always agree."""
        payload = {"type": name, **data, "sequence_number": self._sequence}
        self._sequence += 1
        return f"event: {name}\ndata: {json.dumps(payload)}\n\n"

    def _lean(self, status: str = "in_progress") -> dict[str, Any]:
        """The response object while it is being written. Lean on purpose:
        a keepalive every ten seconds carrying Codex's 21 KB of instructions
        and 40 KB of tools would be most of the traffic of a slow prefill."""
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": [],
        }

    def start(self) -> list[str]:
        """Sent as soon as a backend is chosen, before any output.

        Names the model that was **asked for**; the terminal event names the
        one that answered. Waiting for the first token to send anything is
        what the Anthropic door does, and here it would let Codex's idle
        timer run out during a slow prefill (record §5).
        """
        return [
            self.frame("response.created", {"response": self._lean()}),
            self.frame("response.in_progress", {"response": self._lean()}),
        ]

    def keepalive(self) -> str:
        return self.frame("response.in_progress", {"response": self._lean()})

    # -- items --------------------------------------------------------------

    def _close_open(self, status: str = "completed") -> list[str]:
        item = self._open
        if item is None:
            return []
        self._open = None
        text = "".join(item["text"])
        index = item["output_index"]
        if item["kind"] == "reasoning":
            done = reasoning_item(
                text, include_encrypted=self.include_reasoning, item_id=item["id"]
            )
            events = [
                self.frame(
                    "response.reasoning_text.done",
                    {
                        "item_id": item["id"],
                        "output_index": index,
                        "content_index": 0,
                        "text": text,
                    },
                ),
                self.frame(
                    "response.content_part.done",
                    {
                        "item_id": item["id"],
                        "output_index": index,
                        "content_index": 0,
                        "part": {"type": "reasoning_text", "text": text},
                    },
                ),
            ]
        else:
            done = message_item(text, status=status, item_id=item["id"])
            events = [
                self.frame(
                    "response.output_text.done",
                    {
                        "item_id": item["id"],
                        "output_index": index,
                        "content_index": 0,
                        "text": text,
                        "logprobs": [],
                    },
                ),
                self.frame(
                    "response.content_part.done",
                    {
                        "item_id": item["id"],
                        "output_index": index,
                        "content_index": 0,
                        "part": done["content"][0],
                    },
                ),
            ]
        events.append(
            self.frame("response.output_item.done", {"output_index": index, "item": done})
        )
        self.output.append(done)
        return events

    def _close_calls(self, keep: int | None = None, status: str = "completed") -> list[str]:
        events: list[str] = []
        for call_index, call in sorted(self._calls.items(), key=lambda kv: kv[1]["output_index"]):
            if call_index == keep:
                continue
            arguments = "".join(call["arguments"])
            done = call_item(
                call["call_id"], call["name"], arguments, status=status, item_id=call["id"]
            )
            events.append(
                self.frame(
                    "response.function_call_arguments.done",
                    {
                        "item_id": call["id"],
                        "output_index": call["output_index"],
                        "name": call["name"],
                        "arguments": arguments,
                    },
                )
            )
            events.append(
                self.frame(
                    "response.output_item.done",
                    {"output_index": call["output_index"], "item": done},
                )
            )
            self.output.append(done)
            del self._calls[call_index]
        return events

    def _next_index(self) -> int:
        return len(self.output) + (1 if self._open else 0) + len(self._calls)

    def _open_item(self, kind: str) -> list[str]:
        events = self._close_open() + self._close_calls()
        index = self._next_index()
        prefix = "rs" if kind == "reasoning" else "msg"
        self._open = {"kind": kind, "id": _item_id(prefix), "output_index": index, "text": []}
        item_id = self._open["id"]
        item: dict[str, Any]
        part: dict[str, Any]
        if kind == "reasoning":
            item = {"id": item_id, "type": "reasoning", "summary": [], "content": []}
            part = {"type": "reasoning_text", "text": ""}
        else:
            item = {**message_item("", status="in_progress", item_id=item_id), "content": []}
            part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
        events.append(
            self.frame("response.output_item.added", {"output_index": index, "item": item})
        )
        events.append(
            self.frame(
                "response.content_part.added",
                {"item_id": item_id, "output_index": index, "content_index": 0, "part": part},
            )
        )
        return events

    def reasoning(self, chunk: str) -> list[str]:
        events = (
            [] if self._open and self._open["kind"] == "reasoning" else self._open_item("reasoning")
        )
        assert self._open is not None
        self._open["text"].append(chunk)
        events.append(
            self.frame(
                "response.reasoning_text.delta",
                {
                    "item_id": self._open["id"],
                    "output_index": self._open["output_index"],
                    "content_index": 0,
                    "delta": chunk,
                },
            )
        )
        return events

    def text(self, chunk: str) -> list[str]:
        events = (
            [] if self._open and self._open["kind"] == "message" else self._open_item("message")
        )
        assert self._open is not None
        self._open["text"].append(chunk)
        events.append(
            self.frame(
                "response.output_text.delta",
                {
                    "item_id": self._open["id"],
                    "output_index": self._open["output_index"],
                    "content_index": 0,
                    "delta": chunk,
                    "logprobs": [],
                },
            )
        )
        return events

    def tool_fragments(self, fragments: list[dict[str, Any]]) -> list[str]:
        """Driver tool fragments as `function_call_arguments.delta` runs.

        A fragment is not parseable JSON on its own, so the arguments are
        forwarded exactly as they arrive and the client reassembles them.
        """
        events: list[str] = []
        for position, fragment in enumerate(fragments):
            call_index = fragment.get("index")
            if not isinstance(call_index, int):
                call_index = position
            function = fragment.get("function") or {}
            if call_index not in self._calls:
                events += self._close_open() + self._close_calls(keep=call_index)
                call: dict[str, Any] = {
                    "id": _item_id("fc"),
                    "output_index": self._next_index(),
                    "call_id": fragment.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                    "name": function.get("name") or "",
                    "arguments": [],
                }
                self._calls[call_index] = call
                item = call_item(
                    call["call_id"], call["name"], "", status="in_progress", item_id=call["id"]
                )
                events.append(
                    self.frame(
                        "response.output_item.added",
                        {"output_index": call["output_index"], "item": item},
                    )
                )
            call = self._calls[call_index]
            arguments = function.get("arguments")
            if arguments:
                call["arguments"].append(arguments)
                events.append(
                    self.frame(
                        "response.function_call_arguments.delta",
                        {
                            "item_id": call["id"],
                            "output_index": call["output_index"],
                            "delta": arguments,
                        },
                    )
                )
        return events

    # -- the end --------------------------------------------------------------

    def finish(self, *, model: str, finish: Any, usage: Any) -> list[str]:
        status, details = status_of(finish)
        item_status = "incomplete" if status == "incomplete" else "completed"
        events = self._close_open(item_status) + self._close_calls(status=item_status)
        self.model = model
        response = response_object(
            response_id=self.response_id,
            created_at=self.created_at,
            model=model,
            status=status,
            output=self.output,
            echo=self.echo,
            usage=usage,
            incomplete_details=details,
        )
        name = "response.incomplete" if status == "incomplete" else "response.completed"
        events.append(self.frame(name, {"response": response}))
        return events

    def failed(self, message: str, *, code: str) -> list[str]:
        """`response.failed`, never a bare `error` event -- Codex reports the
        latter without our message (record §4). What had arrived is closed
        as `incomplete` and kept on the response."""
        events = self._close_open("incomplete") + self._close_calls(status="incomplete")
        response = response_object(
            response_id=self.response_id,
            created_at=self.created_at,
            model=self.model,
            status="failed",
            output=self.output,
            echo=self.echo,
            error={"code": code, "message": message},
        )
        events.append(self.frame("response.failed", {"response": response}))
        return events


def failed_frame(response_id: str, message: str, *, code: str) -> str:
    """A `response.failed` for the admission middleware, which ends a stream
    it did not write (a deadline, a key turned off mid-answer) and knows
    neither the model nor the sequence so far."""
    response = {
        "id": response_id,
        "object": "response",
        "status": "failed",
        "output": [],
        "error": {"code": code, "message": message},
    }
    payload = {"type": "response.failed", "response": response}
    return f"event: response.failed\ndata: {json.dumps(payload)}\n\n"


def middleware_code(status: int) -> str:
    """The `response.failed` code for a refusal the middleware writes.

    A deadline that fired or a key turned off would fail the same way on
    the next attempt, so Codex is told to stop (`invalid_prompt`); a rate
    limit is `rate_limit_exceeded`, which it retries.
    """
    if status == 429:
        return "rate_limit_exceeded"
    if status in (401, 403, 504, 400):
        return "invalid_prompt"
    return "server_error"
