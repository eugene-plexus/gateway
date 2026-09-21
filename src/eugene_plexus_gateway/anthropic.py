"""Anthropic's Messages wire, translated at the edge.

R4. This module is a **protocol translation and nothing else**: it turns
an Anthropic request into the same `ChatCompletionRequest` the OpenAI
door builds, and turns what comes back into Anthropic's shapes. Routing,
the cascade, the wake, the settings profile, the truncation detector and
the recording are all one layer up and are *shared*, not re-implemented,
which is what keeps `GET /v1/metrics` from being blind to this door --
M8's finding, in a new place.

**Every shape decision in here came from a capture rather than from the
documentation.** `specs/docs/acceptance/anthropic-messages-measurement.md`
records nine runs of a real Claude Code against a throwaway listener,
and it moved a field from the refusal list to the drop list, inverted
the status code for a bad credential, and named three request shapes
nobody had written down. The comments say which fact is measured where,
because the failure this door is most likely to have is passing its own
tests while refusing the one client it exists for.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import chat_contract, security
from ._generated.models import (
    AnthropicMessagesRequest,
    ChatCompletionMessage,
    ChatCompletionRequest,
    Function,
    FunctionDefinition,
    NamedToolChoice,
    Role1,
    Stop,
    Tool,
    ToolCall,
    ToolChoice,
)
from ._generated.models import (
    FunctionCall as OpenAIFunctionCall,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# The error envelope, and the status table that is not the obvious one
# --------------------------------------------------------------------------- #

# Measured 2026-09-19 against Claude Code 2.1.207, by answering each
# status and counting what the client did next:
#
#   400  1 attempt,  message shown verbatim
#   401  UNBOUNDED retries, user shown NOTHING
#   403  1 attempt,  message shown verbatim
#   404  2 attempts, our message DISCARDED for a generic one
#   429  retried, silent
#   500  retried, silent
#   503  retried, silent
#
# Two consequences are baked into this module and both invert what one
# would otherwise write. A rejected credential is **403** (see
# `authorize`), and a model nothing serves is **400** (see
# `status_for`). `X-Stainless-Retry-Count` stayed 0 on every retry, so
# nothing here can detect a storm from the wire.
_ERROR_TYPE_BY_STATUS = {
    400: "invalid_request_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
    499: "api_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    504: "api_error",
}


def error_body(status: int, message: str, *, kind: str | None = None) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "type": kind or _ERROR_TYPE_BY_STATUS.get(status, "api_error"),
            "message": message,
        },
    }


def error_response(status: int, message: str, *, kind: str | None = None) -> JSONResponse:
    """Anthropic's envelope, at the top level of the body.

    A plain `JSONResponse` rather than a raised `HTTPException` for the
    same reason the OpenAI door gives: FastAPI wraps a raised one in its
    own `{"detail": ...}`, which is precisely the shape the client's SDK
    cannot read -- and here the cost is higher, because a credential
    error the client cannot parse is a credential error it retries.
    """
    return JSONResponse(status_code=status, content=error_body(status, message, kind=kind))


class Refusal(Exception):
    """A request this door will not carry, with the field named.

    Carries its own status so the caller does not have to map it back;
    every refusal raised here is one the client will show to a person.
    """

    def __init__(self, message: str, *, status: int = 400, kind: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.kind = kind

    def response(self) -> JSONResponse:
        return error_response(self.status, self.message, kind=self.kind)


def status_for(openai_status: int) -> int:
    """Our internal failure status, as this door should report it.

    Only one value moves, and it moves because of a measurement rather
    than a preference: **404 becomes 400**. Claude Code discards a 404's
    body and substitutes *"There's an issue with the selected model"*,
    which would throw away the part of our message that matters -- the
    available model names, and the sealed-control-root diagnosis that
    exists precisely so this surface stops naming two healthy places and
    never mentioning the root.
    """
    return 400 if openai_status == 404 else openai_status


# --------------------------------------------------------------------------- #
# Authentication: two headers, and 403 rather than 401
# --------------------------------------------------------------------------- #


def credential(request: Request) -> str | None:
    """The token, from whichever header the client chose.

    **They are alternatives, never both.** Measured: `ANTHROPIC_API_KEY`
    produces `x-api-key` and no `Authorization` header at all, while
    `ANTHROPIC_AUTH_TOKEN` produces `Authorization: Bearer` and no
    `x-api-key`. A door reading only one of them does not merely prefer
    the wrong header -- on half the configurations it sees no credential
    whatsoever, and then answers the status that gets retried forever.
    """
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key.strip()
    header = request.headers.get("authorization")
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


async def authorize(request: Request) -> None:
    """Accept operator, any service token, or a client key -- or 403.

    **403 and not 401, and this is the divergence that must not be
    tidied away.** Measured: a 401 from this endpoint makes Claude Code
    retry without bound -- nine attempts in 79 seconds, still climbing
    when the measurement's timeout fired -- while showing its user
    nothing at all. So a typo in a key becomes a silent retry storm
    against our own gateway, which is step 7's looping symptom one layer
    up, in the client we most want to keep. A 403 is reported on the
    first attempt with our message intact.

    The OpenAI door keeps its 401, because an OpenAI SDK reports one
    properly and a 403 there would read as "this key exists but may not
    do this". `test_the_openai_door_still_answers_401` asserts that half,
    so the pair cannot be "harmonised" by someone who sees only one.
    """
    auth = request.app.state.auth_state
    if auth.auth_disabled:
        return

    token = credential(request)
    if token is None:
        raise Refusal(
            "No credential. Send an Eugene Plexus client key as `x-api-key` or as "
            "`Authorization: Bearer`. With Claude Code, set ANTHROPIC_AUTH_TOKEN to the key "
            "you made under Home -> Use it from your apps.",
            status=403,
            kind="authentication_error",
        )

    assert auth.signing_key is not None  # narrowed by auth_disabled
    try:
        payload = security.decode_token(
            token=token,
            signing_key=auth.signing_key,
            accept_operator=True,
            accept_any_service=True,
            accept_client=True,
        )
    except Exception as e:
        raise Refusal(
            f"That key was rejected: {e}. Make a new one under Home -> Use it from your apps.",
            status=403,
            kind="authentication_error",
        ) from e

    if payload.aud != security.AUDIENCE_CLIENT:
        return
    from .admission import current

    context = current.get()
    if context is not None:
        context.key_id = payload.jti
        context.key_name = payload.sub
    guard = getattr(request.app.state, "client_key_guard", None)
    decision = await guard.decision(payload.jti) if guard is not None else "unavailable"
    if decision == "unavailable":
        raise Refusal(
            "Client-key policy unavailable; check the agent/control root. "
            "Operator sign-in remains available.",
            status=503,
            kind="api_error",
        )
    if decision == "unregistered":
        raise Refusal(
            "This key is not registered. Check migration on the node that made it "
            "under Use it from your apps, or replace it.",
            status=403,
            kind="permission_error",
        )
    if decision == "revoked":
        raise Refusal(
            "This client key was turned off. Make a new one under "
            "Home -> Use it from your apps, and paste it into the app that is failing.",
            status=403,
            kind="permission_error",
        )


# --------------------------------------------------------------------------- #
# Request translation
# --------------------------------------------------------------------------- #

# Block types this gateway will not carry. Refused rather than dropped,
# because each changes what the answer would be: a model that never
# received the image is not answering the question that was asked, and a
# confident text-only reply to "what is in this screenshot" is worse
# than a refusal naming the reason.
_REFUSED_BLOCK_TYPES = {"image", "document"}

# Fields that arrive on every real request and have no equivalent here.
# Accepted for the measured Claude Code client; A2 reports ignored controls
# on a response header, rather than silently implying native support.
# A blanket unknown-field refusal passes every refusal test and then
# fails on the first real request, which is the trap this whole door was
# scoped around.
_DROPPED_TOP_LEVEL = ("thinking", "cache_control", "metadata", "context_management")

# Fields whose presence means the caller wants something this control
# plane cannot do at all, as opposed to something it can ignore.
_REFUSED_TOP_LEVEL = ("mcp_servers", "top_k")


def compatibility_headers(raw: Mapping[str, Any]) -> dict[str, str]:
    ignored = [
        name for name in _DROPPED_TOP_LEVEL if name != "metadata" and raw.get(name) is not None
    ]

    # Cache hints also arrive on system, tool and message blocks. Do not walk
    # tool JSON Schemas or user metadata looking for coincidental property names.
    def has_cache(value: Any) -> bool:
        if isinstance(value, list):
            return any(has_cache(item) for item in value)
        if isinstance(value, dict):
            return value.get("cache_control") is not None or has_cache(value.get("content"))
        return False

    if "cache_control" not in ignored and any(
        has_cache(raw.get(name)) for name in ("system", "messages", "tools")
    ):
        ignored.append("cache_control")
    return {"x-eugene-plexus-ignored-settings": ", ".join(ignored)} if ignored else {}


_MAX_STOP_SEQUENCES = 4


def _validation_refusal(exc: ValidationError) -> Refusal:
    """A Pydantic failure as a 400 naming the field.

    Without this the response is FastAPI's 422 `{"detail": [...]}`,
    which an Anthropic SDK reports as an unhelpful generic failure --
    and a missing `max_tokens`, which the contract says is a 400 naming
    the field, would arrive as exactly that.
    """
    first = exc.errors()[0]
    where = ".".join(str(p) for p in first.get("loc", ()) if p != "body") or "body"
    return Refusal(f"{where}: {first.get('msg', 'invalid')}")


def _blocks(content: Any) -> list[Any]:
    """Content as a list of blocks, whether it arrived as a string."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


def _field(block: Any, name: str) -> Any:
    """One field off a block, whether it is a model or a plain dict.

    Both forms reach here: the validated model for the top-level walk,
    and raw dicts for a `tool_result`'s nested content, which Pydantic
    keeps as-is.
    """
    if isinstance(block, Mapping):
        return block.get(name)
    return getattr(block, name, None)


def _refuse_unsupported_blocks(blocks: Iterable[Any], *, where: str) -> None:
    """Refuse an image or document anywhere it can appear.

    **Including inside a `tool_result`**, which is the case a check that
    only walked top-level blocks would pass its own test on and then
    serve a screenshot-blind answer for.
    """
    for block in blocks:
        kind = _field(block, "type")
        if kind in _REFUSED_BLOCK_TYPES:
            raise Refusal(
                f"This gateway cannot carry an {kind} block ({where}). It routes to local "
                f"text engines, and answering without the {kind} would answer a different "
                f"question than the one asked. Send text, or point this client at a model "
                f"that takes {kind}s."
            )
        nested = _field(block, "content")
        if isinstance(nested, list):
            _refuse_unsupported_blocks(nested, where=f"{where} -> tool_result")


def _system_text(system: Any) -> str | None:
    """Every system block concatenated, in order.

    **`system[0]` is not the prompt on the commonest client.** Claude
    Code puts a billing header there as prose
    (`x-anthropic-billing-header: …`), so code that reads `system[0]` as
    the instruction is wrong about the request it will see most often.
    All blocks are joined and the model sorts it out, exactly as it
    would upstream.
    """
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    parts = [b.text for b in system if getattr(b, "text", None)]
    return "\n\n".join(parts) or None


def _tool_result_text(block: Any) -> str:
    """A tool result as the text an OpenAI `tool` message carries.

    `is_error` is carried into the text rather than dropped: a harness
    that cannot see that its own tool failed will call it again, which
    is the looping symptom in miniature.
    """
    content = _field(block, "content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\n".join(str(_field(b, "text") or "") for b in content if _field(b, "text"))
    else:
        text = ""
    if _field(block, "is_error"):
        return f"[tool error] {text}" if text else "[tool error]"
    return text


def _tool_choice(choice: Any) -> Any:
    """Anthropic's four values onto OpenAI's three plus a named one.

    `any` becomes `required`, which is the closest honest reading: both
    mean "you must call one of these", and neither names which.
    """
    if choice is None:
        return None
    kind = getattr(choice, "type", None)
    kind = getattr(kind, "value", kind)
    if kind == "tool":
        name = getattr(choice, "name", None)
        if not name:
            raise Refusal("tool_choice: type 'tool' needs a 'name'.")
        return NamedToolChoice(type="function", function=Function(name=name))
    return {"auto": ToolChoice.auto, "any": ToolChoice.required, "none": ToolChoice.none}.get(
        str(kind), ToolChoice.auto
    )


def _tools(definitions: Any) -> list[Tool] | None:
    """Anthropic tool definitions in OpenAI's function shape.

    A definition carrying a `type` is a **server-side** tool -- web
    search, code execution, a computer -- which this gateway would have
    to run itself. It has nothing to run them with, so it refuses at the
    door rather than at the moment the model chooses to use one, which
    would be a failure with no good place to report it.
    """
    if not definitions:
        return None
    out: list[Tool] = []
    for definition in definitions:
        server_side = getattr(definition, "type", None)
        if server_side:
            raise Refusal(
                f"tools: {definition.name!r} is a server-side tool ({server_side}), which this "
                "gateway cannot execute. It routes to local engines and hands tool calls back "
                "to you; only client-side tools work here."
            )
        out.append(
            Tool(
                type="function",
                function=FunctionDefinition(
                    name=definition.name,
                    description=definition.description,
                    parameters=definition.input_schema,
                ),
            )
        )
    return out


def translate_request(raw: Mapping[str, Any]) -> ChatCompletionRequest:
    """An Anthropic body as the request the shared path already serves.

    Measured Claude Code hints remain accepted, with response warnings. Unknown
    top-level settings and unsupported top_k are refused, not discarded.
    """
    for name in raw:
        if name not in AnthropicMessagesRequest.model_fields and name not in (
            *_DROPPED_TOP_LEVEL,
            *_REFUSED_TOP_LEVEL,
        ):
            raise Refusal(f"{chat_contract._field_name(name)}: unsupported setting")
    limit = raw.get("max_tokens")
    if type(limit) is not int or limit < 1:
        raise Refusal("max_tokens: must be a positive JSON integer")
    try:
        body = AnthropicMessagesRequest.model_validate(raw)
    except ValidationError as e:
        raise _validation_refusal(e) from e

    for name in _REFUSED_TOP_LEVEL:
        if raw.get(name) is not None:
            raise Refusal(
                f"{name}: this gateway does not implement it. It routes requests to local "
                "inference backends and has nothing to connect on your behalf."
            )

    if body.stop_sequences and len(body.stop_sequences) > _MAX_STOP_SEQUENCES:
        # Refused rather than truncated to the first four: dropping a
        # stop sequence changes where the answer ends, which is a
        # silently different answer rather than a smaller request.
        raise Refusal(
            f"stop_sequences: at most {_MAX_STOP_SEQUENCES} are supported and "
            f"{len(body.stop_sequences)} were sent. Dropping one would change where the "
            "answer stops, so the request was refused instead."
        )

    messages: list[ChatCompletionMessage] = []
    system = _system_text(body.system)
    if system:
        messages.append(ChatCompletionMessage(role=Role1.system, content=system))

    for turn in body.messages:
        blocks = _blocks(turn.content)
        _refuse_unsupported_blocks(blocks, where=f"{turn.role.value} message")

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        results: list[ChatCompletionMessage] = []

        for block in blocks:
            kind = _field(block, "type")
            if kind == "text":
                if _field(block, "text"):
                    text_parts.append(str(_field(block, "text")))
            elif kind == "tool_use":
                # The inbound half. Without it a tool loop cannot
                # continue past its first turn, because the assistant
                # turn handed back to us carries `tool_use` blocks
                # rather than text -- and an assistant message whose
                # calls were dropped reads to the model as though it
                # had never made them.
                calls.append(
                    ToolCall(
                        id=str(_field(block, "id") or f"toolu_{uuid.uuid4().hex[:16]}"),
                        type="function",
                        function=OpenAIFunctionCall(
                            name=str(_field(block, "name") or ""),
                            arguments=json.dumps(_field(block, "input") or {}),
                        ),
                    )
                )
            elif kind == "tool_result":
                # **Anthropic has no `tool` role.** A result is a block
                # inside a USER message, several to a message. Getting
                # this wrong turns the tool's output into the human
                # speaking, which is exactly the defect step 6 found on
                # the other protocol.
                results.append(
                    ChatCompletionMessage(
                        role=Role1.tool,
                        content=_tool_result_text(block),
                        tool_call_id=str(_field(block, "tool_use_id") or ""),
                    )
                )

        # Results first: they answer the assistant turn above them, and
        # an OpenAI backend expects every `tool` message to follow the
        # assistant message that asked for it with nothing between.
        messages.extend(results)
        if text_parts or calls:
            messages.append(
                ChatCompletionMessage(
                    role=Role1(turn.role.value),
                    content="\n\n".join(text_parts) if text_parts else None,
                    tool_calls=calls or None,
                )
            )

    if not messages:
        raise Refusal("messages: nothing to send -- every block was empty.")

    return ChatCompletionRequest(
        model=body.model,
        messages=messages,
        max_tokens=body.max_tokens,
        # Anthropic's range is 0-1 and OpenAI's is 0-2. Passed through
        # unscaled: the number means the same thing to a local engine
        # either way, and rescaling would silently change what the
        # caller asked for.
        temperature=body.temperature,
        top_p=body.top_p,
        stop=Stop(body.stop_sequences) if body.stop_sequences else None,
        stream=bool(body.stream),
        tools=_tools(body.tools),
        tool_choice=_tool_choice(body.tool_choice),
    )


# --------------------------------------------------------------------------- #
# Response translation
# --------------------------------------------------------------------------- #

_STOP_REASON_BY_FINISH = {
    "stop": "end_turn",
    "stop_sequence": "stop_sequence",
    "length": "max_tokens",
    # The value that makes an agent loop terminate correctly. A caller
    # that sees `end_turn` here stops; one that sees `tool_use`
    # dispatches and comes back. Getting it wrong does not look like an
    # error -- it looks like a model that answered instead of using its
    # tools.
    "tool_calls": "tool_use",
    # Anthropic's own name for a classifier stopping the answer, so a
    # client switching on this field gets a value from the vocabulary
    # it already parses rather than one we invented -- the rule
    # `tool_use` was fixed under, applied to the next value along.
    # **Not verified against a live Anthropic SDK**; the OpenAI door's
    # `content_filter` is OpenAI's own value and that one is measured.
    "content_filter": "refusal",
    # There is no Anthropic stop reason for "the backend broke", and
    # inventing one would break a client switching on this field. The
    # truncation is reported where it can be acted on: the log line and
    # the metrics row.
    "error": "end_turn",
}


def stop_reason(finish: Any) -> str:
    return _STOP_REASON_BY_FINISH.get(getattr(finish, "value", str(finish)), "end_turn")


def usage_block(usage: Any) -> dict[str, int]:
    """Token counts, renamed.

    Cache counts are reported as an honest zero rather than omitted: a
    client that reads them should see nothing rather than an absence it
    has to guess about.
    """
    return {
        "input_tokens": getattr(usage, "promptTokens", 0) or 0,
        "output_tokens": getattr(usage, "completionTokens", 0) or 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def content_blocks(content: str | None, tool_calls: Any) -> list[dict[str, Any]]:
    """Text and tool calls as Anthropic content blocks, text first."""
    blocks: list[dict[str, Any]] = []
    if content:
        blocks.append({"type": "text", "text": content})
    for call in tool_calls or []:
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            # A backend that produced unparseable arguments is a real
            # event and the caller has to see it. `input` is a typed
            # object on this wire, so the raw string cannot ride there;
            # it goes somewhere a human will read rather than being
            # dropped.
            log.warning("tool call %s had unparseable arguments; sending an empty input", call.id)
            arguments = {}
        blocks.append(
            {"type": "tool_use", "id": call.id, "name": call.function.name, "input": arguments}
        )
    return blocks


def message_response(
    *, model: str, content: str | None, tool_calls: Any, finish: Any, usage: Any
) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks(content, tool_calls),
        "stop_reason": stop_reason(finish),
        "stop_sequence": None,
        "usage": usage_block(usage),
    }


def envelope_headers(routing: Any) -> dict[str, str]:
    """`x_eugene_plexus` as response headers.

    **Not in the body.** Their wire is typed events and a strict client
    is exactly who this door is for, so an unknown top-level key is a
    risk with no upside. The *recording* is unaffected -- it rides the
    shared path and lands in `GET /v1/metrics` like any other request.
    """
    out: dict[str, str] = {}
    for field, value in routing.model_dump(exclude_none=True).items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        out[f"x-eugene-plexus-{field.replace('_', '-')}"] = str(value)
    return out


# --------------------------------------------------------------------------- #
# Stream translation -- the one piece of real work
# --------------------------------------------------------------------------- #


def frame(name: str, data: dict[str, Any]) -> str:
    """One SSE frame carrying both channels.

    The `event:` name and `data.type` always agree, because a client may
    read either and the two disagreeing is a bug no test that reads only
    one of them can see.
    """
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


class StreamTranslator:
    """Our internal stream as Anthropic's typed events.

    **Anthropic numbers content blocks statefully and we do not**, which
    is the whole of the difficulty. Text deltas carry no index; tool
    fragments carry a per-call index that starts at 0 for the first
    call. So this holds the open text block and a map from call index to
    content-block index, and closes the text block before the first tool
    block opens -- Anthropic allows exactly one open block at a time.

    Getting it wrong means a strict SDK rejects the stream **inside a
    200**, where no status code can report it and the user sees a model
    that produced nothing.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.started = False
        self._next_index = 0
        self._text_index: int | None = None
        self._tool_index: dict[int, int] = {}
        self._message_id = f"msg_{uuid.uuid4().hex}"

    def start(self, model: str | None = None) -> str:
        """`message_start`, emitted on the FIRST DRIVER EVENT.

        Never on request acceptance: it names the model, and until the
        first token the cascade can still change which backend -- and
        therefore which model id -- answers.
        """
        self.started = True
        if model:
            self.model = model
        return frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self._message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

    def _close_text(self) -> list[str]:
        if self._text_index is None:
            return []
        out = [
            frame(
                "content_block_stop",
                {"type": "content_block_stop", "index": self._text_index},
            )
        ]
        self._text_index = None
        return out

    def text(self, chunk: str) -> list[str]:
        out: list[str] = []
        if self._text_index is None:
            self._text_index = self._next_index
            self._next_index += 1
            out.append(
                frame(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": self._text_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
        out.append(
            frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._text_index,
                    "delta": {"type": "text_delta", "text": chunk},
                },
            )
        )
        return out

    def tool_fragments(self, fragments: list[dict[str, Any]]) -> list[str]:
        """Driver tool fragments as `input_json_delta` runs.

        A fragment is not parseable JSON on its own -- that is the
        property that breaks naive readers -- so the arguments are
        forwarded exactly as they arrive and the client reassembles
        them, which is what both wires ask for.
        """
        out: list[str] = []
        for position, fragment in enumerate(fragments):
            call_index = fragment.get("index")
            if not isinstance(call_index, int):
                call_index = position
            if call_index not in self._tool_index:
                # A new call. Close whatever is open first: Anthropic
                # allows one open block at a time, and a text block left
                # open is a stream a strict client rejects.
                out.extend(self._close_text())
                out.extend(self._close_tool_except(call_index))
                index = self._next_index
                self._next_index += 1
                self._tool_index[call_index] = index
                function = fragment.get("function") or {}
                out.append(
                    frame(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": index,
                            "content_block": {
                                "type": "tool_use",
                                "id": fragment.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
                                "name": function.get("name") or "",
                                # Empty on start; the arguments arrive
                                # as deltas, which is Anthropic's own
                                # framing and what lets a client show a
                                # call forming.
                                "input": {},
                            },
                        },
                    )
                )
            index = self._tool_index[call_index]
            arguments = (fragment.get("function") or {}).get("arguments")
            if arguments:
                out.append(
                    frame(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "input_json_delta", "partial_json": arguments},
                        },
                    )
                )
        return out

    def _close_tool_except(self, keep: int) -> list[str]:
        out: list[str] = []
        for call_index, index in list(self._tool_index.items()):
            if call_index == keep:
                continue
            out.append(frame("content_block_stop", {"type": "content_block_stop", "index": index}))
            del self._tool_index[call_index]
        return out

    def close_blocks(self) -> list[str]:
        """Every block that opened also closes, in the order it opened."""
        out = self._close_text()
        for _, index in sorted(self._tool_index.items(), key=lambda kv: kv[1]):
            out.append(frame("content_block_stop", {"type": "content_block_stop", "index": index}))
        self._tool_index.clear()
        return out

    def finish(self, *, reason: str, usage: Any = None) -> list[str]:
        out = self.close_blocks()
        delta: dict[str, Any] = {
            "type": "message_delta",
            "delta": {"stop_reason": reason, "stop_sequence": None},
        }
        if usage is not None:
            delta["usage"] = usage_block(usage)
        out.append(frame("message_delta", delta))
        out.append(frame("message_stop", {"type": "message_stop"}))
        return out

    def failed(self, message: str) -> list[str]:
        """A truncation, once the 200 is long gone.

        Past the first token the slot is committed and cannot fail over,
        so the stream stops and says why -- and then still emits
        `message_stop`, because a client's state machine is waiting for
        one and a stream that simply ends leaves it hanging.
        """
        out = self.close_blocks()
        out.append(
            frame(
                "error",
                {"type": "error", "error": {"type": "api_error", "message": message}},
            )
        )
        out.append(frame("message_stop", {"type": "message_stop"}))
        return out


def now_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
