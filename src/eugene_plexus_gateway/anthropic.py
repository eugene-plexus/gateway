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

import base64
import binascii
import json
import logging
import time
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import chat_contract, images, tokens
from ._generated.models import (
    AnthropicMessagesRequest,
    ChatCompletionMessage,
    ChatCompletionRequest,
    Function,
    FunctionDefinition,
    ImageContentPart,
    ImageUrl,
    NamedToolChoice,
    Role1,
    Stop,
    TextContentPart,
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

    from .dependencies import front_door_claims

    try:
        payload = front_door_claims(auth, token)
    except tokens.TokenError as e:
        raise Refusal(
            f"That key was rejected: {e}. Make a new one under Home -> Use it from your apps.",
            status=403,
            kind="authentication_error",
        ) from e

    if not payload.is_client:
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
            "This key is not registered with the current authority. Make a new one under "
            "Home -> Use it from your apps.",
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
# received the document is not answering the question that was asked.
# `image` left this set on 2026-09-23 -- it is carried now, on the path
# the OpenAI door's images already take, and a backend that cannot see
# it is never asked (see `_image_part`).
_REFUSED_BLOCK_TYPES = {"document"}

# Fields that arrive on every real request and have no equivalent here.
# Accepted for the measured Claude Code client; A2 reports ignored controls
# on a response header, rather than silently implying native support.
# A blanket unknown-field refusal passes every refusal test and then
# fails on the first real request, which is the trap this whole door was
# scoped around.
_DROPPED_TOP_LEVEL = (
    "thinking",
    "cache_control",
    "metadata",
    "context_management",
    # On every Claude Code request since agent-sdk 0.3.280 (captured
    # 2026-09-23) as `{"effort": "high"}`. Refused as an unknown field until
    # then, which failed the first request of every session.
    "output_config",
)

# The only `output_config` key this door accepts. Anything else --
# structured output's `format` above all -- changes what the answer is.
_ACCEPTED_OUTPUT_CONFIG = frozenset({"effort"})

# Fields whose presence means the caller wants something this control
# plane cannot do at all, as opposed to something it can ignore. `top_k`
# left this list on 2026-09-23: the driver's request carries it now.
_REFUSED_TOP_LEVEL = ("mcp_servers",)

# The opaque signature an omitted thinking block carries. Anthropic's own
# signature exists so the server keeps continuity when a client echoes a
# block back without its text; ours does the same job, readably, and says
# whose it is so nobody mistakes it for Anthropic's.
_SIGNATURE_PREFIX = "eugene-plexus-reasoning-v1:"


def thinking_display(raw: Mapping[str, Any]) -> str | None:
    """How this request asked to see the model's reasoning, if at all.

    None -- no `thinking` blocks: the request sent no `thinking`, sent
    null, or sent `{"type": "disabled"}`. Anthropic's own rule, and the
    one that keeps `content[0].text` pointing at the answer for the
    commonest line of SDK code there is.

    `"omitted"` -- blocks with the text empty and the reasoning in the
    signature. **Claude Code asks for this on every request** (measured,
    2.1.207), which is Anthropic's `display: "omitted"`.

    `"text"` -- blocks with the reasoning in them, for `"summarized"` or
    no `display`: ours is the whole reasoning, with nothing to condense
    it and nothing to hide.
    """
    thinking = raw.get("thinking")
    if not isinstance(thinking, Mapping) or thinking.get("type") == "disabled":
        return None
    return "omitted" if thinking.get("display") == "omitted" else "text"


def encode_signature(reasoning: str) -> str:
    return _SIGNATURE_PREFIX + base64.b64encode(reasoning.encode("utf-8")).decode("ascii")


def decode_signature(signature: Any) -> str | None:
    """The reasoning an omitted block carried, or None for anyone else's.

    A signature without our prefix is Anthropic's, or garbage, and is
    ignored rather than guessed at; so is one whose body will not decode.
    """
    if not isinstance(signature, str) or not signature.startswith(_SIGNATURE_PREFIX):
        return None
    try:
        text = base64.b64decode(signature[len(_SIGNATURE_PREFIX) :], validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return None
    return text or None


def thinking_block(reasoning: str, display: str) -> dict[str, Any]:
    if display == "omitted":
        return {"type": "thinking", "thinking": "", "signature": encode_signature(reasoning)}
    return {"type": "thinking", "thinking": reasoning, "signature": ""}


def compatibility_headers(raw: Mapping[str, Any]) -> dict[str, str]:
    def not_honoured(name: str) -> bool:
        value = raw.get(name)
        if name == "metadata" or value is None:
            return False
        if name != "thinking":
            return True
        # `thinking` decides whether reasoning is returned and whether its
        # text is shown, and both are honoured. What is not: a token budget,
        # and `disabled` as an instruction to the model -- the model thinks
        # regardless, and only the operator's `thinkingMode` changes that.
        return not isinstance(value, Mapping) or (
            value.get("budget_tokens") is not None or value.get("type") == "disabled"
        )

    ignored = [name for name in _DROPPED_TOP_LEVEL if not_honoured(name)]

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
    """Refuse a document anywhere it can appear.

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


def _image_part(block: Any, where: str, budget: images.ImageBudget) -> ImageContentPart:
    """An Anthropic `image` block as the part the OpenAI door already carries.

    From there it is the shared path's: the same limits, the same
    PNG/JPEG check, and routing only to a backend that confirms image
    input, so a model that cannot see it is refused rather than asked.
    Named in the caller's own coordinates (`messages.2.content.0...`),
    because that is the list the caller can find the picture in.
    """
    field = f"{where}.source"
    source = _field(block, "source")
    if not isinstance(source, Mapping) or source.get("type") != "base64":
        # A URL is never fetched, as on the OpenAI door: the gateway
        # dialling an address a caller names is the request forgery R2.4
        # closed for node addresses. A Files API id names a store that
        # exists only at Anthropic.
        raise Refusal(
            f"{field}: send the image inline as base64. URLs are not fetched and "
            "file references cannot be resolved here."
        )
    try:
        url = images.data_url_from_base64(source.get("media_type"), source.get("data"), field)
        budget.admit(url, field)
    except images.ImageRefusal as e:
        raise Refusal(str(e)) from None
    return ImageContentPart(type="image_url", image_url=ImageUrl(url=url))


def _tool_result_images(
    block: Any, where: str, budget: images.ImageBudget
) -> list[ImageContentPart]:
    """The pictures a tool returned -- Claude Code's `Read` of an image.

    Captured 2026-09-23: that `tool_result` holds a lone `image` block and
    no text. An OpenAI `tool` message carries text only and image parts
    ride on user messages, so they cannot stay where they arrived.
    """
    content = _field(block, "content")
    if not isinstance(content, list):
        return []
    return [
        _image_part(inner, f"{where}.content.{i}", budget)
        for i, inner in enumerate(content)
        if _field(inner, "type") == "image"
    ]


def _joined(parts: list[TextContentPart | ImageContentPart]) -> Any:
    """A turn's content: a string as ever, or ordered parts when it holds a picture.

    Adjacent text blocks join with a blank line either way, which is how
    this door has always joined them, so a turn's words read the same to
    the model whether or not an image sits beside them.
    """
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


def _parallel_tool_calls(choice: Any) -> bool | None:
    """`disable_parallel_tool_use` as the backend's `parallel_tool_calls`.

    Opposite sense, and only when the caller said something: absent
    stays absent, because llama.cpp and OpenAI disagree about the default
    and filling one in would change what one of them does. **Silently
    dropped until 2026-09-23** -- accepted by the loose model, read by
    nothing -- which is the one outcome a setting that changes the
    answer must never have.
    """
    disable = getattr(choice, "disable_parallel_tool_use", None) if choice is not None else None
    return None if disable is None else not disable


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


def translate_request(
    raw: Mapping[str, Any], *, max_images: int = images.DEFAULT_MAX_IMAGES
) -> ChatCompletionRequest:
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

    output_config = raw.get("output_config")
    if isinstance(output_config, Mapping):
        for key in output_config:
            if key not in _ACCEPTED_OUTPUT_CONFIG:
                raise Refusal(
                    f"output_config.{chat_contract._field_name(str(key))}: unsupported "
                    "setting. Only `effort` is accepted here (and not enforced: the "
                    "model's settings profile governs how it thinks)."
                )

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

    budget = images.ImageBudget(max_images)
    for turn_index, turn in enumerate(body.messages):
        blocks = _blocks(turn.content)
        _refuse_unsupported_blocks(blocks, where=f"{turn.role.value} message")
        role = turn.role.value

        parts: list[TextContentPart | ImageContentPart] = []
        # Pictures a tool returned, carried on this turn's user message
        # ahead of the person's own words: they answer the call above.
        hoisted: list[TextContentPart | ImageContentPart] = []
        reasoning_parts: list[str] = []
        calls: list[ToolCall] = []
        results: list[ChatCompletionMessage] = []

        for block_index, block in enumerate(blocks):
            where = f"messages.{turn_index}.content.{block_index}"
            kind = _field(block, "type")
            if kind == "text":
                if _field(block, "text"):
                    parts.append(TextContentPart(type="text", text=str(_field(block, "text"))))
            elif kind == "image":
                if role != "user":
                    raise Refusal(
                        f"{where}: images are accepted on user turns and inside tool "
                        f"results, not on a {role} turn."
                    )
                parts.append(_image_part(block, where, budget))
            elif kind == "thinking" and turn.role.value == "assistant":
                # A thinking block we returned, handed back as Anthropic
                # tells clients to: its text when it was shown, else the
                # reasoning our signature carried. Either way it becomes
                # this turn's reasoning, which llama.cpp renders back into
                # the prompt for templates that keep it -- so a tool loop
                # resumes with the model's own thinking. Anyone else's
                # signature is ignored. `redacted_thinking` falls through
                # to the drop below: nothing in it is readable here.
                thought = _field(block, "thinking") or decode_signature(_field(block, "signature"))
                if thought:
                    reasoning_parts.append(str(thought))
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
                call_id = str(_field(block, "tool_use_id") or "")
                text = _tool_result_text(block)
                pictures = _tool_result_images(block, where, budget)
                if pictures and role != "user":
                    raise Refusal(f"{where}: a tool result belongs on a user turn.")
                if pictures:
                    # The tool message still answers its call and says where
                    # the picture went: a model reading an empty result and
                    # then an unexplained image cannot tell which call made it.
                    many = len(pictures) > 1
                    note = (
                        f"[The tool returned {len(pictures)} images; they are attached to "
                        "the next user message.]"
                        if many
                        else "[The tool returned an image; it is attached to the next user "
                        "message.]"
                    )
                    text = f"{text}\n{note}" if text else note
                    label = "The images" if many else "The image"
                    hoisted.append(
                        TextContentPart(
                            type="text", text=f"{label} returned by tool call {call_id}:"
                        )
                    )
                    hoisted.extend(pictures)
                results.append(
                    ChatCompletionMessage(role=Role1.tool, content=text, tool_call_id=call_id)
                )

        # Results first: they answer the assistant turn above them, and
        # an OpenAI backend expects every `tool` message to follow the
        # assistant message that asked for it with nothing between. A
        # tool's pictures then follow them at once, on the user message.
        messages.extend(results)
        content = _joined(hoisted + parts)
        if content is not None or calls or reasoning_parts:
            messages.append(
                ChatCompletionMessage(
                    role=Role1(role),
                    content=content,
                    tool_calls=calls or None,
                    reasoning_content="\n\n".join(reasoning_parts) if reasoning_parts else None,
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
        # Refused with a 400 from A2 until 2026-09-23: the driver's request
        # had nowhere to put it. Both local engines read it.
        top_k=body.top_k,
        stop=Stop(body.stop_sequences) if body.stop_sequences else None,
        stream=bool(body.stream),
        tools=_tools(body.tools),
        tool_choice=_tool_choice(body.tool_choice),
        parallel_tool_calls=_parallel_tool_calls(body.tool_choice),
    )


def translate_count_request(
    raw: Mapping[str, Any], *, max_images: int = images.DEFAULT_MAX_IMAGES
) -> ChatCompletionRequest:
    """A `count_tokens` body, translated exactly as a message is.

    It has no `max_tokens` -- Anthropic's count endpoint takes none and
    Claude Code sends none (captured) -- so one is supplied, and the shared
    translation, every refusal in it included, applies unchanged. Nothing
    is generated, so the number is never used for anything.
    """
    return translate_request({"max_tokens": 1, **raw}, max_images=max_images)


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
    """Token counts, renamed -- and split the way Anthropic splits them.

    **Anthropic's `input_tokens` excludes cached input**: a client that
    computes context usage adds `input_tokens`, `cache_read_input_tokens`
    and `cache_creation_input_tokens`. So when the backend reports how
    much of the prompt came from its cache (llama.cpp and vLLM do), that
    part moves to `cache_read_input_tokens` and the three still sum to the
    prompt. Reporting the cached count on top of an unreduced
    `input_tokens` would double it in every such client.

    Cache counts are an honest zero when the backend said nothing, and
    creation is always zero: a local engine's cache is not something a
    request pays to write.
    """
    prompt = getattr(usage, "promptTokens", 0) or 0
    cached = getattr(usage, "cachedPromptTokens", None) or 0
    cached = min(cached, prompt)
    return {
        "input_tokens": prompt - cached,
        "output_tokens": getattr(usage, "completionTokens", 0) or 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
    }


def content_blocks(
    content: str | None,
    tool_calls: Any,
    *,
    reasoning: str | None = None,
    display: str | None = None,
) -> list[dict[str, Any]]:
    """A thinking block when asked for, then text and tool calls, text first."""
    blocks: list[dict[str, Any]] = []
    if reasoning and display is not None:
        blocks.append(thinking_block(reasoning, display))
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
    *,
    model: str,
    content: str | None,
    tool_calls: Any,
    finish: Any,
    usage: Any,
    reasoning: str | None = None,
    display: str | None = None,
    stop_sequence: str | None = None,
) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks(content, tool_calls, reasoning=reasoning, display=display),
        "stop_reason": stop_reason(finish),
        # Named when the backend named it (vLLM does, llama.cpp does not)
        # and only beside the stop reason that says a sequence matched.
        "stop_sequence": stop_sequence if stop_reason(finish) == "stop_sequence" else None,
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

    def __init__(self, model: str, *, display: str | None = None) -> None:
        self.model = model
        self.started = False
        self._next_index = 0
        self._text_index: int | None = None
        self._tool_index: dict[int, int] = {}
        self._message_id = f"msg_{uuid.uuid4().hex}"
        #: How the request asked to see reasoning -- see `thinking_display`.
        #: None hides it: its events still start the message (they are
        #: output, and past the commit point) but open no block.
        self._display = display
        self._thinking_index: int | None = None
        #: Under `"omitted"` the text is not streamed; it is held here and
        #: sent as the block's closing `signature_delta`.
        self._withheld: list[str] = []

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

    def thinking(self, chunk: str) -> list[str]:
        """Reasoning as a `thinking` block, or as nothing when not asked for.

        Opens its block ahead of the answer and is closed before the next
        kind of block starts -- Anthropic allows one open block at a time.
        Under `"omitted"` no `thinking_delta` is sent at all; the text is
        withheld and travels in the closing `signature_delta`, which is
        what Anthropic's own omitted stream looks like.
        """
        if self._display is None:
            return []
        out: list[str] = []
        if self._thinking_index is None:
            out.extend(self._close_text())
            out.extend(self._close_tool_except(None))
            self._thinking_index = self._next_index
            self._next_index += 1
            out.append(
                frame(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": self._thinking_index,
                        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                    },
                )
            )
        if self._display == "omitted":
            self._withheld.append(chunk)
            return out
        out.append(
            frame(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._thinking_index,
                    "delta": {"type": "thinking_delta", "thinking": chunk},
                },
            )
        )
        return out

    def _close_thinking(self) -> list[str]:
        if self._thinking_index is None:
            return []
        out: list[str] = []
        if self._display == "omitted" and self._withheld:
            out.append(
                frame(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._thinking_index,
                        "delta": {
                            "type": "signature_delta",
                            "signature": encode_signature("".join(self._withheld)),
                        },
                    },
                )
            )
        out.append(
            frame(
                "content_block_stop",
                {"type": "content_block_stop", "index": self._thinking_index},
            )
        )
        self._thinking_index = None
        self._withheld = []
        return out

    def text(self, chunk: str) -> list[str]:
        out: list[str] = self._close_thinking()
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
                # allows one open block at a time, and a text or thinking
                # block left open is a stream a strict client rejects.
                out.extend(self._close_thinking())
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

    def _close_tool_except(self, keep: int | None) -> list[str]:
        out: list[str] = []
        for call_index, index in list(self._tool_index.items()):
            if call_index == keep:
                continue
            out.append(frame("content_block_stop", {"type": "content_block_stop", "index": index}))
            del self._tool_index[call_index]
        return out

    def close_blocks(self) -> list[str]:
        """Every block that opened also closes, in the order it opened."""
        out = self._close_thinking()
        out.extend(self._close_text())
        for _, index in sorted(self._tool_index.items(), key=lambda kv: kv[1]):
            out.append(frame("content_block_stop", {"type": "content_block_stop", "index": index}))
        self._tool_index.clear()
        return out

    def finish(
        self, *, reason: str, usage: Any = None, stop_sequence: str | None = None
    ) -> list[str]:
        out = self.close_blocks()
        delta: dict[str, Any] = {
            "type": "message_delta",
            "delta": {
                "stop_reason": reason,
                "stop_sequence": stop_sequence if reason == "stop_sequence" else None,
            },
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
