"""What the serving backend does not honour, said in a door's own words.

A driver answers `POST /v1/generate` and `/v1/generate/stream` with
`x-eugene-plexus-ignored-settings`: the request fields -- in its contract's
spelling, `toolChoice` -- that its backend was sent and is known not to honour
(`docs/design/tools-assumed.md`, 2026-10-12). Strata ignores `tool_choice`
`required` and a named function, measured, so those are sent as it takes them
and named rather than refused. A door says the same on its own header of that
name, spelled the way its caller spelled the request, after the settings the
door itself dropped, so one header lists everything this request was not given.

Three things live here and nowhere else: reading the driver's header, the
driver-to-door spelling, and the merge.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping

log = logging.getLogger(__name__)

HEADER = "x-eugene-plexus-ignored-settings"

#: A driver's request field as the chat door spells it. The names it leaves out
#: are one word and the same on both sides (`temperature`, `seed`, `stop`).
#: Admission words its refusals from this table too.
WIRE_NAMES: dict[str, str] = {
    "maxTokens": "max_tokens",
    "topP": "top_p",
    "toolChoice": "tool_choice",
    "responseFormat": "response_format",
    "topK": "top_k",
    "minP": "min_p",
    "frequencyPenalty": "frequency_penalty",
    "presencePenalty": "presence_penalty",
    "parallelToolCalls": "parallel_tool_calls",
    "logitBias": "logit_bias",
    "reasoningEffort": "reasoning_effort",
    "webSearchOptions": "web_search_options",
    "audioOutput": "audio",
    "reportProgress": "stream_options.include_progress",
}

#: Where the Messages door's caller says it differently. Only what that door
#: reads and sends on is here; everything else is the chat door's spelling.
_MESSAGES: dict[str, str] = {
    "stop": "stop_sequences",
    "responseFormat": "output_config.format",
    "parallelToolCalls": "tool_choice.disable_parallel_tool_use",
}

#: Where the Responses door's caller says it differently.
_RESPONSES: dict[str, str] = {
    "maxTokens": "max_output_tokens",
    "responseFormat": "text.format",
}

_DOORS: dict[str, dict[str, str]] = {"chat": {}, "messages": _MESSAGES, "responses": _RESPONSES}

#: What a request field's name can be. A driver's words go out on the caller's
#: response header, so anything else is left off it: this is a field name or it
#: is nothing, as the caller's own keys are never reflected raw either.
_FIELD = re.compile(r"[A-Za-z][A-Za-z0-9_.]{0,63}")
_CAMEL = re.compile(r"(?<=[a-z0-9])([A-Z])")


def parse(value: str | None) -> tuple[str, ...]:
    """The driver's header as its field names: each once, in the order named.

    A driver is not trusted to send only names, so a token that is not one is
    dropped (and noted at debug, as an unreadable progress frame is) rather
    than passed on or allowed to fail an answer that was fine.
    """
    kept: dict[str, None] = {}
    for token in (value or "").split(","):
        name = token.strip()
        if not name:
            continue
        if _FIELD.fullmatch(name):
            kept.setdefault(name)
        else:
            log.debug("unreadable entry in a driver's %s: %r", HEADER, name[:40])
    return tuple(kept)


def spelled(name: str, door: str = "chat") -> str:
    """One driver field as `door`'s caller spells it.

    A name nobody has listed (a field a newer driver names) is the chat
    door's guess at it: snake_case, which is how that door spells every
    setting it carries beyond OpenAI's one-word ones.
    """
    return (
        _DOORS.get(door, {}).get(name) or WIRE_NAMES.get(name) or _CAMEL.sub(r"_\1", name).lower()
    )


def merge(own: str | None, backend: Iterable[str], door: str = "chat") -> str | None:
    """A door's own header value with the backend's names added.

    The door's own come first, then the backend's in the order its driver gave
    them, each once however it was named: the door's `tool_choice` and the
    backend's `toolChoice` are one entry. None when there is nothing to name,
    so a header is never sent empty.
    """
    named: dict[str, None] = {}
    for part in (own or "").split(","):
        if part.strip():
            named.setdefault(part.strip())
    for name in backend:
        named.setdefault(spelled(name, door))
    return ", ".join(named) or None


def added_to(
    headers: Mapping[str, str], backend: Iterable[str], door: str = "chat"
) -> dict[str, str]:
    """`headers`, with the backend's names merged into its ignored-settings
    entry: added when the door had none, left alone when there is nothing to add."""
    merged = merge(headers.get(HEADER), backend, door)
    out = {name: value for name, value in headers.items() if name != HEADER}
    if merged is not None:
        out[HEADER] = merged
    return out


__all__ = ["HEADER", "WIRE_NAMES", "added_to", "merge", "parse", "spelled"]
