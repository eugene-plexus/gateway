"""What the chosen backend does not honour, said in a door's own words.

A backend's driver lists, in its capabilities, the settings it is sent and is
measured not to honour, by value (`capabilities.settingsNotHonoured`,
`docs/design/tools-assumed.md`, 2026-10-12): Strata ignores `tool_choice`
`required` and a named function, and `parallel_tool_calls: false`. A measured
gap is sent and named, not refused. The list is static per backend, so a stream
reads it from the backend the first attempt goes to **before anything is sent**
(`TieredClient.model_asked_first`: not always the balancer's first pick), and
its head never waits for the backend to speak (an earlier version read a header
off the driver's response instead, which held every stream's head until its
first frame, and broke the Responses door's opening). **That header is not read
by any module here.** A failover after the head is sent cannot change what a
stream's head says; **a whole answer has no head to send early**, so it is named
after the call from the backend that served it (`_Serving.not_honoured_by_answer`).

A door names what the caller sent, with a value the list names, on its own
`x-eugene-plexus-ignored-settings`, spelled the way its caller spelled the
request, after the settings the door itself dropped, so one header lists
everything this request was not given. **Only what the caller sent**: the
gateway's own settings (a search loop's forced first turn) are not the
caller's, and are never named.

Three things live here and nowhere else: matching a listing against a request,
the driver-to-door spelling (of one field, and of a whole request's list of them),
and the merge.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from ._generated.driver_models import GenerateRequest, NamedToolChoice, SettingNotHonoured

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
#: Anthropic's `tool_choice` carries the parallel switch inside it, with the
#: opposite sense (`disable_parallel_tool_use: true` is `parallel_tool_calls:
#: false`), so that is where a caller of this door looks for it.
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

_CAMEL = re.compile(r"(?<=[a-z0-9])([A-Z])")


def spelled(name: str, door: str = "chat") -> str:
    """One driver field as `door`'s caller spells it.

    A name nobody has listed (a field a newer driver names) is the chat
    door's guess at it: snake_case, which is how that door spells every
    setting it carries beyond OpenAI's one-word ones.
    """
    return (
        _DOORS.get(door, {}).get(name) or WIRE_NAMES.get(name) or _CAMEL.sub(r"_\1", name).lower()
    )


#: The settings said first when a request's are listed: the tools, how they are
#: chosen, whether they are called in parallel. The rest follow alphabetically.
_TOOLS_FIRST = ("tools", "toolChoice", "parallelToolCalls")


def carried(settings: Iterable[str]) -> list[str]:
    """What a request carries, in its caller's words, in an order a person reads.

    `settings` are the driver's field names (`GenerateRequest.callerSettings`): each
    is said the way the chat door spells it (`WIRE_NAMES`), the tools first and the
    settings that ride with them next, the others alphabetically, so the same
    request is worded the same way every time (`callerSettings` is a set to the
    code that reads it). Every setting the caller sent is named, `tools` included
    whenever the caller sent the key, with a list or not: an empty one is still
    something the caller said.
    """
    sent_names = set(settings)
    first = [WIRE_NAMES.get(name, name) for name in _TOOLS_FIRST if name in sent_names]
    rest = sorted(WIRE_NAMES.get(name, name) for name in sent_names.difference(_TOOLS_FIRST))
    return [*first, *rest]


def sent(request: GenerateRequest, setting: str) -> str | bool | None:
    """What the caller sent for `setting`, spelled as a listing's `values` spell it.

    None when the caller did not send it (`callerSettings` is the caller's own
    list: never a profile's or the install's default, and never what the
    gateway's own search loop asks of a backend on a request it made itself) or
    when its value is not one a listing can name. A tool choice is its mode:
    `auto`, `none`, `required`, or `function` for a named function. A boolean is
    itself, and `false` is a value: it is sent, and a backend may ignore it.
    """
    if setting not in (request.callerSettings or []):
        return None
    value: Any = getattr(request, setting, None)
    if isinstance(value, NamedToolChoice):
        return "function"
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return str(value)  # a `StrEnum` member (`ToolChoice`) as its plain value
    return None


def not_honoured(
    listed: Iterable[SettingNotHonoured] | None, request: GenerateRequest
) -> tuple[str, ...]:
    """The driver's field names of what `request` carries and the backend does not honour.

    A setting is named when the caller sent it **with a value the listing
    names**: `tool_choice: required` is named for a backend that lists
    `toolChoice` with `required`, and `tool_choice: auto` is not. Each once, in
    the order the listing gives them. A listing that names nothing, or a
    backend with no listing, names nothing: absent means everything carried is
    honoured.

    Only names this gateway built reach a header: a name is returned only if it
    is in the request's own `callerSettings`, so a driver's words are matched
    against what the caller sent and never passed on.

    **`tool_choice: none` to a backend that never reads `tool_choice` does not
    name `parallel_tool_calls`.** Such a backend lists `toolChoice` and cannot be
    told `none`, so its driver honours it as it can, by sending no tools and none
    of what rides with them (`inference-driver`'s `_payload_for`; the driver's
    own header leaves it out the same way). Nothing is left on the request for
    the backend to ignore, though `parallel_tool_calls: false` is in its listing.
    """
    entries = list(listed or ())
    # A listing names `toolChoice` exactly for a backend that never reads it.
    unread = any(entry.setting == "toolChoice" for entry in entries)
    tools_unsent = unread and sent(request, "toolChoice") == "none"
    named: dict[str, None] = {}
    for entry in entries:
        if tools_unsent and entry.setting == "parallelToolCalls":
            continue
        value = sent(request, entry.setting)
        # `false` is a value and a boolean is not a string, so `value in values`
        # does not take `"false"` for it.
        if value is not None and value in entry.values:
            named.setdefault(entry.setting)
    return tuple(named)


def merge(own: str | None, names: Iterable[str], door: str = "chat") -> str | None:
    """A door's own header value with the backend's names added.

    The door's own come first, then the backend's in the order its listing gave
    them, each once however it was named: the door's `tool_choice` and the
    backend's `toolChoice` are one entry. None when there is nothing to name,
    so a header is never sent empty.
    """
    named: dict[str, None] = {}
    for part in (own or "").split(","):
        if part.strip():
            named.setdefault(part.strip())
    for name in names:
        named.setdefault(spelled(name, door))
    return ", ".join(named) or None


def added_to(
    headers: Mapping[str, str], names: Iterable[str], door: str = "chat"
) -> dict[str, str]:
    """`headers`, with the backend's names merged into its ignored-settings
    entry: added when the door had none, left alone when there is nothing to add."""
    merged = merge(headers.get(HEADER), names, door)
    out = {name: value for name, value in headers.items() if name != HEADER}
    if merged is not None:
        out[HEADER] = merged
    return out


__all__ = [
    "HEADER",
    "WIRE_NAMES",
    "added_to",
    "carried",
    "merge",
    "not_honoured",
    "sent",
    "spelled",
]
