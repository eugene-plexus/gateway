"""What a driver's listing says a model can do, read in one place.

Tools are assumed and turned off only on evidence (Troy, 2026-10-12,
`docs/design/tools-assumed.md`): "we should always assume in 2026 that a
model & engine can use tools". The gateway used to read an unreported answer
as no, so a model on an engine nobody had measured, or a custom endpoint a
person was testing, was refused its tools for the developer's lack of a test.

Three answers a listing can give, and they are not the same question:

- **Is it a chat model?** (`chats`) Tools are a chat model's. A model that
  names its surfaces and leaves chat out (an embedding, rerank or speech model)
  is sent to the door that serves it, by name, and is never asked about tools.
- **Did its driver say it calls tools?** (`said_tools`) `capabilities.toolCalling`:
  true, false, or nothing. Only false is evidence.
- **Is it taken to?** (`calls_tools`) The two together: yes for a chat model
  whose driver does not say, whatever its list of settings holds.
"""

from __future__ import annotations

from ._generated.driver_models import DriverModel

#: The settings that come with a request that carries tools. A backend whose
#: driver says nothing of tool calling is taken to call tools, and a driver that
#: reports nothing is not a no, so these two are admitted for it whatever its
#: `supportedSettings` lists (`admission.permitted`). A backend that does say
#: keeps the rule that it lists the settings it takes.
TOOL_SETTINGS = frozenset({"toolChoice", "parallelToolCalls"})


def chats(model: DriverModel | None) -> bool:
    """Whether this entry is a chat model.

    One whose surfaces are listed and leave chat out is not. An entry with no
    surfaces to read (no entry, or an empty list: a model whose kind nobody
    could decide) is not shown to be anything else, so it is taken to be one, as
    every model was before surfaces and as unknown is yes for tools.
    """
    surfaces = getattr(model, "surfaces", None)
    return not surfaces or "chat" in surfaces


def said_tools(model: DriverModel | None) -> bool | None:
    """What the model's driver said of tool calling: `capabilities.toolCalling`,
    or None when it said nothing (no entry, no capabilities, no field)."""
    said = getattr(getattr(model, "capabilities", None), "toolCalling", None)
    return None if said is None else bool(said)


def calls_tools(model: DriverModel | None) -> bool:
    """Whether this model is taken to call tools.

    Yes unless its driver says no: `capabilities.toolCalling: false` is the
    only report there is, and a driver gives it only on evidence (a backend
    that refused tools in its own words, a provider's listing, no way to carry
    them to the engine). An older driver that reports nothing, a listing with
    no capabilities and a model with no entry at all count as models that do.

    **Of a chat model**, which is what the rule is about: an entry that names
    its surfaces and leaves chat out (an embedding or a speech model) is not
    one that calls tools because nothing said it does not.
    """
    said = said_tools(model)
    return chats(model) if said is None else said


def tools_refused(model: str, backends: int = 1) -> str:
    """The refusal of a request that carries tools to a model whose driver says
    they cannot be carried to it.

    **In the driver's words, never as the model failing**: a driver reports
    `toolCalling: false` only on evidence (its backend refused tools in its own
    words, a provider's listing says so, or it has no way to carry them to the
    engine), and the gateway repeats what was reported. One sentence for the
    admission check and for the front door, so the two cannot drift.
    """
    reports = (
        "its driver reports that tools cannot be carried to it"
        if backends == 1
        else "every driver serving it reports that tools cannot be carried to it"
    )
    return (
        f"Tools cannot be sent to {model!r}: {reports} (capabilities.toolCalling: false), so "
        "the request was refused rather than answered without them. "
        "GET /v1/models reports x_eugene_plexus.tool_calling per model. "
        "Nothing was forwarded or woken."
    )
