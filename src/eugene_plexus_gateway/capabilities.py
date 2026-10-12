"""What a driver's listing says a model can do, read in one place.

Tools are assumed and turned off only on evidence (Troy, 2026-10-12,
`docs/design/tools-assumed.md`): "we should always assume in 2026 that a
model & engine can use tools". The gateway used to read an unreported answer
as no, so a model on an engine nobody had measured, or a custom endpoint a
person was testing, was refused its tools for the developer's lack of a test.
"""

from __future__ import annotations

from ._generated.driver_models import DriverModel


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
    said = getattr(getattr(model, "capabilities", None), "toolCalling", None)
    if said is not None:
        return bool(said)
    surfaces = getattr(model, "surfaces", None)
    return not surfaces or "chat" in surfaces
