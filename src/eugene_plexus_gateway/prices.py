"""Row 40: how this gateway sees a model's price, for the root that counts it.

The control root keeps the owner's price for each model and counts every
request of the install; it prices a request itself. What it cannot know is
what this gateway's drivers say: whether every backend serving a model is
local (then a model with no owner's rate costs nothing), and the price its
provider publishes (`DriverModel.price`: OpenRouter's listing, or Eugene's
table for OpenAI, Anthropic and Gemini). This module says that, as a
`ModelPriceHint`, at `acquire` for the model asked for and at `release` for
the model that served it.

A model two backends serve is priced at the highest of their prices, and has
no provider price when one of them has none: a request may land on either,
and a price that is too low would let a money allowance be passed. Integers
throughout: a price is millionths of a dollar for a million tokens.
"""

from __future__ import annotations

from typing import Any

from ._generated.models import ProviderModelPrice

#: The hint for a model this gateway does not route: its cost is unknown.
UNKNOWN: dict[str, Any] = {"locality": "unknown"}


def _locality(backends: list[Any]) -> str:
    said = {str(getattr(b.info.locality, "value", b.info.locality) or "unknown") for b in backends}
    if "external" in said:
        return "external"
    if said == {"local"}:
        return "local"
    return "unknown"


def _price(backend: Any) -> tuple[int, int] | None:
    found = getattr(getattr(backend, "model", None), "price", None)
    if found is None:
        return None
    return int(found.inputPerMillionMicros), int(found.outputPerMillionMicros)


def provider_price(backends: list[Any]) -> tuple[int, int] | None:
    """The provider's price of a model these backends serve: the highest of
    each, and None when any of them has none (or there are none)."""
    prices = [_price(b) for b in backends]
    if not prices or any(p is None for p in prices):
        return None
    known = [p for p in prices if p is not None]
    return max(p[0] for p in known), max(p[1] for p in known)


def hint_for(table: Any, model: str | None) -> dict[str, Any]:
    """The `ModelPriceHint` for `model` as the routing table serves it now."""
    if table is None or not model:
        return dict(UNKNOWN)
    try:
        backends = list(table.backends_for(model))
    except Exception:  # A model the table cannot resolve has no price here.
        return dict(UNKNOWN)
    if not backends:
        return dict(UNKNOWN)
    hint: dict[str, Any] = {"locality": _locality(backends)}
    price = provider_price(backends)
    if price is not None:
        hint["providerPrice"] = {
            "inputPerMillionMicros": price[0],
            "outputPerMillionMicros": price[1],
        }
    return hint


def listed(backends: list[Any]) -> ProviderModelPrice | None:
    """`ModelRoutingInfo.provider_price`: the provider's price, with where the
    drivers read it and when, or None (*no price listed*, never free)."""
    price = provider_price(backends)
    if price is None:
        return None
    sources = {
        str(getattr(getattr(b.model.price, "source", None), "value", b.model.price.source))
        for b in backends
    }
    dates = sorted(str(b.model.price.asOf) for b in backends if b.model.price.asOf is not None)
    out: dict[str, Any] = {
        "input_per_million_micros": price[0],
        "output_per_million_micros": price[1],
    }
    if len(sources) == 1:
        out["source"] = sources.pop()
    if dates:
        out["as_of"] = dates[0]
    return ProviderModelPrice.model_validate(out)
