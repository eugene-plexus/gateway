"""`allowedModels` entries: an exact id, or a pattern with `*` (P1, 2026-09-27).

`*` matches any run of characters, `/` included, and is the only wildcard,
so `openrouter/*` allows every model of the account named `openrouter`
(`openrouter/anthropic/claude-opus-5.5`) and nothing else. Without it,
scoping a key to one account means typing out hundreds of names.

**The same function is copied into the agent and the control root**, which
decide admission for a key before this gateway does; the inference-driver's
catalogue filters use it too. Components share schemas, not code, so it is
duplicated rather than imported -- and it is small enough that the copies
are checked by reading them side by side.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


def matches(pattern: str, value: str) -> bool:
    """`*` matches any run of characters, `/` included; nothing else is special.

    Deliberately not `fnmatch`: `[`, `?` and `\\` are ordinary characters in
    a model id, and one wildcard is a rule an operator can hold in their head.
    """
    if "*" not in pattern:
        return pattern == value
    parts = [re.escape(part) for part in pattern.split("*")]
    return re.fullmatch(".*".join(parts), value, flags=re.DOTALL) is not None


def permits(allowed: Iterable[str] | None, model: str) -> bool:
    """Whether a key's `allowedModels` lets it use `model`. None permits all."""
    if allowed is None:
        return True
    return any(matches(entry, model) for entry in allowed)
