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

from collections.abc import Iterable, Sequence

#: Row 37 review: a model name longer than any id the contract allows is never
#: matched against a list. A caller chooses the name, and the count-tokens
#: doors pass it on uncapped.
MAX_MODEL_NAME = 256


def matches(pattern: str, value: str) -> bool:
    """`*` matches any run of characters, `/` included; nothing else is special.

    Deliberately not `fnmatch`: `[`, `?` and `\\` are ordinary characters in
    a model id, and one wildcard is a rule an operator can hold in their head.

    Linear in the lengths (row 37 review): the parts between `*` are found left
    to right, each as early as it can be, which is exact when `*` is the only
    wildcard. A regular expression of `.*` runs backtracked for a minute on a
    26-character pattern of many stars against a 40-character name, on the
    event loop.
    """
    if "*" not in pattern:
        return pattern == value
    parts = pattern.split("*")
    first, last = parts[0], parts[-1]
    if len(value) < len(first) + len(last):
        return False
    if not value.startswith(first) or not value.endswith(last):
        return False
    position, end = len(first), len(value) - len(last)
    for part in parts[1:-1]:
        if not part:
            continue
        found = value.find(part, position, end)
        if found < 0:
            return False
        position = found + len(part)
    return True


class ModelScope:
    """Row 37: what a request with a verified person may reach: the key's
    `allowedModels` and the model list of every group of theirs that sets one
    (call R37-SELF, Choice 3), all of them.

    A `role:<role>` name is checked against the key's list only, as before:
    the person's groups allow models, and a role's models are checked one by
    one when the role resolves. Used wherever a key's list is (`permits`), so
    every place that already applies the key's list applies the person's.
    """

    __slots__ = ("key", "limits")

    def __init__(
        self, key: Iterable[str] | None, limits: Sequence[tuple[str, Sequence[str]]]
    ) -> None:
        self.key = list(key) if key is not None else None
        #: `(group name, its models)`, in the root's order (by group name).
        self.limits = [(name, list(models)) for name, models in limits]

    def permits(self, model: str) -> bool:
        if not permits(self.key, model):
            return False
        return self.excluded_by(model) is None

    def excluded_by(self, model: str) -> str | None:
        """The first of the person's groups whose list leaves `model` out,
        for the refusal that names it; None when every list allows it."""
        if model.startswith("role:"):
            return None
        for name, models in self.limits:
            if not permits(models, model):
                return name
        return None


#: A key's list, a request's scope (row 37), or None for every model.
type Allowed = Iterable[str] | ModelScope | None


def permits(allowed: Allowed, model: str) -> bool:
    """Whether a key's `allowedModels` lets it use `model`. None permits all."""
    if allowed is None:
        return True
    if len(model) > MAX_MODEL_NAME:
        return False
    if isinstance(allowed, ModelScope):
        return allowed.permits(model)
    return any(matches(entry, model) for entry in allowed)
