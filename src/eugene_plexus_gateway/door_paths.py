"""Which front door a request path is, when a door has a path parameter.

Every door until P5 was one fixed path, and the admission, CORS and
body-limit lists were plain sets of them. `/v1/videos/{video_id}` is the
first door whose path carries a value, so the lists hold the route's template
(which is also what the route-reading test compares) and a request path is
matched against it here, once, for every list.

`/v1/models/{model:path}` (P6) is the first whose value may itself hold
slashes (an account's `openrouter/anthropic/claude-opus-5.5`): a last
segment `{name:path}` stands for the rest of the path, one segment or more.
"""

from __future__ import annotations

from collections.abc import Iterable


def _segment(template: str, part: str) -> bool:
    return (template.startswith("{") and template.endswith("}") and bool(part)) or template == part


def matches(path: str, doors: Iterable[str]) -> bool:
    """Whether `path` is one of `doors`: a template segment (`{video_id}`)
    stands for exactly one non-empty segment, and a last `{name:path}` for
    one or more."""
    if path in doors:
        return True
    parts = path.split("/")
    for door in doors:
        if "{" not in door:
            continue
        template = door.split("/")
        if template[-1].endswith(":path}"):
            head = template[:-1]
            rest = parts[len(head) :]
            if (
                len(parts) > len(head)
                and all(_segment(t, p) for t, p in zip(head, parts, strict=False))
                and all(rest)
            ):
                return True
            continue
        if len(template) == len(parts) and all(
            _segment(t, p) for t, p in zip(template, parts, strict=True)
        ):
            return True
    return False
