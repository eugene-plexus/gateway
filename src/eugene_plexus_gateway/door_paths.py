"""Which front door a request path is, when a door has a path parameter.

Every door until P5 was one fixed path, and the admission, CORS and
body-limit lists were plain sets of them. `/v1/videos/{video_id}` is the
first door whose path carries a value, so the lists hold the route's template
(which is also what the route-reading test compares) and a request path is
matched against it here, once, for every list.
"""

from __future__ import annotations

from collections.abc import Iterable


def matches(path: str, doors: Iterable[str]) -> bool:
    """Whether `path` is one of `doors`, a template segment (`{video_id}`)
    standing for exactly one non-empty segment."""
    if path in doors:
        return True
    parts = path.split("/")
    for door in doors:
        if "{" not in door:
            continue
        template = door.split("/")
        if len(template) == len(parts) and all(
            (t.startswith("{") and t.endswith("}") and p) or t == p
            for t, p in zip(template, parts, strict=True)
        ):
            return True
    return False
