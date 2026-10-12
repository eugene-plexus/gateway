"""Every route that accepts a client key is under client admission and the
body limit, by construction rather than by a list someone remembers.

Twice a door shipped accepting client keys with no row in
`CLIENT_ADMISSION_PATHS`: `/v1/systemone` (B2) and `/v1/audio/speech`
(P3a). Such a door checks the key and then applies none of what the key
says -- no allowed models, no local-only, no rate or concurrency limit --
because the middleware never set the request's client context. Each time
the per-door test was written after the fact. This one reads the routes.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator

from fastapi.routing import APIRoute

from eugene_plexus_gateway.admission import CLIENT_ADMISSION_PATHS
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.body_limit import InferenceBodyLimit
from eugene_plexus_gateway.cors import FRONT_DOOR_PATHS
from eugene_plexus_gateway.dependencies import require_authorized
from eugene_plexus_gateway.settings import Settings


def _routes(routes: list) -> Iterator[APIRoute]:
    for route in routes:
        if isinstance(route, APIRoute):
            yield route
        elif hasattr(route, "original_router"):
            yield from _routes(route.original_router.routes)


def _handler_source(route: APIRoute) -> str:
    """The text of a route's handler, or nothing when it has none to read (a callable
    object, a builtin): such a handler does not call the credential checks by name."""
    try:
        return inspect.getsource(route.endpoint)
    except (OSError, TypeError):
        return ""


def _client_key_routes(settings: Settings) -> list[APIRoute]:
    app = create_app(settings=settings)
    found = [
        r
        for r in _routes(app.routes)
        if require_authorized in [d.call for d in r.dependant.dependencies]
    ]
    # The walk itself must find something, or every assertion below holds
    # of nothing.
    assert {"/v1/chat/completions", "/v1/audio/speech"} <= {r.path for r in found}
    return found


def test_every_route_taking_a_client_key_is_under_client_admission(settings: Settings) -> None:
    missing = sorted(
        r.path for r in _client_key_routes(settings) if r.path not in CLIENT_ADMISSION_PATHS
    )
    assert not missing, f"accepts a client key with no client admission: {missing}"


def test_every_door_that_authorizes_in_its_own_body_is_under_client_admission(
    settings: Settings,
) -> None:
    """The walk above sees a route that declares `require_authorized`. The Anthropic and
    Responses doors check the credential in their own bodies (their refusals are not
    FastAPI's envelope), so it cannot see them, and `/v1/responses/input_tokens` was
    added beside them with no row: a key that may not use a model was counted on it,
    and a refusal admission raised was an unhandled exception. Read from what each
    route's handler calls."""
    app = create_app(settings=settings)
    in_body = [
        r
        for r in _routes(app.routes)
        if any(
            call in _handler_source(r) for call in ("anthropic.authorize(", "responses.authorize(")
        )
    ]
    # The walk itself must find the doors it exists for, or every assertion below
    # holds of nothing.
    assert {
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/responses",
        "/v1/responses/input_tokens",
    } <= {r.path for r in in_body}
    missing = sorted(r.path for r in in_body if r.path not in CLIENT_ADMISSION_PATHS)
    assert not missing, f"checks a client key in its own body, with no client admission: {missing}"


def test_every_body_taking_route_with_a_client_key_is_bounded(settings: Settings) -> None:
    app = create_app(settings=settings)
    limit = next(m for m in app.user_middleware if m.cls is InferenceBodyLimit)
    # A door with a limit of its own (P3b's uploads) is bounded too.
    bounded = set(limit.kwargs["paths"]) | set(limit.kwargs.get("limits") or {})
    missing = sorted(
        r.path
        for r in _client_key_routes(settings)
        if "POST" in r.methods and r.path not in bounded
    )
    assert not missing, f"reads a client's body with no size limit: {missing}"


#: Takes client keys and is not answered cross-origin. Nobody has decided
#: whether a browser should call the decision door (B2 left it out, without
#: a word either way); recorded here rather than widened by this test.
NOT_CROSS_ORIGIN = {"/v1/systemone"}


def test_every_route_taking_a_client_key_answers_a_browser(settings: Settings) -> None:
    missing = sorted(
        r.path
        for r in _client_key_routes(settings)
        if r.path not in FRONT_DOOR_PATHS | NOT_CROSS_ORIGIN
    )
    assert not missing, f"a client key works here but a browser's preflight is refused: {missing}"
