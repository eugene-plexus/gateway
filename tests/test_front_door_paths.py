"""Every route that accepts a client key is under client admission, the body limit and
the browser's cross-origin rules, by construction rather than by a list someone remembers.

Twice a door shipped accepting client keys with no row in
`CLIENT_ADMISSION_PATHS`: `/v1/systemone` (B2) and `/v1/audio/speech`
(P3a). Such a door checks the key and then applies none of what the key
says -- no allowed models, no local-only, no rate or concurrency limit --
because the middleware never set the request's client context. Each time
the per-door test was written after the fact. This one reads the routes.

Two walks find the doors. One reads the routes that declare `require_authorized`.
The other reads the doors that check a credential in their own bodies (the Anthropic
and Responses doors, whose refusals are not FastAPI's envelope), which the first
cannot see: the count doors (`/v1/messages/count_tokens`, `/v1/responses/input_tokens`)
went without a body limit and without the browser's treatment for that reason, and a key
holder could post unbounded JSON to both.
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


def _declared_routes(settings: Settings) -> list[APIRoute]:
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


def _in_body_routes(settings: Settings) -> list[APIRoute]:
    """The doors that check a credential in their own handler, found from what each
    handler calls."""
    app = create_app(settings=settings)
    found = [
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
    } <= {r.path for r in found}
    return found


def _client_key_routes(settings: Settings) -> list[APIRoute]:
    """Every route a client key can reach: those that declare the dependency, and those
    that check the credential in their own body."""
    return [*_declared_routes(settings), *_in_body_routes(settings)]


def test_every_route_taking_a_client_key_is_under_client_admission(settings: Settings) -> None:
    missing = sorted(
        r.path for r in _client_key_routes(settings) if r.path not in CLIENT_ADMISSION_PATHS
    )
    assert not missing, f"accepts a client key with no client admission: {missing}"


def test_every_door_that_authorizes_in_its_own_body_is_under_client_admission(
    settings: Settings,
) -> None:
    """The walk over `require_authorized` sees a route that declares it. The Anthropic and
    Responses doors check the credential in their own bodies, so it cannot see them, and
    `/v1/responses/input_tokens` was added beside them with no row: a key that may not
    use a model was counted on it, and a refusal admission raised was an unhandled
    exception. Read from what each route's handler calls."""
    missing = sorted(
        r.path for r in _in_body_routes(settings) if r.path not in CLIENT_ADMISSION_PATHS
    )
    assert not missing, f"checks a client key in its own body, with no client admission: {missing}"


def _bounded(settings: Settings) -> set[str]:
    app = create_app(settings=settings)
    limit = next(m for m in app.user_middleware if m.cls is InferenceBodyLimit)
    # A door with a limit of its own (P3b's uploads) is bounded too.
    return set(limit.kwargs["paths"]) | set(limit.kwargs.get("limits") or {})


def test_every_body_taking_route_with_a_client_key_is_bounded(settings: Settings) -> None:
    bounded = _bounded(settings)
    missing = sorted(
        r.path
        for r in _client_key_routes(settings)
        if "POST" in r.methods and r.path not in bounded
    )
    assert not missing, f"reads a client's body with no size limit: {missing}"


def test_every_door_that_authorizes_in_its_own_body_is_bounded(settings: Settings) -> None:
    """The same walk the admission rows use, asserted of the body limit for each door it
    finds: a count door is a body-taking door, and reads its JSON whole like any other."""
    bounded = _bounded(settings)
    doors = _in_body_routes(settings)
    assert all("POST" in r.methods for r in doors)
    missing = sorted(r.path for r in doors if r.path not in bounded)
    assert not missing, f"checks a client key in its own body, with no size limit: {missing}"


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


def test_every_door_that_authorizes_in_its_own_body_answers_a_browser(settings: Settings) -> None:
    """A door's count has its parent's treatment: Anthropic's own browser path announces
    itself on every Claude Code request, and `count_tokens` is called by the same client."""
    missing = sorted(r.path for r in _in_body_routes(settings) if r.path not in FRONT_DOOR_PATHS)
    assert not missing, f"checks a client key in its own body, but a browser is refused: {missing}"
