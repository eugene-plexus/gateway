"""FastAPI dependencies for v0.2 bearer auth.

Two dependencies, both pass-through when `AuthState.auth_disabled` is
true (the dev/standalone path):

  * `require_authorized` — accepts an operator-audience OR any
    `service:*`-audience token OR a **client key** (`aud: client`).
    Used for routes reachable from both the UI and peer components (the
    OpenAI-compatible front door), and since S4 the only place in the
    install that accepts a client key at all. A revoked client key is
    refused here, from the list `ClientKeyGuard` keeps.

  * `require_operator` — accepts operator-audience only. Used for
    operator-only routes (config edits, admin/restart, drivers
    list/probe).

Both raise 401 with a Problem JSON body on missing / malformed /
expired / wrong-audience tokens, mirroring the agent's shape so the
UI can render one error path across components.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import security
from ._generated.models import Problem
from .admission import current
from .auth_state import AuthState
from .client_keys import ClientKeyGuard

_bearer_scheme = HTTPBearer(auto_error=False)


def _problem(status_code: int, title: str, detail: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/gateway#{title.replace(' ', '-').lower()}",
            title=title,
            status=status_code,
            detail=detail,
            component="gateway",
        ).model_dump(exclude_none=True),
    )


def _validate(
    request: Request,
    creds: HTTPAuthorizationCredentials | None,
    *,
    accept_operator: bool,
    accept_any_service: bool,
    accept_client: bool = False,
) -> security.TokenPayload | None:
    auth: AuthState = request.app.state.auth_state
    if auth.auth_disabled:
        return None
    if creds is None or not creds.credentials:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Missing token",
            "Provide a bearer token via the Authorization: Bearer header.",
        )
    assert auth.signing_key is not None  # narrowed by auth_disabled
    try:
        return security.decode_token(
            token=creds.credentials,
            signing_key=auth.signing_key,
            accept_operator=accept_operator,
            accept_any_service=accept_any_service,
            accept_client=accept_client,
        )
    except Exception as e:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid token",
            f"Bearer token rejected: {e}",
        ) from e


async def require_authorized(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> security.TokenPayload | None:
    """Operator, any service token, or a client key — the front door.

    Async since S4, because a client key has one more question to
    answer: has the operator turned it off? Nothing about the other two
    audiences awaits anything, and an install where nobody has minted a
    client key never touches the guard.
    """
    payload = _validate(
        request, creds, accept_operator=True, accept_any_service=True, accept_client=True
    )
    if payload is None or payload.aud != security.AUDIENCE_CLIENT:
        return payload
    context = current.get()
    if context is not None:
        context.key_id = payload.jti
        context.key_name = payload.sub
    guard: ClientKeyGuard | None = getattr(request.app.state, "client_key_guard", None)
    decision = await guard.decision(payload.jti) if guard is not None else "unavailable"
    if decision == "unavailable":
        raise _problem(
            503,
            "Client-key policy unavailable",
            "Client access is paused until a fresh key policy is available. "
            "Check the local agent and control root; operator sign-in remains available.",
        )
    if decision == "unregistered":
        raise _problem(
            401,
            "Key not registered",
            "This key is not registered with the current authority. Open Use it from "
            "your apps on the node that originally made it to check migration, or replace it.",
        )
    if decision == "revoked":
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Key revoked",
            "This client key was turned off. Make a new one under "
            "Home -> Use it from your apps, and paste it into the app that is failing.",
        )
    return payload


def require_operator(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> security.TokenPayload | None:
    """Operator-audience tokens only — for config / admin endpoints.

    `accept_client` is left at its default `False`: a key handed to
    Open WebUI must not be able to edit this gateway's config or read
    what every other caller asked it."""
    return _validate(request, creds, accept_operator=True, accept_any_service=False)
