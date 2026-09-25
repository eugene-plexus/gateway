"""FastAPI dependencies for bearer auth, against this machine's trust bundle.

Two dependencies, both pass-through when `AuthState.auth_disabled` is
true (the dev/standalone path):

  * `require_authorized` -- the front door. An operator session
    addressed to this machine, a service token this machine's own agent
    minted for itself or one of its children, or a **client key**
    (`ep-client+jwt`, addressed to `gateway`, signed by the install's
    authority). The only place in the install that accepts a client key
    at all; a turned-off one is refused here, from the policy
    `ClientKeyGuard` keeps.

  * `require_operator` -- an operator session only. Config edits,
    admin/restart, drivers list/probe, metrics.

**No `service:*` wildcard, and no service token from another machine**
(per-node token keys, 2026-09-25). Until then any component's token
from any machine opened the front door, because a service token named a
kind and every machine shared one key.

Both raise 401 with a Problem JSON body on a missing, malformed,
expired or misaddressed token, mirroring the agent's shape so the UI can
render one error path across components.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import tokens
from ._generated.models import Problem
from .admission import current
from .auth_state import AuthState
from .client_keys import ClientKeyGuard

_bearer_scheme = HTTPBearer(auto_error=False)

FRONT_DOOR = (tokens.TYP_SESSION, tokens.TYP_SERVICE, tokens.TYP_CLIENT)


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


def front_door_claims(auth: AuthState, token: str) -> tokens.Claims:
    """A front-door credential's claims, or `TokenError` saying why not.

    Shared with the Anthropic and Responses doors, which read their
    credential from a different header and answer in their own shape.
    """
    claims = auth.verify(token, classes=FRONT_DOOR)
    if claims.is_service and not claims.is_local_service(str(auth.recipient)):
        raise tokens.TokenError(
            f"a service token from {claims.iss!r} is not accepted at this front door; "
            "only this machine's own components and the operator may use it"
        )
    return claims


def _validate(
    request: Request,
    creds: HTTPAuthorizationCredentials | None,
    *,
    front_door: bool,
) -> tokens.Claims | None:
    auth: AuthState = request.app.state.auth_state
    if auth.auth_disabled:
        return None
    if creds is None or not creds.credentials:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Missing token",
            "Provide a bearer token via the Authorization: Bearer header.",
        )
    try:
        if front_door:
            return front_door_claims(auth, creds.credentials)
        return auth.verify(creds.credentials, classes=(tokens.TYP_SESSION,))
    except tokens.TokenError as e:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid token",
            f"Bearer token rejected: {e}",
        ) from e


async def require_authorized(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims | None:
    """A session, this machine's services, or a client key -- the front door.

    Async since S4, because a client key has one more question to
    answer: has the operator turned it off? Nothing about the other two
    classes awaits anything, and an install where nobody has minted a
    client key never touches the guard.
    """
    claims = _validate(request, creds, front_door=True)
    if claims is None or not claims.is_client:
        return claims
    context = current.get()
    if context is not None:
        context.key_id = claims.jti
        context.key_name = claims.sub
    guard: ClientKeyGuard | None = getattr(request.app.state, "client_key_guard", None)
    decision = await guard.decision(claims.jti) if guard is not None else "unavailable"
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
            "This key is not registered with the current authority. Make a new one under "
            "Home -> Use it from your apps.",
        )
    if decision == "revoked":
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Key revoked",
            "This client key was turned off. Make a new one under "
            "Home -> Use it from your apps, and paste it into the app that is failing.",
        )
    return claims


def require_operator(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> tokens.Claims | None:
    """An operator session only -- for config / admin endpoints.

    A key handed to Open WebUI must not be able to edit this gateway's
    config or read what every other caller asked it, and neither may
    any component's token."""
    return _validate(request, creds, front_door=False)
