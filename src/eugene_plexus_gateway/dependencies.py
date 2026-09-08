"""FastAPI dependencies for v0.2 bearer auth.

Two dependencies, both pass-through when `AuthState.auth_disabled` is
true (the dev/standalone path):

  * `require_authorized` — accepts an operator-audience OR any
    `service:*`-audience token. Used for routes reachable from both the
    UI and peer components (the OpenAI-compatible front door).

  * `require_operator` — accepts operator-audience only. Used for
    operator-only routes (config edits, admin/restart, drivers
    list/probe).

Both raise 401 with a Problem JSON body on missing / malformed /
expired / wrong-audience tokens, mirroring the watchdog's shape so the
UI can render one error path across components.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import security
from ._generated.models import Problem
from .auth_state import AuthState

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
        )
    except Exception as e:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid token",
            f"Bearer token rejected: {e}",
        ) from e


def require_authorized(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> security.TokenPayload | None:
    """Operator OR any service-audience token accepted."""
    return _validate(request, creds, accept_operator=True, accept_any_service=True)


def require_operator(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> security.TokenPayload | None:
    """Operator-audience tokens only — for config / admin endpoints."""
    return _validate(request, creds, accept_operator=True, accept_any_service=False)
