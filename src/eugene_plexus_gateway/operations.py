"""Operations rights for people who are not the owner (row 39), on the gateway.

`specs/docs/design/rows/row-39.md`. A person the owner gave any operations
power on this machine holds an `ep-operations+jwt` addressed to it. Here
it reads the model roles and the metrics summary, and nothing else: the
request log, client usage, routing, backends, client keys, admin and the
config stay the owner's (`require_operator`). The gateway has no
operations change, so it records nothing.
"""

from __future__ import annotations

from typing import Annotated

import jwt
from fastapi import Depends, Request, status
from fastapi.security import HTTPAuthorizationCredentials

from . import tokens
from .auth_state import AuthState
from .dependencies import _bearer_scheme, _problem, require_operator

#: The table: reads any operations power may make.
OPERATIONS_READS: frozenset[tuple[str, str]] = frozenset(
    {("GET", "/v1/roles"), ("GET", "/v1/metrics")}
)


def _is_operations(token: str) -> bool:
    try:
        return bool(jwt.get_unverified_header(token).get("typ") == tokens.TYP_OPERATIONS)
    except jwt.InvalidTokenError:
        return False


def operations_read(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> tokens.Claims | None:
    """An operations token on a read the table names, or the owner's gate."""
    if creds is None or not creds.credentials or not _is_operations(creds.credentials):
        return require_operator(request, creds)
    auth: AuthState = request.app.state.auth_state
    if auth.auth_disabled:
        return None
    try:
        claims = auth.verify(creds.credentials, classes=(tokens.TYP_OPERATIONS,))
    except tokens.TokenError as exc:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED, "Invalid token", f"Bearer token rejected: {exc}"
        ) from exc
    route = (request.method.upper(), getattr(request.scope.get("route"), "path", None))
    if route not in OPERATIONS_READS:
        raise _problem(
            status.HTTP_403_FORBIDDEN,
            "Owner only",
            "Only the owner can do this. Your operations powers do not include it.",
        )
    return claims
