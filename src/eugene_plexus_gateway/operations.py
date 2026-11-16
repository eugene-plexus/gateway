"""Operations rights for people who are not the owner (row 39), on the gateway.

`specs/docs/design/rows/row-39.md`. A person the owner gave any operations
power on this machine holds an `ep-operations+jwt` addressed to it. Here
it reads the model roles and the metrics summary, and nothing else: the
request log, client usage, routing, backends, client keys, admin and the
config stay the owner's (`require_operator`). The gateway has no
operations change, so it records nothing.

Row 38: the control root's own service token reads the metrics summary and
the per-key usage, which the active root evaluates alert rules and SLOs on.
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


#: Row 38: the reads the control root makes with its own service token, once a
#: minute, to evaluate alert rules and SLOs. Its token opens nothing else here.
CONTROL_READS: frozenset[tuple[str, str]] = frozenset(
    {("GET", "/v1/metrics"), ("GET", "/v1/metrics/clients")}
)


def _typ(token: str) -> object:
    try:
        return jwt.get_unverified_header(token).get("typ")
    except jwt.InvalidTokenError:
        return None


def _is_operations(token: str) -> bool:
    return _typ(token) == tokens.TYP_OPERATIONS


def _route(request: Request) -> tuple[str, str | None]:
    return (request.method.upper(), getattr(request.scope.get("route"), "path", None))


def _control_read(request: Request, token: str) -> tokens.Claims | None:
    """The control root's own service token on a read `CONTROL_READS` names,
    or None for the owner's gate to answer. Only the root mints a token with
    `iss: control`, and this one must be addressed to this machine."""
    if _typ(token) != tokens.TYP_SERVICE or _route(request) not in CONTROL_READS:
        return None
    auth: AuthState = request.app.state.auth_state
    if auth.auth_disabled:
        return None
    try:
        claims = auth.verify(token, classes=(tokens.TYP_SERVICE,))
    except tokens.TokenError:
        return None
    if claims.iss == tokens.ISSUER_CONTROL and claims.sub == tokens.SUB_CONTROL:
        return claims
    return None


def operations_read(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> tokens.Claims | None:
    """An operations token on a read the table names, the control root on a
    read `CONTROL_READS` names (row 38), or the owner's gate."""
    if creds is not None and creds.credentials:
        root = _control_read(request, creds.credentials)
        if root is not None:
            return root
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
    if _route(request) not in OPERATIONS_READS:
        raise _problem(
            status.HTTP_403_FORBIDDEN,
            "Owner only",
            "Only the owner can do this. Your operations powers do not include it.",
        )
    return claims
