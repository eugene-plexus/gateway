"""v0.2 security primitives — verify-only.

The gateway never issues tokens. The agent (the install's
trust root) generates the per-restart HMAC signing key and distributes
it to spawned children, along with a long-lived service token, via env
vars (`EUGENE_PLEXUS_GATEWAY_AUTH_SIGNING_KEY`,
`EUGENE_PLEXUS_GATEWAY_SERVICE_TOKEN`). This module exposes just the
decode side so route dependencies can validate inbound bearer tokens.

Mirror of the corresponding agent primitives at
`eugene_plexus_agent.security`. Keeping the same constant names
(`AUDIENCE_OPERATOR`, `SERVICE_AUDIENCE_PREFIX`) makes the two-sided
contract obvious when reading either side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jwt

_JWT_ALG = "HS256"

# Audience claim values — kept in sync with the agent.
AUDIENCE_OPERATOR = "operator"
SERVICE_AUDIENCE_PREFIX = "service:"


@dataclass(frozen=True)
class TokenPayload:
    """Decoded JWT claims. `iat` / `exp` are unix seconds."""

    sub: str
    aud: str
    iat: int
    exp: int


def decode_token(
    *,
    token: str,
    signing_key: bytes,
    accept_operator: bool = True,
    accept_any_service: bool = True,
) -> TokenPayload:
    """Verify a bearer token's signature + expiry and return its claims.

    `accept_operator` / `accept_any_service` together decide which
    audiences are acceptable. Common patterns:

      * `accept_operator=True, accept_any_service=True`  — chat-style
        endpoints reachable from both the UI (operator token) and
        peer components (service tokens).
      * `accept_operator=True, accept_any_service=False` — operator-only
        endpoints (config edits, admin/restart).

    Raises:
      jwt.InvalidTokenError — signature mismatch, malformed, expired,
        or audience not in the accept-set. All auth failures collapse
        into this base class so the dependency layer can `except` once.
    """
    if not (accept_operator or accept_any_service):
        raise ValueError("must accept at least one audience class")

    # Decode without strict audience match — pyjwt's `audience` kwarg
    # accepts a list but we need a *prefix* match on `service:*`, which
    # it can't do. Verify signature/expiry here, then audience manually.
    options: Any = {
        "require": ["sub", "aud", "iat", "exp"],
        "verify_aud": False,
    }
    claims = jwt.decode(token, key=signing_key, algorithms=[_JWT_ALG], options=options)

    aud = str(claims["aud"])
    is_operator = accept_operator and aud == AUDIENCE_OPERATOR
    is_service = accept_any_service and aud.startswith(SERVICE_AUDIENCE_PREFIX)
    if not (is_operator or is_service):
        raise jwt.InvalidAudienceError(f"audience {aud!r} not accepted")

    return TokenPayload(
        sub=str(claims["sub"]),
        aud=aud,
        iat=int(claims["iat"]),
        exp=int(claims["exp"]),
    )
