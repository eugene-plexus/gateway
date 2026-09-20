"""Token verification with public Ed25519 PEM.
Legacy 32-byte HS256 verification is supported only while the install
retains its old key. No algorithm is selected from the token header.
This component receives no private signing key in Ed25519 mode.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

log = logging.getLogger(__name__)


def verification_key(key: bytes) -> bytes:
    """Validate public Ed25519 PEM, or an explicitly legacy 32-byte HMAC key."""
    if len(key) == 32:
        return key
    parsed = serialization.load_pem_public_key(key)
    if not isinstance(parsed, Ed25519PublicKey):
        raise ValueError("token verification requires an Ed25519 public key")
    return key


def verification_algorithm(key: bytes) -> str:
    """Select from trusted key material, never an untrusted JWT header."""
    return "HS256" if len(key) == 32 else "EdDSA"


# Audience claim values — kept in sync with the agent.
AUDIENCE_OPERATOR = "operator"
SERVICE_AUDIENCE_PREFIX = "service:"

AUDIENCE_CLIENT = "client"
"""A long-lived key an app outside the install holds (S4, 2026-09-15).

Accepted here on the three OpenAI-compatible paths and refused
everywhere else in the install -- not by a rule anyone maintains, but
because `client` is neither `operator` nor a `service:` audience, and
every other check in every other component tests for one of those two.
This module is the single place that opts in, and `require_authorized`
is the single caller that passes `accept_client=True`.
"""


@dataclass(frozen=True)
class TokenPayload:
    """Decoded JWT claims. `iat` / `exp` are unix seconds."""

    sub: str
    aud: str
    iat: int
    exp: int
    jti: str | None = None
    """The client key's id, on a client key; None on every other token.

    Not in the `require` list, because operator sessions and service
    tokens have never carried one and demanding it would refuse every
    token minted before 2026-09-15 -- including the one the operator is
    holding while the gateway is upgraded under them.
    """


CLOCK_SKEW_LEEWAY_SECONDS = 300
"""How far apart two hosts' clocks may drift before a token is refused.

Five minutes: Kerberos's `MaxClockSkew`, and the window Entra and most
OAuth validators apply to `iat`, `nbf` and `exp`. **It was zero until
2026-09-15.** On the live two-machine install the control root's clock
ran half a second ahead of a worker whose Windows Time service had
stopped, and every token the root minted in the first half of each
second was refused by that worker as "not yet valid (iat)" a few
milliseconds later. Long-lived tokens (the gateway's, the operator's
session) passed, every health check said ok, and the root listed the
node `down` with no reason -- so it read as a key or enrollment fault
and was neither. A skew large enough to matter for security is a
broken clock; a broken clock is *reported* (`_note_clock_skew`), not
enforced by refusing traffic between two healthy hosts.
"""

_SKEW_WARN_AFTER_SECONDS = 2.0
_SKEW_WARN_INTERVAL_SECONDS = 60.0
_last_skew_warning = 0.0


def _note_clock_skew(iat: int, *, now: float | None = None) -> None:
    """Warn, at most once a minute, when a token was issued in this host's future.

    Accepted within `CLOCK_SKEW_LEEWAY_SECONDS`, so nothing breaks. Logged
    so a wrong clock on either host is visible long before the skew grows
    past the leeway and starts refusing traffic.
    """
    global _last_skew_warning
    current = time.time() if now is None else now
    ahead = iat - current
    if ahead <= _SKEW_WARN_AFTER_SECONDS:
        return
    if current - _last_skew_warning < _SKEW_WARN_INTERVAL_SECONDS:
        return
    _last_skew_warning = current
    log.warning(
        "accepted a token issued %.1f s in this host's future: the issuer's clock or "
        "this host's is wrong (tolerated up to %d s, then tokens are refused)",
        ahead,
        CLOCK_SKEW_LEEWAY_SECONDS,
    )


def decode_token(
    *,
    token: str,
    signing_key: bytes,
    accept_operator: bool = True,
    accept_any_service: bool = True,
    accept_client: bool = False,
) -> TokenPayload:
    """Verify a bearer token's signature + expiry and return its claims.

    `accept_operator` / `accept_any_service` together decide which
    audiences are acceptable. Common patterns:

      * `accept_operator=True, accept_any_service=True`  — chat-style
        endpoints reachable from both the UI (operator token) and
        peer components (service tokens).
      * `accept_operator=True, accept_any_service=False` — operator-only
        endpoints (config edits, admin/restart).

    `accept_client` is off by default and turned on in exactly one
    place: the three OpenAI-compatible paths. A client key must never
    reach an operator surface, and the default being `False` is what
    makes a new endpoint safe by omission rather than by vigilance.

    Raises:
      jwt.InvalidTokenError — signature mismatch, malformed, expired,
        or audience not in the accept-set. All auth failures collapse
        into this base class so the dependency layer can `except` once.
    """
    if not (accept_operator or accept_any_service or accept_client):
        raise ValueError("must accept at least one audience class")

    # Decode without strict audience match — pyjwt's `audience` kwarg
    # accepts a list but we need a *prefix* match on `service:*`, which
    # it can't do. Verify signature/expiry here, then audience manually.
    options: Any = {
        "require": ["sub", "aud", "iat", "exp"],
        "verify_aud": False,
    }
    claims = jwt.decode(
        token,
        key=verification_key(signing_key),
        algorithms=[verification_algorithm(signing_key)],
        options=options,
        leeway=CLOCK_SKEW_LEEWAY_SECONDS,
    )
    _note_clock_skew(int(claims["iat"]))

    aud = str(claims["aud"])
    is_operator = accept_operator and aud == AUDIENCE_OPERATOR
    is_service = accept_any_service and aud.startswith(SERVICE_AUDIENCE_PREFIX)
    is_client = accept_client and aud == AUDIENCE_CLIENT
    if not (is_operator or is_service or is_client):
        raise jwt.InvalidAudienceError(f"audience {aud!r} not accepted")

    raw_jti = claims.get("jti")
    return TokenPayload(
        sub=str(claims["sub"]),
        aud=aud,
        iat=int(claims["iat"]),
        exp=int(claims["exp"]),
        jti=str(raw_jti) if raw_jti is not None else None,
    )
