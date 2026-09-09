"""Auth state for the gateway's verify-only role.

Built once at startup from the env vars the agent threads in
when it spawns the gateway:

  * `EUGENE_PLEXUS_GATEWAY_AUTH_SIGNING_KEY` — base64 of the 32-byte HMAC
    key used to validate inbound bearer tokens.
  * `EUGENE_PLEXUS_GATEWAY_SERVICE_TOKEN` — long-lived JWT (`aud:
    service:gateway`) presented as `Authorization: Bearer ...` on
    every outbound call to a peer component.
  * `EUGENE_PLEXUS_GATEWAY_MASTER_KEY` — base64 of the 32-byte secretbox
    key. Populated only after the operator has logged in at the
    agent; absent during the "configured-but-locked" window.
    Reserved for Phase 6 at-rest decryption — not consumed in Phase 3.

If `AUTH_SIGNING_KEY` is unset, the gateway runs in
`auth_disabled=True` mode: route dependencies short-circuit and let
everything through, and outbound clients send no Authorization header.
That's the dev/standalone-test path; production via the agent
always supplies the env var.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthState:
    """Process-wide auth posture. Immutable for the gateway's
    lifetime — rotating the signing key requires a restart, which is
    by design (per-restart key rotation is the v0.2 revocation story)."""

    signing_key: bytes | None
    """32-byte HMAC key, or None when auth is disabled."""

    service_token: str | None
    """Outbound bearer token, or None when auth is disabled."""

    master_key: bytes | None
    """At-rest secretbox key. Only set when the operator has logged in
    at the agent. Phase 6 uses this; Phase 3 leaves it untouched."""

    @property
    def auth_disabled(self) -> bool:
        return self.signing_key is None


def _decode_b64_key(value: str | None, *, expected_len: int, label: str) -> bytes | None:
    """Decode and length-check a base64-encoded key from env. Returns
    None if `value` is None / empty. Raises ValueError on malformed or
    wrong-length input — that's a configuration bug we want loud."""
    if not value:
        return None
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as e:
        raise ValueError(f"{label}: not valid base64 ({e})") from e
    if len(raw) != expected_len:
        raise ValueError(
            f"{label}: expected {expected_len} bytes after base64-decode, got {len(raw)}"
        )
    return raw


def load_auth_state(
    *,
    signing_key_b64: str | None,
    service_token: str | None,
    master_key_b64: str | None,
) -> AuthState:
    """Build an `AuthState` from the three env-var inputs.

    Tolerates the disabled case (no signing key) by returning an
    auth-disabled state with a one-shot warning logged. Any other
    inconsistency (signing-key set but service-token absent, malformed
    base64) raises so the agent operator sees the failure rather
    than mysterious 401s downstream.
    """
    signing_key = _decode_b64_key(signing_key_b64, expected_len=32, label="AUTH_SIGNING_KEY")
    master_key = _decode_b64_key(master_key_b64, expected_len=32, label="MASTER_KEY")

    if signing_key is None:
        if service_token or master_key:
            raise ValueError(
                "auth env vars inconsistent: SERVICE_TOKEN or MASTER_KEY is set but "
                "AUTH_SIGNING_KEY is not — refusing to start in a partially-auth state"
            )
        log.warning(
            "EUGENE_PLEXUS_GATEWAY_AUTH_SIGNING_KEY not set — running unauthenticated "
            "(dev/standalone mode). Production spawns via agent always supply this."
        )
        return AuthState(signing_key=None, service_token=None, master_key=None)

    if not service_token:
        raise ValueError(
            "AUTH_SIGNING_KEY is set but SERVICE_TOKEN is missing — children spawned "
            "by the agent must receive both. Check the supervisor wiring."
        )

    return AuthState(
        signing_key=signing_key,
        service_token=service_token,
        master_key=master_key,
    )
