"""Verifier bootstrap: the trust bundle this gateway checks tokens against.

Per-node token keys (2026-09-25; `specs/docs/design/per-node-token-keys.md`).
The supervising agent hands this process four things and no key:

* `TRUST_BUNDLE_FILE` -- the bundle the agent keeps and rewrites when
  the control root publishes a new one; reloaded when it changes;
* `TRUST_AUTHORITY` -- the key that bundle must be signed by, so a file
  anyone else wrote is refused;
* `AUTH_RECIPIENT` -- which machine this is (`node:<name>`), the
  audience a token must name to be accepted here;
* `SERVICE_TOKEN` -- this gateway's own token, addressed to this
  machine alone.

`MASTER_KEY` is the separate at-rest encryption credential. With none of
the first four, the gateway runs unauthenticated (dev/standalone), as it
always has; with some but not all, it refuses to start rather than run
half-authenticated.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

from . import tokens

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthState:
    """Process-wide auth posture."""

    bundle: tokens.BundleFile | None
    """The trust bundle; None when auth is disabled."""

    recipient: str | None
    """This machine, as a token's audience names it."""

    service_token: str | None
    """This gateway's own token, good on this machine only."""

    master_key: bytes | None
    """At-rest secretbox key, when the operator has unlocked the agent."""

    authority: str | None = None
    """The key the bundle is signed by: the control root's, or a standalone node's."""

    @property
    def auth_disabled(self) -> bool:
        return self.bundle is None

    def verify(self, token: str, *, classes: tuple[str, ...]) -> tokens.Claims:
        """Every check `tokens.verify` makes, against the bundle as it is now."""
        if self.bundle is None or self.recipient is None:
            raise tokens.TokenError("this gateway verifies nothing: it runs without auth")
        bundle = self.bundle.get()
        if bundle is None:
            raise tokens.TokenError(
                "this gateway holds no trust bundle it can read yet; its agent keeps one "
                "beside node.yaml"
            )
        return tokens.verify(token, bundle=bundle, recipient=self.recipient, classes=classes)


def _decode_b64_key(value: str | None, *, expected_len: int, label: str) -> bytes | None:
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
    trust_bundle_file: str | None,
    trust_authority: str | None,
    auth_recipient: str | None,
    service_token: str | None,
    master_key_b64: str | None,
) -> AuthState:
    """Build an `AuthState` from the environment the agent supplies."""
    master_key = _decode_b64_key(master_key_b64, expected_len=32, label="MASTER_KEY")
    given = {
        "TRUST_BUNDLE_FILE": trust_bundle_file,
        "TRUST_AUTHORITY": trust_authority,
        "AUTH_RECIPIENT": auth_recipient,
        "SERVICE_TOKEN": service_token,
    }
    present = [name for name, value in given.items() if value]
    if not present:
        if master_key is not None:
            raise ValueError(
                "MASTER_KEY is set but no trust bundle is -- refusing a partially-auth state"
            )
        log.warning(
            "EUGENE_PLEXUS_GATEWAY_TRUST_BUNDLE_FILE not set -- running unauthenticated "
            "(dev/standalone mode). Production spawns via the agent always supply it."
        )
        return AuthState(bundle=None, recipient=None, service_token=None, master_key=None)
    missing = [name for name, value in given.items() if not value]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} missing beside {', '.join(present)}: the agent hands a "
            "child all four or none. Check the supervisor wiring."
        )
    assert trust_bundle_file and trust_authority and auth_recipient and service_token
    try:
        tokens.load_public(trust_authority)
    except ValueError as exc:
        raise ValueError(f"TRUST_AUTHORITY is not an Ed25519 public key: {exc}") from exc
    return AuthState(
        bundle=tokens.BundleFile(trust_bundle_file, authority=trust_authority),
        recipient=auth_recipient,
        service_token=service_token,
        master_key=master_key,
        authority=trust_authority,
    )
