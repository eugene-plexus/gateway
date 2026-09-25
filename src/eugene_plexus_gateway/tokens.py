"""Tokens and the trust bundle: per-node token keys (2026-09-25).

One module, copied into every component that mints or verifies a token,
because components share schemas and not code. The copies must stay
identical; `tests/test_tokens.py` travels with each one. Design:
`specs/docs/design/per-node-token-keys.md`; wire shapes:
`specs/openapi/components/common.yaml` (`TrustBundle`).

**Every issuer has its own Ed25519 key and no private key crosses the
wire.** A token names its key by `kid`, the RFC 7638 thumbprint. A
verifier trusts exactly the keys the install's trust bundle lists, and
the bundle itself is a JWS signed by the one authority key the verifier
pinned. What a key may issue is the bundle's `grants`, never the
token's own say-so.

The checks, in the order `verify` makes them, each one a line of RFC
8725 (JWT BCP):

1. `typ` is one of the classes the caller accepts (§3.11, explicit
   typing): a session cannot be passed off as a service token, or the
   reverse.
2. `kid` is in the bundle. The algorithm comes from that key and is
   always EdDSA; the header's `alg` chooses nothing (§3.1).
3. Signature, `exp` and `iat`, with 300 s of clock leeway.
4. `iss` is the key's issuer (§3.12), and `aud` names the recipient
   asking (§3.9).
5. The key's grants cover what the token claims to be.
6. The token lives no longer than its class allows.
7. A session is not signed out, and every machine it names, as a
   console or as the one that exchanged it, is still in the bundle.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

log = logging.getLogger(__name__)

TYP_SESSION = "ep-session+jwt"
TYP_SERVICE = "ep-service+jwt"
TYP_CLIENT = "ep-client+jwt"
TYP_BUNDLE = "ep-trust-bundle+jwt"

GRANT_AUTHORITY = "authority"
GRANT_NODE = "node"
GRANT_GATEWAY = "gateway"

ISSUER_CONTROL = "control"
RECIPIENT_CONTROL = "control"
RECIPIENT_GATEWAY = "gateway"
STANDALONE_NODE = "local"

SUB_OPERATOR = "operator"
SUB_AGENT = "agent"
SUB_GATEWAY = "gateway"
SUB_CONTROL = "control"

LEEWAY_SECONDS = 300
"""Clock skew tolerated between two hosts (Kerberos's `MaxClockSkew`).
It was zero until 2026-09-15, when half a second of skew took a worker
out of the install while every health check said ok."""

SKEW_WARN_AFTER_SECONDS = 2.0
SKEW_WARN_INTERVAL_SECONDS = 60.0
_last_skew_warning = 0.0


def note_clock_skew(iat: int, *, now: float | None = None) -> bool:
    """Warn, at most once a minute, when a token was issued in this host's future.

    Accepted within `LEEWAY_SECONDS`, so nothing breaks. Logged so a
    wrong clock on either host is visible long before the skew grows
    past the leeway and starts refusing traffic. Returns whether it
    warned, for the tests.
    """
    global _last_skew_warning
    current = time.time() if now is None else now
    ahead = iat - current
    if ahead <= SKEW_WARN_AFTER_SECONDS:
        return False
    if current - _last_skew_warning < SKEW_WARN_INTERVAL_SECONDS:
        return False
    _last_skew_warning = current
    log.warning(
        "accepted a token issued %.1f s in this host's future: the issuer's clock or this "
        "host's is wrong (tolerated up to %d s, then tokens are refused)",
        ahead,
        LEEWAY_SECONDS,
    )
    return True


DAY = 24 * 3600
MAX_SESSION_SECONDS = 14 * DAY
MAX_EXCHANGED_SECONDS = 10 * 60
MAX_LOCAL_SERVICE_SECONDS = 400 * DAY
MAX_REMOTE_SERVICE_SECONDS = 3600
MAX_CLIENT_SECONDS = 400 * DAY

SESSION_TTL_SECONDS = 14 * DAY
EXCHANGED_TTL_SECONDS = 5 * 60
LOCAL_SERVICE_TTL_SECONDS = 365 * DAY
REMOTE_SERVICE_TTL_SECONDS = 15 * 60
CONTROL_SERVICE_TTL_SECONDS = 5 * 60


class TokenError(jwt.InvalidTokenError):
    """Every refusal. A subclass of PyJWT's base so one `except` catches both."""


class BundleError(ValueError):
    """A trust bundle that must not be trusted: bad signature, wrong authority, rollback."""


# --------------------------------------------------------------------------- #
# Keys
# --------------------------------------------------------------------------- #


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def public_raw(key: Ed25519PublicKey | Ed25519PrivateKey) -> bytes:
    """The raw 32 bytes of a public key, or of a private key's public half."""
    public = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    return public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def public_b64(key: Ed25519PublicKey | Ed25519PrivateKey) -> str:
    """Base64 of the raw public key, the form the bundle and the contract carry."""
    return base64.b64encode(public_raw(key)).decode("ascii")


def thumbprint(key: Ed25519PublicKey | Ed25519PrivateKey) -> str:
    """RFC 7638 JWK thumbprint: SHA-256 over the canonical OKP JWK, base64url.

    The members are exactly `crv`, `kty` and `x`, in that (lexicographic)
    order and with no whitespace, which is what makes two
    implementations agree on the bytes.
    """
    jwk = f'{{"crv":"Ed25519","kty":"OKP","x":"{_b64url(public_raw(key))}"}}'
    return _b64url(hashlib.sha256(jwk.encode("ascii")).digest())


def generate_private_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def private_to_b64(key: Ed25519PrivateKey) -> str:
    """Base64 of the raw 32-byte private key. What `node.yaml` stores."""
    raw = key.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
    )
    return base64.b64encode(raw).decode("ascii")


def load_private(value: str | bytes) -> Ed25519PrivateKey:
    """An Ed25519 private key from base64 raw bytes, or from PKCS8 PEM.

    Anything else is refused, including every 32-byte HMAC key: HS256
    was removed with the shared key and is not a format this accepts.
    """
    raw = value.encode("ascii") if isinstance(value, str) else value
    if raw.startswith(b"-----BEGIN"):
        parsed = serialization.load_pem_private_key(raw, password=None)
        if not isinstance(parsed, Ed25519PrivateKey):
            raise ValueError("a token key must be an Ed25519 private key")
        return parsed
    try:
        decoded = base64.b64decode(raw, validate=True)
    except ValueError as exc:
        raise ValueError(f"a token key must be base64 of 32 raw bytes: {exc}") from exc
    if len(decoded) != 32:
        raise ValueError(f"a token key must be 32 raw bytes, got {len(decoded)}")
    return Ed25519PrivateKey.from_private_bytes(decoded)


def load_public(value: str) -> Ed25519PublicKey:
    """An Ed25519 public key from base64 of its raw 32 bytes."""
    try:
        decoded = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise ValueError(f"a public key must be base64 of 32 raw bytes: {exc}") from exc
    if len(decoded) != 32:
        raise ValueError(f"a public key must be 32 raw bytes, got {len(decoded)}")
    return Ed25519PublicKey.from_public_bytes(decoded)


# --------------------------------------------------------------------------- #
# The trust bundle
# --------------------------------------------------------------------------- #


def node_recipient(name: str) -> str:
    return f"node:{name}"


def node_of(recipient: str) -> str | None:
    """`node:<name>` -> `<name>`; anything else -> None."""
    return recipient[5:] if recipient.startswith("node:") and len(recipient) > 5 else None


@dataclass(frozen=True)
class TrustKey:
    kid: str
    issuer: str
    public: Ed25519PublicKey
    grants: frozenset[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kid": self.kid,
            "issuer": self.issuer,
            "publicKey": public_b64(self.public),
            "grants": sorted(self.grants),
        }


@dataclass(frozen=True)
class TrustBundle:
    """A verified bundle. Built only by `parse_bundle` or `build_bundle`."""

    version: int
    epoch: int
    iat: int
    authority: str
    keys: Mapping[str, TrustKey]
    revoked_sessions: frozenset[str]
    jws: str
    payload: Mapping[str, Any] = field(repr=False)

    @property
    def members(self) -> frozenset[str]:
        """Every node name a key in this bundle is issued to."""
        names = (node_of(k.issuer) for k in self.keys.values())
        return frozenset(n for n in names if n is not None)

    def age_seconds(self, now: float | None = None) -> int:
        return max(0, int((time.time() if now is None else now) - self.iat))


def build_bundle(
    *,
    authority: Ed25519PrivateKey,
    version: int,
    epoch: int,
    keys: Collection[TrustKey],
    revoked_sessions: Collection[tuple[str, int]] = (),
    now: int | None = None,
) -> TrustBundle:
    """Sign a bundle. Expired sign-outs are pruned; nothing else is decided here."""
    issued = int(time.time()) if now is None else now
    kids = [k.kid for k in keys]
    if len(set(kids)) != len(kids):
        raise ValueError("a trust bundle cannot list one key twice")
    payload: dict[str, Any] = {
        "version": version,
        "epoch": epoch,
        "iat": issued,
        "authority": public_b64(authority),
        "keys": [k.to_dict() for k in sorted(keys, key=lambda k: k.kid)],
        "revokedSessions": [
            {"jti": jti, "exp": exp}
            for jti, exp in sorted(set(revoked_sessions))
            if exp + LEEWAY_SECONDS >= issued
        ],
    }
    jws = jwt.encode(payload, authority, algorithm="EdDSA", headers={"typ": TYP_BUNDLE})
    return parse_bundle(jws, authority=public_b64(authority))


def parse_bundle(jws: str, *, authority: str) -> TrustBundle:
    """Verify a bundle against the pinned authority key and parse it.

    The signature is checked over the JWS payload's exact bytes, so
    nothing is re-serialized first. `authority` is base64 of the raw
    public key the caller pinned; a bundle signed by anything else, or
    naming a different authority inside, is refused.
    """
    try:
        header = jwt.get_unverified_header(jws)
    except jwt.InvalidTokenError as exc:
        raise BundleError(f"not a JWS: {exc}") from exc
    if header.get("typ") != TYP_BUNDLE:
        raise BundleError(f"typ is {header.get('typ')!r}, not {TYP_BUNDLE!r}")
    try:
        pinned = load_public(authority)
        claims = jwt.decode(
            jws,
            pinned,
            algorithms=["EdDSA"],
            options={"verify_exp": False, "verify_iat": False, "verify_aud": False},
        )
    except (jwt.InvalidTokenError, ValueError) as exc:
        raise BundleError(f"signature did not verify against the pinned authority: {exc}") from exc
    if claims.get("authority") != authority:
        raise BundleError("the bundle names a different authority from the one that signed it")
    try:
        keys: dict[str, TrustKey] = {}
        for raw in claims["keys"]:
            public = load_public(str(raw["publicKey"]))
            kid = str(raw["kid"])
            if thumbprint(public) != kid:
                raise BundleError(f"key {kid!r} is not the thumbprint of its public key")
            if kid in keys:
                raise BundleError(f"key {kid!r} is listed twice")
            keys[kid] = TrustKey(
                kid=kid,
                issuer=str(raw["issuer"]),
                public=public,
                grants=frozenset(str(g) for g in raw["grants"]),
            )
        revoked = frozenset(str(r["jti"]) for r in claims["revokedSessions"])
        return TrustBundle(
            version=int(claims["version"]),
            epoch=int(claims["epoch"]),
            iat=int(claims["iat"]),
            authority=authority,
            keys=keys,
            revoked_sessions=revoked,
            jws=jws,
            payload=claims,
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, BundleError):
            raise
        raise BundleError(f"malformed bundle: {exc}") from exc


def accepts_replacement(held: TrustBundle | None, offered: TrustBundle) -> str | None:
    """None when `offered` may replace `held`; otherwise why not.

    An equal version replaces: two builds at one log index differ only
    in `iat` and in pruned, already-expired sign-outs.
    """
    if held is None:
        return None
    if offered.epoch < held.epoch:
        return f"epoch {offered.epoch} is below the {held.epoch} already held"
    if offered.version < held.version:
        return f"version {offered.version} is below the {held.version} already held"
    return None


class BundleFile:
    """A bundle on disk, reloaded when the file changes. What children read.

    The agent writes the file atomically; this keeps the newest bundle
    it has verified and refuses one that goes backwards, so a file
    restored from a backup cannot resurrect a revoked key.
    """

    def __init__(
        self, path: str | os.PathLike[str], *, authority: str, check_interval: float = 1.0
    ) -> None:
        self._path = Path(path)
        self._authority = authority
        self._interval = check_interval
        self._bundle: TrustBundle | None = None
        self._stamp: tuple[int, int, int] | None = None
        self._checked = 0.0
        self._lock = threading.Lock()
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        return self._error

    def get(self) -> TrustBundle | None:
        now = time.perf_counter()
        with self._lock:
            if self._bundle is not None and now - self._checked < self._interval:
                return self._bundle
            self._checked = now
            try:
                stat = self._path.stat()
            except OSError as exc:
                self._error = f"cannot read the trust bundle at {self._path}: {exc}"
                return self._bundle
            # The inode as well as the time: `os.replace` makes a new file,
            # and a filesystem with coarse timestamps can give two quick
            # writes of the same size the same mtime.
            stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if stamp == self._stamp:
                return self._bundle
            try:
                document = json.loads(self._path.read_text(encoding="utf-8"))
                offered = parse_bundle(str(document["jws"]), authority=self._authority)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self._error = f"the trust bundle at {self._path} was refused: {exc}"
                return self._bundle
            why = accepts_replacement(self._bundle, offered)
            if why is not None:
                self._error = f"the trust bundle at {self._path} was refused: {why}"
                return self._bundle
            self._bundle = offered
            self._stamp = stamp
            self._error = None
            return offered


def write_bundle_file(path: str | os.PathLike[str], bundle: TrustBundle) -> None:
    """Atomically replace the bundle file: temp, fsync, `os.replace`."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{secrets.token_hex(4)}.tmp")
    data = json.dumps({"jws": bundle.jws}).encode("utf-8")
    with open(temp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)


# --------------------------------------------------------------------------- #
# Minting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Signer:
    """A private token key and the issuer name its tokens carry."""

    key: Ed25519PrivateKey
    issuer: str

    @property
    def kid(self) -> str:
        return thumbprint(self.key)

    def trust_key(self, grants: Collection[str]) -> TrustKey:
        return TrustKey(
            kid=self.kid, issuer=self.issuer, public=self.key.public_key(), grants=frozenset(grants)
        )

    def mint(
        self,
        *,
        typ: str,
        sub: str,
        aud: Collection[str],
        ttl_seconds: int,
        now: int | None = None,
        jti: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> tuple[str, int]:
        """Sign one token. Returns `(token, exp)`.

        The random `jti` is what makes a sign-out mean one token: an
        Ed25519 signature is deterministic and `iat` is whole seconds,
        so two tokens minted in one second would otherwise be identical.
        """
        if not aud:
            raise ValueError("a token must name at least one recipient")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        issued = int(time.time()) if now is None else now
        claims: dict[str, Any] = {
            "iss": self.issuer,
            "sub": sub,
            "aud": list(aud),
            "iat": issued,
            "exp": issued + ttl_seconds,
            "jti": jti or secrets.token_urlsafe(12),
        }
        if extra:
            claims.update(extra)
        token = jwt.encode(
            claims, self.key, algorithm="EdDSA", headers={"typ": typ, "kid": self.kid}
        )
        return token, issued + ttl_seconds


# --------------------------------------------------------------------------- #
# Verifying
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Claims:
    """What a verified token says, after every check in `verify` passed."""

    typ: str
    iss: str
    sub: str
    aud: tuple[str, ...]
    iat: int
    exp: int
    jti: str
    act: str | None = None
    """On an exchanged session: the machine that asked, `node:<name>`."""
    sid: str | None = None
    """On an exchanged session: the `jti` of the session it came from."""

    @property
    def is_session(self) -> bool:
        return self.typ == TYP_SESSION

    @property
    def is_service(self) -> bool:
        return self.typ == TYP_SERVICE

    @property
    def is_client(self) -> bool:
        return self.typ == TYP_CLIENT

    @property
    def issuer_node(self) -> str | None:
        return node_of(self.iss)

    def is_local_service(self, recipient: str) -> bool:
        """A service token its own machine minted for itself."""
        return self.is_service and self.iss == recipient and self.aud == (recipient,)


def _require_str(claims: Mapping[str, Any], name: str) -> str:
    value = claims.get(name)
    if not isinstance(value, str) or not value:
        raise TokenError(f"the token has no {name!r} claim")
    return value


def _aud(claims: Mapping[str, Any]) -> tuple[str, ...]:
    value = claims.get("aud")
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value or not all(isinstance(a, str) for a in value):
        raise TokenError("the token's aud must be a non-empty list of recipients")
    return tuple(value)


def _check_grants(key: TrustKey, typ: str, sub: str, aud: tuple[str, ...]) -> None:
    if GRANT_AUTHORITY in key.grants:
        return
    if typ in (TYP_SESSION, TYP_CLIENT):
        raise TokenError(f"{key.issuer} may not issue a {typ}: only the authority mints one")
    if GRANT_NODE not in key.grants:
        raise TokenError(f"{key.issuer} holds no grant that issues a service token")
    if all(a == key.issuer for a in aud):
        return
    if sub == SUB_AGENT:
        return
    if sub == SUB_GATEWAY and GRANT_GATEWAY in key.grants:
        return
    raise TokenError(
        f"{key.issuer} may not send a {sub!r} service token off its own machine"
        + ("" if sub != SUB_GATEWAY else ": it holds no gateway grant")
    )


def _check_lifetime(typ: str, iss: str, aud: tuple[str, ...], lifetime: int, act: Any) -> None:
    if typ == TYP_SESSION:
        limit = MAX_EXCHANGED_SECONDS if act is not None else MAX_SESSION_SECONDS
    elif typ == TYP_CLIENT:
        limit = MAX_CLIENT_SECONDS
    elif all(a == iss for a in aud):
        limit = MAX_LOCAL_SERVICE_SECONDS
    else:
        limit = MAX_REMOTE_SERVICE_SECONDS
    if lifetime > limit:
        raise TokenError(f"a {typ} to {list(aud)} may live {limit} s; this one claims {lifetime} s")


def verify(
    token: str,
    *,
    bundle: TrustBundle,
    recipient: str,
    classes: Collection[str],
) -> Claims:
    """Every check the module docstring lists, or `TokenError`."""
    try:
        header = jwt.get_unverified_header(token)
    except jwt.InvalidTokenError as exc:
        raise TokenError(f"not a JWT: {exc}") from exc
    typ = header.get("typ")
    if typ not in classes:
        raise TokenError(f"a {typ!r} is not accepted here")
    kid = header.get("kid")
    key = bundle.keys.get(kid) if isinstance(kid, str) else None
    if key is None:
        raise TokenError("the key that signed this token is not in the trust bundle")
    try:
        claims = jwt.decode(
            token,
            key.public,
            algorithms=["EdDSA"],
            options={"require": ["iss", "sub", "aud", "iat", "exp", "jti"], "verify_aud": False},
            leeway=LEEWAY_SECONDS,
        )
    except jwt.InvalidTokenError as exc:
        raise TokenError(str(exc)) from exc

    iss = _require_str(claims, "iss")
    sub = _require_str(claims, "sub")
    jti = _require_str(claims, "jti")
    aud = _aud(claims)
    if iss != key.issuer:
        raise TokenError(f"the token says it is from {iss!r} but its key belongs to {key.issuer!r}")
    wanted = RECIPIENT_GATEWAY if typ == TYP_CLIENT else recipient
    if wanted not in aud:
        raise TokenError(f"the token is addressed to {list(aud)}, not to {wanted!r}")
    try:
        iat = int(claims["iat"])
        exp = int(claims["exp"])
    except (TypeError, ValueError) as exc:
        raise TokenError("iat and exp must be integers") from exc
    note_clock_skew(iat)

    act_claim = claims.get("act")
    act: str | None = None
    if act_claim is not None:
        if typ != TYP_SESSION or not isinstance(act_claim, dict):
            raise TokenError("act is only meaningful on an exchanged session")
        act = _require_str(act_claim, "sub")
    sid_claim = claims.get("sid")
    if sid_claim is not None and not isinstance(sid_claim, str):
        raise TokenError("sid must be a string")

    _check_grants(key, str(typ), sub, aud)
    _check_lifetime(str(typ), iss, aud, exp - iat, act_claim)

    if typ == TYP_SESSION:
        if jti in bundle.revoked_sessions or (sid_claim and sid_claim in bundle.revoked_sessions):
            raise TokenError("this session was signed out")
        if GRANT_AUTHORITY in key.grants and iss == ISSUER_CONTROL:
            members = bundle.members
            named = [node_of(a) for a in aud] + [node_of(act) if act else None]
            gone = sorted(n for n in named if n is not None and n not in members)
            if gone:
                raise TokenError(f"this session names machines no longer in the install: {gone}")
    return Claims(
        typ=str(typ),
        iss=iss,
        sub=sub,
        aud=aud,
        iat=iat,
        exp=exp,
        jti=jti,
        act=act,
        sid=sid_claim,
    )
