"""The token module, tested once and copied with it into every component.

Each case names the RFC 8725 rule or the design decision it pins
(`specs/docs/design/per-node-token-keys.md`), so a copy that drifts
fails for a reason someone can read.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eugene_plexus_gateway import tokens


def _now() -> int:
    """Read at the moment of use: a module-level constant goes stale by
    however long the suite has run, and a leeway test then measures that."""
    return int(time.time())


def _signer(issuer: str) -> tokens.Signer:
    return tokens.Signer(key=tokens.generate_private_key(), issuer=issuer)


@pytest.fixture
def install() -> dict[str, object]:
    authority = tokens.generate_private_key()
    root = _signer("control")
    nas = _signer("node:nas")
    gpu = _signer("node:gpu-box")
    bundle = tokens.build_bundle(
        authority=authority,
        version=10,
        epoch=1,
        keys=[
            root.trust_key([tokens.GRANT_AUTHORITY]),
            nas.trust_key([tokens.GRANT_NODE, tokens.GRANT_GATEWAY]),
            gpu.trust_key([tokens.GRANT_NODE]),
        ],
    )
    return {"authority": authority, "root": root, "nas": nas, "gpu": gpu, "bundle": bundle}


def _verify(install: dict[str, object], token: str, recipient: str, *classes: str) -> tokens.Claims:
    bundle = install["bundle"]
    assert isinstance(bundle, tokens.TrustBundle)
    return tokens.verify(
        token,
        bundle=bundle,
        recipient=recipient,
        classes=classes or (tokens.TYP_SESSION, tokens.TYP_SERVICE, tokens.TYP_CLIENT),
    )


# --------------------------------------------------------------------------- keys


def test_the_thumbprint_is_rfc_8037s_published_value() -> None:
    """RFC 8037 Appendix A.3: the one external vector, so two copies cannot agree on a wrong one."""
    x = "11qYAYKxCrfVS_7TyWQHOg7hcvPapiMlrwIaaPcHURo"
    raw = base64.urlsafe_b64decode(x + "=")
    key = tokens.load_public(base64.b64encode(raw).decode("ascii"))
    assert tokens.thumbprint(key) == "kPrK_qmxVWaYVA9wwBF6Iuo3vVzz7TxHCTwXBygrS4k"


def test_a_private_key_round_trips_and_an_hmac_key_does_not_load() -> None:
    key = tokens.generate_private_key()
    again = tokens.load_private(tokens.private_to_b64(key))
    assert tokens.public_raw(again) == tokens.public_raw(key)
    # 32 random bytes ARE a valid Ed25519 seed, so the old HMAC keys would
    # load silently as a different key; only PEM or base64 raw are formats,
    # and a 16-byte or PEM-RSA input is refused.
    with pytest.raises(ValueError):
        tokens.load_private(base64.b64encode(os.urandom(16)).decode())


# --------------------------------------------------------------------------- the bundle


def test_a_bundle_round_trips_against_its_pinned_authority(install: dict[str, object]) -> None:
    bundle = install["bundle"]
    authority = install["authority"]
    assert isinstance(bundle, tokens.TrustBundle) and isinstance(authority, Ed25519PrivateKey)
    again = tokens.parse_bundle(bundle.jws, authority=tokens.public_b64(authority))
    assert again.version == 10 and again.members == frozenset({"nas", "gpu-box"})


def test_a_bundle_signed_by_anything_else_is_refused(install: dict[str, object]) -> None:
    bundle = install["bundle"]
    assert isinstance(bundle, tokens.TrustBundle)
    impostor = tokens.public_b64(tokens.generate_private_key())
    with pytest.raises(tokens.BundleError):
        tokens.parse_bundle(bundle.jws, authority=impostor)


def test_a_bundle_whose_payload_was_edited_is_refused(install: dict[str, object]) -> None:
    bundle = install["bundle"]
    authority = install["authority"]
    assert isinstance(bundle, tokens.TrustBundle) and isinstance(authority, Ed25519PrivateKey)
    head, body, sig = bundle.jws.split(".")
    claims = json.loads(base64.urlsafe_b64decode(body + "=="))
    claims["keys"] = claims["keys"][:1]
    forged_body = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    with pytest.raises(tokens.BundleError):
        tokens.parse_bundle(f"{head}.{forged_body}.{sig}", authority=tokens.public_b64(authority))


def test_a_bundle_naming_a_kid_that_is_not_its_keys_thumbprint_is_refused() -> None:
    authority = tokens.generate_private_key()
    key = tokens.generate_private_key()
    payload = {
        "version": 1,
        "epoch": 1,
        "iat": _now(),
        "authority": tokens.public_b64(authority),
        "keys": [
            {
                "kid": "not-the-thumbprint",
                "issuer": "control",
                "publicKey": tokens.public_b64(key),
                "grants": ["authority"],
            }
        ],
        "revokedSessions": [],
    }
    jws = jwt.encode(payload, authority, algorithm="EdDSA", headers={"typ": tokens.TYP_BUNDLE})
    with pytest.raises(tokens.BundleError):
        tokens.parse_bundle(jws, authority=tokens.public_b64(authority))


def test_a_lower_version_or_epoch_cannot_replace_the_held_bundle(
    install: dict[str, object],
) -> None:
    held = install["bundle"]
    authority = install["authority"]
    assert isinstance(held, tokens.TrustBundle) and isinstance(authority, Ed25519PrivateKey)
    keys = list(held.keys.values())
    older = tokens.build_bundle(authority=authority, version=9, epoch=1, keys=keys)
    stale_epoch = tokens.build_bundle(authority=authority, version=11, epoch=0, keys=keys)
    same = tokens.build_bundle(authority=authority, version=10, epoch=1, keys=keys)
    assert tokens.accepts_replacement(held, older) is not None
    assert tokens.accepts_replacement(held, stale_epoch) is not None
    assert tokens.accepts_replacement(held, same) is None


def test_expired_sign_outs_are_pruned_and_live_ones_kept() -> None:
    authority = tokens.generate_private_key()
    bundle = tokens.build_bundle(
        authority=authority,
        version=1,
        epoch=1,
        keys=[],
        revoked_sessions=[("gone", _now() - 10 * 3600), ("live", _now() + 3600)],
        now=_now(),
    )
    assert bundle.revoked_sessions == frozenset({"live"})


def test_the_bundle_file_reloads_refuses_rollback_and_keeps_the_last_good(
    tmp_path: Path, install: dict[str, object]
) -> None:
    held = install["bundle"]
    authority = install["authority"]
    assert isinstance(held, tokens.TrustBundle) and isinstance(authority, Ed25519PrivateKey)
    path = tmp_path / "trust_bundle.json"
    tokens.write_bundle_file(path, held)
    source = tokens.BundleFile(path, authority=tokens.public_b64(authority), check_interval=0.0)
    got = source.get()
    assert got is not None and got.version == 10

    keys = list(held.keys.values())
    tokens.write_bundle_file(
        path, tokens.build_bundle(authority=authority, version=12, epoch=1, keys=keys)
    )
    got = source.get()
    assert got is not None and got.version == 12

    tokens.write_bundle_file(
        path, tokens.build_bundle(authority=authority, version=11, epoch=1, keys=keys)
    )
    got = source.get()
    assert got is not None and got.version == 12 and source.error and "below" in source.error

    path.write_text("{not json", encoding="utf-8")
    got = source.get()
    assert got is not None and got.version == 12


# --------------------------------------------------------------------------- verify: accepted


def test_a_session_verifies_at_its_console_and_at_the_root(install: dict[str, object]) -> None:
    root = install["root"]
    assert isinstance(root, tokens.Signer)
    token, _ = root.mint(
        typ=tokens.TYP_SESSION, sub="operator", aud=["node:nas", "control"], ttl_seconds=3600
    )
    assert _verify(install, token, "node:nas").is_session
    assert _verify(install, token, "control").sub == "operator"
    with pytest.raises(tokens.TokenError, match="addressed"):
        _verify(install, token, "node:gpu-box")


def test_a_client_key_is_checked_against_the_gateway_recipient(install: dict[str, object]) -> None:
    root = install["root"]
    assert isinstance(root, tokens.Signer)
    token, _ = root.mint(
        typ=tokens.TYP_CLIENT, sub="Open WebUI", aud=["gateway"], ttl_seconds=86400, jti="k1"
    )
    claims = _verify(install, token, "node:nas")
    assert claims.is_client and claims.jti == "k1"


def test_a_node_may_send_its_own_machine_any_service_token(install: dict[str, object]) -> None:
    gpu = install["gpu"]
    assert isinstance(gpu, tokens.Signer)
    token, _ = gpu.mint(
        typ=tokens.TYP_SERVICE, sub="inference-driver", aud=["node:gpu-box"], ttl_seconds=86400
    )
    claims = _verify(install, token, "node:gpu-box")
    assert claims.is_local_service("node:gpu-box")


def test_an_agent_token_may_leave_its_machine(install: dict[str, object]) -> None:
    gpu = install["gpu"]
    assert isinstance(gpu, tokens.Signer)
    token, _ = gpu.mint(typ=tokens.TYP_SERVICE, sub="agent", aud=["node:nas"], ttl_seconds=600)
    claims = _verify(install, token, "node:nas")
    assert claims.issuer_node == "gpu-box" and not claims.is_local_service("node:nas")


def test_a_gateway_token_leaves_only_a_machine_granted_gateway(install: dict[str, object]) -> None:
    nas = install["nas"]
    gpu = install["gpu"]
    assert isinstance(nas, tokens.Signer) and isinstance(gpu, tokens.Signer)
    good, _ = nas.mint(typ=tokens.TYP_SERVICE, sub="gateway", aud=["node:gpu-box"], ttl_seconds=900)
    assert _verify(install, good, "node:gpu-box").sub == "gateway"
    bad, _ = gpu.mint(typ=tokens.TYP_SERVICE, sub="gateway", aud=["node:nas"], ttl_seconds=900)
    with pytest.raises(tokens.TokenError, match="no gateway grant"):
        _verify(install, bad, "node:nas")


# --------------------------------------------------------------------------- verify: refused


def test_a_node_key_cannot_mint_a_session_or_a_client_key(install: dict[str, object]) -> None:
    """D4: the whole point. A leaked worker key mints nothing the install honours."""
    gpu = install["gpu"]
    assert isinstance(gpu, tokens.Signer)
    session, _ = gpu.mint(typ=tokens.TYP_SESSION, sub="operator", aud=["node:nas"], ttl_seconds=60)
    client, _ = gpu.mint(typ=tokens.TYP_CLIENT, sub="x", aud=["gateway"], ttl_seconds=60)
    for token in (session, client):
        with pytest.raises(tokens.TokenError, match="only the authority"):
            _verify(install, token, "node:nas")


def test_a_node_key_cannot_mint_a_control_or_driver_token_for_another_machine(
    install: dict[str, object],
) -> None:
    gpu = install["gpu"]
    assert isinstance(gpu, tokens.Signer)
    for sub in ("control", "inference-driver", "library"):
        token, _ = gpu.mint(typ=tokens.TYP_SERVICE, sub=sub, aud=["node:nas"], ttl_seconds=60)
        with pytest.raises(tokens.TokenError, match="off its own machine"):
            _verify(install, token, "node:nas")


def test_a_class_the_route_does_not_accept_is_refused(install: dict[str, object]) -> None:
    """RFC 8725 §3.11: a service token cannot be passed off as a session."""
    gpu = install["gpu"]
    assert isinstance(gpu, tokens.Signer)
    token, _ = gpu.mint(typ=tokens.TYP_SERVICE, sub="agent", aud=["node:gpu-box"], ttl_seconds=60)
    with pytest.raises(tokens.TokenError, match="not accepted"):
        _verify(install, token, "node:gpu-box", tokens.TYP_SESSION)


def test_a_key_the_bundle_does_not_list_is_refused(install: dict[str, object]) -> None:
    stranger = _signer("control")
    token, _ = stranger.mint(
        typ=tokens.TYP_SESSION, sub="operator", aud=["control"], ttl_seconds=60
    )
    with pytest.raises(tokens.TokenError, match="not in the trust bundle"):
        _verify(install, token, "control")


def test_hs256_signed_with_the_public_key_is_refused(install: dict[str, object]) -> None:
    """RFC 8725 §3.1, the classic confusion: HMAC keyed with the public key bytes."""
    root = install["root"]
    assert isinstance(root, tokens.Signer)
    header = {"alg": "HS256", "typ": tokens.TYP_SESSION, "kid": root.kid}
    claims = {
        "iss": "control",
        "sub": "operator",
        "aud": ["control"],
        "iat": _now(),
        "exp": _now() + 60,
        "jti": "x",
    }

    def seg(obj: object) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    signing_input = f"{seg(header)}.{seg(claims)}"
    mac = hmac.new(tokens.public_raw(root.key), signing_input.encode(), hashlib.sha256).digest()
    forged = f"{signing_input}.{base64.urlsafe_b64encode(mac).rstrip(b'=').decode()}"
    with pytest.raises(tokens.TokenError):
        _verify(install, forged, "control")


def test_an_issuer_that_does_not_own_the_key_is_refused(install: dict[str, object]) -> None:
    """RFC 8725 §3.12: `iss` and the key must agree."""
    gpu = install["gpu"]
    assert isinstance(gpu, tokens.Signer)
    liar = tokens.Signer(key=gpu.key, issuer="node:nas")
    token, _ = liar.mint(typ=tokens.TYP_SERVICE, sub="agent", aud=["node:nas"], ttl_seconds=60)
    with pytest.raises(tokens.TokenError, match="belongs to"):
        _verify(install, token, "node:nas")


def test_a_token_that_claims_to_live_too_long_is_refused(install: dict[str, object]) -> None:
    nas = install["nas"]
    gpu = install["gpu"]
    assert isinstance(nas, tokens.Signer) and isinstance(gpu, tokens.Signer)
    remote, _ = nas.mint(
        typ=tokens.TYP_SERVICE, sub="gateway", aud=["node:gpu-box"], ttl_seconds=7200
    )
    local, _ = gpu.mint(
        typ=tokens.TYP_SERVICE, sub="agent", aud=["node:gpu-box"], ttl_seconds=401 * tokens.DAY
    )
    with pytest.raises(tokens.TokenError, match="may live"):
        _verify(install, remote, "node:gpu-box")
    with pytest.raises(tokens.TokenError, match="may live"):
        _verify(install, local, "node:gpu-box")


def test_a_signed_out_session_and_everything_exchanged_from_it_are_refused() -> None:
    authority = tokens.generate_private_key()
    root = _signer("control")
    nas = _signer("node:nas")
    bundle = tokens.build_bundle(
        authority=authority,
        version=1,
        epoch=1,
        keys=[root.trust_key(["authority"]), nas.trust_key(["node"])],
        revoked_sessions=[("session-1", _now() + 3600)],
    )
    session, _ = root.mint(
        typ=tokens.TYP_SESSION, sub="operator", aud=["node:nas"], ttl_seconds=600, jti="session-1"
    )
    exchanged, _ = root.mint(
        typ=tokens.TYP_SESSION,
        sub="operator",
        aud=["node:nas"],
        ttl_seconds=300,
        extra={"act": {"sub": "node:nas"}, "sid": "session-1"},
    )
    for token in (session, exchanged):
        with pytest.raises(tokens.TokenError, match="signed out"):
            tokens.verify(token, bundle=bundle, recipient="node:nas", classes=[tokens.TYP_SESSION])


def test_a_session_bound_to_a_revoked_console_is_refused(install: dict[str, object]) -> None:
    """D9: revoking a console machine ends every session bound to it, everywhere."""
    root = install["root"]
    bundle = install["bundle"]
    authority = install["authority"]
    assert isinstance(root, tokens.Signer) and isinstance(bundle, tokens.TrustBundle)
    assert isinstance(authority, Ed25519PrivateKey)
    session, _ = root.mint(
        typ=tokens.TYP_SESSION, sub="operator", aud=["node:gpu-box", "control"], ttl_seconds=600
    )
    exchanged, _ = root.mint(
        typ=tokens.TYP_SESSION,
        sub="operator",
        aud=["node:nas"],
        ttl_seconds=300,
        extra={"act": {"sub": "node:gpu-box"}, "sid": "s"},
    )
    assert _verify(install, session, "control")
    assert _verify(install, exchanged, "node:nas").act == "node:gpu-box"
    without_gpu = tokens.build_bundle(
        authority=authority,
        version=11,
        epoch=1,
        keys=[k for k in bundle.keys.values() if k.issuer != "node:gpu-box"],
    )
    for token, recipient in ((session, "control"), (exchanged, "node:nas")):
        with pytest.raises(tokens.TokenError, match="no longer in the install"):
            tokens.verify(
                token, bundle=without_gpu, recipient=recipient, classes=[tokens.TYP_SESSION]
            )


def test_an_exchanged_token_may_live_ten_minutes_and_no_more(install: dict[str, object]) -> None:
    root = install["root"]
    assert isinstance(root, tokens.Signer)
    token, _ = root.mint(
        typ=tokens.TYP_SESSION,
        sub="operator",
        aud=["node:nas"],
        ttl_seconds=11 * 60,
        extra={"act": {"sub": "node:gpu-box"}, "sid": "s"},
    )
    with pytest.raises(tokens.TokenError, match="may live"):
        _verify(install, token, "node:nas")


def test_an_expired_token_is_refused_past_the_leeway(install: dict[str, object]) -> None:
    root = install["root"]
    assert isinstance(root, tokens.Signer)
    within, _ = root.mint(
        typ=tokens.TYP_SESSION, sub="operator", aud=["control"], ttl_seconds=60, now=_now() - 120
    )
    beyond, _ = root.mint(
        typ=tokens.TYP_SESSION, sub="operator", aud=["control"], ttl_seconds=60, now=_now() - 1000
    )
    assert _verify(install, within, "control")
    with pytest.raises(tokens.TokenError):
        _verify(install, beyond, "control")


# --------------------------------------------------------------------------- clock skew


def test_a_token_from_a_clock_half_a_second_ahead_is_accepted(
    install: dict[str, object],
) -> None:
    """2026-09-15: half a second of skew took a worker out of the install."""
    root = install["root"]
    assert isinstance(root, tokens.Signer)
    token, _ = root.mint(
        typ=tokens.TYP_SESSION, sub="operator", aud=["control"], ttl_seconds=60, now=_now() + 1
    )
    assert _verify(install, token, "control")


def test_skew_inside_the_leeway_is_accepted_and_past_it_refused(
    install: dict[str, object],
) -> None:
    root = install["root"]
    assert isinstance(root, tokens.Signer)
    inside, _ = root.mint(
        typ=tokens.TYP_SESSION, sub="operator", aud=["control"], ttl_seconds=60, now=_now() + 250
    )
    past, _ = root.mint(
        typ=tokens.TYP_SESSION, sub="operator", aud=["control"], ttl_seconds=60, now=_now() + 400
    )
    assert _verify(install, inside, "control")
    with pytest.raises(tokens.TokenError):
        _verify(install, past, "control")


def test_a_future_issued_token_is_logged_at_most_once_a_minute() -> None:
    tokens._last_skew_warning = 0.0
    assert tokens.note_clock_skew(1000, now=900.0) is True
    assert tokens.note_clock_skew(1000, now=910.0) is False
    assert tokens.note_clock_skew(2000, now=1000.0) is True
    assert tokens.note_clock_skew(1000, now=999.5) is False
