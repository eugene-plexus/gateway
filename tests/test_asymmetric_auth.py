"""R7: algorithm selection comes from trusted key material, never token headers."""

import base64
import time

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from eugene_plexus_gateway import security
from eugene_plexus_gateway.auth_state import load_auth_state


def _pair():
    private = Ed25519PrivateKey.generate()
    return (
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
        private.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        ),
    )


def _claims(aud="operator"):
    now = int(time.time())
    return {"sub": "operator", "aud": aud, "iat": now, "exp": now + 60}


def test_public_key_verifies_eddsa():
    private, public = _pair()
    token = jwt.encode(_claims(), private, algorithm="EdDSA")
    assert security.decode_token(token=token, signing_key=public).aud == "operator"


def test_legacy_hs256_remains_available_only_with_legacy_key():
    legacy = b"L" * 32
    token = jwt.encode(_claims(), legacy, algorithm="HS256")
    assert security.decode_token(token=token, signing_key=legacy).aud == "operator"
    _, public = _pair()
    with pytest.raises(jwt.InvalidTokenError):
        security.decode_token(token=token, signing_key=public)


def test_public_material_cannot_be_used_as_hmac_secret():
    _, public = _pair()
    raw_public = serialization.load_pem_public_key(public).public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    token = jwt.encode(_claims(), raw_public, algorithm="HS256")
    with pytest.raises(jwt.InvalidTokenError):
        security.decode_token(token=token, signing_key=public)


def test_wrong_public_key_and_unsigned_tokens_are_refused():
    private, _ = _pair()
    _, other = _pair()
    for token in (
        jwt.encode(_claims(), private, algorithm="EdDSA"),
        jwt.encode(_claims(), key="", algorithm="none"),
    ):
        with pytest.raises(jwt.InvalidTokenError):
            security.decode_token(token=token, signing_key=other)


def test_bootstrap_accepts_public_key_without_retaining_private_material():
    private, public = _pair()
    auth = load_auth_state(
        signing_key_b64=None,
        verify_key_b64=base64.b64encode(public).decode(),
        service_token="outbound-token",
        master_key_b64=None,
    )
    assert not auth.auth_disabled
    assert auth.signing_key == public
    assert private not in repr(auth).encode()


@pytest.mark.parametrize("bad", ["private", "legacy", "garbage", "empty"])
def test_verify_bootstrap_fails_closed(bad):
    private, _ = _pair()
    raw = {"private": private, "legacy": b"L" * 32, "garbage": b"invalid", "empty": b""}[bad]
    with pytest.raises(ValueError):
        load_auth_state(
            signing_key_b64=None,
            verify_key_b64=base64.b64encode(raw).decode(),
            service_token="outbound-token",
            master_key_b64=None,
        )


def test_ambiguous_bootstrap_is_refused():
    _, public = _pair()
    with pytest.raises(ValueError):
        load_auth_state(
            signing_key_b64=base64.b64encode(b"L" * 32).decode(),
            verify_key_b64=base64.b64encode(public).decode(),
            service_token="outbound-token",
            master_key_b64=None,
        )


def test_application_wires_public_bootstrap_before_starting_clients(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from eugene_plexus_gateway import app as app_module
    from eugene_plexus_gateway.settings import Settings

    _, public = _pair()
    encoded = base64.b64encode(public).decode()

    class AuthReached(Exception):
        pass

    def capture(**kwargs):
        assert kwargs["verify_key_b64"] == encoded
        assert kwargs["signing_key_b64"] is None
        raise AuthReached()

    monkeypatch.setattr(app_module, "load_auth_state", capture)
    app = app_module.create_app(
        Settings(
            config_file=tmp_path / "config.yaml", auth_verify_key=encoded, service_token="outbound"
        )
    )
    with pytest.raises(AuthReached), TestClient(app):
        pass


def test_legacy_bootstrap_cannot_smuggle_private_pem():
    private, _ = _pair()
    with pytest.raises(ValueError):
        load_auth_state(
            signing_key_b64=base64.b64encode(private).decode(),
            service_token="outbound",
            master_key_b64=None,
        )
