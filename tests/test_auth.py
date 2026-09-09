"""Tests for v0.2 bearer auth on the gateway.

The gateway is verify-only — the agent issues the signing key
and tokens. These tests stand in for the agent by constructing
JWTs directly via PyJWT against a known key, then asserting the
gateway's dependencies accept / reject the right shapes.

Auth posture is selected by whether `app.state.auth_state` is
pre-populated before the lifespan runs:

  * default fixtures (`client`) leave it unset → lifespan reads env
    vars (empty in tests) → `auth_disabled=True`. Verifies the
    backward-compat / dev path.
  * `authed_app` injects a real `AuthState` with a known signing key,
    exercising the production path.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Iterator

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.auth_state import AuthState
from eugene_plexus_gateway.settings import Settings
from tests.conftest import FakeDriverClient, make_routing_table

_JWT_ALG = "HS256"


def _issue(
    *,
    signing_key: bytes,
    sub: str,
    aud: str,
    ttl_seconds: int = 60,
    iat: int | None = None,
) -> str:
    """Mint a JWT exactly the way the agent would."""
    issued_at = iat if iat is not None else int(time.time())
    claims = {
        "sub": sub,
        "aud": aud,
        "iat": issued_at,
        "exp": issued_at + ttl_seconds,
    }
    return jwt.encode(claims, signing_key, algorithm=_JWT_ALG)


# --------------------------------------------------------------------------- #
# Auth-enabled fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def signing_key() -> bytes:
    return secrets.token_bytes(32)


@pytest.fixture
def authed_app(
    settings: Settings,
    signing_key: bytes,
    fake_driver: FakeDriverClient,
) -> FastAPI:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake_driver)
    # Pre-populate auth_state with a real signing key; the lifespan
    # leaves it alone because hasattr is True.
    app.state.auth_state = AuthState(
        signing_key=signing_key,
        service_token=_issue(
            signing_key=signing_key,
            sub="gateway",
            aud="service:gateway",
            ttl_seconds=365 * 24 * 3600,
        ),
        master_key=None,
    )
    return app


@pytest.fixture
def authed_client(authed_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(authed_app) as c:
        yield c


@pytest.fixture
def operator_token(signing_key: bytes) -> str:
    return _issue(signing_key=signing_key, sub="operator", aud="operator")


@pytest.fixture
def service_token(signing_key: bytes) -> str:
    return _issue(signing_key=signing_key, sub="connector", aud="service:connector")


# --------------------------------------------------------------------------- #
# Auth-disabled path (default `client` fixture)
# --------------------------------------------------------------------------- #


def test_auth_disabled_lets_everything_through(client: TestClient) -> None:
    """When no signing key is wired in, every route should answer
    normally without a bearer header. That's the dev / standalone
    posture — production via the agent supplies the env vars."""
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/admin/drivers").status_code == 200
    assert client.get("/v1/config").status_code == 200


# --------------------------------------------------------------------------- #
# Health stays unauthenticated even when auth is on
# --------------------------------------------------------------------------- #


def test_healthz_is_always_open(authed_client: TestClient) -> None:
    """Supervisors and load balancers must be able to probe /healthz
    without holding credentials. Auth-on must NOT change that."""
    response = authed_client.get("/healthz")
    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# Missing / malformed / wrong-key tokens
# --------------------------------------------------------------------------- #


def test_missing_bearer_rejects_with_401(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/config")
    assert response.status_code == 401
    body = response.json()
    assert "detail" in body
    assert body["detail"]["component"] == "gateway"


def test_wrong_signing_key_rejects(authed_client: TestClient) -> None:
    other_key = secrets.token_bytes(32)
    token = _issue(signing_key=other_key, sub="operator", aud="operator")
    response = authed_client.get("/v1/config", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_garbage_bearer_rejects(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/config", headers={"Authorization": "Bearer not.a.real.jwt"})
    assert response.status_code == 401


def test_expired_token_rejects(authed_client: TestClient, signing_key: bytes) -> None:
    expired = _issue(
        signing_key=signing_key,
        sub="operator",
        aud="operator",
        ttl_seconds=-60,
        iat=int(time.time()) - 120,
    )
    response = authed_client.get("/v1/config", headers={"Authorization": f"Bearer {expired}"})
    assert response.status_code == 401


# --------------------------------------------------------------------------- #
# Operator audience — accepted on operator routes AND mixed routes
# --------------------------------------------------------------------------- #


def test_operator_token_accepted_on_config(authed_client: TestClient, operator_token: str) -> None:
    response = authed_client.get(
        "/v1/config", headers={"Authorization": f"Bearer {operator_token}"}
    )
    assert response.status_code == 200


def test_operator_token_accepted_on_admin(authed_client: TestClient, operator_token: str) -> None:
    response = authed_client.get(
        "/v1/admin/drivers",
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert response.status_code == 200


def test_operator_token_accepted_on_chat(
    authed_client: TestClient, operator_token: str, fake_driver: FakeDriverClient
) -> None:
    fake_driver.responses = ["hello from the model"]
    response = authed_client.post(
        "/v1/chat/completions",
        json={
            "model": fake_driver.model_id,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": f"Bearer {operator_token}"},
    )
    assert response.status_code == 200, response.text


def test_service_token_rejected_on_config(authed_client: TestClient, service_token: str) -> None:
    """Config is operator-only: a peer component has no business editing it."""
    response = authed_client.get("/v1/config", headers={"Authorization": f"Bearer {service_token}"})
    assert response.status_code == 401


def test_service_token_rejected_on_admin_restart(
    authed_client: TestClient, service_token: str
) -> None:
    """And it certainly has no business restarting the gateway."""
    response = authed_client.post(
        "/v1/admin/restart", headers={"Authorization": f"Bearer {service_token}"}
    )
    assert response.status_code == 401


def test_service_token_accepted_on_chat(
    authed_client: TestClient, service_token: str, fake_driver: FakeDriverClient
) -> None:
    """The front door accepts service tokens: another component calling
    in (a Discord connector, say) is a legitimate caller of the one
    OpenAI-compatible endpoint."""
    fake_driver.responses = ["hello"]
    response = authed_client.post(
        "/v1/chat/completions",
        json={
            "model": fake_driver.model_id,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": f"Bearer {service_token}"},
    )
    assert response.status_code == 200, response.text


def test_service_token_accepted_on_models(authed_client: TestClient, service_token: str) -> None:
    response = authed_client.get("/v1/models", headers={"Authorization": f"Bearer {service_token}"})
    assert response.status_code == 200


def test_load_auth_state_disabled_when_no_signing_key() -> None:
    from eugene_plexus_gateway.auth_state import load_auth_state

    state = load_auth_state(signing_key_b64=None, service_token=None, master_key_b64=None)
    assert state.auth_disabled is True


def test_load_auth_state_rejects_partial_auth() -> None:
    """SERVICE_TOKEN without AUTH_SIGNING_KEY is a configuration bug —
    fail loudly rather than silently disabling auth."""
    from eugene_plexus_gateway.auth_state import load_auth_state

    with pytest.raises(ValueError, match="inconsistent"):
        load_auth_state(
            signing_key_b64=None,
            service_token="dummy",
            master_key_b64=None,
        )


def test_load_auth_state_requires_service_token_when_enabled(signing_key: bytes) -> None:
    """AUTH_SIGNING_KEY without SERVICE_TOKEN means the gateway
    has nothing to present on outbound calls — refuse."""
    import base64

    from eugene_plexus_gateway.auth_state import load_auth_state

    with pytest.raises(ValueError, match="SERVICE_TOKEN is missing"):
        load_auth_state(
            signing_key_b64=base64.b64encode(signing_key).decode("ascii"),
            service_token=None,
            master_key_b64=None,
        )


def test_load_auth_state_rejects_wrong_length_signing_key() -> None:
    """Bad config should produce a clear error, not a runtime jwt
    failure half a second after a request lands."""
    import base64

    from eugene_plexus_gateway.auth_state import load_auth_state

    short = base64.b64encode(b"\x00" * 16).decode("ascii")
    with pytest.raises(ValueError, match="32 bytes"):
        load_auth_state(
            signing_key_b64=short,
            service_token="dummy",
            master_key_b64=None,
        )
