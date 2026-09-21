"""Client keys at the front door, and the guard that turns one off.

Hobbyist UX S4. The gateway is the only thing in the install that
accepts `aud: client`, and it accepts it on three paths. Two families of
assertion here, and the second is the load-bearing one:

  * a client key **works** on `/v1/models`, `/v1/chat/completions` and
    `/v1/embeddings`;
  * a client key is **refused** by every operator path on this gateway,
    and stops working within a refresh interval of being revoked.

The guard is exercised against a fake agent rather than a mock, so the
fail-open path -- the one that decides whether an agent restart takes an
install's harnesses down -- is tested by making the agent actually fail.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.auth_state import AuthState
from eugene_plexus_gateway.client_keys import ClientKeyGuard
from eugene_plexus_gateway.settings import Settings
from tests.conftest import FakeDriverClient, make_routing_table

_JWT_ALG = "HS256"

# The model `make_routing_table`'s fake driver serves. Named rather than
# typed inline: a chat test that 404s on a wrong model name is a test
# that passed its auth check and asserted nothing about it.
FAKE_MODEL = "Qwen3-30B-A3B-Q4_K_M"


def _issue(
    *,
    signing_key: bytes,
    sub: str,
    aud: str,
    ttl_seconds: int = 3600,
    jti: str | None = None,
) -> str:
    claims: dict[str, Any] = {
        "sub": sub,
        "aud": aud,
        "iat": int(time.time()),
        "exp": int(time.time()) + ttl_seconds,
    }
    if jti is not None:
        claims["jti"] = jti
    return jwt.encode(claims, signing_key, algorithm=_JWT_ALG)


class FakeAgent:
    """The agent's `/v1/auth/client-keys/revoked`, over a real transport.

    **Not a patched `_fetch`.** An earlier version of this file replaced
    the guard's fetch method, which meant the fail-open branch -- the
    one that decides whether an agent restart takes an install's
    harnesses down -- was asserted about rather than run. `MockTransport`
    puts the real `_fetch` on the real code path: the HTTP call, the
    status check, the JSON parse and the `except` are all exercised, and
    `fail` produces a genuine `httpx.ConnectError` for the guard to
    catch.
    """

    def __init__(self) -> None:
        self.revoked: list[str] = []
        self.revision = 0
        self.fail = False
        self.reads = 0
        self.tokens: list[str | None] = []

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.reads += 1
        self.tokens.append(request.headers.get("authorization"))
        if self.fail:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/v1/auth/client-keys/admission":
            body = json.loads(request.content)
            if body["keyId"] in self.revoked and body["action"] != "release":
                return httpx.Response(401)
            return httpx.Response(
                200, json={"keyId": body["keyId"], "keyName": "Continue", "leaseSeconds": 30}
            )
        assert request.url.path == "/v1/auth/client-keys/policy", request.url
        keys = [
            {
                "id": key,
                "expiresAt": datetime.fromtimestamp(time.time() + 3600, UTC).isoformat(),
                **({"revokedAt": datetime.now(UTC).isoformat()} if key in self.revoked else {}),
            }
            for key in set(["a", "key-1", "key-2", *self.revoked])
        ]
        return httpx.Response(
            200,
            json={
                "authority": "control:test",
                "generatedAt": time.time(),
                "keys": keys,
                "revision": self.revision,
            },
        )

    def as_guard(self, **kwargs: Any) -> ClientKeyGuard:
        guard = ClientKeyGuard(agent_url="http://agent.test", service_token="svc-token", **kwargs)
        guard._client = httpx.AsyncClient(transport=httpx.MockTransport(self._handle))
        return guard

    def revoke(self, key_id: str) -> None:
        self.revoked.append(key_id)
        self.revision += 1


@pytest.fixture
def signing_key() -> bytes:
    return secrets.token_bytes(32)


@pytest.fixture
def agent() -> FakeAgent:
    return FakeAgent()


@pytest.fixture
def authed_app(
    settings: Settings,
    signing_key: bytes,
    fake_driver: FakeDriverClient,
    agent: FakeAgent,
) -> FastAPI:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake_driver)
    app.state.auth_state = AuthState(
        signing_key=signing_key,
        service_token=_issue(
            signing_key=signing_key, sub="gateway", aud="service:gateway", ttl_seconds=86400
        ),
        master_key=None,
    )
    # A short TTL so "the gateway notices a revocation" is a test rather
    # than a fifteen-second wait.
    app.state.client_key_guard = agent.as_guard(ttl_seconds=0.0)
    return app


@pytest.fixture
def authed_client(authed_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(authed_app) as c:
        yield c


@pytest.fixture
def client_key(signing_key: bytes) -> str:
    return _issue(signing_key=signing_key, sub="Continue", aud="client", jti="key-1")


# --------------------------------------------------------------------- #
# It works on the front door
# --------------------------------------------------------------------- #


def test_a_client_key_lists_models(authed_client: TestClient, client_key: str) -> None:
    resp = authed_client.get("/v1/models", headers={"Authorization": f"Bearer {client_key}"})
    assert resp.status_code == 200


def test_a_client_key_completes_a_chat(authed_client: TestClient, client_key: str) -> None:
    resp = authed_client.post(
        "/v1/chat/completions",
        json={"model": FAKE_MODEL, "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {client_key}"},
    )
    assert resp.status_code == 200, resp.text


def test_a_client_key_reaches_embeddings(authed_client: TestClient, client_key: str) -> None:
    """A 200 is not required -- this install may serve no embedding
    model -- but a 401 would mean the audience was refused, which is the
    only thing this test is about."""
    resp = authed_client.post(
        "/v1/embeddings",
        json={"model": FAKE_MODEL, "input": "hello"},
        headers={"Authorization": f"Bearer {client_key}"},
    )
    assert resp.status_code != 401


# --------------------------------------------------------------------- #
# And nowhere else
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/v1/config"),
        ("get", "/v1/config/schema"),
        ("get", "/v1/admin/drivers"),
        ("get", "/v1/admin/routing"),
        ("get", "/v1/metrics"),
        ("get", "/v1/metrics/requests"),
        ("get", "/v1/metrics/clients"),
    ],
)
def test_a_client_key_opens_no_operator_path(
    authed_client: TestClient, client_key: str, method: str, path: str
) -> None:
    """The narrowing the audience exists for.

    A key pasted into Open WebUI must not be able to read what every
    other caller asked this gateway, nor edit its config. `_validate`'s
    `accept_client` defaults to False, so this holds for any endpoint
    added later that forgets to think about it.
    """
    resp = getattr(authed_client, method)(path, headers={"Authorization": f"Bearer {client_key}"})
    assert resp.status_code == 401, f"{path} accepted a client key"


def test_an_operator_token_still_works_everywhere(
    authed_client: TestClient, signing_key: bytes
) -> None:
    token = _issue(signing_key=signing_key, sub="operator", aud="operator")
    assert (
        authed_client.get("/v1/config", headers={"Authorization": f"Bearer {token}"}).status_code
        == 200
    )
    assert (
        authed_client.get("/v1/models", headers={"Authorization": f"Bearer {token}"}).status_code
        == 200
    )


# --------------------------------------------------------------------- #
# Revocation
# --------------------------------------------------------------------- #


def test_a_revoked_key_stops_working_and_says_why(
    authed_client: TestClient, client_key: str, agent: FakeAgent
) -> None:
    assert (
        authed_client.get(
            "/v1/models", headers={"Authorization": f"Bearer {client_key}"}
        ).status_code
        == 200
    )

    agent.revoke("key-1")

    resp = authed_client.get("/v1/models", headers={"Authorization": f"Bearer {client_key}"})
    assert resp.status_code == 401
    # The sentence has to tell a person what to do, not what failed.
    assert "new one" in resp.text or "revoked" in resp.text.lower()


def test_revoking_one_key_leaves_the_others(
    authed_client: TestClient, signing_key: bytes, agent: FakeAgent
) -> None:
    """The whole reason for a key per app.

    If revoking took every key down it would be the signing-key rotation
    with extra steps, and there would be no point minting more than one.
    """
    laptop = _issue(signing_key=signing_key, sub="laptop", aud="client", jti="key-1")
    phone = _issue(signing_key=signing_key, sub="phone", aud="client", jti="key-2")
    agent.revoke("key-1")

    assert (
        authed_client.get("/v1/models", headers={"Authorization": f"Bearer {laptop}"}).status_code
        == 401
    )
    assert (
        authed_client.get("/v1/models", headers={"Authorization": f"Bearer {phone}"}).status_code
        == 200
    )


def test_a_client_token_with_no_jti_is_refused(
    authed_client: TestClient, signing_key: bytes
) -> None:
    """A credential that can never be revoked is not one to accept.

    Every key the agent mints carries a `jti`; a client-audience token
    without one did not come from this install's mint, and there would
    be no way to turn it off.
    """
    token = _issue(signing_key=signing_key, sub="mystery", aud="client")
    resp = authed_client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 401


def test_authority_outage_refuses_discovery_even_with_cached_authentication(
    authed_client: TestClient, client_key: str, agent: FakeAgent
) -> None:
    headers = {"Authorization": f"Bearer {client_key}"}
    assert authed_client.get("/v1/models", headers=headers).status_code == 200
    agent.fail = True
    assert authed_client.get("/v1/models", headers=headers).status_code == 503


def test_a_revocation_survives_the_agent_going_down_afterwards(
    authed_client: TestClient, client_key: str, agent: FakeAgent
) -> None:
    """Fail-open keeps the LAST answer, not an empty one.

    A gateway that forgot the list when a read failed would un-revoke
    every key on the first blip -- which is the shape of failure that
    makes fail-open indefensible, and is not what this does.
    """
    agent.revoke("key-1")
    assert (
        authed_client.get(
            "/v1/models", headers={"Authorization": f"Bearer {client_key}"}
        ).status_code
        == 401
    )
    agent.fail = True
    assert (
        authed_client.get(
            "/v1/models", headers={"Authorization": f"Bearer {client_key}"}
        ).status_code
        == 401
    )


def test_only_a_client_key_consults_the_agent(
    authed_client: TestClient, signing_key: bytes, agent: FakeAgent
) -> None:
    """An install where nobody minted a client key never makes the call."""
    operator = _issue(signing_key=signing_key, sub="operator", aud="operator")
    service = _issue(signing_key=signing_key, sub="library", aud="service:library")
    authed_client.get("/v1/models", headers={"Authorization": f"Bearer {operator}"})
    authed_client.get("/v1/models", headers={"Authorization": f"Bearer {service}"})
    assert agent.reads == 0


# --------------------------------------------------------------------- #
# The guard itself
# --------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_guard_presents_the_gateways_service_token() -> None:
    """The agent narrows this endpoint to `service:gateway` exactly, so a
    guard that forgot the header would read 401 and fail open forever --
    silently, because failing open looks like working."""
    agent = FakeAgent()
    guard = agent.as_guard(ttl_seconds=0.0)
    await guard.is_revoked("a")
    assert agent.tokens == ["Bearer svc-token"]


@pytest.mark.asyncio
async def test_a_401_from_the_agent_is_a_failed_read_not_an_empty_list() -> None:
    """The failure that would silently un-revoke every key.

    A wrong or expired service token answers 401 with a JSON body, which
    a guard that only caught connection errors would parse as `ids: []`.
    `raise_for_status` is what makes it a failure.
    """
    agent = FakeAgent()
    agent.revoke("gone")
    guard = agent.as_guard(ttl_seconds=0.0)
    await guard.is_revoked("gone")
    assert await guard.is_revoked("gone") is True

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "nope"})

    guard._client = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    assert await guard.is_revoked("gone") is True


@pytest.mark.asyncio
async def test_the_guard_caches_within_its_ttl() -> None:
    agent = FakeAgent()
    guard = agent.as_guard(ttl_seconds=60.0)
    assert await guard.is_revoked("a") is False
    assert await guard.is_revoked("a") is False
    assert agent.reads == 1, "a second question inside the TTL must not be a second round trip"


@pytest.mark.asyncio
async def test_a_failed_read_does_not_make_the_copy_look_fresh() -> None:
    """Otherwise one blip would buy a whole TTL of not asking again."""
    agent = FakeAgent()
    agent.fail = True
    guard = agent.as_guard(ttl_seconds=60.0)
    await guard.is_revoked("a")
    await guard.is_revoked("a")
    assert agent.reads == 1
    assert await guard.decision("a") == "unavailable"


def test_no_policy_is_503_for_both_protocols_but_operator_can_repair(
    authed_client, client_key, agent, signing_key
):
    agent.fail = True
    assert (
        authed_client.get(
            "/v1/models", headers={"Authorization": f"Bearer {client_key}"}
        ).status_code
        == 503
    )
    body = {
        "model": FAKE_MODEL,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hello"}],
    }
    response = authed_client.post("/v1/messages", json=body, headers={"x-api-key": client_key})
    assert response.status_code == 503 and response.json()["error"]["type"] == "api_error"
    operator = _issue(signing_key=signing_key, sub="operator", aud="operator")
    assert (
        authed_client.get("/v1/config", headers={"Authorization": f"Bearer {operator}"}).status_code
        == 200
    )


def test_revoked_and_unknown_anthropic_keys_are_403(authed_client, client_key, agent, signing_key):
    agent.revoke("key-1")
    body = {
        "model": FAKE_MODEL,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hello"}],
    }
    response = authed_client.post("/v1/messages", json=body, headers={"x-api-key": client_key})
    assert response.status_code == 403 and "turned off" in response.text.lower()
    unknown = _issue(signing_key=signing_key, sub="app", aud="client", jti="not-in-registry")
    response = authed_client.post("/v1/messages", json=body, headers={"x-api-key": unknown})
    assert response.status_code == 403 and "not registered" in response.text.lower()
