"""Client policy through real gateway middleware, routing and metric storage.

The authority protocol is scripted here; the specs acceptance runner exercises
the durable authority and two independent gateway processes together.
"""

import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import Usage
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.auth_state import AuthState
from tests.conftest import FakeDriverClient, install_snapshot, make_routing_table
from tests.test_client_disconnect import _SCOPE, _Hang
from tests.test_client_keys import FakeAgent, _issue


class Authority(FakeAgent):
    def __init__(self):
        super().__init__()
        self.allowed = ["alias", "allowed"]
        self.calls = []
        self.refuse_acquire = None
        self.refuse_renew = None
        self.lease = 30
        self.local_only = False

    def _handle(self, request):
        if request.url.path.endswith("/admission"):
            body = json.loads(request.content)
            self.calls.append(body)
            action = body["action"]
            code = (
                self.refuse_acquire
                if action == "acquire"
                else self.refuse_renew
                if action == "renew"
                else None
            )
            if code:
                return httpx.Response(code, headers={"Retry-After": "7"})
            return httpx.Response(
                200,
                json={
                    "keyId": body["keyId"],
                    "keyName": "Verified app",
                    "limits": {
                        **({"localOnly": True} if self.local_only else {}),
                        "allowedModels": self.allowed,
                        "maxConcurrentRequests": 1,
                        "requestsPerMinute": 2,
                    },
                    "leaseSeconds": self.lease,
                },
            )
        return super()._handle(request)


@pytest.fixture
def setup(settings):
    authority = Authority()
    signing = b"a" * 32
    token = _issue(signing_key=signing, sub="Original name", aud="client", jti="key-1")
    app = create_app(settings=settings)
    app.state.auth_state = AuthState(signing_key=signing, service_token="service", master_key=None)
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    allowed = FakeDriverClient(name="allowed-driver", model_id="allowed")
    excluded = FakeDriverClient(name="excluded-driver", model_id="excluded")
    app.state.routing = make_routing_table(
        allowed, excluded, slots=[{"model": "alias", "targets": ["allowed", "excluded"]}]
    )
    headers = {"Authorization": "Bearer " + token}
    operator = {
        "Authorization": "Bearer " + _issue(signing_key=signing, sub="operator", aud="operator")
    }
    return app, authority, allowed, excluded, headers, operator


def body(model="alias", **extra):
    return {"model": model, "messages": [{"role": "user", "content": "PRIVATE_PROMPT"}], **extra}


def test_discovery_and_fallback_never_reveal_or_call_excluded_targets(setup):
    app, authority, allowed, excluded, headers, _ = setup
    with TestClient(app) as c:
        models = c.get("/v1/models", headers=headers)
        assert {m["id"] for m in models.json()["data"]} == {"alias", "allowed"}
        assert "excluded" not in models.text
        allowed.generate_error = httpx.ConnectError("offline")
        response = c.post("/v1/chat/completions", json=body(), headers=headers)
        assert response.status_code == 502
        assert len(allowed.calls) == 1 and not excluded.calls
        assert [x["action"] for x in authority.calls] == ["check", "acquire", "renew", "release"]


def test_alias_permission_alone_cannot_authorize_its_target(setup):
    app, authority, allowed, excluded, headers, _ = setup
    authority.allowed = ["alias"]
    with TestClient(app) as c:
        assert c.get("/v1/models", headers=headers).json()["data"] == []
        response = c.post("/v1/chat/completions", json=body(), headers=headers)
        assert response.status_code == 404
        assert "excluded" not in response.text and "allowed-driver" not in response.text
        assert not allowed.calls and not excluded.calls


@pytest.mark.parametrize(
    "path,extra",
    [
        ("/v1/chat/completions", {}),
        ("/v1/chat/completions", {"stream": True}),
        ("/v1/messages", {"max_tokens": 10}),
        ("/v1/messages", {"max_tokens": 10, "stream": True}),
        ("/v1/embeddings", {"input": "PRIVATE_INPUT"}),
    ],
)
def test_all_surfaces_record_verified_identity_and_release(setup, path, extra):
    app, authority, allowed, _, headers, operator = setup
    allowed.usage = Usage(promptTokens=7, completionTokens=3, totalTokens=10)
    if path.endswith("embeddings"):
        allowed.supports_embeddings = True
        install_snapshot(app.state.routing, allowed)
    with TestClient(app) as c:
        payload = (
            {"model": "allowed", **extra}
            if path.endswith("embeddings")
            else body("allowed", **extra)
        )
        if path.endswith("completions"):
            payload["user"] = "FORGED_ID"
        response = c.post(path, json=payload, headers=headers)
        assert response.status_code == 200, response.text
        assert authority.calls[-1]["action"] == "release"
        for _ in range(100):
            usage = c.get("/v1/metrics/clients", headers=operator).json()
            if usage["clients"]:
                break
            time.sleep(0.01)
        row = usage["clients"][0]
        assert row["clientKeyId"] == "key-1" and row["clientKeyName"] == "Verified app"
        assert row["requests"] == row["served"] == row["attempts"] == 1
        assert row["failed"] == row["incompleteUsageRequests"] == 0
        history = c.get("/v1/metrics/requests", headers=operator)
        assert history.json()["requests"][0]["clientKeyId"] == "key-1"
        assert not any(
            secret in history.text + json.dumps(usage)
            for secret in ["PRIVATE_PROMPT", "PRIVATE_INPUT", "FORGED_ID", headers["Authorization"]]
        )


@pytest.mark.parametrize("code", [429, 503])
def test_refusal_precedes_backend_and_keeps_operator_repair_available(setup, code):
    app, authority, allowed, excluded, headers, operator = setup
    authority.refuse_acquire = code
    headers["Origin"] = "http://browser.test"
    with TestClient(app) as c:
        response = c.post("/v1/chat/completions", json=body(), headers=headers)
        assert response.status_code == code and response.headers["Retry-After"] == "7"
        assert response.headers["Access-Control-Allow-Origin"] == "*"
        assert "retry-after" in response.headers["Access-Control-Expose-Headers"].lower()
        assert not allowed.calls and not excluded.calls
        assert c.get("/v1/config", headers=operator).status_code == 200


def test_anthropic_current_authority_revocation_keeps_non_retrying_403(setup):
    app, authority, allowed, _, headers, _ = setup
    authority.refuse_acquire = 401
    with TestClient(app) as c:
        response = c.post("/v1/messages", headers=headers, json=body(max_tokens=10))
        assert response.status_code == 403
        assert not allowed.calls


@pytest.mark.parametrize("surface", ["chat", "stream", "embeddings", "wake"])
async def test_disconnect_cancels_work_and_releases_reservation(setup, surface):
    app, authority, allowed, _, headers, _ = setup
    hang = _Hang()
    allowed.generate_hook = allowed.embed_hook = hang.run
    path = "/v1/embeddings" if surface == "embeddings" else "/v1/chat/completions"
    if surface == "embeddings":
        allowed.supports_embeddings = True
        install_snapshot(app.state.routing, allowed)
    if surface == "stream":

        async def hanging_stream(request):
            await hang.run()
            if False:
                yield

        allowed.stream = hanging_stream
    if surface == "wake":
        from tests.conftest import runtime_facts

        allowed.runtime = "sleeping"
        install_snapshot(
            app.state.routing,
            allowed,
            runtimes=[runtime_facts("sleeping", status="stopped", start_on_demand=True)],
        )

        class Lifecycle:
            async def wake(self, resolution):
                return await hang.run()

        app.state.lifecycle = Lifecycle()

        async def no_refresh():
            return False

        app.state.routing.refresh_if_stale = no_refresh
    payload = (
        {"model": "allowed", "input": "x"}
        if surface == "embeddings"
        else body("allowed", stream=surface == "stream")
    )
    gone = asyncio.Event()
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {
                "type": "http.request",
                "body": json.dumps(payload).encode(),
                "more_body": False,
            }
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    scope = _SCOPE | {
        "path": path,
        "raw_path": path.encode(),
        "headers": [
            (b"authorization", headers["Authorization"].encode()),
            (b"content-type", b"application/json"),
        ],
    }
    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(hang.entered.wait(), 3)
        assert authority.calls[0]["action"] == "acquire"
        gone.set()
        await asyncio.wait_for(task, 2)
        assert hang.cancelled
        assert authority.calls[-1]["action"] == "release"
        assert not any(app.state.routing._inflight.values())
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await app.state.client_key_guard.aclose()
        await app.state.routing.aclose()


async def test_renewal_outage_cancels_generation_before_lease_expiry(setup):
    app, authority, allowed, _, headers, _ = setup
    authority.lease = 0.6
    hang = _Hang()
    allowed.generate_hook = hang.run
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway"
    ) as c:
        task = asyncio.create_task(c.post("/v1/chat/completions", json=body(), headers=headers))
        try:
            await asyncio.wait_for(hang.entered.wait(), 2)
            authority.refuse_renew = 503
            response = await asyncio.wait_for(task, 2)
            assert response.status_code == 503
            assert hang.cancelled and authority.calls[-1]["action"] == "release"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await app.state.client_key_guard.aclose()
            await app.state.routing.aclose()
