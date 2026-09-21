import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.conftest import install_snapshot
from tests.test_admission import body
from tests.test_admission import setup as _setup
from tests.test_local_only import describe_as

setup = _setup


@pytest.mark.parametrize(
    "path,extra",
    [
        ("/v1/chat/completions", {}),
        ("/v1/chat/completions", {"stream": True}),
        ("/v1/messages", {"max_tokens": 10}),
        ("/v1/messages", {"max_tokens": 10, "stream": True}),
        ("/v1/embeddings", {"input": "test"}),
    ],
)
def test_total_deadline_cancels_owned_work_releases_admission_and_never_replays(setup, path, extra):
    app, authority, driver, fallback, headers, _ = setup
    authority.allowed = None
    cancelled = []

    async def hang():
        try:
            await asyncio.sleep(30)
        finally:
            cancelled.append(True)

    driver.generate_hook = driver.embed_hook = hang

    async def hanging_stream(request):
        await hang()
        yield None

    driver.stream = hanging_stream
    driver.supports_embeddings = True
    install_snapshot(app.state.routing, driver, fallback)
    with TestClient(app) as client:
        original = app.state.config_store.get
        app.state.config_store.get = lambda key: (
            0.08 if key == "requestTimeoutSeconds" else original(key)
        )
        before = time.perf_counter()
        response = client.post(
            path,
            json={"model": "allowed", **extra} if path.endswith("embeddings") else body(**extra),
            headers=headers,
        )
        assert time.perf_counter() - before < 2
        assert "total request deadline" in response.text, response.text
        assert response.status_code == (200 if extra.get("stream") else 504)
        assert cancelled and not fallback.calls
        assert authority.calls[-1]["action"] == "release"
        assert response.headers["x-request-id"]


def test_policy_and_request_id_survive_profile_default_replacement(setup):
    app, authority, driver, fallback, headers, _ = setup
    authority.local_only = True
    describe_as(driver, "local")
    describe_as(fallback, "external")
    install_snapshot(app.state.routing, driver, fallback)

    class Profiles:
        async def get(self, path):
            return {"temperature": 0.2}

    with TestClient(app) as client:
        app.state.profile_defaults = Profiles()
        response = client.post("/v1/chat/completions", json=body(), headers=headers)
        assert response.status_code == 200, response.text
        assert driver.calls[0].localOnly
        assert str(driver.calls[0].requestId) == response.headers["x-request-id"]


@pytest.mark.parametrize("phase", ["profile", "wake"])
def test_preparation_and_wake_spend_the_same_deadline(setup, phase):
    from tests.conftest import runtime_facts

    app, authority, driver, fallback, headers, _ = setup
    cancelled = []

    class Slow:
        async def get(self, path):
            try:
                await asyncio.sleep(30)
            finally:
                cancelled.append(True)

        wake = get

    if phase == "wake":
        driver.runtime = "asleep"
        install_snapshot(
            app.state.routing,
            driver,
            runtimes=[runtime_facts("asleep", status="stopped", start_on_demand=True)],
        )
    with TestClient(app) as client:
        if phase == "profile":
            app.state.profile_defaults = Slow()
        else:
            app.state.lifecycle = Slow()
        original = app.state.config_store.get
        app.state.config_store.get = lambda key: (
            0.08 if key == "requestTimeoutSeconds" else original(key)
        )
        response = client.post("/v1/chat/completions", json=body(), headers=headers)
        assert response.status_code == 504, response.text
        assert cancelled and not driver.calls and not fallback.calls
        assert authority.calls[-1]["action"] == "release"


@pytest.mark.parametrize("missing", ["settings", "tools", "changed"])
def test_capability_ineligible_fallback_receives_no_application_content(setup, missing):
    app, authority, driver, fallback, headers, _ = setup
    authority.allowed = None
    driver.supports_tools = fallback.supports_tools = True
    install_snapshot(app.state.routing, driver, fallback)
    info = fallback.describe()
    if missing == "tools":
        info.capabilities.toolCalling = False
    else:
        info.capabilities.supportedSettings = None
    fallback.describe = lambda: info
    if missing != "changed":
        install_snapshot(app.state.routing, driver, fallback)
    driver.generate_error = httpx.ConnectError("offline")
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=body(
                response_format={"type": "json_object"},
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "act", "parameters": {"type": "object"}},
                    }
                ],
            ),
            headers=headers,
        )
        assert response.status_code == 502
        assert len(driver.calls) == 1 and not fallback.calls
