"""R8: defaults cross the real route/cascade seam, not just a merge helper."""

import asyncio
from dataclasses import replace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import GenerateRequest
from eugene_plexus_gateway.profiles import ProfileDefaults

from .conftest import FakeDriverClient, make_routing_table, runtime_facts


class Library:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, Any]] = {
            "/models/a.gguf": {"maxTokens": 91, "temperature": 0, "topP": 0.25},
            "/models/b.gguf": {"maxTokens": 73, "temperature": 1.2, "topP": 0.8},
        }
        self.calls: list[httpx.Request] = []
        self.status = 200
        self.invalid = False

    async def request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        assert request.headers.get("authorization") == "Bearer gateway-service"
        if self.status != 200:
            return httpx.Response(self.status)
        if request.url.path.endswith("/v1/models"):
            path = request.url.params["path"]
            # Opaque ids differ from aliases and are never derived by gateway.
            return httpx.Response(
                200,
                json={
                    "models": [{"id": str(list(self.values).index(path))}]
                    if path in self.values
                    else []
                },
            )
        index = int(request.url.path.split("/")[-2])
        path = list(self.values)[index]
        values = self.values[path]
        return httpx.Response(
            200,
            json={
                "profiles": [
                    {
                        "id": "not-default",
                        "name": "launch",
                        "engine": "llama_cpp",
                        "default": False,
                        "temperature": 1.9,
                    },
                    {
                        "id": "chosen",
                        "name": "chosen",
                        "engine": "llama_cpp",
                        "default": True,
                        **values,
                        **({"maxTokens": -1} if self.invalid else {}),
                    },
                ]
                if values
                else []
            },
        )


@pytest.fixture
def library() -> Library:
    return Library()


def resolver(
    library: Library, config: dict[str, Any] | None = None, **kwargs: Any
) -> ProfileDefaults:
    return ProfileDefaults(
        "http://agent.invalid:8079",
        "gateway-service",
        (config or {}).get,
        transport=httpx.MockTransport(library.request),
        **kwargs,
    )


async def test_cache_outage_expiry_deletion_and_recovery(library: Library) -> None:
    now = [0.0]
    profiles = resolver(library, clock=lambda: now[0])
    try:
        assert (await profiles.get("/models/a.gguf"))["temperature"] == 0
        library.values["/models/a.gguf"]["temperature"] = 0.9
        now[0] = 29
        assert (await profiles.get("/models/a.gguf"))["temperature"] == 0
        assert len(library.calls) == 2
        now[0] = 31
        assert (await profiles.get("/models/a.gguf"))["temperature"] == 0.9
        library.status = 503
        now[0] = 62
        assert (await profiles.get("/models/a.gguf"))["temperature"] == 0.9
        calls = len(library.calls)
        now[0] = 63
        assert await profiles.get("/models/a.gguf")
        assert len(library.calls) == calls
        now[0] = 362
        assert await profiles.get("/models/a.gguf") == {}
        library.status = 200
        now[0] = 368
        library.values["/models/a.gguf"] = {"maxTokens": 999}
        assert await profiles.get("/models/a.gguf") == {"maxTokens": 999}
        library.values["/models/a.gguf"] = {}
        now[0] = 399
        assert await profiles.get("/models/a.gguf") == {}
        assert await profiles.get("/models/missing.gguf") == {}
    finally:
        await profiles.aclose()


async def test_zero_ttls_and_invalid_response_do_not_keep_stale(library: Library) -> None:
    config = {"profileCacheSeconds": 0, "profileMaxStaleSeconds": 0}
    profiles = resolver(library, config)
    try:
        assert (await profiles.get("/models/a.gguf"))["maxTokens"] == 91
        library.invalid = True
        assert await profiles.get("/models/a.gguf") == {}
    finally:
        await profiles.aclose()


async def test_single_flight_and_cancelled_waiter(library: Library) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    original = library.request

    async def slow(request: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return await original(request)

    library.request = slow  # type: ignore[method-assign]
    profiles = resolver(library)
    try:
        first = asyncio.create_task(profiles.get("/models/a.gguf"))
        await entered.wait()
        others = [asyncio.create_task(profiles.get("/models/a.gguf")) for _ in range(10)]
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        assert all(value["maxTokens"] == 91 for value in await asyncio.gather(*others))
        assert len(library.calls) == 2
    finally:
        await profiles.aclose()


async def test_cache_is_bounded_and_cloud_does_no_lookup(library: Library) -> None:
    profiles = resolver(library)
    try:
        assert await profiles.get(None) == {}
        assert not library.calls
        for i in range(260):
            assert await profiles.get(f"/missing/{i}.gguf") == {}
        assert len(profiles._cache) <= 256
    finally:
        await profiles.aclose()


def setup_routes(app: FastAPI, library: Library, *drivers: FakeDriverClient) -> ProfileDefaults:
    facts = [
        replace(
            runtime_facts(d.runtime or "run", alias=d.model_id, node=d.node),
            spec={"modelPath": f"/models/{'a' if i == 0 else 'b'}.gguf"},
        )
        for i, d in enumerate(drivers)
        if d.runtime
    ]
    table = make_routing_table(*drivers, runtimes=facts)
    app.state.routing = table
    profiles = resolver(library)
    app.state.profile_defaults = profiles
    return profiles


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/messages"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_both_doors_preserve_caller_values_and_apply_profile(
    app: FastAPI, library: Library, endpoint: str, stream: bool, explicit: bool
) -> None:
    driver = FakeDriverClient(
        name="driver", base_url="http://driver.invalid", model_id="alias", runtime="run"
    )
    profiles = setup_routes(app, library, driver)
    body: dict[str, Any] = {
        "model": "alias",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
    }
    if endpoint == "/v1/messages":
        body["max_tokens"] = 41  # Anthropic requires it; never fill over it.
    if explicit:
        library.values["/models/a.gguf"]["temperature"] = 0.8
        body.update(max_tokens=17, temperature=0, top_p=0)
        if endpoint.endswith("completions"):
            body["seed"] = 0
    with TestClient(app) as client:
        response = client.post(endpoint, json=body)
        assert response.status_code == 200, response.text
        got = driver.calls[-1]
        assert got.maxTokens == (17 if explicit else 41 if endpoint == "/v1/messages" else 91)
        assert got.temperature == 0
        assert got.topP == (0 if explicit else 0.25)
        if explicit:
            assert not library.calls
            if endpoint.endswith("completions"):
                assert got.seed == 0
        client.portal.call(profiles.aclose)


@pytest.mark.parametrize("stream", [False, True])
def test_fallback_uses_its_own_profile(app: FastAPI, library: Library, stream: bool) -> None:
    first = FakeDriverClient(
        name="d1", base_url="http://one.invalid", model_id="first", runtime="one"
    )
    second = FakeDriverClient(
        name="d2", base_url="http://two.invalid", model_id="second", runtime="two"
    )
    first.generate_error = httpx.ConnectError("offline")
    first.stream_error_after = 0
    first.stream_error = httpx.ConnectError("offline")
    profiles = setup_routes(app, library, first, second)
    app.state.routing._slots = lambda: [{"model": "alias", "targets": ["first", "second"]}]
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "alias",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
            },
        )
        assert response.status_code == 200, response.text
        assert second.calls[-1].maxTokens == 73
        assert second.calls[-1].temperature == 1.2
        assert second.calls[-1].topP == 0.8
        client.portal.call(profiles.aclose)


def test_no_profile_retains_gateway_defaults(app: FastAPI, library: Library) -> None:
    driver = FakeDriverClient(
        name="d", base_url="http://d.invalid", model_id="alias", runtime="run"
    )
    library.values.clear()
    profiles = setup_routes(app, library, driver)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "alias", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert r.status_code == 200
        assert driver.calls[-1].maxTokens == 2048
        assert driver.calls[-1].temperature == 0.7
        assert driver.calls[-1].topP is None
        client.portal.call(profiles.aclose)


def test_replica_names_and_aliases_do_not_cross_model_paths(app: FastAPI, library: Library) -> None:
    a = FakeDriverClient(name="same", node="a", model_id="alias", runtime="same")
    b = FakeDriverClient(name="same", node="b", model_id="alias", runtime="same")
    profiles = setup_routes(app, library, a, b)
    with TestClient(app) as client:
        for _ in range(2):
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "alias",
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.6,
                },
            )
            assert r.status_code == 200
        assert (a.calls[-1].maxTokens, b.calls[-1].maxTokens) == (91, 73)
        assert a.calls[-1].temperature == b.calls[-1].temperature == 0.6
        client.portal.call(profiles.aclose)


def test_hosted_backend_does_not_inherit_a_library_model(app: FastAPI, library: Library) -> None:
    driver = FakeDriverClient(name="cloud", model_id="alias")
    profiles = setup_routes(app, library, driver)
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "alias", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert r.status_code == 200
        assert driver.calls[-1].maxTokens == 2048
        assert not library.calls
        client.portal.call(profiles.aclose)


@pytest.mark.parametrize("stream", [False, True])
async def test_profile_wait_reserves_runtime_and_cancellation_releases_it(stream: bool) -> None:
    driver = FakeDriverClient(name="driver", runtime="run", model_id="alias")
    table = make_routing_table(driver, runtimes=[runtime_facts("run", alias="alias")])
    selected = table.pick(table.resolve("alias"))
    assert selected is not None
    entered = asyncio.Event()

    async def prepare(candidate: Any, request: GenerateRequest) -> GenerateRequest:
        entered.set()
        await asyncio.Event().wait()
        return request

    selected.prepare_request = prepare
    request = GenerateRequest(messages=[{"role": "user", "content": "hello"}])
    iterator = selected.stream(request) if stream else None
    task = asyncio.create_task(iterator.__anext__() if iterator else selected.generate(request))
    await entered.wait()
    assert table.inflight((None, "driver")) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert table.inflight((None, "driver")) == 0
    assert not driver.calls
    if iterator:
        await iterator.aclose()
    await table.aclose()
