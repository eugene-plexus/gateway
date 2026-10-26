"""One fair queue in the gateway (row 11, LD1).

A request over its key's concurrency limit waits instead of a 429; requests
for a busy runtime wait in the gateway by band, then person in turns, then
arrival; a refusal names its cause in the door's own words; and the wait is
kept on every request row. Through the real middleware, routing and metrics
store, with the authority scripted as the root counts leases.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway.app import create_app

from .conftest import FakeDriverClient, make_routing_table
from .test_client_keys import FakeAgent

MODEL = "qwen3-8b"


class Root(FakeAgent):
    """The admission authority as the root answers it: leases counted per key,
    a 429 that says which limit, and a verified person for each person token."""

    def __init__(self, *, limit: int = 2, per_minute: int = 600, max_queued: int = 8) -> None:
        super().__init__()
        self.limit = limit
        self.per_minute = per_minute
        self.max_queued = max_queued
        self.band: str | None = None
        self.person_bands: dict[str, str] = {}
        self.active: dict[str, str] = {}
        self.calls: list[dict[str, Any]] = []
        self.most = 0
        #: Leases another gateway holds, by key: counted, never released here.
        self.elsewhere: dict[str, int] = {}
        self.lock = threading.Lock()

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/auth/client-keys/admission":
            return super()._handle(request)
        body = json.loads(request.content)
        with self.lock:
            self.calls.append(body)
            key, action, rid = body["keyId"], body["action"], body["requestId"]
            limits: dict[str, Any] = {
                "allowedModels": None,
                "maxConcurrentRequests": self.limit,
                "requestsPerMinute": self.per_minute,
            }
            if self.max_queued != 8:  # as the root writes it: only when not the default
                limits["maxQueuedRequests"] = self.max_queued
            if self.band is not None:
                limits["band"] = self.band
            if action == "acquire" and rid not in self.active:
                held = sum(1 for k in self.active.values() if k == key) + self.elsewhere.get(key, 0)
                if held >= self.limit:
                    return httpx.Response(
                        429,
                        json={"detail": "This key's shared concurrent request limit is reached."},
                        headers={"Retry-After": "1", "X-Eugene-Refusal-Reason": "concurrency"},
                    )
                self.active[rid] = key
                self.most = max(self.most, held + 1)
            elif action == "release":
                self.active.pop(rid, None)
            answer: dict[str, Any] = {"keyId": key, "keyName": "Burst", "limits": limits}
            if action in ("acquire", "renew"):
                answer["leaseSeconds"] = 30
            token = body.get("personToken")
            if token:
                answer["personId"] = f"person-{token}"
                answer["appClientId"] = "workbench"
                if token in self.person_bands:
                    answer["queueBand"] = self.person_bands[token]
            return httpx.Response(200, json=answer)


class Holding(FakeDriverClient):
    """A driver whose engine takes `seconds` for each request and counts how
    many it holds at once."""

    def __init__(self, *, seconds: float = 0.3, **kwargs: Any) -> None:
        kwargs.setdefault("name", "llama-driver")
        kwargs.setdefault("model_id", MODEL)
        super().__init__(**kwargs)
        self.seconds = seconds
        self.now = 0
        self.most = 0
        self.order: list[str] = []
        self.generate_hook = self._hold

    async def _hold(self) -> None:
        self.now += 1
        self.most = max(self.most, self.now)
        self.order.append(self.calls[-1].messages[-1].content if self.calls else "")
        try:
            await asyncio.sleep(self.seconds)
        finally:
            self.now -= 1


@pytest.fixture
def gateway(settings, install):  # type: ignore[no-untyped-def]
    def build(root: Root, driver: FakeDriverClient, **table: Any):  # type: ignore[no-untyped-def]
        app = create_app(settings=settings)
        app.state.auth_state = install.auth_state()
        app.state.client_key_guard = root.as_guard(ttl_seconds=0)
        app.state.routing = make_routing_table(driver, **table)
        return app

    return build


def _key(install, jti: str = "key-1") -> dict[str, str]:  # type: ignore[no-untyped-def]
    return {"Authorization": "Bearer " + install.client_key(name="Burst", jti=jti)}


def _chat(text: str = "hello", **extra: Any) -> dict[str, Any]:
    return {"model": MODEL, "messages": [{"role": "user", "content": text}], **extra}


def _rows(c: TestClient, install, count: int) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    operator = {"Authorization": "Bearer " + install.session(sub="operator")}
    rows: list[dict[str, Any]] = []
    for _ in range(300):
        rows = c.get("/v1/metrics/requests?limit=500", headers=operator).json()["requests"]
        if len(rows) >= count:
            break
        time.sleep(0.02)
    return rows


# --- 11b: a key over its limit waits ---------------------------------------------


def test_over_the_concurrency_limit_waits_instead_of_429(gateway, install):
    root, driver = Root(limit=2), Holding(seconds=0.3)
    app = gateway(root, driver)
    headers = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(3) as pool:
        answers = list(
            pool.map(
                lambda n: c.post("/v1/chat/completions", json=_chat(f"t{n}"), headers=headers),
                range(3),
            )
        )
        assert [a.status_code for a in answers] == [200, 200, 200], [a.text for a in answers]
        assert driver.most <= 2 and root.most <= 2
        rows = _rows(c, install, 3)
    waited = sorted(r["queuedMs"] or 0 for r in rows)
    assert waited[-1] >= 250, waited
    assert all(r["band"] == "interactive" and r["outcome"] == "served" for r in rows)
    assert not root.active, "every lease was given back"


def test_a_key_at_its_limit_does_not_block_another_key(gateway, install):
    root, driver = Root(limit=1), Holding(seconds=0.4)
    app = gateway(root, driver)
    busy, other = _key(install, "key-1"), _key(install, "key-2")
    with TestClient(app) as c, ThreadPoolExecutor(4) as pool:
        held = [
            pool.submit(c.post, "/v1/chat/completions", json=_chat(f"a{n}"), headers=busy)
            for n in range(3)
        ]
        time.sleep(0.1)
        began = time.perf_counter()
        answer = c.post("/v1/chat/completions", json=_chat("b"), headers=other)
        took = time.perf_counter() - began
        assert answer.status_code == 200
        assert took < 0.7, f"key-2 waited behind key-1's line: {took:.2f} s"
        assert [f.result().status_code for f in held] == [200, 200, 200]


def test_a_full_line_for_a_key_is_refused_with_its_cause(gateway, install):
    root, driver = Root(limit=1, max_queued=1), Holding(seconds=0.5)
    app = gateway(root, driver)
    headers = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(3) as pool:
        first = pool.submit(c.post, "/v1/chat/completions", json=_chat("one"), headers=headers)
        time.sleep(0.1)
        second = pool.submit(c.post, "/v1/chat/completions", json=_chat("two"), headers=headers)
        time.sleep(0.1)
        third = c.post("/v1/chat/completions", json=_chat("three"), headers=headers)
        assert third.status_code == 429
        assert third.headers["Retry-After"] == "5"
        assert third.headers["X-Eugene-Refusal-Reason"] == "key_queue_full"
        error = third.json()["error"]
        assert error["code"] == "key_queue_full"
        assert error["message"] == (
            "This key has 1 running and 1 waiting for you, its most. Retry when one finishes."
        )
        assert first.result().status_code == second.result().status_code == 200
        rows = _rows(c, install, 3)
        refused = [r for r in rows if r["outcome"] == "refused"]
        assert [r["refusal"] for r in refused] == ["key_queue_full"]
        operator = {"Authorization": "Bearer " + install.session(sub="operator")}
        later = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()
        summary = c.get("/v1/metrics", params={"until": later}, headers=operator).json()
        assert summary["refused"] == {"key_queue_full": 1}
        assert sum(g["requests"] for g in summary["groups"]) == 2, (rows, summary)


def test_zero_queued_refuses_at_once_as_before(gateway, install):
    root, driver = Root(limit=1, max_queued=0), Holding(seconds=0.4)
    app = gateway(root, driver)
    headers = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(2) as pool:
        first = pool.submit(c.post, "/v1/chat/completions", json=_chat("one"), headers=headers)
        time.sleep(0.1)
        began = time.perf_counter()
        second = c.post("/v1/chat/completions", json=_chat("two"), headers=headers)
        assert second.status_code == 429 and time.perf_counter() - began < 0.3
        assert second.json()["error"]["code"] == "key_queue_full"
        assert first.result().status_code == 200


def test_the_root_refusing_for_another_gateway_keeps_the_place(gateway, install):
    """Two gateways share one key: the root counts the other's lease, so this
    gateway's waiter asks again, and leaves when that lease ends."""
    root, driver = Root(limit=1), Holding(seconds=0.05)
    root.elsewhere["key-1"] = 1
    app = gateway(root, driver)
    headers = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(1) as pool:
        waiting = pool.submit(c.post, "/v1/chat/completions", json=_chat("me"), headers=headers)
        time.sleep(1.3)
        assert not waiting.done(), "a root 429 for concurrency is not the request's answer"
        with root.lock:
            root.elsewhere.clear()
        answer = waiting.result(timeout=5)
        assert answer.status_code == 200
    asks = [call for call in root.calls if call["action"] == "acquire"]
    assert len(asks) >= 2, "it asked again"


def test_a_rate_refusal_says_the_keys_minute(gateway, install):
    root, driver = Root(limit=4, per_minute=1), Holding(seconds=0.01)
    app = gateway(root, driver)

    def rate(request: httpx.Request) -> httpx.Response:
        admission = request.url.path.endswith("/admission")
        if admission and json.loads(request.content)["action"] == "acquire":
            return httpx.Response(
                429,
                json={"detail": "This key's shared rolling-minute request limit is reached."},
                headers={"Retry-After": "12", "X-Eugene-Refusal-Reason": "rate"},
            )
        return root._handle(request)

    app.state.client_key_guard._client = httpx.AsyncClient(transport=httpx.MockTransport(rate))
    with TestClient(app) as c:
        answer = c.post("/v1/chat/completions", json=_chat(), headers=_key(install))
        assert answer.status_code == 429 and answer.headers["Retry-After"] == "12"
        assert answer.json()["error"]["code"] == "rate_limited"
        assert answer.json()["error"]["message"] == (
            "This key's 1 requests a minute are used. Retry in 12 s."
        )
        assert not driver.calls


@pytest.mark.parametrize(
    "path,extra,kind",
    [
        ("/v1/messages", {"max_tokens": 10}, "rate_limit_error"),
        ("/v1/messages", {"max_tokens": 10, "stream": True}, "rate_limit_error"),
    ],
)
def test_the_anthropic_door_words_a_full_line_in_its_own_types(gateway, install, path, extra, kind):
    root, driver = Root(limit=1, max_queued=0), Holding(seconds=0.4)
    app = gateway(root, driver)
    headers = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(1) as pool:
        first = pool.submit(c.post, "/v1/chat/completions", json=_chat("one"), headers=headers)
        time.sleep(0.1)
        answer = c.post(path, json=_chat("two", **extra), headers=headers)
        text = answer.text
        if extra.get("stream"):
            # The head is sent before the turn takes its place: the error frame
            # carries the type and the words, and the retry time in them.
            assert "event: error" in text and kind in text
            assert "its most. Retry when one finishes. Retry after 5 s." in text
        else:
            assert answer.status_code == 429
            assert answer.json()["error"]["type"] == kind
        assert first.result().status_code == 200


def test_a_responses_stream_refused_in_line_fails_with_its_retry_code(gateway, install):
    root, driver = Root(limit=1, max_queued=0), Holding(seconds=0.4)
    app = gateway(root, driver)
    headers = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(1) as pool:
        first = pool.submit(c.post, "/v1/chat/completions", json=_chat("one"), headers=headers)
        time.sleep(0.1)
        answer = c.post(
            "/v1/responses", json={"model": MODEL, "input": "two", "stream": True}, headers=headers
        )
        frames = [line for line in answer.text.splitlines() if line.startswith("data: ")]
        failed = [json.loads(f[6:]) for f in frames if '"response.failed"' in f]
        assert failed, answer.text
        error = failed[-1]["response"]["error"]
        assert error["code"] == "rate_limit_exceeded", "Codex retries this code"
        assert "its most" in error["message"] and "Retry after 5 s." in error["message"]
        assert first.result().status_code == 200


def test_a_door_without_a_turn_waits_for_its_key_too(gateway, install):
    """Embeddings name a model and run no chat turn: they wait in the same line,
    for the key's turn only."""
    root = Root(limit=1)
    driver = Holding(seconds=0.3, supports_embeddings=True)

    async def hold() -> None:
        driver.now += 1
        driver.most = max(driver.most, driver.now)
        try:
            await asyncio.sleep(0.3)
        finally:
            driver.now -= 1

    driver.embed_hook = hold
    app = gateway(root, driver)
    headers = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(2) as pool:
        answers = list(
            pool.map(
                lambda n: c.post(
                    "/v1/embeddings", json={"model": MODEL, "input": f"x{n}"}, headers=headers
                ),
                range(2),
            )
        )
        assert [a.status_code for a in answers] == [200, 200], [a.text for a in answers]
        assert driver.most == 1
        rows = _rows(c, install, 2)
    assert max(r["queuedMs"] or 0 for r in rows) >= 200


async def test_cancel_while_waiting_leaks_nothing(gateway, install):
    """A client that hangs up while it waits frees its place at once, and
    nothing was acquired for it, so nothing is released."""
    from eugene_plexus_gateway import fair_queue

    root, driver = Root(limit=1), Holding(seconds=0.5)
    app = gateway(root, driver)
    headers = _key(install)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
            first = asyncio.create_task(
                client.post("/v1/chat/completions", json=_chat("one"), headers=headers)
            )
            await asyncio.sleep(0.1)
            gone = asyncio.create_task(
                client.post("/v1/chat/completions", json=_chat("gone"), headers=headers)
            )
            await asyncio.sleep(0.1)
            queue = fair_queue.of(app.state.routing)
            assert queue.waiting() == 1
            gone.cancel()
            await asyncio.gather(gone, return_exceptions=True)
            assert queue.waiting() == 0, "the place was freed at once"
            assert (await first).status_code == 200
            assert queue.key_running("key-1") == 0
            later = await client.post("/v1/chat/completions", json=_chat("later"), headers=headers)
            assert later.status_code == 200
    finally:
        await app.state.client_key_guard.aclose()
        await app.state.routing.aclose()
    acquired = [c["requestId"] for c in root.calls if c["action"] == "acquire"]
    assert len(acquired) == 2, "the request that hung up never acquired"
    assert not root.active


def test_zero_queued_is_refused_at_once_when_another_gateway_holds_the_key(gateway, install):
    """A key that may wait for nothing (`maxQueuedRequests` 0) is refused at once
    when the root counts its leases on another gateway, as before the queue."""
    root, driver = Root(limit=1, max_queued=0), Holding(seconds=0.05)
    root.elsewhere["key-1"] = 1
    app = gateway(root, driver)
    with TestClient(app) as c:
        began = time.perf_counter()
        answer = c.post("/v1/chat/completions", json=_chat(), headers=_key(install))
        assert answer.status_code == 429 and time.perf_counter() - began < 0.5
        assert answer.json()["error"]["code"] == "key_queue_full"
        assert not driver.calls
