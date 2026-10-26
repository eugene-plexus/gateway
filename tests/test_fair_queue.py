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
        #: Whether the key is local-only, as the operator set it now.
        self.local_only = False
        #: Seconds each acquire takes to answer, so several are in flight at once.
        self.acquire_delay = 0.0
        #: A band for one key, beside `band` for every key.
        self.key_bands: dict[str, str] = {}
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
            if self.local_only:
                limits["localOnly"] = True
            band = self.key_bands.get(key, self.band)
            if band is not None:
                limits["band"] = band
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

    def as_guard(self, **kwargs: Any) -> Any:
        guard = super().as_guard(**kwargs)

        async def handle(request: httpx.Request) -> httpx.Response:
            admission = request.url.path.endswith("/admission")
            if (
                self.acquire_delay
                and admission
                and json.loads(request.content)["action"] == "acquire"
            ):
                await asyncio.sleep(self.acquire_delay)
            return self._handle(request)

        guard._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        return guard


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
    asks = [call for call in root.calls if call["action"] == "acquire"]
    assert len(asks) == 3, "the gateway held the third back itself, asking once for each"


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


# --- 11c: a busy runtime is waited for in the gateway, fairly -----------------------


class Engine(FakeDriverClient):
    """llama-server as the gateway meets it: `slots` requests at a time, the rest
    queued inside in arrival order, each taking `seconds`."""

    def __init__(self, *, slots: int = 2, seconds: float = 0.2, **kwargs: Any) -> None:
        kwargs.setdefault("name", "llama-driver")
        kwargs.setdefault("model_id", MODEL)
        kwargs.setdefault("runtime", "llama")
        super().__init__(**kwargs)
        self.slots = asyncio.Semaphore(slots) if slots else None
        self.seconds = seconds
        self.now = 0
        self.most = 0
        self.order: list[str] = []
        self.generate_hook = self._serve

    async def _serve(self) -> None:
        text = self.calls[-1].messages[-1].content if self.calls else ""
        assert self.slots is not None
        async with self.slots:
            self.now += 1
            self.most = max(self.most, self.now)
            self.order.append(str(text))
            try:
                await asyncio.sleep(self.seconds)
            finally:
                self.now -= 1


def _slots(count: int, **extra: Any) -> list[Any]:
    from .conftest import runtime_facts

    return [runtime_facts("llama", parallel_slots=count, **extra)]


def test_one_persons_burst_does_not_hold_back_another(gateway, install):
    """Persons A and B share one app key (as Workbench's people do). A sends 51
    requests at once to a runtime of two slots; B sends one 50 ms later. B waits
    no more than one service time and 100 ms, where it waited behind all of A's
    in the engine's own queue before."""
    root, engine = Root(limit=64, max_queued=64), Engine(slots=2, seconds=0.2)
    app = gateway(root, engine, runtimes=_slots(2))
    key = _key(install)
    a = {**key, "X-Eugene-Person-Token": "a"}
    b = {**key, "X-Eugene-Person-Token": "b"}
    with TestClient(app) as c, ThreadPoolExecutor(52) as pool:
        burst = [
            pool.submit(c.post, "/v1/chat/completions", json=_chat(f"a{n}"), headers=a)
            for n in range(51)
        ]
        time.sleep(0.05)
        began = time.perf_counter()
        answer = c.post("/v1/chat/completions", json=_chat("b"), headers=b)
        took = time.perf_counter() - began
        assert answer.status_code == 200, answer.text
        assert [f.result().status_code for f in burst] == [200] * 51
        rows = _rows(c, install, 52)
    mine = next(r for r in rows if r["personId"] == "person-b")
    assert (mine["queuedMs"] or 0) <= 300, mine["queuedMs"]
    assert took <= 0.2 + 0.3 + 0.3, f"B took {took:.2f} s behind A's burst"
    assert engine.most <= 2, "the runtime never saw more than its slots"


def test_background_waits_for_interactive(gateway, install):
    root, engine = Root(limit=8), Engine(slots=1, seconds=0.3)
    app = gateway(root, engine, runtimes=_slots(1))
    key = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(3) as pool:
        first = pool.submit(c.post, "/v1/chat/completions", json=_chat("first"), headers=key)
        time.sleep(0.1)
        later = {**key, "X-Eugene-Band": "background"}
        background = pool.submit(
            c.post, "/v1/chat/completions", json=_chat("background"), headers=later
        )
        time.sleep(0.1)
        interactive = pool.submit(
            c.post, "/v1/chat/completions", json=_chat("interactive"), headers=key
        )
        assert [f.result().status_code for f in (first, background, interactive)] == [200] * 3
        rows = _rows(c, install, 3)
    assert engine.order == ["first", "interactive", "background"], engine.order
    assert sorted(r["band"] for r in rows) == ["background", "interactive", "interactive"]


def test_a_header_cannot_raise_the_band(gateway, install):
    """key-2's band is background: its `X-Eugene-Band: interactive` is ignored,
    and an interactive request that arrives after it leaves first."""
    root, engine = Root(limit=8), Engine(slots=1, seconds=0.3)
    root.key_bands["key-2"] = "background"
    app = gateway(root, engine, runtimes=_slots(1))
    one, two = _key(install, "key-1"), _key(install, "key-2")
    with TestClient(app) as c, ThreadPoolExecutor(3) as pool:
        first = pool.submit(c.post, "/v1/chat/completions", json=_chat("first"), headers=one)
        time.sleep(0.1)
        asks = {**two, "X-Eugene-Band": "interactive"}
        raised = pool.submit(c.post, "/v1/chat/completions", json=_chat("raised"), headers=asks)
        time.sleep(0.1)
        plain = pool.submit(c.post, "/v1/chat/completions", json=_chat("plain"), headers=one)
        assert [f.result().status_code for f in (first, raised, plain)] == [200] * 3
        rows = _rows(c, install, 3)
    assert engine.order == ["first", "plain", "raised"], engine.order
    assert {r["clientKeyId"]: r["band"] for r in rows if r["clientKeyId"] == "key-2"} == {
        "key-2": "background"
    }


def test_a_persons_groups_set_the_band_and_the_key_does_not_raise_it(gateway, install):
    """The verified person's band from their groups (the root's `queueBand`)
    replaces the key's: a person whose groups say background waits behind an
    interactive person on the same app key."""
    root, engine = Root(limit=8), Engine(slots=1, seconds=0.3)
    root.person_bands["slow"] = "background"
    app = gateway(root, engine, runtimes=_slots(1))
    key = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(3) as pool:
        first = pool.submit(
            c.post,
            "/v1/chat/completions",
            json=_chat("first"),
            headers={**key, "X-Eugene-Person-Token": "fast"},
        )
        time.sleep(0.1)
        slow = pool.submit(
            c.post,
            "/v1/chat/completions",
            json=_chat("slow"),
            headers={**key, "X-Eugene-Person-Token": "slow", "X-Eugene-Band": "interactive"},
        )
        time.sleep(0.1)
        fast = pool.submit(
            c.post,
            "/v1/chat/completions",
            json=_chat("fast"),
            headers={**key, "X-Eugene-Person-Token": "fast"},
        )
        assert [f.result().status_code for f in (first, slow, fast)] == [200] * 3
    assert engine.order == ["first", "fast", "slow"], engine.order


def test_a_slot_is_held_through_the_acquire_round_trip(gateway, install):
    """The slot is reserved when a turn leaves the line, before its lease is
    asked for, so two turns never leave on one slot."""
    root, holding = Root(limit=8), Holding(seconds=0.2, runtime="llama")
    app = gateway(root, holding, runtimes=_slots(1))
    key = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(3) as pool:
        answers = list(
            pool.map(
                lambda n: c.post("/v1/chat/completions", json=_chat(f"r{n}"), headers=key),
                range(3),
            )
        )
    assert [a.status_code for a in answers] == [200] * 3
    assert holding.most == 1


# --- table-level: no overtaking, moves, refusals, cancel and shutdown -------------


def _pooled(fake: FakeDriverClient, **table: Any):  # type: ignore[no-untyped-def]
    from .conftest import runtime_facts

    fake.usage = None
    runtimes = [runtime_facts("rt", parallel_slots=4, context_pool=10_000)]
    return make_routing_table(fake, runtimes=runtimes, **table)


def _turn(text: str, max_tokens: int = 100) -> Any:
    from eugene_plexus_gateway._generated.driver_models import GenerateRequest

    return GenerateRequest(messages=[{"role": "user", "content": text}], maxTokens=max_tokens)


async def test_a_big_turn_is_not_overtaken_on_its_runtime():
    """A turn too big for what is left of the pool claims the runtime: a small
    turn that arrives after it and would fit does not go first, or a steady
    stream of small turns would starve it."""
    fake = FakeDriverClient(name="rt-driver", model_id=MODEL, runtime="rt")
    table = _pooled(fake)
    table._ledger.take((None, "rt"), 5_000)  # what runs there now
    big = asyncio.ensure_future(table.pick(table.resolve(MODEL)).generate(_turn("x" * 20_000)))
    await asyncio.sleep(0.05)
    small = asyncio.ensure_future(table.pick(table.resolve(MODEL)).generate(_turn("hi")))
    await asyncio.sleep(0.1)
    assert not fake.calls, "the small turn overtook the big one on its runtime"
    table._ledger.give((None, "rt"), 5_000)
    await asyncio.wait_for(asyncio.gather(big, small), 2)
    assert [len(c.messages[-1].content) for c in fake.calls] == [20_000, 2]


async def test_ttl_moves_to_the_next_tier_then_refuses():
    from eugene_plexus_gateway import admission, fair_queue

    from .conftest import runtime_facts

    local = FakeDriverClient(name="local-driver", model_id="local-model", runtime="rt")
    cloud = FakeDriverClient(name="cloud-driver", model_id="cloud-model")
    gate = asyncio.Event()

    async def hold() -> None:
        await gate.wait()

    local.generate_hook = hold
    table = make_routing_table(
        local,
        cloud,
        runtimes=[runtime_facts("rt", parallel_slots=1)],
        slots=[{"model": "slotty", "targets": ["local-model", "cloud-model"]}],
    )
    running = asyncio.ensure_future(table.pick(table.resolve("local-model")).generate(_turn("a")))
    await asyncio.sleep(0.05)
    moving = table.pick(table.resolve("slotty"))
    moving.queue_wait_seconds = 0.2
    began = time.perf_counter()
    answer = await asyncio.wait_for(moving.generate(_turn("b")), 2)
    assert time.perf_counter() - began >= 0.18
    assert moving.served_by == "cloud-driver" and moving.tier == 2 and moving.tier_moves == 1
    assert answer.content

    alone = table.pick(table.resolve("local-model"))
    alone.queue_wait_seconds = 0.2
    with pytest.raises(admission.AdmissionFailure) as refused:
        await alone.generate(_turn("c"))
    assert refused.value.status == 503 and refused.value.code == "queue_timeout"
    assert refused.value.retry == "10"
    assert refused.value.message == (
        "Waited 0.2 s for room on local-model, 0 ahead. Retry shortly."
    )
    assert fair_queue.of(table).waiting() == 0
    gate.set()
    await running


async def test_cancel_while_waiting_for_room_leaks_nothing():
    from eugene_plexus_gateway import fair_queue

    fake = FakeDriverClient(name="rt-driver", model_id=MODEL, runtime="rt")
    gate = asyncio.Event()

    async def hold() -> None:
        await gate.wait()

    fake.generate_hook = hold
    from .conftest import runtime_facts

    table = make_routing_table(
        fake, runtimes=[runtime_facts("rt", parallel_slots=1, context_pool=10_000)]
    )
    queue = fair_queue.of(table)
    running = asyncio.ensure_future(table.pick(table.resolve(MODEL)).generate(_turn("a")))
    await asyncio.sleep(0.05)
    waiting = asyncio.ensure_future(table.pick(table.resolve(MODEL)).generate(_turn("b")))
    await asyncio.sleep(0.05)
    assert queue.waiting() == 1 and queue.running((None, "rt")) == 1
    held = table._ledger.in_flight((None, "rt"))
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    assert queue.waiting() == 0, "its place was freed at once"
    assert table._ledger.in_flight((None, "rt")) == held, "it held no room"
    gate.set()
    await running
    assert queue.running((None, "rt")) == 0 and table._ledger.in_flight((None, "rt")) == 0


async def test_close_refuses_every_waiter_shutting_down():
    from eugene_plexus_gateway import admission, fair_queue

    fake = FakeDriverClient(name="rt-driver", model_id=MODEL, runtime="rt")
    gate = asyncio.Event()

    async def hold() -> None:
        await gate.wait()

    fake.generate_hook = hold
    from .conftest import runtime_facts

    table = make_routing_table(fake, runtimes=[runtime_facts("rt", parallel_slots=1)])
    running = asyncio.ensure_future(table.pick(table.resolve(MODEL)).generate(_turn("a")))
    await asyncio.sleep(0.05)
    waiting = asyncio.ensure_future(table.pick(table.resolve(MODEL)).generate(_turn("b")))
    await asyncio.sleep(0.05)
    fair_queue.of(table).close()
    with pytest.raises(admission.AdmissionFailure) as refused:
        await asyncio.wait_for(waiting, 1)
    assert refused.value.code == "shutting_down" and refused.value.status == 503
    assert refused.value.message == "Eugene's gateway is restarting. Retry in a few seconds."
    with pytest.raises(admission.AdmissionFailure) as later:
        await table.pick(table.resolve(MODEL)).generate(_turn("c"))
    assert later.value.code == "shutting_down"
    gate.set()
    await running


def test_stopping_the_app_refuses_its_waiters(gateway, install):
    from eugene_plexus_gateway.app import close_queue

    root, driver = Root(limit=1), Holding(seconds=0.5)
    app = gateway(root, driver)
    headers = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(2) as pool:
        first = pool.submit(c.post, "/v1/chat/completions", json=_chat("one"), headers=headers)
        time.sleep(0.1)
        waiter = pool.submit(c.post, "/v1/chat/completions", json=_chat("two"), headers=headers)
        time.sleep(0.1)
        c.portal.call(close_queue, app)  # type: ignore[union-attr]
        refused = waiter.result(timeout=2)
        assert refused.status_code == 503
        assert refused.json()["error"]["code"] == "shutting_down"
        assert refused.headers["X-Eugene-Refusal-Reason"] == "shutting_down"
        assert first.result().status_code == 200


# --- WAKE: a starting runtime is waited for --------------------------------------


def _starting_app(gateway, install, *, ready_after: float | None):  # type: ignore[no-untyped-def]
    from eugene_plexus_gateway.lifecycle import WakeResult

    from .conftest import install_snapshot, runtime_facts

    root = Root(limit=8)
    driver = FakeDriverClient(name="llama-driver", model_id=MODEL, runtime="llama")
    app = gateway(root, driver, runtimes=[runtime_facts("llama", status="starting")])
    table = app.state.routing
    began: list[float] = []

    class Lifecycle:
        async def wake(self, resolution: Any) -> WakeResult:
            # As the real one answers when nothing is startable and a runtime is
            # coming up on its own.
            return WakeResult(
                ok=False, runtime="llama", waited_ms=0, message="still coming up: llama (starting)"
            )

        async def aclose(self) -> None:
            return None

    async def refresh_if_stale(max_age_seconds: float = 1.0) -> bool:
        began.append(time.perf_counter())
        if ready_after is not None and began[-1] - began[0] >= ready_after:
            install_snapshot(table, driver, runtimes=[runtime_facts("llama", status="ready")])
        return True

    table.refresh_if_stale = refresh_if_stale
    app.state.lifecycle = Lifecycle()
    return app, driver


def test_a_starting_runtime_is_waited_for(gateway, install):
    app, driver = _starting_app(gateway, install, ready_after=0.5)
    headers = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(4) as pool:
        answers = list(
            pool.map(
                lambda n: c.post("/v1/chat/completions", json=_chat(f"s{n}"), headers=headers),
                range(4),
            )
        )
    assert [a.status_code for a in answers] == [200] * 4, [a.text for a in answers]
    assert len(driver.calls) == 4


def test_a_runtime_still_starting_is_refused_in_its_words(gateway, install, monkeypatch):
    from eugene_plexus_gateway.routes import inference

    monkeypatch.setattr(inference, "_swap_wait_seconds", lambda request: 1.0)
    app, driver = _starting_app(gateway, install, ready_after=None)
    headers = _key(install)
    with TestClient(app) as c:
        answer = c.post("/v1/chat/completions", json=_chat(), headers=headers)
        assert answer.status_code == 503
        assert answer.headers["Retry-After"] == "10"
        assert answer.headers["X-Eugene-Refusal-Reason"] == "still_starting"
        error = answer.json()["error"]
        assert error["code"] == "still_starting"
        assert error["message"] == "llama is still starting after 1 s. Retry shortly."
        messages = c.post("/v1/messages", json=_chat(max_tokens=5), headers=headers)
        assert messages.status_code == 503
        assert messages.json()["error"]["type"] == "overloaded_error"
        rows = _rows(c, install, 2)
    assert {r["refusal"] for r in rows} == {"still_starting"}
    assert all(r["outcome"] == "refused" for r in rows)
    assert not driver.calls


# --- the review's ten findings (Codex, 2026-10-26) -----------------------------------


def test_a_policy_changed_while_waiting_is_refused_not_forwarded(gateway, install):
    """Blocker 1: the key is made local-only while a request waits. Its lease is
    granted under the new policy, but the backend was chosen and the request
    shaped under the old one: it is refused, never sent with `localOnly` false."""
    root, driver = Root(limit=1), Holding(seconds=0.4)
    app = gateway(root, driver)
    headers = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(2) as pool:
        first = pool.submit(c.post, "/v1/chat/completions", json=_chat("one"), headers=headers)
        time.sleep(0.1)
        second = pool.submit(c.post, "/v1/chat/completions", json=_chat("two"), headers=headers)
        time.sleep(0.1)
        with root.lock:
            root.local_only = True
        refused = second.result(timeout=5)
        assert refused.status_code == 409, refused.text
        assert "changed while it waited" in refused.json()["error"]["message"]
        assert first.result().status_code == 200
    assert [call.messages[-1].content for call in driver.calls] == ["one"]
    assert not root.active, "the refused request's lease was given back"


@pytest.mark.parametrize("gate", ["slot", "pool"])
async def test_a_fallback_waits_for_room_where_it_goes(gate):
    """Blocker 2: the first replica fails, and the fallback's runtime is full. The
    fallback waits in line for its slot (or its pool's room) like any turn,
    instead of entering a runtime that has none."""
    from eugene_plexus_gateway import fair_queue

    from .conftest import runtime_facts

    first = FakeDriverClient(name="a-driver", base_url="http://a", model_id=MODEL, runtime="ra")
    second = FakeDriverClient(name="b-driver", base_url="http://b", model_id=MODEL, runtime="rb")
    first.generate_error = httpx.ConnectError("refused")
    pooled = {"context_pool": 10_000} if gate == "pool" else {}
    table = make_routing_table(
        first,
        second,
        runtimes=[
            runtime_facts("ra", parallel_slots=1, **pooled),
            runtime_facts("rb", parallel_slots=1, **pooled),
        ],
    )
    queue = fair_queue.of(table)
    rb = (None, "rb")
    if gate == "slot":
        queue._take_seat(rb)  # a request runs there now
    else:
        table._ledger.take(rb, 9_500)  # its pool is all but full
    client = table.pick(table.resolve(MODEL))
    task = asyncio.ensure_future(client.generate(_turn("x" * 3_500, max_tokens=1_000)))
    await asyncio.sleep(0.2)
    assert first.calls, "the first replica was tried"
    assert not second.calls, "the fallback entered a runtime with no room"
    assert queue.waiting() == 1
    if gate == "slot":
        queue._give_seat(rb)
    else:
        table._ledger.give(rb, 9_500)
    await asyncio.wait_for(task, 2)
    assert client.served_by == "b-driver"
    assert queue.running(rb) == 0 and table._ledger.in_flight(rb) == 0


def test_root_refusals_do_not_let_a_key_wait_past_its_bound(gateway, install):
    """Blocker 3: five requests leave the line at once, and the root refuses all
    five because another gateway holds the key's leases. One may wait (the key's
    `maxQueuedRequests`), the others are refused, and the one is served when the
    other gateway lets go."""
    from eugene_plexus_gateway import fair_queue

    root, driver = Root(limit=5, max_queued=1), Holding(seconds=0.05)
    root.elsewhere["key-1"] = 5
    root.acquire_delay = 0.3  # all five ask the root before any answer comes back
    app = gateway(root, driver)
    headers = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(5) as pool:
        futures = [
            pool.submit(c.post, "/v1/chat/completions", json=_chat(f"r{n}"), headers=headers)
            for n in range(5)
        ]
        most = 0
        for _ in range(30):
            most = max(most, fair_queue.of(app.state.routing).waiting())
            time.sleep(0.05)
        assert most <= 1, f"{most} waited for a key that allows 1"
        with root.lock:
            root.elsewhere.clear()
        answers = [f.result(timeout=10) for f in futures]
    assert sorted(a.status_code for a in answers) == [200, 429, 429, 429, 429]
    assert {a.json()["error"]["code"] for a in answers if a.status_code == 429} == {
        "key_queue_full"
    }


def test_requests_for_a_starting_runtime_wait_in_the_bounded_line(gateway, install, monkeypatch):
    """Blocker 4: requests that meet a starting runtime wait in the line, so the
    gateway's bound applies to them, the Metrics page counts them, and stopping
    the gateway refuses them `shutting_down`."""
    from eugene_plexus_gateway import admission, fair_queue
    from eugene_plexus_gateway.app import close_queue
    from eugene_plexus_gateway.routes import inference

    monkeypatch.setattr(inference, "_swap_wait_seconds", lambda request: 30.0)
    monkeypatch.setattr(admission, "queue_settings", lambda app: (60.0, 16))
    app, driver = _starting_app(gateway, install, ready_after=None)
    app.state.client_key_guard = Root(limit=64, max_queued=64).as_guard(ttl_seconds=0)
    headers = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(20) as pool:
        futures = [
            pool.submit(c.post, "/v1/chat/completions", json=_chat(f"s{n}"), headers=headers)
            for n in range(20)
        ]
        queue = fair_queue.of(app.state.routing)
        for _ in range(100):
            if queue.waiting() == 16 and sum(f.done() for f in futures) == 4:
                break
            time.sleep(0.05)
        assert queue.waiting() == 16, queue.waiting()
        full = [f.result() for f in futures if f.done()]
        assert [a.json()["error"]["code"] for a in full] == ["queue_full"] * 4
        assert sum(sum(q["waiting"].values()) for q in queue.snapshot()) == 16
        c.portal.call(close_queue, app)  # type: ignore[union-attr]
        stopped = [f.result(timeout=5) for f in futures if f.result(timeout=5) not in full]
    assert len(stopped) == 16
    assert {(a.status_code, a.json()["error"]["code"]) for a in stopped} == {(503, "shutting_down")}
    assert not driver.calls


def test_a_key_turn_before_a_wake_that_runs_out_is_key_queue_timeout(
    settings, install, monkeypatch
):
    """Should-fix 6: the key's turn a wake waits for ends in `key_queue_timeout`,
    not an unhandled error."""
    from eugene_plexus_gateway import admission

    from .conftest import runtime_facts

    monkeypatch.setattr(admission, "queue_settings", lambda app: (0.3, 256))
    root = Root(limit=1)
    busy = Holding(seconds=1.0, name="busy-driver", model_id="busy")
    sleeper = FakeDriverClient(name="sleepy-driver", model_id=MODEL, runtime="sleepy")
    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    app.state.client_key_guard = root.as_guard(ttl_seconds=0)
    app.state.routing = make_routing_table(
        busy,
        sleeper,
        runtimes=[runtime_facts("sleepy", status="stopped", start_on_demand=True)],
    )

    class Lifecycle:
        async def wake(self, resolution: Any) -> Any:
            raise AssertionError("a key at its limit started a runtime")

        async def aclose(self) -> None:
            return None

    app.state.lifecycle = Lifecycle()
    headers = _key(install)
    with TestClient(app) as c, ThreadPoolExecutor(1) as pool:
        running = pool.submit(
            c.post, "/v1/chat/completions", json={**_chat("one"), "model": "busy"}, headers=headers
        )
        time.sleep(0.1)
        answer = c.post("/v1/chat/completions", json=_chat("wake"), headers=headers)
        assert answer.status_code == 429, answer.text
        assert answer.json()["error"]["code"] == "key_queue_timeout"
        assert running.result().status_code == 200


async def test_a_failed_reservation_gives_back_what_it_took():
    """Should-fix 7: a reservation that fails as an entry leaves takes nothing with it."""
    from eugene_plexus_gateway import fair_queue

    queue = fair_queue.FairQueue()
    runtime = (None, "rt")

    def broken(seat: Any) -> None:
        raise RuntimeError("the ledger broke")

    entry = fair_queue.Entry(
        key="k", limit=2, seats=[fair_queue.Seat(runtime=runtime, slots=2)], reserve=broken
    )
    with pytest.raises(RuntimeError, match="the ledger broke"):
        await queue.wait(entry)
    assert queue.key_running("k") == 0 and queue.running(runtime) == 0
    assert queue.waiting() == 0


async def test_a_refused_key_is_forgotten_when_its_last_waiter_leaves():
    """Should-fix 8: what the root refused is not kept once nobody waits for it."""
    from eugene_plexus_gateway import fair_queue

    queue = fair_queue.FairQueue()
    queue.root_full("k")
    with pytest.raises(fair_queue.Waited):
        await queue.wait(fair_queue.Entry(key="k", limit=2, wait_seconds=0.1))
    assert queue._root_full == {}
    queue.root_full("k")
    task = asyncio.ensure_future(queue.wait(fair_queue.Entry(key="k", limit=2)))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert queue._root_full == {}


def _line(queue: Any, flows: list[int], runtime: Any = (None, "busy")) -> None:
    """`flows[i]` waiting entries for person i, all held back by a full runtime."""
    from eugene_plexus_gateway import fair_queue

    loop = asyncio.new_event_loop()
    try:
        seat = fair_queue.Seat(runtime=runtime, slots=1)
        queue._take_seat(runtime)
        seq = 0
        for person, depth in enumerate(flows):
            for _ in range(depth):
                seq += 1
                entry = fair_queue.Entry(fairness=f"person:{person}", seats=[seat])
                entry.seq, entry.joined, entry.future = seq, 0.0, loop.create_future()
                queue._add(entry)
    finally:
        loop.close()


def _cost(call: Any) -> float:
    best = float("inf")
    for _ in range(3):
        began = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - began)
    return best


def test_ordering_a_skewed_line_costs_about_what_an_even_one_does():
    """Should-fix 9: one person with 4,000 waiting beside 96 with one each is
    ordered in about the time one person with 4,096 is: the work follows the
    entries, not depth times persons."""
    from eugene_plexus_gateway import fair_queue

    skewed, even = fair_queue.FairQueue(), fair_queue.FairQueue()
    _line(skewed, [4_000] + [1] * 96)
    _line(even, [4_096])
    assert len(skewed._ordered()) == len(even._ordered()) == 4_096
    assert _cost(skewed._ordered) <= 3 * _cost(even._ordered) + 0.005


async def test_a_request_that_needs_no_wait_does_not_pay_for_the_line():
    """Should-fix 9: with 4,096 waiting on a busy runtime, a request for a free one
    that competes with none of them leaves without a pass over the line."""
    from eugene_plexus_gateway import fair_queue

    queue = fair_queue.FairQueue()
    _line(queue, [4_000] + [1] * 96)
    passes = 0
    real = queue._dispatch

    def counted() -> None:
        nonlocal passes
        passes += 1
        real()

    queue._dispatch = counted  # type: ignore[method-assign]
    free = fair_queue.Seat(runtime=(None, "free"), slots=1)
    began = time.perf_counter()
    ticket = await queue.wait(fair_queue.Entry(key="other", limit=2, seats=[free]))
    took = time.perf_counter() - began
    assert ticket.seat is free and passes == 0, passes
    assert took < 0.01, f"{took * 1000:.1f} ms for a request that waited for nothing"
    ticket.release()
