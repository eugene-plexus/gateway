"""Retained request metrics.

Two of these tests exist because reading the code found a defect a
fixture would not have: the streaming path never set `x_eugene_plexus`,
so anything recording off the response envelope would have been blind to
every streaming client, and `latency_ms` includes failed attempts, so
attributing it to the backend that answered reports a fast backend as
slow in exactly the cascade an operator is investigating.

Those two are `test_a_streamed_completion_is_recorded` and
`test_a_cascade_attributes_its_own_time_to_each_backend`. The live
acceptance run checks the same two things against real engines; these
check them where they can be checked in a second.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import Usage
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.metrics import AttemptRow, MetricsStore, RequestRow
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table, runtime_facts


class SlowFakeDriver(FakeDriverClient):
    """A fake that takes measurable time to answer.

    The plain fake returns in under a millisecond, which floors
    `elapsed_ms` to 0 — and tokens per second is not computable from a
    zero, so the store correctly reports null. Real generation takes
    hundreds of milliseconds at least, so a fake that answers
    instantaneously is not a simpler version of a backend, it is a
    different one. Anything asserting throughput needs this.
    """

    async def generate(self, request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0.02)
        return await super().generate(request)


@pytest.fixture
def metrics_app(settings: Settings) -> FastAPI:
    """One driver, metrics on, writing to the test's own tmp file."""
    app = create_app(settings=settings)
    fake = SlowFakeDriver(name="qwen-box", model_id="qwen", runtime="qwen-rt")
    fake.usage = Usage(promptTokens=10, completionTokens=40, totalTokens=50)
    app.state.routing = make_routing_table(fake, runtimes=[runtime_facts("qwen-rt")])
    return app


@pytest.fixture
def metrics_client(metrics_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(metrics_app) as c:
        yield c


def _drain(client: TestClient) -> None:
    """Wait for the off-path writer to commit what has been queued.

    The writer is deliberately asynchronous, so a test that reads
    immediately after a completion is racing it. Polling the endpoint is
    what a caller would do and keeps the production path unchanged; a
    test-only flush hook would be API existing only for tests.
    """
    for _ in range(100):
        body = client.get("/v1/metrics").json()
        if body["groups"]:
            return
        import time as _t

        _t.sleep(0.05)


def test_a_completion_is_recorded_with_the_backend_that_served_it(
    metrics_client: TestClient,
) -> None:
    metrics_client.post(
        "/v1/chat/completions",
        json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
    )
    _drain(metrics_client)

    body = metrics_client.get("/v1/metrics").json()
    assert body["rowsDropped"] == 0
    assert len(body["groups"]) == 1
    group = body["groups"][0]
    assert group["model"] == "qwen"
    assert group["driver"] == "qwen-box"
    # By runtime name, so two replicas of one model stay distinguishable -
    # which is the whole case load balancing exists for.
    assert group["runtime"] == "qwen-rt"
    assert group["requests"] == 1
    assert group["errors"] == 0
    assert group["tokensPerSecond"]["samples"] == 1


def test_a_streamed_completion_is_recorded(metrics_client: TestClient) -> None:
    """The defect this whole module was shaped around.

    The streaming path builds a `StreamingResponse` and used to set no
    routing extension at all, so a recorder hooked to the response would
    have silently ignored every streaming client - which is most real
    clients, the UI playground included. Recording happens at the routing
    hooks instead, which fire on both paths.
    """
    with metrics_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        frames = [line for line in response.iter_lines() if line.startswith("data: ")]

    _drain(metrics_client)
    body = metrics_client.get("/v1/metrics").json()
    assert body["groups"], "a streamed completion recorded nothing"
    assert body["groups"][0]["requests"] == 1

    page = metrics_client.get("/v1/metrics/requests").json()
    assert page["requests"][0]["streamed"] is True

    # And the stream now says which backend answered, on its final frame
    # beside `usage` - the gap that hid the recording gap.
    final = json.loads(frames[-2].removeprefix("data: "))
    assert final["x_eugene_plexus"]["driver"] == "qwen-box"
    assert final["usage"]["completion_tokens"] == 40


def test_a_cascade_attributes_its_own_time_to_each_backend(
    settings: Settings,
) -> None:
    """Throughput must come from the serving attempt, not the request.

    A request's total includes every failed attempt before the one that
    answered. Dividing tokens by that total would report the healthy
    backend as slower the longer its dead neighbour took to fail - which
    is backwards, and is why an attempt is its own row.
    """
    dead = FakeDriverClient(name="dead", model_id="qwen")
    dead.generate_error = httpx.ConnectError("refused")
    alive = FakeDriverClient(name="alive", model_id="qwen-backup")
    alive.usage = Usage(promptTokens=10, completionTokens=40, totalTokens=50)

    app = create_app(settings=settings)
    app.state.routing = make_routing_table(
        dead,
        alive,
        slots=[{"model": "qwen", "targets": ["qwen", "qwen-backup"]}],
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200
        _drain(client)

        page = client.get("/v1/metrics/requests").json()
        record = page["requests"][0]
        assert record["attempts"] == 2
        assert record["outcome"] == "served"
        # Both backends recorded, in the order tried, and only one served.
        assert [t["driver"] for t in record["tries"]] == ["dead", "alive"]
        assert [t["served"] for t in record["tries"]] == [False, True]
        # The class name, never the message: a driver error can carry a
        # provider's response body and this string is retained.
        assert record["tries"][0]["error"] == "ConnectError"
        assert "refused" not in json.dumps(record)

        served = record["tries"][1]
        assert served["elapsedMs"] <= record["totalMs"]

        cascade = client.get("/v1/metrics/requests?outcome=cascaded").json()
        assert len(cascade["requests"]) == 1


def test_a_request_that_no_backend_served_is_still_recorded(settings: Settings) -> None:
    """A failing backend must not read as an idle one."""
    dead = FakeDriverClient(name="dead", model_id="qwen")
    dead.generate_error = httpx.ConnectError("refused")

    app = create_app(settings=settings)
    app.state.routing = make_routing_table(dead)
    with TestClient(app) as client:
        assert (
            client.post(
                "/v1/chat/completions",
                json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
            ).status_code
            == 502
        )
        _drain(client)
        page = client.get("/v1/metrics/requests").json()
        assert page["requests"][0]["outcome"] == "error"
        assert page["requests"][0]["servedModel"] is None
        body = client.get("/v1/metrics").json()
        group = body["groups"][0]
        assert group["errors"] == 1
        # And attributed to the backend that failed, not to nobody.
        # Joining the summary on the SERVING attempt alone put every
        # failed request in a null-driver group, so "is anything
        # failing" could be answered with a count but not with a name -
        # found live the first time a real backend failed.
        assert group["driver"] == "dead"
        assert group["model"] == "qwen"


def test_metrics_off_is_reported_as_off_not_as_empty(settings: Settings) -> None:
    """503, not an empty 200.

    "Metrics are disabled" and "nothing was served in this window" are
    different answers, and a UI that cannot tell them apart shows an
    empty chart for a switched-off feature.
    """
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(FakeDriverClient(name="d", model_id="qwen"))
    app.state.metrics = None  # what metricsEnabled: false produces
    with TestClient(app) as client:
        response = client.get("/v1/metrics")
        assert response.status_code == 503
        assert "metricsEnabled" in response.json()["detail"]["detail"]


def test_no_metrics_file_is_written_in_safe_mode(tmp_path: Path) -> None:
    """Safe mode exists to get a broken install back to a config endpoint;
    opening a database is one more thing that can fail on the way."""
    app = create_app(
        settings=Settings(
            config_file=tmp_path / "config.yaml",
            metrics_file=tmp_path / "metrics.sqlite3",
            safe_mode=True,
        )
    )
    with TestClient(app) as client:
        assert client.get("/v1/metrics").status_code == 503
    assert not (tmp_path / "metrics.sqlite3").exists()


# --------------------------------------------------------------------------- #
# The store itself
# --------------------------------------------------------------------------- #


def _row(
    *,
    driver: str = "d",
    tokens: int | None = 40,
    elapsed: int = 1000,
    when: datetime | None = None,
    served: bool = True,
) -> RequestRow:
    return RequestRow(
        started_at=when or datetime.now(UTC),
        requested_model="qwen",
        served_model="qwen" if served else None,
        attempts=1,
        tier=1 if served else None,
        total_ms=elapsed,
        outcome="served" if served else "error",
        completion_tokens=tokens,
        tries=[AttemptRow(driver=driver, elapsed_ms=elapsed, served=served)],
    )


async def _store(path: Path, **kwargs: object) -> MetricsStore:
    store = MetricsStore(path, **kwargs)  # type: ignore[arg-type]
    await store.start()
    return store


def test_throughput_is_null_when_no_backend_reports_tokens(tmp_path: Path) -> None:
    """Null, not zero.

    A backend that reported no usage leaves the rate uncomputable. A
    zero here says "slow" where it means "unmeasured", and a UI cannot
    recover the difference.

    Not a property of a backend kind, which an earlier version of this
    docstring claimed: the CLI subscription backends do report usage,
    and omit it only when their CLI emitted none.
    """

    async def run() -> None:
        store = await _store(tmp_path / "m.sqlite3")
        try:
            store._insert([_row(tokens=None)])
            groups = store.summary(
                since=datetime.now(UTC) - timedelta(hours=1),
                until=datetime.now(UTC) + timedelta(hours=1),
            )
            assert groups[0]["tokensPerSecond"] is None
            assert groups[0]["requests"] == 1
        finally:
            await store.aclose()

    asyncio.run(run())


def test_rows_past_retention_are_dropped_and_the_hour_is_kept(tmp_path: Path) -> None:
    """Rollup before prune, always: the other order loses the traffic it
    was about to summarise."""

    async def run() -> None:
        store = await _store(tmp_path / "m.sqlite3", retention_days=1)
        try:
            old = datetime.now(UTC) - timedelta(days=3)
            store._insert([_row(when=old), _row()])
            store.maintain()

            surviving = store.summary(
                since=datetime.now(UTC) - timedelta(days=30),
                until=datetime.now(UTC) + timedelta(hours=1),
            )
            assert sum(g["requests"] for g in surviving) == 1

            with store._reader() as conn:
                assert conn is not None
                rolled = conn.execute(
                    "SELECT requests, completion_tokens FROM rollup WHERE bucket_start = ?",
                    (
                        old.replace(minute=0, second=0, microsecond=0)
                        .isoformat()
                        .replace("+00:00", "Z"),
                    ),
                ).fetchone()
            assert rolled == (1, 40), "the pruned hour left no aggregate behind"

            # Idempotent. The rollup runs hourly on the writer's own loop,
            # so it WILL see hours it has already summarised; without the
            # `rollup_through` guard an INSERT OR REPLACE over a widening
            # window double-counts, and a metrics table that inflates
            # slowly is worse than one that is obviously broken.
            store.maintain()
            store.maintain()
            with store._reader() as conn:
                assert conn is not None
                again = conn.execute("SELECT SUM(requests) FROM rollup").fetchone()
            assert again == (1,), "re-running maintenance double-counted"
        finally:
            await store.aclose()

    asyncio.run(run())


def test_a_full_queue_drops_and_says_so(tmp_path: Path) -> None:
    """Recording degrades before inference does, and reports it.

    Silently under-reporting would make a saturated install look like a
    quiet one, which is the opposite of what the numbers are for.
    """

    async def run() -> None:
        store = await _store(tmp_path / "m.sqlite3", queue_max=1)
        try:
            for _ in range(50):
                store.record(_row())
            assert store.rows_dropped > 0
        finally:
            await store.aclose()

    asyncio.run(run())


def test_rows_survive_a_restart(tmp_path: Path) -> None:
    """The reason this is SQLite and not a counter.

    The agent respawns every child on operator login, so an in-memory
    history would reset every time someone signed in.
    """

    async def run() -> None:
        path = tmp_path / "m.sqlite3"
        first = await _store(path)
        first._insert([_row()])
        await first.aclose()

        second = await _store(path)
        try:
            groups = second.summary(
                since=datetime.now(UTC) - timedelta(hours=1),
                until=datetime.now(UTC) + timedelta(hours=1),
            )
            assert groups[0]["requests"] == 1
        finally:
            await second.aclose()

    asyncio.run(run())


def test_paging_is_stable_while_rows_arrive(tmp_path: Path) -> None:
    """Keyset-paged on the id, not OFFSET, so inserting during a walk
    cannot make a page repeat or skip a row."""

    async def run() -> None:
        store = await _store(tmp_path / "m.sqlite3")
        try:
            store._insert([_row(driver=f"d{i}") for i in range(5)])
            first, cursor = store.requests(limit=2)
            assert len(first) == 2
            assert cursor is not None

            store._insert([_row(driver="newcomer")])
            second, _ = store.requests(limit=2, cursor=cursor)
            seen = [r["tries"][0]["driver"] for r in (*first, *second)]
            assert "newcomer" not in seen
            assert len(set(seen)) == len(seen)
        finally:
            await store.aclose()

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Phase decomposition and balancer decisions
# --------------------------------------------------------------------------- #


def test_the_routing_phase_is_measured_separately_from_the_backend(
    metrics_client: TestClient,
) -> None:
    """`routingMs` covers work that used to happen before any clock started.

    Resolve, pick and any routing-table refresh run before `totalMs`
    begins, so they were invisible - and a refresh does HTTP to the
    agent and to every driver inside the request that triggered it.
    """
    metrics_client.post(
        "/v1/chat/completions",
        json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
    )
    _drain(metrics_client)

    record = metrics_client.get("/v1/metrics/requests").json()["requests"][0]
    assert record["routingMs"] is not None
    assert record["refreshed"] is False
    # Present and not conflated with the wake, which is its own field.
    assert record["routingMs"] >= 0
    assert record["strategy"] == "least_busy"


def test_the_control_planes_own_overhead_is_derivable(metrics_client: TestClient) -> None:
    """`elapsedMs - backendMs` is the local hop plus the driver's work.

    gateway.yaml has asserted since M0 that the extra hop is
    "sub-millisecond against a multi-second generation" and "not a cost
    worth optimising away". Both numbers were already being produced;
    keeping the driver's own measurement makes that claim checkable
    rather than asserted.
    """
    metrics_client.post(
        "/v1/chat/completions",
        json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
    )
    _drain(metrics_client)

    served = metrics_client.get("/v1/metrics/requests").json()["requests"][0]["tries"][0]
    assert served["backendMs"] is not None
    assert served["backendMs"] <= served["elapsedMs"]

    group = metrics_client.get("/v1/metrics").json()["groups"][0]
    assert group["overheadMs"] is not None
    assert group["overheadMs"]["p50"] >= 0
    assert group["routingMs"] is not None


class SilentLatencyDriver(FakeDriverClient):
    """A backend that answers without reporting its own latency.

    `GenerateResponse.latencyMs` is optional, and a driver that omits it
    leaves the control plane's overhead uncomputable - not zero.
    """

    async def generate(self, request):  # type: ignore[no-untyped-def]
        response = await super().generate(request)
        response.latencyMs = None
        return response


def test_a_backend_that_reports_no_latency_leaves_overhead_unmeasured(
    settings: Settings,
) -> None:
    """Null, not zero - the same rule as throughput.

    A zero overhead would read as "the hop is free", which is a claim,
    where the data says "this backend did not tell us how long it took".
    """
    fake = SilentLatencyDriver(name="quiet", model_id="qwen")
    fake.usage = Usage(promptTokens=1, completionTokens=1, totalTokens=2)
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake)
    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
        _drain(client)
        record = client.get("/v1/metrics/requests").json()["requests"][0]
        assert record["tries"][0]["backendMs"] is None
        group = client.get("/v1/metrics").json()["groups"][0]
        assert group["overheadMs"] is None, "overhead was invented from a missing measurement"
        # The routing phase is ours to measure and is still there.
        assert group["routingMs"] is not None


def test_the_balancer_records_what_it_saw_when_there_was_a_choice(
    settings: Settings,
) -> None:
    """Two replicas of one model, one of them asleep.

    The recorded candidates are the *inputs* to the decision - eligible,
    why not, in-flight, capacity - not a score. There is no score:
    least-busy is a sort, and inventing a number to display would be
    inventing the smarter balancer that was deliberately deferred.
    """
    a = FakeDriverClient(name="replica-a", model_id="qwen", runtime="rt-a")
    b = FakeDriverClient(name="replica-b", model_id="qwen", runtime="rt-b")
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(
        a,
        b,
        runtimes=[runtime_facts("rt-a"), runtime_facts("rt-b", status="stopped")],
    )
    with TestClient(app) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
        _drain(client)
        record = client.get("/v1/metrics/requests").json()["requests"][0]

    considered = {c["driver"]: c for c in record["candidates"]}
    assert set(considered) == {"replica-a", "replica-b"}
    assert considered["replica-a"]["eligible"] is True
    assert considered["replica-a"]["reason"] is None
    # The rejection carries its reason, which is the answer to "why did
    # this not go to the other replica".
    assert considered["replica-b"]["eligible"] is False
    assert "stopped" in considered["replica-b"]["reason"]
    # And the load signal the sort actually runs on.
    assert considered["replica-a"]["inFlight"] == 0
    assert considered["replica-a"]["slots"] == 1
    assert all(c["tier"] == 1 for c in record["candidates"])


def test_one_eligible_backend_records_no_candidate_list(
    metrics_client: TestClient,
) -> None:
    """A row per request saying "the only option was chosen" is noise.

    The list is for answering "why there and not the other one", so with
    no other one there is nothing to answer.
    """
    metrics_client.post(
        "/v1/chat/completions",
        json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
    )
    _drain(metrics_client)
    record = metrics_client.get("/v1/metrics/requests").json()["requests"][0]
    assert record["candidates"] == []


def test_asking_what_the_balancer_saw_does_not_change_what_it_does(
    settings: Settings,
) -> None:
    """The observation must not move the thing observed.

    `_order` rotates a per-target cursor so an idle install alternates
    between replicas. Reading it to find out what it did would advance
    that cursor - so the recorded candidates are the inputs, and this
    test pins the property that reading them is free.
    """
    a = FakeDriverClient(name="replica-a", model_id="qwen", runtime="rt-a")
    b = FakeDriverClient(name="replica-b", model_id="qwen", runtime="rt-b")
    table = make_routing_table(a, b, runtimes=[runtime_facts("rt-a"), runtime_facts("rt-b")])

    firsts = []
    for _ in range(4):
        resolution = table.resolve("qwen")
        # Interleave the read with the pick, the way the route does.
        table.candidates_considered(resolution)
        client = table.pick(resolution)
        assert client is not None
        firsts.append(client.candidates[0].name)
        table.candidates_considered(resolution)

    # Still strictly alternating, as it would be with no reads at all.
    assert firsts == ["replica-a", "replica-b", "replica-a", "replica-b"]
