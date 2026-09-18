"""What a slot leaves behind when a stream ends in a way nobody planned.

`TieredClient` reports every attempt to the routing table through
`RoutingHooks`, and the table counts in flight per driver and per
runtime. That counter is what the balancer reads, what the idle pass
consults before unloading, and what an eviction consults before making
room for a wake -- so an attempt that is opened and never closed takes a
runtime out of the lifecycle permanently, and one that is closed twice,
or closed against the wrong runtime, reports a busy engine as idle.

Three ways for an attempt to end that the `except`/`else` pair cannot
see, each with a test here:

* the consumer abandons the generator (the playground's Stop button, a
  browser tab closed mid-answer) -- `GeneratorExit`, a `BaseException`;
* the request task is cancelled (starlette cancels the response task
  group on `http.disconnect`) -- `CancelledError`, also a
  `BaseException`;
* the driver's stream simply stops without ever saying `done` -- no
  exception at all, so the `else` arm runs and calls it served.

Plus the pairing itself: the runtime a slot decrements has to be the one
it incremented, and the snapshot it looked that up in can be replaced
between the two.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import (
    EmbedRequest,
    GenerateRequest,
    GenerateResponse,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import StreamEvent, TieredClient
from eugene_plexus_gateway.routing import RoutingTable, collect_attempts
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, install_snapshot, make_routing_table, runtime_facts

DRIVER = "qwen-box"
RUNTIME = "engine-a"
#: Every demand counter is keyed by `(node, name)` since R1.6: a
#: driver or runtime NAME is unique per agent, not per install. These
#: fixtures are single-host, so the node half is `None`.
DRIVER_KEY = (None, DRIVER)
RUNTIME_KEY = (None, RUNTIME)
MODEL = "Qwen3-30B-A3B-Q4_K_M"


def _request() -> GenerateRequest:
    return GenerateRequest(messages=[{"role": "user", "content": "hi"}])


def _chat(**extra: object) -> dict[str, object]:
    body: dict[str, object] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(extra)
    return body


class _ParkingDriver(FakeDriverClient):
    """One token, then it waits forever: a backend still generating.

    The state a cancellation actually arrives in. A fake that returns
    promptly cannot be cancelled mid-stream at all, so it would prove
    nothing about the path a closed tab takes.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.parked = asyncio.Event()

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        self.calls.append(request)
        yield StreamEvent(text="one ")
        self.parked.set()
        await asyncio.Event().wait()  # never set
        raise AssertionError("unreachable")


class _TruncatedDriver(FakeDriverClient):
    """Tokens, then the connection closes: no `done` event ever arrives.

    What a backend killed mid-answer looks like from here -- the driver
    is supposed to raise on this (M10), and a gateway that trusts it to
    is a gateway that records a truncation as a completion.
    """

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        self.calls.append(request)
        yield StreamEvent(text="half an ")
        yield StreamEvent(text="answer")


def _slot(*fakes: FakeDriverClient, table: RoutingTable) -> TieredClient:
    return TieredClient(name=MODEL, tiers=[list(fakes)], hooks=table)


def _one_ready_backend(fake: FakeDriverClient) -> RoutingTable:
    return make_routing_table(fake, runtimes=[runtime_facts(RUNTIME)])


# --------------------------------------------------------------------------- #
# The counter balances -- baseline first, so a fix that never increments
# cannot pass the leak tests by accident.
# --------------------------------------------------------------------------- #


async def test_a_stream_consumed_to_the_end_leaves_the_counters_at_zero() -> None:
    fake = FakeDriverClient(name=DRIVER, model_id=MODEL, runtime=RUNTIME)
    fake.responses = ["one two three"]
    table = _one_ready_backend(fake)

    with collect_attempts() as rows:
        events = [event async for event in _slot(fake, table=table).stream(_request())]

    assert any(event.done for event in events)
    assert [row.served for row in rows] == [True]
    assert table.inflight(DRIVER_KEY) == 0
    assert table.runtime_inflight(RUNTIME_KEY) == 0


# --------------------------------------------------------------------------- #
# §6.1 #3 -- a stream that ends without returning
# --------------------------------------------------------------------------- #


async def test_a_consumer_that_closes_the_stream_returns_the_counters_to_zero() -> None:
    """The closed tab. `aclose()` raises `GeneratorExit` at the yield, so
    neither `except Exception` nor `else` runs."""
    fake = _ParkingDriver(name=DRIVER, model_id=MODEL, runtime=RUNTIME)
    table = _one_ready_backend(fake)

    with collect_attempts() as rows:
        stream = _slot(fake, table=table).stream(_request())
        first = await stream.__anext__()
        assert first.text == "one "
        # The attempt really is open -- otherwise the assertions below
        # pass against an implementation that never counted at all.
        assert table.inflight(DRIVER_KEY) == 1
        assert table.runtime_inflight(RUNTIME_KEY) == 1

        await stream.aclose()

    assert table.inflight(DRIVER_KEY) == 0
    assert table.runtime_inflight(RUNTIME_KEY) == 0
    # Closed is not served: the row says what happened, and the
    # backend's "last served" mark is not refreshed by an abandonment.
    assert [row.served for row in rows] == [False]
    assert table._last_request.get(DRIVER) is None


async def test_a_cancelled_request_returns_the_counters_to_zero() -> None:
    """Starlette cancels the response task group on `http.disconnect`.
    The cancellation lands inside the driver's own await, so it unwinds
    through the slot -- as a `BaseException`, past both arms."""
    fake = _ParkingDriver(name=DRIVER, model_id=MODEL, runtime=RUNTIME)
    table = _one_ready_backend(fake)
    slot = _slot(fake, table=table)

    async def consume() -> None:
        async for _ in slot.stream(_request()):
            pass

    task = asyncio.create_task(consume())
    await fake.parked.wait()
    assert table.runtime_inflight(RUNTIME_KEY) == 1

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert table.inflight(DRIVER_KEY) == 0
    assert table.runtime_inflight(RUNTIME_KEY) == 0


async def test_a_leaked_counter_would_keep_a_runtime_out_of_the_idle_pass() -> None:
    """Why the leak matters rather than that it happens.

    `runtime_inflight() > 0` is what the idle pass and the eviction both
    consult; a runtime stuck above zero is never unloaded and can never
    be evicted to make room for a wake, for the life of the process."""
    fake = _ParkingDriver(name=DRIVER, model_id=MODEL, runtime=RUNTIME)
    table = _one_ready_backend(fake)

    for _ in range(3):
        stream = _slot(fake, table=table).stream(_request())
        await stream.__anext__()
        await stream.aclose()

    assert table.runtime_inflight(RUNTIME_KEY) == 0


async def test_a_cascade_records_one_row_per_attempt() -> None:
    """One attempt is one row. The arms that close an attempt have to
    agree that it is closed, or a failed attempt is counted twice and
    every cascade in the metrics reads as one attempt longer than it
    was."""
    dead = FakeDriverClient(name="dead-box", model_id=MODEL, runtime=RUNTIME)
    dead.stream_error_after = 0
    dead.stream_error = httpx.ConnectError("connection refused")
    alive = FakeDriverClient(name=DRIVER, model_id=MODEL, runtime=RUNTIME)
    alive.responses = ["served by the survivor"]
    table = make_routing_table(dead, alive, runtimes=[runtime_facts(RUNTIME)])

    with collect_attempts() as rows:
        async for _ in _slot(dead, alive, table=table).stream(_request()):
            pass

    assert [(row.driver, row.served) for row in rows] == [
        ("dead-box", False),
        (DRIVER, True),
    ]
    assert table.runtime_inflight(RUNTIME_KEY) == 0


# --------------------------------------------------------------------------- #
# §6.3 #38 -- a stream with no `done` frame is not a stream that finished
# --------------------------------------------------------------------------- #


async def test_a_stream_that_never_says_done_is_not_recorded_served() -> None:
    fake = _TruncatedDriver(name=DRIVER, model_id=MODEL, runtime=RUNTIME)
    table = _one_ready_backend(fake)

    with collect_attempts() as rows:
        events = [event async for event in _slot(fake, table=table).stream(_request())]

    assert [event.text for event in events] == ["half an ", "answer"]
    assert [row.served for row in rows] == [False], (
        "the answer was truncated; recording it served hides truncation from "
        "the one surface that could show it"
    )
    assert rows[0].error is not None
    assert table.runtime_inflight(RUNTIME_KEY) == 0


async def test_a_truncated_stream_does_not_mark_the_backend_as_having_served() -> None:
    """`_last_request` is the balancer's and the idle pass's idea of when
    this backend last did useful work. A truncation is not that."""
    fake = _TruncatedDriver(name=DRIVER, model_id=MODEL, runtime=RUNTIME)
    table = _one_ready_backend(fake)

    async for _ in _slot(fake, table=table).stream(_request()):
        pass

    assert table._last_request.get(DRIVER) is None
    assert table._runtime_last_request.get(RUNTIME_KEY) is None


def test_a_stream_that_never_says_done_still_gets_a_terminal_frame(
    settings: Settings,
) -> None:
    """An OpenAI client reads the answer as finished when the terminal
    chunk carries a `finish_reason`. Today this path emits `[DONE]` with
    no terminal chunk at all, so a client that switches on the field
    never sees one."""
    fake = _TruncatedDriver(name=DRIVER, model_id=MODEL, runtime=RUNTIME)
    app: FastAPI = create_app(settings=settings)
    app.state.routing = make_routing_table(fake, runtimes=[runtime_facts(RUNTIME)])

    with TestClient(app) as c:
        response = c.post("/v1/chat/completions", json=_chat(stream=True))

    assert response.status_code == 200
    frames = [
        line[len("data: ") :] for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert frames[-1] == "[DONE]"
    chunks = [json.loads(f) for f in frames[:-1]]
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "half an answer"
    assert chunks[-1]["choices"][0]["finish_reason"] is not None


# --------------------------------------------------------------------------- #
# The pairing -- the runtime decremented is the runtime incremented
# --------------------------------------------------------------------------- #


async def test_an_attempt_decrements_the_runtime_it_incremented() -> None:
    """A refresh landing mid-request on a node whose read failed makes
    the second lookup answer `None`, and the increment is never undone."""
    fake = _ParkingDriver(name=DRIVER, model_id=MODEL, runtime=RUNTIME)
    table = _one_ready_backend(fake)

    stream = _slot(fake, table=table).stream(_request())
    await stream.__anext__()
    assert table.runtime_inflight(RUNTIME_KEY) == 1

    # The node did not answer this refresh, so its runtime facts are gone.
    install_snapshot(table, fake)
    await stream.aclose()

    assert table.runtime_inflight(RUNTIME_KEY) == 0


async def test_a_finished_request_does_not_zero_a_counter_it_never_raised() -> None:
    """The mirror case, and the dangerous one: a request that starts
    while a node's facts are missing and ends after they come back
    decrements a counter it never incremented. `max(0, ...)` absorbs it,
    a live request reads as zero in flight, and the idle pass unloads a
    runtime mid-answer."""
    live = _ParkingDriver(name=DRIVER, model_id=MODEL, runtime=RUNTIME)
    # The same driver and the same runtime, on a request that finishes:
    # B has to end through the ORDINARY path, or this passes because of
    # the leak it sits beside rather than because of the pairing.
    quick = FakeDriverClient(name=DRIVER, model_id=MODEL, runtime=RUNTIME)
    quick.responses = ["done and dusted"]
    table = _one_ready_backend(live)

    # A is under way and counted.
    a = _slot(live, table=table).stream(_request())
    await a.__anext__()
    assert table.runtime_inflight(RUNTIME_KEY) == 1

    # The node's read fails; B starts against a snapshot with no facts.
    install_snapshot(table, live)
    b = _slot(quick, table=table).stream(_request())
    await b.__anext__()

    # The read succeeds again, and B runs to a clean finish.
    install_snapshot(table, live, runtimes=[runtime_facts(RUNTIME)])
    async for _ in b:
        pass

    assert table.runtime_inflight(RUNTIME_KEY) >= 1, (
        "request A is still streaming; reporting the runtime idle here is "
        "what unloads an engine mid-answer"
    )
    await a.aclose()


async def test_embed_pairs_its_counter_the_same_way() -> None:
    """`embed` is `generate`'s cascade over one model's replicas, and
    the same three lines. A fix applied to two of the three methods
    leaves the third leaking on every vector request."""

    class _SwappingEmbedder(FakeDriverClient):
        table: RoutingTable

        async def embed(self, request: Any) -> Any:
            install_snapshot(self.table, self)
            return await super().embed(request)

    swapping = _SwappingEmbedder(
        name=DRIVER, model_id=MODEL, runtime=RUNTIME, supports_embeddings=True
    )
    table = _one_ready_backend(swapping)
    swapping.table = table

    await _slot(swapping, table=table).embed(EmbedRequest(input=["one"]))

    assert table.runtime_inflight(RUNTIME_KEY) == 0
    assert table._runtime_last_request.get(RUNTIME_KEY) is not None


async def test_generate_pairs_its_counter_the_same_way() -> None:
    """`generate` and `embed` share the shape and so share the defect."""

    class _SwappingDriver(FakeDriverClient):
        table: RoutingTable

        async def generate(self, request: GenerateRequest) -> GenerateResponse:
            # A refresh lands while this attempt is in flight.
            install_snapshot(self.table, self)
            return await super().generate(request)

    swapping = _SwappingDriver(name=DRIVER, model_id=MODEL, runtime=RUNTIME)
    table = _one_ready_backend(swapping)
    swapping.table = table

    await _slot(swapping, table=table).generate(_request())

    assert table.runtime_inflight(RUNTIME_KEY) == 0
    assert table._runtime_last_request.get(RUNTIME_KEY) is not None, (
        "the attempt served; the runtime that served it is the one it started on"
    )


async def test_a_completed_stream_is_still_recorded_served() -> None:
    """The guard on the §6.3 #38 fix: `saw_done` must not turn every
    stream into a failure."""
    fake = FakeDriverClient(name=DRIVER, model_id=MODEL, runtime=RUNTIME)
    fake.responses = ["a complete answer"]
    table = _one_ready_backend(fake)
    slot = _slot(fake, table=table)

    with collect_attempts() as rows:
        async for _ in slot.stream(_request()):
            pass

    assert [row.served for row in rows] == [True]
    assert slot.served_by == DRIVER
    assert table._runtime_last_request.get(RUNTIME_KEY) is not None


def test_a_normal_streamed_completion_still_reports_stop(
    settings: Settings,
) -> None:
    """The route-level guard on the same fix."""
    fake = FakeDriverClient(name=DRIVER, model_id=MODEL, runtime=RUNTIME)
    fake.responses = ["a complete answer"]
    fake.usage = None
    app: FastAPI = create_app(settings=settings)
    app.state.routing = make_routing_table(fake, runtimes=[runtime_facts(RUNTIME)])

    with TestClient(app) as c:
        response = c.post("/v1/chat/completions", json=_chat(stream=True))

    frames = [
        line[len("data: ") :] for line in response.text.splitlines() if line.startswith("data: ")
    ]
    chunks = [json.loads(f) for f in frames[:-1]]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
