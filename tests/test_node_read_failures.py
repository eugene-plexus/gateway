"""What a node that did not answer is allowed to mean.

Every refresh reads two things from every agent: `/v1/components` for
the drivers it declares and `/v1/runtimes` for what its engines are
doing. Both reads treated *failure* as *an answer*, in opposite and
equally wrong directions:

* `/v1/components` failing returned `[]`, so the refresh concluded every
  driver on that node had left the topology, closed their cached HTTP
  clients **under whatever requests were using them**, and un-routed the
  node's models until some later refresh put them back;
* `/v1/runtimes` failing returned `{}`, so every driver on that node had
  `runtime is None` -- which the eligibility rule reads as "follows
  nothing of ours, route to it whenever it is reachable". A stopped, a
  loading and a crashed engine all became routable.

One 401 from clock skew, one 5 s timeout, or the agent restart the Reach
switch performs on purpose is enough for either. The control root's
read, eighty lines away, already kept the previous map on a failure.

The third case here is the idle-unload race: `idle_pass` and
`_evict_for` check `runtime_inflight == 0`, then **await** an HTTP call
to stop the engine, and a request arriving inside that await is routed
to a runtime that is being shut down. Until R1.4 a leaked in-flight
counter was what accidentally kept runtimes out of the idle pass; with
the counter honest, this is reachable.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import httpx
import pytest

from eugene_plexus_gateway.lifecycle import AgentLifecycleClient, LifecycleManager
from eugene_plexus_gateway.routing import RoutingTable

AGENT = "http://agent-a:8079"
DRIVER_URL = "http://127.0.0.1:8090"
MODEL = "qwen3-1.7b"
DRIVER = "qwen-driver"
RUNTIME = "qwen"
#: Every demand counter is keyed by `(node, name)` since R1.6: a
#: driver or runtime NAME is unique per agent, not per install. These
#: fixtures are single-host, so the node half is `None`.
DRIVER_KEY = (None, DRIVER)
RUNTIME_KEY = (None, RUNTIME)


class OneAgent:
    """One agent, one companion driver, one runtime — and switches for
    the two reads to fail the way a real one fails."""

    def __init__(self) -> None:
        self.components_error: Exception | None = None
        self.components_status = 200
        self.runtimes_error: Exception | None = None
        self.runtimes_status = 200
        self.declared: list[dict[str, Any]] = [
            {
                "name": DRIVER,
                "kind": "inference-driver",
                "url": DRIVER_URL,
                "status": "running",
            }
        ]
        self.runtime: dict[str, Any] = {
            "name": RUNTIME,
            "engine": "llama_cpp",
            "modelPath": "/m/qwen.gguf",
            "modelAlias": MODEL,
            "status": "ready",
            "url": DRIVER_URL,
            "idleUnloadSeconds": 10,
            "startOnDemand": True,
        }
        self.probes = 0
        self.lifecycle: list[tuple[str, str]] = []
        """`(action, runtime)` for every stop/start the gateway asked for."""
        self.on_stop: Any = None
        """Called inside the stop handler, before it answers — the seam
        the race needs. A request arriving *during* the stop is what
        `idle_pass` leaves a window for."""
        self.stop_status = 202
        self.admission: dict[str, Any] = {
            "decision": "admit",
            "fit": "fits",
            "reason": "ok",
            "blockers": [],
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        host, port, path = request.url.host, request.url.port, request.url.path

        if host == "agent-a" and port == 8079:
            if path == "/v1/node":
                return httpx.Response(200, json={"enrolled": False, "name": None})
            if path == "/v1/components":
                if self.components_error is not None:
                    raise self.components_error
                if self.components_status != 200:
                    return httpx.Response(self.components_status, json={"detail": "nope"})
                return httpx.Response(200, json={"components": list(self.declared)})
            if path == "/v1/runtimes":
                if self.runtimes_error is not None:
                    raise self.runtimes_error
                if self.runtimes_status != 200:
                    return httpx.Response(self.runtimes_status, json={"detail": "nope"})
                return httpx.Response(200, json={"runtimes": [dict(self.runtime)]})
            if path == "/v1/runtimes/admission":
                return httpx.Response(200, json=self.admission)
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[1] == "runtimes" and parts[3] in ("stop", "start"):
                self.lifecycle.append((parts[3], parts[2]))
                if self.on_stop is not None and parts[3] == "stop":
                    self.on_stop()
                if self.stop_status >= 400 and parts[3] == "stop":
                    return httpx.Response(self.stop_status, json={"detail": "refused"})
                self.runtime["status"] = "stopped" if parts[3] == "stop" else "ready"
                return httpx.Response(202, json={"scheduled": True, "delayMs": 0})
            return httpx.Response(404, json={"detail": path})

        if path == "/v1/info":
            self.probes += 1
            return httpx.Response(
                200,
                json={
                    "backend": "openai_compat_http",
                    "version": "0.1.0",
                    "modelId": MODEL,
                    "runtime": RUNTIME,
                },
            )
        if path == "/v1/generate":
            return httpx.Response(
                200,
                json={
                    "content": "hi",
                    "finishReason": "stop",
                    "backend": "openai_compat_http",
                    "modelId": MODEL,
                    "latencyMs": 1,
                },
            )
        return httpx.Response(404, json={"detail": f"{host}:{port}{path}"})


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch) -> OneAgent:
    fake = OneAgent()
    real_init = httpx.AsyncClient.__init__

    def init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = httpx.MockTransport(fake.handler)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
    return fake


async def _table() -> RoutingTable:
    table = RoutingTable(agent_url=AGENT, refresh_seconds=3600)
    await table.refresh()
    return table


def _manager(table: RoutingTable) -> LifecycleManager:
    return LifecycleManager(
        table,
        client=AgentLifecycleClient(service_token=None),
        swap_wait_seconds=lambda: 5.0,
        idle_check_seconds=lambda: 3600.0,
        poll_seconds=0.01,
    )


# --------------------------------------------------------------------------- #
# §6.1 #9 — a failed `/v1/components` read
# --------------------------------------------------------------------------- #


async def test_a_failed_components_read_keeps_the_node_routable(agent: OneAgent) -> None:
    table = await _table()
    assert [b.name for b in table.backends_for(MODEL)] == [DRIVER]

    agent.components_error = httpx.ConnectError("connection refused")
    await table.refresh()

    assert [b.name for b in table.backends_for(MODEL)] == [DRIVER], (
        "an agent that did not answer is not an agent with nothing on it"
    )


async def test_a_failed_components_read_does_not_close_a_client_in_use(agent: OneAgent) -> None:
    """The sharp end of §6.1 #9. The refresh closes the HTTP clients of
    drivers it believes have left — and a request already dispatched is
    holding one. Closing it turns an in-flight completion into a
    transport error."""
    table = await _table()
    backend = table.backends_for(MODEL)[0]
    in_flight = backend.client  # what a dispatched request is holding

    agent.components_error = httpx.ConnectError("connection refused")
    await table.refresh()

    # Not "the dict still has an entry" — the object a request is using
    # still works. A closed httpx client raises on its next send.
    info = await in_flight.info()
    assert info.modelId == MODEL


async def test_a_successful_read_that_drops_a_driver_still_closes_its_client(
    agent: OneAgent,
) -> None:
    """The guard. Keeping clients for a node that did not answer must not
    turn into keeping them forever: an agent that answers, and no longer
    declares this driver, has really removed it."""
    table = await _table()
    # `(node, name, url)` since R1.6; single-host, so the node is None.
    key = (None, DRIVER, DRIVER_URL)
    assert key in table._clients

    agent.declared = []
    await table.refresh()

    assert key not in table._clients
    assert table.backends_for(MODEL) == []


async def test_a_5xx_components_read_is_a_failure_like_any_other(agent: OneAgent) -> None:
    """A 500 from an agent that is up but broken is not a declaration
    that it supervises nothing."""
    table = await _table()
    agent.components_status = 503
    await table.refresh()
    assert [b.name for b in table.backends_for(MODEL)] == [DRIVER]


# --------------------------------------------------------------------------- #
# §6.2 #18 — a failed `/v1/runtimes` read
# --------------------------------------------------------------------------- #


async def test_a_failed_runtimes_read_keeps_the_previous_facts(agent: OneAgent) -> None:
    """`runtime is None` means "follows nothing of ours, always
    eligible". A failed read must not be spelled that way: the engine is
    stopped and was stopped a second ago."""
    agent.runtime["status"] = "stopped"
    table = await _table()
    assert table.backends_for(MODEL)[0].eligible is False

    agent.runtimes_error = httpx.ReadTimeout("timed out")
    await table.refresh()

    backend = table.backends_for(MODEL)[0]
    assert backend.eligible is False, (
        "a stopped engine became routable because its agent missed one read"
    )
    assert backend.runtime is not None
    assert backend.runtime.status == "stopped"


async def test_a_failed_runtimes_read_does_not_make_a_loading_engine_routable(
    agent: OneAgent,
) -> None:
    agent.runtime["status"] = "loading"
    table = await _table()
    assert table.backends_for(MODEL)[0].eligible is False

    agent.runtimes_error = httpx.ConnectError("connection refused")
    await table.refresh()
    assert table.backends_for(MODEL)[0].eligible is False


async def test_a_401_from_clock_skew_is_a_failed_read(agent: OneAgent) -> None:
    """The trigger this install has actually produced: half a second of
    clock skew, every token refused, and a worker that reads as having
    nothing running."""
    agent.runtime["status"] = "stopped"
    table = await _table()
    agent.runtimes_status = 401
    await table.refresh()
    assert table.backends_for(MODEL)[0].eligible is False


async def test_the_facts_return_when_the_agent_answers_again(agent: OneAgent) -> None:
    agent.runtime["status"] = "stopped"
    table = await _table()
    agent.runtimes_error = httpx.ConnectError("refused")
    await table.refresh()

    agent.runtimes_error = None
    agent.runtime["status"] = "ready"
    await table.refresh()
    assert table.backends_for(MODEL)[0].eligible is True


async def test_a_successful_read_that_drops_a_runtime_keeps_the_faith_rule(
    agent: OneAgent,
) -> None:
    """The boundary, pinned deliberately. A driver naming a runtime no
    agent reports **while answering** is the pre-existing case: routable
    on faith, logged at DEBUG. That is a different question from a read
    that failed, and this slice does not change it."""
    table = await _table()
    agent.runtime = {"name": "something-else", "status": "ready", "url": DRIVER_URL}
    await table.refresh()

    backend = table.backends_for(MODEL)[0]
    assert backend.runtime is None
    assert backend.eligible is True


# --------------------------------------------------------------------------- #
# Two nodes — the shape the caches are keyed for
# --------------------------------------------------------------------------- #


CONTROL = "http://control:8083"
AGENT_B = "http://agent-b:8079"
MODEL_B = "gemma-4-e4b"


class TwoNodes:
    """A control root listing two nodes, each with one driver and one
    runtime, and a switch for node A's reads to fail."""

    def __init__(self) -> None:
        self.a_fails = False
        self.nodes = ["node-a", "node-b"]

    def handler(self, request: httpx.Request) -> httpx.Response:
        host, port, path = request.url.host, request.url.port, request.url.path
        if host == "control":
            if path == "/v1/nodes":
                listed = [
                    {"name": n, "url": AGENT if n == "node-a" else AGENT_B, "reachable": True}
                    for n in self.nodes
                ]
                return httpx.Response(200, json={"nodes": listed})
            return httpx.Response(404, json={"detail": path})

        if port == 8079:
            node = "node-a" if host == "agent-a" else "node-b"
            if node == "node-a" and self.a_fails and path in ("/v1/components", "/v1/runtimes"):
                raise httpx.ConnectError("connection refused")
            if path == "/v1/node":
                return httpx.Response(200, json={"enrolled": True, "name": node})
            if path == "/v1/components":
                return httpx.Response(
                    200,
                    json={
                        "components": [
                            {
                                "name": f"{node}-driver",
                                "kind": "inference-driver",
                                "url": f"http://{host}:8090",
                                "status": "running",
                            }
                        ]
                    },
                )
            if path == "/v1/runtimes":
                return httpx.Response(
                    200,
                    json={
                        "runtimes": [
                            {
                                "name": f"{node}-runtime",
                                "status": "stopped",
                                "url": f"http://{host}:8090",
                                "node": node,
                            }
                        ]
                    },
                )
            return httpx.Response(404, json={"detail": path})

        if path == "/v1/info":
            model = MODEL if host == "agent-a" else MODEL_B
            node = "node-a" if host == "agent-a" else "node-b"
            return httpx.Response(
                200,
                json={
                    "backend": "openai_compat_http",
                    "version": "0.1.0",
                    "modelId": model,
                    "runtime": f"{node}-runtime",
                },
            )
        return httpx.Response(404, json={"detail": f"{host}:{port}{path}"})


@pytest.fixture
def two(monkeypatch: pytest.MonkeyPatch) -> TwoNodes:
    fake = TwoNodes()
    real_init = httpx.AsyncClient.__init__

    def init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = httpx.MockTransport(fake.handler)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
    return fake


async def test_one_nodes_failed_read_does_not_reach_the_other(two: TwoNodes) -> None:
    """The caches are per node, and that is the whole reason they are.
    A node whose agent is down keeps its own last answer; the node beside
    it keeps reading fresh, and neither inherits the other's."""
    table = RoutingTable(agent_url=AGENT, control_url=CONTROL, refresh_seconds=3600)
    await table.refresh()
    assert not table.backends_for(MODEL)[0].eligible
    assert not table.backends_for(MODEL_B)[0].eligible

    two.a_fails = True
    await table.refresh()

    assert [b.name for b in table.backends_for(MODEL)] == ["node-a-driver"]
    assert not table.backends_for(MODEL)[0].eligible, "A's last-known facts, not faith"
    assert not table.backends_for(MODEL_B)[0].eligible, "B answered; nothing of A's reached it"


async def test_a_node_that_leaves_the_install_keeps_nothing(two: TwoNodes) -> None:
    """Keeping what a node last said must not become keeping it forever.
    A read that failed is a node that did not answer; a node the control
    root no longer lists has left, and its cache goes with it."""
    table = RoutingTable(agent_url=AGENT, control_url=CONTROL, refresh_seconds=3600)
    await table.refresh()
    assert "node-b" in table._last_entries
    assert "node-b" in table._last_facts

    two.nodes = ["node-a"]
    await table.refresh()

    assert "node-b" not in table._last_entries
    assert "node-b" not in table._last_facts


# --------------------------------------------------------------------------- #
# §6.2 #17 — the idle-unload race
# --------------------------------------------------------------------------- #


async def test_a_runtime_being_stopped_is_not_routed_to(agent: OneAgent) -> None:
    table = await _table()
    assert table.backends_for(MODEL)[0].eligible is True

    with table.stopping(RUNTIME_KEY):
        backend = table.backends_for(MODEL)[0]
        assert backend.eligible is False
        assert backend.ineligible_reason is not None
        assert "stop" in backend.ineligible_reason

    assert table.backends_for(MODEL)[0].eligible is True


async def test_two_overlapping_stops_do_not_release_each_other(agent: OneAgent) -> None:
    """A count, not a flag. `idle_pass` and an eviction can both be
    stopping the same runtime, and the inner span finishing must not
    un-reserve it for the outer one."""
    table = await _table()
    with table.stopping(RUNTIME_KEY):
        with table.stopping(RUNTIME_KEY):
            assert table.is_stopping(RUNTIME_KEY)
        assert table.is_stopping(RUNTIME_KEY)
    assert not table.is_stopping(RUNTIME_KEY)


async def test_the_idle_pass_reserves_the_runtime_across_the_stop_call(
    agent: OneAgent,
) -> None:
    """Check-await-act. The check is `runtime_inflight == 0`; the act is
    an HTTP round trip to the agent; and a request arriving in between
    was routed to an engine that is going away. On the streamed path
    M10's commit point turns that from a retry into a visible
    truncation."""
    table = await _table()
    # Idle since it became ready, and past its timeout.
    table._ready_since[RUNTIME_KEY] = table._ready_since[RUNTIME_KEY] - 3600

    eligible_during: list[bool] = []
    agent.on_stop = lambda: eligible_during.append(
        any(b.eligible for b in table.backends_for(MODEL))
    )

    stopped = await _manager(table).idle_pass()

    assert stopped == [RUNTIME]
    assert eligible_during == [False], (
        "a request arriving while the stop was in flight would have been "
        "routed to the engine being stopped"
    )


async def test_an_eviction_reserves_each_victim(agent: OneAgent) -> None:
    """`_evict_for`'s `for _ in range(8)` reopens the same window up to
    eight times, on a path a user is actively waiting on."""
    table = await _table()
    manager = _manager(table)
    agent.admission = {
        "decision": "refuse",
        "fit": "no",
        "reason": "not enough memory",
        "blockers": [{"name": RUNTIME, "evictable": True}],
    }

    eligible_during: list[bool] = []
    agent.on_stop = lambda: eligible_during.append(
        any(b.eligible for b in table.backends_for(MODEL))
    )

    facts = next(r for r in table.runtimes() if r.name == RUNTIME)
    evicted = await manager._evict_for(facts, AGENT)

    assert evicted == [RUNTIME]
    assert eligible_during == [False]


async def test_a_stop_that_fails_releases_the_reservation(agent: OneAgent) -> None:
    """A reservation is a promise that something is about to happen. The
    agent refusing means it is not, and the runtime has to go back to
    being routable — otherwise one 500 removes it from the install."""
    table = await _table()
    table._ready_since[RUNTIME_KEY] = table._ready_since[RUNTIME_KEY] - 3600
    agent.stop_status = 500

    stopped = await _manager(table).idle_pass()

    assert stopped == []
    assert not table.is_stopping(RUNTIME_KEY)
    assert table.backends_for(MODEL)[0].eligible is True


async def test_a_stop_that_raises_releases_the_reservation(agent: OneAgent) -> None:
    """The `finally`, not the happy path. A reservation that survives an
    exception removes a healthy runtime from the install for good.

    **The exception has to be one `stop()` does not catch**, which is
    the thing the first version of this check got wrong: `stop()`
    catches every `httpx.HTTPError` and returns False, so a transport
    failure never reaches the `with` block at all and a `finally`
    removed from the reservation escaped the sabotage that removed it.
    The real trigger is the idle loop's task being cancelled mid-stop at
    shutdown — a `BaseException`, caught by nothing on the way out."""
    table = await _table()
    table._ready_since[RUNTIME_KEY] = table._ready_since[RUNTIME_KEY] - 3600

    def boom() -> None:
        raise asyncio.CancelledError

    agent.on_stop = boom
    with contextlib.suppress(asyncio.CancelledError):
        await _manager(table).idle_pass()

    assert not table.is_stopping(RUNTIME_KEY)


async def test_a_stop_whose_transport_fails_also_releases_it(agent: OneAgent) -> None:
    """And the ordinary case, which takes a different path out:
    `stop()` turns an `httpx.HTTPError` into `False` and the `with`
    block exits normally."""
    table = await _table()
    table._ready_since[RUNTIME_KEY] = table._ready_since[RUNTIME_KEY] - 3600

    def refused() -> None:
        raise httpx.ConnectError("the agent went away mid-stop")

    agent.on_stop = refused
    assert await _manager(table).idle_pass() == []

    assert not table.is_stopping(RUNTIME_KEY)
    assert table.backends_for(MODEL)[0].eligible is True


async def test_the_idle_pass_leaves_no_reservation_behind(agent: OneAgent) -> None:
    """The span is the stop call and nothing longer. After it the runtime
    is ineligible for the honest reason — its status — not because a flag
    was left set."""
    table = await _table()
    table._ready_since[RUNTIME_KEY] = table._ready_since[RUNTIME_KEY] - 3600

    assert await _manager(table).idle_pass() == [RUNTIME]

    assert not table.is_stopping(RUNTIME_KEY)
    backend = table.backends_for(MODEL)[0]
    assert backend.eligible is False
    assert backend.runtime is not None
    assert backend.runtime.status == "stopped"
