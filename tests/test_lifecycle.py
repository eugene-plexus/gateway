"""Idle unload, start on demand, and eviction — against a fake agent.

The agent is an httpx MockTransport that keeps a tiny runtime table:
stop and start flip status, a start on a runtime marked too big is a
422 with blockers, and `/v1/runtimes/{name}` reports the current state.
The routing table's refresh is stubbed to re-read that same fake, so a
wake that polls until `ready` then refreshes sees the runtime become
eligible the way it would against a real agent.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.lifecycle import AgentLifecycleClient, LifecycleManager
from eugene_plexus_gateway.routing import RoutingTable
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, install_snapshot, make_routing_table, runtime_facts

LOCAL = "qwen3-1.7b"
BIG = "qwen3-27b"


class FakeAgent:
    """Just enough of the agent's runtime surface for lifecycle policy."""

    def __init__(self, runtimes: dict[str, dict[str, Any]]) -> None:
        self.runtimes = runtimes
        self.calls: list[tuple[str, str, Any]] = []
        self.too_big: dict[str, int] = {}
        """Runtime name -> how many other runtimes may be running for it
        to fit. Its start is refused, with those runtimes as blockers,
        while more than that many are up."""
        self.start_becomes_ready_after: int = 0
        """How many status polls a started runtime spends `loading`."""
        self._pending: dict[str, int] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        parts = path.strip("/").split("/")
        body = json.loads(request.content) if request.content else None
        if path == "/v1/runtimes" and request.method == "GET":
            return httpx.Response(200, json={"runtimes": list(self.runtimes.values())})
        if path == "/v1/runtimes/admission":
            return httpx.Response(200, json=self._admission(body["name"]))
        if len(parts) == 4 and parts[3] == "stop":
            name = parts[2]
            self.calls.append(("stop", name, body))
            self.runtimes[name]["status"] = "stopped"
            self.runtimes[name]["stopReason"] = (body or {}).get("reason", "operator")
            return httpx.Response(202, json={"scheduled": True, "delayMs": 0})
        if len(parts) == 4 and parts[3] == "start":
            name = parts[2]
            self.calls.append(("start", name, None))
            admission = self._admission(name)
            if admission["decision"] == "refuse":
                return httpx.Response(
                    422,
                    json={"detail": {"title": "Admission refused", "detail": admission["reason"]}},
                )
            self.runtimes[name]["status"] = "loading"
            self.runtimes[name].pop("stopReason", None)
            self._pending[name] = self.start_becomes_ready_after
            return httpx.Response(202, json={"scheduled": True, "delayMs": 0})
        if len(parts) == 3 and parts[0] == "v1" and parts[1] == "runtimes":
            name = parts[2]
            runtime = self.runtimes.get(name)
            if runtime is None:
                return httpx.Response(404, json={"detail": "no such runtime"})
            if runtime["status"] == "loading":
                left = self._pending.get(name, 0)
                if left <= 0:
                    runtime["status"] = "ready"
                else:
                    self._pending[name] = left - 1
            return httpx.Response(200, json=runtime)
        if path == "/v1/components":
            return httpx.Response(200, json={"components": []})
        return httpx.Response(404, json={"detail": path})

    def _admission(self, name: str) -> dict[str, Any]:
        running = [
            r
            for n, r in self.runtimes.items()
            if n != name and r["status"] in ("ready", "loading", "starting")
        ]
        capacity = self.too_big.get(name)
        if capacity is not None and len(running) > capacity:
            blockers = [
                {
                    "name": r["name"],
                    "status": r["status"],
                    "idleUnloadSeconds": r.get("idleUnloadSeconds"),
                    "evictable": bool(r.get("idleUnloadSeconds")),
                }
                for r in running
            ]
            return {
                "decision": "refuse",
                "fit": "split",
                "basis": "file_size",
                "reason": f"refuse: {name} does not fit while {[b['name'] for b in blockers]} hold the device",
                "blockers": blockers,
            }
        return {
            "decision": "admit",
            "fit": "fits",
            "basis": "file_size",
            "reason": "admit",
            "blockers": [],
        }


def _runtime(name: str, alias: str, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "engine": "llama_cpp",
        "modelPath": f"/models/{alias}.gguf",
        "modelAlias": alias,
        "status": "ready",
        "url": "http://127.0.0.1:8090",
    }
    body.update(overrides)
    return body


def _facts_from(agent: FakeAgent):  # type: ignore[no-untyped-def]
    return [
        runtime_facts(
            r["name"],
            alias=r["modelAlias"],
            status=r["status"],
            idle_unload_seconds=r.get("idleUnloadSeconds"),
            start_on_demand=bool(r.get("startOnDemand")),
            stop_reason=r.get("stopReason"),
        )
        for r in agent.runtimes.values()
    ]


@pytest.fixture
def agent() -> FakeAgent:
    return FakeAgent(
        {
            "qwen3-a": _runtime("qwen3-a", LOCAL, idleUnloadSeconds=20, startOnDemand=True),
            "qwen3-b": _runtime("qwen3-b", LOCAL, idleUnloadSeconds=20, startOnDemand=True),
            "big": _runtime(
                "big", BIG, status="stopped", stopReason="autoStart", startOnDemand=True
            ),
        }
    )


@pytest.fixture
def drivers() -> dict[str, FakeDriverClient]:
    return {
        "qwen3-a-driver": FakeDriverClient(
            name="qwen3-a-driver", base_url="http://a", model_id=LOCAL, runtime="qwen3-a"
        ),
        "qwen3-b-driver": FakeDriverClient(
            name="qwen3-b-driver", base_url="http://b", model_id=LOCAL, runtime="qwen3-b"
        ),
        "big-driver": FakeDriverClient(
            name="big-driver", base_url="http://big", model_id=BIG, runtime="big"
        ),
    }


@pytest.fixture
def table(
    agent: FakeAgent, drivers: dict[str, FakeDriverClient], monkeypatch: pytest.MonkeyPatch
) -> RoutingTable:
    """A table whose refresh re-reads the fake agent's runtime table."""
    table = make_routing_table(
        *drivers.values(), runtimes=_facts_from(agent), agent_url="http://agent"
    )

    async def refresh() -> None:
        install_snapshot(table, *drivers.values(), runtimes=_facts_from(agent))

    monkeypatch.setattr(table, "refresh", refresh)
    return table


@pytest.fixture
def manager(agent: FakeAgent, table: RoutingTable) -> LifecycleManager:
    client = AgentLifecycleClient(service_token="t")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(agent.handler))
    return LifecycleManager(
        table,
        client=client,
        swap_wait_seconds=lambda: 5.0,
        idle_check_seconds=lambda: 15.0,
        poll_seconds=0.01,
    )


# --- idle unload -----------------------------------------------------------------


@pytest.mark.anyio
async def test_an_idle_runtime_past_its_timeout_is_stopped_with_reason_idle(
    agent: FakeAgent,
    table: RoutingTable,
    manager: LifecycleManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both replicas ready, both idle since "long ago".
    monkeypatch.setattr(table, "idle_seconds", lambda name: 100.0)
    stopped = await manager.idle_pass()
    assert sorted(stopped) == ["qwen3-a", "qwen3-b"]
    assert [c for c in agent.calls if c[0] == "stop"] == [
        ("stop", "qwen3-a", {"reason": "idle"}),
        ("stop", "qwen3-b", {"reason": "idle"}),
    ]
    assert agent.runtimes["qwen3-a"]["status"] == "stopped"
    # The refresh after unloading makes the table see them stopped.
    assert table.resolve(LOCAL).eligible_backends() == []
    assert manager.stopped_idle == ["qwen3-a", "qwen3-b"]


@pytest.mark.anyio
async def test_a_runtime_within_its_timeout_or_with_no_timeout_is_left_alone(
    agent: FakeAgent,
    table: RoutingTable,
    manager: LifecycleManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent.runtimes["qwen3-b"].pop("idleUnloadSeconds")
    await table.refresh()
    # `idle_seconds` takes a `(node, name)` key since R1.6.
    monkeypatch.setattr(
        table, "idle_seconds", lambda key: 10.0 if key[1] == "qwen3-a" else 10_000.0
    )
    assert await manager.idle_pass() == []
    assert agent.calls == []


@pytest.mark.anyio
async def test_a_runtime_with_a_request_in_flight_is_never_unloaded(
    agent: FakeAgent,
    table: RoutingTable,
    manager: LifecycleManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(table, "idle_seconds", lambda name: 100.0)
    table.on_attempt_start("qwen3-a-driver")
    assert await manager.idle_pass() == ["qwen3-b"]


@pytest.mark.anyio
async def test_a_runtime_that_never_served_counts_idle_from_when_it_became_ready(
    table: RoutingTable,
) -> None:
    # `install_snapshot` bypasses refresh, so mark readiness the way a
    # refresh would.
    import time

    table._ready_since[(None, "qwen3-a")] = time.perf_counter() - 50
    idle = table.idle_seconds((None, "qwen3-a"))
    assert idle is not None and 49 <= idle <= 60
    assert table.idle_seconds((None, "never-seen")) is None


# --- start on demand --------------------------------------------------------------


@pytest.mark.anyio
async def test_a_request_for_a_sleeping_model_wakes_it_and_is_served(
    settings: Settings,
    agent: FakeAgent,
    drivers: dict[str, FakeDriverClient],
    table: RoutingTable,
    manager: LifecycleManager,
) -> None:
    for name in ("qwen3-a", "qwen3-b"):
        agent.runtimes[name]["status"] = "stopped"
        agent.runtimes[name]["stopReason"] = "idle"
    await table.refresh()
    agent.start_becomes_ready_after = 3
    drivers["qwen3-a-driver"].responses = ["awake"]
    drivers["qwen3-b-driver"].responses = ["awake"]

    app = create_app(settings=settings)
    app.state.routing = table
    app.state.lifecycle = manager
    with TestClient(app) as c:
        response = c.post(
            "/v1/chat/completions",
            json={"model": LOCAL, "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "awake"
    info = body["x_eugene_plexus"]
    assert info["swapped_in"] is True
    assert info["waited_ms"] >= 0
    # The first startable runtime in tier order was woken, and only it.
    assert [c for c in agent.calls if c[0] == "start"] == [("start", "qwen3-a", None)]
    assert agent.runtimes["qwen3-a"]["status"] == "ready"
    assert agent.runtimes["qwen3-b"]["status"] == "stopped"
    assert info["runtime"] == "qwen3-a"


@pytest.mark.anyio
async def test_a_wake_that_does_not_reach_ready_in_time_is_a_503_naming_the_runtime(
    settings: Settings, agent: FakeAgent, table: RoutingTable, manager: LifecycleManager
) -> None:
    for name in ("qwen3-a", "qwen3-b"):
        agent.runtimes[name]["status"] = "stopped"
    await table.refresh()
    agent.start_becomes_ready_after = 10_000
    manager._swap_wait = lambda: 0.05  # type: ignore[assignment]

    app = create_app(settings=settings)
    app.state.routing = table
    app.state.lifecycle = manager
    with TestClient(app) as c:
        response = c.post(
            "/v1/chat/completions",
            json={"model": LOCAL, "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 503, response.text
    message = response.json()["error"]["message"]
    assert "qwen3-a" in message and "still loading" in message


@pytest.mark.anyio
async def test_a_stopped_model_that_did_not_opt_in_is_a_503_without_a_wake(
    settings: Settings, agent: FakeAgent, table: RoutingTable, manager: LifecycleManager
) -> None:
    for name in ("qwen3-a", "qwen3-b"):
        agent.runtimes[name]["status"] = "stopped"
        agent.runtimes[name]["startOnDemand"] = False
    await table.refresh()
    app = create_app(settings=settings)
    app.state.routing = table
    app.state.lifecycle = manager
    with TestClient(app) as c:
        response = c.post(
            "/v1/chat/completions",
            json={"model": LOCAL, "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 503
    assert "asked to be started on demand" in response.json()["error"]["message"]
    assert agent.calls == []


@pytest.mark.anyio
async def test_concurrent_requests_share_one_wake(
    agent: FakeAgent, table: RoutingTable, manager: LifecycleManager
) -> None:
    import asyncio

    for name in ("qwen3-a", "qwen3-b"):
        agent.runtimes[name]["status"] = "stopped"
    await table.refresh()
    agent.start_becomes_ready_after = 5
    resolution = table.resolve(LOCAL)
    results = await asyncio.gather(
        manager.wake(resolution), manager.wake(resolution), manager.wake(resolution)
    )
    assert all(r.ok for r in results)
    assert [c for c in agent.calls if c[0] == "start"] == [("start", "qwen3-a", None)]


# --- eviction ------------------------------------------------------------------------


@pytest.mark.anyio
async def test_a_refused_wake_evicts_idle_opted_in_runtimes_then_starts(
    agent: FakeAgent,
    table: RoutingTable,
    manager: LifecycleManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent.too_big["big"] = 1  # fits once one of the two replicas is gone
    # b has been idle longer than a; both opted in via idleUnloadSeconds.
    monkeypatch.setattr(
        table, "idle_seconds", lambda key: {"qwen3-a": 30.0, "qwen3-b": 300.0}.get(key[1], 0.0)
    )
    result = await manager.wake(table.resolve(BIG))
    assert result.ok, result.message
    # Most idle first; the fake admits as soon as one blocker is gone.
    assert result.evicted == ["qwen3-b"]
    stops = [c for c in agent.calls if c[0] == "stop"]
    assert stops == [("stop", "qwen3-b", {"reason": "idle"})]
    assert agent.runtimes["big"]["status"] == "ready"
    assert agent.runtimes["qwen3-a"]["status"] == "ready"


@pytest.mark.anyio
async def test_eviction_never_touches_a_runtime_without_a_timeout_or_with_traffic(
    agent: FakeAgent,
    table: RoutingTable,
    manager: LifecycleManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent.too_big["big"] = 0  # fits only with nothing else running
    agent.runtimes["qwen3-a"].pop("idleUnloadSeconds")  # never evictable
    await table.refresh()
    monkeypatch.setattr(table, "idle_seconds", lambda name: 1000.0)
    table.on_attempt_start("qwen3-b-driver")  # busy right now
    result = await manager.wake(table.resolve(BIG))
    assert result.ok is False
    assert result.evicted == []
    assert [c for c in agent.calls if c[0] == "stop"] == []
    assert "could not start runtime 'big'" in result.message


@pytest.mark.anyio
async def test_a_copying_runtime_is_coming_up_rather_than_nobody_asked_to_start(
    agent: FakeAgent, table: RoutingTable, manager: LifecycleManager
) -> None:
    """A node making its own local copy of a model is not stopped.

    `copying` (node-local-model-copy.md) is emitted for the minutes a
    node spends copying a 25 GB model to its own disk, with no process
    spawned yet. It is neither `stopped` — so nothing is startable —
    nor, before this, in the set `waking()` matched, so a request for
    that model fell through to "none of its runtimes asked to be started
    on demand": false, and unactionable for someone whose model is four
    minutes away.

    This is the gateway learning the status BEFORE any agent emits it:
    the gateway re-pins independently, so a newer agent against an older
    gateway is the ordinary case rather than the exotic one.
    """
    for name in ("qwen3-a", "qwen3-b"):
        agent.runtimes[name]["status"] = "copying"
    await table.refresh()

    result = await manager.wake(table.resolve(LOCAL))

    assert result.ok is False
    assert "still coming up" in result.message
    assert "(copying)" in result.message
    assert [c for c in agent.calls if c[0] == "start"] == []


@pytest.mark.anyio
async def test_a_status_this_build_has_never_heard_of_is_carried_through(
    agent: FakeAgent, table: RoutingTable, manager: LifecycleManager
) -> None:
    """Version skew is the normal state of a pinned polyrepo.

    The gateway does not codegen `agent.yaml`, so it parses `status` as
    text on purpose. An unknown value must not raise, must not make a
    runtime look ready, and must be reported as itself — the operator
    reading "is invented-state" learns more than one reading "unknown".
    """
    agent.runtimes["qwen3-a"]["status"] = "invented-state"
    agent.runtimes["qwen3-b"]["status"] = "invented-state"
    await table.refresh()

    resolution = table.resolve(LOCAL)
    assert [b for b in resolution.backends() if b.eligible] == []
    assert any("is invented-state" in (b.ineligible_reason or "") for b in resolution.backends())

    result = await manager.wake(resolution)
    assert result.ok is False
    assert "asked to be started on demand" in result.message
