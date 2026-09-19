"""Two agents behind one control root — the fan-out M6 wired and never
saw two of.

M6 §11's first departure made the gateway read the control root's
`/v1/nodes` and then each node's agent directly. Every test of it ran
with one node. These run with two, in-process, dispatching on the host
in each request's URL, and assert the three things a two-host install
turns on: runtimes from every node land in one table tagged with the
node that reported them; a component on another node is probed where
its agent says peers reach it and not at the loopback address it binds;
and a lifecycle action for a runtime on node B goes to node B's agent,
never to the default one.

Fakes rather than sockets because "which host answered" is the thing
under test and a MockTransport can see it directly; the live two-agent
run (`specs/scripts/m7-acceptance.sh`) repeats the last assertion with
two real agents on two ports.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import pytest

from eugene_plexus_gateway.lifecycle import AgentLifecycleClient, LifecycleManager
from eugene_plexus_gateway.routing import RoutingTable

CONTROL = "http://control:8083"
AGENT_A = "http://agent-a:8079"
AGENT_B = "http://agent-b:8079"
MODEL = "qwen3-1.7b"
#: What the UI really names them: the runtime after the model, the
#: companion driver after the runtime. The same two strings on every
#: node that runs this model, which is the whole of review §6.1 #8.
RUNTIME = "qwen"
DRIVER = "qwen-driver"


def _info(model_id: str, runtime: str | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "backend": "openai_compat_http",
        "version": "0.1.0",
        "modelId": model_id,
    }
    if runtime is not None:
        body["runtime"] = runtime
    return body


class TwoAgents:
    """A control root that lists two nodes, and the two agents it lists.

    Node B's companion binds loopback (`url`) and is advertised at
    `agent-b:8091` (`advertiseUrl`), exactly as a real remote agent
    reports it. Loopback on port 8091 is **not** reachable from this
    gateway — that is the M6 gap — so a probe sent there is a connection
    error, and the test can tell which address the gateway used.
    """

    def __init__(self, *, advertise: bool = True) -> None:
        self.probed: list[str] = []
        self.lifecycle: list[tuple[str, str, str, Any]] = []
        """`(agent host, action, runtime, body)` for every stop/start."""
        self.control_status = 200
        # What agent A's own `/v1/node` says: enrolled to this root, or not
        # enrolled. And whether A's topology declares a `control` component.
        # Both off by default so every test written before the gateway
        # derived its control root still sees the single-host world it
        # was written for.
        self.enrolled_to: str | None = None
        self.declares_control = False
        # When true, `/v1/info` names no runtime -- a hand-written
        # `baseUrl`, or a driver older than M4 -- so the table has to
        # fall back to the alias, per node.
        self.drop_runtime_from_info = False
        # **The names are IDENTICAL on both nodes, and that is the
        # reproduction** (R1.6, review §6.1 #8). This fixture used to say
        # `qwen-a` / `qwen-b`, which no install produces: the UI names a
        # runtime after the model and a companion driver after the
        # runtime, so one model launched on two machines really is `qwen`
        # and `qwen-driver` twice. The test written for the scenario had
        # been passing on a shape that cannot occur.
        #
        # Keyed by `(node, name)` here for the same reason the gateway
        # must be: a dict keyed by name alone cannot hold both.
        self.runtimes: dict[tuple[str, str], dict[str, Any]] = {
            ("node-a", RUNTIME): {
                "name": RUNTIME,
                "engine": "llama_cpp",
                "modelPath": "/m/qwen.gguf",
                "modelAlias": MODEL,
                "status": "ready",
                "url": "http://127.0.0.1:8090",
                "node": "node-a",
            },
            ("node-b", RUNTIME): {
                "name": RUNTIME,
                "engine": "llama_cpp",
                "modelPath": "/m/qwen.gguf",
                "modelAlias": MODEL,
                "status": "ready",
                "url": "http://127.0.0.1:8090",
                "node": "node-b",
                "idleUnloadSeconds": 10,
                "startOnDemand": True,
            },
        }
        b_component: dict[str, Any] = {
            "name": DRIVER,
            "kind": "inference-driver",
            "url": "http://127.0.0.1:8091",
            "status": "running",
        }
        if advertise:
            b_component["advertiseUrl"] = "http://agent-b:8091"
        self.components = {
            "agent-a": [
                {
                    "name": DRIVER,
                    "kind": "inference-driver",
                    "url": "http://127.0.0.1:8090",
                    "status": "running",
                }
            ],
            "agent-b": [b_component],
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        host, port, path = request.url.host, request.url.port, request.url.path
        body = json.loads(request.content) if request.content else None

        if host == "control":
            if path == "/v1/nodes":
                if self.control_status != 200:
                    # The shape a sealed control root really sends: FastAPI
                    # wraps the Problem under `detail`.
                    return httpx.Response(
                        self.control_status,
                        json={
                            "detail": {
                                "type": "https://github.com/eugene-plexus/control#locked",
                                "title": "Locked",
                                "status": self.control_status,
                                "detail": "control is having a moment",
                            }
                        },
                    )
                return httpx.Response(
                    200,
                    json={
                        "nodes": [
                            # A trailing slash, the way Pydantic renders a URL.
                            {
                                "name": "node-a",
                                "url": AGENT_A + "/",
                                "role": "control",
                                "reachable": True,
                            },
                            {"name": "node-b", "url": AGENT_B, "role": "agent", "reachable": True},
                        ]
                    },
                )
            return httpx.Response(404, json={"detail": path})

        if host in ("agent-a", "agent-b") and port == 8079:
            node = "node-a" if host == "agent-a" else "node-b"
            if path == "/v1/node":
                enrolled = host == "agent-a" and self.enrolled_to is not None
                return httpx.Response(
                    200,
                    json={
                        "enrolled": enrolled,
                        "name": node if enrolled else None,
                        "controlUrl": self.enrolled_to if enrolled else None,
                    },
                )
            if path == "/v1/components":
                components = list(self.components[host])
                if host == "agent-a" and self.declares_control:
                    components.append(
                        {"name": "control", "kind": "control", "url": CONTROL, "status": "running"}
                    )
                return httpx.Response(200, json={"components": components})
            if path == "/v1/runtimes":
                mine = [r for r in self.runtimes.values() if r["node"] == node]
                return httpx.Response(200, json={"runtimes": mine})
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[1] == "runtimes" and parts[3] in ("stop", "start"):
                name = parts[2]
                self.lifecycle.append((host, parts[3], name, body))
                # **An agent knows only its own runtimes**, which is what
                # makes a misdirected lifecycle action visible: a stop for
                # node A's replica sent to node B's agent is a 404 here,
                # exactly as it is on a real host.
                runtime = self.runtimes.get((node, name))
                if runtime is None:
                    return httpx.Response(404, json={"detail": f"{name} is not on this node"})
                runtime["status"] = "stopped" if parts[3] == "stop" else "ready"
                if parts[3] == "stop":
                    runtime["stopReason"] = (body or {}).get("reason", "operator")
                else:
                    runtime.pop("stopReason", None)
                return httpx.Response(202, json={"scheduled": True, "delayMs": 0})
            if len(parts) == 3 and parts[1] == "runtimes":
                runtime = self.runtimes.get((node, parts[2]))
                if runtime is None:
                    return httpx.Response(404, json={"detail": "no such runtime here"})
                return httpx.Response(200, json=runtime)
            if path == "/v1/runtimes/admission":
                return httpx.Response(
                    200, json={"decision": "admit", "fit": "fits", "reason": "ok", "blockers": []}
                )
            return httpx.Response(404, json={"detail": path})

        if path == "/v1/info":
            self.probed.append(f"{host}:{port}")
            # Both drivers name the same runtime, because both really
            # do: `RuntimeSpec.name` is unique per node, not per install.
            named = None if self.drop_runtime_from_info else RUNTIME
            if (host, port) == ("127.0.0.1", 8090):
                return httpx.Response(200, json=_info(MODEL, named))
            if (host, port) == ("agent-b", 8091):
                return httpx.Response(200, json=_info(MODEL, named))
            # Loopback:8091 is node B's driver as node B binds it. From this
            # gateway, on another host, there is nothing there.
            raise httpx.ConnectError("connection refused", request=request)

        return httpx.Response(404, json={"detail": f"{host}:{port}{path}"})


@pytest.fixture
def route_http(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Route every httpx.AsyncClient in the process through one handler —
    the same seam test_routing.py uses, local to this module because a
    fixture defined in a test module is not shared."""

    def install(handler: Any, *, handle_runtimes: bool = False) -> None:
        real_init = httpx.AsyncClient.__init__

        def init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = httpx.MockTransport(handler)
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", init)

    return install


@pytest.fixture
def two(route_http: Any) -> TwoAgents:
    fake = TwoAgents()
    route_http(fake.handler, handle_runtimes=True)
    return fake


async def _table() -> RoutingTable:
    table = RoutingTable(agent_url=AGENT_A, control_url=CONTROL, refresh_seconds=3600)
    await table.refresh()
    return table


def _nodes_for(table: RoutingTable, model: str = MODEL) -> list[str | None]:
    """Which NODES serve this model, since the driver names no longer
    tell them apart. Every assertion in this file that used to read a
    list of distinct driver names reads this instead -- and that is the
    point: the identity of a replica is `(node, name)`, and a test that
    can only see `name` cannot tell a two-node install from a one-node
    one."""
    return sorted(
        (b.node for b in table.backends_for(model)),
        key=lambda n: (n is None, n or ""),
    )


def _facts_by_node(table: RoutingTable) -> dict[str | None, object]:
    return {r.node: r for r in table.runtimes()}


# --------------------------------------------------------------------------- #
# fan-out and attribution
# --------------------------------------------------------------------------- #


async def test_the_table_fans_out_over_every_node_the_control_root_lists(two: TwoAgents) -> None:
    """Two agents, one alias, one table: the replica set differentiator #7
    describes as 'across two GPUs' when the GPUs are on two hosts."""
    table = await _table()

    backends = table.backends_for(MODEL)
    assert [b.name for b in backends] == [DRIVER, DRIVER], "one name, two machines"
    assert _nodes_for(table) == ["node-a", "node-b"]
    assert all(b.eligible for b in backends)

    # **TWO runtimes, not one.** Merged by bare name this was a single
    # entry whose node was whichever agent answered last, and every
    # lifecycle action for the loser went to the wrong host.
    assert len(table.runtimes()) == 2
    facts = _facts_by_node(table)
    assert set(facts) == {"node-a", "node-b"}
    assert all(r.name == RUNTIME for r in facts.values())

    # The agent map is keyed by node, trailing slash trimmed, and the
    # default agent is no longer the only one.
    assert table.agent_url_for(facts["node-a"]) == AGENT_A
    assert table.agent_url_for(facts["node-b"]) == AGENT_B

    view = table.as_routing_view()
    rows = [b for s in view.slots for t in s.tiers for b in t.backends]
    assert sorted(b.node or "" for b in rows) == ["node-a", "node-b"]
    assert all(b.driver == DRIVER for b in rows)
    await table.aclose()


async def test_a_remote_component_is_probed_where_its_agent_says_peers_reach_it(
    two: TwoAgents,
) -> None:
    """The M6 gap, closed: node B's companion binds loopback, and from this
    host loopback:8091 is nothing. The gateway probes the advertise URL and
    never the bind URL."""
    table = await _table()
    assert "agent-b:8091" in two.probed
    assert "127.0.0.1:8091" not in two.probed
    assert [u.name for u in table.as_driver_health() if not u.reachable] == []
    await table.aclose()


async def test_without_an_advertise_url_the_remote_companion_is_unreachable(
    route_http: Any,
) -> None:
    """What the first two-host run would have met before M7, kept as a
    test so the gap cannot quietly reopen: the same topology with no
    `advertiseUrl` on node B's driver is probed at loopback and lost."""
    fake = TwoAgents(advertise=False)
    route_http(fake.handler, handle_runtimes=True)
    table = RoutingTable(agent_url=AGENT_A, control_url=CONTROL, refresh_seconds=3600)
    await table.refresh()
    assert "127.0.0.1:8091" in fake.probed
    unreachable = [u.name for u in table.as_driver_health() if not u.reachable]
    assert unreachable == [DRIVER]
    # A control root IS configured here, so both nodes are named; only
    # node A's driver is reachable.
    assert _nodes_for(table) == ["node-a"]
    await table.aclose()


async def test_a_control_root_that_stops_answering_keeps_the_agent_map(two: TwoAgents) -> None:
    """Management being down must not empty the routing table — M5's
    surviving-data-path guarantee, at the node-discovery step."""
    table = await _table()
    two.control_status = 503
    await table.refresh()
    facts = _facts_by_node(table)
    assert set(facts) == {"node-a", "node-b"}
    assert table.agent_url_for(facts["node-b"]) == AGENT_B
    await table.aclose()


# --------------------------------------------------------------------------- #
# lifecycle actions cross hosts
# --------------------------------------------------------------------------- #


def _manager(table: RoutingTable) -> LifecycleManager:
    return LifecycleManager(
        table,
        client=AgentLifecycleClient(None),
        swap_wait_seconds=lambda: 5.0,
        idle_check_seconds=lambda: 15.0,
        poll_seconds=0.01,
    )


async def test_an_idle_unload_goes_to_the_agent_that_owns_the_runtime(
    two: TwoAgents, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Node B's replica declared an idle timeout; node A's did not. The
    stop must reach node B's agent and name node B's replica.

    **Both replicas are called `qwen`**, so this is the assertion the
    merge broke: node A's entry was the one the map kept, its node was
    `node-a`, and the unload went to node A's agent -- which stopped a
    runtime whose operator never asked for one and left the idle one
    serving. Worse on the idle clock itself: one `_runtime_last_request`
    entry for two replicas means node B's traffic keeps node A's looking
    recently used, and the idle pass iterates the merged map so one of
    the two is never considered at all.
    """
    table = await _table()
    manager = _manager(table)
    monkeypatch.setattr(table, "idle_seconds", lambda _key: 100.0)

    stopped = await manager.idle_pass()
    assert stopped == [RUNTIME]
    assert two.lifecycle == [("agent-b", "stop", RUNTIME, {"reason": "idle"})]
    assert two.runtimes[("node-b", RUNTIME)]["status"] == "stopped"
    assert two.runtimes[("node-a", RUNTIME)]["status"] == "ready", "no timeout, never touched"
    await manager.aclose()
    await table.aclose()


async def test_a_wake_goes_to_the_agent_that_owns_the_runtime(two: TwoAgents) -> None:
    """Both replicas asleep; only node B's asked to be woken. The start goes
    to node B's agent and the runtime is polled there."""
    two.runtimes[("node-a", RUNTIME)]["status"] = "stopped"
    two.runtimes[("node-b", RUNTIME)]["status"] = "stopped"
    table = await _table()
    manager = _manager(table)
    assert table.resolve(MODEL).eligible_backends() == []

    result = await manager.wake(table.resolve(MODEL))
    assert result.ok, result.message
    assert result.runtime == RUNTIME
    assert two.lifecycle == [("agent-b", "start", RUNTIME, None)]
    eligible = table.resolve(MODEL).eligible_backends()
    assert [b.node for b in eligible] == ["node-b"], "the one that was woken, on its own host"
    await manager.aclose()
    await table.aclose()


async def test_one_nodes_traffic_does_not_keep_the_other_nodes_replica_alive(
    two: TwoAgents,
) -> None:
    """The first consequence the review did not name (R1.6).

    `_runtime_last_request` and `_ready_since` were keyed by bare name,
    so both replicas of one model shared one idle clock: node B serving
    a request kept node A's replica looking recently used, and node A's
    engine sat in memory indefinitely on a machine nobody was asking
    anything of.
    """
    table = await _table()
    b = next(f for f in table.runtimes() if f.node == "node-b")
    a = next(f for f in table.runtimes() if f.node == "node-a")

    # A request served on node B, and nothing on node A.
    runtime = table.on_attempt_start(DRIVER, node="node-b")
    assert runtime == b.key, "the attempt counted against the replica on the node it hit"
    table.on_attempt_end(DRIVER, node="node-b", runtime=runtime, served=True, elapsed_ms=1)

    assert table.runtime_inflight(a.key) == 0
    assert table._runtime_last_request.get(b.key) is not None
    assert table._runtime_last_request.get(a.key) is None, (
        "node B's traffic must not mark node A's replica as recently used"
    )
    await table.aclose()


async def test_the_idle_pass_considers_both_replicas_not_one(
    two: TwoAgents, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second consequence (R1.6): the idle pass iterates the runtime
    map, and the merged map held one entry for two engines -- so one of
    them was never considered for unload at all, on any schedule.

    Both replicas declare a timeout here, so a correct pass stops two
    engines on two hosts.
    """
    two.runtimes[("node-a", RUNTIME)]["idleUnloadSeconds"] = 10
    table = await _table()
    assert len(table.runtimes()) == 2
    manager = _manager(table)
    monkeypatch.setattr(table, "idle_seconds", lambda _key: 100.0)

    stopped = await manager.idle_pass()

    assert stopped == [RUNTIME, RUNTIME], "one name, two engines, two unloads"
    assert sorted(c[0] for c in two.lifecycle) == ["agent-a", "agent-b"]
    assert two.runtimes[("node-a", RUNTIME)]["status"] == "stopped"
    assert two.runtimes[("node-b", RUNTIME)]["status"] == "stopped"
    await manager.aclose()
    await table.aclose()


async def test_a_ready_replica_on_one_node_does_not_make_a_stopped_one_routable(
    two: TwoAgents,
) -> None:
    """The review's own symptom: B `ready` masked A `stopped`.

    The driver on node A follows node A's runtime by name, and the name
    matched node B's entry just as well -- so a driver in front of a
    stopped engine read `ready`, stayed eligible, was picked by the
    balancer and answered 502.
    """
    two.runtimes[("node-a", RUNTIME)]["status"] = "stopped"
    table = await _table()

    by_node = {b.node: b for b in table.backends_for(MODEL)}
    assert set(by_node) == {"node-a", "node-b"}
    assert by_node["node-b"].eligible
    assert not by_node["node-a"].eligible, "the driver in front of a stopped engine is not routable"
    assert by_node["node-a"].ineligible_reason is not None
    assert [b.node for b in table.resolve(MODEL).eligible_backends()] == ["node-b"]
    await table.aclose()


async def test_a_driver_that_follows_a_runtime_its_own_node_does_not_report_is_warned_about(
    two: TwoAgents, caplog: pytest.LogCaptureFixture
) -> None:
    """ "Routable on faith" is the eligibility rule saying yes to an engine
    whose state it does not know. M7's record asked for this to be louder
    than DEBUG; R1.6 is where it became reachable by accident rather than
    only by a hand-written `baseUrl`, because the per-node join no longer
    silently borrows another machine's runtime of the same name."""
    two.runtimes.pop(("node-a", RUNTIME))
    caplog.set_level(logging.DEBUG, logger="eugene_plexus_gateway.routing")

    table = await _table()

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "routable on faith" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "node-a" in warnings[0].getMessage()
    await table.aclose()


async def test_a_stop_in_flight_holds_out_only_the_replica_it_is_for(
    two: TwoAgents,
) -> None:
    """The reservation is per `(node, name)` too.

    A sabotage proved this had no check: `_Backend.eligible` reads the
    reservation map, and reading it by bare name means a stop taken for
    node B holds node A's replica out of routing as well -- so an idle
    unload on one machine makes a healthy engine on the other
    unroutable for the length of an HTTP round trip, on the load-bearing
    path. The single-host test for this cannot see it, because there
    both keys are `(None, name)`.
    """
    table = await _table()
    b = next(f for f in table.runtimes() if f.node == "node-b")

    with table.stopping(b.key):
        by_node = {x.node: x for x in table.backends_for(MODEL)}
        assert not by_node["node-b"].eligible
        assert by_node["node-a"].eligible, "a stop on node B must not unroute node A"

    assert all(x.eligible for x in table.backends_for(MODEL))
    await table.aclose()


async def test_the_idle_pass_reserves_the_replica_it_is_stopping(
    two: TwoAgents, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reservation as the lifecycle manager WRITES it.

    A sabotage proved the case above could not see this: it takes the
    reservation itself, so it tests the read side and says nothing about
    the key the write uses. Written under the wrong key, the reservation
    lands where nothing reads it -- and the window it exists to close,
    between deciding to unload and the engine actually going, is open
    again on the load-bearing path.

    The stop is held open here so the assertion happens inside that
    window, which is the only place it is observable.
    """
    table = await _table()
    manager = _manager(table)
    monkeypatch.setattr(table, "idle_seconds", lambda _key: 100.0)

    seen: list[dict[str | None, bool]] = []
    real_stop = manager._client.stop

    async def watching_stop(agent_url: str, name: str, *, reason: str) -> bool:
        seen.append({b.node: b.eligible for b in table.backends_for(MODEL)})
        return await real_stop(agent_url, name, reason=reason)

    monkeypatch.setattr(manager._client, "stop", watching_stop)

    assert await manager.idle_pass() == [RUNTIME]

    assert seen == [{"node-a": True, "node-b": False}], (
        "inside the stop, node B is held out of routing and node A is not"
    )
    await manager.aclose()
    await table.aclose()


async def test_two_wakes_for_one_replica_are_one_start(two: TwoAgents) -> None:
    """A wake in flight is de-duplicated by `(node, name)`.

    **Driven through `wake()`, because `_wake_runtime` bypasses the map
    the key is in** -- which a sabotage proved by escaping the first
    version of this case. Keyed wrongly, the lookup misses its own
    entry, so de-duplication stops working at all and two callers
    arriving together each start the engine.
    """
    two.runtimes[("node-b", RUNTIME)]["status"] = "stopped"
    table = await _table()
    manager = _manager(table)
    resolution = table.resolve(MODEL)

    first, second = await asyncio.gather(manager.wake(resolution), manager.wake(resolution))

    assert first.ok and second.ok, (first.message, second.message)
    starts = [c for c in two.lifecycle if c[1] == "start"]
    assert starts == [("agent-b", "start", RUNTIME, None)], "one start, not two"
    assert manager._waking == {}, "the finished wake forgot the key it stored"
    await manager.aclose()
    await table.aclose()


async def test_two_replicas_asleep_are_two_wakes_not_one(two: TwoAgents) -> None:
    """A wake in flight is de-duplicated by `(node, name)`.

    Keyed by name, the second replica's wake returned the first's task:
    the caller was told its runtime was starting while nothing on its
    machine had been asked to start, and the wake it was handed could
    succeed for the other host entirely.
    """
    two.runtimes[("node-a", RUNTIME)]["startOnDemand"] = True
    two.runtimes[("node-a", RUNTIME)]["status"] = "stopped"
    two.runtimes[("node-b", RUNTIME)]["status"] = "stopped"
    table = await _table()
    manager = _manager(table)

    facts = sorted(table.runtimes(), key=lambda f: f.node or "")
    a, b = facts[0], facts[1]
    assert (a.node, b.node) == ("node-a", "node-b")

    await manager._wake_runtime(a)
    await manager._wake_runtime(b)

    starts = [c for c in two.lifecycle if c[1] == "start"]
    assert sorted(c[0] for c in starts) == ["agent-a", "agent-b"], "each on its own host"
    await manager.aclose()
    await table.aclose()


async def test_a_driver_that_names_no_runtime_takes_its_own_nodes_alias(
    two: TwoAgents,
) -> None:
    """The alias fallback is per node as well.

    It is the route for a hand-written `baseUrl` or a driver older than
    M4, and it was keyed by the alias alone -- which every replica of a
    model shares by definition. A driver on node A with no `runtime` in
    its `/v1/info` therefore followed whichever machine's engine hashed
    first, and read its status.
    """
    two.drop_runtime_from_info = True
    table = await _table()

    by_node = {b.node: b for b in table.backends_for(MODEL)}
    assert by_node["node-a"].runtime is not None
    assert by_node["node-a"].runtime.node == "node-a"
    assert by_node["node-b"].runtime is not None
    assert by_node["node-b"].runtime.node == "node-b"
    await table.aclose()


# --------------------------------------------------------------------------- #
# the control root's address is read live
# --------------------------------------------------------------------------- #


async def test_control_url_is_read_on_every_refresh_not_captured(two: TwoAgents) -> None:
    """`controlUrl` set on a *running* gateway must reach the next refresh.

    Found on a real two-machine install, 2026-09-11. The field's own
    description says it "takes effect on the next routing refresh" and
    its `requiresRestart` is false, but the constructor captured the
    string, so a `PATCH /v1/config` changed nothing until the process was
    restarted. The symptom is the worst kind: the worker node was
    enrolled, reachable, its driver `running` in the control root's own
    union view -- and invisible to routing, with an empty
    `unreachable_drivers` list saying nothing was wrong.

    Until 2026-09-13 nothing set `controlUrl` automatically -- not the
    installers, not the container, not the agent's first-boot topology
    -- so every multi-host install passed through exactly this path. The
    gateway derives it from its own agent now (the tests at the bottom of
    this file); this one keeps the override live and the fake agent here
    knows of no control root, so unset still means single host.
    """
    configured: str | None = None
    table = RoutingTable(agent_url=AGENT_A, control_url=lambda: configured, refresh_seconds=3600)

    # Single-host until told otherwise: only agent A, so only its driver.
    await table.refresh()
    # Unenrolled, so this host has no node name and `None` is its key.
    assert _nodes_for(table) == [None]

    # The operator sets it on the running gateway. No restart.
    configured = CONTROL
    await table.refresh()
    assert _nodes_for(table) == ["node-a", "node-b"]

    # And clearing it collapses back, because a UI that empties the field
    # sends "" rather than removing the key.
    configured = "   "
    await table.refresh()
    # Unenrolled, so this host has no node name and `None` is its key.
    assert _nodes_for(table) == [None]


async def test_a_plain_control_url_string_still_works(two: TwoAgents) -> None:
    """The callable is an addition, not a replacement: most callers have
    a string and the two existing tests above pass one."""
    table = RoutingTable(agent_url=AGENT_A, control_url=CONTROL + "/", refresh_seconds=3600)
    await table.refresh()
    assert _nodes_for(table) == ["node-a", "node-b"]


# --------------------------------------------------------------------------- #
# the control root is derived from the gateway's own agent
# --------------------------------------------------------------------------- #


def _root(table: RoutingTable) -> dict[str, Any]:
    view = table.as_routing_view().control_root
    assert view is not None
    return {
        "source": str(view.source.value if hasattr(view.source, "value") else view.source),
        "url": str(view.url).rstrip("/") if view.url is not None else None,
        "reachable": view.reachable,
        "error": view.error,
        "nodes": view.nodes,
    }


async def test_the_control_root_is_derived_from_the_enrolled_agent(two: TwoAgents) -> None:
    """Nothing set `controlUrl` -- not the installers, not the container,
    not the agent's first boot -- so every multi-host install came up with
    its workers enrolled, reachable and invisible to routing (2026-09-11).
    The gateway asks its own agent now, and an enrolled node knows its
    root."""
    two.enrolled_to = CONTROL
    table = RoutingTable(agent_url=AGENT_A, refresh_seconds=3600)

    await table.refresh()

    assert _nodes_for(table) == ["node-a", "node-b"]
    assert _root(table) == {
        "source": "agent",
        "url": CONTROL,
        "reachable": True,
        "error": None,
        "nodes": 2,
    }


async def test_an_unenrolled_agent_that_runs_the_control_root_still_names_it(
    two: TwoAgents,
) -> None:
    """First boot seeds control, gateway and library before anybody has
    enrolled anything. The gateway finds the root in the topology."""
    two.declares_control = True
    table = RoutingTable(agent_url=AGENT_A, refresh_seconds=3600)

    await table.refresh()

    assert _nodes_for(table) == ["node-a", "node-b"]
    assert _root(table)["source"] == "agent"
    assert _root(table)["url"] == CONTROL


async def test_no_control_root_anywhere_is_a_single_host_install(two: TwoAgents) -> None:
    table = RoutingTable(agent_url=AGENT_A, refresh_seconds=3600)

    await table.refresh()

    # Unenrolled, so this host has no node name and `None` is its key.
    assert _nodes_for(table) == [None]
    assert _root(table) == {
        "source": "none",
        "url": None,
        "reachable": False,
        "error": None,
        "nodes": None,
    }


async def test_a_configured_control_url_wins_even_when_it_is_wrong(two: TwoAgents) -> None:
    """The override is topmost: set, it is used as given and the agent's
    better answer is not consulted. An expert naming an address gets that
    address."""
    two.enrolled_to = CONTROL
    table = RoutingTable(agent_url=AGENT_A, control_url="http://nowhere:1", refresh_seconds=3600)

    await table.refresh()

    # Unenrolled, so this host has no node name and `None` is its key.
    assert _nodes_for(table) == [None]
    root = _root(table)
    assert root["source"] == "config"
    assert root["url"] == "http://nowhere:1"
    assert root["reachable"] is False
    assert root["error"]


async def test_the_derived_root_is_read_on_every_refresh(two: TwoAgents) -> None:
    """Enrollment happens after the gateway is up -- the wizard's Start
    enrolls the control host's own agent -- so the answer has to be
    re-asked, not captured."""
    table = RoutingTable(agent_url=AGENT_A, refresh_seconds=3600)
    await table.refresh()
    assert _root(table)["source"] == "none"

    two.enrolled_to = CONTROL
    await table.refresh()

    assert _root(table)["source"] == "agent"
    assert _nodes_for(table) == ["node-a", "node-b"]


async def test_a_sealed_control_root_is_reported_not_hidden(two: TwoAgents) -> None:
    """A container's root comes back locked after every restart while
    every health check says ok. The routing view says what the gateway
    got, the previous node list stays in force, and the no-models 404
    says where the gateway looked."""
    two.enrolled_to = CONTROL
    two.components = {"agent-a": [], "agent-b": []}  # nothing routable, so the 404 explains
    table = RoutingTable(agent_url=AGENT_A, refresh_seconds=3600)
    await table.refresh()
    assert _root(table)["nodes"] == 2

    two.control_status = 503
    await table.refresh()

    root = _root(table)
    assert root["reachable"] is False
    assert root["error"] == "503 Locked"
    assert root["nodes"] == 2, "the count is from the last read that answered"
    assert sorted(table._snapshot.agents) == ["node-a", "node-b"], "the previous list stays"

    from eugene_plexus_gateway.routes.inference import _no_such_model

    message = json.loads(_no_such_model("x", table).as_openai().body)["error"]["message"]
    assert "did not answer on the last refresh (503 Locked)" in message
    assert CONTROL in message


async def test_a_single_host_404_says_only_this_host_was_read(two: TwoAgents) -> None:
    two.components = {"agent-a": [], "agent-b": []}
    table = RoutingTable(agent_url=AGENT_A, refresh_seconds=3600)
    await table.refresh()

    from eugene_plexus_gateway.routes.inference import _no_such_model

    message = json.loads(_no_such_model("x", table).as_openai().body)["error"]["message"]
    assert "Only this host's agent was read" in message


async def test_the_control_root_is_announced_once_not_every_refresh(
    two: TwoAgents, caplog: pytest.LogCaptureFixture
) -> None:
    two.enrolled_to = CONTROL
    caplog.set_level(logging.INFO, logger="eugene_plexus_gateway.routing")
    table = RoutingTable(agent_url=AGENT_A, refresh_seconds=3600)

    await table.refresh()
    await table.refresh()
    await table.refresh()

    announcements = [r for r in caplog.records if "derived from the agent" in r.getMessage()]
    assert len(announcements) == 1
    assert "the root this node is enrolled to" in announcements[0].getMessage()


async def test_a_root_that_stays_sealed_is_warned_about_once(
    two: TwoAgents, caplog: pytest.LogCaptureFixture
) -> None:
    """An uninitialized root before the wizard, or a sealed one after a
    container restart, is that way for minutes to hours; a warning every
    refresh is a log nobody reads. One line when it stops answering, one
    when it answers again."""
    two.enrolled_to = CONTROL
    caplog.set_level(logging.DEBUG, logger="eugene_plexus_gateway.routing")
    table = RoutingTable(agent_url=AGENT_A, refresh_seconds=3600)
    await table.refresh()

    two.control_status = 503
    for _ in range(4):
        await table.refresh()
    two.control_status = 200
    await table.refresh()

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "did not answer" in r.getMessage()
    ]
    recoveries = [r for r in caplog.records if "answers again" in r.getMessage()]
    assert len(warnings) == 1
    assert "503 Locked" in warnings[0].getMessage()
    assert len(recoveries) == 1
    assert _root(table)["reachable"] is True
