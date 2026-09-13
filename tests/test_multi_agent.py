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


def _info(model_id: str, runtime: str) -> dict[str, Any]:
    return {
        "backend": "openai_compat_http",
        "version": "0.1.0",
        "modelId": model_id,
        "runtime": runtime,
    }


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
        self.runtimes: dict[str, dict[str, Any]] = {
            "qwen-a": {
                "name": "qwen-a",
                "engine": "llama_cpp",
                "modelPath": "/m/qwen.gguf",
                "modelAlias": MODEL,
                "status": "ready",
                "url": "http://127.0.0.1:8090",
                "node": "node-a",
            },
            "qwen-b": {
                "name": "qwen-b",
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
            "name": "qwen-b-driver",
            "kind": "inference-driver",
            "url": "http://127.0.0.1:8091",
            "status": "running",
        }
        if advertise:
            b_component["advertiseUrl"] = "http://agent-b:8091"
        self.components = {
            "agent-a": [
                {
                    "name": "qwen-a-driver",
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
                if self.runtimes[name]["node"] != node:
                    return httpx.Response(404, json={"detail": f"{name} is not on this node"})
                self.runtimes[name]["status"] = "stopped" if parts[3] == "stop" else "ready"
                if parts[3] == "stop":
                    self.runtimes[name]["stopReason"] = (body or {}).get("reason", "operator")
                else:
                    self.runtimes[name].pop("stopReason", None)
                return httpx.Response(202, json={"scheduled": True, "delayMs": 0})
            if len(parts) == 3 and parts[1] == "runtimes":
                runtime = self.runtimes.get(parts[2])
                if runtime is None or runtime["node"] != node:
                    return httpx.Response(404, json={"detail": "no such runtime here"})
                return httpx.Response(200, json=runtime)
            if path == "/v1/runtimes/admission":
                return httpx.Response(
                    200, json={"decision": "admit", "fit": "fits", "reason": "ok", "blockers": []}
                )
            return httpx.Response(404, json={"detail": path})

        if path == "/v1/info":
            self.probed.append(f"{host}:{port}")
            if (host, port) == ("127.0.0.1", 8090):
                return httpx.Response(200, json=_info(MODEL, "qwen-a"))
            if (host, port) == ("agent-b", 8091):
                return httpx.Response(200, json=_info(MODEL, "qwen-b"))
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


# --------------------------------------------------------------------------- #
# fan-out and attribution
# --------------------------------------------------------------------------- #


async def test_the_table_fans_out_over_every_node_the_control_root_lists(two: TwoAgents) -> None:
    """Two agents, one alias, one table: the replica set differentiator #7
    describes as 'across two GPUs' when the GPUs are on two hosts."""
    table = await _table()

    backends = table.backends_for(MODEL)
    assert sorted(b.name for b in backends) == ["qwen-a-driver", "qwen-b-driver"]
    assert all(b.eligible for b in backends)

    facts = {r.name: r for r in table.runtimes()}
    assert facts["qwen-a"].node == "node-a"
    assert facts["qwen-b"].node == "node-b"

    # The agent map is keyed by node, trailing slash trimmed, and the
    # default agent is no longer the only one.
    assert table.agent_url_for(facts["qwen-a"]) == AGENT_A
    assert table.agent_url_for(facts["qwen-b"]) == AGENT_B

    view = table.as_routing_view()
    by_driver = {b.driver: b for s in view.slots for t in s.tiers for b in t.backends}
    assert by_driver["qwen-a-driver"].node == "node-a"
    assert by_driver["qwen-b-driver"].node == "node-b"
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
    assert unreachable == ["qwen-b-driver"]
    assert [b.name for b in table.backends_for(MODEL)] == ["qwen-a-driver"]
    await table.aclose()


async def test_a_control_root_that_stops_answering_keeps_the_agent_map(two: TwoAgents) -> None:
    """Management being down must not empty the routing table — M5's
    surviving-data-path guarantee, at the node-discovery step."""
    table = await _table()
    two.control_status = 503
    await table.refresh()
    facts = {r.name: r for r in table.runtimes()}
    assert set(facts) == {"qwen-a", "qwen-b"}
    assert table.agent_url_for(facts["qwen-b"]) == AGENT_B
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
    """qwen-b declared an idle timeout and lives on node B. The stop must
    reach node B's agent — not the default agent, which is node A's and
    would answer 404 for a runtime it does not have."""
    table = await _table()
    manager = _manager(table)
    monkeypatch.setattr(table, "idle_seconds", lambda name: 100.0)

    stopped = await manager.idle_pass()
    assert stopped == ["qwen-b"]
    assert two.lifecycle == [("agent-b", "stop", "qwen-b", {"reason": "idle"})]
    assert two.runtimes["qwen-b"]["status"] == "stopped"
    assert two.runtimes["qwen-a"]["status"] == "ready", "no timeout, never touched"
    await manager.aclose()
    await table.aclose()


async def test_a_wake_goes_to_the_agent_that_owns_the_runtime(two: TwoAgents) -> None:
    """Both replicas asleep; only qwen-b asked to be woken. The start goes
    to node B's agent and the runtime is polled there."""
    two.runtimes["qwen-a"]["status"] = "stopped"
    two.runtimes["qwen-b"]["status"] = "stopped"
    table = await _table()
    manager = _manager(table)
    assert table.resolve(MODEL).eligible_backends() == []

    result = await manager.wake(table.resolve(MODEL))
    assert result.ok, result.message
    assert result.runtime == "qwen-b"
    assert two.lifecycle == [("agent-b", "start", "qwen-b", None)]
    assert [b.name for b in table.resolve(MODEL).eligible_backends()] == ["qwen-b-driver"]
    await manager.aclose()
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
    assert sorted(b.name for b in table.backends_for(MODEL)) == ["qwen-a-driver"]

    # The operator sets it on the running gateway. No restart.
    configured = CONTROL
    await table.refresh()
    assert sorted(b.name for b in table.backends_for(MODEL)) == [
        "qwen-a-driver",
        "qwen-b-driver",
    ]

    # And clearing it collapses back, because a UI that empties the field
    # sends "" rather than removing the key.
    configured = "   "
    await table.refresh()
    assert sorted(b.name for b in table.backends_for(MODEL)) == ["qwen-a-driver"]


async def test_a_plain_control_url_string_still_works(two: TwoAgents) -> None:
    """The callable is an addition, not a replacement: most callers have
    a string and the two existing tests above pass one."""
    table = RoutingTable(agent_url=AGENT_A, control_url=CONTROL + "/", refresh_seconds=3600)
    await table.refresh()
    assert sorted(b.name for b in table.backends_for(MODEL)) == [
        "qwen-a-driver",
        "qwen-b-driver",
    ]


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

    assert sorted(b.name for b in table.backends_for(MODEL)) == ["qwen-a-driver", "qwen-b-driver"]
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

    assert sorted(b.name for b in table.backends_for(MODEL)) == ["qwen-a-driver", "qwen-b-driver"]
    assert _root(table)["source"] == "agent"
    assert _root(table)["url"] == CONTROL


async def test_no_control_root_anywhere_is_a_single_host_install(two: TwoAgents) -> None:
    table = RoutingTable(agent_url=AGENT_A, refresh_seconds=3600)

    await table.refresh()

    assert sorted(b.name for b in table.backends_for(MODEL)) == ["qwen-a-driver"]
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

    assert sorted(b.name for b in table.backends_for(MODEL)) == ["qwen-a-driver"]
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
    assert sorted(b.name for b in table.backends_for(MODEL)) == ["qwen-a-driver", "qwen-b-driver"]


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

    message = json.loads(_no_such_model("x", table).body)["error"]["message"]
    assert "did not answer on the last refresh (503 Locked)" in message
    assert CONTROL in message


async def test_a_single_host_404_says_only_this_host_was_read(two: TwoAgents) -> None:
    two.components = {"agent-a": [], "agent-b": []}
    table = RoutingTable(agent_url=AGENT_A, refresh_seconds=3600)
    await table.refresh()

    from eugene_plexus_gateway.routes.inference import _no_such_model

    message = json.loads(_no_such_model("x", table).body)["error"]["message"]
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
