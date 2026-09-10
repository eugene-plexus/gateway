"""The routing table: model id -> slot -> tiers -> backends.

The gateway stores no backend URLs and no model list. It reads the
agent topology for `inference-driver` entries, asks each one's
`/v1/info` what it is serving, and groups the answers by model id. So:

  * backend addresses live in exactly one place, the agent topology
  * adding a model is not a config edit — start an engine, point a
    driver at it (or let the agent's companion do it), and it becomes
    routable on the next refresh
  * two drivers serving the same model are a **replica set**: one tier,
    load-balanced, cascading within itself on failure

M6 adds the rest of differentiator #7 on top of that:

  * **Every model is a slot; a slot is an ordered list of tiers; a tier
    is every eligible backend serving one model id.** A `modelSlots`
    config entry extends a model's tiers with targets — model ids tried
    in order when the tiers before them have nothing that answers. The
    natural no-config case is the one-tier slot `[model]`.
  * **Drivers are joined to runtimes by name.** A driver reports the
    runtime it follows on `/v1/info` (M4's `runtime`), and the agent's
    `/v1/runtimes` says what state that runtime is in. The join is what
    enforces the rule the contract has stated since M0: **a driver whose
    runtime is not `ready` is not routed to.** A driver that follows no
    runtime — a cloud provider, a CLI subscription — is eligible
    whenever it is reachable.
  * **Least outstanding requests within a tier**, weighted by the
    runtime's `parallelSlots`, ties broken round-robin. The signal is
    the gateway's own in-flight count, which exists for every engine.
  * **Per-runtime demand tracking** — the last request and the in-flight
    count behind each runtime — which is what `lifecycle.py` reads to
    unload an idle runtime and what keeps one with a request in flight
    from ever being stopped.

Topology comes from every agent in the install: one when no control
root is configured, or the node list the control root reports
(`GET /v1/nodes`) when one is. Each agent's `/v1/components` and
`/v1/runtimes` are read directly rather than through the control root's
union views, because the union `RuntimePlacement` carries no lifecycle
fields and because a poll that dies with the control root would take
model swapping down with management — the surviving data path M5 tested.

The refresh is periodic rather than on-demand because a request should
never pay for topology discovery, and because engines come and go under
the agent without telling anyone.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from ._generated.driver_models import DriverInfo
from ._generated.models import (
    BackendKind,
    DriverHealth,
    Model,
    ModelRoutingInfo,
    RoutingBackendView,
    RoutingSlotView,
    RoutingTableView,
    RoutingTierView,
)
from .driver_client import DriverClient, HttpDriverClient, TieredClient

log = logging.getLogger(__name__)

READY = "ready"
STOPPED = "stopped"
LEAST_BUSY = "least_busy"
ROUND_ROBIN = "round_robin"

# The fields of the agent's `Runtime` that are also its `RuntimeSpec`.
# Carried so a wake can ask the agent's admission dry run about exactly
# the declaration it would start.
_SPEC_FIELDS = (
    "name",
    "engine",
    "modelPath",
    "modelAlias",
    "host",
    "port",
    "autoStart",
    "autoDriver",
    "idleUnloadSeconds",
    "startOnDemand",
    "flags",
    "extraArgs",
    "env",
    "workingDirectory",
    "binary",
)


@dataclass(frozen=True)
class _RuntimeFacts:
    """What the agent knows about one engine runtime."""

    name: str
    alias: str | None
    status: str | None
    node: str | None
    url: str | None
    context_length: int | None
    parallel_slots: int | None
    idle_unload_seconds: int | None
    start_on_demand: bool
    stop_reason: str | None
    spec: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Backend:
    """One reachable driver, what it told us it serves, and the runtime
    behind it when the agent supervises one."""

    name: str
    url: str
    client: DriverClient
    info: DriverInfo
    runtime: _RuntimeFacts | None = None

    @property
    def eligible(self) -> bool:
        """Routable right now. A driver following a runtime is eligible
        only when that runtime is `ready`; one following nothing of ours
        is eligible whenever it is reachable, which it is by being here."""
        return self.runtime is None or self.runtime.status == READY

    @property
    def ineligible_reason(self) -> str | None:
        if self.eligible:
            return None
        assert self.runtime is not None
        return f"runtime {self.runtime.name!r} is {self.runtime.status or 'unknown'}"

    @property
    def parallel_slots(self) -> int:
        if self.runtime is not None and self.runtime.parallel_slots:
            return max(1, self.runtime.parallel_slots)
        return 1


@dataclass
class _Unreachable:
    """A driver in the topology that didn't answer `/v1/info`."""

    name: str
    url: str
    error: str


@dataclass
class _Snapshot:
    """One resolved view of the world. Replaced wholesale on refresh so a
    request never sees a half-rebuilt table."""

    by_model: dict[str, list[_Backend]] = field(default_factory=dict)
    reachable: list[_Backend] = field(default_factory=list)
    unreachable: list[_Unreachable] = field(default_factory=list)
    # Keyed by runtime NAME. Aliases are not unique — two runtimes under
    # one alias are replicas — so the alias is a field, not the key.
    runtimes: dict[str, _RuntimeFacts] = field(default_factory=dict)
    # node name -> that node's agent URL. `None` is the default agent,
    # which is the only agent when no control root is configured.
    agents: dict[str | None, str] = field(default_factory=dict)
    refreshed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class _Tier:
    target: str
    backends: list[_Backend]

    def eligible(self) -> list[_Backend]:
        return [b for b in self.backends if b.eligible]


@dataclass
class Resolution:
    """A request's model, resolved to its slot."""

    model: str
    configured: bool
    tiers: list[_Tier]

    def has_backends(self) -> bool:
        return any(t.backends for t in self.tiers)

    def backends(self) -> list[_Backend]:
        return [b for t in self.tiers for b in t.backends]

    def eligible_backends(self) -> list[_Backend]:
        return [b for t in self.tiers for b in t.eligible()]

    def runtimes(self) -> list[_RuntimeFacts]:
        """Distinct runtimes behind this slot, in tier order."""
        seen: set[str] = set()
        out: list[_RuntimeFacts] = []
        for backend in self.backends():
            facts = backend.runtime
            if facts is not None and facts.name not in seen:
                seen.add(facts.name)
                out.append(facts)
        return out

    def startable(self) -> list[_RuntimeFacts]:
        """Stopped runtimes that asked to be woken on demand, tier order."""
        return [r for r in self.runtimes() if r.status == STOPPED and r.start_on_demand]

    def waking(self) -> list[_RuntimeFacts]:
        return [r for r in self.runtimes() if r.status in ("starting", "loading")]


class RoutingTable:
    """Resolves a requested model to the backends that can serve it."""

    def __init__(
        self,
        *,
        agent_url: str,
        service_token: str | None = None,
        request_timeout_seconds: float = 180.0,
        refresh_seconds: float = 15.0,
        slots: Callable[[], Any] | None = None,
        strategy: Callable[[], Any] | None = None,
        control_url: str | None = None,
    ) -> None:
        self._agent_url = agent_url.rstrip("/")
        self._control_url = control_url.rstrip("/") if control_url else None
        self._service_token = service_token
        self._request_timeout = request_timeout_seconds
        self._refresh_seconds = refresh_seconds
        # Read live on every resolve, so a PATCH to `modelSlots` or
        # `loadBalancing` takes effect without a restart.
        self._slots = slots or (lambda: [])
        self._strategy = strategy or (lambda: LEAST_BUSY)
        self._snapshot = _Snapshot(agents={None: self._agent_url})
        # Clients are cached by (name, url) and reused across refreshes.
        # Rebuilding an httpx.AsyncClient every 15s would throw away the
        # connection pool and leak sockets.
        self._clients: dict[tuple[str, str], HttpDriverClient] = {}
        self._task: asyncio.Task[None] | None = None
        # One refresh at a time; on-demand callers share it.
        self._refreshing: asyncio.Task[None] | None = None
        # Demand, by driver and by runtime. Monotonic seconds; never
        # persisted, never replicated — a promoted control root re-reads
        # what is loaded from the agents, not from here.
        self._inflight: dict[str, int] = {}
        self._last_request: dict[str, float] = {}
        self._runtime_inflight: dict[str, int] = {}
        self._runtime_last_request: dict[str, float] = {}
        self._ready_since: dict[str, float] = {}
        self._cursor: dict[str, int] = {}

    # --- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Do one refresh now, then keep refreshing in the background.

        The first refresh is awaited so the gateway is routable as soon
        as it is serving, rather than 404ing for the first interval.
        """
        await self.refresh()
        if self._task is None:
            self._task = asyncio.create_task(self._refresh_loop(), name="routing-refresh")

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await self._task
            self._task = None
        for client in self._clients.values():
            with contextlib.suppress(BaseException):
                await client.aclose()
        self._clients.clear()
        self._snapshot = _Snapshot(agents={None: self._agent_url})

    async def _refresh_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._refresh_seconds)
                try:
                    await self.refresh()
                except Exception as e:
                    # Never let a bad refresh kill the loop — the previous
                    # snapshot stays serving, which is strictly better than
                    # a gateway that stops routing because the agent
                    # blipped.
                    log.warning("routing refresh failed; keeping previous table: %s", e)
        except asyncio.CancelledError:
            return

    # --- refresh ----------------------------------------------------------

    async def refresh_if_stale(self, max_age_seconds: float = 1.0) -> bool:
        """Refresh now unless the snapshot is younger than `max_age_seconds`.

        The first live run found the seam: the periodic refresh can be
        `routingRefreshSeconds` behind the agent about readiness, so a
        request arriving in the seconds after an engine turned `ready` —
        or after one was killed — met a stale table and a 503. A request
        that finds nothing eligible pays for one topology read before
        concluding anything; a request that finds a backend never does.
        Concurrent callers share one refresh, and the age floor keeps a
        burst during a long load from turning into a poll storm.
        """
        age = (datetime.now(UTC) - self._snapshot.refreshed_at).total_seconds()
        if age < max_age_seconds:
            return False
        if self._refreshing is None or self._refreshing.done():
            self._refreshing = asyncio.create_task(self.refresh(), name="routing-refresh-on-demand")
        try:
            await asyncio.shield(self._refreshing)
        except Exception as e:
            log.warning("on-demand routing refresh failed; keeping previous table: %s", e)
            return False
        return True

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._service_token}"} if self._service_token else {}

    async def _get_json(self, url: str) -> Any | None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url, headers=self._headers())
            if response.status_code >= 400:
                log.debug("%s returned %d", url, response.status_code)
                return None
            return response.json()
        except (httpx.HTTPError, ValueError) as e:
            log.debug("could not read %s: %s", url, e)
            return None

    async def discover_agents(self) -> dict[str | None, str]:
        """Which agents to read topology from.

        Without a control root: the one configured agent. With one: the
        node list it reports, each with its agent URL — the lookup that
        closes the loopback:8079 assumption for anything that has to
        reach a runtime on another host. A control root that does not
        answer leaves the previous agent map in place: management being
        down must not empty the routing table.
        """
        if self._control_url is None:
            return {None: self._agent_url}
        body = await self._get_json(f"{self._control_url}/v1/nodes")
        nodes = body.get("nodes") if isinstance(body, dict) else None
        if not isinstance(nodes, list):
            log.warning(
                "control root at %s did not answer /v1/nodes; keeping the previous agent map",
                self._control_url,
            )
            return dict(self._snapshot.agents) or {None: self._agent_url}
        agents: dict[str | None, str] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            name, url = node.get("name"), node.get("url")
            if isinstance(name, str) and name and isinstance(url, str) and url:
                agents[name] = url.rstrip("/")
        if not agents:
            # An enrolled-nothing control root: fall back to the agent we
            # were told about, which is the single-host case anyway.
            agents = {None: self._agent_url}
        return agents

    async def refresh(self) -> None:
        agents = await self.discover_agents()

        # Every agent, concurrently: its drivers and its runtimes.
        fetched = await asyncio.gather(
            *(self._fetch_agent(node, url) for node, url in agents.items()),
            return_exceptions=True,
        )
        entries: list[tuple[str, str]] = []
        runtimes: dict[str, _RuntimeFacts] = {}
        for result in fetched:
            if isinstance(result, BaseException):
                log.warning("reading an agent raised unexpectedly: %s", result)
                continue
            agent_entries, agent_runtimes = result
            entries.extend(agent_entries)
            runtimes.update(agent_runtimes)

        # Drop clients for drivers that left the topology.
        live_keys = {(name, url) for name, url in entries}
        for key in list(self._clients):
            if key not in live_keys:
                client = self._clients.pop(key)
                log.info("driver %r left the topology; closing its client", key[0])
                with contextlib.suppress(BaseException):
                    await client.aclose()

        results = await asyncio.gather(
            *(self._probe(name, url) for name, url in entries),
            return_exceptions=True,
        )

        snapshot = _Snapshot(runtimes=runtimes, agents=agents)
        for probe in results:
            if isinstance(probe, BaseException):
                log.warning("driver probe raised unexpectedly: %s", probe)
                continue
            if isinstance(probe, _Unreachable):
                snapshot.unreachable.append(probe)
                continue
            snapshot.reachable.append(probe)

        # Aliases claimed by exactly one runtime, for the drivers that do
        # not name the runtime they follow (a hand-written `baseUrl`, or a
        # driver older than M4). By name is the rule; this is the
        # fallback, and it is deliberately silent on a replica pair
        # because naming one of them would be a coin flip.
        by_alias: dict[str, _RuntimeFacts | None] = {}
        for facts in runtimes.values():
            if facts.alias:
                by_alias[facts.alias] = None if facts.alias in by_alias else facts

        for backend in snapshot.reachable:
            model_id = backend.info.modelId
            if not model_id:
                # A driver that won't say what it serves cannot be routed
                # to — there is no key to route on. Its health still shows
                # up on the admin surface.
                log.debug(
                    "driver %r reports no modelId; not routable (it is probably in degraded mode)",
                    backend.name,
                )
                continue
            if backend.info.runtime:
                backend.runtime = runtimes.get(backend.info.runtime)
                if backend.runtime is None:
                    log.debug(
                        "driver %r follows runtime %r, which no agent reports; treating it as "
                        "routable on faith",
                        backend.name,
                        backend.info.runtime,
                    )
            else:
                backend.runtime = by_alias.get(model_id)
            snapshot.by_model.setdefault(model_id, []).append(backend)

        # Stable base order per model; the balancer rotates from here.
        for backends in snapshot.by_model.values():
            backends.sort(key=lambda b: b.name)

        # Ready-since, so a runtime that loaded and was never asked for
        # anything still counts as idle from the moment it became ready.
        now = time.monotonic()
        for name, facts in runtimes.items():
            if facts.status == READY:
                self._ready_since.setdefault(name, now)
            else:
                self._ready_since.pop(name, None)
        for name in list(self._ready_since):
            if name not in runtimes:
                self._ready_since.pop(name, None)

        self._snapshot = snapshot
        log.debug(
            "routing table refreshed: %d model(s) across %d reachable driver(s), "
            "%d unreachable, %d runtime(s), %d agent(s)",
            len(snapshot.by_model),
            len(snapshot.reachable),
            len(snapshot.unreachable),
            len(snapshot.runtimes),
            len(snapshot.agents),
        )

    async def _fetch_agent(
        self, node: str | None, url: str
    ) -> tuple[list[tuple[str, str]], dict[str, _RuntimeFacts]]:
        entries, runtimes = await asyncio.gather(
            self._fetch_driver_entries(url),
            self._fetch_runtime_facts(url, node),
        )
        return entries, runtimes

    async def fetch_driver_entries(self) -> list[tuple[str, str]]:
        """`(name, url)` for every inference-driver across every agent.

        Public because `POST /v1/config/test` reads the topology fresh
        rather than off the last refresh — the point of a Test button is
        the world as it is now.
        """
        agents = await self.discover_agents()
        out: list[tuple[str, str]] = []
        for url in agents.values():
            out.extend(await self._fetch_driver_entries(url))
        return out

    async def _fetch_driver_entries(self, agent_url: str) -> list[tuple[str, str]]:
        """An unreachable agent yields an empty list, which degrades to
        "nothing is routable from it" rather than crashing — the config
        endpoints stay up so the operator can fix whatever is wrong."""
        body = await self._get_json(f"{agent_url}/v1/components")
        if body is None:
            log.warning("could not read %s/v1/components; nothing from it is routable", agent_url)
            return []
        components = body.get("components") if isinstance(body, dict) else None
        if not isinstance(components, list):
            return []
        out: list[tuple[str, str]] = []
        for entry in components:
            if not isinstance(entry, dict) or entry.get("kind") != "inference-driver":
                continue
            name, url = entry.get("name"), entry.get("url")
            if isinstance(name, str) and name and isinstance(url, str) and url:
                out.append((name, url.rstrip("/")))
        return out

    async def _fetch_runtime_facts(
        self, agent_url: str, node: str | None
    ) -> dict[str, _RuntimeFacts]:
        """Runtime name -> what the agent knows about it.

        Where the `ready` gate, attribution, context, capacity and the
        lifecycle policy fields all come from. An unreachable agent
        yields an empty map, so its drivers are routed to on faith and
        the cascade sorts out the dead ones — nothing stops routing over
        a missing runtime list.
        """
        body = await self._get_json(f"{agent_url}/v1/runtimes")
        if body is None:
            log.debug("could not read %s/v1/runtimes; no runtime facts from it", agent_url)
            return {}
        runtimes = body.get("runtimes") if isinstance(body, dict) else None
        if not isinstance(runtimes, list):
            return {}
        facts: dict[str, _RuntimeFacts] = {}
        for entry in runtimes:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                continue
            capabilities = entry.get("capabilities")
            context = capabilities.get("contextLength") if isinstance(capabilities, dict) else None
            slots = capabilities.get("parallelSlots") if isinstance(capabilities, dict) else None
            idle = entry.get("idleUnloadSeconds")
            # `node` as the agent reports it (its own identity), else the
            # node we asked — which is the same thing, and fills it in
            # for an unenrolled agent.
            reported_node = entry.get("node")
            facts[name] = _RuntimeFacts(
                name=name,
                alias=entry.get("modelAlias") if isinstance(entry.get("modelAlias"), str) else None,
                status=str(entry["status"]) if entry.get("status") else None,
                node=reported_node if isinstance(reported_node, str) and reported_node else node,
                url=entry.get("url") if isinstance(entry.get("url"), str) else None,
                context_length=context if isinstance(context, int) and context > 0 else None,
                parallel_slots=slots if isinstance(slots, int) and slots > 0 else None,
                idle_unload_seconds=idle if isinstance(idle, int) and idle > 0 else None,
                start_on_demand=bool(entry.get("startOnDemand")),
                stop_reason=str(entry["stopReason"]) if entry.get("stopReason") else None,
                spec={k: entry[k] for k in _SPEC_FIELDS if entry.get(k) is not None},
            )
        return facts

    async def _probe(self, name: str, url: str) -> _Backend | _Unreachable:
        client = self._clients.get((name, url))
        if client is None:
            client = HttpDriverClient(
                name=name,
                base_url=url,
                timeout_seconds=self._request_timeout,
                service_token=self._service_token,
            )
            self._clients[(name, url)] = client
        try:
            info = await client.info()
        except (httpx.HTTPError, ValueError) as e:
            return _Unreachable(name=name, url=url, error=str(e))
        return _Backend(name=name, url=url, client=client, info=info)

    # --- demand -------------------------------------------------------------
    #
    # The `RoutingHooks` protocol `TieredClient` calls around every attempt.

    def _runtime_name_for_driver(self, driver: str) -> str | None:
        for backend in self._snapshot.reachable:
            if backend.name == driver:
                return backend.runtime.name if backend.runtime is not None else None
        return None

    def on_attempt_start(self, driver: str) -> None:
        self._inflight[driver] = self._inflight.get(driver, 0) + 1
        runtime = self._runtime_name_for_driver(driver)
        if runtime is not None:
            self._runtime_inflight[runtime] = self._runtime_inflight.get(runtime, 0) + 1

    def on_attempt_end(self, driver: str, *, served: bool) -> None:
        self._inflight[driver] = max(0, self._inflight.get(driver, 0) - 1)
        runtime = self._runtime_name_for_driver(driver)
        if runtime is not None:
            self._runtime_inflight[runtime] = max(0, self._runtime_inflight.get(runtime, 0) - 1)
        if served:
            now = time.monotonic()
            self._last_request[driver] = now
            if runtime is not None:
                self._runtime_last_request[runtime] = now

    def inflight(self, driver: str) -> int:
        return self._inflight.get(driver, 0)

    def runtime_inflight(self, runtime: str) -> int:
        return self._runtime_inflight.get(runtime, 0)

    def idle_seconds(self, runtime: str) -> float | None:
        """How long since this runtime last served a request through the
        gateway — or, for one that never has, since it became ready.
        None when it is not ready and never served."""
        last = self._runtime_last_request.get(runtime)
        since = self._ready_since.get(runtime)
        marks = [m for m in (last, since) if m is not None]
        if not marks:
            return None
        return time.monotonic() - max(marks)

    def driver_idle_seconds(self, driver: str) -> int | None:
        last = self._last_request.get(driver)
        return int(time.monotonic() - last) if last is not None else None

    # --- resolution -------------------------------------------------------

    def _slot_targets(self, model: str) -> tuple[bool, list[str]]:
        """`(configured, targets)` for a model: itself first, then the
        configured targets in order, duplicates dropped."""
        configured: dict[str, Any] | None = None
        raw = self._slots()
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and item.get("model") == model:
                    configured = item
                    break
        targets = [model]
        if configured is not None:
            for target in configured.get("targets") or []:
                if isinstance(target, str) and target and target not in targets:
                    targets.append(target)
        return configured is not None, targets

    def configured_slots(self) -> list[str]:
        raw = self._slots()
        if not isinstance(raw, list):
            return []
        return [
            item["model"]
            for item in raw
            if isinstance(item, dict) and isinstance(item.get("model"), str) and item["model"]
        ]

    def resolve(self, model: str) -> Resolution:
        """The slot for `model`: its own replicas, then each configured
        target's, empty tiers dropped."""
        configured, targets = self._slot_targets(model)
        tiers = [
            _Tier(target=t, backends=list(self._snapshot.by_model.get(t, [])))
            for t in targets
            if self._snapshot.by_model.get(t)
        ]
        return Resolution(model=model, configured=configured, tiers=tiers)

    def _order(self, target: str, backends: list[_Backend]) -> list[_Backend]:
        """Balance one tier.

        Rotate by a per-target cursor so an idle install alternates, then
        under `least_busy` stable-sort by in-flight requests per slot of
        capacity so a busy one fills capacity before queueing on a
        saturated replica. `round_robin` is the rotation alone.
        """
        if len(backends) <= 1:
            return list(backends)
        cursor = self._cursor.get(target, 0)
        self._cursor[target] = cursor + 1
        rotated = backends[cursor % len(backends) :] + backends[: cursor % len(backends)]
        if self._strategy() == ROUND_ROBIN:
            return rotated
        return sorted(rotated, key=lambda b: self.inflight(b.name) / b.parallel_slots)

    def pick(self, resolution: Resolution) -> TieredClient | None:
        """The client to send a request through, or None when nothing in
        the slot is eligible right now.

        Each tier's eligible backends are ordered by the balancer; the
        client walks the first tier, then the next, cascading on
        transport errors, 5xx and timeouts and failing hard on a 4xx.
        """
        tiers: list[list[DriverClient]] = []
        for tier in resolution.tiers:
            eligible = tier.eligible()
            # An empty tier stays in the list, so the response's `tier`
            # counts the slot's tiers rather than the eligible ones.
            tiers.append([b.client for b in self._order(tier.target, eligible)] if eligible else [])
        if not any(tiers):
            return None
        return TieredClient(name=resolution.model, tiers=tiers, hooks=self)

    def backends_for(self, model: str) -> list[_Backend]:
        return self.resolve(model).backends()

    def runtime_for(self, model: str, driver: str | None = None) -> str | None:
        """Name of the engine runtime behind `driver` — or, without one,
        behind the only runtime serving `model`. Absent for a hosted or
        CLI backend, which has no runtime of ours."""
        for backend in self.resolve(model).backends():
            if (driver is None or backend.name == driver) and backend.runtime is not None:
                return backend.runtime.name
        return None

    def agent_url_for(self, facts: _RuntimeFacts) -> str:
        """Which agent to talk to about this runtime: its node's, else
        the default."""
        agents = self._snapshot.agents
        if facts.node is not None and facts.node in agents:
            return agents[facts.node]
        return agents.get(None) or next(iter(agents.values()), self._agent_url)

    def runtimes(self) -> list[_RuntimeFacts]:
        return list(self._snapshot.runtimes.values())

    def known_models(self) -> list[str]:
        """Every model a request could name: each served model id, plus
        each configured slot that resolves to anything."""
        names = set(self._snapshot.by_model)
        for slot in self.configured_slots():
            if self.resolve(slot).has_backends():
                names.add(slot)
        return sorted(names)

    def is_empty(self) -> bool:
        return not self._snapshot.by_model

    # --- read models ------------------------------------------------------

    def as_model_list(self) -> list[Model]:
        """The OpenAI-compatible model list.

        Two drivers serving one model produce one entry: replicas are a
        routing detail, and a client should not have to know how many
        GPUs are behind a name. A model whose runtimes are all asleep is
        still listed — a client has to be able to name it to wake it —
        with `ready_backends: 0` and `on_demand` saying what a request
        will meet.
        """
        out: list[Model] = []
        for model_id in self.known_models():
            resolution = self.resolve(model_id)
            backends = resolution.backends()
            providers = {b.info.provider for b in backends if b.info.provider}
            eligible = resolution.eligible_backends()
            out.append(
                Model(
                    id=model_id,
                    object="model",
                    # OpenAI sends an organisation here; we send the
                    # provider. `local` when the driver didn't name one,
                    # which is the openai_compat_http local-engine case.
                    owned_by=sorted(providers)[0] if len(providers) == 1 else "local",
                    x_eugene_plexus=ModelRoutingInfo(
                        drivers=[b.name for b in backends],
                        backends=sorted({_backend_kind(b) for b in backends}),
                        context_length=_smallest_context(backends),
                        tiers=[[b.name for b in t.backends] for t in resolution.tiers],
                        ready_backends=len(eligible),
                        on_demand=not eligible and bool(resolution.startable()),
                    ),
                )
            )
        return out

    def as_driver_health(self) -> list[DriverHealth]:
        """Per-driver snapshot for the admin surface, reachable first."""
        out = [
            DriverHealth(
                name=b.name,
                reachable=True,
                url=b.url,  # type: ignore[arg-type]
                backend=_backend_kind(b),
                modelId=b.info.modelId,
                # Straight off the driver's /v1/info: which supervised
                # runtime it follows. A reachable driver serving nothing,
                # next to a `ready` runtime routed to by nobody, is the
                # visible shape of a mis-wired install.
                runtime=b.info.runtime,
                version=b.info.version,
            )
            for b in sorted(self._snapshot.reachable, key=lambda b: b.name)
        ]
        out += [
            DriverHealth(
                name=u.name,
                reachable=False,
                url=u.url,  # type: ignore[arg-type]
                error=u.error,
            )
            for u in sorted(self._snapshot.unreachable, key=lambda u: u.name)
        ]
        return out

    def as_routing_view(self) -> RoutingTableView:
        """The table opened up: every slot, its tiers, and what each
        backend is doing right now."""
        slots: list[RoutingSlotView] = []
        for model_id in self.known_models():
            resolution = self.resolve(model_id)
            slots.append(
                RoutingSlotView(
                    model=model_id,
                    configured=resolution.configured,
                    tiers=[
                        RoutingTierView(
                            target=tier.target,
                            backends=[self._backend_view(b) for b in tier.backends],
                        )
                        for tier in resolution.tiers
                    ],
                )
            )
        return RoutingTableView(
            refreshed_at=self._snapshot.refreshed_at,
            load_balancing=str(self._strategy() or LEAST_BUSY),
            slots=slots,
            unreachable_drivers=sorted(u.name for u in self._snapshot.unreachable),
        )

    def _backend_view(self, backend: _Backend) -> RoutingBackendView:
        facts = backend.runtime
        return RoutingBackendView(
            driver=backend.name,
            url=backend.url,  # type: ignore[arg-type]
            eligible=backend.eligible,
            ineligible_reason=backend.ineligible_reason,
            in_flight=self.inflight(backend.name),
            parallel_slots=backend.parallel_slots,
            runtime=facts.name if facts else None,
            runtime_status=facts.status if facts else None,
            node=facts.node if facts else None,
            stop_reason=facts.stop_reason if facts else None,
            idle_unload_seconds=facts.idle_unload_seconds if facts else None,
            start_on_demand=facts.start_on_demand if facts else None,
            idle_seconds=self.driver_idle_seconds(backend.name),
        )


def _backend_kind(backend: _Backend) -> BackendKind:
    """Bridge inference-driver.yaml's BackendKind to gateway.yaml's.

    Same wire values, two generated classes — the price of "components
    share schemas, not code", and cheaper than the coupling a shared
    module would create. Crossing via `.value` keeps the boundary
    explicit instead of hiding it behind a cast.
    """
    return BackendKind(backend.info.backend.value)


def _smallest_context(backends: list[_Backend]) -> int | None:
    """Smallest context window among the backends serving one model.

    The honest number to report, since a request may land on any of
    them — promising the largest would mean a prompt that fits the
    advertised window can still be rejected.

    Each supervised runtime counts as one of those numbers. An
    `openai_compat_http` driver reports no capabilities at all (it can
    only see an HTTP endpoint), so without this the local-engine case —
    the one where we *do* know the answer, because the engine told the
    agent after it loaded — would report nothing.
    """
    lengths = [
        b.info.capabilities.maxContextTokens
        for b in backends
        if b.info.capabilities is not None and b.info.capabilities.maxContextTokens
    ]
    lengths += [
        b.runtime.context_length
        for b in backends
        if b.runtime is not None and b.runtime.context_length
    ]
    return min(lengths) if lengths else None


__all__ = [
    "LEAST_BUSY",
    "READY",
    "ROUND_ROBIN",
    "STOPPED",
    "BackendKind",
    "Resolution",
    "RoutingTable",
]
