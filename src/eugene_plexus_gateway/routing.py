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
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from ._generated.driver_models import DriverInfo
from ._generated.models import (
    BackendKind,
    ControlRootView,
    DriverHealth,
    Model,
    ModelRoutingInfo,
    RoutingBackendView,
    RoutingSlotView,
    RoutingTableView,
    RoutingTierView,
    Surface,
)
from ._http import internal_client
from .config import DEFAULT_REQUEST_TIMEOUT_SECONDS
from .driver_client import DriverClient, HttpDriverClient, TieredClient
from .driver_client import Key as DriverKey
from .metrics import AttemptRow, CandidateRow

# Deadline for one topology read (an agent's `/v1/components`, a control
# root's `/v1/nodes`). Short because a refresh can run inside a request
# that found nothing eligible, so this is latency a user feels.
_TOPOLOGY_TIMEOUT = 5.0

log = logging.getLogger(__name__)

# Per-request collection of attempt rows (M8). A contextvar rather than
# state on the table, because the table is shared and long-lived while
# these rows belong to one request; and because the DriverClient
# protocol has no room for a request id to key them by.
_attempts: ContextVar[list[AttemptRow] | None] = ContextVar("ep_attempts", default=None)


@contextmanager
def collect_attempts() -> Iterator[list[AttemptRow]]:
    """Collect the attempts made inside this block.

    Outside such a block the hooks record nothing, so `generate()` stays
    usable from tests and from anything that is not a served request.
    """
    # The admission middleware owns the HTTP lifetime. Route-local collection
    # shares its rows so cancellation before a route records its result still
    # retains the actual attempts (including an interrupted stream).
    existing = _attempts.get()
    if existing is not None:
        yield existing
        return
    rows: list[AttemptRow] = []
    token = _attempts.set(rows)
    try:
        yield rows
    finally:
        _attempts.reset(token)


READY = "ready"
STOPPED = "stopped"
LEAST_BUSY = "least_busy"
ROUND_ROBIN = "round_robin"

# Statuses that mean "this runtime is on its way to `ready`, wait rather
# than despair". Deliberately a set of strings and not a generated enum:
# the gateway reads another node's `/v1/runtimes` over HTTP and does not
# codegen `agent.yaml`, so a newer agent can and does report statuses
# this build has never heard of. An unknown one is carried through as
# text everywhere else here, which is what keeps that safe.
#
# `copying` joined at node-local-model-copy.md step 2, BEFORE any agent
# could emit it: a node making its own copy of a 25 GB model sits there
# for minutes with no process spawned, and without this it is neither
# `startable()` (not stopped) nor `waking()` — so a request for that
# model would have been told "none of its runtimes asked to be started
# on demand", which is both false and unactionable.
COMING_UP = frozenset({"copying", "starting", "loading"})

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


#: How this table identifies a runtime, and a driver.
#:
#: **`name` alone is not an identity** (R1.6, review §6.1 #8).
#: `RuntimeSpec.name` is unique per agent, not per install, and the UI
#: names a runtime after the model and a companion driver after the
#: runtime -- so one model launched on two machines is `qwen` and
#: `qwen-driver` on both. Merged by name, the last agent read won: the
#: surviving entry carried the wrong node, so an idle unload decided for
#: A was sent to B's agent; B `ready` masked A `stopped`, so A's driver
#: stayed eligible and 502'd; and both replicas shared one in-flight
#: counter and one idle clock, which means B's traffic kept A looking
#: recently used and the idle pass, iterating the merged map, never
#: considered one of the two at all.
#:
#: `None` is the unenrolled single-host case, where there is no node
#: name anywhere and one bare name really is unique.
#:
#: The rest of the system is already written this way: control keys
#: placements by `(node, name)` and sorts by it, and the metrics rollup's
#: primary key already carries `node` -- only the value was wrong,
#: because it came from the merged survivor.
Key = DriverKey


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

    @property
    def key(self) -> Key:
        """`(node, name)` — how every map in this module keys it."""
        return (self.node, self.name)


@dataclass
class _Backend:
    """One reachable driver, what it told us it serves, and the runtime
    behind it when the agent supervises one."""

    name: str
    url: str
    client: DriverClient
    info: DriverInfo
    #: Which machine's agent reported this driver. `None` on an
    #: unenrolled single-host install. Part of its identity for the same
    #: reason a runtime's node is part of that runtime's -- see `Key`.
    node: str | None = None
    runtime: _RuntimeFacts | None = None
    stopping: Mapping[Key, int] = field(default_factory=dict)
    """The routing table's own reservation counter, shared by reference
    with every backend in the snapshot — so a runtime the lifecycle
    manager is *in the middle of stopping* stops being routable at the
    moment the decision is taken rather than at the next refresh. See
    `RoutingTable.stopping`."""

    @property
    def eligible(self) -> bool:
        """Routable right now. A driver following a runtime is eligible
        only when that runtime is `ready`; one following nothing of ours
        is eligible whenever it is reachable, which it is by being here.

        And not while a stop for it is in flight: `idle_pass` and
        `_evict_for` decide on a runtime with nothing in flight and then
        *await* an HTTP call to the agent, and a request arriving inside
        that await used to be routed to an engine that was going away.
        On the streamed path M10's commit-point rule turns that from a
        retry into a visible truncation."""
        if self.runtime is not None and self.stopping.get(self.runtime.key, 0) > 0:
            return False
        return self.runtime is None or self.runtime.status == READY

    @property
    def key(self) -> Key:
        """`(node, name)`. Two machines running one model produce two
        drivers with one name; the counters must not share a bucket."""
        return (self.node, self.name)

    @property
    def ineligible_reason(self) -> str | None:
        if self.eligible:
            return None
        assert self.runtime is not None
        if self.stopping.get(self.runtime.key, 0) > 0:
            return f"runtime {self.runtime.name!r} is being stopped"
        return f"runtime {self.runtime.name!r} is {self.runtime.status or 'unknown'}"

    @property
    def embeds(self) -> bool:
        """Whether this backend serves the embeddings surface.

        The driver determines it from the backend and reports it on
        `/v1/info`; absent reads as False, which is the honest default
        for a capability nobody has confirmed."""
        caps = self.info.capabilities
        return caps is not None and bool(caps.embeddings)

    @property
    def chats(self) -> bool:
        """Whether this backend serves chat completions.

        **Not simply `not embeds`.** Measured: an Ollama runner is one
        or the other, but `llama-server` given `--embedding` still
        serves chat perfectly well. A backend that says nothing about
        embeddings is a chat backend, which is every driver that existed
        before this surface did -- so the default has to be True or
        re-pinning would silently unroute every model in the install.
        """
        return not self.embeds or self.info.capabilities is None

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


@dataclass(frozen=True)
class ControlRootFacts:
    """Where the node list came from on the last refresh, and whether it
    answered. `source` is `config` (the `controlUrl` field, used as
    given), `agent` (derived from this gateway's own agent) or `none`
    (a single-host install). `derived_from` is which of the agent's two
    answers it was -- `node` for the root the node is enrolled to,
    `components` for a `control` component it declares -- for the log."""

    source: str = "none"
    url: str | None = None
    reachable: bool = False
    error: str | None = None
    nodes: int | None = None
    derived_from: str | None = None


@dataclass
class _Snapshot:
    """One resolved view of the world. Replaced wholesale on refresh so a
    request never sees a half-rebuilt table."""

    by_model: dict[str, list[_Backend]] = field(default_factory=dict)
    reachable: list[_Backend] = field(default_factory=list)
    unreachable: list[_Unreachable] = field(default_factory=list)
    # Keyed by `(node, runtime name)`. Neither half is unique on its
    # own: an alias is shared by every replica of a model, and a NAME is
    # shared by the same model launched on two machines -- see `Key`.
    runtimes: dict[Key, _RuntimeFacts] = field(default_factory=dict)
    # node name -> that node's agent URL. `None` is the default agent,
    # which is the only agent when no control root is configured.
    agents: dict[str | None, str] = field(default_factory=dict)
    control_root: ControlRootFacts = field(default_factory=ControlRootFacts)
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

    def restricted(self, allowed: set[str] | None, *, local_only: bool = False) -> Resolution:
        if allowed is None and not local_only:
            return self
        return Resolution(
            self.model,
            self.configured,
            [
                _Tier(
                    t.target,
                    [
                        b
                        for b in t.backends
                        if (allowed is None or (t.target in allowed and b.info.modelId in allowed))
                        and (
                            not local_only
                            or (b.info.locality == "local" and b.info.localOnlyEnforced is True)
                        )
                    ],
                )
                for t in self.tiers
            ],
        )

    def surfaces(self) -> list[str]:
        backends = self.backends()
        return (["chat"] if any(b.chats for b in backends) else []) + (
            ["embeddings"] if any(b.embeds for b in backends) else []
        )

    def has_backends(self) -> bool:
        return any(t.backends for t in self.tiers)

    def backends(self) -> list[_Backend]:
        return [b for t in self.tiers for b in t.backends]

    def eligible_backends(self) -> list[_Backend]:
        return [b for t in self.tiers for b in t.eligible()]

    def runtimes(self) -> list[_RuntimeFacts]:
        """Distinct runtimes behind this slot, in tier order.

        By `(node, name)`: deduplicating on the name alone dropped one
        of two replicas of one model on two machines, so `startable()`
        below could only ever wake whichever of them came first.
        """
        seen: set[Key] = set()
        out: list[_RuntimeFacts] = []
        for backend in self.backends():
            facts = backend.runtime
            if facts is not None and facts.key not in seen:
                seen.add(facts.key)
                out.append(facts)
        return out

    def startable(self) -> list[_RuntimeFacts]:
        """Stopped runtimes that asked to be woken on demand, tier order."""
        return [r for r in self.runtimes() if r.status == STOPPED and r.start_on_demand]

    def waking(self) -> list[_RuntimeFacts]:
        """Runtimes coming up on their own, so a caller should wait."""
        return [r for r in self.runtimes() if r.status in COMING_UP]


class RoutingTable:
    """Resolves a requested model to the backends that can serve it."""

    def __init__(
        self,
        *,
        agent_url: str,
        service_token: str | None = None,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        refresh_seconds: float = 15.0,
        slots: Callable[[], Any] | None = None,
        strategy: Callable[[], Any] | None = None,
        control_url: str | Callable[[], Any] | None = None,
    ) -> None:
        self._agent_url = agent_url.rstrip("/")
        # Read live on every refresh, for the same reason `slots` and
        # `strategy` are: `controlUrl` is a config field whose own
        # description promises it "takes effect on the next routing
        # refresh", and whose `requiresRestart` is false. Captured as a
        # string it did neither -- setting it on a running gateway
        # changed nothing until someone restarted the process, which is
        # how a two-machine install came up with its worker enrolled,
        # reachable, and invisible to routing (2026-09-11). A plain
        # string is still accepted, because most callers have one.
        self._control_url_source: Callable[[], Any] = (
            control_url if callable(control_url) else (lambda: control_url)
        )
        self._service_token = service_token
        self._request_timeout = request_timeout_seconds
        self._refresh_seconds = refresh_seconds
        # Read live on every resolve, so a PATCH to `modelSlots` or
        # `loadBalancing` takes effect without a restart.
        self._slots = slots or (lambda: [])
        self._strategy = strategy or (lambda: LEAST_BUSY)
        self._snapshot = _Snapshot(agents={None: self._agent_url})
        # Clients are cached by (node, name, url) and reused across
        # refreshes. Rebuilding an httpx.AsyncClient every 15s would
        # throw away the connection pool and leak sockets. The node is in
        # the key because the name is not unique across machines.
        self._clients: dict[tuple[str | None, str, str], HttpDriverClient] = {}
        #: The one client this table reads topology with -- five to seven
        #: GETs per refresh, every 15 s, and again inside whatever request
        #: triggered `refresh_if_stale`. Built once here rather than per
        #: GET: `httpx.AsyncClient()` with no `verify=` parses certifi's
        #: PEM bundle, ~104 ms of synchronous CPU on the event loop, so a
        #: refresh used to spend ~0.6 s blocking the loop that serves
        #: every in-flight completion. `trust_env=False` because every
        #: URL it dials is this install's own agent or control root, and
        #: a user's `HTTP_PROXY` -- which the Windows logon task
        #: inherits -- would otherwise swallow the lot and leave the
        #: install at `starting` while actually serving.
        self._json_client = internal_client(timeout=_TOPOLOGY_TIMEOUT)
        self._task: asyncio.Task[None] | None = None
        # One refresh at a time; on-demand callers share it.
        self._refreshing: asyncio.Task[None] | None = None
        # And refreshes never overlap. Three things call `refresh()` — the
        # periodic loop, the lifecycle manager after a stop, and a request
        # that found nothing eligible — and two in flight at once is a
        # lost update: the M7 live run saw a periodic refresh that began
        # before an idle unload finish AFTER the post-unload one and put a
        # `ready` runtime back over a `stopped` one, so the next request
        # went to a driver whose engine was gone. Snapshots now land in
        # the order the refreshes began.
        self._refresh_lock = asyncio.Lock()
        # Demand, by driver and by runtime. Monotonic seconds; never
        # persisted, never replicated — a promoted control root re-reads
        # what is loaded from the agents, not from here.
        # Every one of these is keyed by `(node, name)` -- see `Key`.
        self._inflight: dict[Key, int] = {}
        self._last_request: dict[Key, float] = {}
        self._runtime_inflight: dict[Key, int] = {}
        self._runtime_last_request: dict[Key, float] = {}
        self._ready_since: dict[Key, float] = {}
        # By model, which really is install-wide: the balancer's rotation
        # cursor for one slot.
        self._cursor: dict[str, int] = {}
        # Runtimes with a stop in flight, counted rather than flagged
        # because `idle_pass` and `_evict_for` can both be stopping one.
        # Shared by reference with every `_Backend` in the snapshot; see
        # `stopping()`.
        self._stopping: dict[Key, int] = {}
        # The last SUCCESSFUL read from each node, by node name. An agent
        # that does not answer is not an agent with nothing on it: its
        # entry here is what the refresh uses instead of an empty list,
        # which is the whole of review §6.1 #9 and §6.2 #18.
        self._last_entries: dict[str | None, list[tuple[str, str]]] = {}
        self._last_facts: dict[str | None, dict[Key, _RuntimeFacts]] = {}
        # `(url, source)` last written to the log, so the control root is
        # announced when it is first found and when it changes -- not on
        # every refresh.
        self._announced_control_root: tuple[str | None, str] | None = None
        # The last "did not answer" reason written to the log at WARNING,
        # so a root that is sealed for an hour costs one line, not one
        # every refresh -- and its recovery costs one more.
        self._announced_unreachable: str | None = None

    @property
    def _control_url(self) -> str | None:
        """The control root's address as *configured* right now -- the
        `controlUrl` field, which is the operator's override. `None` means
        the gateway works it out itself; see `_derive_control_url`.

        Normalized on every read rather than at construction, so a
        `PATCH /v1/config` reaches the next refresh. An empty string and
        a whitespace-only value both mean "not set", which is what a UI
        that clears the field sends.
        """
        value = self._control_url_source()
        text = str(value).strip() if value is not None else ""
        return text.rstrip("/") or None

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
        with contextlib.suppress(BaseException):
            await self._json_client.aclose()
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
        body, error = await self._get_json_or_error(url)
        if error is not None:
            log.debug("could not read %s: %s", url, error)
        return body

    async def _get_json_or_error(self, url: str) -> tuple[Any | None, str | None]:
        """The body, or why there is none.

        The reason is the HTTP status plus the problem's title when the
        far side sent one -- `503 Locked` is what a sealed control root
        says -- else the transport error. Kept short, because it ends up
        in a 404 an operator reads.
        """
        try:
            response = await self._json_client.get(url, headers=self._headers())
        except httpx.HTTPError as e:
            return None, str(e) or type(e).__name__
        if response.status_code >= 400:
            title: str | None = None
            try:
                problem = response.json()
                detail = problem.get("detail") if isinstance(problem, dict) else None
                inner = detail if isinstance(detail, dict) else problem
                title = inner.get("title") if isinstance(inner, dict) else None
            except ValueError:
                title = None
            code = str(response.status_code)
            return None, f"{code} {title}" if isinstance(title, str) and title else f"HTTP {code}"
        try:
            return response.json(), None
        except ValueError as e:
            return None, f"not JSON: {e}"

    async def _derive_control_url(self) -> tuple[str | None, str | None]:
        """Ask this gateway's own agent where the install's control root is.

        Two places a host already knows it: `GET /v1/node` names the root
        this node is enrolled to, which is the answer on a control host
        (its agent enrolls too, since M9) and on a worker; failing that,
        `GET /v1/components` names the `control` component the agent
        declares, which is the answer on a control host nobody has
        enrolled yet -- first boot seeds control, gateway and library
        before the wizard's Start. Neither means a single-host install.

        Returns `(url, "node" | "components")`, or `(None, None)`. Read
        on every refresh, like `controlUrl` itself, so an enrollment or a
        promotion reaches the next refresh with no restart.
        """
        node = await self._get_json(f"{self._agent_url}/v1/node")
        if isinstance(node, dict) and node.get("enrolled"):
            url = node.get("controlUrl")
            if isinstance(url, str) and url.strip():
                return url.strip().rstrip("/"), "node"
        body = await self._get_json(f"{self._agent_url}/v1/components")
        components = body.get("components") if isinstance(body, dict) else None
        for entry in components if isinstance(components, list) else []:
            if isinstance(entry, dict) and entry.get("kind") == "control":
                url = entry.get("url")
                if isinstance(url, str) and url.strip():
                    return url.strip().rstrip("/"), "components"
        return None, None

    def _announce_control_root(self, url: str | None, source: str, via: str | None) -> None:
        """One log line when the control root is first found or changes."""
        key = (url, source)
        if key == self._announced_control_root:
            return
        self._announced_control_root = key
        if url is None:
            log.info(
                "no control root: the agent at %s is neither enrolled nor declares one, so this "
                "is a single-host install (set controlUrl to override)",
                self._agent_url,
            )
        elif source == "config":
            log.info("control root %s, from the controlUrl config field (used as given)", url)
        else:
            what = (
                "the root this node is enrolled to"
                if via == "node"
                else "the control component its agent declares"
            )
            log.info(
                "control root %s, derived from the agent at %s: %s", url, self._agent_url, what
            )

    async def discover_agents(self) -> dict[str | None, str]:
        """Which agents to read topology from. See `_discover`."""
        agents, _facts = await self._discover()
        return agents

    async def _discover(self) -> tuple[dict[str | None, str], ControlRootFacts]:
        """Which agents to read topology from, and where that answer came from.

        The control root is the `controlUrl` field when an operator set
        one -- used as given, even when it is wrong, because an expert
        naming an address gets that address -- and otherwise whatever
        this gateway's own agent knows (`_derive_control_url`). Until
        2026-09-13 nothing set the field at all and nothing derived it,
        so every multi-host install came up with its workers enrolled,
        reachable, and invisible to routing.

        Without a control root: the one configured agent. With one: the
        node list it reports, each with its agent URL — the lookup that
        closes the loopback:8079 assumption for anything that has to
        reach a runtime on another host. A control root that does not
        answer leaves the previous agent map in place: management being
        down must not empty the routing table -- and the facts say it
        did not answer, so the admin view and the no-models 404 can.
        """
        configured = self._control_url
        via: str | None = None
        if configured is not None:
            control_url: str | None = configured
            source = "config"
        else:
            control_url, via = await self._derive_control_url()
            source = "agent" if control_url is not None else "none"
        self._announce_control_root(control_url, source, via)
        if control_url is None:
            return {None: self._agent_url}, ControlRootFacts()

        body, error = await self._get_json_or_error(f"{control_url}/v1/nodes")
        nodes = body.get("nodes") if isinstance(body, dict) else None
        if not isinstance(nodes, list):
            reason = error or "the response carried no node list"
            if reason != self._announced_unreachable:
                log.warning(
                    "control root at %s did not answer /v1/nodes (%s); keeping the previous "
                    "agent map until it does",
                    control_url,
                    reason,
                )
                self._announced_unreachable = reason
            else:
                log.debug("control root at %s still not answering (%s)", control_url, reason)
            previous = dict(self._snapshot.agents) or {None: self._agent_url}
            facts = ControlRootFacts(
                source=source,
                url=control_url,
                reachable=False,
                error=reason,
                nodes=self._snapshot.control_root.nodes,
                derived_from=via,
            )
            return previous, facts
        agents: dict[str | None, str] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            name, url = node.get("name"), node.get("url")
            if isinstance(name, str) and name and isinstance(url, str) and url:
                agents[name] = url.rstrip("/")
        listed = len(agents)
        if self._announced_unreachable is not None:
            log.info("control root at %s answers again: %d node(s)", control_url, listed)
            self._announced_unreachable = None
        if not agents:
            # An enrolled-nothing control root: fall back to the agent we
            # were told about, which is the single-host case anyway.
            agents = {None: self._agent_url}
        facts = ControlRootFacts(
            source=source, url=control_url, reachable=True, nodes=listed, derived_from=via
        )
        return agents, facts

    async def refresh(self) -> None:
        async with self._refresh_lock:
            await self._refresh_locked()

    async def _refresh_locked(self) -> None:
        agents, control_root = await self._discover()

        # A node that has left the install keeps nothing. The per-node
        # caches exist so a missed READ does not erase a node; a node
        # the control root no longer lists really is gone.
        for known in list(self._last_entries):
            if known not in agents:
                self._last_entries.pop(known, None)
        for known in list(self._last_facts):
            if known not in agents:
                self._last_facts.pop(known, None)

        # Every agent, concurrently: its drivers and its runtimes.
        fetched = await asyncio.gather(
            *(self._fetch_agent(node, url) for node, url in agents.items()),
            return_exceptions=True,
        )
        # `(node, name, url)` per driver, so two machines running one
        # model produce two entries rather than one that overwrites the
        # other.
        entries: list[tuple[str | None, str, str]] = []
        runtimes: dict[Key, _RuntimeFacts] = {}
        for node, result in zip(agents, fetched, strict=True):
            if isinstance(result, BaseException):
                log.warning("reading an agent raised unexpectedly: %s", result)
                continue
            agent_entries, agent_runtimes = result
            entries.extend((node, name, url) for name, url in agent_entries)
            runtimes.update(agent_runtimes)

        # Drop clients for drivers that left the topology.
        live_keys = set(entries)
        for client_key in list(self._clients):
            if client_key not in live_keys:
                client = self._clients.pop(client_key)
                log.info("driver %r left the topology; closing its client", client_key[1])
                with contextlib.suppress(BaseException):
                    await client.aclose()

        results = await asyncio.gather(
            *(self._probe(node, name, url) for node, name, url in entries),
            return_exceptions=True,
        )

        snapshot = _Snapshot(runtimes=runtimes, agents=agents, control_root=control_root)
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
        # Per NODE, so a driver on node A cannot be joined to a runtime
        # on node B by an alias they both carry -- which is the normal
        # state of a replica pair.
        by_alias: dict[Key, _RuntimeFacts | None] = {}
        for facts in runtimes.values():
            if facts.alias:
                alias_key = (facts.node, facts.alias)
                by_alias[alias_key] = None if alias_key in by_alias else facts

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
                # **On this driver's own node.** `/v1/info` reports a
                # bare name, and a bare name matches a replica on another
                # machine just as well -- which is how a driver on node A
                # came to follow node B's runtime and read its status.
                backend.runtime = runtimes.get((backend.node, backend.info.runtime))
                if backend.runtime is None:
                    # Raised from DEBUG (M7's record asked for this):
                    # "routable on faith" is the eligibility rule saying
                    # yes to an engine whose state it does not know, and
                    # it is worth a line an operator will see.
                    log.warning(
                        "driver %r on node %r follows runtime %r, which that node does not "
                        "report; treating it as routable on faith",
                        backend.name,
                        backend.node,
                        backend.info.runtime,
                    )
            else:
                backend.runtime = by_alias.get((backend.node, model_id))
            snapshot.by_model.setdefault(model_id, []).append(backend)

        # Stable base order per model; the balancer rotates from here.
        # By `(node, name)`, because the names of two replicas of one
        # model on two machines are equal and `sort` would then leave
        # their order to whichever agent answered first.
        for backends in snapshot.by_model.values():
            backends.sort(key=lambda b: (b.node or "", b.name))

        # Ready-since, so a runtime that loaded and was never asked for
        # anything still counts as idle from the moment it became ready.
        now = time.perf_counter()
        for key, facts in runtimes.items():
            if facts.status == READY:
                self._ready_since.setdefault(key, now)
            else:
                self._ready_since.pop(key, None)
        for key in list(self._ready_since):
            if key not in runtimes:
                self._ready_since.pop(key, None)

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
    ) -> tuple[list[tuple[str, str]], dict[Key, _RuntimeFacts]]:
        """One node's two reads, with the last successful answer kept.

        **A node that did not answer is not a node with nothing on it.**
        Both reads report whether they got an answer, and a failure
        reuses what that node last said rather than the empty value —
        which the rest of the refresh would otherwise act on as fact
        (review §6.1 #9 and §6.2 #18). The control root's own read has
        worked this way since the day it was written, eighty lines
        above; these two did not.

        Stale facts are bounded by the thing that actually checks
        liveness: every entry is probed at its own `/v1/info` below, so
        a driver on a node whose agent is down lands in `unreachable`
        and is not routed to. What survives is the *shape* of that node
        — which engines exist and what state they were last in — which
        is exactly what a missed read should not be allowed to erase.
        """
        (entries, entries_ok), (runtimes, runtimes_ok) = await asyncio.gather(
            self._fetch_driver_entries(url),
            self._fetch_runtime_facts(url, node),
        )
        if entries_ok:
            self._last_entries[node] = entries
        else:
            entries = list(self._last_entries.get(node, []))
        if runtimes_ok:
            self._last_facts[node] = runtimes
        else:
            runtimes = dict(self._last_facts.get(node, {}))
        return entries, runtimes

    async def fetch_driver_entries(self) -> list[tuple[str, str]]:
        """`(name, url)` for every inference-driver across every agent.

        Public because `POST /v1/config/test` reads the topology fresh
        rather than off the last refresh — the point of a Test button is
        the world as it is now. Which is also why this one does **not**
        fall back to the last successful read: a Test button that
        answers from a cache is answering the wrong question.
        """
        agents = await self.discover_agents()
        out: list[tuple[str, str]] = []
        for url in agents.values():
            entries, _ = await self._fetch_driver_entries(url)
            out.extend(entries)
        return out

    async def _fetch_driver_entries(self, agent_url: str) -> tuple[list[tuple[str, str]], bool]:
        """`(entries, answered)`. Entries are `(name, url)` for each
        inference-driver one agent declares, at the address a peer
        reaches it — `Component.advertiseUrl` when the agent stamps one,
        else `url`.

        **The flag is the point.** An empty list used to mean both "this
        agent supervises no drivers" and "this agent did not answer", and
        the refresh acts on the difference: it closes the HTTP clients of
        drivers it believes have left, *under whatever requests are using
        them*, and un-routes their models. One 401 from clock skew, one
        5 s timeout, or the agent restart the Reach switch performs on
        purpose was enough (review §6.1 #9). The caller keeps the last
        successful answer for a node that did not answer this time; the
        per-driver `/v1/info` probe below is the real liveness check and
        still runs.
        """
        body = await self._get_json(f"{agent_url}/v1/components")
        if body is None:
            log.warning("could not read %s/v1/components; keeping what it last declared", agent_url)
            return [], False
        components = body.get("components") if isinstance(body, dict) else None
        if not isinstance(components, list):
            return [], False
        out: list[tuple[str, str]] = []
        for entry in components:
            if not isinstance(entry, dict) or entry.get("kind") != "inference-driver":
                continue
            # `advertiseUrl` is where a peer on another host reaches the
            # component — the agent's advertise host with that component's
            # port — and `url` is what the agent binds and probes, which is
            # loopback for everything an agent spawns. Prefer the first; a
            # single-host agent sends no advertiseUrl and the two would say
            # the same thing. This one line is what makes a companion
            # driver on another node routable from here (M7).
            name = entry.get("name")
            url = entry.get("advertiseUrl") or entry.get("url")
            if isinstance(name, str) and name and isinstance(url, str) and url:
                out.append((name, url.rstrip("/")))
        return out, True

    async def _fetch_runtime_facts(
        self, agent_url: str, node: str | None
    ) -> tuple[dict[Key, _RuntimeFacts], bool]:
        """`(facts, answered)`. Facts are `(node, runtime name)` -> what the agent
        knows about it: where the `ready` gate, attribution, context,
        capacity and the lifecycle policy fields all come from.

        **The flag is the point, and this is the wider of the two.** An
        empty map leaves every driver on that node with `runtime is
        None`, which the eligibility rule reads as "follows nothing of
        ours, route to it whenever it is reachable" — so a stopped, a
        loading and a crashed engine all become routable, from one
        missed read (review §6.2 #18). Its widest trigger is not the
        post-unload window this project diagnosed in 2026-09-11 but a
        401 from clock skew, a 5 s timeout, or the agent restart the
        Reach switch performs on purpose. A failed read means **keep the
        previous facts**, not keep the faith.
        """
        body = await self._get_json(f"{agent_url}/v1/runtimes")
        if body is None:
            log.warning(
                "could not read %s/v1/runtimes; keeping the facts from its last answer", agent_url
            )
            return {}, False
        runtimes = body.get("runtimes") if isinstance(body, dict) else None
        if not isinstance(runtimes, list):
            return {}, False
        facts: dict[Key, _RuntimeFacts] = {}
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
            owner = reported_node if isinstance(reported_node, str) and reported_node else node
            facts[(owner, name)] = _RuntimeFacts(
                name=name,
                alias=entry.get("modelAlias") if isinstance(entry.get("modelAlias"), str) else None,
                status=str(entry["status"]) if entry.get("status") else None,
                node=owner,
                url=entry.get("url") if isinstance(entry.get("url"), str) else None,
                context_length=context if isinstance(context, int) and context > 0 else None,
                parallel_slots=slots if isinstance(slots, int) and slots > 0 else None,
                idle_unload_seconds=idle if isinstance(idle, int) and idle > 0 else None,
                start_on_demand=bool(entry.get("startOnDemand")),
                stop_reason=str(entry["stopReason"]) if entry.get("stopReason") else None,
                spec={k: entry[k] for k in _SPEC_FIELDS if entry.get(k) is not None},
            )
        return facts, True

    async def _probe(self, node: str | None, name: str, url: str) -> _Backend | _Unreachable:
        client = self._clients.get((node, name, url))
        if client is None:
            client = HttpDriverClient(
                name=name,
                base_url=url,
                timeout_seconds=self._request_timeout,
                service_token=self._service_token,
                node=node,
            )
            self._clients[(node, name, url)] = client
        try:
            info = await client.info()
        except (httpx.HTTPError, ValueError) as e:
            return _Unreachable(name=name, url=url, error=str(e))
        # The reservation map is passed BY REFERENCE: every backend in
        # every snapshot reads the one the table owns, so a stop taken
        # after this refresh is visible to a backend built before it.
        return _Backend(
            name=name, url=url, client=client, info=info, node=node, stopping=self._stopping
        )

    # --- demand -------------------------------------------------------------
    #
    # The `RoutingHooks` protocol `TieredClient` calls around every attempt.

    def _runtime_key_for_driver(self, key: Key) -> Key | None:
        """The runtime behind this driver, by `(node, name)` both ways.

        Scanning by bare driver name returned the FIRST backend with
        that name, which on a two-machine replica pair is a coin flip --
        so an attempt against node B's driver could be counted against
        node A's runtime, and undone against it too.
        """
        for backend in self._snapshot.reachable:
            if backend.key == key:
                return backend.runtime.key if backend.runtime is not None else None
        return None

    def on_attempt_start(self, driver: str, *, node: str | None = None) -> Key | None:
        """Count this attempt, and answer with the runtime it counted
        against so the other end undoes exactly this.

        Resolving the runtime again at the end would resolve it against
        whatever snapshot has since been installed, and the two answers
        need not agree -- see `RoutingHooks.on_attempt_start`.

        `node` arrives from the client, which got it from the agent that
        reported the driver; without it two machines' replicas of one
        model share every counter here.
        """
        key: Key = (node, driver)
        self._inflight[key] = self._inflight.get(key, 0) + 1
        runtime = self._runtime_key_for_driver(key)
        if runtime is not None:
            self._runtime_inflight[runtime] = self._runtime_inflight.get(runtime, 0) + 1
        return runtime

    def on_attempt_end(
        self,
        driver: str,
        *,
        node: str | None = None,
        runtime: Key | None,
        served: bool,
        elapsed_ms: int = 0,
        error: str | None = None,
        retry_disposition: str | None = None,
        usage: Any = None,
        first_ms: int | None = None,
    ) -> None:
        key: Key = (node, driver)
        self._inflight[key] = max(0, self._inflight.get(key, 0) - 1)
        if runtime is not None:
            self._runtime_inflight[runtime] = max(0, self._runtime_inflight.get(runtime, 0) - 1)
        if served:
            now = time.perf_counter()
            self._last_request[key] = now
            if runtime is not None:
                self._runtime_last_request[runtime] = now
        self._note_attempt(
            AttemptRow(
                driver=driver,
                elapsed_ms=elapsed_ms,
                served=served,
                # The row keeps the NAMES, which is what the contract and
                # every UI render -- but taken from the key the attempt
                # was actually counted against rather than from a second,
                # independent scan that could disagree with it.
                runtime=runtime[1] if runtime is not None else None,
                node=runtime[0] if runtime is not None else node,
                backend=self._backend_for_driver(key),
                error=error,
                retry_disposition=retry_disposition,
                usage_known=usage is not None
                and getattr(usage, "promptTokens", None) is not None
                and getattr(usage, "completionTokens", 0) is not None,
                prompt_tokens=getattr(usage, "promptTokens", None),
                completion_tokens=getattr(usage, "completionTokens", 0)
                if usage is not None
                else None,
                first_ms=first_ms,
            )
        )

    # --- per-attempt collection (M8) ----------------------------------------
    #
    # The hooks fire on whatever task is running the request, so the rows
    # are collected into a contextvar rather than onto the table: the
    # table is shared and long-lived, and a dict keyed by anything else
    # would need a request id the driver protocol has no room for.
    #
    # The route opens a collection scope, reads it back after the
    # response is built, and joins it to the facts only it holds (tokens,
    # wake cost). Without a scope open, noting an attempt is a no-op —
    # which is what keeps the existing tests, and anything calling
    # `generate()` outside a request, working unchanged.

    def _note_attempt(self, row: AttemptRow) -> None:
        collected = _attempts.get()
        if collected is not None:
            collected.append(row)

    def _backend_for_driver(self, key: Key) -> str | None:
        for backend in self._snapshot.reachable:
            if backend.key == key:
                kind = getattr(backend.info, "backend", None)
                return getattr(kind, "value", None) if kind is not None else None
        return None

    # --- stop reservations --------------------------------------------------

    @contextmanager
    def stopping(self, runtime: Key) -> Iterator[None]:
        """Hold a runtime out of routing while a stop for it is in flight.

        `idle_pass` and `_evict_for` are check-await-act: they establish
        that nothing is in flight, then **await** an HTTP call to the
        agent. A request arriving inside that await is routed to an
        engine that is being shut down — a 502 on the non-streamed path
        and, because M10's commit point forbids failover once a token is
        out, a visibly truncated answer on the streamed one.

        A context manager rather than a pair of calls because the one
        property that matters is that it is always released: a
        reservation left behind removes a healthy runtime from the
        install for the life of the process. Counted rather than
        flagged because an idle pass and an eviction can be stopping the
        same runtime, and the inner span finishing must not un-reserve
        it for the outer one.

        The span is the stop call and nothing longer. Afterwards the
        runtime's own `status` carries the ineligibility, which is the
        honest source for it.
        """
        self._stopping[runtime] = self._stopping.get(runtime, 0) + 1
        try:
            yield
        finally:
            remaining = self._stopping.get(runtime, 1) - 1
            if remaining > 0:
                self._stopping[runtime] = remaining
            else:
                self._stopping.pop(runtime, None)

    def is_stopping(self, runtime: Key) -> bool:
        return self._stopping.get(runtime, 0) > 0

    def inflight(self, driver: Key) -> int:
        return self._inflight.get(driver, 0)

    def runtime_inflight(self, runtime: Key) -> int:
        return self._runtime_inflight.get(runtime, 0)

    def idle_seconds(self, runtime: Key) -> float | None:
        """How long since this runtime last served a request through the
        gateway — or, for one that never has, since it became ready.
        None when it is not ready and never served."""
        last = self._runtime_last_request.get(runtime)
        since = self._ready_since.get(runtime)
        marks = [m for m in (last, since) if m is not None]
        if not marks:
            return None
        return time.perf_counter() - max(marks)

    def driver_idle_seconds(self, driver: Key) -> int | None:
        last = self._last_request.get(driver)
        return int(time.perf_counter() - last) if last is not None else None

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
        target's, **every configured tier kept — including empty ones**.

        Empty tiers used to be dropped here, and that quietly renumbered
        the ones after them. A slot configured `[local-8b, cloud]` whose
        `local-8b` had never been launched (no runtime, so no companion
        driver, so nothing advertising that model id) collapsed to one
        tier, and the cloud fallback answered reporting `tier: 1` — so an
        operator whose primary was missing was told the primary served
        the request.

        Routing was never wrong; the order is the same either way. Only
        the label was, and `TieredClient` had already been written the
        other way ("empty tiers are kept so `tier` counts the slot's
        tiers, not the eligible ones"). It kept a tier whose target
        existed but was ineligible, while this dropped one whose target
        had nothing at all — two rules for one question, and the
        contract's `tier` description states this one.

        **The slot's own name is the one conditional tier**, and the
        condition is *does this install declare a runtime under that
        name*. `_slot_targets` always puts it first, and it is implicit —
        "this model's own replicas" — rather than something the operator
        listed. Two shapes hide behind that:

        * `{model: "chat", targets: [local-8b, cloud]}`, where `chat` is
          a **virtual alias** nothing is ever launched under. An empty
          self tier there would push both of the operator's targets up a
          number, which is the same defect in the other direction and is
          how the first attempt at this fix broke five tests.
        * `{model: "qwen3-8b", targets: [cloud]}`, where the slot's name
          **is a real local model** — the shape the UI produces. There
          the self tier IS the primary, and dropping it because its
          companion driver happened to be down at this refresh made the
          cloud answer `tier: 1`: *the primary served this*, which is
          exactly the question tiered failover exists to answer,
          answered backwards. Review §6.2 #20, the second case the
          2026-09-10 carve-out left behind; fixed at R3 item 3.

        `_declares_runtime` tells them apart. Note which map it reads and
        which it does not: `runtimes` is what the install DECLARES, read
        per node from each agent and independent of what is advertising
        anything right now; `by_model` is the instant, and its being
        empty is the condition under test.

        A slot where nothing resolves is still a 404, because
        `has_backends()` asks whether any tier has backends rather than
        whether any tier exists. And `GET /v1/admin/routing` now shows a
        configured target with an empty backend list, which is the
        diagnosis an operator wants: *you asked for `local-8b` and
        nothing serves it.*
        """
        configured, targets = self._slot_targets(model)
        declared = self._declares_runtime(model)
        tiers: list[_Tier] = []
        for index, target in enumerate(targets):
            backends = list(self._snapshot.by_model.get(target, []))
            if index == 0 and not backends and not declared:
                continue
            tiers.append(_Tier(target=target, backends=backends))
        return Resolution(model=model, configured=configured, tiers=tiers)

    def _declares_runtime(self, model: str) -> bool:
        """Does any node in this install declare an engine runtime under
        this name? — which is what separates a primary that is currently
        down from a name that was never a primary at all.

        **By alias OR by name, and both halves earn their place.** The
        alias is what a companion driver advertises as its `modelId`
        (M6), so it is the match that fires on the live install. But
        `RuntimeSpec.modelAlias` is optional, and a runtime declared
        without one still gets a companion driver advertising *something*
        — matching only on `alias` would drop the self tier for every
        operator who never typed one, which is this same defect reached
        by a different route.

        Over-matching is the safe direction and is worth saying why: this
        decides a **label**, never an order. `_slot_targets` produces the
        same sequence either way and `pick` walks it the same way, so the
        cost of keeping a tier that should have gone is one number, while
        the cost of dropping one is a fallback claiming to be the
        primary.

        Iterates rather than looking up, because `runtimes` is keyed by
        `(node, name)` (R1.6) and the question is install-wide: one
        model on two machines is two entries, and either answers it.
        """
        return any(
            facts.alias == model or facts.name == model
            for facts in self._snapshot.runtimes.values()
        )

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
        return sorted(rotated, key=lambda b: self.inflight(b.key) / b.parallel_slots)

    def pick(self, resolution: Resolution, *, images: bool = False) -> TieredClient | None:
        """The client to send a request through, or None when nothing in
        the slot is eligible right now.

        Each tier's eligible backends are ordered by the balancer; the
        client walks the first tier, then the next, cascading on
        transport errors, 5xx and timeouts and failing hard on a 4xx.
        """
        tiers: list[list[DriverClient]] = []
        for tier in resolution.tiers:
            eligible = [
                b
                for b in tier.eligible()
                if not images
                or (b.info.capabilities is not None and b.info.capabilities.imageInput is True)
            ]
            # An empty tier stays in the list, so the response's `tier`
            # counts the slot's tiers rather than the eligible ones.
            tiers.append([b.client for b in self._order(tier.target, eligible)] if eligible else [])
        if not any(tiers):
            return None
        return TieredClient(name=resolution.model, tiers=tiers, hooks=self)

    def pick_embedding(self, resolution: Resolution) -> TieredClient | None:
        """A client that can only ever reach ONE model.

        **This is where the no-cross-model rule is enforced**, and it is
        enforced structurally rather than by a check at the end: the
        returned client is handed a single tier containing only the
        eligible backends whose driver reports serving `resolution.model`
        AND reports an embeddings surface. There is no second tier for a
        cascade to walk into, so the rule cannot be lost later by
        someone editing the cascade logic.

        Replicas of one model still balance and fail over exactly as
        they do for chat -- those are interchangeable by definition.
        """
        eligible = [
            b
            for b in resolution.eligible_backends()
            if b.embeds and b.info.modelId == resolution.model
        ]
        if not eligible:
            return None
        ordered = [b.client for b in self._order(resolution.model, eligible)]
        return TieredClient(name=resolution.model, tiers=[ordered], hooks=self)

    def surfaces_for(self, model: str) -> list[Surface]:
        """Which OpenAI surfaces this model can be sent to.

        A list because the two are not always disjoint: an Ollama runner
        is one or the other, but `llama-server` in `--embedding` mode
        still chats. Derived from every backend serving the name, so a
        surface appears when at least one of them offers it -- the
        opposite of `tool_calling`, which is the weakest-backend answer.
        That difference is deliberate: `tool_calling` promises a request
        will be carried whichever replica takes it, while `surfaces`
        answers "is there any point sending this here at all".
        """
        backends = self.resolve(model).backends()
        out: list[Surface] = []
        if any(b.chats for b in backends):
            out.append(Surface.chat)
        if any(b.embeds for b in backends):
            out.append(Surface.embeddings)
        return out

    def candidates_considered(self, resolution: Resolution) -> list[CandidateRow]:
        """What the balancer saw for each candidate, in tier order.

        Deliberately does **not** call `_order`. That method rotates a
        per-target cursor, so asking it a second time in order to find
        out what it did would change what it does next — the observation
        would move the thing observed. Recording the *inputs* to the
        decision avoids that, and is what the contract promises for this
        reason as much as for the absence of a score.
        """
        rows: list[CandidateRow] = []
        for index, tier in enumerate(resolution.tiers, start=1):
            for backend in tier.backends:
                rows.append(
                    CandidateRow(
                        driver=backend.name,
                        tier=index,
                        eligible=backend.eligible,
                        reason=backend.ineligible_reason,
                        in_flight=self.inflight(backend.key),
                        slots=backend.parallel_slots,
                    )
                )
        return rows

    def backends_for(self, model: str) -> list[_Backend]:
        return self.resolve(model).backends()

    def context_length_for(
        self, model: str, driver: str | None, node: str | None = None
    ) -> int | None:
        """The window of the one backend that answered, not the smallest.

        `_smallest_context` is the right number for `GET /v1/models`,
        where a caller is choosing a model and a request may land on any
        replica. It is the wrong number once a request has landed: what
        explains a truncated answer is the window that actually applied.

        Absent when the backend does not expose one. A supervised
        runtime's reading wins over the driver's own, since the agent
        reads it back from the engine after it loads.
        """
        if driver is None:
            return None
        for backend in self._matching(model, driver, node):
            if backend.runtime is not None and backend.runtime.context_length:
                return backend.runtime.context_length
            caps = backend.info.capabilities
            if caps is not None and caps.maxContextTokens:
                return caps.maxContextTokens
            return None
        return None

    def _matching(self, model: str, driver: str, node: str | None) -> list[_Backend]:
        """Backends serving `model` under this driver name, the one on
        `node` first.

        Two machines running one model give two backends with one name,
        so a first-match scan answered about whichever sorted first.
        `node` is what the client that answered reported; absent, the
        old behaviour stands and the answer is the first match.
        """
        named = [b for b in self.resolve(model).backends() if b.name == driver]
        if node is None:
            return named
        return sorted(named, key=lambda b: b.node != node)

    def runtime_for(
        self, model: str, driver: str | None = None, node: str | None = None
    ) -> str | None:
        """Name of the engine runtime behind `driver` — or, without one,
        behind the only runtime serving `model`. Absent for a hosted or
        CLI backend, which has no runtime of ours."""
        candidates = (
            self.resolve(model).backends()
            if driver is None
            else self._matching(model, driver, node)
        )
        for backend in candidates:
            if backend.runtime is not None:
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

    def as_model_list(
        self, allowed: set[str] | None = None, *, local_only: bool = False
    ) -> list[Model]:
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
            if allowed is not None and model_id not in allowed:
                continue
            resolution = self.resolve(model_id).restricted(allowed, local_only=local_only)
            if not resolution.has_backends():
                continue
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
                        surfaces=[Surface(value) for value in resolution.surfaces()],
                        context_length=_smallest_context(backends),
                        tool_calling=_all_carry_tools(backends),
                        image_input=any(
                            b.info.capabilities is not None
                            and b.info.capabilities.imageInput is True
                            for b in backends
                        ),
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
            control_root=self._control_root_view(),
        )

    def control_root(self) -> ControlRootFacts:
        """Where the last refresh got its node list, and whether it
        answered -- for the no-models 404, which used to name two
        healthy places and never this."""
        return self._snapshot.control_root

    def _control_root_view(self) -> ControlRootView:
        facts = self._snapshot.control_root
        return ControlRootView.model_validate(
            {
                "source": facts.source,
                "url": facts.url,
                "reachable": facts.reachable,
                "error": facts.error,
                "nodes": facts.nodes,
            }
        )

    def _backend_view(self, backend: _Backend) -> RoutingBackendView:
        facts = backend.runtime
        return RoutingBackendView(
            driver=backend.name,
            url=backend.url,  # type: ignore[arg-type]
            eligible=backend.eligible,
            ineligible_reason=backend.ineligible_reason,
            in_flight=self.inflight(backend.key),
            parallel_slots=backend.parallel_slots,
            runtime=facts.name if facts else None,
            runtime_status=facts.status if facts else None,
            node=facts.node if facts else None,
            stop_reason=facts.stop_reason if facts else None,
            idle_unload_seconds=facts.idle_unload_seconds if facts else None,
            start_on_demand=facts.start_on_demand if facts else None,
            idle_seconds=self.driver_idle_seconds(backend.key),
        )


def _backend_kind(backend: _Backend) -> BackendKind:
    """Bridge inference-driver.yaml's BackendKind to gateway.yaml's.

    Same wire values, two generated classes — the price of "components
    share schemas, not code", and cheaper than the coupling a shared
    module would create. Crossing via `.value` keeps the boundary
    explicit instead of hiding it behind a cast.
    """
    return BackendKind(backend.info.backend.value)


def _all_carry_tools(backends: list[_Backend]) -> bool:
    """Whether a request for this model may carry `tools`.

    True only when **every** backend serving it can, by the same
    reasoning `_smallest_context` follows: a request may land on any of
    them, so the honest answer is the weakest one. A model that is
    tool-capable on two replicas and not on a third is not tool-capable,
    because the third will refuse and which one answers is our choice,
    not the caller's.

    False for a model with no backends at all -- nothing serves it, so
    nothing about it can be promised.
    """
    if not backends:
        return False
    return all(
        b.info.capabilities is not None and bool(b.info.capabilities.toolCalling) for b in backends
    )


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
