"""Drains (row 13): the gateway's own, and its part in a runtime's.

**The gateway's own drain** (`GatewayDrain`). A restart or a stop used to end
the process 500 ms after it was asked, and the agent killed it after 5 s, so
every answer in flight was cut. Now `POST /v1/admin/restart` and
`POST /v1/admin/drain` start a drain:

1. new inference requests are refused `shutting_down` (row 11's refusal), and
   so is every request waiting in line (`FairQueue.close()`);
2. the answers in flight finish, for up to `shutdownDrainSeconds`;
3. a stream still open at that limit gets its door's error frame ("Cut:
   Eugene's gateway restarted after waiting 120 s for this answer. Ask
   again.") and its metrics row's outcome is `cut`;
4. the process exits 0, through the server's own shutdown, so the lifespan
   closes normally.

Config, health and admin routes answer throughout: only the front doors'
requests are counted and refused (`DrainMiddleware`).

**A runtime's drain** is the agent's (`agent/runtime_drain.py`): the runtime
reports `status: draining`, and a gateway routes only to `ready` runtimes, so
an older gateway obeys it too. What this gateway adds is the report the agent
waits on (`DrainReporter`): on every refresh while a drain is open, its own
count of answers running on the runtime, how long the runtime has been idle,
and whether the model's other copies can carry the load (DI-C3).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from starlette.types import ASGIApp, Receive, Scope, Send

from . import admission, door_paths, fair_queue
from ._generated.models import GatewayDrain as GatewayDrainView
from ._generated.models import GatewayDrainState
from .metrics import RequestRow

log = logging.getLogger(__name__)

#: `shutdownDrainSeconds` when the config does not say (R13-DRAIN, Troy 2026-10-21).
DEFAULT_SECONDS = 120.0
#: How long a cut request is given to send its error frame and end.
CUT_SETTLE_SECONDS = 5.0
#: What a drain reporter waits between reports while a drain is open.
REPORT_SECONDS = 2.0
#: The runtime drain states in which a gateway reports.
REPORTED = frozenset({"waiting_idle", "waiting_covered", "draining", "drained", "cut"})
#: Methods that never count as an answer in flight: a read of a model list or a
#: finished video is not what a drain waits for.
_READS = frozenset({"GET", "HEAD", "OPTIONS"})


def cut_words(seconds: float) -> str:
    return (
        f"Cut: Eugene's gateway restarted after waiting {seconds:g} s for this answer. Ask again."
    )


async def wait_inflight(count: Callable[[], int], seconds: float, *, poll: float = 0.1) -> bool:
    """Wait until `count()` is 0, at most `seconds`. Whether it reached 0.

    The one wait every drain shares: a model switch waits on its two
    runtimes' answers with it, as the gateway's drain waits on its own."""
    deadline = time.perf_counter() + max(0.0, seconds)
    while count() > 0:
        if time.perf_counter() >= deadline:
            return False
        await asyncio.sleep(poll)
    return True


@dataclass(eq=False)
class _Open:
    """One inference request in flight, and the event that cuts it."""

    cut: asyncio.Event = field(default_factory=asyncio.Event)
    streamed: bool = False


class GatewayDrain:
    """The gateway's own drain: one per app, started at most once."""

    def __init__(
        self,
        app: Any,
        *,
        seconds: Callable[[], float],
        exit_process: Callable[[], None] | None = None,
    ) -> None:
        self._app = app
        self._seconds = seconds
        self._exit = exit_process
        self.state = GatewayDrainState.serving
        self.started_at: datetime | None = None
        self.deadline_at: datetime | None = None
        self.limit = 0.0
        self.cut = 0
        self.message: str | None = None
        self._open: set[_Open] = set()
        self._idle = asyncio.Event()
        self._idle.set()
        self._task: asyncio.Task[None] | None = None

    # --- reading ----------------------------------------------------------

    @property
    def serving(self) -> bool:
        return self.state is GatewayDrainState.serving

    def running(self) -> int:
        return len(self._open)

    def view(self) -> GatewayDrainView:
        return GatewayDrainView(
            state=self.state,
            running=self.running(),
            startedAt=self.started_at,
            deadline=self.deadline_at,
            cut=self.cut,
            message=self.message,
        )

    # --- the requests in flight -------------------------------------------

    def enter(self) -> _Open:
        entry = _Open()
        self._open.add(entry)
        self._idle.clear()
        return entry

    def leave(self, entry: _Open) -> None:
        self._open.discard(entry)
        if not self._open:
            self._idle.set()

    # --- the drain --------------------------------------------------------

    def start(self) -> GatewayDrainView:
        """Begin draining, once; a second call answers the drain running."""
        if self._task is None:
            self.limit = max(0.0, float(self._seconds()))
            now = datetime.now(UTC)
            self.state = GatewayDrainState.draining
            self.started_at = now
            self.deadline_at = now + timedelta(seconds=self.limit)
            running = self.running()
            self.message = (
                f"Finishing {running} answer{'s' if running != 1 else ''}, "
                f"then restarting (at most {self.limit:g} s)."
            )
            log.warning(
                "draining: %d answer(s) in flight, at most %.0f s, then the process exits",
                running,
                self.limit,
            )
            # Waiters in line are refused `shutting_down` now (row 11).
            routing = getattr(self._app.state, "routing", None)
            if routing is not None:
                fair_queue.of(routing).close()
            self._task = asyncio.create_task(self._run(), name="gateway-drain")
        return self.view()

    async def _run(self) -> None:
        try:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self.limit):
                    await self._idle.wait()
            if self._open:
                self.cut = len(self._open)
                log.warning(
                    "draining: cutting %d answer(s) still running after %.0f s",
                    self.cut,
                    self.limit,
                )
                for entry in list(self._open):
                    entry.cut.set()
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(CUT_SETTLE_SECONDS):
                        await self._idle.wait()
            self.message = (
                f"{self.cut} answer{'s were' if self.cut != 1 else ' was'} cut at the "
                f"{self.limit:g} s limit."
                if self.cut
                else "Every answer finished."
            )
        finally:
            self.state = GatewayDrainState.exiting
            self._request_exit()

    def _request_exit(self) -> None:
        exit_process = self._exit or getattr(self._app.state, "request_exit", None)
        if callable(exit_process):
            log.warning("drained; exiting")
            exit_process()
            return
        # No server to ask (the app was served by something other than
        # `__main__`): the old way, once the response has had time to flush.
        log.warning("drained; no server hook, so exiting the process directly")
        asyncio.get_running_loop().call_later(0.5, lambda: os._exit(0))

    async def aclose(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await self._task


def of(app: Any) -> GatewayDrain | None:
    found = getattr(getattr(app, "state", None), "gateway_drain", None)
    return found if isinstance(found, GatewayDrain) else None


class DrainMiddleware:
    """Counts the front doors' requests in flight, refuses new ones while the
    gateway drains, and cuts those still running at the limit.

    Mounted inside `ClientAdmissionMiddleware`, so a refusal or a cut raised
    here is answered by it, in each door's own shape: a status and
    `Retry-After` before a response started, the door's error frame after
    (`response.failed` on the Responses door, `event: error` on Anthropic's).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        drain = of(scope.get("app")) if scope["type"] == "http" else None
        if (
            drain is None
            or str(scope.get("method", "")).upper() in _READS
            or not door_paths.matches(scope["path"], admission.CLIENT_ADMISSION_PATHS)
        ):
            await self.app(scope, receive, send)
            return
        if not drain.serving:
            raise admission.refused(fair_queue.shutting_down())
        entry = drain.enter()
        inner: asyncio.Future[None] = asyncio.ensure_future(self.app(scope, receive, send))
        cut: asyncio.Future[Any] = asyncio.ensure_future(entry.cut.wait())
        either: set[asyncio.Future[Any]] = {inner, cut}
        try:
            await asyncio.wait(either, return_when=asyncio.FIRST_COMPLETED)
            if inner.done():
                inner.result()
                return
            inner.cancel()
            await asyncio.gather(inner, return_exceptions=True)
            _record_cut(scope)
            raise admission.AdmissionFailure(
                503, cut_words(drain.limit), "5", kind="service_unavailable"
            )
        finally:
            if not inner.done():
                inner.cancel()
                await asyncio.gather(inner, return_exceptions=True)
            cut.cancel()
            drain.leave(entry)


def _record_cut(scope: Scope) -> None:
    """The metrics row of an answer cut at the limit: outcome `cut`. Written
    here because the door's own recorder never ran (its stream was cancelled),
    and marked recorded so the admission middleware does not write `error`."""
    context = admission.current.get()
    app = scope.get("app")
    metrics = getattr(getattr(app, "state", None), "metrics", None)
    if context is None or context.recorded or context.model is None or metrics is None:
        return
    try:
        elapsed = int((time.perf_counter() - context.started) * 1000)
        metrics.record(
            RequestRow(
                started_at=context.at,
                requested_model=context.model,
                total_ms=elapsed,
                attempts=len(context.tries),
                outcome="cut",
                queued_ms=context.queued_ms,
                band=context.band,
                streamed=context.streamed,
                tries=context.tries,
                client_key_id=context.key_id,
                client_key_name=context.key_name,
                person_id=context.person_id,
                app_client_id=context.app_client_id,
                request_id=context.id,
                elapsed_ms=elapsed,
            )
        )
        context.recorded = True
    except Exception:
        log.debug("could not record a cut request", exc_info=True)


# --- the gateway's part in a runtime's drain ------------------------------------


@dataclass(frozen=True)
class DrainReport:
    """What this gateway reports for one runtime (`RuntimeDrainReport`)."""

    running: int
    idle_seconds: float | None
    other_copies: int
    covered: bool

    def body(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "running": self.running,
            "otherCopies": self.other_copies,
            "covered": self.covered,
        }
        if self.idle_seconds is not None:
            out["idleSeconds"] = round(max(0.0, self.idle_seconds), 1)
        return out


#: Statuses of a copy that is running or on its way, for `otherCopies`.
_LIVE = frozenset({"ready", "draining", "copying", "starting", "loading", "exited"})


def report_for(table: Any, facts: Any) -> DrainReport:
    """This gateway's report on one runtime with an open drain.

    `covered` (DI-C3, Troy 2026-10-21): the model's other ready copies, not
    themselves draining, have a free slot for every answer running on any copy
    of the model and every request waiting in line for it. A copy whose agent
    reports no slot count counts as one slot."""
    model = facts.alias or facts.name
    copies = [f for f in table.runtimes() if (f.alias or f.name) == model and f.key != facts.key]
    live = [f for f in copies if f.status in _LIVE]
    ready = [f for f in live if _covers(f, facts)]
    demand = sum(table.runtime_inflight(f.key) for f in [facts, *copies])
    queue = fair_queue.of(table)
    demand += sum(
        sum(group["waiting"].values()) for group in queue.snapshot() if group.get("model") == model
    )
    capacity = sum(f.parallel_slots or 1 for f in ready)
    return DrainReport(
        running=table.runtime_inflight(facts.key),
        idle_seconds=table.idle_seconds(facts.key),
        other_copies=len(live),
        covered=bool(ready) and demand <= capacity,
    )


def _drain_state(facts: Any) -> str | None:
    drain = getattr(facts, "drain", None)
    return str(drain.get("state")) if isinstance(drain, dict) and drain.get("state") else None


def _open(facts: Any) -> bool:
    return _drain_state(facts) in REPORTED


def _order(facts: Any) -> tuple[str, str]:
    return (facts.node or "", facts.name)


def _covers(other: Any, facts: Any) -> bool:
    """Whether `other` can take `facts`'s load: it is ready, and not leaving.

    A copy waiting for idleness is still routed, so it covers. Of two copies
    that both wait for cover, only the later in `(node, name)` order covers
    the earlier: the earlier leaves first and the other waits for it to come
    back, rather than both leaving at once or each waiting on the other."""
    if other.status != "ready":
        return False
    state = _drain_state(other)
    if state is None or state in ("waiting_idle", "cancelled"):
        return True
    if state == "waiting_covered":
        return _order(other) > _order(facts)
    return False


class DrainReporter:
    """Reports on every runtime whose agent holds a drain open, every
    `REPORT_SECONDS` while one is (the routing table refreshes that often then).

    `on_draining` runs before a runtime is reported with nothing running: the
    hook where row 44 moves conversations off a machine going into
    maintenance."""

    def __init__(
        self,
        table: Any,
        *,
        report: Callable[[str, str, dict[str, Any], str | None], Awaitable[bool]],
        interval: float = REPORT_SECONDS,
    ) -> None:
        self._table = table
        self._report = report
        self._interval = interval
        self._task: asyncio.Task[None] | None = None
        self.on_draining: Callable[[Any], Awaitable[None]] | None = None
        #: The last report sent for each runtime, for tests and the log.
        self.sent: dict[Any, DrainReport] = {}

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="drain-reports")

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._interval)
                try:
                    await self.report_once()
                except Exception as e:
                    log.warning("drain report failed: %s", e)
        except asyncio.CancelledError:
            return

    async def report_once(self) -> list[Any]:
        """One round. Returns the runtimes reported on."""
        reported: list[Any] = []
        for facts in self._table.runtimes():
            if not _open(facts):
                continue
            report = report_for(self._table, facts)
            if report.running == 0 and facts.status == "draining" and self.on_draining is not None:
                await self.on_draining(facts)
            agent = self._table.agent_url_for(facts)
            if await self._report(agent, facts.name, report.body(), facts.node):
                self.sent[facts.key] = report
                reported.append(facts.key)
        return reported


__all__ = [
    "DEFAULT_SECONDS",
    "DrainMiddleware",
    "DrainReport",
    "DrainReporter",
    "GatewayDrain",
    "cut_words",
    "of",
    "report_for",
    "wait_inflight",
]
