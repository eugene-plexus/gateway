"""Lifecycle policy: the gateway decides, the owning agent executes.

Three components each see one third of the picture — the gateway sees
requests, the control root owns declarations, the agent owns the
process — and idle is a property of traffic, so the decision lives here.
The agent is asked to stop or start through its own API with the
gateway's service token, saying why, and the control root is **not in
the path**: a request for a sleeping model has to wake it whether or not
management is up, or M5's surviving-data-path guarantee turns into an
inference outage the moment the root is down.

Nothing here is replicated state. Which runtimes are loaded is an
observation one gateway made at one moment; a promoted control root
re-reads it from the agents. The policy fields (`idleUnloadSeconds`,
`startOnDemand`) ride on each `RuntimeSpec` and replicate with the
declaration. Design: `docs/design/m6-lifecycle-policy.md` §5.

Two behaviours:

* **Idle unload.** Every `idleCheckSeconds`, each `ready` runtime that
  declared `idleUnloadSeconds`, has nothing in flight through this
  gateway, and has been idle that long is stopped with `reason: idle`.
  A runtime with no timeout is never touched.
* **Start on demand.** A request whose slot has no eligible backend but
  a `stopped` runtime that declared `startOnDemand` wakes it: start,
  poll until `ready` or `swapWaitSeconds`, refresh the table, serve.
  If the agent refuses the start on admission, the most-idle runtimes
  that themselves declared an idle timeout and have nothing in flight
  are stopped to make room, once — eviction bounded by opt-in.

Concurrent requests for the same sleeping model share one wake.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx

from .routing import READY, Resolution, RoutingTable, _RuntimeFacts

log = logging.getLogger(__name__)

# The agent's stop/start are quick acknowledgements; anything slower is
# the agent being down, and a wake should fail rather than hang.
_AGENT_TIMEOUT_SECONDS = 10.0


class AgentLifecycleClient:
    """The four calls this module makes on an agent, with the gateway's
    own service token — `service:gateway` is the one non-operator
    audience the agent's stop and start accept."""

    def __init__(self, service_token: str | None) -> None:
        self._headers = {"Authorization": f"Bearer {service_token}"} if service_token else {}
        self._client = httpx.AsyncClient(timeout=_AGENT_TIMEOUT_SECONDS)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stop(self, agent_url: str, name: str, *, reason: str) -> bool:
        try:
            response = await self._client.post(
                f"{agent_url.rstrip('/')}/v1/runtimes/{quote(name, safe='')}/stop",
                json={"reason": reason},
                headers=self._headers,
            )
        except httpx.HTTPError as e:
            log.warning("could not ask %s to stop runtime %r: %s", agent_url, name, e)
            return False
        if response.status_code >= 400:
            log.warning(
                "agent %s refused to stop runtime %r: %d %s",
                agent_url,
                name,
                response.status_code,
                response.text[:200],
            )
            return False
        return True

    async def start(self, agent_url: str, name: str) -> tuple[int, str]:
        """`(status, detail)`. 202 means started or already running; 422
        means admission refused, with the arithmetic in `detail`."""
        try:
            response = await self._client.post(
                f"{agent_url.rstrip('/')}/v1/runtimes/{quote(name, safe='')}/start",
                headers=self._headers,
            )
        except httpx.HTTPError as e:
            return 0, f"the agent at {agent_url} did not answer: {e}"
        return response.status_code, _detail(response)

    async def admission(self, agent_url: str, spec: dict[str, Any]) -> dict[str, Any] | None:
        try:
            response = await self._client.post(
                f"{agent_url.rstrip('/')}/v1/runtimes/admission",
                json=spec,
                headers=self._headers,
            )
            if response.status_code >= 400:
                return None
            body = response.json()
            return body if isinstance(body, dict) else None
        except (httpx.HTTPError, ValueError) as e:
            log.info("admission dry run at %s failed: %s", agent_url, e)
            return None

    async def runtime(self, agent_url: str, name: str) -> dict[str, Any] | None:
        try:
            response = await self._client.get(
                f"{agent_url.rstrip('/')}/v1/runtimes/{quote(name, safe='')}",
                headers=self._headers,
            )
            if response.status_code >= 400:
                return None
            body = response.json()
            return body if isinstance(body, dict) else None
        except (httpx.HTTPError, ValueError):
            return None


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            return str(detail.get("detail") or detail.get("title") or detail)
        if detail:
            return str(detail)
    return response.text[:300]


@dataclass
class WakeResult:
    """What a wake did, for the response's routing extension and for the
    503 that says why it could not."""

    ok: bool
    runtime: str | None
    waited_ms: int
    message: str
    evicted: list[str] = field(default_factory=list)


class LifecycleManager:
    """Owns the idle loop and the wake path over one routing table."""

    def __init__(
        self,
        table: RoutingTable,
        *,
        client: AgentLifecycleClient,
        swap_wait_seconds: Callable[[], float],
        idle_check_seconds: Callable[[], float],
        poll_seconds: float = 1.0,
    ) -> None:
        self._table = table
        self._client = client
        self._swap_wait = swap_wait_seconds
        self._idle_check = idle_check_seconds
        self._poll = poll_seconds
        self._task: asyncio.Task[None] | None = None
        # One wake per runtime at a time; concurrent requesters await it.
        self._waking: dict[str, asyncio.Task[WakeResult]] = {}
        self.stopped_idle: list[str] = []
        """Runtimes this gateway unloaded for idleness, most recent last.
        For the admin view and the acceptance run; not authoritative."""

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._idle_loop(), name="lifecycle-idle")

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await self._task
            self._task = None
        for task in list(self._waking.values()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await task
        self._waking.clear()
        await self._client.aclose()

    # --- idle unload ------------------------------------------------------

    async def _idle_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(max(1.0, float(self._idle_check())))
                try:
                    await self.idle_pass()
                except Exception as e:
                    log.warning("idle pass failed: %s", e)
        except asyncio.CancelledError:
            return

    async def idle_pass(self) -> list[str]:
        """One sweep. Returns the runtimes stopped. Public so a test — or
        the acceptance run — can drive it without waiting on the clock."""
        stopped: list[str] = []
        for facts in self._table.runtimes():
            if facts.status != READY or not facts.idle_unload_seconds:
                continue
            if self._table.runtime_inflight(facts.name) > 0:
                continue
            idle = self._table.idle_seconds(facts.name)
            if idle is None or idle < facts.idle_unload_seconds:
                continue
            agent_url = self._table.agent_url_for(facts)
            log.info(
                "runtime %r idle for %ds (timeout %ds); asking %s to unload it",
                facts.name,
                int(idle),
                facts.idle_unload_seconds,
                agent_url,
            )
            if await self._client.stop(agent_url, facts.name, reason="idle"):
                stopped.append(facts.name)
                self.stopped_idle.append(facts.name)
        if stopped:
            # Refresh so the next request sees `stopped` rather than
            # trying a driver whose engine is gone.
            with contextlib.suppress(Exception):
                await self._table.refresh()
        return stopped

    # --- start on demand ----------------------------------------------------

    async def wake(self, resolution: Resolution) -> WakeResult:
        """Wake the first startable runtime in the slot, in tier order.

        A tier with a startable runtime is awaited rather than skipped:
        the operator put it first, and falling through to a cloud target
        because the local model was idle would spend money precisely
        because idle unload worked.
        """
        candidates = resolution.startable()
        if not candidates:
            waking = resolution.waking()
            if waking:
                names = ", ".join(f"{r.name} ({r.status})" for r in waking)
                return WakeResult(
                    ok=False,
                    runtime=waking[0].name,
                    waited_ms=0,
                    message=f"a runtime serving this model is still coming up: {names}",
                )
            return WakeResult(
                ok=False,
                runtime=None,
                waited_ms=0,
                message="nothing serving this model is ready, and none of its runtimes asked to "
                "be started on demand",
            )
        facts = candidates[0]
        task = self._waking.get(facts.name)
        if task is None:
            task = asyncio.create_task(self._wake_runtime(facts), name=f"wake:{facts.name}")
            self._waking[facts.name] = task

            def forget(_task: asyncio.Task[WakeResult], name: str = facts.name) -> None:
                self._waking.pop(name, None)

            task.add_done_callback(forget)
        return await asyncio.shield(task)

    async def _wake_runtime(self, facts: _RuntimeFacts) -> WakeResult:
        started = time.monotonic()
        agent_url = self._table.agent_url_for(facts)
        evicted: list[str] = []

        status, detail = await self._client.start(agent_url, facts.name)
        if status == 422:
            # Admission refused. Make room, bounded by opt-in, then once
            # more. The agent's dry run names what holds the device.
            evicted = await self._evict_for(facts, agent_url)
            if evicted:
                status, detail = await self._client.start(agent_url, facts.name)
        if status != 202:
            return WakeResult(
                ok=False,
                runtime=facts.name,
                waited_ms=int((time.monotonic() - started) * 1000),
                message=(
                    f"could not start runtime {facts.name!r} on demand: {detail}"
                    + (f" (after unloading {', '.join(evicted)})" if evicted else "")
                ),
                evicted=evicted,
            )

        deadline = started + max(1.0, float(self._swap_wait()))
        last_status: str | None = None
        while time.monotonic() < deadline:
            body = await self._client.runtime(agent_url, facts.name)
            last_status = str(body.get("status")) if body and body.get("status") else last_status
            if last_status == READY:
                await self._table.refresh()
                waited = int((time.monotonic() - started) * 1000)
                log.info("runtime %r woke on demand in %dms", facts.name, waited)
                return WakeResult(
                    ok=True,
                    runtime=facts.name,
                    waited_ms=waited,
                    message=f"runtime {facts.name!r} started on demand",
                    evicted=evicted,
                )
            if last_status == "crashed":
                return WakeResult(
                    ok=False,
                    runtime=facts.name,
                    waited_ms=int((time.monotonic() - started) * 1000),
                    message=f"runtime {facts.name!r} crashed while starting on demand; "
                    f"GET /v1/runtimes/{facts.name} on its agent has the captured error",
                    evicted=evicted,
                )
            await asyncio.sleep(self._poll)

        return WakeResult(
            ok=False,
            runtime=facts.name,
            waited_ms=int((time.monotonic() - started) * 1000),
            message=(
                f"runtime {facts.name!r} was started on demand and is still "
                f"{last_status or 'starting'} after {int(self._swap_wait())}s; retry shortly, "
                f"or raise swapWaitSeconds"
            ),
            evicted=evicted,
        )

    async def _evict_for(self, facts: _RuntimeFacts, agent_url: str) -> list[str]:
        """Stop the most-idle evictable runtimes holding the device, until
        the agent's dry run says the wake would fit. Only runtimes that
        declared `idleUnloadSeconds` and have nothing in flight here."""
        evicted: list[str] = []
        for _ in range(8):
            admission = await self._client.admission(agent_url, facts.spec)
            if admission is None or admission.get("decision") != "refuse":
                break
            blockers = [
                b
                for b in admission.get("blockers") or []
                if isinstance(b, dict)
                and b.get("evictable")
                and isinstance(b.get("name"), str)
                and b["name"] not in evicted
                and self._table.runtime_inflight(b["name"]) == 0
            ]
            if not blockers:
                break

            # Most idle first, by this gateway's own knowledge; the agent
            # cannot know idleness and did not try to order by it.
            def most_idle_first(blocker: dict[str, Any]) -> float:
                return -(self._table.idle_seconds(str(blocker["name"])) or 0.0)

            blockers.sort(key=most_idle_first)
            victim = blockers[0]["name"]
            log.info(
                "evicting idle runtime %r to make room for %r on %s", victim, facts.name, agent_url
            )
            if not await self._client.stop(agent_url, victim, reason="idle"):
                break
            evicted.append(victim)
            self.stopped_idle.append(victim)
        return evicted


__all__ = ["AgentLifecycleClient", "LifecycleManager", "WakeResult"]
