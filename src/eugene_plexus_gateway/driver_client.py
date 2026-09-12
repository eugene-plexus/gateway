"""HTTP clients for talking to inference-driver instances.

Thin wrappers around httpx.AsyncClient: one client per driver, cached and
reused by the routing table. Each carries the driver's topology `name` so
a response can say which backend answered it.

`TieredClient` composes several into a slot: an ordered list of tiers,
each an ordered list of backends. That is where failover lives — within
a tier first (the other replicas of the same model), then the next tier
(the next target in the slot's priority list) — and because the routing
table groups drivers by the model they serve, a slot's members come from
the topology rather than from configuration. `FailoverDriverClient` is
the one-tier case, kept under its old name.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from ._generated.driver_models import (
    DriverInfo,
    EmbedRequest,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
    Problem,
)

log = logging.getLogger(__name__)


class DriverError(Exception):
    """Raised when an inference-driver responds with 4xx/5xx.

    Carries the driver's parsed `Problem` body (when present) so the
    chat route can surface the *actual* upstream error instead of the
    generic "502 Bad Gateway" httpx text. Drivers return Problem JSON
    via FastAPI's `HTTPException(detail=Problem(...).model_dump())`,
    which produces a `{"detail": {...}}` envelope — we look inside.
    """

    def __init__(
        self,
        *,
        driver_name: str,
        driver_url: str,
        status_code: int,
        problem: Problem | None,
        raw_body: str,
    ) -> None:
        self.driver_name = driver_name
        self.driver_url = driver_url
        self.status_code = status_code
        self.problem = problem
        self.raw_body = raw_body
        super().__init__(self._summary())

    def _summary(self) -> str:
        prefix = f"driver {self.driver_name!r} ({self.driver_url}) returned {self.status_code}"
        if self.problem is not None:
            parts = [prefix, self.problem.title]
            if self.problem.detail:
                parts.append(self.problem.detail)
            if self.problem.component:
                parts.append(f"component={self.problem.component}")
            return " — ".join(parts)
        snippet = self.raw_body[:300] if self.raw_body else "<empty body>"
        return f"{prefix} (no problem+json body): {snippet}"


@dataclass(frozen=True)
class StreamEvent:
    """One event from a driver's token stream, as the gateway sees it.

    The gateway's own shape rather than the driver's `Chunk`: the two
    components share schemas, not code, so this is assembled from the
    parsed SSE rather than imported.
    """

    text: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    """Tool-call fragments on this event, when the driver is streaming a
    call rather than text. Kept as parsed JSON rather than a model: the
    gateway's only job with a fragment is to re-frame it as an OpenAI
    delta, and validating a *fragment* against the whole-call shape
    would reject the normal case -- `id` and `name` arrive once, and
    `arguments` arrives split at arbitrary points."""
    done: bool = False
    result: GenerateResponse | None = None


def _problem_from_bytes(raw: bytes) -> Problem | None:
    """`_problem_from_response`, for a body already read off a stream."""
    try:
        body: Any = json.loads(raw)
    except ValueError:
        return None
    candidates: list[Any] = []
    if isinstance(body, dict):
        if isinstance(body.get("detail"), dict):
            candidates.append(body["detail"])
        candidates.append(body)
    for candidate in candidates:
        try:
            return Problem.model_validate(candidate)
        except ValidationError:
            continue
    return None


def _problem_from_response(response: httpx.Response) -> Problem | None:
    """Best-effort extraction of a Problem from a driver's error response.

    Drivers return either:
        a) a bare Problem JSON: `{"type": ..., "title": ..., ...}`
        b) FastAPI's HTTPException-wrapped form: `{"detail": {<problem>}}`

    Try (b) first (the common case), fall back to (a). Any parse failure
    returns None — the caller falls back to the raw body.
    """
    try:
        body: Any = response.json()
    except ValueError:
        return None
    candidates: list[Any] = []
    if isinstance(body, dict):
        if isinstance(body.get("detail"), dict):
            candidates.append(body["detail"])
        candidates.append(body)
    for candidate in candidates:
        try:
            return Problem.model_validate(candidate)
        except ValidationError:
            continue
    return None


class DriverClient(Protocol):
    """Contract every driver client implements (real or fake-for-tests)."""

    name: str
    base_url: str

    async def info(self) -> DriverInfo: ...
    async def generate(self, request: GenerateRequest) -> GenerateResponse: ...
    async def embed(self, request: EmbedRequest) -> EmbedResponse: ...
    def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]: ...
    async def aclose(self) -> None: ...


class RoutingHooks(Protocol):
    """What a slot tells the routing table around every attempt, so the
    table can count in-flight requests per backend — the load signal the
    balancer runs on — and remember when each backend last served.

    Since M8 this is also the **only** place per-attempt facts are
    observable. `elapsed_ms` is this attempt alone: a request's total
    includes every failed attempt before it, so attributing that total
    to the backend that finally answered reports a fast backend as slow
    in exactly the cascade an operator is investigating. And the route
    above cannot supply it — it sees one exception, not which of four
    backends produced it.

    The other half of why it is here rather than on the response: the
    streaming path returns a `StreamingResponse` and never builds the
    routing envelope, but it does call `generate()`. Recording here
    covers streams; recording off the response would not.
    """

    def on_attempt_start(self, driver: str) -> None: ...
    def on_attempt_end(
        self,
        driver: str,
        *,
        served: bool,
        elapsed_ms: int = 0,
        error: str | None = None,
    ) -> None: ...


class HttpDriverClient:
    """Real HTTP-backed client. Talks to an inference-driver over its OpenAPI."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        timeout_seconds: float = 180.0,
        service_token: str | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        # Mirrors TieredClient's surface so the route can read these off
        # either without asking which kind it holds.
        self.attempts = 1
        self.served_by: str | None = name
        self.tier = 1
        # When the agent threaded a service token in, attach it to
        # every outbound call. The driver validates against the shared
        # HMAC signing key. Headers stay unset when running unauthenticated
        # (dev / standalone) so the existing test path still works.
        headers = {"Authorization": f"Bearer {service_token}"} if service_token else None
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            headers=headers,
        )

    async def info(self) -> DriverInfo:
        response = await self._client.get("/v1/info")
        response.raise_for_status()
        return DriverInfo.model_validate(response.json())

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        payload = request.model_dump(mode="json", exclude_none=True)
        response = await self._client.post("/v1/generate", json=payload)
        if response.status_code >= 400:
            raise DriverError(
                driver_name=self.name,
                driver_url=self.base_url,
                status_code=response.status_code,
                problem=_problem_from_response(response),
                raw_body=response.text,
            )
        return GenerateResponse.model_validate(response.json())

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        payload = request.model_dump(mode="json", exclude_none=True)
        response = await self._client.post("/v1/embed", json=payload)
        if response.status_code >= 400:
            raise DriverError(
                driver_name=self.name,
                driver_url=self.base_url,
                status_code=response.status_code,
                problem=_problem_from_response(response),
                raw_body=response.text,
            )
        return EmbedResponse.model_validate(response.json())

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        """Consume one driver's `/v1/generate/stream`.

        Parses the three contracted event types -- `token`, `done`,
        `error` -- and nothing else. This is deliberately not a general
        SSE client: both ends of this wire are our own contract, single
        JSON payload per event, and the gateway already hand-*frames*
        SSE on its way out. If we ever consume a third party's SSE, use
        a library rather than reaching for this.

        An `error` event becomes a `DriverError`, so it flows into the
        same cascade taxonomy as a failed POST -- which is what lets a
        driver that dies *before* its first token still fail over.

        A non-200 raises `DriverError` before anything is yielded, for
        the same reason: nothing has been forwarded, so it is still an
        ordinary failure.
        """
        payload = request.model_dump(mode="json", exclude_none=True)
        async with self._client.stream(
            "POST",
            "/v1/generate/stream",
            json=payload,
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise DriverError(
                    driver_name=self.name,
                    driver_url=self.base_url,
                    status_code=response.status_code,
                    problem=_problem_from_bytes(body),
                    raw_body=body.decode("utf-8", "replace"),
                )
            event_name: str | None = None
            async for line in response.aiter_lines():
                if not line.strip():
                    event_name = None
                    continue
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                try:
                    parsed = json.loads(data)
                except ValueError:
                    log.debug("driver %r sent an unparseable SSE payload", self.name)
                    continue
                if event_name == "error":
                    raise DriverError(
                        driver_name=self.name,
                        driver_url=self.base_url,
                        status_code=int(parsed.get("status") or 502),
                        problem=Problem.model_validate(parsed)
                        if isinstance(parsed, dict)
                        else None,
                        raw_body=data,
                    )
                if event_name == "done":
                    yield StreamEvent(done=True, result=GenerateResponse.model_validate(parsed))
                    return
                if not isinstance(parsed, dict):
                    continue
                # A token frame carries text or tool-call fragments,
                # never both. Both count as output, which is what makes
                # the commit point in `TieredClient.stream` cover tool
                # calls without knowing anything about them.
                calls = parsed.get("toolCalls")
                if isinstance(calls, list) and calls:
                    yield StreamEvent(tool_calls=[c for c in calls if isinstance(c, dict)])
                    continue
                text = parsed.get("text")
                if isinstance(text, str) and text:
                    yield StreamEvent(text=text)

    async def aclose(self) -> None:
        await self._client.aclose()


def _is_cascade_eligible(exc: Exception) -> bool:
    """Failure taxonomy for priority-list failover (v0.2.1).

    Cascade-eligible (try the next backend in the slot):
      * transport-level errors — connection refused, DNS, read/connect
        timeout (every `httpx.HTTPError` that isn't a clean response)
      * upstream 5xx — the backend is reachable but broken/overloaded

    NOT cascade-eligible (fail the slot HARD, re-raise immediately):
      * upstream 4xx — a request / auth / config bug. The next backend
        would hit the same bad request, and cascading past it would
        mask the real problem (e.g. an expired service token reading as
        "all backends down" instead of "fix your token").

    Locked taxonomy per the v0.2.1 plan: 5xx / transport / timeout
    cascade; 4xx hard-fails. Timeouts surface as `httpx.TimeoutException`
    (an `httpx.HTTPError`), so they fall into the transport branch.
    """
    if isinstance(exc, DriverError):
        return exc.status_code >= 500
    return isinstance(exc, httpx.HTTPError)


class TieredClient:
    """A slot: ordered tiers of ordered backends, walked on failure.

    Implements the same `DriverClient` protocol as `HttpDriverClient`,
    so the caller is oblivious to failover — it holds a client and calls
    `.generate()`. Internally this tries the first tier's candidates in
    order, then the next tier's, cascading on a cascade-eligible failure
    (transport / 5xx / timeout) and failing hard on a 4xx. See
    `_is_cascade_eligible`.

    Granularity is per-call: each `.generate()` independently walks the
    list from the top, so a slot that fell over to its backup recovers to
    the primary as soon as the primary is healthy again — no operator
    intervention, and no sticky state to reset. Within a tier the order
    is the balancer's, decided when the routing table built this client.
    """

    def __init__(
        self,
        *,
        name: str,
        tiers: list[list[DriverClient]],
        hooks: RoutingHooks | None = None,
    ) -> None:
        candidates = [c for tier in tiers for c in tier]
        if not candidates:
            raise ValueError(f"driver slot {name!r} needs at least one backend")
        self.name = name
        # Empty tiers are kept so `tier` counts the slot's tiers, not the
        # eligible ones: a cloud target answering because every local
        # replica was asleep is tier 2, whatever tier 1 held.
        self._tiers = [list(tier) for tier in tiers]
        self._hooks = hooks
        # `base_url` is the primary backend — used for labelling / logs.
        # The active backend on a given turn may differ after failover,
        # but the slot's identity is its primary.
        self.base_url = candidates[0].base_url

        # What the last call actually did, for the response's routing
        # extension. Per-instance mutable state, which is safe ONLY
        # because an instance is per-request: `RoutingTable.pick` builds
        # a fresh wrapper around the shared, long-lived HttpDriverClients
        # on every call. Do not cache one of these.
        self.attempts = 0
        self.served_by: str | None = None
        self.tier = 0

    @property
    def candidates(self) -> list[DriverClient]:
        return [c for tier in self._tiers for c in tier]

    async def info(self) -> DriverInfo:
        """Report the first reachable backend's `/v1/info`.

        Mirrors generate()'s failover so the admin drivers listing
        reflects what a real chat turn would actually reach.
        """
        last_exc: Exception | None = None
        for index, candidate in enumerate(self.candidates):
            try:
                return await candidate.info()
            except Exception as exc:
                if not _is_cascade_eligible(exc):
                    raise
                last_exc = exc
                self._log_cascade("info", index, candidate, exc)
        assert last_exc is not None  # candidates is non-empty (checked in __init__)
        raise last_exc

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        last_exc: Exception | None = None
        total = len(self.candidates)
        index = 0
        for tier_index, tier in enumerate(self._tiers):
            for candidate in tier:
                self.attempts = index + 1
                driver = getattr(candidate, "name", None)
                if self._hooks is not None and driver:
                    self._hooks.on_attempt_start(driver)
                started = time.monotonic()
                try:
                    result = await candidate.generate(request)
                except Exception as exc:
                    if self._hooks is not None and driver:
                        self._hooks.on_attempt_end(
                            driver,
                            served=False,
                            elapsed_ms=int((time.monotonic() - started) * 1000),
                            # The exception CLASS, never its message: a
                            # driver error can carry a provider's response
                            # body, and this string is retained and
                            # rendered in a UI.
                            error=type(exc).__name__,
                        )
                    if not _is_cascade_eligible(exc):
                        # 4xx / non-HTTP error — surface it without trying
                        # the next backend. A 4xx is the same bad request
                        # everywhere.
                        raise
                    last_exc = exc
                    self._log_cascade("generate", index, candidate, exc, total=total)
                    index += 1
                else:
                    if self._hooks is not None and driver:
                        self._hooks.on_attempt_end(
                            driver,
                            served=True,
                            elapsed_ms=int((time.monotonic() - started) * 1000),
                        )
                    self.served_by = driver
                    self.tier = tier_index + 1
                    return result
        # Every backend failed in a cascade-eligible way. Re-raise the
        # last failure so the chat route's existing DriverError
        # / httpx.HTTPError handlers surface it as they would for a
        # single-backend slot — no new error path to maintain.
        assert last_exc is not None  # candidates is non-empty (checked in __init__)
        raise last_exc

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        """`generate()`'s cascade, over replicas of ONE model.

        The failure taxonomy is identical -- transport and 5xx cascade,
        4xx fails hard. What differs is what it is allowed to cascade
        *to*, and that is enforced upstream rather than here:
        `RoutingTable.pick_embedding` builds a single tier containing
        only backends serving the requested model id, so there is
        structurally no other model to reach. Replicas of one model are
        interchangeable; two models are not, and a fallback between them
        would write vectors from a different space into the caller's
        store with a 200 and no marker.
        """
        last_exc: Exception | None = None
        total = len(self.candidates)
        index = 0
        for tier_index, tier in enumerate(self._tiers):
            for candidate in tier:
                self.attempts = index + 1
                driver = getattr(candidate, "name", None)
                if self._hooks is not None and driver:
                    self._hooks.on_attempt_start(driver)
                started = time.monotonic()
                try:
                    result = await candidate.embed(request)
                except Exception as exc:
                    if self._hooks is not None and driver:
                        self._hooks.on_attempt_end(
                            driver,
                            served=False,
                            elapsed_ms=int((time.monotonic() - started) * 1000),
                            error=type(exc).__name__,
                        )
                    if not _is_cascade_eligible(exc):
                        raise
                    last_exc = exc
                    self._log_cascade("embed", index, candidate, exc, total=total)
                    index += 1
                else:
                    if self._hooks is not None and driver:
                        self._hooks.on_attempt_end(
                            driver,
                            served=True,
                            elapsed_ms=int((time.monotonic() - started) * 1000),
                        )
                    self.served_by = driver
                    self.tier = tier_index + 1
                    return result
        assert last_exc is not None
        raise last_exc

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        """`generate()`'s cascade, with a commit point.

        **The rule this method exists for:** failover is possible until
        the first token is emitted, and impossible after it.

        `generate()` can walk the whole list freely because nothing has
        reached the client until it returns. Once a token has been
        forwarded, retrying on another backend would splice two models'
        output into one answer with no marker at the seam -- a wrong
        answer that looks like a right one, which is worse than a
        truncated one. So a failure after the first token propagates and
        the route turns it into an OpenAI `error` frame.

        The practical consequence, contracted in `gateway.yaml`: a
        streamed request can be truncated where a non-streamed one would
        have cascaded. A caller that needs the failover guarantee should
        not stream.

        Attempt accounting follows `generate()` exactly, with one
        deliberate difference: an attempt that emitted tokens and then
        broke is recorded `served=False`. It did not serve the request,
        and calling it served would hide truncation from the one surface
        that could show it.
        """
        last_exc: Exception | None = None
        total = len(self.candidates)
        index = 0
        for tier_index, tier in enumerate(self._tiers):
            for candidate in tier:
                self.attempts = index + 1
                driver = getattr(candidate, "name", None)
                if self._hooks is not None and driver:
                    self._hooks.on_attempt_start(driver)
                started = time.monotonic()
                committed = False
                stream = candidate.stream(request)
                try:
                    async for event in stream:
                        committed = True
                        yield event
                except Exception as exc:
                    if self._hooks is not None and driver:
                        self._hooks.on_attempt_end(
                            driver,
                            served=False,
                            elapsed_ms=int((time.monotonic() - started) * 1000),
                            error=type(exc).__name__,
                        )
                    if committed:
                        # Past the commit point. The client already holds
                        # part of this answer; another backend's tokens
                        # cannot be appended to it.
                        log.warning(
                            "driver slot %r lost %s mid-stream after emitting tokens; "
                            "truncating rather than failing over: %s",
                            self.name,
                            driver,
                            exc,
                        )
                        raise
                    if not _is_cascade_eligible(exc):
                        raise
                    last_exc = exc
                    self._log_cascade("stream", index, candidate, exc, total=total)
                    index += 1
                else:
                    if self._hooks is not None and driver:
                        self._hooks.on_attempt_end(
                            driver,
                            served=True,
                            elapsed_ms=int((time.monotonic() - started) * 1000),
                        )
                    self.served_by = driver
                    self.tier = tier_index + 1
                    return
                finally:
                    # Runs on every path, including the consumer
                    # abandoning this generator: that is what releases
                    # the driver's response and, through it, the engine.
                    await stream.aclose()
        assert last_exc is not None  # candidates is non-empty (checked in __init__)
        raise last_exc

    def _log_cascade(
        self,
        op: str,
        index: int,
        candidate: DriverClient,
        exc: Exception,
        *,
        total: int | None = None,
    ) -> None:
        """Emit a WARNING when a backend fails and we cascade.

        Failover that silently always-works hides a broken primary
        (v0.3-plan risk). This WARNING is the "failover happened"
        surface operators grep for; the UI failover badge reads the
        same signal in a later release.
        """
        total = total if total is not None else len(self.candidates)
        is_last = index == total - 1
        next_action = (
            "no more backends in slot — failing"
            if is_last
            else f"trying backend {index + 2}/{total}"
        )
        log.warning(
            "failover[%s]: slot %r backend %d/%d (%s) failed (%s); %s",
            op,
            self.name,
            index + 1,
            total,
            candidate.base_url,
            exc,
            next_action,
        )

    async def aclose(self) -> None:
        for candidate in self.candidates:
            await candidate.aclose()


class FailoverDriverClient(TieredClient):
    """A one-tier slot: an ordered priority list of backends.

    The pre-M6 shape, kept under its name because the cascade rules it
    documented are unchanged — M6 added tiers above it, not a different
    walk within one.
    """

    def __init__(
        self,
        *,
        name: str,
        candidates: list[DriverClient],
        hooks: RoutingHooks | None = None,
    ) -> None:
        if not candidates:
            raise ValueError(f"driver slot {name!r} needs at least one backend URL")
        super().__init__(name=name, tiers=[list(candidates)], hooks=hooks)
