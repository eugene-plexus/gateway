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

import asyncio
import json
import logging
import sys
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
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
    RetryDisposition,
)
from ._http import internal_client
from .circuit import Circuit

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
        summary = self._summary()
        if retry_disposition(self) == "indeterminate":
            summary += " Outcome unknown: no automatic replay; work may have occurred."
        super().__init__(summary)

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


#: How the routing table identifies a driver and a runtime:
#: `(node, name)`. Defined in this module because `routing.py` imports
#: it and not the reverse; `RoutingHooks` is the protocol that carries
#: the value, so the type belongs beside it. See `routing.Key`, which is
#: an alias of this, for why a bare name is not an identity.
type Key = tuple[str | None, str]


class DriverClient(Protocol):
    """Contract every driver client implements (real or fake-for-tests)."""

    name: str
    base_url: str
    #: Which machine's agent reported this driver, when the table knows.
    #: On the protocol because `TieredClient` has to pass it to the hooks
    #: and a driver NAME is not unique across machines (R1.6, review
    #: §6.1 #8). Optional, so a fake in a single-host test needs nothing.
    node: str | None

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

    def on_attempt_start(self, driver: str, *, node: str | None = None) -> Key | None: ...

    """Open an attempt, and answer with the runtime it was counted
    against -- which the caller hands straight back to
    `on_attempt_end`.

    The return value exists because the two ends of one attempt used to
    resolve the runtime independently, each by scanning whatever
    snapshot was installed at the time, and a refresh landing mid-request
    made them disagree. A node that failed its read answers `None` on
    the way out, so the increment is never undone and that runtime never
    idle-unloads again; the mirror case decrements a counter it never
    raised, `max(0, ...)` swallows it, and a live request reads as zero
    in flight -- after which the idle pass unloads an engine mid-answer.
    An attempt is one thing and is counted once, against one runtime.

    **And `runtime` is a `(node, name)` pair, not a name** (R1.6).
    `RuntimeSpec.name` is unique per agent, not per install, so one model
    on two machines is one name twice; counted by name, two replicas
    shared every counter. The value stays an opaque handle from this
    side: it is whatever `on_attempt_start` answered, handed straight
    back.
    """

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
        node: str | None = None,
    ) -> None:
        self.name = name
        self.circuit = Circuit()
        self.base_url = base_url.rstrip("/")
        #: Which machine's agent reported this driver. Carried so
        #: `TieredClient` can hand it to the routing hooks, whose
        #: counters are keyed by `(node, name)` -- a driver name is
        #: unique per agent, not per install (R1.6).
        self.node = node
        # Mirrors TieredClient's surface so the route can read these off
        # either without asking which kind it holds.
        self.attempts = 1
        self.served_by: str | None = name
        self.served_by_node: str | None = node
        self.tier = 1
        # When the agent threaded a service token in, attach it to
        # every outbound call. The driver validates against the shared
        # HMAC signing key. Headers stay unset when running unauthenticated
        # (dev / standalone) so the existing test path still works.
        headers = {"Authorization": f"Bearer {service_token}"} if service_token else None
        # `internal_client`: one SSL context for the process instead of a
        # fresh certifi parse per client (~104 ms of synchronous CPU on
        # the event loop), and no proxy, because a driver is this machine
        # or another node of this install and a user's `HTTP_PROXY` --
        # inherited by the Windows logon task -- would swallow every hop.
        self._client = internal_client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            headers=headers,
        )

    async def info(self) -> DriverInfo:
        response = await self._client.get("/v1/info")
        response.raise_for_status()
        return DriverInfo.model_validate(response.json())

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        response = await self._client.post(
            "/v1/generate",
            json=payload,
            headers={"X-Request-ID": str(request.requestId)} if request.requestId else None,
        )
        if response.status_code >= 400:
            raise DriverError(
                driver_name=self.name,
                driver_url=self.base_url,
                status_code=response.status_code,
                problem=_problem_from_response(response),
                raw_body=response.text,
            )
        try:
            return GenerateResponse.model_validate(response.json())
        except ValueError as exc:
            raise self._invalid_reply() from exc

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        response = await self._client.post(
            "/v1/embed",
            json=payload,
            headers={"X-Request-ID": str(request.requestId)} if request.requestId else None,
        )
        if response.status_code >= 400:
            raise DriverError(
                driver_name=self.name,
                driver_url=self.base_url,
                status_code=response.status_code,
                problem=_problem_from_response(response),
                raw_body=response.text,
            )
        try:
            return EmbedResponse.model_validate(response.json())
        except ValueError as exc:
            raise self._invalid_reply() from exc

    def _invalid_reply(self) -> DriverError:
        return DriverError(
            driver_name=self.name,
            driver_url=self.base_url,
            status_code=502,
            problem=Problem(
                type="about:blank",
                title="Driver returned an invalid response",
                status=502,
                retryDisposition=RetryDisposition.indeterminate,
            ),
            raw_body="",
        )

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
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        async with self._client.stream(
            "POST",
            "/v1/generate/stream",
            json=payload,
            headers={
                "Accept": "text/event-stream",
                **({"X-Request-ID": str(request.requestId)} if request.requestId else {}),
            },
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
                except ValueError as exc:
                    raise self._invalid_reply() from exc
                if not isinstance(parsed, dict):
                    raise self._invalid_reply()
                if event_name == "error":
                    try:
                        problem = Problem.model_validate(parsed)
                    except ValueError as exc:
                        raise self._invalid_reply() from exc
                    raise DriverError(
                        driver_name=self.name,
                        driver_url=self.base_url,
                        status_code=problem.status,
                        problem=problem,
                        raw_body=data,
                    )
                if event_name == "done":
                    try:
                        result = GenerateResponse.model_validate(parsed)
                    except ValueError as exc:
                        raise self._invalid_reply() from exc
                    yield StreamEvent(done=True, result=result)
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


def retry_disposition(exc: BaseException) -> str:
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "safe"
    if isinstance(exc, DriverError):
        if exc.status_code == 504:
            return "indeterminate"
        if exc.problem is not None and exc.problem.retryDisposition is not None:
            return exc.problem.retryDisposition.value
        if 400 <= exc.status_code < 500:
            return "terminal"
    return "indeterminate"


def _is_cascade_eligible(exc: Exception) -> bool:
    return retry_disposition(exc) == "safe"


def _cooling_error(delay: float = 1) -> DriverError:
    return DriverError(
        driver_name="routing",
        driver_url="",
        status_code=503,
        problem=Problem(
            type="about:blank",
            title="Backends cooling down",
            status=503,
            retryDisposition=RetryDisposition.safe,
            retryAfterSeconds=max(1, delay),
            detail="Eligible backends are cooling down or already probing recovery. Try a new "
            "request later.",
        ),
        raw_body="",
    )


class TieredClient:
    """Ordered tiers, retried only after a proven pre-execution failure.

    Selection is per request; each HTTP client's circuit is shared across
    requests and limits recovery probes. No replay follows uncertain work,
    a deadline, or any output. See retry_disposition and Circuit.
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
        self.authorize_attempt: Callable[[], Awaitable[None]] | None = None
        self.prepare_request: (
            Callable[[DriverClient, GenerateRequest], Awaitable[GenerateRequest]] | None
        ) = None
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
        #: `TieredClient` is passed where a `DriverClient` is expected,
        #: so it carries the protocol's `node` too. It is a slot over
        #: several machines' backends and has no one node of its own;
        #: `served_by_node` is the meaningful answer and is set once a
        #: backend has answered.
        self.node: str | None = None
        self.served_by: str | None = None
        #: Which machine's driver answered. Beside `served_by` because
        #: the name alone no longer identifies one: two machines running
        #: one model give two drivers with one name (R1.6).
        self.served_by_node: str | None = None
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
                if self.authorize_attempt is not None:
                    await self.authorize_attempt()
                circuit = getattr(candidate, "circuit", None)
                if circuit is not None and not circuit.acquire():
                    if last_exc is None:
                        last_exc = _cooling_error(circuit.until - time.perf_counter())
                    continue
                probe_epoch = circuit.epoch if circuit is not None and circuit.probing else None
                self.attempts = index + 1
                driver = getattr(candidate, "name", None)
                node = getattr(candidate, "node", None)
                runtime: Key | None = None
                if self._hooks is not None and driver:
                    runtime = self._hooks.on_attempt_start(driver, node=node)
                started = time.perf_counter()
                try:
                    prepared = (
                        await self.prepare_request(candidate, request)
                        if self.prepare_request
                        else request
                    )
                    result = await candidate.generate(prepared)
                except BaseException as exc:
                    self._finish_circuit(candidate, exc, probe_epoch=probe_epoch)
                    if self._hooks is not None and driver:
                        self._hooks.on_attempt_end(
                            driver,
                            node=node,
                            runtime=runtime,
                            served=False,
                            elapsed_ms=int((time.perf_counter() - started) * 1000),
                            # The exception CLASS, never its message: a
                            # driver error can carry a provider's response
                            # body, and this string is retained and
                            # rendered in a UI.
                            error=type(exc).__name__,
                            retry_disposition=retry_disposition(exc),
                        )
                    if not isinstance(exc, Exception) or not _is_cascade_eligible(exc):
                        # 4xx / non-HTTP error — surface it without trying
                        # the next backend. A 4xx is the same bad request
                        # everywhere.
                        raise
                    last_exc = exc
                    self._log_cascade("generate", index, candidate, exc, total=total)
                    index += 1
                else:
                    self._finish_circuit(candidate, probe_epoch=probe_epoch)
                    if self._hooks is not None and driver:
                        self._hooks.on_attempt_end(
                            driver,
                            node=node,
                            runtime=runtime,
                            served=True,
                            usage=result.usage,
                            elapsed_ms=int((time.perf_counter() - started) * 1000),
                        )
                    self.served_by = driver
                    self.served_by_node = node
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

        The failure taxonomy is identical: only proven pre-execution
        failures cascade. What differs is what it is allowed to cascade
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
                if self.authorize_attempt is not None:
                    await self.authorize_attempt()
                circuit = getattr(candidate, "circuit", None)
                if circuit is not None and not circuit.acquire():
                    if last_exc is None:
                        last_exc = _cooling_error(circuit.until - time.perf_counter())
                    continue
                probe_epoch = circuit.epoch if circuit is not None and circuit.probing else None
                self.attempts = index + 1
                driver = getattr(candidate, "name", None)
                node = getattr(candidate, "node", None)
                runtime: Key | None = None
                if self._hooks is not None and driver:
                    runtime = self._hooks.on_attempt_start(driver, node=node)
                started = time.perf_counter()
                try:
                    result = await candidate.embed(request)
                except BaseException as exc:
                    self._finish_circuit(candidate, exc, probe_epoch=probe_epoch)
                    if self._hooks is not None and driver:
                        self._hooks.on_attempt_end(
                            driver,
                            node=node,
                            runtime=runtime,
                            served=False,
                            elapsed_ms=int((time.perf_counter() - started) * 1000),
                            error=type(exc).__name__,
                            retry_disposition=retry_disposition(exc),
                        )
                    if not isinstance(exc, Exception) or not _is_cascade_eligible(exc):
                        raise
                    last_exc = exc
                    self._log_cascade("embed", index, candidate, exc, total=total)
                    index += 1
                else:
                    self._finish_circuit(candidate, probe_epoch=probe_epoch)
                    if self._hooks is not None and driver:
                        self._hooks.on_attempt_end(
                            driver,
                            node=node,
                            runtime=runtime,
                            served=True,
                            usage=result.usage,
                            elapsed_ms=int((time.perf_counter() - started) * 1000),
                        )
                    self.served_by = driver
                    self.served_by_node = node
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
        have cascaded after a safe refusal. Neither surface replays an
        ambiguous attempt.

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
                if self.authorize_attempt is not None:
                    await self.authorize_attempt()
                circuit = getattr(candidate, "circuit", None)
                if circuit is not None and not circuit.acquire():
                    if last_exc is None:
                        last_exc = _cooling_error(circuit.until - time.perf_counter())
                    continue
                probe_epoch = circuit.epoch if circuit is not None and circuit.probing else None
                self.attempts = index + 1
                driver = getattr(candidate, "name", None)
                node = getattr(candidate, "node", None)
                runtime: Key | None = None
                if self._hooks is not None and driver:
                    runtime = self._hooks.on_attempt_start(driver, node=node)
                started = time.perf_counter()
                committed = False
                # Whether the driver ever said the answer was finished.
                # An SSE stream that simply stops is indistinguishable
                # from one that ended, at the transport layer -- the
                # `done` event is the only thing that tells them apart,
                # and M10's own rule is that the absence of it is a
                # truncation rather than a completion.
                saw_done = False
                usage = None
                reported = False
                stream = None
                try:
                    prepared = (
                        await self.prepare_request(candidate, request)
                        if self.prepare_request
                        else request
                    )
                    stream = candidate.stream(prepared)
                    async for event in stream:
                        committed = True
                        if event.done:
                            saw_done = True
                            usage = event.result.usage if event.result is not None else None
                        yield event
                except Exception as exc:
                    self._finish_circuit(candidate, exc, probe_epoch=probe_epoch)
                    reported = True
                    if self._hooks is not None and driver:
                        self._hooks.on_attempt_end(
                            driver,
                            node=node,
                            runtime=runtime,
                            served=False,
                            elapsed_ms=int((time.perf_counter() - started) * 1000),
                            error=type(exc).__name__,
                            retry_disposition="indeterminate"
                            if committed
                            else retry_disposition(exc),
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
                            type(exc).__name__,
                        )
                        raise
                    if not _is_cascade_eligible(exc):
                        raise
                    last_exc = exc
                    self._log_cascade("stream", index, candidate, exc, total=total)
                    index += 1
                else:
                    self._finish_circuit(
                        candidate,
                        None if saw_done else RuntimeError("incomplete"),
                        probe_epoch=probe_epoch,
                    )
                    reported = True
                    if self._hooks is not None and driver:
                        self._hooks.on_attempt_end(
                            driver,
                            node=node,
                            runtime=runtime,
                            # A stream that stopped without a `done`
                            # event did not serve this request, however
                            # tidily it ended. It is NOT cascaded past
                            # -- tokens are already out, so M10's commit
                            # point forbids it -- but calling it served
                            # would put a truncation in the metrics as a
                            # completion and refresh this backend's
                            # "last served" mark, which is what the
                            # balancer and the idle pass read.
                            served=saw_done,
                            usage=usage,
                            retry_disposition=None if saw_done else "indeterminate",
                            elapsed_ms=int((time.perf_counter() - started) * 1000),
                            error=None if saw_done else "IncompleteStream",
                        )
                    self.served_by = driver
                    self.served_by_node = node
                    self.tier = tier_index + 1
                    return
                finally:
                    # Runs on every path, including the consumer
                    # abandoning this generator: that is what releases
                    # the driver's response and, through it, the engine.
                    #
                    # And it is the only arm that sees the two ways this
                    # attempt ends WITHOUT reaching either branch above.
                    # A consumer that closes this generator -- the
                    # playground's Stop button, a tab shut mid-answer --
                    # gets `GeneratorExit` raised at the `yield`, and a
                    # cancelled request task gets `CancelledError`
                    # (starlette cancels the response task group on
                    # `http.disconnect`). Both are `BaseException`, so
                    # neither `except Exception` nor `else` runs, the
                    # attempt is never closed, and the runtime's
                    # in-flight counter stays above zero for the life of
                    # the process -- after which it never idle-unloads
                    # and can never be evicted to make room for a wake.
                    if not reported:
                        self._finish_circuit(
                            candidate,
                            sys.exc_info()[1] or RuntimeError("abandoned"),
                            probe_epoch=probe_epoch,
                        )
                    if not reported and self._hooks is not None and driver:
                        ending = sys.exc_info()[1]
                        self._hooks.on_attempt_end(
                            driver,
                            node=node,
                            runtime=runtime,
                            served=False,
                            elapsed_ms=int((time.perf_counter() - started) * 1000),
                            error=type(ending).__name__ if ending is not None else "Abandoned",
                            retry_disposition="indeterminate",
                        )
                    if stream is not None:
                        await stream.aclose()
        assert last_exc is not None  # candidates is non-empty (checked in __init__)
        raise last_exc

    @staticmethod
    def _finish_circuit(
        candidate: DriverClient,
        error: BaseException | None = None,
        *,
        probe_epoch: int | None = None,
    ) -> None:
        circuit = getattr(candidate, "circuit", None)
        if circuit is not None:
            if isinstance(error, asyncio.CancelledError | GeneratorExit):
                # A caller's Stop/disconnect/deadline is not a failed backend.
                # Nor is it a successful recovery probe. Keep uncertain usage
                # in attempt metrics, but do not punish the next caller.
                circuit.abandon(probe_epoch=probe_epoch)
                return
            delay = (
                error.problem.retryAfterSeconds
                if isinstance(error, DriverError) and error.problem
                else None
            )
            circuit.finish(
                failed=error is not None and retry_disposition(error) != "terminal",
                retry_after=delay,
                probe_epoch=probe_epoch,
            )

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
            type(exc).__name__,
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
