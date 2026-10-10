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
import contextlib
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from ._generated.driver_models import (
    DecisionRequest,
    DecisionResponse,
    DriverInfo,
    EmbedRequest,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
    ImagePartial,
    ImageRequest,
    ImageResponse,
    ModerateRequest,
    ModerateResponse,
    Problem,
    RetryDisposition,
    SpeakRequest,
    TokenCount,
    TranscribeRequest,
    TranscribeResponse,
    VideoJob,
    VideoRequest,
)
from ._http import internal_client
from .affinity import EVICTED, HIT
from .budget import Conversation, TurnBudget, request_chars
from .circuit import Circuit
from .execution import Executor, StreamPolicy
from .repetition import STOP_MESSAGE, Guard, Policy

log = logging.getLogger(__name__)

# A count is two small calls to the backend beside its slots, never a
# prefill, so it does not get a generation's minutes.
_COUNT_TIMEOUT_SECONDS = 30.0


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


class RepetitionStopped(DriverError):
    """An intentional, terminal gateway stop, never a backend health failure."""

    def __init__(self, candidate: DriverClient) -> None:
        super().__init__(
            driver_name=candidate.name,
            driver_url=candidate.base_url,
            status_code=422,
            problem=Problem(
                type="urn:eugene-plexus:problem#repetition-detected",
                title="Response repetition detected",
                status=422,
                detail=STOP_MESSAGE,
                retryDisposition=RetryDisposition.terminal,
            ),
            raw_body="",
        )

    def _summary(self) -> str:
        return STOP_MESSAGE


@dataclass(frozen=True)
class StreamEvent:
    """One event from a driver's token stream, as the gateway sees it.

    The gateway's own shape rather than the driver's `Chunk`: the two
    components share schemas, not code, so this is assembled from the
    parsed SSE rather than imported.
    """

    text: str = ""
    reasoning: str = ""
    """A fragment of the model's reasoning, from a backend that reports it
    apart from the answer. Output like any other event, so it is also the
    commit point: a caller shown the model thinking cannot then be handed
    another model's answer."""
    tool_calls: list[dict[str, Any]] | None = None
    """Tool-call fragments on this event, when the driver is streaming a
    call rather than text. Kept as parsed JSON rather than a model: the
    gateway's only job with a fragment is to re-frame it as an OpenAI
    delta, and validating a *fragment* against the whole-call shape
    would reject the normal case -- `id` and `name` arrive once, and
    `arguments` arrives split at arbitrary points."""
    logprobs: dict[str, Any] | None = None
    """The log probabilities of this event's tokens (P2c), riding with its
    `text`, which may be empty. Parsed JSON in OpenAI's shape."""
    annotations: list[dict[str, Any]] | None = None
    """Citations from a provider's web search (P2c), OpenAI's shape."""
    audio: dict[str, Any] | None = None
    """A fragment of a spoken answer (P2b), as the driver's `AudioDelta`
    (camelCase, parsed JSON). Output like text, so a commit point."""
    progress: dict[str, Any] | None = None
    """What the backend is doing while it produces no output, as the
    driver's `StreamProgress` (camelCase, parsed JSON). **Not output**:
    `TieredClient.stream` neither commits on it nor times a first token by
    it, so a backend that fails after reporting progress still fails over.
    Only a request with `reportProgress` gets any."""
    done: bool = False
    result: GenerateResponse | None = None
    search: Any = None
    """A search the gateway itself is running for this answer (P8), as a
    `server_tools.SearchUpdate`. Never from a driver: the search loop
    yields it between a model's turns, and each door renders it in its own
    vocabulary (a `web_search_call` item, a `server_tool_use` block) or
    not at all (chat)."""
    image: Any = None
    """An image the gateway itself is making for this answer (P8e), as a
    `server_tools.SearchUpdate` over an `ImageExecution`. Only on the
    Responses door, the one door whose API defines `image_generation`."""


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

    async def info(self, *, models: bool = True, model: str | None = None) -> DriverInfo: ...
    async def generate(self, request: GenerateRequest) -> GenerateResponse: ...
    async def embed(self, request: EmbedRequest) -> EmbedResponse: ...

    async def decide(self, request: DecisionRequest) -> DecisionResponse: ...
    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse: ...
    async def moderate(self, request: ModerateRequest) -> ModerateResponse: ...
    async def count_tokens(self, request: GenerateRequest) -> int: ...
    def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]: ...
    def speak(self, request: SpeakRequest) -> AsyncGenerator[str | bytes, None]: ...
    async def image(self, request: ImageRequest) -> ImageResponse: ...
    def image_stream(
        self, request: ImageRequest
    ) -> AsyncGenerator[ImagePartial | ImageResponse, None]: ...
    async def video(self, request: VideoRequest) -> VideoJob: ...
    async def aclose(self) -> None: ...


class VideoJobs(Protocol):
    """What a poll needs of the one driver that holds a job (P5). Not on
    `DriverClient`: a poll goes to that driver and never over tiers."""

    async def video_job(self, job_id: str) -> VideoJob: ...
    def video_content(self, job_id: str) -> AsyncGenerator[bytes, None]: ...


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
        first_ms: int | None = None,
        model: str | None = None,
    ) -> None: ...


class HttpDriverClient:
    """Real HTTP-backed client. Talks to an inference-driver over its OpenAPI."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        timeout_seconds: float = 180.0,
        auth: httpx.Auth | None = None,
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
        # `auth` presents the token for this driver's machine on every
        # call: the gateway's own token when the driver is beside it, a
        # fifteen-minute one for another machine (`outbound.py`). None
        # when running unauthenticated (dev / standalone), and on the
        # admin probe, which dials whatever URL an operator typed.
        # `internal_client`: one SSL context for the process instead of a
        # fresh certifi parse per client (~104 ms of synchronous CPU on
        # the event loop), and no proxy, because a driver is this machine
        # or another node of this install and a user's `HTTP_PROXY` --
        # inherited by the Windows logon task -- would swallow every hop.
        self._client = internal_client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            auth=auth,
        )

    async def info(self, *, models: bool = True, model: str | None = None) -> DriverInfo:
        """The driver's `/v1/info`. `models=False` leaves an account's list
        out; `model=` narrows it to one entry -- the per-request re-check,
        which must not re-read six hundred entries to confirm one."""
        params: dict[str, str] = {}
        if not models:
            params["models"] = "false"
        if model is not None:
            params["model"] = model
        response = await self._client.get("/v1/info", params=params or None)
        response.raise_for_status()
        body = response.json()
        return DriverInfo.model_validate(body)

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

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        response = await self._client.post(
            "/v1/decide",
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
            return DecisionResponse.model_validate(response.json())
        except ValueError as exc:
            raise self._invalid_reply() from exc

    async def moderate(self, request: ModerateRequest) -> ModerateResponse:
        """The driver's `/v1/moderate` (P6)."""
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        response = await self._client.post(
            "/v1/moderate",
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
            return ModerateResponse.model_validate(response.json())
        except ValueError as exc:
            raise self._invalid_reply() from exc

    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        """The driver's `/v1/transcribe` (P3b): the audio as base64 in JSON."""
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        response = await self._client.post(
            "/v1/transcribe",
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
            return TranscribeResponse.model_validate(response.json())
        except ValueError as exc:
            raise self._invalid_reply() from exc

    async def image(self, request: ImageRequest) -> ImageResponse:
        """The driver's `/v1/image` (P4): reference images as base64 in JSON."""
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        response = await self._client.post(
            "/v1/image",
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
            return ImageResponse.model_validate(response.json())
        except ValueError as exc:
            raise self._invalid_reply() from exc

    async def image_stream(
        self, request: ImageRequest
    ) -> AsyncGenerator[ImagePartial | ImageResponse, None]:
        """The driver's `/v1/image/stream` (P4): `partial` events, then
        `done`. A non-200 and an `error` event are `DriverError`s, so a
        failure before the first event cascades as a failed POST does."""
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        async with self._client.stream(
            "POST",
            "/v1/image/stream",
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
                    if event_name == "error":
                        problem = Problem.model_validate(parsed)
                        raise DriverError(
                            driver_name=self.name,
                            driver_url=self.base_url,
                            status_code=problem.status,
                            problem=problem,
                            raw_body=data,
                        )
                    if event_name == "partial":
                        yield ImagePartial.model_validate(parsed)
                    elif event_name == "done":
                        yield ImageResponse.model_validate(parsed)
                        return
                except ValueError as exc:
                    raise self._invalid_reply() from exc
            raise self._invalid_reply()

    async def video(self, request: VideoRequest) -> VideoJob:
        """The driver's `POST /v1/video` (P5): the job, accepted."""
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        response = await self._client.post(
            "/v1/video",
            json=payload,
            headers={"X-Request-ID": str(request.requestId)} if request.requestId else None,
        )
        return self._video_job_from(response)

    async def video_job(self, job_id: str) -> VideoJob:
        """The driver's `GET /v1/video/{jobId}`: the job now."""
        response = await self._client.get(f"/v1/video/{quote(job_id, safe='')}")
        return self._video_job_from(response)

    def _video_job_from(self, response: httpx.Response) -> VideoJob:
        if response.status_code >= 400:
            raise DriverError(
                driver_name=self.name,
                driver_url=self.base_url,
                status_code=response.status_code,
                problem=_problem_from_response(response),
                raw_body=response.text,
            )
        try:
            return VideoJob.model_validate(response.json())
        except ValueError as exc:
            raise self._invalid_reply() from exc

    async def video_content(self, job_id: str) -> AsyncGenerator[bytes, None]:
        """The driver's `/v1/video/{jobId}/content`, streamed. A non-200
        raises before anything is yielded."""
        path = f"/v1/video/{quote(job_id, safe='')}/content"
        async with self._client.stream("GET", path) as response:
            if response.status_code >= 400:
                body = await response.aread()
                raise DriverError(
                    driver_name=self.name,
                    driver_url=self.base_url,
                    status_code=response.status_code,
                    problem=_problem_from_bytes(body),
                    raw_body=body.decode("utf-8", "replace"),
                )
            async for chunk in response.aiter_raw():
                if chunk:
                    yield chunk

    async def count_tokens(self, request: GenerateRequest) -> int:
        """The driver's `/v1/generate/count`: the backend's own tokenizer, no generation."""
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        response = await self._client.post(
            "/v1/generate/count", json=payload, timeout=_COUNT_TIMEOUT_SECONDS
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
            return TokenCount.model_validate(response.json()).promptTokens
        except ValueError as exc:
            raise self._invalid_reply() from exc

    async def speak(self, request: SpeakRequest) -> AsyncGenerator[str | bytes, None]:
        """The driver's `/v1/speak` (P3a): the media type first, then the
        audio, as it arrives. A non-200 raises before anything is yielded."""
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=True)
        async with self._client.stream(
            "POST",
            "/v1/speak",
            json=payload,
            headers={"X-Request-ID": str(request.requestId)} if request.requestId else None,
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
            yield response.headers.get("content-type", "application/octet-stream")
            async for chunk in response.aiter_raw():
                if chunk:
                    yield chunk

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
                if event_name == "progress":
                    yield StreamEvent(progress=parsed)
                    continue
                if not isinstance(parsed, dict):
                    continue
                # A token frame carries text, reasoning or tool-call
                # fragments, one kind only. All three count as output,
                # which is what makes the commit point in
                # `TieredClient.stream` cover them without knowing
                # anything about them.
                calls = parsed.get("toolCalls")
                if isinstance(calls, list) and calls:
                    yield StreamEvent(tool_calls=[c for c in calls if isinstance(c, dict)])
                    continue
                reasoning = parsed.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    yield StreamEvent(reasoning=reasoning)
                    continue
                audio = parsed.get("audio")
                if isinstance(audio, dict) and audio:
                    yield StreamEvent(audio=audio)
                    continue
                annotations = parsed.get("annotations")
                if isinstance(annotations, list) and annotations:
                    yield StreamEvent(annotations=[a for a in annotations if isinstance(a, dict)])
                    continue
                text = parsed.get("text")
                logprobs = parsed.get("logprobs")
                if (isinstance(text, str) and text) or isinstance(logprobs, dict):
                    yield StreamEvent(
                        text=text if isinstance(text, str) else "",
                        logprobs=logprobs if isinstance(logprobs, dict) else None,
                    )

    async def aclose(self) -> None:
        await self._client.aclose()


class BoundClient:
    """One model on one driver: what a tier holds since P1 (2026-09-27).

    A driver can serve many models now -- a provider account serves every
    model its backend lists -- so a routing candidate is `(node, driver,
    model)`, not `(node, driver)`. This wraps the driver's one long-lived
    HTTP client and does three things on every call:

    * **names the model**, unprefixed, on the request, so the driver asks
      its backend for the model this candidate is and not for whatever it
      holds;
    * **publishes the answer under the public id** -- `openrouter/x`, where
      the driver said `x` -- so a caller sees the name it asked for;
    * **owns its own circuit**, so a failing model cools down alone rather
      than taking the other models of its account with it.

    Cached by the routing table across refreshes, like the client it
    wraps, so its circuit survives them. It never closes the inner client:
    the table owns that.
    """

    def __init__(self, inner: DriverClient, *, model: str, public_model: str) -> None:
        self._inner = inner
        self.name = inner.name
        self.base_url = inner.base_url
        self.node: str | None = getattr(inner, "node", None)
        #: The driver's own id for this model -- what a request names.
        self.model = model
        #: The id callers use -- `<driver>/<model>` for an account.
        self.public_model = public_model
        self.circuit = Circuit()
        # Mirrors TieredClient's surface, as HttpDriverClient does.
        self.attempts = 1
        self.served_by: str | None = inner.name
        self.served_by_node: str | None = self.node
        self.served_model: str | None = public_model
        self.tier = 1

    def _bind[
        R: (
            GenerateRequest,
            EmbedRequest,
            DecisionRequest,
            SpeakRequest,
            TranscribeRequest,
            ImageRequest,
            VideoRequest,
            ModerateRequest,
        )
    ](self, request: R) -> R:
        return request.model_copy(update={"model": self.model})

    def _publish(self, reported: str | None) -> str | None:
        """The driver's name for the model, as callers know it."""
        if reported is None or reported == self.model:
            return self.public_model
        return reported

    async def info(self, *, models: bool = True, model: str | None = None) -> DriverInfo:
        info = self._inner.info
        try:
            return await info(models=models, model=model)
        except TypeError:
            # A test double from before the parameters existed.
            return await info()

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        result = await self._inner.generate(self._bind(request))
        result.modelId = self._publish(result.modelId)
        return result

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        result = await self._inner.embed(self._bind(request))
        result.modelId = self._publish(result.modelId)
        return result

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        result = await self._inner.decide(self._bind(request))
        result.modelId = self._publish(result.modelId)
        return result

    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        result = await self._inner.transcribe(self._bind(request))
        result.modelId = self._publish(result.modelId)
        return result

    async def moderate(self, request: ModerateRequest) -> ModerateResponse:
        result = await self._inner.moderate(self._bind(request))
        result.modelId = self._publish(result.modelId)
        return result

    async def count_tokens(self, request: GenerateRequest) -> int:
        return await self._inner.count_tokens(self._bind(request))

    async def image(self, request: ImageRequest) -> ImageResponse:
        result = await self._inner.image(self._bind(request))
        result.modelId = self._publish(result.modelId)
        return result

    async def video(self, request: VideoRequest) -> VideoJob:
        result = await self._inner.video(self._bind(request))
        result.modelId = self._publish(result.modelId)
        return result

    async def image_stream(
        self, request: ImageRequest
    ) -> AsyncGenerator[ImagePartial | ImageResponse, None]:
        events = self._inner.image_stream(self._bind(request))
        try:
            async for item in events:
                if isinstance(item, ImageResponse):
                    item.modelId = self._publish(item.modelId)
                yield item
        finally:
            closer = getattr(events, "aclose", None)
            if closer is not None:
                await closer()

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        events = self._inner.stream(self._bind(request))
        try:
            async for event in events:
                if event.done and event.result is not None:
                    event.result.modelId = self._publish(event.result.modelId)
                yield event
        finally:
            closer = getattr(events, "aclose", None)
            if closer is not None:
                await closer()

    async def speak(self, request: SpeakRequest) -> AsyncGenerator[str | bytes, None]:
        events = self._inner.speak(self._bind(request))
        try:
            async for item in events:
                yield item
        finally:
            closer = getattr(events, "aclose", None)
            if closer is not None:
                await closer()

    async def aclose(self) -> None:
        """Nothing: the inner client is the routing table's to close."""


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


def capacity_refused(exc: BaseException) -> bool:
    """The driver's `#backend-capacity`: the engine's shared KV pool was
    full of other requests' prompts (CB3). Load, not a broken backend, so
    it may cascade (`safe`) and the circuit does not count it: counted, a
    few overflows on an 8B at 64k took every healthy replica out as
    "cooling down" and 158 of 204 agent turns failed (gateway#8)."""
    return (
        isinstance(exc, DriverError)
        and exc.problem is not None
        and str(exc.problem.type or "").endswith("#backend-capacity")
    )


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
        self.repetition_policy = Policy()
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
        #: PC4: `hit`, `new` or `moved` when the balancer had a conversation
        #: to place, else None. Set by `RoutingTable.pick`.
        self.affinity: str | None = None
        #: CB3: this turn's place in its replicas' shared pools, when any
        #: has one. Set by `RoutingTable.pick`; see `budget.py`.
        self.budget: TurnBudget | None = None
        #: The conversation this turn continues, when it has a key: its
        #: last prompt is kept for the budget's next estimate, and a hit is
        #: told from an eviction by it (CB5). Set by `RoutingTable.pick`.
        self.conversation: Conversation | None = None
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
        #: The published id of the model that answered -- a slot's target,
        #: prefixed for an account. Set with `served_by`.
        self.served_model: str | None = None
        #: The candidate that answered the last call, for `pin`.
        self.served_candidate: DriverClient | None = None
        self.tier = 0

    @property
    def candidates(self) -> list[DriverClient]:
        return [c for tier in self._tiers for c in tier]

    def pin(self) -> None:
        """Send every later call to the backend that answered, and only to it.

        P8's rule: **failover ends at the first search**. A model call
        after a search is a turn of one answer whose conversation now holds
        that search's results; another backend continuing it would be the
        seam M10 forbids for tokens. The empty tiers in front keep `tier`
        counting the slot's tiers, so a pinned tier-2 backend still reports
        2 on every turn.
        """
        served = self.served_candidate
        if served is None:
            return
        self._tiers = [[] for _ in range(max(0, self.tier - 1))] + [[served]]

    async def info(self, *, models: bool = True, model: str | None = None) -> DriverInfo:
        """Report the first reachable backend's `/v1/info`.

        Mirrors generate()'s failover so the admin drivers listing
        reflects what a real chat turn would actually reach.
        """
        last_exc: Exception | None = None
        for index, candidate in enumerate(self.candidates):
            try:
                return await candidate.info(models=models, model=model)
            except Exception as exc:
                if not _is_cascade_eligible(exc):
                    raise
                last_exc = exc
                self._log_cascade("info", index, candidate, exc)
        assert last_exc is not None  # candidates is non-empty (checked in __init__)
        raise last_exc

    async def _admit(self, request: GenerateRequest) -> None:
        """CB3: wait for room in a shared pool, and try first where there is."""
        if self.budget is None:
            return
        for index, tier in enumerate(self._tiers):
            if tier:
                self._tiers[index] = await self.budget.admit(request, tier)
                return

    def _enter(self, candidate: DriverClient) -> None:
        if self.budget is not None:
            self.budget.enter(candidate)

    def _served(self, usage: Any, candidate: DriverClient, request: GenerateRequest) -> None:
        """What the answer said about the cache: remember this conversation's
        prompt, and call a hit whose home no longer held it `evicted` (CB5).

        Evicted is the measured definition: the turn went back to the replica
        that served its last one, and the engine reused less than that turn's
        whole prompt. A backend that does not report cached tokens cannot be
        judged, and its hit stays a hit.
        """
        convo = self.conversation
        if convo is None or usage is None:
            return
        cached = getattr(usage, "cachedPromptTokens", None)
        if (
            self.affinity == HIT
            and candidate is convo.home
            and convo.previous_prompt is not None
            and isinstance(cached, int)
            and cached < convo.previous_prompt
        ):
            self.affinity = EVICTED
        prompt = getattr(usage, "promptTokens", None)
        if isinstance(prompt, int) and prompt > 0:
            chars = self.budget.chars if self.budget is not None else request_chars(request)
            convo.sizes.put(convo.target, convo.key, prompt, chars)

    def _release(self) -> None:
        if self.budget is not None:
            self.budget.close()

    def _keyed(self, request: GenerateRequest) -> GenerateRequest:
        """The conversation key, for a driver that pins slots (CB4). Internal:
        the driver never sends it upstream."""
        if self.conversation is None or request.conversationKey is not None:
            return request
        return request.model_copy(update={"conversationKey": self.conversation.key})

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        request = self._keyed(request)
        await self._admit(request)
        try:
            return await self._generate(request)
        finally:
            self._release()

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        async def call(candidate: DriverClient) -> GenerateResponse:
            prepared = (
                await self.prepare_request(candidate, request) if self.prepare_request else request
            )
            result = await candidate.generate(prepared)
            # Whole replies have already consumed their compute. Observe them
            # without cutting or reinterpreting an otherwise complete answer.
            guard = Guard(
                self.repetition_policy,
                structured=True,
                request_id=str(prepared.requestId) if prepared.requestId else None,
            )
            guard.take(
                StreamEvent(
                    text=result.content or "",
                    reasoning=result.reasoning or "",
                    tool_calls=[
                        {"index": i, "function": {"arguments": call.function.arguments}}
                        for i, call in enumerate(result.toolCalls or [])
                    ],
                )
            )
            guard.take(StreamEvent(done=True))
            return result

        return await Executor(self).whole("generate", call, generation=request)

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        # pick_embedding supplies only replicas of this model.
        return await Executor(self).whole("embed", lambda candidate: candidate.embed(request))

    async def _over_tiers[T](self, label: str, call: Callable[[Any], Awaitable[T]]) -> T:
        return await Executor(self).whole(label, call)

    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        """Transcription over the slot's tiers, as chat (P3b, call #4).

        A transcript from a fallback model is still a transcript, so unlike
        speech and embeddings this walks every tier `pick_transcription`
        built, each holding only backends that transcribe.
        """
        return await self._over_tiers("transcribe", lambda c: c.transcribe(request))

    async def moderate(self, request: ModerateRequest) -> ModerateResponse:
        """A moderation over replicas of ONE model (P6-2): `pick_moderation`
        builds a single tier of the slot's first model, as `pick_embedding`
        does, since categories and thresholds are that model's own."""
        return await self._over_tiers("moderate", lambda c: c.moderate(request))

    async def image(self, request: ImageRequest) -> ImageResponse:
        """Images over the slot's tiers, as chat (P4, call #4): a fallback
        model's image is still an image. `pick_image` built every tier from
        models whose listing takes this request."""
        return await self._over_tiers("image", lambda c: c.image(request))

    async def video(self, request: VideoRequest) -> VideoJob:
        """A video submit over the slot's tiers (P5): **failover at submit
        only** (section 5, #4). Once one backend accepts the job it is that
        backend's, and the handle names it."""
        return await self._over_tiers("video", lambda c: c.video(request))

    async def image_stream(
        self, request: ImageRequest
    ) -> AsyncGenerator[ImagePartial | ImageResponse, None]:
        stream = Executor(self).stream(
            "image stream", lambda candidate: candidate.image_stream(request)
        )
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        # pick_decision supplies only replicas of this decision model.
        return await Executor(self).whole("decide", lambda candidate: candidate.decide(request))

    async def speak(self, request: SpeakRequest) -> AsyncGenerator[str | bytes, None]:
        async def call(candidate: DriverClient) -> AsyncGenerator[str | bytes, None]:
            events = candidate.speak(request)
            try:
                media = await anext(events)
                first = await anext(events, b"")
                # Hold the media header until the first bytes are available.
                yield media
                if first:
                    yield first
                async for chunk in events:
                    yield chunk
            finally:
                await events.aclose()

        stream = Executor(self).stream("speak", call)
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        """`_stream`, inside this turn's room in a shared pool (CB3).

        The inner generator is closed explicitly, so a consumer abandoning
        this one still runs its attempt accounting at once."""
        request = self._keyed(request)
        await self._admit(request)
        try:
            async with contextlib.aclosing(self._stream(request)) as events:
                async for event in events:
                    yield event
        finally:
            self._release()

    async def _stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        async def call(candidate: DriverClient) -> AsyncGenerator[StreamEvent, None]:
            prepared = (
                await self.prepare_request(candidate, request) if self.prepare_request else request
            )
            events = candidate.stream(prepared)
            guard = Guard(
                self.repetition_policy,
                structured=(
                    prepared.responseFormat is not None and prepared.responseFormat.type != "text"
                )
                or prepared.audioOutput is not None,
                request_id=str(prepared.requestId) if prepared.requestId else None,
            )
            try:
                async for event in events:
                    stopped = guard.take(event)
                    if stopped:
                        # Close the owned transport before yielding to a slow
                        # downstream client. Its partial output still stands.
                        await events.aclose()
                    yield event
                    if stopped:
                        raise RepetitionStopped(candidate)
            finally:
                await events.aclose()

        policy = StreamPolicy(
            progress=lambda event: event.progress is not None,
            complete=lambda event: bool(event.done),
            usage=lambda event: event.result.usage if event.result is not None else None,
            eof_is_success=False,
        )
        stream = Executor(self).stream("stream", call, policy=policy, generation=request)
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()

    @staticmethod
    def _finish_circuit(
        candidate: DriverClient,
        error: BaseException | None = None,
        *,
        probe_epoch: int | None = None,
    ) -> None:
        circuit = getattr(candidate, "circuit", None)
        if circuit is not None:
            if isinstance(error, asyncio.CancelledError | GeneratorExit | RepetitionStopped):
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
                failed=error is not None
                and retry_disposition(error) != "terminal"
                and not capacity_refused(error),
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

    async def count_tokens(self, request: GenerateRequest) -> int:
        """Counted by a backend of the tier that would serve, or not at all.

        **Never past the first tier that holds anything**, and that is the
        rule rather than a shortcut: a later tier is another model, whose
        tokenizer answers a different question -- a count of the fallback's
        prompt presented as the primary's. The embeddings door's
        no-cross-model rule, applied to counts.

        Within that tier each replica is asked in the balancer's order
        until one counts. That replays nothing: a count generates nothing,
        so asking the next replica is not the uncertain-work replay the
        generate cascade refuses. A 400 is the request's fault and would
        be the same on every replica, so it is raised at once.
        """
        tier = next((t for t in self._tiers if t), [])
        last: Exception | None = None
        for candidate in tier:
            try:
                count = await candidate.count_tokens(request)
            except DriverError as exc:
                if exc.status_code == 400:
                    raise
                last = exc
                continue
            except httpx.HTTPError as exc:
                last = exc
                continue
            self.served_by, self.served_by_node = candidate.name, candidate.node
            self.served_model = getattr(candidate, "public_model", None)
            return count
        if last is None:  # pragma: no cover - pick() never builds an empty slot
            raise ValueError(f"driver slot {self.name!r} has no backend to count with")
        raise last

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
