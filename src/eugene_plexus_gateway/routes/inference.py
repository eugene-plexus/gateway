"""The front doors: /v1/models, /v1/chat/completions, /v1/embeddings and,
since R4, /v1/messages.

These are the only operations in Eugene Plexus that use snake_case field
names and somebody else's error envelope. That is deliberate:
"OpenAI-compatible" is worth nothing unless an unmodified OpenAI SDK can
point its `base_url` here and work, and those SDKs parse the error shape
to build their exceptions. Renaming `max_tokens` to `maxTokens` for house
consistency would break the entire audience.

`/v1/messages` speaks Anthropic's wire for the same reason one layer
over: Claude Code and everything built on the Anthropic SDKs cannot read
OpenAI's shape at all, so without that door the most capable client in
the field can be pointed at every competing product and not at us.
**It is a translation, not a second routing engine** -- `_prepare` is
shared, so both doors get the same refusals, the same cascade, the same
wake and the same recording, and `GET /v1/metrics` sees both. The
translation itself lives in `..anthropic`.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import struct
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace
from dataclasses import field as dc_field
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .. import admission, anthropic, chat_contract
from .._generated.driver_models import (
    EmbedRequest,
    GenerateRequest,
    GenerateResponse,
    Message,
    Role,
)
from .._generated.driver_models import NamedToolChoice as DriverNamedToolChoice
from .._generated.driver_models import ResponseFormat as DriverResponseFormat
from .._generated.driver_models import Tool as DriverTool
from .._generated.driver_models import ToolChoice as DriverToolChoice
from .._generated.models import (
    BackendKind,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRoutingInfo,
    CompletionUsage,
    Delta,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    FinishReason,
    FunctionCall,
    ModelList,
    Object,
    Object1,
    ResponseFormat,
    Role1,
    Role2,
    Tool,
    ToolCall,
    ToolCallDelta,
)
from ..config import ConfigStore
from ..dependencies import require_authorized
from ..disconnect import ClientGone, serve_while_connected
from ..driver_client import DriverClient, DriverError, TieredClient
from ..images import has_images
from ..lifecycle import LifecycleManager, WakeResult
from ..metrics import AttemptRow, CandidateRow, MetricsStore, RequestRow
from ..routing import Resolution, RoutingTable, collect_attempts

log = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])

# Operator OR service token: a UI playground and another component
# (a Discord connector, say) are both legitimate callers of the front
# door. Config and admin stay operator-only.
_auth = [Depends(require_authorized)]


def _timeout_seconds(store: ConfigStore | None) -> float:
    """What the operator's own knob says, for a message that names it."""
    if store is None:
        return 600.0
    try:
        return float(store.get("requestTimeoutSeconds") or 600)
    except (TypeError, ValueError):
        return 600.0


def _routing(request: Request) -> RoutingTable | None:
    return getattr(request.app.state, "routing", None)


def _lifecycle(request: Request) -> LifecycleManager | None:
    return getattr(request.app.state, "lifecycle", None)


def _store(request: Request) -> ConfigStore | None:
    return getattr(request.app.state, "config_store", None)


def _metrics(request: Request) -> MetricsStore | None:
    return getattr(request.app.state, "metrics", None)


@dataclass(slots=True)
class _Recording:
    """The per-request facts settled before the backend call.

    One value rather than nine keyword arguments threaded through the
    route and into the streaming generator. Everything here is known by
    the time a backend is chosen; what `_record` adds afterwards is only
    what the response taught us.
    """

    metrics: MetricsStore | None
    started: float
    waited_ms: int = 0
    swapped_in: bool = False
    routing_ms: int | None = None
    refreshed: bool = False
    strategy: str | None = None
    candidates: list[CandidateRow] = dc_field(default_factory=list)
    streamed: bool = False


def _record(
    rec: _Recording,
    body: ChatCompletionRequest,
    tries: list[AttemptRow],
    *,
    served_model: str | None = None,
    tier: int | None = None,
    usage: Any = None,
    backend_ms: int | None = None,
) -> None:
    """Join the per-attempt rows to the facts only this route holds.

    The hooks see every attempt and its own elapsed time; the route sees
    the requested model, the wake cost, the token counts and which tier
    answered. Neither is sufficient alone, which is why the two row
    shapes exist rather than one.

    Called on the failure paths as well as the success path, deliberately:
    a request that exhausted every backend is the most interesting row in
    the table, and recording only successes would make a failing backend
    look like an idle one.

    Never raises. An instrumentation bug must not turn a served
    completion into a 500.
    """
    if rec.metrics is None:
        return
    try:
        served = next((a for a in tries if a.served), None)
        # The driver's own measurement belongs to the attempt that
        # produced it, and only the serving attempt has one — a failed
        # attempt returned an exception, not a response with a latency
        # on it.
        if served is not None and backend_ms is not None:
            served.backend_ms = backend_ms
        # Keep the candidate list only when there was a decision to
        # make. One eligible backend and nothing rejected is not an
        # audit trail, it is a row per request saying "the only option
        # was chosen".
        considered = rec.candidates
        if len(considered) < 2 and all(c.eligible for c in considered):
            considered = []
        context = admission.current.get()
        rec.metrics.record(
            RequestRow(
                started_at=datetime.now(UTC),
                requested_model=body.model,
                served_model=served_model,
                attempts=max(1, len(tries)),
                tier=tier if served is not None else None,
                total_ms=int((time.perf_counter() - rec.started) * 1000),
                waited_ms=rec.waited_ms,
                swapped_in=rec.swapped_in,
                streamed=rec.streamed,
                prompt_tokens=getattr(usage, "promptTokens", None),
                completion_tokens=getattr(usage, "completionTokens", None),
                outcome="served" if served is not None else "error",
                tries=list(tries),
                routing_ms=rec.routing_ms,
                refreshed=rec.refreshed,
                strategy=rec.strategy,
                candidates=considered,
                client_key_id=context.key_id if context else None,
                client_key_name=context.key_name if context else None,
                request_id=context.id if context else None,
                elapsed_ms=int((time.perf_counter() - context.started) * 1000) if context else None,
            )
        )
        if (context := admission.current.get()) is not None:
            context.recorded = True
    except Exception:
        log.debug("could not record request metrics", exc_info=True)


def _error(
    *,
    code: int,
    message: str,
    error_type: str,
    param: str | None = None,
) -> JSONResponse:
    """OpenAI's error envelope, not RFC 7807.

    Returned as a plain JSONResponse rather than raised as an
    HTTPException because FastAPI would wrap the latter in its own
    `{"detail": ...}`, which is precisely the shape an OpenAI SDK cannot
    read.
    """
    body: dict[str, Any] = {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": None,
        }
    }
    return JSONResponse(status_code=code, content=body)


@dataclass(frozen=True)
class _Failure:
    """A refusal decided once and rendered per door.

    R4 put a second front door on the same routing. The refusals it
    shares -- no such model, the wrong surface, nothing ready, a
    toolless backend -- have to *say* the same thing on both and be
    *shaped* differently on each, because an OpenAI SDK and an Anthropic
    SDK each parse their own envelope and report anything else as a
    generic failure.

    So the decision and its wording live here and each door renders it.
    The alternative -- a second copy of these messages inside the
    Anthropic translator -- is how two surfaces start disagreeing about
    what the install is doing, which is the failure this project has
    already had between a contract and its implementation more than
    once.
    """

    code: int
    message: str
    error_type: str
    param: str | None = None
    retry_after: str | None = None

    def as_openai(self) -> JSONResponse:
        response = _error(
            code=self.code, message=self.message, error_type=self.error_type, param=self.param
        )
        if self.retry_after:
            response.headers["Retry-After"] = self.retry_after
        return response

    def as_anthropic(self) -> JSONResponse:
        # `status_for` moves exactly one code, and it moves on a
        # measurement: a 404's body is discarded by Claude Code and
        # replaced with a generic message blaming the model, which
        # would throw away the available-model list and the
        # sealed-control-root diagnosis.
        status = anthropic.status_for(self.code)
        response = anthropic.error_response(status, self.message)
        if self.retry_after:
            response.headers["Retry-After"] = self.retry_after
        return response


def _control_root_hint(table: RoutingTable | None) -> str:
    """The sentence this 404 never said: whether the gateway could see
    past this host at all.

    A control root sealed after a container restart left every surface
    reporting healthy and this response naming two healthy places
    (2026-09-12). The routing table knows where it looked and what it
    got; say it. Nothing to add when the root answered -- then the
    problem really is on the agents.
    """
    if table is None:
        return ""
    root = table.control_root()
    if root.source == "none":
        return (
            " Only this host's agent was read: it is not enrolled and runs no control "
            "root, so there is no other node to look at (set controlUrl to override)."
        )
    if not root.reachable:
        return (
            f" The control root at {root.url} did not answer on the last refresh"
            f" ({root.error or 'no node list'}), so only the previously known agents were read."
        )
    return ""


def _no_such_model(model: str, table: RoutingTable | None) -> _Failure:
    if (context := admission.current.get()) is not None and context.key_id is not None:
        return _Failure(
            code=404,
            message="The requested model is unavailable or not permitted.",
            error_type="model_not_found",
        )
    known = table.known_models() if table is not None else []
    if known:
        hint = f" Available models: {', '.join(known)}."
    else:
        hint = (
            " No models are currently routable. Check the agent's "
            "GET /v1/runtimes for an engine in `ready` state, and that an "
            "inference-driver in GET /v1/components is pointed at its url."
        ) + _control_root_hint(table)
    return _Failure(
        code=404,
        message=f"The model {model!r} does not exist.{hint}",
        error_type="invalid_request_error",
        param="model",
    )


def _any_backend_carries_tools(resolution: Resolution) -> bool:
    """Whether anything serving this model can carry tool definitions.

    Deliberately `any` and not the `all` that `GET /v1/models` reports.
    The two answer different questions: `/v1/models` advertises what a
    caller can *rely* on across every replica, and this decides whether
    to refuse a request outright. Refusing when one capable backend
    exists would fail a request that would have worked.
    """
    return any(
        b.info.capabilities is not None and bool(b.info.capabilities.toolCalling)
        for b in resolution.backends()
    )


def _tools_unsupported(model: str) -> _Failure:
    """400, in OpenAI's envelope so a harness's SDK raises properly.

    The alternative -- dropping `tools` and answering -- is the failure
    this whole milestone exists to prevent: a harness cannot tell "the
    model chose not to call anything" from "nobody ever offered it the
    tools", so it retries, re-prompts, and loops. That is the reported
    symptom this project set out to fix, and producing it ourselves
    while claiming to route around it would be worse than not shipping
    tools at all.
    """
    return _Failure(
        code=400,
        message=(
            f"No backend serving {model!r} can carry tool definitions, so the request "
            "was refused rather than answered without them. "
            "GET /v1/models reports x_eugene_plexus.tool_calling per model."
        ),
        error_type="invalid_request_error",
        param="tools",
    )


def _wrong_surface(model: str, surfaces: Sequence[str], *, wanted: str, instead: str) -> _Failure:
    """400 when a model was sent to a surface it does not serve.

    Named rather than generic, because the alternative is what happened
    before this existed: the request went down to the backend and came
    back as `"nomic-embed-text" does not support chat` or, worse, an
    embedding-shaped nothing. The caller could not tell whether they had
    picked the wrong model or hit a broken install.
    """
    return _Failure(
        code=400,
        message=(
            f"The model {model!r} serves {', '.join(surfaces)} and not {wanted}. "
            f"Send this request to {instead} instead. "
            "GET /v1/models reports x_eugene_plexus.surfaces per model."
        ),
        error_type="invalid_request_error",
        param="model",
    )


def _encode_embedding(vector: list[float], encoding_format: str | None) -> list[float] | str:
    """Floats, or the base64 the OpenAI SDKs ask for by default.

    **Done here rather than passed to the backend**, because backends
    differ on whether they implement `encoding_format` and a caller
    should not be able to tell which one answered. Little-endian
    `float32` then base64 -- verified byte-for-byte against a real
    backend's own base64 output rather than inferred from OpenAI's
    documentation, so an SDK that decodes it gets the same numbers it
    would get from OpenAI.
    """
    if encoding_format != "base64":
        return vector
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


def _not_ready(resolution: Resolution, wake: WakeResult | None) -> _Failure:
    """503: something serves this model and none of it can take a request
    right now. Retryable, so deliberately not a 502."""
    states = sorted(
        {
            f"{b.runtime.name}={b.runtime.status}"
            for b in resolution.backends()
            if b.runtime is not None and b.runtime.status
        }
    )
    detail = wake.message if wake is not None else "; ".join(states) or "no backend is ready"
    return _Failure(
        code=503,
        message=(
            f"No backend serving {resolution.model!r} is ready: {detail}. "
            f"Runtimes: {', '.join(states) if states else 'none reported'}."
        ),
        error_type="service_unavailable",
    )


# --------------------------------------------------------------------------- #
# /v1/models
# --------------------------------------------------------------------------- #


@router.get("/v1/models", response_model=ModelList, dependencies=_auth)
async def list_models(request: Request) -> ModelList:
    await admission.authorize(request)
    context = admission.current.get()
    table = _routing(request)
    if table is None:
        return ModelList(object="list", data=[])
    return ModelList(
        object="list",
        data=table.as_model_list(
            context.allowed_models if context else None, local_only=admission.local_only()
        ),
    )


# --------------------------------------------------------------------------- #
# /v1/chat/completions
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _Serving:
    """Everything a door needs once routing has decided.

    R4's one real refactor. `create_chat_completion` used to hold the
    routing phase inline, and a second front door would either have
    duplicated it or reached past it into a driver. Duplicating it is
    the worse of the two: it is where the surface refusal, the tools
    refusal, the refresh-and-wake and -- crucially -- the **recording**
    live, and a door that skipped the recording would be invisible to
    `GET /v1/metrics`. That is M8's finding (the envelope is the wrong
    recording point, and the streaming path was not it) arriving in a
    new place before it could bite twice.
    """

    client: DriverClient
    table: RoutingTable
    generate: GenerateRequest
    rec: _Recording
    started: float
    waited_ms: int
    swapped_in: bool


async def _prepare(
    request: Request, body: ChatCompletionRequest, *, surface: str, instead: str
) -> _Serving | _Failure:
    """Resolve, refuse, refresh, wake, and start the clock.

    Shared by both front doors. Returns the thing to serve with, or the
    refusal to render -- and the refusal is a `_Failure` rather than a
    response, because the two doors shape errors differently and must
    not word them differently.
    """
    # The routing phase starts here, and it used to be unmeasured: the
    # clock below is taken after the wake, so resolving, picking and any
    # refresh happened before anything was timing. A refresh does HTTP
    # to the agent and to every driver, inside the request that
    # triggered it, so "before the clock starts" was not the same as
    # "free".
    await admission.authorize(request, body.model, streamed=bool(body.stream))
    arrived = time.perf_counter()
    refreshed = False

    constraints = _to_generate_request(body, _store(request))
    table = _routing(request)
    if table is None:
        return _Failure(
            code=503,
            message="The gateway is starting up or in safe mode; no routing table exists yet.",
            error_type="service_unavailable",
        )

    resolution = await admission.permitted(table.resolve(body.model), requirements=constraints)
    if not resolution.has_backends():
        return _no_such_model(body.model, table)

    # An embeddings-only model, named on the chat surface. Refused by
    # name here rather than passed down to fail as whatever the backend
    # happens to say -- the library will happily discover, download and
    # launch a dedicated embedding model, so this is a mistake an
    # operator can make entirely inside our own UI.
    surfaces = resolution.surfaces()
    if surfaces and surface not in surfaces:
        return _wrong_surface(body.model, surfaces, wanted=surface, instead=instead)

    # **Refused here, never stripped.** Checked before a backend is
    # picked, because the answer must not depend on which replica the
    # balancer happens to choose: a model that is tool-capable on two of
    # three backends would otherwise work twice and fail once, which is
    # the worst possible way to learn about it.
    if body.tools and not _any_backend_carries_tools(resolution):
        return _tools_unsupported(body.model)

    # Nothing eligible: first make sure that is still true — the snapshot
    # can be a refresh interval behind the agent about readiness, and the
    # first live run met exactly that seam. Then wake a `startOnDemand`
    # runtime if the slot has one, wait for it, and try again. A tier
    # with a startable runtime is awaited rather than skipped — see the
    # lifecycle module.
    images = has_images(body.messages)
    wake: WakeResult | None = None
    client = table.pick(resolution, images=images)
    if client is None and await table.refresh_if_stale():
        refreshed = True
        resolution = await admission.permitted(table.resolve(body.model), requirements=constraints)
        if not resolution.has_backends():
            return _no_such_model(body.model, table)
        client = table.pick(resolution, images=images)
    # What the balancer saw, read before the wake so the numbers are the
    # ones the decision was made on. Read even when only one backend is
    # eligible; `_record` decides whether it is worth keeping.
    considered = table.candidates_considered(resolution)
    if client is None:
        lifecycle = _lifecycle(request)
        if lifecycle is not None:
            wake = await lifecycle.wake(resolution)
            if wake.ok:
                resolution = await admission.permitted(
                    table.resolve(body.model), requirements=constraints
                )
                client = table.pick(resolution, images=images)
                considered = table.candidates_considered(resolution)
        if client is None:
            if images and resolution.eligible_backends():
                return _Failure(
                    code=400,
                    error_type="invalid_request_error",
                    param="messages",
                    message="No ready backend serving this model confirms image input. "
                    "Select a vision model with its projector loaded. GET /v1/models "
                    "reports x_eugene_plexus.image_input; images were not discarded.",
                )
            return _not_ready(resolution, wake)

    store = _store(request)
    if isinstance(client, TieredClient):
        client.authorize_attempt = admission.before_attempt
    generate = _to_generate_request(body, store)
    generate.localOnly = admission.local_only()
    profiles = getattr(request.app.state, "profile_defaults", None)
    if profiles is not None and isinstance(client, TieredClient):
        # Capture the routing snapshot that selected these candidates. A later
        # topology refresh must not join this request to a replacement driver.
        paths = {
            (backend.node, backend.name): backend.runtime.spec.get("modelPath")
            if backend.runtime is not None
            else None
            for backend in resolution.backends()
        }

        async def prepare_candidate(
            candidate: DriverClient, original: GenerateRequest
        ) -> GenerateRequest:
            if (
                body.max_tokens is not None
                and body.temperature is not None
                and body.top_p is not None
            ):
                return original
            defaults = await profiles.get(paths.get((candidate.node, candidate.name)))
            prepared = _to_generate_request(body, store, defaults)
            prepared.localOnly = original.localOnly
            prepared.requestId = original.requestId
            return prepared

        client.prepare_request = prepare_candidate
    waited_ms = wake.waited_ms if wake is not None and wake.ok else 0
    # Excludes the wake, which is `waited_ms` and already reported. The
    # two must not overlap or a swap would be counted twice.
    routing_ms = int((time.perf_counter() - arrived) * 1000) - waited_ms
    swapped_in = bool(wake is not None and wake.ok)

    started = time.perf_counter()
    return _Serving(
        client=client,
        table=table,
        generate=generate,
        started=started,
        waited_ms=waited_ms,
        swapped_in=swapped_in,
        rec=_Recording(
            metrics=_metrics(request),
            started=started,
            waited_ms=waited_ms,
            swapped_in=swapped_in,
            routing_ms=max(0, routing_ms),
            refreshed=refreshed,
            strategy=str(store.get("loadBalancing")) if store is not None else None,
            candidates=considered,
            streamed=bool(body.stream),
        ),
    )


@router.post(
    "/v1/chat/completions", dependencies=_auth, openapi_extra=chat_contract.request_body_schema()
)
async def create_chat_completion(request: Request) -> Any:
    try:
        raw = await request.json()
    except (ValueError, UnicodeDecodeError):
        return _error(
            code=400,
            message="body: must be valid JSON.",
            error_type="invalid_request_error",
            param="body",
        )
    try:
        body = await run_in_threadpool(chat_contract.parse_request, raw)
    except chat_contract.Refusal as exc:
        return _error(
            code=400, message=exc.message, error_type="invalid_request_error", param=exc.field
        )
    prepared = await _prepare(request, body, surface="chat", instead="/v1/embeddings")
    if isinstance(prepared, _Failure):
        return prepared.as_openai()
    client = prepared.client
    table = prepared.table
    generate = prepared.generate
    rec = prepared.rec
    started = prepared.started
    waited_ms = prepared.waited_ms
    swapped_in = prepared.swapped_in
    store = _store(request)

    if body.stream:
        # The generator runs AFTER this function returns, so its
        # collection scope has to live inside it — a `with` block here
        # would be closed before the first attempt was made. The stream
        # records itself.
        return StreamingResponse(
            _stream_completion(body, generate, client, table, started, rec=rec),
            media_type="text/event-stream",
        )

    with collect_attempts() as tries:
        try:
            response = await serve_while_connected(
                request, client.generate(generate), what="a chat completion"
            )
        except ClientGone:
            # The tab closed, or the SDK's own deadline fired and it is
            # already retrying. The backend call is cancelled by now, so
            # the engine is free rather than computing an answer into a
            # closed socket. Still recorded: a run that nobody read is a
            # real cost, and hiding it from /v1/metrics would hide the
            # one signal that says "your clients are giving up on you".
            _record(rec, body, tries)
            return _error(
                code=499,
                message="The client disconnected; the backend call was cancelled.",
                error_type="client_disconnected",
            )
        except DriverError as e:
            _record(rec, body, tries)
            return _driver_failure(e).as_openai()
        except httpx.TimeoutException as e:
            # **Not the cascade's 502, and not a cascade at all.** This
            # is our own read deadline onto the driver: the engine is
            # almost certainly still computing. Cascading recomputes the
            # prompt on every replica and every tier and reports a total
            # failure at the sum of the deadlines, which is what R2.5
            # removed. Say the one thing that helps instead.
            _record(rec, body, tries)
            return _error(
                code=504,
                message=(
                    f"No backend serving {body.model!r} answered within the gateway's "
                    f"requestTimeoutSeconds ({_timeout_seconds(store):g}s). The backend was "
                    f"most likely still working rather than broken -- a large model on CPU, "
                    f"or a partial offload, can take minutes. Raise that setting under "
                    f"Config -> Gateway -> Routing. ({type(e).__name__})"
                ),
                error_type="timeout",
            )
        except httpx.HTTPError as e:
            _record(rec, body, tries)
            return _error(
                code=502,
                message=(
                    f"A backend serving {body.model!r} failed ({type(e).__name__}). "
                    "Work may have occurred; an uncertain outcome is not replayed automatically. "
                    f"The engine may have stopped — check GET /v1/runtimes on the agent."
                ),
                error_type="upstream_error",
            )

        result = _to_chat_completion(
            body, response, client, table, started, waited_ms=waited_ms, swapped_in=swapped_in
        )
        _record(
            rec,
            body,
            tries,
            served_model=result.model,
            tier=getattr(client, "tier", 1),
            usage=response.usage,
            backend_ms=response.latencyMs,
        )
        return result


# --------------------------------------------------------------------------- #
# /v1/messages -- the Anthropic door
# --------------------------------------------------------------------------- #


def _serve_failure(
    e: Exception, body: ChatCompletionRequest, store: ConfigStore | None
) -> _Failure:
    """One backend failure, classified once for both front doors.

    R2.5's distinction is the load-bearing one and it is preserved here
    rather than restated: a **deadline that fired** is 504 and is not a
    cascade, because the engine is almost certainly still computing this
    prompt and the next replica would take the same time on the same
    input. A connect timeout is a dead host and cascades like any other
    transport error, which is why it never reaches this function as a
    `TimeoutException`.
    """
    if isinstance(e, ClientGone):
        return _Failure(
            code=499,
            message="The client disconnected; the backend call was cancelled.",
            error_type="client_disconnected",
        )
    if isinstance(e, DriverError):
        return _driver_failure(e)
    if isinstance(e, httpx.TimeoutException):
        return _Failure(
            code=504,
            message=(
                f"No backend serving {body.model!r} answered within the gateway's "
                f"requestTimeoutSeconds ({_timeout_seconds(store):g}s). The backend was "
                f"most likely still working rather than broken -- a large model on CPU, "
                f"or a partial offload, can take minutes. Raise that setting under "
                f"Config -> Gateway -> Routing. ({type(e).__name__})"
            ),
            error_type="timeout",
        )
    return _Failure(
        code=502,
        message=(
            f"A backend serving {body.model!r} failed ({type(e).__name__}). "
            "Work may have occurred; an uncertain outcome is not replayed automatically. "
            f"The engine may have stopped — check GET /v1/runtimes on the agent."
        ),
        error_type="upstream_error",
    )


async def _stream_anthropic(
    body: ChatCompletionRequest,
    prepared: _Serving,
    *,
    model: str,
) -> AsyncIterator[str]:
    """The driver's stream as Anthropic's typed events.

    Deliberately a sibling of `_stream_completion` rather than a wrapper
    around it: that one emits OpenAI SSE text, and re-parsing our own
    output to re-frame it would put a serializer and a parser between
    the backend and the client for no gain. What the two share is the
    thing that matters -- the same `client.stream`, the same commit
    point, the same `_record` -- and they share it by construction
    because both are handed the same `_Serving`.
    """
    translator = anthropic.StreamTranslator(model)
    with collect_attempts() as tries:
        response: GenerateResponse | None = None
        try:
            async for event in prepared.client.stream(prepared.generate):
                if event.done:
                    # Captured, NOT broken out of -- see
                    # `_stream_completion`. Abandoning the generator at
                    # its yield loses the attempt bookkeeping entirely,
                    # which is R1.4's finding and cost every streamed
                    # request its metrics row.
                    response = event.result
                    continue
                if not translator.started:
                    # On the first DRIVER event, never on acceptance:
                    # `message_start` names the model, and until the
                    # first token the cascade can still change which
                    # backend answers.
                    yield translator.start()
                if event.tool_calls:
                    for chunk in translator.tool_fragments(event.tool_calls):
                        yield chunk
                    continue
                if event.text:
                    for chunk in translator.text(event.text):
                        yield chunk
        except (DriverError, httpx.HTTPError) as e:
            log.warning("anthropic stream for %r failed: %s", body.model, e)
            _record(prepared.rec, body, tries)
            if not translator.started:
                yield translator.start()
            for chunk in translator.failed(str(e) or type(e).__name__):
                yield chunk
            return

        if not translator.started:
            # A backend that produced no tokens at all still owes the
            # client a well-formed stream, or its state machine hangs
            # waiting for a `message_start` that never comes.
            yield translator.start(response.modelId if response is not None else None)

        if response is None:
            # The driver ended without a `done` event: a truncation, not
            # a completion. There is no Anthropic stop reason for "the
            # backend broke mid-stream", so the stream says so as an
            # error event rather than inventing one -- the same call
            # `_stream_completion` makes, in this wire's vocabulary.
            log.warning("driver stream for %r ended without a done event", body.model)
            _record(prepared.rec, body, tries)
            for chunk in translator.failed(
                "The backend stopped before it finished. What arrived above is partial."
            ):
                yield chunk
            return

        _record(
            prepared.rec,
            body,
            tries,
            served_model=response.modelId or body.model,
            tier=getattr(prepared.client, "tier", 1),
            usage=response.usage,
            backend_ms=response.latencyMs,
        )
        truncated = _prompt_truncated(body, response.usage)
        _warn_if_truncated(
            truncated, body, response.usage, getattr(prepared.client, "served_by", None)
        )
        for chunk in translator.finish(
            reason=anthropic.stop_reason(response.finishReason), usage=response.usage
        ):
            yield chunk


@router.post("/v1/messages")
async def create_anthropic_message(request: Request) -> Any:
    """Anthropic's Messages API, translated onto the shared path.

    **Auth is done in the body of this function rather than as a
    `dependencies=` entry, and that is not an oversight.** This door
    accepts a second header (`x-api-key`) and answers **403** where the
    rest of the gateway answers 401, and neither is expressible through
    the shared `HTTPBearer` dependency: it reads only `Authorization`,
    and a raised `HTTPException` is wrapped by FastAPI in its own
    `{"detail": ...}`, which is exactly the shape an Anthropic SDK
    cannot read. A credential error the client cannot parse is a
    credential error it retries.

    **The body is read raw rather than declared as a parameter**, for
    the same family of reason: a declared model hands a validation
    failure to FastAPI, which answers 422 in its own envelope, and the
    contract says a missing `max_tokens` is a 400 naming the field.
    `anthropic.translate_request` validates against the generated model
    and turns a failure into our own shape.
    """
    try:
        await anthropic.authorize(request)
    except anthropic.Refusal as refusal:
        return refusal.response()

    try:
        raw = await request.json()
    except Exception:
        return anthropic.error_response(400, "The request body is not valid JSON.")
    if not isinstance(raw, dict):
        return anthropic.error_response(400, "The request body must be a JSON object.")

    try:
        body = anthropic.translate_request(raw)
    except anthropic.Refusal as refusal:
        return refusal.response()

    prepared = await _prepare(request, body, surface="chat", instead="/v1/embeddings")
    if isinstance(prepared, _Failure):
        return prepared.as_anthropic()

    if body.stream:
        # The generator runs AFTER this function returns, so the
        # attempt-collection scope has to live inside it.
        return StreamingResponse(
            _stream_anthropic(body, prepared, model=body.model),
            media_type="text/event-stream",
            headers=anthropic.compatibility_headers(raw),
        )

    store = _store(request)
    with collect_attempts() as tries:
        try:
            response = await serve_while_connected(
                request, prepared.client.generate(prepared.generate), what="an Anthropic message"
            )
        except (ClientGone, DriverError, httpx.HTTPError) as e:
            _record(prepared.rec, body, tries)
            return _serve_failure(e, body, store).as_anthropic()

        served_model = response.modelId or body.model
        truncated = _prompt_truncated(body, response.usage)
        _warn_if_truncated(
            truncated, body, response.usage, getattr(prepared.client, "served_by", None)
        )
        _record(
            prepared.rec,
            body,
            tries,
            served_model=served_model,
            tier=getattr(prepared.client, "tier", 1),
            usage=response.usage,
            backend_ms=response.latencyMs,
        )
        return JSONResponse(
            content=anthropic.message_response(
                model=served_model,
                content=response.content,
                tool_calls=response.toolCalls,
                finish=response.finishReason,
                usage=response.usage,
            ),
            headers={
                **anthropic.compatibility_headers(raw),
                **anthropic.envelope_headers(
                    _routing_info(
                        prepared.client,
                        prepared.table,
                        body.model,
                        prepared.started,
                        waited_ms=prepared.waited_ms,
                        swapped_in=prepared.swapped_in,
                        prompt_truncated=truncated,
                    )
                ),
            },
        )


# --------------------------------------------------------------------------- #
# /v1/embeddings
# --------------------------------------------------------------------------- #


@router.post("/v1/embeddings", dependencies=_auth)
async def create_embedding(request: Request, body: EmbeddingRequest) -> Any:
    """Text in, vectors out.

    **The one place in this gateway where a slot does not cascade.**
    `pick_embedding` hands back a client over replicas of the requested
    model and nothing else, so a `modelSlots` fallback to a different
    target cannot run here. Vectors from two models occupy different
    spaces; substituting one for the other writes noise into the
    caller's vector store with a 200 and no marker, and the damage
    outlives the request. Same trade as M10's commit point: give up a
    retry rather than return a wrong answer that looks right.
    """
    await admission.authorize(request, body.model)
    table = _routing(request)
    if table is None:
        return _error(
            code=503,
            message="The gateway is starting up or in safe mode; no routing table exists yet.",
            error_type="service_unavailable",
        )

    resolution = await admission.permitted(table.resolve(body.model))
    if not resolution.has_backends():
        return _no_such_model(body.model, table).as_openai()

    surfaces = resolution.surfaces()
    if surfaces and "embeddings" not in surfaces:
        return _wrong_surface(
            body.model, surfaces, wanted="embeddings", instead="/v1/chat/completions"
        ).as_openai()

    # `oneOf: [string, array]` generates `str | Input`, where `Input` is
    # a RootModel wrapping the list -- so the array case needs `.root`
    # rather than iterating the model, which yields its fields.
    #
    # **No validation here, deliberately.** A hand-written check for
    # "non-empty list of strings" was written, then measured to be
    # unreachable: the schema types `input` as string-or-array-of-string
    # with `minItems: 1`, so a token array and an empty batch are both
    # rejected as 422 before this function runs. Keeping the check would
    # have meant shipping a branch that cannot fire, with a test
    # asserting that it does -- which is this project's most familiar
    # mistake, one layer in.
    inputs: list[str] = (
        [body.input] if isinstance(body.input, str) else list(getattr(body.input, "root", []))
    )

    client = table.pick_embedding(resolution)
    if client is None and await table.refresh_if_stale():
        resolution = await admission.permitted(table.resolve(body.model))
        client = table.pick_embedding(resolution)
    if client is None:
        # Deliberately NOT a wake. Idle-unload and start-on-demand are
        # chat-path lifecycle; an embeddings request that woke a runtime
        # would be the first thing in this gateway to do so, and doing
        # it silently here rather than designing it is how surfaces
        # drift apart.
        return _not_ready(resolution, None).as_openai()

    if isinstance(client, TieredClient):
        client.authorize_attempt = admission.before_attempt
    started = time.perf_counter()
    try:
        result = await serve_while_connected(
            request,
            client.embed(
                EmbedRequest(
                    input=inputs,
                    localOnly=admission.local_only(),
                    requestId=str(admission.request_id()) if admission.request_id() else None,
                )
            ),
            what="an embeddings request",
        )
    except ClientGone:
        return _error(
            code=499,
            message="The client disconnected; the backend call was cancelled.",
            error_type="client_disconnected",
        )
    except DriverError as e:
        return _driver_failure(e).as_openai()
    except httpx.TimeoutException as e:
        return _error(
            code=504,
            message=(
                f"No backend serving {body.model!r} answered within the gateway's "
                f"requestTimeoutSeconds ({_timeout_seconds(_store(request)):g}s). It was not "
                f"retried on a replica, because a replica would take the same time on the "
                f"same input. ({type(e).__name__})"
            ),
            error_type="timeout",
        )
    except httpx.HTTPError as e:
        return _error(
            code=502,
            message=(
                f"A backend serving {body.model!r} failed ({type(e).__name__}). "
                "Work may have occurred; an uncertain outcome is not replayed automatically. "
                f"The cascade never reached a different model -- see the contract."
            ),
            error_type="upstream_error",
        )

    # The generated default is the literal string "float", not the enum
    # member -- datamodel-code-generator emits the schema default
    # verbatim -- so an unset field has no `.value`.
    context = admission.current.get()
    if context is not None:
        context.embedding_result = result
    fmt = getattr(body.encoding_format, "value", body.encoding_format) or "float"
    usage = None
    if result.usage is not None:
        usage = EmbeddingUsage(
            prompt_tokens=result.usage.promptTokens,
            total_tokens=result.usage.totalTokens,
        )
    return EmbeddingResponse(
        object=Object.list,
        data=[
            EmbeddingData(object=Object1.embedding, index=i, embedding=_encode_embedding(v, fmt))
            for i, v in enumerate(result.embeddings)
        ],
        model=result.modelId or body.model,
        usage=usage,
        x_eugene_plexus=_routing_info(client, table, body.model, started),
    )


# --------------------------------------------------------------------------- #
# translation
# --------------------------------------------------------------------------- #


def _to_generate_request(
    body: ChatCompletionRequest, store: ConfigStore | None, defaults: dict[str, Any] | None = None
) -> GenerateRequest:
    """OpenAI request -> the driver's uniform surface.

    Every output-affecting parameter is filled in here and sent
    explicitly. A driver never substitutes a default of its own, so if a
    value reaches a backend, the gateway put it there. When the caller
    omits one, use the selected model's default profile, then the install
    default. Profile lookup is per actual candidate, including fallbacks.
    """
    defaults = defaults or {}
    max_tokens = body.max_tokens if body.max_tokens is not None else defaults.get("maxTokens")
    temperature = body.temperature if body.temperature is not None else defaults.get("temperature")
    if store is not None:
        if max_tokens is None:
            max_tokens = store.get("defaultMaxTokens")
        if temperature is None:
            temperature = store.get("defaultTemperature")

    return GenerateRequest(
        callerSettings=[
            target
            for source, target in (
                ("max_tokens", "maxTokens"),
                ("temperature", "temperature"),
                ("top_p", "topP"),
                ("seed", "seed"),
                ("stop", "stop"),
                ("tools", "tools"),
                ("tool_choice", "toolChoice"),
                ("response_format", "responseFormat"),
            )
            if getattr(body, source) is not None
        ]
        or None,
        messages=[_to_driver_message(m) for m in body.messages],
        maxTokens=max_tokens,
        temperature=temperature,
        # **Carried since 2026-09-19 and dropped on the floor before
        # that** -- `GenerateRequest` had no field for either, so the
        # two were accepted here, range-validated by the schema above,
        # and then went nowhere. R8 lets a profile supply top-p, but
        # still invents neither an install-wide top-p nor a seed.
        topP=body.top_p if body.top_p is not None else defaults.get("topP"),
        seed=body.seed,
        stop=([body.stop] if isinstance(body.stop, str) else body.stop.root)
        if body.stop is not None
        else None,
        requestId=admission.request_id(),
        # Tools are the caller's, not ours. Unlike every parameter above
        # them, there is no install default to fall back to and nothing
        # sensible to invent: a tool the caller did not offer is one it
        # cannot execute, so an absent field stays absent.
        tools=_to_driver_tools(body.tools),
        toolChoice=_to_driver_tool_choice(body.tool_choice),
        responseFormat=_to_driver_response_format(body.response_format),
    )


def _to_driver_message(message: ChatCompletionMessage) -> Message:
    """One OpenAI message as the driver's shared `Message`.

    `toolCalls` is typed loosely in `common.yaml` on purpose -- the
    shared schema declines to be a third definition of OpenAI's object
    -- so the calls go across as plain dicts and the driver re-shapes
    them for its backend.
    """
    return Message(
        role=Role(message.role.value),
        content=message.model_dump(mode="json", exclude_none=True).get("content"),
        toolCalls=[c.model_dump(mode="json", exclude_none=True) for c in message.tool_calls]
        if message.tool_calls
        else None,
        toolCallId=message.tool_call_id,
    )


def _to_driver_tools(tools: list[Tool] | None) -> list[DriverTool] | None:
    """Re-shape rather than re-import.

    The two documents define the same OpenAI object and codegen gives us
    two unrelated classes for it; components share schemas, not code, so
    crossing the boundary is a dump and a validate. `parameters` is a
    free-form JSON Schema and rides through untouched.
    """
    if not tools:
        return None
    return [DriverTool.model_validate(t.model_dump(mode="json", exclude_none=True)) for t in tools]


def _to_driver_tool_choice(choice: Any) -> Any:
    """`none` / `auto` / `required`, or an object naming one function.

    The enum members are distinct classes on the two sides even though
    they carry identical values, which is what "share schemas, not code"
    costs at every boundary and buys in not coupling two repos'
    releases.
    """
    if choice is None:
        return None
    if hasattr(choice, "model_dump"):
        return DriverNamedToolChoice.model_validate(
            choice.model_dump(mode="json", exclude_none=True)
        )
    return DriverToolChoice(str(getattr(choice, "value", choice)))


def _to_driver_response_format(fmt: ResponseFormat | None) -> DriverResponseFormat | None:
    if fmt is None:
        return None
    return DriverResponseFormat.model_validate(
        fmt.model_dump(mode="json", by_alias=True, exclude_none=True)
    )


def _to_openai_tool_calls(calls: Any) -> list[ToolCall] | None:
    """The driver's tool calls in OpenAI's shape, for a client SDK."""
    if not calls:
        return None
    return [
        ToolCall(
            id=c.id,
            type="function",
            function=FunctionCall(name=c.function.name, arguments=c.function.arguments),
        )
        for c in calls
    ]


def _to_openai_tool_call_deltas(fragments: list[dict[str, Any]]) -> list[ToolCallDelta]:
    """Driver fragments re-framed as OpenAI deltas.

    The two wires use the same field names here, so this is a validate
    rather than a translation -- but it stays explicit, because the one
    field the gateway must not lose is `index`: it is what lets a client
    reassemble two interleaved calls, and a fragment that arrived
    without one would silently merge them into a third call that was
    never made.
    """
    out: list[ToolCallDelta] = []
    for position, fragment in enumerate(fragments):
        data = dict(fragment)
        if not isinstance(data.get("index"), int):
            data["index"] = position
        out.append(ToolCallDelta.model_validate(data))
    return out


_FINISH_BY_DRIVER_REASON = {
    "stop": FinishReason.stop,
    "stop_sequence": FinishReason.stop,
    "length": FinishReason.length,
    # The value that makes an agent loop terminate correctly. A caller
    # that sees `stop` here stops; one that sees `tool_calls` dispatches
    # and comes back. Getting this wrong does not look like an error --
    # it looks like a model that answered instead of using its tools.
    "tool_calls": FinishReason.tool_calls,
    # `content_filter` is OpenAI's OWN value, so this is a pass-through
    # rather than an invention. Until 2026-09-19 the chain was
    # `content_filter` -> the driver's `error` -> `stop`, and a refusal
    # arrived as a natural end: the same mistake as `tool_calls` ->
    # `stop` before step 6, one row along, and with the same shape --
    # the map had no member for a state nobody had used yet, so the
    # state was reported as its nearest neighbour.
    "content_filter": FinishReason.content_filter,
    # A truncated-by-error generation is reported as `stop` with the text
    # that did arrive, matching what OpenAI does — there is no OpenAI
    # finish reason for "the backend broke mid-stream", and inventing one
    # would break clients that switch on this field. This is why
    # `content_filter` had to become its own driver value first: folded
    # in here it was indistinguishable from a backend that broke, and
    # only one of the two has a name the caller already knows.
    "error": FinishReason.stop,
}


def _finish_reason(response: GenerateResponse) -> FinishReason:
    return _FINISH_BY_DRIVER_REASON.get(response.finishReason.value, FinishReason.stop)


# A prompt whose reported token count falls below one token per this
# many characters did not arrive intact. **Chosen to be far below any
# real tokenizer**, because the cost of the two errors is not
# symmetrical: a false positive accuses a healthy backend, while the
# condition itself is so gross when it happens that no tight bound is
# needed to see it. Measured on this hardware: 66,389 characters came
# back as 86 prompt tokens -- one token per 772 characters. English
# prose runs about 4, and the most token-efficient real input anyone
# sends (long runs of whitespace) does not reach 20.
#
# Note this is NOT the inverse of an estimator. `chars/4` was measured
# to underestimate a real prompt by 19.4% (16,568 against 20,560), which
# makes it unsafe for deciding whether something will *fit*. Deciding
# whether something *arrived* is a different question with a much wider
# margin, and it is the only one asked here.
_TRUNCATION_CHARS_PER_TOKEN = 20

# Below this the ratio stops meaning anything: no context window in use
# anywhere is small enough for a prompt this size to be truncated, and
# short inputs are where token density varies most.
_TRUNCATION_MIN_CHARS = 1000


def _prompt_chars(body: ChatCompletionRequest) -> int:
    """How much text we handed the backend.

    Counts message content only. Tool definitions and the chat template
    add tokens we do not count here, which makes this an **under**count
    of what was really sent -- and therefore biases the detector toward
    silence rather than toward false accusation, which is the direction
    to be wrong in.
    """
    total = 0
    for message in body.messages or []:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            total += len(content)
    return total


def _prompt_truncated(body: ChatCompletionRequest, usage: Any) -> bool | None:
    """Did the backend silently drop most of the input?

    The failure this catches: a server that fits an over-long prompt into
    its window by discarding the middle of the conversation and answering
    anyway, with a 200 and no flag of its own. Ollama does this --
    verified with canaries, where the system message and the last turn
    survived and the first and middle did not. A coding harness sends
    file contents and gets a confident answer about code the model never
    received, which is indistinguishable from the model being wrong.

    `None` means not evaluated, and is a different claim from `False`:
    a backend that reports no usage cannot be checked at all, and a
    prompt below the floor is too small for the ratio to mean anything.

    Detected after the fact on purpose. Predicting it would need a
    tokenizer we do not have for the backends that do it -- Ollama
    exposes none -- while this needs nothing but the number the backend
    already returns.
    """
    reported = getattr(usage, "promptTokens", None)
    if not isinstance(reported, int) or reported <= 0:
        return None
    chars = _prompt_chars(body)
    if chars < _TRUNCATION_MIN_CHARS:
        return None
    return reported * _TRUNCATION_CHARS_PER_TOKEN < chars


def _warn_if_truncated(
    truncated: bool | None,
    body: ChatCompletionRequest,
    usage: Any,
    driver: str | None,
) -> None:
    """Say it in the log too, because the flag rides a field most clients
    drop. An operator chasing "the model keeps ignoring my files" needs
    this line to exist somewhere they will look."""
    if not truncated:
        return
    log.warning(
        "backend %r reported consuming only %s prompt tokens for %d characters of input: "
        "it discarded most of the prompt and answered anyway. The answer is about what "
        "survived. Raise the backend's context window, or send less.",
        driver,
        getattr(usage, "promptTokens", None),
        _prompt_chars(body),
    )


def _routing_info(
    client: DriverClient,
    table: RoutingTable,
    model: str,
    started: float,
    *,
    waited_ms: int = 0,
    swapped_in: bool = False,
    prompt_truncated: bool | None = None,
) -> CompletionRoutingInfo:
    """Which backend actually served this, and whether failover fired.

    Not part of OpenAI's schema; clients ignore unknown fields. It exists
    because the failure mode of a routing layer is opacity — with several
    backends behind one name, an operator seeing a slow or odd response
    needs to know which one answered without going to the logs.
    `attempts > 1` is the visible evidence the cascade ran; `tier > 1`
    says a later target answered; `swapped_in` is the visible cost of
    idle unload, next to the request that paid it.
    """
    # `served_by` is the backend that actually answered, which after a
    # cascade is not the primary. Both client kinds expose it.
    served_by = getattr(client, "served_by", None) or getattr(client, "name", model)
    # Which machine's driver answered. Two machines running one model
    # give two drivers with one name, so a lookup by name alone answered
    # about whichever sorted first (R1.6, review §6.1 #8).
    served_node = getattr(client, "served_by_node", None)
    attempts = getattr(client, "attempts", 1)
    tier = getattr(client, "tier", 1)
    # inference-driver.yaml and gateway.yaml each generate their own
    # BackendKind with the same wire values; cross via `.value`.
    backend_kind = next(
        (
            BackendKind(b.info.backend.value)
            for b in table.backends_for(model)
            if b.name == served_by and (served_node is None or b.node == served_node)
        ),
        None,
    )
    return CompletionRoutingInfo(
        driver=served_by,
        # The engine process behind the driver that answered, when the
        # agent supervises one — by name, so replicas are attributed too.
        runtime=table.runtime_for(model, served_by, served_node),
        backend=backend_kind,
        latency_ms=int((time.perf_counter() - started) * 1000),
        attempts=attempts if isinstance(attempts, int) and attempts >= 1 else 1,
        tier=tier if isinstance(tier, int) and tier >= 1 else 1,
        swapped_in=swapped_in,
        waited_ms=waited_ms,
        # The window that applied to *this* request, which is not the
        # smallest across every backend serving the name -- that is what
        # `GET /v1/models` reports, and it is the right number there and
        # the wrong one here.
        context_length=table.context_length_for(model, served_by, served_node),
        prompt_truncated=prompt_truncated,
    )


def _to_chat_completion(
    body: ChatCompletionRequest,
    response: GenerateResponse,
    client: DriverClient,
    table: RoutingTable,
    started: float,
    *,
    waited_ms: int = 0,
    swapped_in: bool = False,
) -> ChatCompletionResponse:
    usage = None
    if response.usage is not None:
        usage = CompletionUsage(
            prompt_tokens=response.usage.promptTokens,
            completion_tokens=response.usage.completionTokens,
            total_tokens=response.usage.totalTokens,
        )
    truncated = _prompt_truncated(body, response.usage)
    _warn_if_truncated(truncated, body, response.usage, getattr(client, "served_by", None))
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        # What actually served the request. Equal to the requested id in
        # the normal case; after a cascade it names what answered.
        model=response.modelId or body.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role=Role1.assistant,
                    content=response.content,
                    tool_calls=_to_openai_tool_calls(response.toolCalls),
                ),
                finish_reason=_finish_reason(response),
            )
        ],
        usage=usage,
        x_eugene_plexus=_routing_info(
            client,
            table,
            body.model,
            started,
            waited_ms=waited_ms,
            swapped_in=swapped_in,
            prompt_truncated=truncated,
        ),
    )


async def _stream_completion(
    body: ChatCompletionRequest,
    generate: GenerateRequest,
    client: DriverClient,
    table: RoutingTable,
    started: float,
    *,
    rec: _Recording,
) -> AsyncIterator[str]:
    """SSE framed exactly as OpenAI frames it, one frame per token.

    From M0 to M9 this framed correctly and **did not stream**: it
    awaited the whole completion and emitted it as a single content
    chunk, because the driver's own stream endpoint was a 501 stub. The
    framing was right, which is why every OpenAI client worked
    unmodified, and the delivery was not, which is why the playground
    sat silent and then printed everything at once. M10 plumbed the
    driver's SSE through the cascade and this loop forwards tokens as
    they arrive.

    **The rule that came with it**: past the first token the slot is
    committed and cannot fail over — see `TieredClient.stream`. So the
    error path below carries two cases that look identical from here: a
    backend that never produced anything (the cascade ran and lost) and
    one that broke mid-answer (truncation). Both become an error frame,
    because the 200 is long gone either way.

    Since M8 this path also records itself and carries the routing
    extension on its final frame. Both were missing, and the second
    caused the first to be nearly overlooked: the non-streaming response
    carried `x_eugene_plexus` and this one did not, so a recorder written
    against the response would have been blind to exactly the clients
    that stream.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def frame(chunk: ChatCompletionChunk) -> str:
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    def envelope(
        *,
        delta: Delta,
        finish: FinishReason | None,
        model: str,
        usage: Any = None,
        routing: CompletionRoutingInfo | None = None,
    ) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id=completion_id,
            object="chat.completion.chunk",
            created=created,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=delta,
                    finish_reason=finish,  # type: ignore[arg-type]
                )
            ],
            usage=usage,
            x_eugene_plexus=routing,
        )

    with collect_attempts() as tries:
        emitted_role = False
        response: GenerateResponse | None = None
        try:
            async for event in client.stream(generate):
                if event.done:
                    # Captured, NOT broken out of. Breaking here abandons
                    # the generator while it is suspended at its yield, so
                    # `TieredClient.stream` never reaches the branch that
                    # reports the attempt as served -- and the request is
                    # recorded with zero attempts. The driver's stream ends
                    # immediately after `done`, so letting the loop finish
                    # costs nothing and keeps the bookkeeping honest.
                    response = event.result
                    continue
                if not emitted_role:
                    # OpenAI puts the role on the first chunk, and it has
                    # to wait until there is a first chunk: emitting it
                    # before the stream opens would commit a model name
                    # the cascade might still change.
                    emitted_role = True
                    yield frame(
                        envelope(
                            delta=Delta(role=Role2.assistant),
                            finish=None,
                            model=body.model,
                        )
                    )
                if event.tool_calls:
                    # Forwarded as they land rather than accumulated to
                    # the terminal frame: a harness wants to start
                    # dispatching, and holding fragments back would make
                    # a streamed tool call strictly worse than a
                    # non-streamed one. Note this is past the commit
                    # point -- `TieredClient.stream` set `committed` on
                    # the first event of any kind, so a backend that
                    # dies here truncates rather than splicing half a
                    # call onto another model's.
                    yield frame(
                        envelope(
                            delta=Delta(tool_calls=_to_openai_tool_call_deltas(event.tool_calls)),
                            finish=None,
                            model=body.model,
                        )
                    )
                    continue
                yield frame(
                    envelope(delta=Delta(content=event.text), finish=None, model=body.model)
                )
        except (DriverError, httpx.HTTPError) as e:
            # An error after the stream has been opened cannot become an
            # HTTP status — the 200 is already sent. OpenAI's own
            # behaviour is to emit an error frame, so do that and then
            # terminate normally.
            #
            # Since M10 this also covers the truncation case: past the
            # first token the slot is committed, so `TieredClient.stream`
            # re-raises instead of failing over, and what the client has
            # already received stands.
            log.warning("streaming completion for %r failed: %s", body.model, e)
            _record(rec, body, tries)
            yield (
                "data: "
                + json.dumps(
                    {
                        "error": {
                            "message": str(e),
                            "type": "upstream_error",
                            "param": None,
                            "code": None,
                        }
                    }
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"
            return

        if response is None:
            # The driver ended without a `done` event. Nothing to
            # summarise: no usage, no served model, no finish reason of
            # its own, and inventing any of them would report a
            # truncation as a completion. The attempt is already
            # recorded `served=False` by the slot, which is where that
            # belongs.
            #
            # What the client is still owed is a well-formed stream. An
            # OpenAI client reads an answer as finished when the
            # terminal chunk carries a `finish_reason`; until 2026-09-18
            # this path emitted `[DONE]` with no terminal chunk at all,
            # so a client switching on that field never saw one and a
            # truncated answer was indistinguishable from a complete
            # one. `stop` rather than an invented value, for the reason
            # `_FINISH_BY_DRIVER_REASON` already gives: there is no
            # OpenAI finish reason for "the backend broke mid-stream",
            # and the truncation is reported where it can be acted on --
            # the log line above and the metrics row.
            log.warning("driver stream for %r ended without a done event", body.model)
            _record(rec, body, tries)
            if not emitted_role:
                yield frame(
                    envelope(delta=Delta(role=Role2.assistant), finish=None, model=body.model)
                )
            yield frame(envelope(delta=Delta(), finish=FinishReason.stop, model=body.model))
            yield "data: [DONE]\n\n"
            return

        served_model = response.modelId or body.model
        if not emitted_role:
            # A backend that produced no tokens at all still owes the
            # client a well-formed stream.
            yield frame(
                envelope(delta=Delta(role=Role2.assistant), finish=None, model=served_model)
            )

        usage = None
        if response.usage is not None:
            usage = CompletionUsage(
                prompt_tokens=response.usage.promptTokens,
                completion_tokens=response.usage.completionTokens,
                total_tokens=response.usage.totalTokens,
            )
        _record(
            rec,
            body,
            tries,
            served_model=served_model,
            tier=getattr(client, "tier", 1),
            usage=response.usage,
            backend_ms=response.latencyMs,
        )
        # A truncated prompt is reported on the streamed path too, and it
        # can only be a flag here: the answer has already been delivered
        # and M10's rule is that a stream cannot be unsent. Flagging one
        # path while failing the other would report one condition two
        # different ways depending on a parameter the caller chose for
        # unrelated reasons.
        truncated = _prompt_truncated(body, response.usage)
        _warn_if_truncated(truncated, body, response.usage, getattr(client, "served_by", None))
        # The routing extension rides the final frame, beside `usage` —
        # the same place OpenAI puts its end-of-stream extras, and the
        # earliest point at which any of it is known.
        yield frame(
            envelope(
                delta=Delta(),
                finish=_finish_reason(response),
                model=served_model,
                usage=usage if body.stream_options is None else None,
                routing=_routing_info(
                    client,
                    table,
                    body.model,
                    started,
                    waited_ms=rec.waited_ms,
                    swapped_in=rec.swapped_in,
                    prompt_truncated=truncated,
                ),
            )
        )
        if (
            body.stream_options is not None
            and body.stream_options.include_usage
            and usage is not None
        ):
            yield frame(
                ChatCompletionChunk(
                    id=completion_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=served_model,
                    choices=[],
                    usage=usage,
                )
            )
        yield "data: [DONE]\n\n"


def _driver_failure(e: DriverError) -> _Failure:
    from ..driver_client import retry_disposition

    failure = _driver_failure_status(e)
    if retry_disposition(e) == "indeterminate":
        failure = replace(
            failure,
            message=failure.message
            + " Outcome unknown: work may have occurred; no automatic replay.",
        )
    delay = e.problem.retryAfterSeconds if e.problem is not None else None
    if delay is not None and math.isfinite(delay):
        failure = replace(failure, retry_after=str(math.ceil(delay)))
    return failure


def _driver_failure_status(e: DriverError) -> _Failure:
    """Map a driver failure onto the status the client should act on.

    The split matters operationally. A 4xx from the backend is a request
    or config bug that the next backend would hit identically, so it does
    not cascade and comes back as a 400. A 503 means something serves
    this model but isn't ready — usually an engine still loading — and is
    worth retrying, which is why it is not folded into the 502 that means
    "the cascade ran and every backend lost".
    """
    upstream = e.status_code
    detail = e.problem.detail if e.problem is not None and e.problem.detail else str(e)

    if upstream == 504:
        # The driver's own deadline fired. Not folded into the 502,
        # because 502 is what the cascade says when every backend was
        # tried and lost -- and this one was deliberately NOT retried
        # elsewhere (R2.5): the next replica would take the same time
        # to compute the same prompt. The caller is told what to turn.
        return _Failure(
            code=504,
            message=(
                f"A backend was still working when the deadline passed: {detail} "
                f"It was not retried on another backend, because another backend would "
                f"take the same time on the same prompt. Raise requestTimeoutSeconds on "
                f"the gateway (Config -> Gateway -> Routing) if this model needs longer."
            ),
            error_type="timeout",
        )
    if upstream == 503:
        return _Failure(
            code=503,
            message=(
                f"A backend serving this model is not ready yet: {detail}. "
                f"If an engine is still loading its weights this will clear on its own."
            ),
            error_type="service_unavailable",
        )
    if upstream in (401, 403):
        # **Not the caller's 400, since 2026-09-19.** Everything else in
        # the 4xx range here means the BACKEND refused the request --
        # a prompt longer than the context window, overwhelmingly --
        # which is the caller's to fix. These two mean the DRIVER
        # refused us: the gateway's own `service:gateway` token was
        # rejected, which is what a rotated signing key on one node
        # looks like from here. The request was correct and the install
        # is not, so reporting it as `invalid_request_error` sent a
        # harness to re-read a prompt that never had a problem, and
        # gave the operator no hint that a credential was involved.
        #
        # 502 and not 401: a 401 from this door is about the CALLER's
        # bearer, and there is exactly one thing worse than blaming the
        # caller's prompt, which is blaming their key. R4 measured what
        # a 401 costs on the other door -- an unbounded silent retry
        # loop -- and that is reason enough not to reuse the status for
        # something the caller cannot act on either.
        return _Failure(
            code=502,
            message=(
                f"The driver {e.driver_name!r} at {e.driver_url} refused the gateway's "
                f"credential (HTTP {upstream}): {detail} Nothing is wrong with this "
                f"request. The gateway holds a `service:gateway` token minted by its own "
                f"node's agent; re-check that driver's auth keys, or restart it so it "
                f"picks up the install's current signing key."
            ),
            error_type="upstream_auth_error",
        )
    if 400 <= upstream < 500:
        return _Failure(
            code=400,
            message=f"The backend rejected the request: {detail}",
            error_type="invalid_request_error",
        )
    return _Failure(
        code=502,
        message=f"Every backend serving this model failed. Last error: {detail}",
        error_type="upstream_error",
    )
