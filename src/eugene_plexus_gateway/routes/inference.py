"""The OpenAI-compatible front door: /v1/models and /v1/chat/completions.

These two operations are the only ones in Eugene Plexus that use
snake_case field names and OpenAI's error envelope. That is deliberate:
"OpenAI-compatible" is worth nothing unless an unmodified OpenAI SDK can
point its `base_url` here and work, and those SDKs parse the error shape
to build their exceptions. Renaming `max_tokens` to `maxTokens` for house
consistency would break the entire audience.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .._generated.driver_models import (
    GenerateRequest,
    GenerateResponse,
    Message,
    Role,
)
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
    FinishReason,
    ModelList,
    Role1,
    Role2,
)
from ..config import ConfigStore
from ..dependencies import require_authorized
from ..driver_client import DriverClient, DriverError
from ..lifecycle import LifecycleManager, WakeResult
from ..metrics import AttemptRow, CandidateRow, MetricsStore, RequestRow
from ..routing import Resolution, RoutingTable, collect_attempts

log = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])

# Operator OR service token: a UI playground and another component
# (a Discord connector, say) are both legitimate callers of the front
# door. Config and admin stay operator-only.
_auth = [Depends(require_authorized)]


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
        rec.metrics.record(
            RequestRow(
                started_at=datetime.now(UTC),
                requested_model=body.model,
                served_model=served_model,
                attempts=max(1, len(tries)),
                tier=tier if served is not None else None,
                total_ms=int((time.monotonic() - rec.started) * 1000),
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
            )
        )
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


def _no_such_model(model: str, table: RoutingTable | None) -> JSONResponse:
    known = table.known_models() if table is not None else []
    if known:
        hint = f" Available models: {', '.join(known)}."
    else:
        hint = (
            " No models are currently routable. Check the agent's "
            "GET /v1/runtimes for an engine in `ready` state, and that an "
            "inference-driver in GET /v1/components is pointed at its url."
        )
    return _error(
        code=404,
        message=f"The model {model!r} does not exist.{hint}",
        error_type="invalid_request_error",
        param="model",
    )


def _not_ready(resolution: Resolution, wake: WakeResult | None) -> JSONResponse:
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
    return _error(
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
    table = _routing(request)
    if table is None:
        return ModelList(object="list", data=[])
    return ModelList(object="list", data=table.as_model_list())


# --------------------------------------------------------------------------- #
# /v1/chat/completions
# --------------------------------------------------------------------------- #


@router.post("/v1/chat/completions", dependencies=_auth)
async def create_chat_completion(request: Request, body: ChatCompletionRequest) -> Any:
    # The routing phase starts here, and it used to be unmeasured: the
    # clock below is taken after the wake, so resolving, picking and any
    # refresh happened before anything was timing. A refresh does HTTP
    # to the agent and to every driver, inside the request that
    # triggered it, so "before the clock starts" was not the same as
    # "free".
    arrived = time.monotonic()
    refreshed = False

    table = _routing(request)
    if table is None:
        return _error(
            code=503,
            message="The gateway is starting up or in safe mode; no routing table exists yet.",
            error_type="service_unavailable",
        )

    resolution = table.resolve(body.model)
    if not resolution.has_backends():
        return _no_such_model(body.model, table)

    # Nothing eligible: first make sure that is still true — the snapshot
    # can be a refresh interval behind the agent about readiness, and the
    # first live run met exactly that seam. Then wake a `startOnDemand`
    # runtime if the slot has one, wait for it, and try again. A tier
    # with a startable runtime is awaited rather than skipped — see the
    # lifecycle module.
    wake: WakeResult | None = None
    client = table.pick(resolution)
    if client is None and await table.refresh_if_stale():
        refreshed = True
        resolution = table.resolve(body.model)
        if not resolution.has_backends():
            return _no_such_model(body.model, table)
        client = table.pick(resolution)
    # What the balancer saw, read before the wake so the numbers are the
    # ones the decision was made on. Read even when only one backend is
    # eligible; `_record` decides whether it is worth keeping.
    considered = table.candidates_considered(resolution)
    if client is None:
        lifecycle = _lifecycle(request)
        if lifecycle is not None:
            wake = await lifecycle.wake(resolution)
            if wake.ok:
                resolution = table.resolve(body.model)
                client = table.pick(resolution)
                considered = table.candidates_considered(resolution)
        if client is None:
            return _not_ready(resolution, wake)

    store = _store(request)
    generate = _to_generate_request(body, store)
    waited_ms = wake.waited_ms if wake is not None and wake.ok else 0
    # Excludes the wake, which is `waited_ms` and already reported. The
    # two must not overlap or a swap would be counted twice.
    routing_ms = int((time.monotonic() - arrived) * 1000) - waited_ms
    swapped_in = bool(wake is not None and wake.ok)

    started = time.monotonic()
    rec = _Recording(
        metrics=_metrics(request),
        started=started,
        waited_ms=waited_ms,
        swapped_in=swapped_in,
        routing_ms=max(0, routing_ms),
        refreshed=refreshed,
        strategy=str(store.get("loadBalancing")) if store is not None else None,
        candidates=considered,
        streamed=bool(body.stream),
    )

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
            response = await client.generate(generate)
        except DriverError as e:
            _record(rec, body, tries)
            return _driver_error_response(e)
        except httpx.HTTPError as e:
            _record(rec, body, tries)
            return _error(
                code=502,
                message=(
                    f"Every backend serving {body.model!r} failed. Last error: {e}. "
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
# translation
# --------------------------------------------------------------------------- #


def _to_generate_request(body: ChatCompletionRequest, store: ConfigStore | None) -> GenerateRequest:
    """OpenAI request -> the driver's uniform surface.

    Every output-affecting parameter is filled in here and sent
    explicitly. A driver never substitutes a default of its own, so if a
    value reaches a backend, the gateway put it there. When the caller
    omits one we fall back to the install default rather than leaving it
    unset — "unset" would hand the decision to whatever the backend
    happens to do.
    """
    max_tokens = body.max_tokens
    temperature = body.temperature
    if store is not None:
        if max_tokens is None:
            max_tokens = store.get("defaultMaxTokens")
        if temperature is None:
            temperature = store.get("defaultTemperature")

    return GenerateRequest(
        messages=[Message(role=Role(m.role.value), content=m.content) for m in body.messages],
        maxTokens=max_tokens,
        temperature=temperature,
        stop=body.stop,
        requestId=None,
    )


_FINISH_BY_DRIVER_REASON = {
    "stop": FinishReason.stop,
    "stop_sequence": FinishReason.stop,
    "length": FinishReason.length,
    # A truncated-by-error generation is reported as `stop` with the text
    # that did arrive, matching what OpenAI does — there is no OpenAI
    # finish reason for "the backend broke mid-stream", and inventing one
    # would break clients that switch on this field.
    "error": FinishReason.stop,
}


def _finish_reason(response: GenerateResponse) -> FinishReason:
    return _FINISH_BY_DRIVER_REASON.get(response.finishReason.value, FinishReason.stop)


def _routing_info(
    client: DriverClient,
    table: RoutingTable,
    model: str,
    started: float,
    *,
    waited_ms: int = 0,
    swapped_in: bool = False,
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
    attempts = getattr(client, "attempts", 1)
    tier = getattr(client, "tier", 1)
    # inference-driver.yaml and gateway.yaml each generate their own
    # BackendKind with the same wire values; cross via `.value`.
    backend_kind = next(
        (
            BackendKind(b.info.backend.value)
            for b in table.backends_for(model)
            if b.name == served_by
        ),
        None,
    )
    return CompletionRoutingInfo(
        driver=served_by,
        # The engine process behind the driver that answered, when the
        # agent supervises one — by name, so replicas are attributed too.
        runtime=table.runtime_for(model, served_by),
        backend=backend_kind,
        latency_ms=int((time.monotonic() - started) * 1000),
        attempts=attempts if isinstance(attempts, int) and attempts >= 1 else 1,
        tier=tier if isinstance(tier, int) and tier >= 1 else 1,
        swapped_in=swapped_in,
        waited_ms=waited_ms,
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
                message=ChatCompletionMessage(role=Role1.assistant, content=response.content),
                finish_reason=_finish_reason(response),
            )
        ],
        usage=usage,
        x_eugene_plexus=_routing_info(
            client, table, body.model, started, waited_ms=waited_ms, swapped_in=swapped_in
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
    """SSE framed exactly as OpenAI frames it.

    M0 streams the whole completion as a single content chunk rather than
    proxying the driver's token stream. That is a deliberate, visible
    limitation: the framing is what clients depend on, so getting it
    right matters more than incremental delivery, and true token
    pass-through needs the driver's SSE surface plumbed through the
    cascade logic — which is its own piece of work.

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
        try:
            response = await client.generate(generate)
        except (DriverError, httpx.HTTPError) as e:
            # An error after the stream has been opened cannot become an
            # HTTP status — the 200 is already sent. OpenAI's own
            # behaviour is to emit an error frame, so do that and then
            # terminate normally.
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

        served_model = response.modelId or body.model
        # First chunk carries the role, per OpenAI.
        yield frame(envelope(delta=Delta(role=Role2.assistant), finish=None, model=served_model))
        if response.content:
            yield frame(
                envelope(delta=Delta(content=response.content), finish=None, model=served_model)
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
        # The routing extension rides the final frame, beside `usage` —
        # the same place OpenAI puts its end-of-stream extras, and the
        # earliest point at which any of it is known.
        yield frame(
            envelope(
                delta=Delta(),
                finish=_finish_reason(response),
                model=served_model,
                usage=usage,
                routing=_routing_info(
                    client,
                    table,
                    body.model,
                    started,
                    waited_ms=rec.waited_ms,
                    swapped_in=rec.swapped_in,
                ),
            )
        )
        yield "data: [DONE]\n\n"


def _driver_error_response(e: DriverError) -> JSONResponse:
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

    if upstream == 503:
        return _error(
            code=503,
            message=(
                f"A backend serving this model is not ready yet: {detail}. "
                f"If an engine is still loading its weights this will clear on its own."
            ),
            error_type="service_unavailable",
        )
    if 400 <= upstream < 500:
        return _error(
            code=400,
            message=f"The backend rejected the request: {detail}",
            error_type="invalid_request_error",
        )
    return _error(
        code=502,
        message=f"Every backend serving this model failed. Last error: {detail}",
        error_type="upstream_error",
    )
