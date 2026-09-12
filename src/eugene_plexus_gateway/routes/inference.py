"""The OpenAI-compatible front door: /v1/models and /v1/chat/completions.

These two operations are the only ones in Eugene Plexus that use
snake_case field names and OpenAI's error envelope. That is deliberate:
"OpenAI-compatible" is worth nothing unless an unmodified OpenAI SDK can
point its `base_url` here and work, and those SDKs parse the error shape
to build their exceptions. Renaming `max_tokens` to `maxTokens` for house
consistency would break the entire audience.
"""

from __future__ import annotations

import base64
import json
import logging
import struct
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

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


def _tools_unsupported(model: str) -> JSONResponse:
    """400, in OpenAI's envelope so a harness's SDK raises properly.

    The alternative -- dropping `tools` and answering -- is the failure
    this whole milestone exists to prevent: a harness cannot tell "the
    model chose not to call anything" from "nobody ever offered it the
    tools", so it retries, re-prompts, and loops. That is the reported
    symptom this project set out to fix, and producing it ourselves
    while claiming to route around it would be worse than not shipping
    tools at all.
    """
    return _error(
        code=400,
        message=(
            f"No backend serving {model!r} can carry tool definitions, so the request "
            "was refused rather than answered without them. "
            "GET /v1/models reports x_eugene_plexus.tool_calling per model."
        ),
        error_type="invalid_request_error",
        param="tools",
    )


def _wrong_surface(
    model: str, surfaces: Sequence[str], *, wanted: str, instead: str
) -> JSONResponse:
    """400 when a model was sent to a surface it does not serve.

    Named rather than generic, because the alternative is what happened
    before this existed: the request went down to the backend and came
    back as `"nomic-embed-text" does not support chat` or, worse, an
    embedding-shaped nothing. The caller could not tell whether they had
    picked the wrong model or hit a broken install.
    """
    return _error(
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

    # An embeddings-only model, named on the chat surface. Refused by
    # name here rather than passed down to fail as whatever the backend
    # happens to say -- the library will happily discover, download and
    # launch a dedicated embedding model, so this is a mistake an
    # operator can make entirely inside our own UI.
    surfaces = table.surfaces_for(body.model)
    if surfaces and "chat" not in surfaces:
        return _wrong_surface(body.model, surfaces, wanted="chat", instead="/v1/embeddings")

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

    surfaces = table.surfaces_for(body.model)
    if surfaces and "embeddings" not in surfaces:
        return _wrong_surface(
            body.model, surfaces, wanted="embeddings", instead="/v1/chat/completions"
        )

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
        resolution = table.resolve(body.model)
        client = table.pick_embedding(resolution)
    if client is None:
        # Deliberately NOT a wake. Idle-unload and start-on-demand are
        # chat-path lifecycle; an embeddings request that woke a runtime
        # would be the first thing in this gateway to do so, and doing
        # it silently here rather than designing it is how surfaces
        # drift apart.
        return _not_ready(resolution, None)

    started = time.monotonic()
    try:
        result = await client.embed(EmbedRequest(input=inputs))
    except DriverError as e:
        return _driver_error_response(e)
    except httpx.HTTPError as e:
        return _error(
            code=502,
            message=(
                f"Every backend serving {body.model!r} failed. Last error: {e}. "
                f"The cascade never reached a different model -- see the contract."
            ),
            error_type="upstream_error",
        )

    # The generated default is the literal string "float", not the enum
    # member -- datamodel-code-generator emits the schema default
    # verbatim -- so an unset field has no `.value`.
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
        messages=[_to_driver_message(m) for m in body.messages],
        maxTokens=max_tokens,
        temperature=temperature,
        stop=body.stop,
        requestId=None,
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
        content=message.content,
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
    return DriverResponseFormat.model_validate(fmt.model_dump(mode="json", exclude_none=True))


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
    # A truncated-by-error generation is reported as `stop` with the text
    # that did arrive, matching what OpenAI does — there is no OpenAI
    # finish reason for "the backend broke mid-stream", and inventing one
    # would break clients that switch on this field.
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
        # The window that applied to *this* request, which is not the
        # smallest across every backend serving the name -- that is what
        # `GET /v1/models` reports, and it is the right number there and
        # the wrong one here.
        context_length=table.context_length_for(model, served_by),
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
            # summarise, and pretending otherwise would invent a usage
            # and a finish reason nobody reported.
            log.warning("driver stream for %r ended without a done event", body.model)
            _record(rec, body, tries)
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
                usage=usage,
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
