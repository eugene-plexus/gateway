"""A client request owns one shared reservation, including wake and streaming."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import door_paths
from ._generated.client_key_models import ClientAdmissionResult
from .capabilities import TOOL_SETTINGS, calls_tools, chats, said_tools, tools_refused
from .ignored_settings import WIRE_NAMES
from .metrics import AttemptRow, RequestRow


class AdmissionFailure(Exception):
    def __init__(
        self, status: int, message: str, retry: str | None = None, param: str | None = None
    ):
        self.status, self.message, self.retry = status, message, retry
        self.param = param
        super().__init__(message)


current: ContextVar[ClientRequest | None] = ContextVar("client_request", default=None)


class ClientRequest:
    def __init__(self, scope: Scope) -> None:
        self.scope = scope
        self.id = str(uuid.uuid4())
        self.key_id: str | None = None
        self.key_name: str | None = None
        self.model: str | None = None
        self.streamed = False
        self.recorded = False
        self.tries: list[AttemptRow] = []
        self.embedding_result: Any = None
        self.started = time.perf_counter()
        self.at = datetime.now(UTC)
        self.guard: Any = None
        self.access: dict[str, Any] | None = None
        self.attempted = False
        self.ready = asyncio.Event()
        self.deadline = 0.0
        self.lease_seconds = 30.0
        self.expires = float("inf")
        # Renew and release take turns, so a renewal in flight cannot land
        # after the release and be refused (which would cut the reply short).
        self.lease_lock = asyncio.Lock()
        self.released = False

    @property
    def allowed_models(self) -> set[str] | None:
        limits = (self.access or {}).get("limits")
        allowed = limits.get("allowedModels") if limits else None
        return set(allowed) if allowed is not None else None

    @property
    def local_only(self) -> bool:
        return ((self.access or {}).get("limits") or {}).get("localOnly") is True

    @property
    def allowed_tools(self) -> list[str] | None:
        """The key's `allowedTools` (P8): None permits every tool the
        install runs, as on every key minted before P8."""
        tools = ((self.access or {}).get("limits") or {}).get("allowedTools")
        return list(tools) if isinstance(tools, list) else None

    async def contact(self, action: str) -> dict[str, Any]:
        began = time.perf_counter()
        try:
            timeout = min(4.0, self.deadline - began) if action == "renew" else 4.0
            if timeout <= 0:
                raise TimeoutError
            async with asyncio.timeout(timeout):
                response = await self.guard.admission(
                    {
                        "action": action,
                        "keyId": self.key_id,
                        "requestId": self.id,
                        **({"model": self.model} if self.model else {}),
                    }
                )
            if response.status_code >= 400:
                if action == "acquire" and response.status_code < 500:
                    self.attempted = False
                status = response.status_code
                if status not in (401, 403, 409, 429):
                    status = 503
                message = {
                    401: "This client key is unregistered, expired or revoked.",
                    403: "This client key does not permit the requested model.",
                    409: "Request permissions or reservation changed; start a new request.",
                    429: "This key's shared concurrency or request-rate limit is reached. "
                    "Wait or cancel an active request; switching gateways does not add allowance.",
                    503: "Client access needs the admission authority. Check the local agent and "
                    "active control root; operator management remains available.",
                }[status]
                retry = response.headers.get("Retry-After", "5") if status in (429, 503) else None
                if retry is not None and (not retry.isdigit() or len(retry) > 5):
                    retry = "5"
                raise AdmissionFailure(status, message, retry)
            result = ClientAdmissionResult.model_validate(response.json())
            if result.keyId != self.key_id:
                raise ValueError("authority returned another key")
            if action in ("acquire", "renew"):
                if not result.leaseSeconds:
                    raise ValueError("no reservation duration")
                self.lease_seconds = result.leaseSeconds
                self.deadline = began + result.leaseSeconds * 0.75
                if self.deadline <= time.perf_counter():
                    raise TimeoutError
            self.key_name = result.keyName
            return result.model_dump(mode="json")
        except (httpx.HTTPError, ValueError, TimeoutError) as exc:
            raise AdmissionFailure(
                503,
                "Client admission is unavailable. Restore the agent/control "
                "connection; this gateway will not create a separate allowance.",
                "5",
            ) from exc

    async def authorize(self, request: Request, model: str | None, streamed: bool = False) -> None:
        self.model, self.streamed = model, streamed
        if self.key_id is None:
            return
        self.guard = getattr(request.app.state, "client_key_guard", None)
        if self.guard is None:
            raise AdmissionFailure(503, "Client admission is unavailable.", "5")
        self.attempted = model is not None
        self.access = await self.contact("acquire" if model is not None else "check")
        if model is not None:
            self.ready.set()

    async def renew(self) -> None:
        async with self.lease_lock:
            if self.ready.is_set() and not self.released:
                self.access = await self.contact("renew")

    async def monitor(self) -> None:
        while not self.released:
            await asyncio.sleep(
                min(self.lease_seconds / 3, max(0.0, self.deadline - time.perf_counter()))
            )
            await self.renew()
        # Released: nothing left to renew. Wait to be cancelled with the
        # request; returning would leave a finished task in the wait set.
        await asyncio.Event().wait()

    async def release(self) -> None:
        """Release once. A cancelled release stays unreleased, so the
        middleware's `finally` tries again."""
        if not self.attempted or self.guard is None:
            return
        async with self.lease_lock:
            if self.released:
                return
            with contextlib.suppress(Exception):
                async with asyncio.timeout(2):
                    await self.contact("release")
            self.released = True

    def record_missing(self, app: Any, status: int) -> None:
        metrics = getattr(app.state, "metrics", None)
        if self.model is None or self.recorded or metrics is None:
            return
        metrics.record(
            RequestRow(
                started_at=self.at,
                requested_model=self.model,
                total_ms=int((time.perf_counter() - self.started) * 1000),
                attempts=len(self.tries),
                outcome="served" if status < 400 and self.embedding_result is not None else "error",
                streamed=self.streamed,
                tries=self.tries,
                served_model=getattr(self.embedding_result, "modelId", None),
                prompt_tokens=getattr(
                    getattr(self.embedding_result, "usage", None), "promptTokens", None
                ),
                completion_tokens=0
                if getattr(self.embedding_result, "usage", None) is not None
                else None,
                client_key_id=self.key_id,
                client_key_name=self.key_name,
                request_id=self.id,
                elapsed_ms=int((time.perf_counter() - self.started) * 1000),
            )
        )
        self.recorded = True


async def authorize(request: Request, model: str | None = None, *, streamed: bool = False) -> None:
    context = current.get()
    if context is not None:
        await context.authorize(request, model, streamed)


def request_id() -> uuid.UUID | None:
    context = current.get()
    return uuid.UUID(context.id) if context is not None else None


def local_only() -> bool:
    context = current.get()
    return context.local_only if context else False


#: How long one backend's listing is given to answer the re-check in `permitted`.
CONFIRM_TIMEOUT_SECONDS = 4.0


def _who(backend: Any) -> str:
    """The driver a candidate is, as a person can find it: its name, the node when
    it has one (two machines run one name), and where it listens."""
    node = f" on node {backend.node!r}" if backend.node else ""
    return f"driver {backend.name!r}{node} ({backend.url})"


def unread_because(backend: Any, exc: Exception) -> str:
    """What was observed when `backend`'s `/v1/info` could not be read, in words.

    The driver and the cause, never empty: our own deadline as such, and every
    other failure as the routing table words an unreachable driver
    (`routing.unreachable_reason`: nothing listening, a firewall dropping the
    connection, an answer that was not a 2xx). Whitespace is collapsed and the
    length kept, because a malformed listing's validation error is many lines.
    """
    from .routing import unreachable_reason

    if isinstance(exc, TimeoutError):
        cause = f"no answer to GET /v1/info within {CONFIRM_TIMEOUT_SECONDS:g} s"
    else:
        cause = unreachable_reason(exc, url=backend.url, timeout=CONFIRM_TIMEOUT_SECONDS)
    return f"{_who(backend)}: {' '.join(cause.split())[:240]}"


def unconfirmed_refusal(model: str, carried: Sequence[str], observed: Sequence[str]) -> str:
    """The refusal of a request none of whose backends could be checked against its
    listing: a 503 that says what was observed, not a 400 that blames the request.

    **The request is not what failed.** A driver that does not answer (a timeout,
    a restart) was left out without a word, and when it was the only one that could
    have served, the refusal read as the request's own fault: "takes tools, remove
    it", a 400 that a client such as Codex does not retry, for what the next try
    would have got through. One sentence for every way a backend could not be
    confirmed, so the causes are worded in one place.
    """
    return (
        f"Could not confirm that a backend serving {model!r} takes what this request "
        f"carries ({', '.join(carried)}): {'; '.join(observed)}. This is not a refusal of "
        "the request: try again in a few seconds. Nothing was forwarded or woken."
    )


async def permitted(resolution: Any, requirements: Any = None) -> Any:
    """The backends of `resolution` this request may go to, confirmed afresh.

    A request that carries settings or tools is checked against each backend's
    own listing, read again now rather than trusted from the last refresh. A
    backend that cannot take what was sent is left out, and when none can, the
    request is refused saying which setting nobody takes. What a listing owes a
    request (2026-10-12, `docs/design/tools-assumed.md`):

    - **A model that does not chat is not asked about tools.** A request that
      carries tools is a chat request's, and a model that names its surfaces and
      leaves chat out (an embedding, rerank or speech model) is refused by the
      door, by name, as the wrong one (`_wrong_surface`: "serves embeddings and
      not chat"). It is kept here for that when nothing in the slot chats, and
      left out, uncounted, when something does.
    - **`tools` are assumed**: only a driver's `toolCalling: false` turns a chat
      backend away, and its list of settings, which may be none at all, does not
      decide them. When every chat backend turned away was turned away for that,
      the refusal says so, in the driver's words (`tools_refused`).
    - **`toolChoice` and `parallelToolCalls`, sent with tools, are admitted for a
      backend whose driver says nothing of tool calling**: its capabilities are
      not reported at all, or carry no `toolCalling`. Whatever its
      `supportedSettings` holds, because a driver that reports nothing is not a
      no. A backend that does say keeps the rule that its list names the
      settings it takes.
    - **Every other setting keeps the existing rule, for every backend**: the
      listing must name it, and a backend whose capabilities are not reported
      takes none. Unknown is yes for tools and the settings that ride with them,
      and nowhere else.

    A backend turned away for its tools does not count toward what "some backend
    takes", so a refusal for another setting names that setting, not one a
    backend that cannot serve the request anyway happens to list.

    **A backend that could not be confirmed is not a backend that refused.** Its
    listing may not be readable (a timeout, a driver restarting), or may answer
    differently from the table (the model is gone from it, or it follows another
    runtime): it is left out like any other, but when nothing is left, **no
    refusal blames the request for what the listing did not say** ("takes tools,
    remove it" is a 400, which a client such as Codex does not retry, for a
    state the next try may not meet). It is a retryable 503 that names each
    driver and what was observed (`unconfirmed_refusal`). Not for a key that
    requires local-only inference, whose refusal is the policy's and stays 403.
    """
    context = current.get()
    allowed = context.allowed_models if context else None
    resolution = resolution.restricted(allowed)
    settings = set(requirements.callerSettings or []) if requirements is not None else set()
    tools = bool(requirements.tools) if requirements is not None else False
    if not (local_only() or settings or tools) or not resolution.has_backends():
        return resolution

    # No prompt accompanies this probe. Reconfirm before wake/selection, then
    # carry localOnly to the driver to close the probe-to-execution race.
    semaphore = asyncio.Semaphore(8)
    # Every setting some backend that can serve this request confirms, so a
    # refusal can name the one nobody does rather than "the required settings".
    confirmed_anywhere: set[str] = set()
    # Backends turned away because their driver says tools cannot be carried to
    # them: when that is every chat backend turned away, the refusal says so
    # (2026-10-12), in the driver's words and not as the model failing.
    told_no_tools = 0
    # Backends left out for not being chat models, which says nothing of tools:
    # they are not counted among those a tools refusal is about.
    set_aside = 0
    something_chats = any(chats(b.model) for b in resolution.backends())
    # Drivers whose listing could not confirm the request, by `(node, name)`: what
    # was observed, in words. Such a backend is left out, but its silence is not
    # its answer, so when nothing is left the refusal says so (`unconfirmed_refusal`).
    unconfirmed: dict[tuple[str | None, str], str] = {}

    async def confirm(backend: Any) -> Any:
        nonlocal told_no_tools, set_aside
        try:
            model = backend.model.id if backend.model is not None else None
            driver = (backend.node, backend.name)
            try:
                async with semaphore, asyncio.timeout(CONFIRM_TIMEOUT_SECONDS):
                    # This candidate's entry alone (P1): an account would
                    # otherwise re-read every model it lists, on every request
                    # that carries a setting.
                    info = await backend.client.info(model=model)
            except Exception as exc:
                unconfirmed.setdefault(driver, unread_because(backend, exc))
                return None
            if info.runtime != backend.info.runtime:
                # Cached runtime facts must not wake a replacement.
                now = f"runtime {info.runtime!r}" if info.runtime else "no runtime"
                was = f"runtime {backend.info.runtime!r}" if backend.info.runtime else "no runtime"
                unconfirmed.setdefault(
                    driver,
                    f"{_who(backend)}: it now follows {now}, not {was} as at the last refresh",
                )
                return None
            entry = next((m for m in info.models or [] if m.id == model), None)
            if entry is None:
                # The model left the driver since the last refresh.
                unconfirmed.setdefault(
                    driver,
                    f"{_who(backend)}: it answered GET /v1/info without {model!r}, which it "
                    "listed at the last refresh",
                )
                return None
            confirmed = replace(backend, info=info.model_copy(update={"models": None}), model=entry)
            if tools and not chats(entry):
                if something_chats:
                    set_aside += 1
                    return None
                return confirmed  # the door refuses it by name, and that is the answer
            caps = entry.capabilities
            listed = set(caps.supportedSettings or []) if caps is not None else set()
            # Tools are assumed and turned off only on evidence (2026-10-12): a
            # backend whose driver says nothing of tool calling takes them, so
            # its list of settings, which may be none at all, does not decide
            # `tools`. Only `toolCalling: false` does.
            carries = calls_tools(entry)
            if tools and not carries:
                told_no_tools += 1
                return None
            wanted = settings - {"tools"} if carries else set(settings)
            taken = set(listed)
            if tools:
                taken.add("tools")
                if said_tools(entry) is None:
                    # Nothing reported: the settings that come with tools are
                    # not refused for a list the driver may not have written.
                    wanted -= TOOL_SETTINGS
                    taken |= settings & TOOL_SETTINGS
            confirmed_anywhere.update(taken)
            if wanted and not wanted <= listed:
                return None
            return confirmed
        except Exception:
            return None

    tiers = []
    turned_away = 0
    for tier in resolution.tiers:
        checked = await asyncio.gather(*(confirm(b) for b in tier.backends))
        turned_away += sum(1 for b in checked if b is None)
        tiers.append(replace(tier, backends=[b for b in checked if b is not None]))
    result = replace(resolution, tiers=tiers).restricted(allowed, local_only=local_only())
    if not result.has_backends():
        if not local_only():
            if unconfirmed:
                carried = (["tools"] if tools else []) + sorted(
                    WIRE_NAMES.get(m, m) for m in settings - {"tools"}
                )
                raise AdmissionFailure(
                    503,
                    unconfirmed_refusal(resolution.model, carried, list(unconfirmed.values())),
                    "5",
                )
            chat_turned_away = turned_away - set_aside
            if tools and told_no_tools and told_no_tools == chat_turned_away:
                raise AdmissionFailure(
                    400,
                    tools_refused(resolution.model, told_no_tools),
                    param="tools",
                )
            missing = sorted(settings - confirmed_anywhere)
            if missing:
                # Named in the caller's words (2026-09-28): "the required
                # settings" sent a caller to guess which of theirs it was.
                wire = [WIRE_NAMES.get(m, m) for m in missing]
                raise AdmissionFailure(
                    400,
                    f"No permitted backend serving this model takes {', '.join(wire)}. Remove "
                    f"{'it' if len(wire) == 1 else 'them'} or choose a model whose backend does. "
                    "Nothing was forwarded or woken.",
                    param=wire[0],
                )
            raise AdmissionFailure(
                400,
                "No permitted backend confirms the required tools or explicit settings "
                "together (tool_calling / supportedSettings). Nothing was forwarded or woken.",
                param="tools" if tools else None,
            )
        raise AdmissionFailure(
            403,
            "This key requires local-only inference, but no permitted backend "
            "confirms local execution and policy enforcement. Nothing was forwarded or woken.",
        )
    return result


async def before_attempt() -> None:
    context = current.get()
    if context is not None:
        if time.perf_counter() >= context.expires:
            raise AdmissionFailure(
                504, "The total request deadline expired. No further backend was attempted."
            )
        await context.renew()
        if time.perf_counter() >= context.expires:
            raise AdmissionFailure(
                504, "The total request deadline expired. No further backend was attempted."
            )


#: Every path a client key is accepted on. **A door added without a row
#: here is a door with no client admission at all** — no scopes, no
#: local-only, no rate or concurrency limits, no usage attribution — and
#: that is precisely how /v1/systemone shipped for a few hours: the B2
#: acceptance run's scoped key served a model outside its allowedModels
#: because this middleware never saw the path, so `current` was never
#: set and every check downstream read "no client, nothing to limit".
CLIENT_ADMISSION_PATHS = frozenset(
    {
        "/v1/models",
        "/v1/chat/completions",
        "/v1/messages",
        # A count verifies the key and applies its scope, and is a check
        # rather than an admission -- see the route.
        "/v1/messages/count_tokens",
        "/v1/embeddings",
        # MR6: with its row from the first commit, as the rule above asks.
        "/v1/rerank",
        "/v1/systemone",
        # Slice 4, 2026-09-23: Codex CLI's only door.
        "/v1/responses",
        # P3a, 2026-09-28. Shipped without this row for one commit, and a
        # scoped key could speak through any model it was not allowed:
        # the rule above, broken exactly as it describes.
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/images/variations",
        # P5: the first doors whose path carries a value, matched by
        # `door_paths.matches` against the route's template.
        "/v1/videos",
        "/v1/videos/{video_id}",
        "/v1/videos/{video_id}/content",
        # P6. The model id takes the rest of the path (`door_paths`).
        "/v1/models/{model:path}",
        "/v1/moderations",
        "/v1/completions",
    }
)


class ClientAdmissionMiddleware:
    """Own the whole ASGI lifetime; headers do not end a streamed reservation."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not door_paths.matches(scope["path"], CLIENT_ADMISSION_PATHS):
            await self.app(scope, receive, send)
            return
        context = ClientRequest(scope)
        token = current.set(context)
        queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=1)
        disconnected = asyncio.Event()
        response_started = response_complete = False
        response_status = 499

        async def pump() -> None:
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    disconnected.set()
                    await queue.put(message)
                    return
                await queue.put(message)

        async def incoming() -> Message:
            if disconnected.is_set() and queue.empty():
                return {"type": "http.disconnect"}
            return await queue.get()

        async def outgoing(message: Message) -> None:
            nonlocal response_started, response_complete, response_status
            if message["type"] == "http.response.start":
                response_started = True
                response_status = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", context.id.encode()))
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                # The answer is decided: free the key's slot before the client
                # can read it, or a next request sent at once (on any gateway)
                # can be refused 429 for a slot still held (gateway #9).
                await context.release()
                response_complete = True
            await send(message)

        async def refuse(exc: AdmissionFailure) -> None:
            anthropic = scope["path"] in ("/v1/messages", "/v1/messages/count_tokens")
            status = 403 if anthropic and exc.status == 401 else exc.status
            kind = (
                "authentication_error"
                if exc.status == 401
                else "rate_limit_error"
                if exc.status == 429
                else "permission_error"
                if exc.status == 403
                else "invalid_request_error"
                if exc.status == 400
                else "api_error"
            )
            error: dict[str, Any] = {"error": {"type": kind, "message": exc.message}}
            if exc.param is not None:
                error["error"]["param"] = exc.param
            if anthropic:
                error["type"] = "error"
            if not response_started:
                await JSONResponse(
                    error,
                    status_code=status,
                    headers={"Retry-After": exc.retry} if exc.retry else None,
                )(scope, incoming, outgoing)
            elif not response_complete:
                if scope["path"] == "/v1/responses":
                    # `response.failed`, never a bare `error` event: Codex
                    # reports the latter without our message, and the code
                    # decides whether Codex tries again (record §4).
                    from . import responses

                    frame = responses.failed_frame(
                        responses.new_id(uuid.UUID(context.id)),
                        exc.message,
                        code=responses.middleware_code(exc.status),
                    )
                else:
                    prefix = "event: error\n" if anthropic else ""
                    frame = prefix + "data: " + json.dumps(error) + "\n\n"
                await outgoing(
                    {"type": "http.response.body", "body": frame.encode(), "more_body": False}
                )

        store = getattr(scope["app"].state, "config_store", None)
        budget = float(store.get("requestTimeoutSeconds") or 600) if store is not None else 600.0
        context.expires = context.started + budget

        async def expire() -> None:
            await asyncio.sleep(max(0, context.expires - time.perf_counter()))
            raise AdmissionFailure(
                504,
                "The total request deadline expired, including preparation, loading and "
                "inference. Owned work was cancelled; a remote provider may still have acted. "
                "No automatic replay.",
            )

        expiry = asyncio.create_task(expire())
        tasks: list[asyncio.Task[Any]] = [asyncio.create_task(pump()), expiry]

        async def serve() -> None:
            from .routing import collect_attempts

            with collect_attempts() as tries:
                context.tries = tries
                await self.app(scope, incoming, outgoing)

        app_task = asyncio.create_task(serve())
        ready = asyncio.create_task(context.ready.wait())
        gone = asyncio.create_task(disconnected.wait())
        tasks.extend([app_task, ready, gone])
        monitor: asyncio.Task[None] | None = None
        try:
            while True:
                watching = {app_task, gone, expiry, monitor if monitor else ready}
                done, _ = await asyncio.wait(watching, return_when=asyncio.FIRST_COMPLETED)
                if app_task in done:
                    await app_task
                    break
                if expiry in done:
                    await expiry
                if gone in done:
                    if response_complete:
                        await app_task
                    break
                if monitor is not None and monitor in done:
                    await monitor  # Raises the authority failure; stop upstream before replying.
                if ready in done and monitor is None:
                    monitor = asyncio.create_task(context.monitor())
                    tasks.append(monitor)
        except AdmissionFailure as exc:
            response_status = exc.status
            app_task.cancel()
            await asyncio.gather(app_task, return_exceptions=True)
            if not disconnected.is_set():
                await refuse(exc)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await context.release()
            finally:
                with contextlib.suppress(Exception):
                    context.record_missing(scope["app"], response_status)
                current.reset(token)
