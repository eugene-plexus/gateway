"""A client request owns one shared reservation, including wake and streaming."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from contextvars import ContextVar
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ._generated.client_key_models import ClientAdmissionResult
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

    @property
    def allowed_models(self) -> set[str] | None:
        limits = (self.access or {}).get("limits")
        allowed = limits.get("allowedModels") if limits else None
        return set(allowed) if allowed is not None else None

    @property
    def local_only(self) -> bool:
        return ((self.access or {}).get("limits") or {}).get("localOnly") is True

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
        if self.ready.is_set():
            self.access = await self.contact("renew")

    async def monitor(self) -> None:
        while True:
            await asyncio.sleep(
                min(self.lease_seconds / 3, max(0.0, self.deadline - time.perf_counter()))
            )
            await self.renew()

    async def release(self) -> None:
        if self.attempted and self.guard is not None:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(2):
                    await self.contact("release")

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


async def permitted(resolution: Any, requirements: Any = None) -> Any:
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

    async def confirm(backend: Any) -> Any:
        try:
            async with semaphore, asyncio.timeout(4):
                info = await backend.client.info()
            if info.runtime != backend.info.runtime:
                return None  # cached runtime facts must not wake a replacement
            caps = info.capabilities
            if settings and (
                caps is None or not settings.issubset(set(caps.supportedSettings or []))
            ):
                return None
            if tools and (caps is None or caps.toolCalling is not True):
                return None
            return replace(backend, info=info)
        except Exception:
            return None

    tiers = []
    for tier in resolution.tiers:
        checked = await asyncio.gather(*(confirm(b) for b in tier.backends))
        tiers.append(replace(tier, backends=[b for b in checked if b is not None]))
    result = replace(resolution, tiers=tiers).restricted(allowed, local_only=local_only())
    if not result.has_backends():
        if not local_only():
            raise AdmissionFailure(
                400,
                "No permitted backend confirms the required tools or explicit settings "
                "(tool_calling / supportedSettings). Nothing was forwarded or woken.",
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
        "/v1/systemone",
    }
)


class ClientAdmissionMiddleware:
    """Own the whole ASGI lifetime; headers do not end a streamed reservation."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] not in CLIENT_ADMISSION_PATHS:
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
