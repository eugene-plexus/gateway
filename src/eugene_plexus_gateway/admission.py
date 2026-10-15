"""A client request owns one shared reservation, including wake and streaming."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import door_paths
from ._generated.client_key_models import ClientAdmissionResult
from .capabilities import TOOL_SETTINGS, calls_tools, chats, said_tools, tools_refused
from .driver_client import credential_refused_next_step, each_named
from .ignored_settings import carried as carried_words
from .ignored_settings import door_of, worded
from .metrics import AttemptRow, RequestRow
from .outbound import token_refused_words


class AdmissionFailure(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        retry: str | None = None,
        param: str | None = None,
        kind: str | None = None,
    ):
        self.status, self.message, self.retry = status, message, retry
        self.param = param
        #: The error `type` the caller reads, when it is not the one the status says
        #: (a driver's refusal of the gateway's credential is `upstream_auth_error`, as
        #: the front door's own 502 for it is). Not on the Anthropic doors, whose error
        #: types are a closed set.
        self.kind = kind
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
                # What the authority answered, said as it was: every status outside the four
                # it refuses a key with is a 503 here, and a person told only "needs the
                # admission authority" could not tell a 500 from a 404 from a 504.
                answered = response.status_code
                status = answered if answered in (401, 403, 409, 429) else 503
                message = {
                    401: "This client key is unregistered, expired or revoked.",
                    403: "This client key does not permit the requested model.",
                    409: "Request permissions or reservation changed; start a new request.",
                    429: "This key's shared concurrency or request-rate limit is reached. "
                    "Wait or cancel an active request; switching gateways does not add allowance.",
                    503: f"Client access needs the admission authority, which answered {answered}. "
                    "Check the local agent and active control root; operator management "
                    "remains available.",
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
                f"Client admission is unavailable ({_observed(exc)}). Restore the agent/control "
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

#: How many characters of one driver's cause a refusal keeps.
MAX_CAUSE_CHARS = 240

#: What to look at for a cause waiting cannot clear: the drivers it is for, each as
#: `_who` names it, answered with one sentence that says which is which. **A remedy names
#: its drivers**: "the address above" and "that driver" said nothing of which, when a
#: refusal names several.
Remedy = Callable[[Sequence[str]], str]


def not_an_inference_driver(drivers: Sequence[str]) -> str:
    """What to look at when a driver does not answer GET /v1/info as an inference driver
    does (a 404, any other 4xx, a body that is not a listing). Observed, not guessed:
    the address is where the agent said the driver is."""
    which = each_named(drivers) if len(drivers) == 1 else f"each of {each_named(drivers)}"
    return (
        f"Make sure {which} is an inference driver of this install, at a version this "
        "gateway reads."
    )


def serves_nothing(drivers: Sequence[str]) -> str:
    """What to look at when a driver could not build its engine from its configuration and
    says so (`DriverInfo.degraded`): it stays that way until the configuration is fixed and
    it is restarted, so the health and the configuration are what to read."""
    which = "it serves" if len(drivers) == 1 else "each serves"
    return (
        f"Look at the health and configuration of {each_named(drivers)}: {which} nothing "
        "until its configuration is fixed and it is restarted."
    )


def as_sentence(text: str) -> str:
    """`text` ended with one full stop: none is added after one it already ends with,
    which a driver's own detail often does (and a cut one ends with an ellipsis), so a
    refusal never reads "..".
    """
    flat = text.rstrip()
    return flat if flat.endswith(".") else f"{flat}."


def _without_full_stop(text: str) -> str:
    """`text` without the one full stop that ends it, for joining it to another sentence
    part. An ellipsis is not one: it says the cause was cut."""
    return text[:-1] if text.endswith(".") and not text.endswith("...") else text


#: Statuses of a 4xx that are the driver asking to be asked again.
_RETRYABLE_4XX = frozenset({408, 429})

#: Transport failures that a restart or a dropped connection leaves: a deadline, a
#: refused or reset connection, a driver that hung up. A proxy or a protocol that
#: cannot be spoken is the install's, and a retry does not change it.
_TRANSIENT_TRANSPORT = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


@dataclass(frozen=True)
class Unconfirmed:
    """One driver whose listing could not confirm a request, and what was observed.

    `text` names the driver and the cause in plain words. `transient` is whether
    waiting could change it: a deadline, a connection that was refused or dropped,
    a 5xx, a driver whose engine has not loaded yet, a request that went out with no
    token because this node's agent would not give one (it is asked again within
    seconds). A driver that refuses the gateway's own credential, does not answer GET
    /v1/info as an inference driver does, answers it with something that is not a
    listing or says it could not build its engine is not changed by a retry, and
    `remedy` says what to look at, for `driver` (`_who`): **only such a cause has one**.
    `kind` is the error type a refusal made only of such causes carries.
    """

    text: str
    transient: bool = True
    remedy: Remedy | None = None
    kind: str = "upstream_error"
    driver: str = ""


def _who(backend: Any) -> str:
    """The driver a candidate is, as a person can find it: its name, the node when
    it has one (two machines run one name), and where it listens."""
    node = f" on node {backend.node!r}" if backend.node else ""
    return f"driver {backend.name!r}{node} ({backend.url})"


def _plain(text: str, limit: int = MAX_CAUSE_CHARS) -> str:
    """`text` on one line and at most `limit` characters, cut at a word boundary
    and ended with an ellipsis when it was cut.

    A malformed listing's validation error is many lines, and a long answer from a
    driver would otherwise bury the rest of the refusal. A cut never splits a
    word (unless one word is the whole of it).
    """
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    room = limit - len("...")
    cut = flat[:room]
    if flat[room] != " ":
        # The cut falls inside a word: keep the words before it.
        cut = cut.rpartition(" ")[0] or cut
    return cut.rstrip() + "..."


def _observed(exc: Exception) -> str:
    """What `exc` was, in words, never empty: its class, and its own words when it has any,
    on one line and capped (`_plain`).

    The class is always said, because an exception's own words may be none (an httpx timeout's
    text is empty, as is a bare `TimeoutError`) and may be many lines (a malformed answer's
    validation error), and a failure that names no cause leaves a person to guess it. It is
    said inside a parenthesis, so without the full stop its own words may end with (a cut
    one keeps its ellipsis, `_without_full_stop`): "response.)." reads as a mistake.
    """
    said = " ".join(str(exc).split())
    words = f"{type(exc).__name__}: {said}" if said else type(exc).__name__
    return _without_full_stop(_plain(words))


def _changed(backend: Any, what: str) -> Unconfirmed:
    """A driver that answered, but not as it did at the last refresh: a state a
    retry may find changed again (its engine was loading, it was reconfigured)."""
    who = _who(backend)
    return Unconfirmed(f"{who}: {_plain(what)}", driver=who)


def _serves_nothing(backend: Any, why: str) -> Unconfirmed:
    """A driver that could not build its engine from its configuration (it reports
    `DriverInfo.degraded`, in its own words) serves nothing, and the same on the next try:
    only a fixed configuration and a restart change it."""
    who = _who(backend)
    cause = f"it says it serves nothing: {why}"
    return Unconfirmed(
        f"{who}: {_plain(cause)}", transient=False, remedy=serves_nothing, driver=who
    )


def _no_token_because(backend: Any) -> str | None:
    """Why the gateway sent this driver no credential, when it sent none, in the words
    the routing table says it (`outbound.token_refused_words`): this node's agent would
    not mint a token for the driver's machine. None when a credential went out, and for a
    client that has no way to say (a fake, an unauthenticated install)."""
    ask = getattr(getattr(backend, "client", None), "credential_refusal", None)
    refusal = ask() if callable(ask) else None
    return token_refused_words(refusal) if refusal else None


def unread_because(backend: Any, exc: Exception) -> Unconfirmed:
    """What was observed when `backend`'s `/v1/info` could not be read, in words,
    and whether a retry could change it.

    The driver and the cause, never empty: our own deadline as such, and every
    other failure as the routing table words an unreachable driver
    (`routing.unreachable_reason`: nothing listening, a firewall dropping the
    connection, an answer that was not a 2xx), classed by what it is.

    - **Transient**: a deadline, a connection refused, reset or timed out, a 5xx,
      a 408 or 429, and a 401 or 403 to a request that went out with no credential
      because this node's agent would not give the gateway a token for that machine
      (`_no_token_because`): the agent is asked again within seconds, and the answer
      says why it would not.
    - **A driver refusing the gateway's credential (401, 403)**, that was sent one: the
      install's fault and not the request's, and a retry does not mend it.
    - **A driver that is not answering as one** (a 404 or any other 4xx, a body that
      is not JSON or not a listing): the address is not an inference driver this
      gateway can read.
    - **Anything else** is not claimed to be transient: said as what it is.

    Whitespace is collapsed and the length kept (`_plain`), because a malformed
    listing's validation error is many lines.
    """
    from .routing import unreachable_reason

    who = _who(backend)
    if isinstance(exc, TimeoutError):
        return Unconfirmed(
            f"{who}: no answer to GET /v1/info within {CONFIRM_TIMEOUT_SECONDS:g} s", driver=who
        )
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        answered = unreachable_reason(exc, url=backend.url, timeout=CONFIRM_TIMEOUT_SECONDS)
        if status in (401, 403):
            if (unsent := _no_token_because(backend)) is not None:
                # What the driver answered and why nothing was sent, each capped on its
                # own: a long answer must not cut off the reason, which is what to act on.
                return Unconfirmed(f"{who}: {_plain(answered)} ({_plain(unsent)})", driver=who)
            cause = f"refused the gateway's credential ({answered})"
            return Unconfirmed(
                f"{who}: {_plain(cause)}",
                transient=False,
                remedy=credential_refused_next_step,
                kind="upstream_auth_error",
                driver=who,
            )
        if status >= 500 or status in _RETRYABLE_4XX:
            return Unconfirmed(f"{who}: {_plain(answered)}", driver=who)
        cause = f"does not serve GET /v1/info ({answered})"
        return Unconfirmed(
            f"{who}: {_plain(cause)}", transient=False, remedy=not_an_inference_driver, driver=who
        )
    if isinstance(exc, ValueError):
        # An answer that was not JSON, or JSON that was not a driver's listing.
        cause = f"answered GET /v1/info with something that is not a driver listing ({exc})"
        return Unconfirmed(
            f"{who}: {_plain(cause)}", transient=False, remedy=not_an_inference_driver, driver=who
        )
    cause = unreachable_reason(exc, url=backend.url, timeout=CONFIRM_TIMEOUT_SECONDS)
    return Unconfirmed(
        f"{who}: {_plain(cause)}", transient=isinstance(exc, _TRANSIENT_TRANSPORT), driver=who
    )


def unconfirmed_refusal(model: str, carried: Sequence[str], observed: Sequence[Unconfirmed]) -> str:
    """The refusal of a request none of whose backends could be checked against its
    listing, saying what was observed and not blaming the request.

    **The request is not what failed.** A driver that does not answer (a timeout,
    a restart) was left out without a word, and when it was the only one that could
    have served, the refusal read as the request's own fault: "takes tools, remove
    it", a 400 that a client such as Codex does not retry, for what the next try
    would have got through. One sentence for every way a backend could not be
    confirmed, so the causes are worded in one place.

    "Try again in a few seconds" is said only when some cause can pass: for a
    driver that refuses the gateway's credential, is not an inference driver or
    could not build its engine, waiting changes nothing and the sentence says what to
    look at instead: **once for each kind of cause, naming every driver it is for**.
    """
    retry = ": try again in a few seconds." if any(o.transient for o in observed) else "."
    fixes: dict[Remedy, list[str]] = {}
    for o in observed:
        if o.remedy is not None:
            fixes.setdefault(o.remedy, []).append(o.driver)
    return (
        f"Could not confirm that a backend serving {model!r} takes what this request "
        f"carries ({', '.join(carried)}): {as_sentence(_causes(observed))} This is not a "
        f"refusal of the request{retry}"
        + "".join(f" {fix(drivers)}" for fix, drivers in fixes.items())
        + " Nothing was forwarded or woken."
    )


def _causes(observed: Sequence[Unconfirmed]) -> str:
    """What was observed of each driver, in one run: each without the full stop its own
    words may end with, so that joining and ending them never says two."""
    return "; ".join(_without_full_stop(o.text) for o in observed)


def unconfirmed_failure(
    model: str, carried: Sequence[str], observed: Sequence[Unconfirmed]
) -> AdmissionFailure:
    """The failure for `unconfirmed_refusal`: a 503 with a `Retry-After` when waiting
    could clear any of the causes, and otherwise a 502 with no retry advice and no
    `Retry-After`, the treatment the front door gives a driver that refuses its
    credential (`_driver_failure_status`): the request was fine and the install is not.
    """
    message = unconfirmed_refusal(model, carried, observed)
    if any(o.transient for o in observed):
        return AdmissionFailure(503, message, "5")
    refused = any(o.kind == "upstream_auth_error" for o in observed)
    return AdmissionFailure(
        502, message, kind="upstream_auth_error" if refused else "upstream_error"
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
      and nowhere else. **That holds for a model whose tool calling is off, with or
      without tools in the request**: a setting is admitted where the driver lists
      it. A current driver lists `tool_choice` and `parallel_tool_calls` for every chat
      model whose tools are off (the Claude Code and Codex CLI subscriptions, OpenAI's
      Responses-only models, a listing that names no tools, a learned refusal) and drops
      them from a request that carries no tools, so such a model is admitted them by the
      same rule and serves the request itself. A driver that does not list the two (an
      older one, which refused them with a 400 that does not cascade, so a slot
      [subscription model, local model] would stop at the first) is turned away on its
      listing, and the next target serves.

    A backend turned away for its tools does not count toward what "some backend
    takes", so a refusal for another setting names that setting, not one a
    backend that cannot serve the request anyway happens to list.

    **A backend that could not be confirmed is not a backend that refused.** Its
    listing may not be readable (a timeout, a driver restarting, a driver that
    refuses the gateway's credential or is not an inference driver, or one that
    says it could not build its engine), or may answer differently from the table
    (the model is gone from it, or it follows another runtime): it is left out
    like any other, but when nothing is left, **no refusal blames the request for
    what the listing did not say** ("takes tools, remove it" is a 400, which a
    client such as Codex does not retry, for a state the next try may not meet).
    It names each driver and what was observed (`unconfirmed_refusal`), as a
    retryable 503 when waiting could clear any of the causes, and as a 502 with no
    retry advice when none can (`unconfirmed_failure`): a driver that refuses a
    credential it was sent, is not an inference driver, or reports `degraded` is
    the same on the next try, where a listing merely missing the model, or a 401
    to a request the gateway sent no token because its agent would not give one,
    is not. For a key that requires local-only inference the refusal is the policy's
    and stays a 403, with the causes appended. A setting is named in the words of
    the door the request came in by (`ignored_settings.door_of`).
    """
    context = current.get()
    allowed = context.allowed_models if context else None
    resolution = resolution.restricted(allowed)
    settings = set(requirements.callerSettings or []) if requirements is not None else set()
    # Effort is a preference: a backend that cannot carry it can still answer.
    settings.discard("reasoningEffort")
    tools = bool(requirements.tools) if requirements is not None else False
    if not (local_only() or settings or tools) or not resolution.has_backends():
        return resolution

    # The words a refusal names a setting in are those of the door the request came
    # in by: a Messages caller wrote `disable_parallel_tool_use`, not `parallel_tool_calls`.
    door = door_of(context.scope.get("path", "")) if context is not None else "chat"

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
    unconfirmed: dict[tuple[str | None, str], Unconfirmed] = {}

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
            if info.degraded:
                # A driver that could not build its engine from its configuration says
                # so, in its own words, and serves nothing: whatever else its listing
                # says (an empty one, a model it still lists, another runtime) is moot,
                # and waiting does not build it.
                unconfirmed.setdefault(driver, _serves_nothing(backend, info.degraded))
                return None
            if info.runtime != backend.info.runtime:
                # Cached runtime facts must not wake a replacement.
                now = f"runtime {info.runtime!r}" if info.runtime else "no runtime"
                was = f"runtime {backend.info.runtime!r}" if backend.info.runtime else "no runtime"
                unconfirmed.setdefault(
                    driver,
                    _changed(backend, f"it now follows {now}, not {was} as at the last refresh"),
                )
                return None
            entry = next((m for m in info.models or [] if m.id == model), None)
            if entry is None:
                # The model left the driver since the last refresh.
                gone = (
                    f"it answered GET /v1/info without {model!r}, which it listed at the "
                    "last refresh"
                )
                unconfirmed.setdefault(driver, _changed(backend, gone))
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
                # What the caller sent: every setting it named, `tools` too when it
                # sent the key (an empty list is still something it said).
                raise unconfirmed_failure(
                    resolution.model,
                    carried_words(settings | ({"tools"} if tools else set()), door),
                    list(unconfirmed.values()),
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
                wire = [worded(m, door) for m in missing]
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
        # The policy's refusal, so its status; a driver that could not be read is
        # not why a key may not use it, but it is why nothing could confirm local
        # execution, so what was observed is said after it.
        unread = (
            f" Could not confirm: {as_sentence(_causes(list(unconfirmed.values())))}"
            if unconfirmed
            else ""
        )
        raise AdmissionFailure(
            403,
            "This key requires local-only inference, but no permitted backend "
            f"confirms local execution and policy enforcement.{unread} "
            "Nothing was forwarded or woken.",
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
        # Its count: a check and not an admission, as `count_tokens` is -- see the
        # route. It went without a row for a while, so a key's scope, its local-only
        # rule and its attribution applied to nothing here, and a refusal admission
        # raised was an unhandled exception (a 500).
        "/v1/responses/input_tokens",
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
            if exc.kind is not None and not anthropic:
                kind = exc.kind
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
