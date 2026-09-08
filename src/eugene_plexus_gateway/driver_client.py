"""HTTP clients for talking to inference-driver instances.

Thin wrappers around httpx.AsyncClient: one client per driver, cached and
reused by the routing table. Each carries the driver's topology `name` so
a response can say which backend answered it.

`FailoverDriverClient` composes several into a priority list. That is
where failover lives — and because the routing table groups drivers by
the model they serve, a slot's members come from the topology rather than
from configuration.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from ._generated.driver_models import (
    DriverInfo,
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
    async def aclose(self) -> None: ...


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
        # Mirrors FailoverDriverClient's surface so the route can read
        # these off either without asking which kind it holds.
        self.attempts = 1
        self.served_by: str | None = name
        # When the watchdog threaded a service token in, attach it to
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


class FailoverDriverClient:
    """A driver *slot* backed by an ordered priority list of backends.

    Implements the same `DriverClient` protocol as
    `HttpDriverClient`, so the caller is oblivious to failover — it
    holds a client and calls `.generate()`. Internally this tries its
    candidates in order, cascading on a cascade-eligible failure
    (transport / 5xx / timeout) and failing hard on a 4xx. See
    `_is_cascade_eligible`.

    Granularity is per-call: each `.generate()` independently walks the
    list from the top, so a slot that fell over to its backup recovers to
    the primary as soon as the primary is healthy again — no operator
    intervention, and no sticky state to reset.
    """

    def __init__(self, *, name: str, candidates: list[DriverClient]) -> None:
        if not candidates:
            raise ValueError(f"driver slot {name!r} needs at least one backend URL")
        self.name = name
        self._candidates = candidates
        # `base_url` is the primary backend — used for labelling / logs.
        # The active backend on a given turn may differ after failover,
        # but the slot's identity is its primary.
        self.base_url = candidates[0].base_url

        # What the last call actually did, for the response's routing
        # extension. Per-instance mutable state, which is safe ONLY
        # because an instance is per-request: `RoutingTable.resolve`
        # builds a fresh wrapper around the shared, long-lived
        # HttpDriverClients on every call. Do not cache one of these.
        self.attempts = 0
        self.served_by: str | None = None

    async def info(self) -> DriverInfo:
        """Report the first reachable backend's `/v1/info`.

        Mirrors generate()'s failover so the admin drivers listing
        reflects what a real chat turn would actually reach.
        """
        last_exc: Exception | None = None
        for index, candidate in enumerate(self._candidates):
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
        for index, candidate in enumerate(self._candidates):
            self.attempts = index + 1
            try:
                result = await candidate.generate(request)
            except Exception as exc:
                if not _is_cascade_eligible(exc):
                    # 4xx / non-HTTP error — surface it without trying the
                    # next backend. A 4xx is the same bad request everywhere.
                    raise
                last_exc = exc
                self._log_cascade("generate", index, candidate, exc)
            else:
                self.served_by = getattr(candidate, "name", None)
                return result
        # Every backend failed in a cascade-eligible way. Re-raise the
        # last failure so the chat route's existing DriverError
        # / httpx.HTTPError handlers surface it as they would for a
        # single-backend slot — no new error path to maintain.
        assert last_exc is not None  # candidates is non-empty (checked in __init__)
        raise last_exc

    def _log_cascade(self, op: str, index: int, candidate: DriverClient, exc: Exception) -> None:
        """Emit a WARNING when a backend fails and we cascade.

        Failover that silently always-works hides a broken primary
        (v0.3-plan risk). This WARNING is the "failover happened"
        surface operators grep for; the UI failover badge reads the
        same signal in a later release.
        """
        is_last = index == len(self._candidates) - 1
        next_action = (
            "no more backends in slot — failing"
            if is_last
            else f"trying backend {index + 2}/{len(self._candidates)}"
        )
        log.warning(
            "failover[%s]: slot %r backend %d/%d (%s) failed (%s); %s",
            op,
            self.name,
            index + 1,
            len(self._candidates),
            candidate.base_url,
            exc,
            next_action,
        )

    async def aclose(self) -> None:
        for candidate in self._candidates:
            await candidate.aclose()
