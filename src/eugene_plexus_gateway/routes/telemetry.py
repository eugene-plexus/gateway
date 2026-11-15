"""`GET /metrics` and `GET /v1/metrics/export` (row 12).

What a Prometheus scraper reads, and where the numbers go. Both admit an
operator session, or a client key with `readMetrics`: a scraper holds a key that
may read numbers and run nothing (the console makes one with `allowedModels: []`).
No credential is a 401. A key without the grant is a 403 that says what to turn on.

Whether a key may is the key authority's answer, asked with admission purpose
`metrics`: a check that names no model and takes no lease. The answer is
checked here again (`limits.readMetrics`) so a stale or odd authority cannot
widen the grant, and it is kept for `_STANDING_SECONDS`, so a scraper every
fifteen seconds is not a question to the root every fifteen seconds. A key that
is revoked stops at the gateway's own key policy at once (`check_client_key`).

These routes are not front-door routes: no model is named, no queue is joined
and no lease is taken, so a scrape never competes with a completion.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .. import telemetry, tokens
from .._generated.models import MetricsExport, Problem
from ..dependencies import check_client_key

router = APIRouter(tags=["metrics"])

_bearer_scheme = HTTPBearer(auto_error=False)

_STANDING_SECONDS = 10.0
_TEXT = "text/plain; version=0.0.4; charset=utf-8"

NO_GRANT = "This key may not read metrics. Turn on May read metrics for it under Keys."


def _problem(code: int, title: str, detail: str) -> HTTPException:
    return HTTPException(
        status_code=code,
        detail=Problem(
            type=f"https://github.com/eugene-plexus/gateway#{title.replace(' ', '-').lower()}",
            title=title,
            status=code,
            detail=detail,
            component="gateway",
        ).model_dump(exclude_none=True),
    )


@dataclass(frozen=True)
class _Standing:
    name: str
    may_read: bool
    until: float


async def _standing(request: Request, claims: tokens.Claims) -> _Standing:
    cache: dict[str, _Standing] | None = getattr(request.app.state, "metrics_keys", None)
    if cache is None:
        cache = {}
        request.app.state.metrics_keys = cache
    now = time.perf_counter()
    cached = cache.get(claims.jti)
    if cached is not None and cached.until > now:
        return cached
    guard = getattr(request.app.state, "client_key_guard", None)
    if guard is None:
        raise _problem(503, "Client admission unavailable", "Client admission is unavailable.")
    try:
        response = await guard.admission(
            {
                "action": "check",
                "keyId": claims.jti,
                "requestId": secrets.token_hex(8),
                "purpose": "metrics",
            }
        )
    except (httpx.HTTPError, OSError) as exc:
        raise _problem(
            503,
            "Client admission unavailable",
            f"Client admission is unavailable ({type(exc).__name__}). Restore the "
            "agent and control connection; an operator session still reads metrics.",
        ) from exc
    if response.status_code == 403:
        raise _problem(403, "This key may not read metrics", NO_GRANT)
    if response.status_code == 401:
        raise _problem(401, "Key refused", "This client key is unregistered, expired or revoked.")
    if response.status_code >= 400:
        raise _problem(
            503,
            "Client admission unavailable",
            f"Client access needs the admission authority, which answered {response.status_code}.",
        )
    try:
        body = response.json()
        limits = body.get("limits") or {}
        name = str(body.get("keyName") or "")
    except (ValueError, AttributeError) as exc:
        raise _problem(
            503, "Client admission unavailable", "The admission authority's answer was not JSON."
        ) from exc
    standing = _Standing(
        name=name or claims.sub,
        may_read=limits.get("readMetrics") is True,
        until=now + _STANDING_SECONDS,
    )
    cache[claims.jti] = standing
    return standing


async def require_metrics_reader(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)] = None,
) -> str:
    """The operator session, or the key that may read metrics. Returns who, for
    `lastScrapeKeyName`: `operator`, or the key's name."""
    auth = request.app.state.auth_state
    if auth.auth_disabled:
        return "operator"
    if creds is None or not creds.credentials:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED,
            "Missing token",
            "Send an operator session, or a client key that may read metrics, as "
            "Authorization: Bearer <key>.",
        )
    try:
        claims = auth.verify(creds.credentials, classes=(tokens.TYP_SESSION, tokens.TYP_CLIENT))
    except tokens.TokenError as exc:
        raise _problem(
            status.HTTP_401_UNAUTHORIZED, "Invalid token", f"Bearer token rejected: {exc}"
        ) from exc
    if claims.is_session:
        return "operator"
    await check_client_key(request, claims)
    standing = await _standing(request, claims)
    if not standing.may_read:
        raise _problem(403, "This key may not read metrics", NO_GRANT)
    return standing.name


def _telemetry(request: Request) -> telemetry.Telemetry:
    found: telemetry.Telemetry | None = getattr(request.app.state, "telemetry", None)
    if found is None:
        raise _problem(
            503,
            "Metrics export unavailable",
            "This gateway is starting up or in safe mode; it has no numbers to export yet.",
        )
    return found


@router.get("/metrics", operation_id="getPrometheusMetrics", response_class=Response)
async def get_prometheus_metrics(
    request: Request, who: str = Depends(require_metrics_reader)
) -> Response:
    numbers = _telemetry(request)
    store = getattr(request.app.state, "metrics", None)
    numbers.note_scrape(who)
    gauges = telemetry.read_gauges(
        getattr(request.app.state, "routing", None),
        rows_dropped=store.rows_dropped if store is not None else None,
        enabled=store is not None,
    )
    return Response(numbers.render(gauges), media_type=_TEXT)


@router.get("/v1/metrics/export", operation_id="getMetricsExport", response_model=MetricsExport)
async def get_metrics_export(
    request: Request, who: str = Depends(require_metrics_reader)
) -> MetricsExport:
    numbers = _telemetry(request)
    exporter: telemetry.OtlpExporter | None = getattr(request.app.state, "otlp_exporter", None)
    return MetricsExport.model_validate(
        {
            "prometheus": {
                "path": "/metrics",
                "lastScrapeAt": numbers.last_scrape_at,
                "lastScrapeKeyName": numbers.last_scrape_key,
            },
            "otlp": exporter.snapshot() if exporter is not None else {"state": "off"},
        }
    )
