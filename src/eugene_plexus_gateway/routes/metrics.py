"""Retained request metrics.

Two reads over what `metrics.MetricsStore` kept: aggregates for "which
backend is faster for this model on this box", and the raw rows behind
them for "what happened at 02:14".

Operator-only, mounted that way in `app.py`. Elsewhere a read accepts a
`service:*` token because a peer component legitimately needs it; no
component needs this, and model names plus traffic volumes are not
nothing on a shared tailnet.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request, status

from .._generated.models import (
    MetricRequest,
    MetricRequestPage,
    MetricsGroup,
    MetricsSummary,
    Problem,
)
from ..metrics import MetricsStore

router = APIRouter(tags=["metrics"])

# A day, because that is the window an operator means by "how has it
# been", and short enough that the default response stays small.
DEFAULT_WINDOW = timedelta(days=1)


def _store(request: Request) -> MetricsStore:
    store: MetricsStore | None = getattr(request.app.state, "metrics", None)
    if store is None:
        # 503 rather than an empty 200: "metrics are off" and "nothing
        # was served in this window" are different answers, and a UI that
        # cannot tell them apart shows an empty chart for a disabled
        # feature.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=Problem(
                type="https://github.com/eugene-plexus/gateway#metrics-disabled",
                title="Metrics disabled",
                status=503,
                detail=(
                    "This gateway is not retaining request metrics — `metricsEnabled` "
                    "is false, or the store could not be opened. Set it on the Config "
                    "page; new requests are recorded from then on, and nothing "
                    "reconstructs the traffic served while it was off."
                ),
                component="gateway",
            ).model_dump(exclude_none=True),
        )
    return store


@router.get("/v1/metrics", response_model=MetricsSummary)
async def get_metrics(
    request: Request,
    since: Annotated[datetime | None, Query()] = None,
    until: Annotated[datetime | None, Query()] = None,
    model: Annotated[str | None, Query()] = None,
    driver: Annotated[str | None, Query()] = None,
    bucket: Annotated[Literal["none", "hour"], Query()] = "none",
) -> MetricsSummary:
    store = _store(request)
    end = until or datetime.now(UTC)
    start = since or (end - DEFAULT_WINDOW)

    groups = store.summary(since=start, until=end, model=model, driver=driver, bucket=bucket)
    return MetricsSummary(
        windowStart=start,
        windowEnd=end,
        gatewayStartedAt=store.started_at,
        rowsDropped=store.rows_dropped,
        # Raw rows are all that is served, so a window reaching past
        # retention describes less than it was asked for and says so.
        truncated=start < store.oldest_retained(),
        groups=[MetricsGroup.model_validate(g) for g in groups],
    )


@router.get("/v1/metrics/requests", response_model=MetricRequestPage)
async def get_metric_requests(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    cursor: Annotated[str | None, Query()] = None,
    model: Annotated[str | None, Query()] = None,
    outcome: Annotated[Literal["served", "error", "cascaded"] | None, Query()] = None,
) -> MetricRequestPage:
    store = _store(request)
    rows, next_cursor = store.requests(limit=limit, cursor=cursor, model=model, outcome=outcome)
    return MetricRequestPage(
        requests=[MetricRequest.model_validate(r) for r in rows],
        nextCursor=next_cursor,
    )
