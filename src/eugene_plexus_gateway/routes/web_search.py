"""Search as an external engine, with front-door authentication and attribution."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request
from starlette.responses import JSONResponse

from .. import admission, web_search_door
from ..dependencies import require_authorized
from ..metrics import RequestRow, ToolExecutionRow
from ..model_patterns import permits
from ..server_tools import why_not

router = APIRouter(tags=["tools"], dependencies=[Depends(require_authorized)])


def problem(status: int, detail: str, retry: str | None = None) -> JSONResponse:
    return JSONResponse(
        {
            "type": "about:blank",
            "title": "Search unavailable" if status >= 500 else "Search refused",
            "status": status,
            "detail": detail,
            "component": "gateway",
        },
        status_code=status,
        media_type="application/problem+json",
        headers={"Retry-After": retry} if retry is not None else None,
    )


@router.post("/v1/web_search", operation_id="webSearch")
async def web_search(request: Request) -> JSONResponse:
    try:
        body = await request.json()
        query, count = web_search_door.parse(body)
    except ValueError as exc:
        return problem(400, str(exc))
    await admission.authorize(request)
    context = admission.current.get()
    table = getattr(request.app.state, "routing", None)
    reason = why_not(table, context)
    if reason:
        forbidden = context is not None and (
            context.local_only or not permits(context.allowed_tools, "web_search")
        )
        return problem(403 if forbidden else 503, reason)
    if context is not None and context.key_id is not None:
        # The pinned authority only reserves inference for a named model.
        # Never execute uncharged searches or invent a model to evade its policy.
        return problem(
            503,
            "Web search needs shared tool admission, which this authority contract does not "
            "support yet. Update the agent and gateway together.",
        )
    began = time.perf_counter()
    at = datetime.now(UTC)
    result, shaped, status, retry = await web_search_door.run(table, query, count)
    metrics = getattr(request.app.state, "metrics", None)
    if metrics is not None:
        metrics.record(
            RequestRow(
                started_at=at,
                requested_model="web_search",
                door="web_search",
                total_ms=int((time.perf_counter() - began) * 1000),
                attempts=1,
                outcome="served" if status == 200 else "error",
                client_key_id=context.key_id if context else None,
                client_key_name=context.key_name if context else None,
                person_id=context.person_id if context else None,
                app_client_id=context.app_client_id if context else None,
                request_id=context.id if context else None,
                tool_executions=[
                    ToolExecutionRow(
                        tool="web_search",
                        outcome=result.outcome,
                        driver=result.driver,
                        node=result.node,
                        provider=result.provider,
                        version=result.version,
                        results=len(result.results) if status == 200 else None,
                        elapsed_ms=result.elapsed_ms,
                    )
                ],
            )
        )
        if context:
            context.recorded = True
    return (
        JSONResponse(shaped)
        if status == 200
        else problem(status, result.error or "The search failed.", retry)
    )
