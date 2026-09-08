"""POST /v1/events — the single afferent-event injection door.

Accepts one `AfferentEvent`, enqueues it for the continuous consciousness
loop, and returns `202` immediately. Fire-and-forget: this call does not
block for a reply and does not guarantee one — whether/when/how Eugene
responds is the loop's decision. Replies leave asynchronously as
`EfferentSpeechAct`s on `GET /v1/stream/consciousness`.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status

from .._generated.models import AfferentEvent, Problem

router = APIRouter(tags=["chat"])

log = logging.getLogger(__name__)


@router.post("/v1/events", status_code=status.HTTP_202_ACCEPTED)
async def inject_event(request: Request, body: AfferentEvent) -> dict[str, object]:
    app = request.app
    if getattr(app.state, "safe_mode", False) or len(getattr(app.state, "drivers", [])) < 2:
        # Nothing to think with — refuse the event rather than enqueue it
        # for a loop that can't act on it. Mirrors the v0.2 chat 503 in the
        # same conditions (safe mode / fewer than two driver slots).
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=Problem(
                type="https://github.com/eugene-plexus/gateway#loop-unavailable",
                title="Consciousness loop unavailable",
                status=503,
                detail=(
                    "The gateway has fewer than two driver slots resolved "
                    "(or is in safe mode), so it cannot deliberate. Check the "
                    "`drivers` config and the watchdog topology, then restart."
                ),
                component="gateway",
            ).model_dump(exclude_none=True),
        )
    accepted = app.state.loop.submit(body)
    if not accepted:
        # The loop's event queue is full — Eugene is receiving events faster
        # than it can think through them. Shed rather than grow memory with
        # work it can't drain in time (lossy-not-a-queue).
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=Problem(
                type="https://github.com/eugene-plexus/gateway#loop-saturated",
                title="Consciousness loop saturated",
                status=503,
                detail=(
                    "The event queue is full — Eugene is processing events "
                    "slower than they arrive. Retry shortly."
                ),
                component="gateway",
            ).model_dump(exclude_none=True),
        )
    return {"eventId": str(body.eventId), "accepted": True}
