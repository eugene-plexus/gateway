"""GET /v1/stream/consciousness — the SSE observability stream.

Subscribes to the `ConsciousnessBroker` and relays Eugene's live inner
activity — `thought`, `nt_update`, `gate_decision`, `tool_call`,
`speech`, `focus_switch`, `phase_change` — as Server-Sent Events. This is
the UI's stream-of-consciousness view (the live evolution of the v0.2
bicameral rail). External channels only *hear speech*; this is the fMRI.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..runtime.stream import ConsciousnessBroker

router = APIRouter(tags=["chat"])

log = logging.getLogger(__name__)


@router.get("/v1/stream/consciousness")
async def stream_consciousness(request: Request) -> StreamingResponse:
    broker: ConsciousnessBroker = request.app.state.broker
    queue = broker.subscribe()

    async def events() -> AsyncIterator[str]:
        # On client disconnect the `finally` unsubscribes so the broker
        # doesn't fan out to a dead queue forever. Detection can't depend on
        # publish cadence (Starlette only notices a dead socket on the next
        # write): poll `is_disconnected` each iteration and emit a heartbeat
        # on idle, which forces a write that surfaces the dead socket (and
        # keeps proxies from idle-timing-out the stream).
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event_type, data = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        finally:
            broker.unsubscribe(queue)

    return StreamingResponse(events(), media_type="text/event-stream")
