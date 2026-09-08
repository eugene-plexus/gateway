"""Pub/sub for the consciousness stream (`GET /v1/stream/consciousness`).

The loop publishes typed observability events — `thought`, `nt_update`,
`gate_decision`, `tool_call`, `speech`, `focus_switch`, `phase_change` —
and any number of subscribers (the UI's stream-of-consciousness view)
receive them as Server-Sent Events.

**Lossy by design.** A slow or absent subscriber's queue fills and
further events are *dropped for that subscriber* rather than
back-pressuring the loop. Observability must never stall cognition — the
same "lossy, not a queue" principle the design applies to
presence/salience. A subscriber that falls behind misses events; it does
not slow Eugene down.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)

# Per-subscriber buffer. Generous enough to absorb one turn's burst of
# events; a subscriber that can't keep up past this is genuinely behind
# and drops events (debug-logged) rather than blocking the loop.
_SUBSCRIBER_QUEUE_MAX = 256

# (event_type, json-serializable data). `event_type` is the SSE `event:`
# field; `data` is the SSE `data:` payload (serialized by the route).
ConsciousnessEvent = tuple[str, dict[str, Any]]


class ConsciousnessBroker:
    """Fan-out of consciousness events to all live SSE subscribers."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[ConsciousnessEvent]] = set()

    def subscribe(self) -> asyncio.Queue[ConsciousnessEvent]:
        """Register a subscriber and return its private event queue."""
        queue: asyncio.Queue[ConsciousnessEvent] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_MAX)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[ConsciousnessEvent]) -> None:
        """Drop a subscriber (its SSE connection closed)."""
        self._subscribers.discard(queue)

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """Fan one event out to every subscriber. Non-blocking.

        Drops the event for any subscriber whose queue is full — the loop
        never waits on a slow consumer.
        """
        event: ConsciousnessEvent = (event_type, data)
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                log.debug(
                    "consciousness subscriber behind; dropping %s event",
                    event_type,
                )

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
