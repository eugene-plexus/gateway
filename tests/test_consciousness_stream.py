"""Consciousness-stream events must survive the SSE route's serialization.

`GET /v1/stream/consciousness` does `json.dumps(data)` on every event the
loop publishes; a payload with a non-JSON-serializable field (e.g. an
NTState `lastUpdated` datetime that skipped `mode="json"`) would raise and
kill the subscriber's stream. The direct-loop tests read the event dict
straight off the broker queue and never exercise json.dumps, so this test
guards the serialization contract explicitly.
"""

from __future__ import annotations

import json

from eugene_plexus_gateway.settings import Settings
from tests.conftest import (
    FakeHemisphereClient,
    build_loop_app,
    drive_message,
    make_message_event,
)


async def test_all_published_events_are_json_serializable(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    left_fake.responses = ["hi", "hi voice"]
    right_fake.responses = ["hi"]
    app = build_loop_app(settings, [left_fake, right_fake])

    events = await drive_message(app, make_message_event("hello"))

    types = [t for t, _ in events]
    # A normal turn emits at least these — guard that the turn actually ran.
    assert "thought" in types
    assert "gate_decision" in types
    assert "nt_update" in types
    assert "speech" in types

    # Every event the SSE route would `json.dumps` must serialize cleanly —
    # regression for the nt_update datetime leak.
    for _event_type, data in events:
        json.dumps(data)
