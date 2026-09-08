"""The continuous loop emits a `tool_call` event per ToolRunner invocation.

Every gateway operation runs through the ToolRunner; in the
continuous runtime each invocation is published to the consciousness
stream as a `tool_call` event — the live evolution of the v0.2
`ChatResponse.toolInvocations` trace — so the UI can render the
perception/action layer beneath deliberation.
"""

from __future__ import annotations

from uuid import uuid4

from eugene_plexus_gateway.settings import Settings
from tests.conftest import (
    FakeHemisphereClient,
    build_loop_app,
    drive_message,
    make_message_event,
)


def _tool_calls(events: list[tuple[str, dict]]) -> list[dict]:
    return [data for kind, data in events if kind == "tool_call"]


async def test_loop_emits_tool_calls_across_all_three_channels(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    left_fake.responses = ["hi", "hi voice"]
    right_fake.responses = ["hi"]
    app = build_loop_app(settings, [left_fake, right_fake])

    # A non-NIL personId makes the afferent memory recall fire (it's
    # skipped for the NIL person), so this single turn exercises all three
    # channels even with identity off.
    events = await drive_message(app, make_message_event("hello", person_id=uuid4()))

    calls = _tool_calls(events)
    assert calls, "expected tool_call events on the consciousness stream"

    names = {c["name"] for c in calls}
    assert "memory_person_recent" in names  # afferent read
    assert "memory_append_entry" in names  # efferent write
    assert "nt_observe" in names  # internal regimented call

    channels = {c["channel"] for c in calls}
    assert channels == {"afferent", "efferent", "internal"}

    # Reversibility class rides along for the System-1/2 gate; memory
    # writes are reversible, reads/internal are read_only.
    for c in calls:
        if c["name"] == "memory_append_entry":
            assert c["effect"] == "reversible"
        else:
            assert c["effect"] == "read_only"
        assert c["name"]
        assert c["channel"] in {"afferent", "efferent", "internal"}
        assert "latencyMs" in c


async def test_tool_trace_is_fresh_each_turn(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """Each turn begins a fresh trace — the second turn must not carry the
    first turn's invocations (guards the task-local contextvar reset)."""
    left_fake.responses = ["hi", "hi voice", "hi", "hi voice"]
    right_fake.responses = ["hi", "hi"]
    app = build_loop_app(settings, [left_fake, right_fake])

    first = _tool_calls(await drive_message(app, make_message_event("one")))
    second = _tool_calls(await drive_message(app, make_message_event("two")))

    # NIL person → no afferent recall; each turn runs the same small set
    # (two efferent writes + one internal nt_observe).
    assert {c["name"] for c in second} == {"memory_append_entry", "nt_observe"}
    assert len(second) == len(first)
