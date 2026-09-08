"""NT-system integration through the continuous loop's message turn.

Verify (translated from the v0.2 `POST /v1/chat` round-trips):
  - NT state evolves on the turn and is emitted on the consciousness
    stream as an `nt_update` event (the live evolution of v0.2's
    `ChatResponse.ntStateAtEnd`).
  - A settled (high-agreement) turn moves GABA + dopamine up. The impulse
    map keys "calm" on whether the bout SETTLED (final agreement ≥
    threshold), not on pass count — under the plateau-stop a clean
    convergence is no longer synonymous with "exactly one pass."
  - Low per-pass latency pulls norepinephrine down.
  - NT state evolves across turns on the same app — it is mutated in
    place, not reset per message (the live evolution of v0.2's
    `ntStateAtStart`/`ntStateAtEnd` continuity + the admin nt-state route,
    which just returns `app.state.nt_state`).

DROPPED: `test_high_cortisol_widens_modulated_max_passes` — NT no longer
modulates the pass count (deliberation depth is owned by the plateau-stop
gate; `modulated_max_passes` was deleted, since a count is the kind of
arbitrary cognitive limit the design forbids).
"""

from __future__ import annotations

from eugene_plexus_gateway.settings import Settings
from tests.conftest import (
    FakeHemisphereClient,
    build_loop_app,
    drive_message,
    make_message_event,
)


def _nt_updates(events: list[tuple[str, dict]]) -> list[dict]:
    return [data for kind, data in events if kind == "nt_update"]


async def test_turn_emits_nt_update_with_convergence_directions(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """A settled turn evolves NT and publishes the new state on the stream
    so the UI can render Eugene's cognitive arc per turn → GABA up,
    dopamine up."""
    # Both hemispheres say the same thing every pass — the fake repeats its
    # last response once its script drains, so agreement stays 1.0 across
    # however many passes the plateau gate runs (and the voice pass reuses
    # it too). A settled bout → GABA up, dopamine up.
    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    app = build_loop_app(settings, [left_fake, right_fake])

    events = await drive_message(app, make_message_event("hello"))

    updates = _nt_updates(events)
    assert len(updates) == 1, "expected exactly one nt_update on the stream"
    emitted = updates[0]
    assert emitted["gaba"]["level"] > 0.5
    assert emitted["dopamine"]["level"] > 0.5

    # The loop mutates app.state.nt_state in place; the emitted event must
    # mirror the live state (the admin nt-state route just returns it).
    assert app.state.nt_state.gaba.level > 0.5
    assert app.state.nt_state.dopamine.level > 0.5
    assert app.state.nt_state.gaba.level == emitted["gaba"]["level"]
    assert app.state.nt_state.dopamine.level == emitted["dopamine"]["level"]


async def test_low_latency_pulls_norepinephrine_down(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """Each pass's max-of-hemispheres latency feeds NT.norepinephrine. The
    fakes report 1ms, so the short-latency clause pulls NE down from its
    0.5 neutral start — assert direction, not magnitude."""
    left_fake.responses = ["hi", "hi voice"]
    right_fake.responses = ["hi"]
    app = build_loop_app(settings, [left_fake, right_fake])

    events = await drive_message(app, make_message_event("hello"))

    updates = _nt_updates(events)
    assert len(updates) == 1
    assert updates[0]["norepinephrine"]["level"] < 0.5
    assert app.state.nt_state.norepinephrine.level < 0.5


async def test_nt_state_evolves_across_turns_on_same_app(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """NT state is carried on app.state and mutated in place, not reset per
    message. After two consecutive agreeing turns the live state still
    reflects accumulated convergence and `lastUpdated` advanced."""
    # Repeat-last keeps every pass (and the voice pass) at "hi" across both
    # turns, so each turn settles regardless of how many passes the gate runs.
    left_fake.responses = ["hi"]
    right_fake.responses = ["hi"]
    app = build_loop_app(settings, [left_fake, right_fake])

    await drive_message(app, make_message_event("first"))
    after_first = app.state.nt_state
    first_last_updated = after_first.lastUpdated
    assert after_first.dopamine.level > 0.5

    await drive_message(app, make_message_event("second"))
    after_second = app.state.nt_state

    # Same object lineage, evolved further — not reset to neutral (0.5).
    assert after_second.dopamine.level > 0.5
    assert after_second.gaba.level > 0.5
    # The second turn ticked NT again: its timestamp advanced past the first.
    assert after_second.lastUpdated >= first_last_updated
