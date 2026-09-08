"""Continuous-loop memory + personId wiring tests (M2 slice 2 rewrite).

These verify the memory contract the loop preserves from the v0.2 chat
handler, now driven through `drive_message` instead of `POST /v1/chat`:

  - The loop writes full MemoryEntries (personId, NT snapshot,
    hemisphereAttribution="voice" on the reply, reply content = the voice
    pass output) — not bare Messages.
  - When identity resolves an operator personId, both writes for the
    turn use that personId.
  - An explicit (body) personId wins over the operator fallback.
  - With neither identity nor an explicit personId, writes fall back to
    NIL_PERSON_ID without failing the turn.
  - `person_recent` enrichment is injected into the hemisphere prompts
    only for non-NIL personIds, excluding the active conversation.
  - For a NIL person, `person_recent` is skipped — no NIL-bucket leak.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from eugene_plexus_gateway._generated.hemisphere_models import Role as HemisphereRole
from eugene_plexus_gateway._generated.models import (
    Constitution,
    MemoryEntry,
    Person,
    Role,
)
from eugene_plexus_gateway.identity import InProcessIdentity
from eugene_plexus_gateway.memory import NIL_PERSON_ID, InProcessMemory
from eugene_plexus_gateway.settings import Settings
from tests.conftest import (
    FakeHemisphereClient,
    build_loop_app,
    drive_message,
    make_message_event,
)


def _speech_conversation_id(events: list[tuple[str, dict]]) -> UUID:
    """Pull the conversationId the loop spoke into off the `speech` event."""
    speech = next(data for kind, data in events if kind == "speech")
    return UUID(speech["conversationId"])


def _system_message(fake: FakeHemisphereClient) -> str:
    """The system prompt that reached a hemisphere on its first call."""
    return next(m.content for m in fake.calls[0].messages if m.role == HemisphereRole.system)


async def test_loop_writes_full_memory_entries_with_resolved_personid(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """When identity is wired and the message posts NIL (a "this is the
    operator" UI marker), both writes carry the operator's personId, an NT
    snapshot, and the reply is tagged `hemisphereAttribution="voice"` with
    content = the voice pass output."""
    operator_id = uuid4()
    operator = Person(
        personId=operator_id,
        displayName="Troy",
        isOperator=True,
        createdAt=datetime.now(UTC),
    )

    memory = InProcessMemory()
    identity = InProcessIdentity(persons=[operator])
    app = build_loop_app(settings, [left_fake, right_fake], memory=memory, identity=identity)

    # Voice pass runs after deliberation; the extra left response covers it.
    left_fake.responses = ["hi", "hi voice"]
    right_fake.responses = ["hi"]

    events = await drive_message(app, make_message_event("hello"))

    cid = _speech_conversation_id(events)
    entries = memory._conversations[cid]
    assert len(entries) == 2

    user_entry, reply_entry = entries
    assert user_entry.personId == operator_id
    assert reply_entry.personId == operator_id
    assert user_entry.role == Role.user
    assert reply_entry.role == Role.assistant
    # The reply Eugene actually sends is the voice pass output, tagged
    # hemisphereAttribution="voice".
    assert reply_entry.hemisphereAttribution == "voice"
    assert reply_entry.content == "hi voice"
    # NT snapshot present on both — gives the v0.3+ analyser something to
    # correlate output style against state.
    assert user_entry.ntStateSnapshot is not None
    assert reply_entry.ntStateSnapshot is not None


async def test_loop_body_personid_wins_over_operator_fallback(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """Connector adapters supply an explicit personId on the message. Even
    when an operator is registered, that explicit personId must win — the
    speaker isn't the operator."""
    operator_id = uuid4()
    other_id = uuid4()
    operator = Person(
        personId=operator_id,
        displayName="Troy",
        isOperator=True,
        createdAt=datetime.now(UTC),
    )
    other = Person(
        personId=other_id,
        displayName="Sarah",
        isOperator=False,
        createdAt=datetime.now(UTC),
    )

    memory = InProcessMemory()
    identity = InProcessIdentity(persons=[operator, other])
    app = build_loop_app(settings, [left_fake, right_fake], memory=memory, identity=identity)

    left_fake.responses = ["hi", "hi voice"]
    right_fake.responses = ["hi"]

    events = await drive_message(app, make_message_event("hello", person_id=other_id))

    cid = _speech_conversation_id(events)
    entries = memory._conversations[cid]
    assert entries  # the turn produced writes
    assert all(e.personId == other_id for e in entries)


async def test_loop_falls_back_to_NIL_personid_when_no_identity_and_no_body(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """No identity component + NIL personId on the message = NIL_PERSON_ID
    on the writes. The turn still completes — the loop never fails over
    personId resolution alone."""
    memory = InProcessMemory()
    app = build_loop_app(settings, [left_fake, right_fake], memory=memory, identity=None)

    left_fake.responses = ["hi", "hi voice"]
    right_fake.responses = ["hi"]

    events = await drive_message(app, make_message_event("hello"))

    cid = _speech_conversation_id(events)
    entries = memory._conversations[cid]
    assert len(entries) == 2
    assert all(e.personId == NIL_PERSON_ID for e in entries)


async def test_loop_injects_recent_turns_from_prior_conversations(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """Memory's recent turns with the speaker get injected into the
    hemisphere prompts — concrete context, not just identity's summary.
    Turns from the CURRENT conversation are excluded (already in history)."""
    operator_id = uuid4()
    operator = Person(
        personId=operator_id,
        displayName="Troy",
        isOperator=True,
        createdAt=datetime.now(UTC),
    )

    memory = InProcessMemory()
    # Pre-seed a prior conversation with two turns from the operator.
    prior_cid = uuid4()
    memory._conversations[prior_cid] = [
        MemoryEntry(
            entryId=uuid4(),
            personId=operator_id,
            conversationId=prior_cid,
            role=Role.user,
            content="we talked about cats last week",
            timestamp=datetime.now(UTC),
        ),
        MemoryEntry(
            entryId=uuid4(),
            personId=operator_id,
            conversationId=prior_cid,
            role=Role.assistant,
            content="and I mentioned my favorite species",
            timestamp=datetime.now(UTC),
        ),
    ]

    identity = InProcessIdentity(
        constitution=Constitution(name="Eugene"),
        persons=[operator],
    )
    app = build_loop_app(settings, [left_fake, right_fake], memory=memory, identity=identity)

    left_fake.responses = ["hi", "hi voice"]
    right_fake.responses = ["hi"]

    await drive_message(app, make_message_event("remember?"))

    # The system message in each hemisphere's first GenerateRequest carries
    # the prior-conversation recap.
    for fake in (left_fake, right_fake):
        sys_msg = _system_message(fake)
        assert "cats last week" in sys_msg
        assert "favorite species" in sys_msg
        assert "Recent turns with this person" in sys_msg


async def test_loop_skips_recent_turns_for_nil_person(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """When the speaker resolves to NIL, the loop must NOT call
    `person_recent` — that would surface unrelated NIL-bucket entries from
    prior anonymous turns, which would be confusing rather than helpful."""
    memory = InProcessMemory()
    # Pre-seed a NIL-bucket entry from a prior anonymous turn.
    prior_cid = uuid4()
    memory._conversations[prior_cid] = [
        MemoryEntry(
            entryId=uuid4(),
            personId=NIL_PERSON_ID,
            conversationId=prior_cid,
            role=Role.user,
            content="UNRELATED_NIL_BUCKET_LEAK_CANARY",
            timestamp=datetime.now(UTC),
        ),
    ]

    app = build_loop_app(settings, [left_fake, right_fake], memory=memory, identity=None)

    left_fake.responses = ["hi", "hi voice"]
    right_fake.responses = ["hi"]

    await drive_message(app, make_message_event("hello"))

    for fake in (left_fake, right_fake):
        sys_msg = _system_message(fake)
        assert "UNRELATED_NIL_BUCKET_LEAK_CANARY" not in sys_msg
