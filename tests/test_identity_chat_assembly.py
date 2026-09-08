"""Identity-aware turn assembly in the continuous loop.

Each message turn the loop pulls constitution + relevant self-model
entries + per-person relationship context, then builds *per-hemisphere*
system prompts (`turn.build_per_driver_system_prompts`) which reach the
drivers as the `system` message of each `GenerateRequest`.

These tests verify, by inspecting the recorded `GenerateRequest`s on the
fake drivers:

  - With identity configured, both drivers receive a system prompt
    carrying the constitution (name/pronouns/coreValues/freeText).
  - Both drivers receive the SAME persona body — v0.2.x dropped the
    per-hemisphere preamble; divergence comes from the models being
    different and from per-pass cross-talk, not from preamble framing.
  - An explicit non-NIL `personId` resolves that person's relationship
    context; the operator does NOT leak in.
  - A NIL `personId` + identity wired resolves the operator's personId.
  - Self-model entries are filtered to the related person.
  - An identity-service outage degrades gracefully: the loop still
    speaks, using the `defaultSystemPrompt` fallback (mentions Eugene).
  - The no-preamble invariant: left_sys == right_sys, no per-side
    labels, no architecture meta.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx

from eugene_plexus_gateway._generated.hemisphere_models import (
    GenerateRequest,
    Role,
)
from eugene_plexus_gateway._generated.models import (
    Constitution,
    Person,
    RelationshipSummary,
    SelfModelEntry,
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


def _system_message(req: GenerateRequest) -> str:
    """Pluck the system prompt out of a recorded GenerateRequest."""
    return next((m.content for m in req.messages if m.role == Role.system), "")


def _single_pass(left: FakeHemisphereClient, right: FakeHemisphereClient) -> None:
    """Script both fakes for one agreeing bicameral pass + the voice pass.

    Identical "hi"/"hi" => Jaccard agreement 1.0 => terminate after one
    pass; the voice driver (left by default) then speaks once more.
    """
    left.responses = ["hi", "hi voice"]
    right.responses = ["hi"]


def _speech_events(events: list[tuple[str, dict]]) -> list[dict]:
    return [data for kind, data in events if kind == "speech"]


async def test_turn_falls_back_to_default_prompt_when_identity_unconfigured(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """With identity off, the persona body IS `defaultSystemPrompt`
    (which mentions Eugene) for both hemispheres. No per-side labels —
    divergence comes from the models, not from preamble framing."""
    _single_pass(left_fake, right_fake)
    app = build_loop_app(settings, [left_fake, right_fake], identity=None)

    await drive_message(app, make_message_event("hello"))

    left_sys = _system_message(left_fake.calls[0])
    right_sys = _system_message(right_fake.calls[0])
    assert "Eugene" in left_sys
    assert "Eugene" in right_sys
    # Both hemispheres see the same persona-only system prompt.
    assert left_sys == right_sys


async def test_turn_with_identity_threads_constitution_into_both_hemispheres(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """When identity is wired in, both hemispheres' system prompts
    include constitution fields (name, pronouns, coreValues, freeText)
    and stay identical."""
    _single_pass(left_fake, right_fake)
    identity = InProcessIdentity(
        constitution=Constitution(
            name="Eugene",
            pronouns="they/them",
            coreValues=["honesty", "intellectual humility"],
            freeText="You like to cite analogies from neuroscience.",
        ),
    )
    app = build_loop_app(settings, [left_fake, right_fake], identity=identity)

    await drive_message(app, make_message_event("hello"))

    for fake in (left_fake, right_fake):
        sys_msg = _system_message(fake.calls[0])
        assert "Your name: Eugene." in sys_msg
        assert "Pronouns: they/them." in sys_msg
        assert "honesty" in sys_msg
        assert "intellectual humility" in sys_msg
        assert "cite analogies from neuroscience" in sys_msg

    left_sys = _system_message(left_fake.calls[0])
    right_sys = _system_message(right_fake.calls[0])
    # Per-hemisphere persona variation is a planned v0.3 knob; today the
    # two hemispheres see the same persona-only prompt.
    assert left_sys == right_sys


async def test_turn_uses_explicit_personId_relationship_when_supplied(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """A connector adapter supplies a real `personId` on the message
    event. The loop fetches that person's relationship context and
    injects it; the operator is NOT used as fallback."""
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
        relationshipNote="my wife",
    )
    other_summary = RelationshipSummary(
        personId=other_id,
        lastUpdated=datetime.now(UTC),
        summary="Sarah and you chat several times a day, mostly casual.",
        turnCount=42,
    )
    identity = InProcessIdentity(
        persons=[operator, other],
        relationships={other_id: other_summary},
    )
    app = build_loop_app(settings, [left_fake, right_fake], identity=identity)

    _single_pass(left_fake, right_fake)
    await drive_message(app, make_message_event("hello", person_id=other_id))

    sys_msg = _system_message(left_fake.calls[0])
    assert "Sarah" in sys_msg
    assert "my wife" in sys_msg
    assert "several times a day" in sys_msg
    # Operator's name does NOT leak into another-person's turn.
    assert "Troy" not in sys_msg


async def test_turn_resolves_operator_personId_for_nil_person(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """UI turns post NIL_PERSON_ID to mean "this is the operator". With
    identity wired in, the loop looks up the operator's personId and
    uses their relationship context ("your operator" marker)."""
    operator_id = uuid4()
    operator = Person(
        personId=operator_id,
        displayName="Troy",
        isOperator=True,
        createdAt=datetime.now(UTC),
    )
    identity = InProcessIdentity(persons=[operator])
    app = build_loop_app(settings, [left_fake, right_fake], identity=identity)

    _single_pass(left_fake, right_fake)
    await drive_message(app, make_message_event("hello", person_id=NIL_PERSON_ID))

    sys_msg = _system_message(left_fake.calls[0])
    assert "Troy" in sys_msg
    # The "your operator" marker tells Eugene who they're talking to.
    assert "your operator" in sys_msg


async def test_turn_filters_self_model_entries_by_related_person(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """Self-model entries scoped to other persons must not leak when
    talking to a different person."""
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
    entry_about_operator = SelfModelEntry(
        id=uuid4(),
        topic="user-troy",
        content="With Troy you tend to switch into engineering mode.",
        createdAt=datetime.now(UTC),
        relatedPersonIds=[operator_id],
    )
    entry_about_other = SelfModelEntry(
        id=uuid4(),
        topic="user-sarah",
        content="With Sarah you tend to be more playful.",
        createdAt=datetime.now(UTC),
        relatedPersonIds=[other_id],
    )
    identity = InProcessIdentity(
        persons=[operator, other],
        self_model_entries=[entry_about_operator, entry_about_other],
    )
    app = build_loop_app(settings, [left_fake, right_fake], identity=identity)

    _single_pass(left_fake, right_fake)
    await drive_message(app, make_message_event("hello", person_id=other_id))

    sys_msg = _system_message(left_fake.calls[0])
    assert "more playful" in sys_msg
    assert "engineering mode" not in sys_msg


async def test_turn_degrades_when_identity_service_is_unreachable(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """If identity is configured but unreachable mid-assembly, the loop
    must still speak — degrade to the `defaultSystemPrompt` path.
    Identity enriches context; it is not load-bearing for the turn."""

    class _FailingIdentity:
        async def get_constitution(self) -> Constitution:
            raise httpx.ConnectError("identity dead")

        async def query_self_model(
            self,
            *,
            topic: str | None = None,
            person_id: UUID | None = None,
            limit: int = 5,
        ) -> list[SelfModelEntry]:
            raise httpx.ConnectError("identity dead")

        async def get_relationship(self, person_id: UUID) -> RelationshipSummary | None:
            raise httpx.ConnectError("identity dead")

        async def list_persons(self) -> list[Person]:
            raise httpx.ConnectError("identity dead")

        async def aclose(self) -> None:
            return None

    app = build_loop_app(settings, [left_fake, right_fake], identity=_FailingIdentity())

    _single_pass(left_fake, right_fake)
    events = await drive_message(app, make_message_event("hello"))

    # Critically: the loop still replied. (There is no HTTP status to
    # check in the continuous runtime — the recoverable-outage signal is
    # that speech was still emitted.)
    assert _speech_events(events), "identity outage must not silence the turn"

    sys_msg = _system_message(left_fake.calls[0])
    # Fallback persona came from defaultSystemPrompt (which mentions Eugene).
    assert "Eugene" in sys_msg


async def test_pass0_preambles_are_identical_for_both_hemispheres(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """v0.2.x dropped the per-hemisphere preamble entirely. The system
    prompt is the persona body for both drivers — no architecture meta,
    no per-side labels, no "you are the left hemisphere" framing.

    Earlier versions added a distinguishing preamble ("you are the left
    hemisphere, your twin is the right hemisphere running on backend X")
    to make cross-vendor bicameral commitment visible. That backfired:
    the LLMs started addressing the gateway and treating each other
    as siblings to chat with. We now derive divergence from the
    underlying models being different (cross-vendor) and from per-pass
    cross-talk, which surfaces the other side's content as the SAME
    Eugene's prior thought.

    This test pins the no-preamble invariant so a future change that
    re-introduces architectural framing gets caught with the why
    attached. Per-hemisphere persona variation is planned for v0.3
    (operator-selectable).
    """
    _single_pass(left_fake, right_fake)
    app = build_loop_app(settings, [left_fake, right_fake], identity=None)

    await drive_message(app, make_message_event("hello"))

    left_sys = _system_message(left_fake.calls[0])
    right_sys = _system_message(right_fake.calls[0])
    assert left_sys == right_sys
    # No per-side labels surfaced into the prompt.
    assert "left hemisphere" not in left_sys
    assert "right hemisphere" not in left_sys
    # No architecture meta.
    assert "synthetic consciousness" not in left_sys
    assert "inner voice" not in left_sys


async def test_explicit_person_relationship_persisted_to_memory(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    """The reply is attributed to the supplied person in memory — the
    durable counterpart to the relationship context injected into the
    prompt (replaces the v0.2 ChatResponse personId echo)."""
    person_id = uuid4()
    person = Person(
        personId=person_id,
        displayName="Sarah",
        isOperator=False,
        createdAt=datetime.now(UTC),
    )
    identity = InProcessIdentity(persons=[person])
    memory = InProcessMemory()
    app = build_loop_app(settings, [left_fake, right_fake], memory=memory, identity=identity)

    _single_pass(left_fake, right_fake)
    events = await drive_message(app, make_message_event("hello", person_id=person_id))

    cid = next(data["conversationId"] for kind, data in events if kind == "speech")
    stored = memory._conversations[UUID(cid)]
    # User turn + assistant reply, both attributed to the explicit person.
    assert [e.personId for e in stored] == [person_id, person_id]
    reply = stored[-1]
    assert reply.role == Role.assistant
    assert reply.hemisphereAttribution == "voice"
    assert reply.content == "hi voice"
