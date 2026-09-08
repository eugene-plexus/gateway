"""Turn-assembly helpers for the continuous loop.

These are the Request-free building blocks the loop uses to turn one
incoming message into a deliberation + reply: history shaping, memory
entry construction, NT observation distillation, per-hemisphere system
prompt assembly, and the identity-stack renderers.

They were the proven core of the v0.2 `chat()` handler — moved here
verbatim (not rewritten) so the continuous loop inherits known-good
prompt assembly. The HTTP handler itself is gone (no backwards compat);
the loop in `runtime/loop.py` calls these.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from .._generated.models import (
    Constitution,
    MemoryEntry,
    Message,
    Person,
    RelationshipSummary,
    Role,
    SelfModelEntry,
)
from ..bicameral.loop import BICAMERAL_SUBSTRATE_NOTE, BicameralOutcome
from ..bicameral.nt import Observations
from ..driver_client import DriverClient
from ..tools import (
    TOOL_IDENTITY_GET_CONSTITUTION,
    TOOL_IDENTITY_GET_RELATIONSHIP,
    TOOL_IDENTITY_LIST_PERSONS,
    TOOL_IDENTITY_QUERY_SELF_MODEL,
    ToolContext,
    ToolRunner,
    new_call,
)

log = logging.getLogger(__name__)


def build_history(history: list[Message], user_message: str) -> list[Message]:
    """Conversation history to send to hemispheres for this turn.

    Returns user / assistant turns plus the new user message — no system
    prompt (the bicameral loop prepends per-driver system prompts).
    Hemisphere-tagged intermediate messages from prior turns are
    observability artifacts and don't belong in subsequent prompts.
    """
    out: list[Message] = []
    for msg in history:
        if msg.role in (Role.user, Role.assistant):
            out.append(msg)
    out.append(Message(role=Role.user, content=user_message))
    return out


async def resolve_operator_person_id(tool_runner: ToolRunner) -> UUID | None:
    """Look up the operator's personId from the identity component.

    UI turns may post `NIL_PERSON_ID` as a "this is the operator" marker;
    the gateway resolves the real id on demand. Returns None when no
    operator person exists yet (e.g. an install whose first-run wizard
    hasn't run `ensure_operator`).
    """
    invocation = await tool_runner.run(new_call(TOOL_IDENTITY_LIST_PERSONS))
    persons: list[Person] = invocation.payload
    for p in persons:
        if p.isOperator:
            return p.personId
    return None


def build_memory_entry(
    *,
    conversation_id: UUID,
    person_id: UUID,
    message: Message,
    nt_snapshot: Any = None,
    hemisphere_attribution: str | None = None,
) -> MemoryEntry:
    """Wrap a `Message` in a `MemoryEntry` with full metadata.

    The gateway is the source of truth for the NT snapshot and the
    hemisphere attribution — the memory component stores both as opaque
    blobs.
    """
    return MemoryEntry(
        entryId=uuid4(),
        personId=person_id,
        conversationId=conversation_id,
        role=message.role,
        content=message.content,
        timestamp=message.timestamp or datetime.now(UTC),
        ntStateSnapshot=nt_snapshot,
        hemisphereAttribution=hemisphere_attribution,
    )


def build_observations(outcome: BicameralOutcome, agreement_threshold: float) -> Observations:
    """Distill the bicameral outcome into NT-system observations.

    Final-pass agreement, whether termination was convergence vs cap, and
    average pass latency are the observables. The active
    `agreement_threshold` rides along so the NT system can interpret
    `final_agreement` relative to the bar the operator set.
    """
    final_callosum = outcome.passes[-1].callosum
    final_agreement = final_callosum.agreement
    pass_count = len(outcome.passes)
    avg_latency_ms = (
        sum(outcome.pass_latencies_ms) / len(outcome.pass_latencies_ms)
        if outcome.pass_latencies_ms
        else 0.0
    )
    return Observations(
        final_agreement=final_agreement,
        pass_count=pass_count,
        agreement_threshold=agreement_threshold,
        avg_pass_latency_ms=avg_latency_ms,
    )


def render_recent_turns(entries: list[MemoryEntry]) -> str:
    """Render recent memory turns as a compact prompt section."""
    if not entries:
        return ""
    # Newest first → reverse for readability (oldest mention first).
    lines = ["Recent turns with this person (oldest first):"]
    for e in reversed(entries):
        label = "you" if e.role == Role.assistant else "they"
        content = e.content if len(e.content) <= 280 else e.content[:280] + "…"
        lines.append(f"- {label}: {content}")
    return "\n".join(lines)


def resolve_voice_driver(drivers: list[DriverClient], configured_name: object) -> DriverClient:
    """Pick which driver performs the voice pass (the speak effector).

    Operator-configurable via `voiceDriver`. When unset (or unknown),
    defaults to the first driver; an invalid name is logged, not fatal.
    """
    if isinstance(configured_name, str) and configured_name.strip():
        target = configured_name.strip()
        for driver in drivers:
            if driver.name == target:
                return driver
        log.warning(
            "voiceDriver=%r not found in topology %s; falling back to first driver",
            target,
            [d.name for d in drivers],
        )
    return drivers[0]


def render_constitution(c: Constitution) -> str:
    parts: list[str] = []
    parts.append(f"Your name: {c.name}.")
    if c.pronouns:
        parts.append(f"Pronouns: {c.pronouns}.")
    if c.coreValues:
        parts.append("Core values: " + "; ".join(c.coreValues) + ".")
    if c.freeText:
        parts.append(c.freeText.strip())
    return "\n".join(parts)


def render_self_model(entries: list[SelfModelEntry]) -> str:
    if not entries:
        return ""
    lines = ["What you've come to notice about yourself:"]
    for e in entries:
        lines.append(f"- [{e.topic}] {e.content}")
    return "\n".join(lines)


def render_relationship(summary: RelationshipSummary, person: Person | None) -> str:
    """Render relationship context for the speaker."""
    parts: list[str] = []
    if person is not None:
        intro = f"You are talking to {person.displayName}"
        if person.isOperator:
            intro += " (your operator — the person who set up your install)"
        intro += "."
        parts.append(intro)
        if person.relationshipNote:
            parts.append(f"Operator note about them: {person.relationshipNote}")
    if summary.summary:
        parts.append(summary.summary)
    elif summary.turnCount and summary.turnCount > 0:
        parts.append(f"You've shared {summary.turnCount} prior turn(s) with this person.")
    return "\n".join(parts)


async def build_per_driver_system_prompts(
    *,
    drivers: list[DriverClient],
    tool_runner: ToolRunner,
    person_id: UUID | None,
    operator_person_id: UUID | None,
    fallback_default: str,
    recent_turns: list[MemoryEntry] | None = None,
    cross_pass_framing: str = "parallel_thread",
) -> dict[str, str]:
    """Assemble per-hemisphere system prompts.

    When identity is configured (its tools are registered on the runner),
    each driver gets the full identity stack (constitution + self-model +
    relationship). Otherwise every driver gets the shared
    `fallback_default`. Both hemispheres receive the same persona body —
    exposing the bicameral architecture in the system prompt made the
    LLMs address the gateway and treat cross-pass content as
    conversation with a sibling.
    """
    if len(drivers) != 2:
        raise ValueError(
            f"per-driver prompt assembly requires exactly two drivers; got {len(drivers)}"
        )

    left, right = drivers[0], drivers[1]
    identity_available = tool_runner.has(TOOL_IDENTITY_GET_CONSTITUTION)

    persona_body: str
    if identity_available:
        # Resolve the speaker: explicit person_id wins; else operator.
        effective_person_id = person_id or operator_person_id
        constitution = (await tool_runner.run(new_call(TOOL_IDENTITY_GET_CONSTITUTION))).payload
        self_model_entries = (
            await tool_runner.run(
                new_call(TOOL_IDENTITY_QUERY_SELF_MODEL),
                ToolContext(inputs={"topic": None, "person_id": effective_person_id, "limit": 5}),
            )
        ).payload
        if effective_person_id is not None:
            relationship = (
                await tool_runner.run(
                    new_call(TOOL_IDENTITY_GET_RELATIONSHIP),
                    ToolContext(inputs={"person_id": effective_person_id}),
                )
            ).payload
            persons = (await tool_runner.run(new_call(TOOL_IDENTITY_LIST_PERSONS))).payload
        else:
            relationship = None
            persons = []
        person_record = (
            next((p for p in persons if p.personId == effective_person_id), None)
            if effective_person_id is not None
            else None
        )

        sections: list[str] = [render_constitution(constitution)]
        sm_text = render_self_model(self_model_entries)
        if sm_text:
            sections.append(sm_text)
        if relationship is not None:
            rel_text = render_relationship(relationship, person_record)
            if rel_text:
                sections.append(rel_text)
        elif person_record is not None:
            intro = f"You are talking to {person_record.displayName}"
            if person_record.isOperator:
                intro += " (your operator — the person who set up your install)"
            intro += "."
            if person_record.relationshipNote:
                intro += f"\nOperator note about them: {person_record.relationshipNote}"
            sections.append(intro)
        persona_body = "\n\n".join(s for s in sections if s)
    else:
        persona_body = fallback_default

    if recent_turns:
        recent_section = render_recent_turns(recent_turns)
        if recent_section:
            persona_body = f"{persona_body}\n\n{recent_section}".strip()

    # When `parallel_thread` framing is in use, append the substrate
    # explanation so the model knows what <parallel_thread>…</parallel_thread>
    # tags mean when they appear in the conversation.
    if cross_pass_framing == "parallel_thread":
        persona_body = f"{persona_body}\n\n{BICAMERAL_SUBSTRATE_NOTE}".strip()

    return {left.name: persona_body, right.name: persona_body}
