"""The consciousness loop — Eugene's single, continuous cognitive task.

One Eugene = one consciousness = one workspace = one attention. This loop
is the *only* thing that thinks: HTTP handlers inject `AfferentEvent`s
onto its queue (`POST /v1/events`) and subscribe to its observability
stream (`GET /v1/stream/consciousness`). Because exactly one task mutates
cognitive state, there is no locking anywhere in the cognition path.

**Cognition flow** on a `message` event: recall → per-driver prompts →
bicameral deliberation (depth owned by the plateau-stop gate, not a fixed
pass count) → NT tick → the action gate elects *speak* or *stay-silent* on
anticipated net NT valence → (if speaking) voice pass → persist → emit an
`EfferentSpeechAct`. The whole thing is fire-and-forget — no caller waits
on a response; the reply leaves asynchronously as the speech act — and
every step publishes to the consciousness stream. Adenosine/sleep,
presence, self-initiated (unaddressed) speech, and multi-focus salience
switching layer on in later increments — see
`docs/design/m1-continuous-runtime.md` in `specs`.

Errors NEVER crash the loop: there is no caller waiting on an HTTP
response, so a failed recall / deliberation / voice pass is logged (and,
where useful, surfaced on the stream) and the loop moves on.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI

from .._generated.models import (
    AfferentEvent,
    EfferentSpeechAct,
    GateDecision,
    Message,
    NTState,
    Role,
    ToolInvocationRecord,
)
from ..bicameral.action import Action, ActionPolicyParams, select_action
from ..bicameral.loop import run_bicameral_loop
from ..bicameral.nt import modulated_temperature, net_valence
from ..bicameral.plateau import BoutGate, PlateauParams
from ..bicameral.voice import run_voice_pass
from ..driver_client import DriverClient, DriverError
from ..memory import NIL_PERSON_ID, MemoryClient
from ..tools import (
    TOOL_MEMORY_APPEND_ENTRY,
    TOOL_MEMORY_PERSON_RECENT,
    TOOL_NT_OBSERVE,
    ToolContext,
    ToolRunner,
    begin_tool_trace,
    new_call,
)
from . import turn
from .stream import ConsciousnessBroker
from .workspace import Workspace

log = logging.getLogger(__name__)

# Bound on pending afferent events. Single-attention means the loop drains
# one event per multi-second turn; an unbounded queue lets a faster-than-
# realtime producer grow memory without limit (and makes Eugene act on a
# stale flood). When full, POST /v1/events 503s — lossy-not-a-queue, the
# same principle the observability stream applies.
_EVENT_QUEUE_MAX = 256


def _enum_value(value: object) -> object:
    """Normalize a generated enum field to its underlying value.

    datamodel-code-generator emits `Enum` members for closed string enums;
    `--collapse-root-models` can also leave plain strings. Compare on the
    value either way.
    """
    return getattr(value, "value", value)


class ConsciousnessLoop:
    """Eugene's single continuous cognitive task.

    Constructed in the FastAPI lifespan after `app.state` is built, then
    `start()`ed. It reads its dependencies (drivers, memory, identity,
    scorer, tool_runner, config, NT) off `app.state` at handling time so
    it always sees current state (NT mutates turn to turn).
    """

    def __init__(self, app: FastAPI, broker: ConsciousnessBroker) -> None:
        self._app = app
        self._broker = broker
        self._queue: asyncio.Queue[AfferentEvent] = asyncio.Queue(maxsize=_EVENT_QUEUE_MAX)
        self._workspace = Workspace()
        self._running = False
        self._task: asyncio.Task[None] | None = None

    # -- lifecycle -----------------------------------------------------

    @property
    def queue(self) -> asyncio.Queue[AfferentEvent]:
        return self._queue

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def submit(self, event: AfferentEvent) -> bool:
        """Enqueue an afferent event (called by `POST /v1/events`).

        Returns False when the queue is full — the loop can't keep up, so
        the route sheds the event (503) rather than growing memory with
        work it provably can't drain in time.
        """
        try:
            self._queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            return False

    def start(self) -> None:
        if self._task is None:
            self._running = True
            self._task = asyncio.create_task(self.run(), name="consciousness-loop")
            self._task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        """Clear the task handle and shout if cognition died unexpectedly.

        A dead loop that silently keeps accepting events (the queue fills,
        nothing drains it) is the worst failure mode; surface it loudly and
        clear the handle so a future `start()` can recreate the loop.
        """
        self._task = None
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.error("consciousness loop DIED: %r — cognition has stopped", exc)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def run(self) -> None:
        """The loop. Blocks on the queue; processes one event at a time.

        Single-attention is literal: one event is fully handled before the
        next is dequeued. Idle (no events) is a cheap block here; the
        seeking/mind-wandering idle floor + adenosine wake/sleep arrive in
        a later increment.
        """
        log.info("consciousness loop started")
        try:
            while self._running:
                event = await self._queue.get()
                try:
                    await self._handle(event)
                except Exception:
                    # The loop must never die on a single bad event.
                    log.exception("consciousness loop: unhandled error processing an event")
        except asyncio.CancelledError:
            log.info("consciousness loop cancelled")
            raise

    # -- dispatch ------------------------------------------------------

    async def _handle(self, event: AfferentEvent) -> None:
        kind = _enum_value(event.kind)
        if kind == "message":
            await self._handle_message(event)
        elif kind == "presence":
            # Presence (afferent occupancy) lands in M3 — NT-modulated
            # lossy salience + initiation. For now, observed and dropped.
            log.debug("presence event %s observed (M3 not yet wired)", event.eventId)
        else:
            log.warning("unknown afferent event kind %r — ignoring", kind)

    # -- the message turn (port of the v0.2 chat handler) --------------

    async def _handle_message(self, event: AfferentEvent) -> None:
        msg = event.message
        if msg is None:
            log.warning("message event %s has no message payload — dropping", event.eventId)
            return

        app = self._app
        if getattr(app.state, "safe_mode", False):
            log.warning("loop received a message in safe mode — nothing to think with; dropping")
            return
        drivers: list[DriverClient] = app.state.drivers
        if len(drivers) < 2:
            log.warning(
                "loop received a message but %d driver slot(s) resolved (need 2) — dropping",
                len(drivers),
            )
            return

        store = app.state.config_store
        memory: MemoryClient = app.state.memory
        tool_runner: ToolRunner = app.state.tool_runner
        scorer = app.state.scorer

        # Per-event tool trace. The loop is a single task that handles one
        # event fully before the next, and begin_tool_trace() re-sets the
        # task-local trace at the top of every turn — so records never bleed
        # across turns. (M3 note: any future non-message handler that runs
        # tools on this task must open its own begin_tool_trace scope.)
        trace = begin_tool_trace()
        # msg.channelContext (ambient platform messages preceding a mention)
        # rides on IncomingMessage but is not yet surfaced to hemispheres —
        # deferred like presence (M3), not dropped by oversight.

        # Focus shifts to this conversation. (Salience-driven switching
        # across multiple foci is a later increment; here every message
        # captures attention.)
        new_focus = str(msg.conversationId) if msg.conversationId else None
        if new_focus != self._workspace.focus:
            self._publish("focus_switch", {"from": self._workspace.focus, "to": new_focus})
            self._workspace.focus = new_focus

        nt_at_start: NTState = app.state.nt_state

        # Resolve the speaker. The connector always supplies a real
        # personId; a UI turn may post NIL_PERSON_ID to mean "this is the
        # operator" — resolve it from identity then.
        identity = getattr(app.state, "identity", None)
        operator_person_id = None
        if identity is not None and msg.personId == NIL_PERSON_ID:
            try:
                operator_person_id = await turn.resolve_operator_person_id(tool_runner)
            except httpx.HTTPError as e:
                log.warning(
                    "identity unreachable resolving operator: %s (degrading to no-person-context)",
                    e,
                )
        effective_person_id = (
            msg.personId if msg.personId != NIL_PERSON_ID else (operator_person_id or NIL_PERSON_ID)
        )

        # Load history + persist the incoming message. Memory unreachable
        # is fatal for this turn (no context, no durable trace) — log and
        # drop; the loop survives.
        try:
            if msg.conversationId is not None:
                existing = await memory.get(msg.conversationId)
                if existing is None:
                    conversation_id = await memory.create()
                    history: list[Message] = []
                else:
                    conversation_id = msg.conversationId
                    history = list(existing.messages)
            else:
                conversation_id = await memory.create()
                history = []

            user_message = Message(role=Role.user, content=msg.content)
            await tool_runner.run(
                new_call(TOOL_MEMORY_APPEND_ENTRY),
                ToolContext(
                    inputs={
                        "conversation_id": conversation_id,
                        "entry": turn.build_memory_entry(
                            conversation_id=conversation_id,
                            person_id=effective_person_id,
                            message=user_message,
                            nt_snapshot=nt_at_start,
                        ),
                    }
                ),
            )
        except httpx.HTTPError as e:
            log.warning(
                "memory unreachable handling message %s: %s — dropping turn",
                event.eventId,
                e,
            )
            return

        # Recent turns with this person (relationship-context enrichment).
        # Explicit None-check, not `or`: personRecentLimit has minimum=0 and
        # "0 disables" is documented — `or 30` would silently rewrite a
        # deliberate 0 to 30. A non-positive limit skips the lookup entirely
        # (the operator turned relationship-context injection OFF).
        limit_cfg = store.get("personRecentLimit")
        person_recent_limit = int(limit_cfg) if limit_cfg is not None else 30
        recent_with_person: list = []
        if effective_person_id != NIL_PERSON_ID and person_recent_limit > 0:
            try:
                invocation = await tool_runner.run(
                    new_call(TOOL_MEMORY_PERSON_RECENT),
                    ToolContext(
                        inputs={
                            "person_id": effective_person_id,
                            "limit": person_recent_limit,
                        }
                    ),
                )
                recent_with_person = [
                    e for e in invocation.payload if e.conversationId != conversation_id
                ]
            except httpx.HTTPError as e:
                log.warning("person_recent unreachable: %s (continuing without it)", e)

        cross_pass_framing = str(store.get("crossPassFraming") or "parallel_thread")
        try:
            system_prompts = await turn.build_per_driver_system_prompts(
                drivers=drivers,
                tool_runner=tool_runner,
                person_id=(None if msg.personId == NIL_PERSON_ID else msg.personId),
                operator_person_id=operator_person_id,
                fallback_default=str(store.get("defaultSystemPrompt") or ""),
                recent_turns=recent_with_person,
                cross_pass_framing=cross_pass_framing,
            )
        except httpx.HTTPError as e:
            log.warning("identity unreachable assembling prompts: %s (using default)", e)
            fallback = str(store.get("defaultSystemPrompt") or "")
            system_prompts = {drivers[0].name: fallback, drivers[1].name: fallback}

        history_for_drivers = turn.build_history(history, msg.content)

        # Deliberation depth is owned by the plateau-stop gate (a noisy
        # dopamine-RPE accumulator), NOT a fixed pass count. `defaultMaxPasses`
        # is the runaway cost fuse only. Temperature stays NT-modulated (a
        # parameter, not a limit). `agreementThreshold` no longer terminates
        # the loop — it survives only to (a) center the post-turn dopamine
        # impulse and (b) key the calm-vs-stress GABA/cortisol impulse, both
        # via build_observations below. It does NOT feed the voice register
        # bands (those use absolute cutoffs; see voice.py::_agreement_directive).
        # Explicit None-check: agreementThreshold has minimum=0.0, and a
        # configured 0.0 ("treat every bout as settled") is meaningful — `or`
        # would silently rewrite it to 0.5.
        max_passes_cfg = store.get("defaultMaxPasses")
        max_passes = int(max_passes_cfg) if max_passes_cfg is not None else 8
        threshold_cfg = store.get("agreementThreshold")
        agreement_threshold = float(threshold_cfg) if threshold_cfg is not None else 0.5
        temp_cfg = store.get("defaultTemperature")
        temperature = modulated_temperature(
            nt_at_start, float(temp_cfg) if temp_cfg is not None else None
        )
        max_tokens_cfg = store.get("defaultMaxTokens")
        max_tokens = int(max_tokens_cfg) if max_tokens_cfg is not None else None

        # Build the plateau gate for this bout. Explicit None-checks (not
        # `or`): a configured 0.0 is a MEANINGFUL value for these gains
        # (e.g. disable the improvement/valence coupling or the noise), and
        # `x or default` would silently treat 0.0 as unset. Seed: null in
        # production (OS entropy → real stochasticity); a fixed int makes
        # the noisy stop reproducible (tests / clamp-and-sample debugging).
        def _gate_float(key: str, default: float) -> float:
            value = store.get(key)
            return float(value) if value is not None else default

        seed_cfg = store.get("plateauSeed")
        gate = BoutGate(
            PlateauParams(
                base_drift=_gate_float("plateauBaseDrift", 1.0),
                rpe_gain=_gate_float("plateauRpeGain", 3.0),
                valence_gain=_gate_float("plateauValenceGain", 0.5),
                # 0.1, not 0.0: "noise on (low)" is the locked default, so the
                # code-level fallback must match the config default — a missing
                # key must not silently make the gate deterministic.
                noise_sigma=_gate_float("plateauNoiseSigma", 0.1),
            ),
            random.Random(int(seed_cfg) if seed_cfg is not None else None),
        )

        # THINK — the deliberative flavor of a thought is the bicameral pair.
        try:
            outcome = await run_bicameral_loop(
                history=history_for_drivers,
                system_prompts=system_prompts,
                drivers=drivers,
                nt_state=nt_at_start,
                max_passes=max_passes,
                gate=gate,
                temperature=temperature,
                max_tokens=max_tokens,
                scorer=scorer,
                cross_pass_framing=cross_pass_framing,
            )
        except (DriverError, httpx.HTTPError) as e:
            log.warning("deliberation failed for message %s: %s — no reply", event.eventId, e)
            return

        for pass_record in outcome.passes:
            self._publish("thought", pass_record.model_dump(mode="json", exclude_none=True))

        # EVOLVE NT from the bout's observations (the `internal` tool) BEFORE
        # the gate — Eugene thought either way, so his affect updates whether
        # or not he ends up speaking, and the action gate must read this
        # POST-bout state to decide how he feels having thought it through. A
        # failure here must NOT abort the turn — fall back to the pre-turn
        # state; the gate (and any reply) still proceed. NTState carries a
        # required `lastUpdated` datetime, so the publish uses mode="json"
        # like every other event — without it `json.dumps` in the SSE route
        # would raise and kill the subscriber's stream.
        observations = turn.build_observations(outcome, agreement_threshold)
        nt_at_end: NTState = nt_at_start
        try:
            nt_invocation = await tool_runner.run(
                new_call(TOOL_NT_OBSERVE),
                ToolContext(inputs={"state": nt_at_start, "observations": observations}),
            )
            nt_at_end = nt_invocation.payload
            app.state.nt_state = nt_at_end
            self._publish("nt_update", nt_at_end.model_dump(mode="json"))
        except Exception:
            log.exception(
                "NT observe failed for message %s — keeping prior state, gate still decides",
                event.eventId,
            )

        # GATE — the action SELECTOR: speak vs stay silent, chosen by
        # anticipated net NT valence (a seedable softmax sample). Being
        # addressed is a strong innate drive toward SPEAK, but a sufficiently
        # aversive post-bout state can tip Eugene to silence — emergent, not
        # legislated; there is no "addressed → always reply" rule. `addressed`
        # is True here because a message captured his attention; self-
        # initiated speech (addressed=False, idle mind-wandering) is a later
        # increment. THINK_MORE / SWITCH / SLEEP join the candidate set the
        # same way without changing the mechanism.
        action_seed = store.get("actionSeed")
        choice = select_action(
            valence=net_valence(nt_at_end),
            addressed=True,
            params=ActionPolicyParams(
                response_drive=_gate_float("actionResponseDrive", 0.6),
                engagement_gain=_gate_float("actionEngagementGain", 0.5),
                idle_floor=_gate_float("actionIdleFloor", 0.0),
                selection_temperature=_gate_float("actionSelectionTemperature", 0.15),
            ),
            rng=random.Random(int(action_seed) if action_seed is not None else None),
        )
        self._publish_gate(
            choice.action.value, focus=new_focus, anticipated_valence=choice.anticipated_valence
        )

        if choice.action is not Action.SPEAK:
            # Silence is a real outcome: Eugene thought, his NT evolved, and
            # he chose not to emit. No voice pass, no reply persisted, no
            # EfferentSpeechAct. The perception + NT-evolve tools still ran,
            # so surface their trace for the observability stream.
            log.info(
                "gate chose %s over speak (valence=%.3f, p_speak=%.2f) for message %s — silent",
                choice.action.value,
                net_valence(nt_at_end),
                choice.probabilities.get(Action.SPEAK, 0.0),
                event.eventId,
            )
            self._flush_trace(trace)
            return

        # SPEAK — the voice pass IS the speech effector. (Voice register reads
        # nt_at_start — Eugene's affect entering the turn — unchanged from
        # v0.2; unifying it onto nt_at_end is a clean follow-up.)
        voice_driver = turn.resolve_voice_driver(drivers, store.get("voiceDriver"))
        voice_temp_cfg = store.get("voiceTemperature")
        voice_temperature = float(voice_temp_cfg) if voice_temp_cfg is not None else temperature
        deliberation_finals = outcome.passes[-1].hemispheres if outcome.passes else []
        final_agreement = outcome.passes[-1].callosum.agreement if outcome.passes else 0.5
        voice_persona = system_prompts.get(voice_driver.name) or next(
            iter(system_prompts.values()), ""
        )
        try:
            voice_outcome = await run_voice_pass(
                voice_driver=voice_driver,
                user_message=Message(role=Role.user, content=msg.content),
                history=history,
                system_prompt=voice_persona,
                deliberation_finals=list(deliberation_finals),
                final_agreement=final_agreement,
                nt_state=nt_at_start,
                temperature=voice_temperature,
                max_tokens=max_tokens,
            )
        except (DriverError, httpx.HTTPError) as e:
            log.warning("voice pass failed for message %s: %s — no reply", event.eventId, e)
            return

        final_message = voice_outcome.output

        # Persist the reply. A persistence failure here is non-fatal — the
        # reply was already decided; log and still speak.
        try:
            await tool_runner.run(
                new_call(TOOL_MEMORY_APPEND_ENTRY),
                ToolContext(
                    inputs={
                        "conversation_id": conversation_id,
                        "entry": turn.build_memory_entry(
                            conversation_id=conversation_id,
                            person_id=effective_person_id,
                            message=final_message,
                            nt_snapshot=nt_at_end,
                            hemisphere_attribution="voice",
                        ),
                    }
                ),
            )
        except httpx.HTTPError as e:
            log.warning("memory unreachable persisting reply to %s: %s", event.eventId, e)

        # Surface the turn's tool trace on the stream (the M0.5 debug lens,
        # now live rather than bundled into a response).
        self._flush_trace(trace)

        # SPEAK effector — emit the utterance. Destination mirrors the
        # afferent source; `inResponseTo` correlates it to the event. For
        # now the speech leaves on the consciousness stream; connector
        # delivery to external channels is a later slice.
        speech = EfferentSpeechAct(
            destination=msg.source,
            content=final_message.content,
            inResponseTo=event.eventId,
            conversationId=conversation_id,
            timestamp=datetime.now(UTC),
        )
        self._publish("speech", speech.model_dump(mode="json", exclude_none=True))

    # -- observability -------------------------------------------------

    def _publish(self, event_type: str, data: dict) -> None:
        self._broker.publish(event_type, data)

    def _flush_trace(self, trace: list[ToolInvocationRecord]) -> None:
        """Publish the turn's accumulated tool-invocation records.

        Called on BOTH the speak and the silent path — perception and NT
        evolution ran regardless of whether Eugene chose to reply, so their
        trace belongs on the stream either way.
        """
        for record in trace:
            self._publish("tool_call", record.model_dump(mode="json", exclude_none=True))

    def _publish_gate(
        self, action: str, *, focus: str | None, anticipated_valence: float | None = None
    ) -> None:
        decision = GateDecision.model_validate(
            {"action": action, "focus": focus, "anticipatedValence": anticipated_valence}
        )
        self._broker.publish("gate_decision", decision.model_dump(mode="json", exclude_none=True))
