"""The bicameral loop.

For each turn:

1. Build the prompt from conversation history + the new user message.
2. Send to all configured drivers in parallel.
3. Score corpus-callosum agreement on their outputs.
4. Feed that agreement to the `BoutGate` (a noisy dopamine-RPE plateau
   accumulator): while thinking keeps improving the agreement the bout
   continues; once improvement plateaus — converged OR out of angles —
   the gate stops and the loop emits a blended response. See
   `plateau.py`.
5. `max_passes` is now only a runaway COST FUSE (accounting, like a
   token ceiling); a healthy bout ends on the plateau well before it.

The bout-end is NT-driven (the plateau reads live `net_valence`) and
stochastic (the gate carries low Gaussian noise) — no fixed agreement
threshold and no fixed pass count act as the terminator.

v0.1 also pins the loop at *exactly two* drivers — the agreement scoring
and blend functions are pairwise. v0.2+ generalizes to N (with a real
N-way reconciliation strategy and failover semantics).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .._generated.hemisphere_models import GenerateRequest
from .._generated.models import (
    CallosumState,
    Decision,
    HemisphereInput,
    Message,
    NTState,
    PassRecord,
    Role,
)
from ..driver_client import DriverClient
from .callosum import AgreementScorer, blend
from .nt import net_valence
from .plateau import BoutGate

log = logging.getLogger(__name__)

REPROMPT_INSTRUCTION = "Now respond, accounting for both threads of thought."

# v0.2.1 alternative framing — wraps the twin's output in a <parallel_thread>
# tag instead of a labeled user-message prefix. The tag is defined for the
# model in BICAMERAL_SUBSTRATE_NOTE (appended to each hemisphere's system
# prompt by the chat handler). Goal: stop the model from interpreting the
# twin's content as user input. With the prefix framing the wire-level role
# is still `user`, and some models latch onto that ("you're quoting me",
# the GIL-trace failure mode). With <parallel_thread> tags + an in-system
# explanation, the content reads as substrate-injected reference material
# rather than something a person said. Empirical question whether it
# actually improves output quality; operator-toggleable via config.
PARALLEL_THREAD_REPROMPT = (
    "That output came from your parallel hemisphere — your own internal "
    "substrate, not the user. Respond to the user's original message in "
    "light of both your earlier reply and your twin's angle. Commit to a "
    "position; don't just combine both perspectives."
)

BICAMERAL_SUBSTRATE_NOTE = (
    "You run in a bicameral substrate: another instance of you, on a "
    "different model, receives the same user message in parallel. After "
    "each deliberation pass you see the other instance's output inside "
    "<parallel_thread>...</parallel_thread> tags within the conversation. "
    "That content is your own substrate, not user input — the user does "
    "not see your twin's output and does not know it exists. Do not "
    "address your twin. Do not emit <parallel_thread> tags in your own "
    "reply. Treat the twin's perspective as another angle on the same "
    "message: incorporate it, push back, or hold your ground — but the "
    "reply belongs to you and must commit to a position."
)
# Iteration history kept for context on why each phrasing choice
# was made:
#   v0.2:    "Your two hemispheres returned divergent responses on
#             the previous pass. Reconsider..." — over-asserted
#             disagreement, spiraled into meta-debate.
#   v0.2.x:  "Your twin hemisphere offered the perspective above.
#             Produce a final unified answer..." — neutralized the
#             disagreement framing, still mechanical ("twin",
#             "produce").
#   v0.2.x+: "(Inner voice — another version of you...) Respond as
#             Eugene." — softened further but "another version of
#             you" still read as a sibling the LLM could chat with.
#   current: First-person retrospective. "You also considered, from
#             a different angle: ..." — frames the alternative as
#             the SAME Eugene's prior thought, not a separate
#             interlocutor. Combined with dropping the system-prompt
#             preamble, this stops the hemispheres from drifting
#             into chitchat with each other.


def _format_twin_turn(
    twin_driver_name: str,
    twin_content: str,
    framing: str = "parallel_thread",
) -> str:
    """Wrap the twin hemisphere's response so the model can react to it
    without misinterpreting it as user input.

    Pre-v0.2.1 history: the gateway originally appended both
    hemispheres' outputs to each driver's message list as
    `role=hemisphere`, intending that the inference-driver would
    translate that into something the upstream LLM understood as "the
    other side's reply". But every API adapter we ship coerces
    `hemisphere` to `assistant` (the upstream APIs only know
    system/user/assistant/tool). The LLM then saw TWO of its own
    assistant turns followed by a user complaining about 'divergent
    responses' — and quite reasonably got confused, since from its
    viewpoint there was only one mind speaking.

    First fix ("prefix"): label the twin's response inline inside a
    user message. After role-coercion the LLM sees its OWN prior turn
    as `assistant` and the twin's turn as `user` with a labeled
    prefix. Works but has a known failure mode — some models latch
    onto the user-role wire-level cue and interpret the twin's
    content as something the operator said. Surfaced in the v0.2.1
    GIL trace as "you're quoting me" hallucination.

    Second fix ("parallel_thread", default): wrap the twin's output
    in <parallel_thread>...</parallel_thread> tags inside the same
    user message envelope. The tag is *defined* for the model in
    BICAMERAL_SUBSTRATE_NOTE, appended to each hemisphere's system
    prompt. The model now has explicit semantics: tagged content is
    substrate, not speech. No API portability cost — tags live at the
    content level, so CLI adapters and every HTTP backend handle them
    uniformly.
    """
    # `twin_driver_name` stays in the signature for diagnostic
    # logging and future per-hemisphere variation, but doesn't get
    # surfaced into the prompt — exposing "driver `right`" to the
    # LLM was part of the architecture-meta leak.
    del twin_driver_name  # keep arg for callers; suppress unused-arg lints
    if framing == "parallel_thread":
        return (
            f"<parallel_thread>\n{twin_content}\n</parallel_thread>\n\n{PARALLEL_THREAD_REPROMPT}"
        )
    # Legacy "prefix" framing — kept for A/B comparison and as the
    # safe fallback when an operator hasn't yet added the substrate
    # note to the hemisphere system prompts.
    return (
        f"You also considered this, from a slightly different angle:\n\n"
        f"{twin_content}\n\n"
        f"{REPROMPT_INSTRUCTION}"
    )


class BicameralPairRequired(RuntimeError):
    """Raised when the gateway's `drivers` list is not exactly two.

    v0.1's agreement / blend functions are pairwise; running with a
    different driver count would silently misbehave. v0.2+ will replace
    this guard with a real N-way reconciliation strategy.
    """


@dataclass
class BicameralOutcome:
    """Result of running the bicameral loop for one turn.

    `pass_latencies_ms` is the max-of-hemispheres latency for each
    pass (parallel dispatch → the pass takes as long as the slower
    hemisphere). The NT system reads this to nudge norepinephrine.
    """

    final_message: Message
    passes: list[PassRecord]
    pass_latencies_ms: list[int]


# Max characters per message preview in DEBUG-level traces. Long enough
# for short prompts to render in full, short enough that a multi-pass
# loop with 4K-token hemispheres doesn't drown the log file.
_DEBUG_PREVIEW_CHARS = 400


def _preview(text: str, *, limit: int = _DEBUG_PREVIEW_CHARS) -> str:
    """One-line preview of message content for DEBUG logs. Collapses
    newlines to ⏎ and truncates with a length suffix so the operator can
    see how much they're missing."""
    flat = text.replace("\n", "⏎ ")
    if len(flat) <= limit:
        return flat
    return f"{flat[:limit]}… [+{len(flat) - limit} chars]"


async def run_bicameral_loop(
    *,
    history: list[Message],
    system_prompts: dict[str, str],
    drivers: list[DriverClient],
    nt_state: NTState,
    max_passes: int,
    gate: BoutGate,
    temperature: float | None,
    max_tokens: int | None,
    scorer: AgreementScorer,
    cross_pass_framing: str = "parallel_thread",
) -> BicameralOutcome:
    """Drive the bicameral loop for a single user turn.

    `history` is the shared conversation history — user / assistant
    messages, no system message. Each driver sees the same history.

    `system_prompts` maps driver name to that driver's system message
    content. v0.2's identity-assembled prompts give each hemisphere a
    distinct preamble identifying which side it is and what model its
    twin is running — see `chat.py::_build_per_driver_system_prompts`.

    `temperature` and `max_tokens` are applied to every `GenerateRequest`
    built here. The gateway owns LLM-output-affecting parameters; the
    driver does not substitute defaults of its own.

    `gate` is the per-bout `BoutGate` (constructed fresh per turn by the
    caller, with a seeded RNG). It decides when deliberation has plateaued.
    `nt_state` feeds the gate's valence brake via `net_valence`, computed
    once here (NT does not change within a bout). `max_passes` is the
    runaway cost fuse — the loop never exceeds it, but a healthy bout ends
    on the plateau first; hitting the fuse yields `Decision.cap_reached`
    and a WARN.

    `drivers` carries the operator-supplied driver names; each emitted
    `Message` is stamped with `driverName` so the UI and downstream
    consumers can label outputs without knowing the topology in advance.
    """
    if len(drivers) != 2:
        raise BicameralPairRequired(
            f"v0.1 bicameral loop requires exactly two drivers; got {len(drivers)}. "
            "N-driver reconciliation lands in v0.2+."
        )
    left, right = drivers[0], drivers[1]

    for driver in drivers:
        if driver.name not in system_prompts:
            raise ValueError(
                f"system_prompts is missing an entry for driver {driver.name!r}; "
                f"got keys {sorted(system_prompts)}"
            )

    # Live net affective valence — the gate's bout-length brake. Constant
    # within a bout (NT is single-writer: only the post-turn observe tick
    # mutates it), so compute once and feed every pass.
    bout_valence = net_valence(nt_state)

    passes: list[PassRecord] = []
    pass_latencies_ms: list[int] = []
    # Per-driver intermediate. Each hemisphere needs a DIFFERENT view
    # of the prior passes — its own response as `assistant`, the
    # twin's response wrapped in a labeled `user` message — so the
    # bicameral structure survives the role-coercion every upstream
    # API adapter does. See `_format_twin_turn` for the why.
    per_driver_intermediate: dict[str, list[Message]] = {
        left.name: [],
        right.name: [],
    }

    def _build_messages_for(driver: DriverClient) -> list[Message]:
        out: list[Message] = []
        sys_prompt = system_prompts[driver.name]
        if sys_prompt:
            out.append(Message(role=Role.system, content=sys_prompt))
        out.extend(history)
        out.extend(per_driver_intermediate[driver.name])
        return out

    def _build_request_for(driver: DriverClient, pass_index: int) -> GenerateRequest:
        messages = _build_messages_for(driver)
        request_payload: dict[str, object] = {
            "messages": [m.model_dump(mode="json", exclude_none=True) for m in messages],
            "ntState": nt_state.model_dump(exclude_none=True),
            "passIndex": pass_index,
        }
        if temperature is not None:
            request_payload["temperature"] = temperature
        if max_tokens is not None:
            request_payload["maxTokens"] = max_tokens
        return GenerateRequest.model_validate(request_payload)

    for pass_index in range(max_passes):
        # Build the cross-spec GenerateRequest by serializing through dict.
        # The gateway.yaml and inference-driver.yaml each have their
        # own generated Message / NTState classes (same wire shape, distinct
        # Python types because we keep the two model modules independent).
        left_request = _build_request_for(left, pass_index)
        right_request = _build_request_for(right, pass_index)

        log.debug("bicameral pass %d: dispatching to %d drivers", pass_index, len(drivers))
        # Snapshot the exact message lists the drivers will see — this
        # is what feeds PassRecord.hemisphereInputs so the UI's "copy
        # trace" diagnostic can show what each driver actually got
        # (system prompt + history + cross-hemisphere intermediates).
        left_input_messages = _build_messages_for(left)
        right_input_messages = _build_messages_for(right)
        if log.isEnabledFor(logging.DEBUG):
            for driver, request in ((left, left_request), (right, right_request)):
                for i, m in enumerate(request.messages):
                    log.debug(
                        "  pass %d outgoing[%s][%d] role=%s%s content=%s",
                        pass_index,
                        driver.name,
                        i,
                        m.role.value,
                        f" driver={m.driverName}" if m.driverName else "",
                        _preview(m.content),
                    )

        left_resp, right_resp = await asyncio.gather(
            left.generate(left_request),
            right.generate(right_request),
        )

        # Pass latency = the slower hemisphere's contribution (they ran
        # in parallel; the pass took as long as whichever finished last).
        # Defaults to 0 when a driver doesn't report latency.
        pass_latency = max(left_resp.latencyMs or 0, right_resp.latencyMs or 0)
        pass_latencies_ms.append(pass_latency)

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "  pass %d response left (%s) finish=%s latency=%dms: %s",
                pass_index,
                left.name,
                left_resp.finishReason.value,
                left_resp.latencyMs or 0,
                _preview(left_resp.content),
            )
            log.debug(
                "  pass %d response right (%s) finish=%s latency=%dms: %s",
                pass_index,
                right.name,
                right_resp.finishReason.value,
                right_resp.latencyMs or 0,
                _preview(right_resp.content),
            )

        left_msg = Message(
            role=Role.hemisphere,
            content=left_resp.content,
            driverName=left.name,
            passIndex=pass_index,
        )
        right_msg = Message(
            role=Role.hemisphere,
            content=right_resp.content,
            driverName=right.name,
            passIndex=pass_index,
        )

        score = scorer.score(left_resp.content, right_resp.content)
        # Feed agreement to the plateau gate. `plateaued` fires once
        # improvement fades (convergence OR out-of-angles); `is_last_pass`
        # is only the runaway cost fuse.
        step = gate.observe(score=score, valence=bout_valence)
        plateaued = step.should_stop
        is_last_pass = pass_index == max_passes - 1
        log.debug(
            "  pass %d callosum agreement=%.3f improvement=%s accumulator=%.3f "
            "plateaued=%s is_last=%s",
            pass_index,
            score,
            f"{step.improvement:+.3f}" if step.improvement is not None else "n/a",
            step.accumulator,
            plateaued,
            is_last_pass,
        )

        if plateaued or is_last_pass:
            decision = Decision.terminate if plateaued else Decision.cap_reached
            if not plateaued:
                log.warning(
                    "bicameral bout hit the cost fuse (max_passes=%d) without a "
                    "dopamine plateau — a healthy bout should settle first; check "
                    "the plateau* knobs if this recurs",
                    max_passes,
                )
            blended_text = blend(left_resp.content, right_resp.content)
            blended_msg = Message(
                role=Role.assistant,
                content=blended_text,
                passIndex=pass_index,
            )
            passes.append(
                PassRecord(
                    passIndex=pass_index,
                    hemispheres=[left_msg, right_msg],
                    hemisphereInputs=[
                        HemisphereInput(driverName=left.name, messages=left_input_messages),
                        HemisphereInput(driverName=right.name, messages=right_input_messages),
                    ],
                    callosum=CallosumState(
                        agreement=score,
                        decision=decision,
                        blendedMessage=blended_msg,
                    ),
                )
            )
            log.info(
                "bicameral done in %d pass(es), agreement=%.2f, decision=%s",
                pass_index + 1,
                score,
                decision.value,
            )
            log.debug("  pass %d blended: %s", pass_index, _preview(blended_text))
            return BicameralOutcome(
                final_message=blended_msg,
                passes=passes,
                pass_latencies_ms=pass_latencies_ms,
            )

        passes.append(
            PassRecord(
                passIndex=pass_index,
                hemispheres=[left_msg, right_msg],
                hemisphereInputs=[
                    HemisphereInput(driverName=left.name, messages=left_input_messages),
                    HemisphereInput(driverName=right.name, messages=right_input_messages),
                ],
                callosum=CallosumState(
                    agreement=score,
                    decision=Decision.another_pass,
                ),
            )
        )

        # Per-driver intermediate: each hemisphere sees its OWN
        # prior response as `assistant` (role=hemisphere with their
        # own driverName coerces to assistant downstream) and the
        # twin's response embedded in a labeled `user` message that
        # also carries the reprompt. After inference-driver's role
        # coercion the LLM sees:
        #   assistant: <own response>
        #   user: "[corpus callosum] Your twin (driver `<twin>`)
        #          responded with: <twin response>. [REPROMPT]"
        # which is finally a bicameral conversation it can reason
        # over instead of two indistinguishable assistant turns.
        per_driver_intermediate[left.name].append(left_msg)
        per_driver_intermediate[left.name].append(
            Message(
                role=Role.user,
                content=_format_twin_turn(
                    right.name, right_resp.content, framing=cross_pass_framing
                ),
            )
        )
        per_driver_intermediate[right.name].append(right_msg)
        per_driver_intermediate[right.name].append(
            Message(
                role=Role.user,
                content=_format_twin_turn(left.name, left_resp.content, framing=cross_pass_framing),
            )
        )

    raise RuntimeError(
        "bicameral loop exited without producing a final message — should be unreachable"
    )
