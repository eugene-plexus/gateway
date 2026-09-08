"""The voice pass — post-deliberation user-facing reply generation.

The bicameral deliberation loop in `loop.py` produces hemispheres
that talk to each other to reconcile. That works for internal
deliberation but produces a register problem: by pass 1+, each
hemisphere is responding to its OTHER half ("Glad we're on the same
page," "Thanks for not contradicting me"), not to the actual user.
The last hemisphere output therefore isn't appropriate to show as
Eugene's reply — it's deliberation overheard, not communication.

The voice pass is one additional LLM call after the deliberation
loop terminates. It receives:
  - Eugene's persona prompt (the same one used during deliberation)
  - The conversation history
  - The user's current message
  - An inline summary of "what you just considered" — both
    hemispheres' final deliberated content
  - A directive: "now respond to the user. Not to yourself."

The voice pass's output IS what the user sees. The hemispheres'
raw outputs stop being user-facing — they become internal
artifacts the voice pass draws from. This also implicitly solves
the `<think>`-leak risk (the voice pass regenerates clean text
regardless of how messy the deliberation got).

v0.2.x: voice pass always runs. Operator picks which driver does
it via `voiceDriver` config. v0.3+ can add NT modulation of voice
temperature, multi-voice (chorus), or skip-the-voice-pass under
certain NT states.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .._generated.hemisphere_models import GenerateRequest
from .._generated.models import (
    Message,
    NTState,
    Role,
)
from ..driver_client import DriverClient

log = logging.getLogger(__name__)

VOICE_PASS_SCRATCHPAD_SUFFIX = (
    "\n\n---\n\n"
    "You are in VOICE mode. The conversation history (if any) is in "
    "the message turns above. The user has just sent you ONE new "
    "message — the last message below. You already deliberated about "
    "that one new message privately; that internal phase is over. "
    "Your job now is the single reply the user will read. Not more "
    "thinking. Not a summary of what you thought. The reply.\n\n"
    "Your private deliberation notes about the user's latest message:\n\n"
    "{deliberation_summary}\n\n"
    "These notes are internal scratch. They are NOT conversation. The "
    "user did not say any of this, you did not say any of this to the "
    "user, and the user has no idea these notes exist. The notes may "
    'appear to reference each other ("you caught the split", "you\'re '
    'right about the echo", "both") — that is parallel-pass '
    "cross-talk between your two trains of thought, not back-and-forth "
    "with the user. Do not respond as if any of this content was "
    'previously exchanged with the user. Words like "still", '
    '"again", "returning to that", "as you said", "like we '
    'discussed" imply exchanges that did not happen — do not use them '
    "based on the notes. Do not quote, recite, paraphrase, or address "
    "the notes themselves.\n\n"
    "---\n\n"
    "{agreement_directive}\n\n"
    "---\n\n"
    "Banned outputs — every one of these is a failure mode. The "
    "model that produced this skill has strong defaults toward "
    "several of them; override every default:\n\n"
    '- Narrating your deliberation back at the user ("I was '
    'thinking about...", "part of me wanted to...", "on one '
    'hand... on the other..."). The deliberation is private. The '
    "user wants a reply, not a tour of your thought process.\n"
    '- Helpful-assistant openers ("Great question", "I\'d be happy '
    'to help", "Let me explain", "That\'s a thoughtful point"). '
    "Real people do not preface their responses with customer-"
    "service phrases.\n"
    '- Cataloguing options for the user to pick from ("There are a '
    'few ways to look at this", "It depends on what you mean"). '
    "Pick one. Commit.\n"
    "- Reciting back what the user just said before responding. "
    "They were there. They know what they said.\n"
    "- Padding to seem thorough or balanced. Length is not value. "
    "The shortest response that says the thing IS the correct "
    "response.\n"
    "- Asking clarifying questions when the message is clear enough "
    "to react to. React first; ask only if you genuinely cannot "
    "proceed.\n\n"
    "How to respond:\n\n"
    "React first. If the message is confusing, gibberish, a "
    "non-sequitur, weird, hostile, or a strange opener — react to "
    "THAT, briefly. Real people do not owe strangers a coherent "
    "on-topic answer. Be confused, dismissive, playful, defensive, "
    "curious, terse, amused, or thrown off the rails. Pick what "
    "actually fits and commit to it.\n\n"
    "Load-bearing rule: the first three responses a helpful-"
    "assistant model would generate to this message are off-limits. "
    "Push past them into the actual reaction a person would have."
)
# v0.2.1 directive rewrite (post-v0.2.0 tag). Three changes layered
# onto the v0.2.x scratchpad-as-system-message structure:
#
#   1. Explicit "VOICE mode" framing — names the phase. The model is
#      not deliberating; the model is producing the reply. Naming the
#      phase is mechanical role assignment, which carries across
#      RLHF distributions more reliably than aspirational "be a
#      person" language.
#
#   2. Banned outputs section — names specific failure modes with
#      one-line "this is wrong because" reasons. Negative-space
#      directives ("don't do X") are more actionable than positive
#      ones ("be terse"), because the model knows what "X" looks
#      like in its own output distribution.
#
#   3. Load-bearing language for the "first three responses banned"
#      directive. The construction is borrowed from divergent-
#      ideation skill design — it specifies negative space rather
#      than positive aspiration, which is sharper than "be creative"
#      or "don't sound like an assistant."
#
# Still presumes "stranger, first turn, weird-or-normal opener"
# social context. v0.3 work: variant selection based on (a) person
# familiarity from identity, (b) conversation turn count, (c) NT
# state. For now we are validating whether sharpening the directive
# language meaningfully moves the helpful-assistant ceiling for the
# same hemispheres + same architecture — which is the test for
# whether the bicameral inner-thought-process is producing genuine
# value over a single-model baseline.


@dataclass(frozen=True)
class VoicePassOutcome:
    """Result of running the voice pass for one turn."""

    driver_name: str
    input_messages: list[Message]
    output: Message
    latency_ms: int


def _agreement_directive(final_agreement: float) -> str:
    """Map the deliberation's final-pass agreement score to a register
    directive for the voice pass.

    This is the architectural payoff of the bicameral loop: the
    agreement score is a measure of how aligned Eugene's two trains
    of thought were on this turn, and the voice pass surfaces that
    internal state as register / tone in the user-facing reply. Eugene
    who agreed with himself sounds confident; Eugene who didn't sounds
    of-two-minds. A single-driver agent has no analogous axis.

    Three bands — high (≥0.75), mid (0.4-0.75), low (<0.4). Bands are
    absolute, not relative to the configured agreement threshold,
    because the threshold is a *termination* decision (\"stop
    deliberating\") and the bands are a *register* decision (\"how
    certain do I sound\"). Conflating them would couple two separate
    knobs in confusing ways.
    """
    if final_agreement >= 0.75:
        return (
            "Internal-state directive: your two trains of thought "
            "converged on this. This is what internal certainty feels "
            'like. Speak with conviction. Cut hedges — "I think", '
            '"maybe", "perhaps", "could be" are off-limits unless '
            "your reaction itself is genuinely uncertain about something "
            "the deliberation didn't settle. Confidence in the reply "
            "should match the confidence in the deliberation."
        )
    if final_agreement >= 0.4:
        return (
            "Internal-state directive: your two trains of thought "
            "mostly agree but with real friction. This is what genuine-"
            "but-imperfect certainty feels like. Some hedging is "
            "appropriate; do not oversell what you actually know. If "
            "you have a real take, commit to it — but allow that you "
            "might be wrong about the edges."
        )
    return (
        "Internal-state directive: your two trains of thought did NOT "
        "agree on this. You are genuinely of two minds. Let it show in "
        "the reply — be tentative, conflicted, terse, ambivalent, "
        "skeptical, or evasive. Honest uncertainty is more authentic "
        "than synthesized confidence. You do not owe the user a "
        "definitive answer when you don't have one. A short, "
        "noncommittal reaction is the correct shape here — not a long "
        "balanced essay covering both perspectives. Real people under "
        "internal conflict get terse, not expansive."
    )


def _format_deliberation_summary(deliberation_finals: list[Message]) -> str:
    """Render the deliberation's last-pass outputs as a labeled
    bullet list for the voice pass's prompt.

    `deliberation_finals` is one Message per hemisphere — the LAST
    response each side produced during the loop. Their driverName
    is used as a label so the voice pass sees which thread is which,
    even though the user-facing reply won't mention them.
    """
    if not deliberation_finals:
        return "(no deliberation summary — direct voice pass)"
    lines: list[str] = []
    for msg in deliberation_finals:
        label = msg.driverName or "thread"
        lines.append(f"- ({label}) {msg.content.strip()}")
    return "\n".join(lines)


async def run_voice_pass(
    *,
    voice_driver: DriverClient,
    user_message: Message,
    history: list[Message],
    system_prompt: str,
    deliberation_finals: list[Message],
    final_agreement: float,
    nt_state: NTState,
    temperature: float | None,
    max_tokens: int | None,
) -> VoicePassOutcome:
    """Convert deliberation into Eugene's user-facing reply.

    The voice pass uses the SAME persona/system prompt as deliberation
    (Eugene-as-a-person, not Eugene-the-assistant). What's different
    is the LAST user-role message: it carries an inline summary of the
    just-completed deliberation and instructs the model to address the
    actual user.

    `final_agreement` is the agreement score from the last deliberation
    pass — used to shape the register directive injected into the
    scratchpad. This is how the bicameral loop's "did Eugene's two
    minds agree?" signal becomes audible in his actual reply.

    `history` is the conversation history WITHOUT the current
    `user_message` — the current message is appended after history,
    then the voice directive is bracketed in a final user message.
    """
    summary = _format_deliberation_summary(deliberation_finals)
    directive = _agreement_directive(final_agreement)
    # Scratchpad lives in the system message as the model's private
    # notes. Without this restructure the model treats the trailing
    # directive as another user turn to respond to and explainers
    # the deliberation summary back at the user verbatim.
    scratchpad_suffix = VOICE_PASS_SCRATCHPAD_SUFFIX.format(
        deliberation_summary=summary,
        agreement_directive=directive,
    )
    voice_system_prompt = (system_prompt or "") + scratchpad_suffix

    input_messages: list[Message] = []
    if voice_system_prompt:
        input_messages.append(Message(role=Role.system, content=voice_system_prompt))
    input_messages.extend(history)
    input_messages.append(user_message)

    request_payload: dict[str, object] = {
        "messages": [m.model_dump(mode="json", exclude_none=True) for m in input_messages],
        "ntState": nt_state.model_dump(exclude_none=True),
        # Voice pass is always pass index 0 from the driver's POV —
        # this isn't a deliberation pass.
        "passIndex": 0,
    }
    if temperature is not None:
        request_payload["temperature"] = float(temperature)
    if max_tokens is not None:
        request_payload["maxTokens"] = max_tokens

    request = GenerateRequest.model_validate(request_payload)

    started = time.monotonic()
    response = await voice_driver.generate(request)
    latency_ms = response.latencyMs or int((time.monotonic() - started) * 1000)

    output = Message(
        role=Role.assistant,
        content=response.content,
        driverName=voice_driver.name,
    )

    log.info(
        "voice pass: driver=%s latency=%dms output_len=%d",
        voice_driver.name,
        latency_ms,
        len(response.content),
    )
    return VoicePassOutcome(
        driver_name=voice_driver.name,
        input_messages=input_messages,
        output=output,
        latency_ms=latency_ms,
    )


__all__ = ["VoicePassOutcome", "_agreement_directive", "run_voice_pass"]
