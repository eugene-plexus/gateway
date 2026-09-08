"""The action-selection gate — what Eugene does once a think-bout settles.

This is the *inter-action* half of the NT-valence gate; `plateau.py` is the
*intra-bout* half ("take another pass?"). Together they are the locked
principle **behavior = pursuit of anticipated net NT valence**, applied at
two timescales. After deliberation, Eugene faces a choice among candidate
actions and picks the one with the highest *anticipated* value — the reward
he expects to feel for taking it.

Mechanism: **Boltzmann (softmax) action selection** — the RL-canonical
policy over action values. Each action's value is a linear function of
Eugene's current (post-bout) affect:

  value(SPEAK)  = response_drive·[addressed] + engagement_gain · net_valence
  value(IDLE)   = idle_floor

The choice is a temperature-controlled softmax SAMPLE from a seedable RNG —
not an argmax — so the higher-value action usually wins but the choice stays
stochastic (the "seed the RNG, don't disable the noise" contract, identical
to the plateau gate's noise term).

There is no hard rule ("addressed → always reply"). Being addressed enters
only as a strong positive value floor for SPEAK (`response_drive`), which a
sufficiently aversive state can override: when `net_valence` goes negative
(e.g. high cortisol / stress), the mood term drags SPEAK's value toward the
idle floor and silence becomes likely. Silence is therefore *emergent and
NT-gated*, never legislated — exactly the consciousness-not-a-chatbot shape.

The gain / floor / temperature are operator-tunable behavioral knobs (unlike
the valence sign-weights in `nt.py`, which are a fixed convention). The v1
defaults are hand-authored; revisit when self-initiated speech and the
tool-action increments land (v0.3).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import StrEnum

# Floor on the softmax temperature. A misconfigured 0 (or anything below the
# config minimum) would divide-by-zero; below this τ the policy is already a
# de-facto argmax, so clamping here is safe and never changes a sane config.
_MIN_SELECTION_TEMPERATURE = 1e-3


class Action(StrEnum):
    """The candidate actions the selector chooses between after a bout.

    v1 offers SPEAK vs IDLE (stay silent — return to the low-grade seeking
    floor). THINK_MORE (run another bout), SWITCH (change focus) and SLEEP
    (consolidate) are deferred increments: they slot in as extra candidates
    with their own value functions without changing the selection mechanism.
    The string values ARE the wire `GateDecision.action` enum values.
    """

    SPEAK = "speak"
    IDLE = "idle"


@dataclass(frozen=True)
class ActionPolicyParams:
    """Operator-tunable shape of the action-value function."""

    response_drive: float
    """Innate pull to act when addressed — SPEAK's value floor before mood.
    The "someone spoke to me" drive, strong but NOT a rule. Applies only when
    `addressed`; self-initiated speech has no innate drive yet (v1 never
    speaks unprompted — that is a later increment)."""

    engagement_gain: float
    """How strongly current net NT valence (mood) biases toward speaking.
    Positive: feeling good → more eager to engage; feeling bad (stress) →
    SPEAK's value falls toward the idle floor and silence becomes likely.
    This is what makes silence NT-gated and emergent rather than legislated.
    Set ≤ 0 to decouple mood from the speak/silent choice."""

    idle_floor: float
    """Constant value of staying silent — the bar SPEAK must clear. 0.0 makes
    a neutral-mood address almost always answered; raise it to make Eugene
    more reticent across the board, lower it to make him chattier."""

    selection_temperature: float
    """Softmax temperature. Low = greedy (nearly always the higher-value
    action); high = exploratory (closer to a coin-flip). The stochasticity is
    real and seedable, the same contract as the plateau gate's noise."""


@dataclass(frozen=True)
class ActionChoice:
    """The selector's outcome for one decision point."""

    action: Action
    anticipated_valence: float
    """The chosen action's value — the expected reward that drove the choice.
    Surfaced on `GateDecision.anticipatedValence`."""
    values: dict[Action, float]
    """Every candidate's value (diagnostic / stream transparency)."""
    probabilities: dict[Action, float]
    """The softmax distribution the choice was sampled from (diagnostic)."""


def action_value(
    action: Action, *, valence: float, addressed: bool, params: ActionPolicyParams
) -> float:
    """The anticipated value of one action given Eugene's current affect.

    SPEAK is worth an innate response drive (when addressed) plus a mood
    term; IDLE is a flat floor. "Anticipated" because the mood term encodes
    the expectation that engaging while he feels good will feel good
    (mood-congruent) and engaging while he feels bad will not.
    """
    if action is Action.SPEAK:
        drive = params.response_drive if addressed else 0.0
        return drive + params.engagement_gain * valence
    if action is Action.IDLE:
        return params.idle_floor
    raise ValueError(f"no value function for action {action!r}")


def select_action(
    *,
    valence: float,
    addressed: bool,
    params: ActionPolicyParams,
    rng: random.Random,
    candidates: tuple[Action, ...] = (Action.SPEAK, Action.IDLE),
) -> ActionChoice:
    """Boltzmann (softmax) selection over the candidate actions' values.

    `valence` is the live `net_valence(nt_state)` at decision time (the
    POST-bout state). Returns the sampled action plus the full value +
    probability maps. Consumes exactly one `rng.random()` draw, so a fixed
    seed makes the choice reproducible.
    """
    values = {
        a: action_value(a, valence=valence, addressed=addressed, params=params) for a in candidates
    }
    tau = max(params.selection_temperature, _MIN_SELECTION_TEMPERATURE)
    # Subtract the max before exp() for numerical stability: the top action's
    # weight is then exp(0)=1, so the denominator is always ≥ 1 (no div-by-
    # zero) and large negative exponents underflow harmlessly to 0.
    top = max(values.values())
    weights = {a: math.exp((v - top) / tau) for a, v in values.items()}
    total = sum(weights.values())
    probabilities = {a: w / total for a, w in weights.items()}

    draw = rng.random()
    cumulative = 0.0
    chosen = candidates[-1]  # float-rounding fallback for draw just under 1.0
    for a in candidates:
        cumulative += probabilities[a]
        if draw < cumulative:
            chosen = a
            break
    return ActionChoice(
        action=chosen,
        anticipated_valence=values[chosen],
        values=values,
        probabilities=probabilities,
    )
