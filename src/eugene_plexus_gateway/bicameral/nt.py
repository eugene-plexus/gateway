"""Neurotransmitter state — evolution, valence, and bicameral modulation.

The NT system tracks six neurotransmitters as `{level, baseline, decay}`
triples. Between chat turns the level decays exponentially toward
baseline. Each turn's `Observations` apply impulses to specific NTs.
The live levels then feed two consumers:

  - **net_valence(state)** = the signed scalar "how good does this state
    feel" — dopamine/GABA appetitive, cortisol/NE aversive. This is the
    reward axis the action gate hill-climbs ("behavior = pursuit of
    anticipated net NT valence") and the per-pass brake the plateau-stop
    accumulator reads (see `bicameral/plateau.py`).
  - **temperature** = f(dopamine, GABA) — dopamine pushes toward
    exploration / creativity; GABA pulls toward determinism.

(The old `max_passes = f(cortisol, NE)` modulation is gone: a count is an
arbitrary cognitive limit, and deliberation depth is now owned by the
improvement-driven plateau-stop. Pressure is applied through NT dynamics +
noise, not by adding passes.)

Observation → NT impulse map:

  - high final agreement → dopamine up (reward for converging well)
  - the bout settled (final agreement ≥ threshold) → GABA up (calm)
  - the bout did NOT settle → cortisol up, scaled by passes spent (stress)
  - long average pass latency → norepinephrine up
  - acetylcholine (novelty / attention) and serotonin (mood / steady
    state) accumulate from baseline decay only in v0.2 — the topic-
    shift detector that would drive ACh lands in v0.3.

Note the impulse map keys "calm vs stress" on *whether the bout settled*
(final agreement), NOT on pass count: under the plateau-stop a clean
convergence is no longer synonymous with "exactly one pass."

State persistence: in-memory only. A restart resets to neutral state,
matching the "Eugene cools down after a reboot" anatomy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from .._generated.models import NTLevel, NTState

# Configurable per-NT defaults. Each NT has its own decay rate because
# they don't all settle on the same timescale in vivo: dopamine and
# norepinephrine are fast; serotonin is the slow baseline regulator;
# cortisol persists for many minutes; GABA / ACh are intermediate.
# Decays expressed as "natural log decay constant per second" — the
# half-life is ln(2) / decay.
_DEFAULT_DECAYS: dict[str, float] = {
    "dopamine": 0.010,  # ~70s half-life
    "serotonin": 0.001,  # ~11min half-life
    "norepinephrine": 0.020,  # ~35s half-life
    "gaba": 0.005,  # ~2.3min half-life
    "cortisol": 0.002,  # ~5.7min half-life
    "acetylcholine": 0.010,  # ~70s half-life
}


def _neutral(name: str) -> NTLevel:
    return NTLevel(level=0.5, baseline=0.5, decay=_DEFAULT_DECAYS[name])


def neutral_state() -> NTState:
    """A baseline NT state: every NT at 0.5, baseline 0.5, with per-NT decay."""
    return NTState(
        lastUpdated=datetime.now(UTC),
        dopamine=_neutral("dopamine"),
        serotonin=_neutral("serotonin"),
        norepinephrine=_neutral("norepinephrine"),
        gaba=_neutral("gaba"),
        cortisol=_neutral("cortisol"),
        acetylcholine=_neutral("acetylcholine"),
    )


@dataclass(frozen=True)
class Observations:
    """Observables collected from a finished bicameral turn.

    The chat handler builds one of these from `BicameralOutcome` and
    feeds it to `tick()`; the NT system converts it into per-NT
    impulses. v0.3 will add topic-shift signals + connector-side
    metadata to this struct.
    """

    final_agreement: float
    """Corpus-callosum agreement score on the FINAL pass, in [0, 1]."""

    pass_count: int
    """How many passes ran. 1 = quick convergence; >1 = divergence."""

    agreement_threshold: float
    """Configured agreement bar. Used to decide if `final_agreement`
    represents a true convergence or a cap-reached termination."""

    avg_pass_latency_ms: float
    """Mean of (max over hemispheres of latencyMs) across passes."""


@dataclass(frozen=True)
class _Impulse:
    dopamine: float = 0.0
    serotonin: float = 0.0
    norepinephrine: float = 0.0
    gaba: float = 0.0
    cortisol: float = 0.0
    acetylcholine: float = 0.0


# Latency normalization. Past 30s the NE impulse saturates; below this
# range latency contributes proportionally to the boost.
_LATENCY_SATURATION_MS = 30_000.0


def _impulses_from_observations(obs: Observations) -> _Impulse:
    """Map a turn's `Observations` to per-NT impulses in [-0.5, +0.5].

    Each impulse is a one-shot nudge layered on top of the level after
    decay. Magnitudes are kept modest so single turns don't slam the
    state to extremes — the dynamics emerge from repeated observations
    over many turns.
    """
    # Agreement → dopamine. Centered at the agreement threshold so
    # "barely converged" produces no impulse and "perfect agreement"
    # is strongly rewarding.
    above_threshold = obs.final_agreement - obs.agreement_threshold
    dopamine = max(-0.3, min(0.3, above_threshold * 0.6))

    # Did the bout SETTLE? → cortisol / GABA. Keyed on the outcome
    # quality (final agreement vs threshold), NOT on pass count: under
    # the plateau-stop a clean convergence is no longer "exactly one
    # pass," so pass count alone no longer distinguishes calm from
    # struggle. A bout that settled is calming (GABA up); one that ran
    # without settling is stressful (cortisol up), and the stress scales
    # with how many passes were spent failing to settle.
    if obs.final_agreement >= obs.agreement_threshold:
        gaba = 0.15
        cortisol = -0.05  # mild relaxation
    else:
        # Each pass spent without settling adds cortisol stress,
        # saturating at +0.3.
        cortisol = min(0.3, max(0, obs.pass_count - 1) * 0.08)
        gaba = -0.05

    # Latency → norepinephrine. Long latency = pay-attention nudge.
    norm_latency = min(1.0, obs.avg_pass_latency_ms / _LATENCY_SATURATION_MS)
    norepinephrine = (norm_latency - 0.3) * 0.4
    norepinephrine = max(-0.15, min(0.3, norepinephrine))

    return _Impulse(
        dopamine=dopamine,
        gaba=gaba,
        cortisol=cortisol,
        norepinephrine=norepinephrine,
    )


def _decay_level(level: NTLevel, elapsed_seconds: float) -> float:
    """Exponential decay of `level.level` toward `level.baseline`.

    `new = baseline + (level - baseline) * exp(-decay * elapsed)`

    decay==0 means no decay (the level holds). Negative elapsed is
    treated as zero — protects against clock-skew or test sequencing.
    """
    if elapsed_seconds <= 0 or level.decay <= 0:
        return level.level
    return level.baseline + (level.level - level.baseline) * math.exp(
        -level.decay * elapsed_seconds
    )


def _apply(level: NTLevel, impulse: float, elapsed_seconds: float) -> NTLevel:
    decayed = _decay_level(level, elapsed_seconds)
    new = max(0.0, min(1.0, decayed + impulse))
    return NTLevel(level=new, baseline=level.baseline, decay=level.decay)


def tick(
    state: NTState,
    *,
    observations: Observations | None,
    now: datetime | None = None,
) -> NTState:
    """Advance NT state: decay every level toward baseline, then apply
    impulses from `observations` (if supplied).

    Pure function — does not mutate `state`. Callers store the
    returned value back into `app.state.nt_state`.

    `now` is overridable for tests; defaults to `datetime.now(UTC)`.
    """
    now = now or datetime.now(UTC)
    elapsed_seconds = (now - state.lastUpdated).total_seconds()
    impulse = _impulses_from_observations(observations) if observations is not None else _Impulse()

    return NTState(
        lastUpdated=now,
        dopamine=_apply(state.dopamine, impulse.dopamine, elapsed_seconds),
        serotonin=_apply(state.serotonin, impulse.serotonin, elapsed_seconds),
        norepinephrine=_apply(state.norepinephrine, impulse.norepinephrine, elapsed_seconds),
        gaba=_apply(state.gaba, impulse.gaba, elapsed_seconds),
        cortisol=_apply(state.cortisol, impulse.cortisol, elapsed_seconds),
        acetylcholine=_apply(state.acetylcholine, impulse.acetylcholine, elapsed_seconds),
    )


# -------- Net affective valence --------
#
# The signed scalar "how good does the current internal state feel."
# This is the v0.2 reward primitive — the axis the action gate hill-
# climbs ("behavior = pursuit of anticipated net NT valence") and the
# brake the plateau-stop accumulator reads. Valence is measured as a
# *deviation from baseline*, so a fully-neutral Eugene (every level at
# its baseline) has net_valence == 0.0 exactly: a clean zero reference.
#
# Per-NT sign + weight. These are a SIGN CONVENTION (what valence IS),
# not operator-tunable behavioral knobs: exposing them would let a
# config edit invert Eugene's reward sign, which is a footgun, not a
# feature. Only the four NTs that actually receive impulses in v0.2
# carry weight; serotonin/acetylcholine are decay-only (they never leave
# baseline), so their contribution is 0 today — the entries are present
# so wiring their impulses in v0.3 is a weight edit, not a signature
# change.
_VALENCE_WEIGHTS: dict[str, float] = {
    "dopamine": +1.0,  # appetitive — the reward/wanting signal; dominant term
    "gaba": +0.4,  # calm / settledness — mildly pleasant
    "cortisol": -0.8,  # stress / unresolved tension — aversive
    "norepinephrine": -0.3,  # arousal / effort cost — mildly aversive at load
    "serotonin": 0.0,  # decay-only in v0.2 → no valence contribution yet
    "acetylcholine": 0.0,  # decay-only in v0.2 → no valence contribution yet
}


def net_valence(state: NTState) -> float:
    """Signed hedonic tone of an NTState, roughly in [-1, +1].

    Sum over the active NTs of `(level - baseline) * weight`. Positive =
    appetitive (feels good to keep doing this), negative = aversive.
    Neutral state (every level == baseline) returns exactly 0.0.

    Pure function of the NT state — sibling to `modulated_temperature`.
    Reused by the plateau-stop gate and (later) the action-selection
    gate; surfaced on `GateDecision.anticipatedValence`.
    """
    total = 0.0
    for name, weight in _VALENCE_WEIGHTS.items():
        if weight == 0.0:
            continue
        level: NTLevel = getattr(state, name)
        total += weight * (level.level - level.baseline)
    return total


# -------- Bicameral parameter modulation --------
#
# `modulated_temperature` returns a plain value the bicameral loop
# consumes. Pure function of the NT state + the configured baseline, so
# the loop pre-computes it once per turn and passes it down unchanged.


def modulated_temperature(state: NTState, base_temperature: float | None) -> float | None:
    """`temperature = base * (1 + 0.5*(dopamine-0.5) - 0.5*(gaba-0.5))`.

    At neutral NT, returns `base_temperature`. Dopamine pushes
    temperature up (exploration); GABA pulls it down (determinism).
    Clamped to [0.0, 2.0] to match the config field's allowed range.

    Returns None if `base_temperature` is None (operator explicitly
    didn't configure a default; NT shouldn't fabricate one).
    """
    if base_temperature is None:
        return None
    dopamine = state.dopamine.level
    gaba = state.gaba.level
    factor = 1.0 + 0.5 * (dopamine - 0.5) - 0.5 * (gaba - 0.5)
    return max(0.0, min(2.0, base_temperature * factor))
