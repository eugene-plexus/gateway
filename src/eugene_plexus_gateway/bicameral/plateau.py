"""The plateau-stop gate — when a think-bout ends.

This is the intra-bout half of the NT-valence gate (the "should I think
another pass?" decision), the first real piece of the locked principle
*behavior = pursuit of anticipated net NT valence*. It replaces the v0.2
"stop when agreement ≥ a fixed threshold, else run to a fixed cap" rule —
both of which are arbitrary cognitive limits the design forbids.

The mechanism is a basal-ganglia-style **drift-diffusion accumulator**.
Each pass produces a reward-prediction-error on thought quality — the
improvement in cross-hemisphere agreement over the previous pass. While
thinking keeps paying off (positive RPE) the accumulator is held back;
once improvement fades — whether because the hemispheres *converged*
(high, flat agreement: nothing more to gain) or *ran out of angles* (low,
flat agreement: not improving) — a resting drift carries the accumulator
to its bound and the bout ends. Both plateau modes are improvement → 0,
exactly as the design locks. Live net NT valence modulates the drift
(a good-feeling state lingers; a stressed one commits sooner), and a low
Gaussian noise term makes the stop *stochastic* (the brain-like part):
near a marginal plateau, two identical trajectories can stop a pass
apart.

There is no fixed counter and no slope-epsilon acting as a behavioral
rule: improvement and valence enter as *continuous drift terms*, the stop
is the noisy first-passage of the accumulator to a fixed geometric bound,
and the actual stop pass is a random variable. The only integer ceiling
(`max_passes`, in the loop) is a pure runaway **cost fuse** — accounting,
like a token ceiling — not the normal terminator.

The RNG is injected and seedable: production seeds from OS entropy (real
stochasticity); tests pass a fixed seed for bit-exact runs, or sweep
seeds to characterise the stop distribution (clamp-and-sample).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Fixed geometry of the accumulator: the bound the diffusion variable must
# cross to end the bout. NOT a knob — it trades off 1:1 with `base_drift`
# (mean passes-to-stop ≈ bound / base_drift), so `base_drift` is the single
# exposed mean-decision-time control and the bound stays a unit constant.
_STOP_BOUND = 1.0

# The valence cushion (a good-feeling state lingering) can SLOW the resting
# drift to stop, but never below this fraction of `base_drift`. Without the
# floor a contented enough state (high net valence times a high valence_gain)
# would cancel the resting drift entirely and the bout would never plateau —
# it would always run to the cost fuse, turning the fuse into the de-facto
# terminator. The floor keeps the plateau the terminator: mood modulates how
# long Eugene deliberates, it cannot make him deliberate forever. (Stress —
# negative valence — is NOT capped: committing sooner is always safe.)
_MIN_RESTING_FRACTION = 0.25


@dataclass(frozen=True)
class PlateauParams:
    """Tunable shape of the plateau accumulator (all operator config).

    These are gain / noise knobs, not stop rules — none of them is a
    "stop after N" or "stop when slope < ε" cutoff.
    """

    base_drift: float
    """Per-pass push toward STOP in the absence of improvement — the
    resting urge to commit. Mean passes-to-stop ≈ 1 / base_drift, so
    larger = Eugene plateaus sooner."""

    rpe_gain: float
    """How strongly per-pass agreement improvement (positive RPE) cancels
    the stop-drift — how much a refining thought buys another pass."""

    valence_gain: float
    """How strongly live net NT valence biases the bout length. Positive
    valence (good-feeling state) cushions against stopping; negative
    valence (e.g. high cortisol) removes the cushion and commits sooner.
    Set negative to invert (stress → ruminate longer)."""

    noise_sigma: float
    """Std-dev of the Gaussian diffusion noise added each pass. The
    brain-like stochasticity. 0 makes stopping deterministic given the
    inputs; low (default) gives slight run-to-run variation."""


@dataclass(frozen=True)
class PlateauStep:
    """The accumulator's reading after observing one pass."""

    should_stop: bool
    accumulator: float
    improvement: float | None
    """Agreement RPE this pass — `None` on pass 0 (no prior to compare)."""


class BoutGate:
    """Stateful per-bout plateau accumulator. Construct one per bout.

    Carries the diffusion variable and the previous pass's agreement
    across `observe()` calls. The owning RNG is injected so the noisy
    stop is reproducible under a fixed seed.
    """

    def __init__(self, params: PlateauParams, rng: random.Random) -> None:
        self._params = params
        self._rng = rng
        self._x = 0.0
        self._prev_score: float | None = None

    def observe(self, *, score: float, valence: float) -> PlateauStep:
        """Advance the accumulator by one pass and decide whether to stop.

        `score` is this pass's cross-hemisphere agreement in [0, 1];
        `valence` is the live `net_valence(nt_state)` (constant within a
        bout — NT does not change mid-bout). Pass 0 has no predecessor, so
        no improvement can be measured and the bout never stops on it: you
        cannot know a thought has plateaued until you have taken a second
        one to compare. From pass 1 the diffusion runs.
        """
        if self._prev_score is None:
            self._prev_score = score
            return PlateauStep(should_stop=False, accumulator=self._x, improvement=None)

        improvement = score - self._prev_score
        self._prev_score = score

        # Valence sets the RESTING drift toward STOP: a good-feeling state
        # lingers (slower drift), a stressed one commits sooner (faster).
        # Floor it so a contented mood can only slow the bout, never stall it
        # (see _MIN_RESTING_FRACTION) — keeps the plateau, not the cost fuse,
        # as the terminator.
        resting_drift = self._params.base_drift - self._params.valence_gain * valence
        resting_drift = max(self._params.base_drift * _MIN_RESTING_FRACTION, resting_drift)

        # Improvement is a separate anti-stop term that CAN push the step
        # negative — that is how a genuinely-refining bout keeps going. Only
        # its positive part counts (a *drop* in agreement is not a reason to
        # keep grinding). It is self-limiting: agreement is bounded in [0, 1],
        # so sustained positive RPE is impossible — improvement → 0 and the
        # resting drift takes over. Unlike valence it needs no floor.
        step = resting_drift - self._params.rpe_gain * max(0.0, improvement)

        noise = self._rng.gauss(0.0, self._params.noise_sigma)
        # Floor the accumulator at 0: evidence-for-stopping is non-negative,
        # so a long improving streak doesn't bank unbounded anti-stop credit —
        # once improvement ceases the bout ends promptly (within ~1 pass)
        # rather than after a tail proportional to how long it improved.
        self._x = max(0.0, self._x + step + noise)

        return PlateauStep(
            should_stop=self._x >= _STOP_BOUND,
            accumulator=self._x,
            improvement=improvement,
        )
