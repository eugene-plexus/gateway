"""The plateau-stop gate: the noisy dopamine-RPE bout-end.

Three layers of coverage:
  1. BoutGate unit behavior with noise OFF (exact arithmetic): pass 0
     never stops, a flat (no-improvement) bout plateaus, sustained
     improvement extends it, and net valence shifts the stop.
  2. Seeded determinism with noise ON — the production path is stochastic
     but bit-exact under a fixed seed (the "seed the RNG, don't disable
     the noise" contract).
  3. Clamp-and-sample: hold the inputs fixed, sweep N seeds, and confirm
     the stop-pass DISTRIBUTION shifts in the predicted directions — the
     statistical validation of the non-deterministic component.
  4. Loop integration: a healthy bout ends on the plateau (Decision.
     terminate); a degenerate gate that never plateaus hits the cost fuse
     (Decision.cap_reached).
"""

from __future__ import annotations

import random

from eugene_plexus_gateway.bicameral.plateau import BoutGate, PlateauParams
from eugene_plexus_gateway.settings import Settings
from tests.conftest import (
    FakeHemisphereClient,
    build_loop_app,
    drive_message,
    make_message_event,
)


def _params(
    *,
    base_drift: float = 1.0,
    rpe_gain: float = 3.0,
    valence_gain: float = 0.5,
    noise_sigma: float = 0.0,
) -> PlateauParams:
    # Defaults mirror the shipped config defaults so the unit behavior
    # reflects production (noise off here only for exact arithmetic).
    return PlateauParams(
        base_drift=base_drift,
        rpe_gain=rpe_gain,
        valence_gain=valence_gain,
        noise_sigma=noise_sigma,
    )


def _run(gate: BoutGate, scores: list[float], *, valence: float = 0.0) -> int:
    """Feed `scores` pass by pass; return the 1-based pass count at stop
    (or len(scores) if it never stops within the supplied trajectory)."""
    for i, score in enumerate(scores):
        step = gate.observe(score=score, valence=valence)
        if step.should_stop:
            return i + 1
    return len(scores)


# --- 1. exact unit behavior (noise off) ----------------------------------


def test_pass_zero_never_stops() -> None:
    # No prior agreement on pass 0 → no improvement signal → cannot plateau
    # before a second thought exists to compare against.
    gate = BoutGate(_params(), random.Random(0))
    step = gate.observe(score=1.0, valence=0.0)
    assert step.should_stop is False
    assert step.improvement is None


def test_flat_agreement_plateaus_at_two_passes() -> None:
    # Identical agreement every pass → zero improvement → the resting drift
    # (1.0) reaches the unit bound on the first evaluated pass (pass 1).
    gate = BoutGate(_params(), random.Random(0))
    assert _run(gate, [0.8, 0.8, 0.8, 0.8]) == 2


def test_low_flat_agreement_also_plateaus() -> None:
    # "Ran out of angles" (low, flat) is a plateau too — improvement → 0
    # regardless of the absolute agreement level.
    gate = BoutGate(_params(), random.Random(0))
    assert _run(gate, [0.15, 0.15, 0.15, 0.15]) == 2


def test_sustained_improvement_extends_the_bout() -> None:
    # Rising agreement (positive RPE) cancels most of the stop-drift, so a
    # genuinely-refining bout runs SEVERAL passes longer than a flat one.
    # The >= flat + 2 margin guards against the regression where base_drift
    # == bound makes rpe_gain nearly inert (improvement could buy at most
    # one extra pass).
    flat = _run(BoutGate(_params(), random.Random(0)), [0.5] * 6)
    rising = _run(BoutGate(_params(), random.Random(0)), [0.2, 0.5, 0.8, 0.95, 0.95, 0.95])
    assert rising >= flat + 2


def test_agreement_drop_does_not_extend_the_bout() -> None:
    # A DROP in agreement is not a reason to keep grinding (only positive
    # RPE buys passes), so a diverging bout plateaus as fast as a flat one.
    gate = BoutGate(_params(), random.Random(0))
    assert _run(gate, [0.8, 0.5, 0.2, 0.1]) == 2


def test_stress_commits_sooner_than_contentment() -> None:
    # Negative valence (stress) removes the keep-thinking cushion; positive
    # valence (contentment) adds it. Same flat trajectory, opposite moods.
    stressed = _run(BoutGate(_params(), random.Random(0)), [0.5] * 8, valence=-0.8)
    content = _run(BoutGate(_params(), random.Random(0)), [0.5] * 8, valence=+0.8)
    assert stressed < content


def test_extreme_contentment_cannot_stall_the_bout() -> None:
    # The valence cushion is floored: even the highest legal valence_gain
    # (2.0) with a maximally-content state (net_valence ~1.25) cannot cancel
    # the resting drift forever. A flat bout must still plateau well within
    # the cost fuse — otherwise a contented mood would turn the fuse into the
    # de-facto terminator (the runaway the review caught).
    gate = BoutGate(_params(valence_gain=2.0), random.Random(0))
    stop = _run(gate, [0.5] * 30, valence=1.25)
    assert stop <= 6  # bound / (base_drift * _MIN_RESTING_FRACTION) = 1/0.25 = 4, + margin


# --- 2. seeded determinism with noise ON ---------------------------------


def test_noise_on_is_bit_exact_under_a_fixed_seed() -> None:
    scores = [0.3, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    a = _run(BoutGate(_params(noise_sigma=0.2), random.Random(99)), scores)
    b = _run(BoutGate(_params(noise_sigma=0.2), random.Random(99)), scores)
    assert a == b  # same seed → identical stochastic stop


def test_different_seeds_can_differ() -> None:
    # Near a marginal plateau, the noise can move the stop pass. Over a
    # spread of seeds we expect at least one disagreement (proves the noise
    # is live, not a no-op). base_drift just under the bound makes pass 1
    # a coin-flip around the bound.
    scores = [0.5] * 6
    stops = {
        _run(BoutGate(_params(base_drift=0.9, noise_sigma=0.2), random.Random(s)), scores)
        for s in range(40)
    }
    assert len(stops) > 1


# --- 3. clamp-and-sample: the stop-pass distribution shifts as predicted --


def _sample_stop_passes(*, valence: float, n: int = 20_000) -> list[int]:
    scores = [0.5] * 12  # flat → improvement 0 from pass 1; pure drift+noise
    return [
        _run(BoutGate(_params(noise_sigma=0.2), random.Random(s)), scores, valence=valence)
        for s in range(n)
    ]


def test_clamp_and_sample_valence_monotonicity() -> None:
    # Hold the agreement trajectory fixed, sweep the seed, and compare the
    # mean stop pass across three clamped moods. Stress should commit
    # earliest, contentment latest. With N=20k the standard error on the
    # mean is ~0.01 passes, so the (much larger) ordering gaps are
    # overwhelmingly significant — the statistical contract for the
    # stochastic stop.
    stressed = _sample_stop_passes(valence=-0.8)
    neutral = _sample_stop_passes(valence=0.0)
    content = _sample_stop_passes(valence=+0.8)

    def mean(xs: list[int]) -> float:
        return sum(xs) / len(xs)

    assert mean(stressed) < mean(neutral) < mean(content)
    # Every sampled bout stops well within the cost fuse (12 passes here).
    assert max(content) <= 12


def test_clamp_and_sample_is_genuinely_stochastic() -> None:
    # A flat bout at neutral valence does NOT collapse to a single pass —
    # the noise spreads it across at least two stop passes.
    passes = set(_sample_stop_passes(valence=0.0, n=2_000))
    assert len(passes) >= 2


# --- 4. loop integration: plateau vs cost fuse ---------------------------


async def test_loop_bout_ends_on_plateau_not_fuse(settings: Settings) -> None:
    # A normal turn (the seeded gate from build_loop_app) should end on a
    # dopamine plateau — the last thought's decision is `terminate`, and
    # the bout runs fewer passes than the fuse.
    left = FakeHemisphereClient(name="left")
    right = FakeHemisphereClient(name="right")
    left.responses = ["a settled thought"]
    right.responses = ["a settled thought"]
    app = build_loop_app(settings, [left, right])

    events = await drive_message(app, make_message_event("hello"))

    thoughts = [data for kind, data in events if kind == "thought"]
    assert thoughts, "expected at least one thought"
    assert thoughts[-1]["callosum"]["decision"] == "terminate"
    fuse = int(app.state.config_store.get("defaultMaxPasses"))
    assert len(thoughts) < fuse


async def test_loop_hits_cost_fuse_when_gate_never_plateaus(settings: Settings) -> None:
    # A degenerate gate (no drift, no noise) never moves the accumulator,
    # so the bout runs to the fuse and ends as `cap_reached` — proving the
    # runaway guard still bounds spend and is distinguishable from a
    # healthy plateau.
    left = FakeHemisphereClient(name="left")
    right = FakeHemisphereClient(name="right")
    left.responses = ["left view"]
    right.responses = ["right view"]
    app = build_loop_app(settings, [left, right])
    store = app.state.config_store
    store._values["plateauBaseDrift"] = 0.0
    store._values["plateauRpeGain"] = 0.0
    store._values["plateauValenceGain"] = 0.0
    store._values["plateauNoiseSigma"] = 0.0
    store._values["defaultMaxPasses"] = 4

    events = await drive_message(app, make_message_event("hello"))

    thoughts = [data for kind, data in events if kind == "thought"]
    assert len(thoughts) == 4  # ran to the fuse
    assert thoughts[-1]["callosum"]["decision"] == "cap_reached"
