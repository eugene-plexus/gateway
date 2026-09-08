"""The action-selection gate: speak vs stay-silent over anticipated valence.

Four layers, matching the plateau gate's coverage shape:
  1. Exact value arithmetic (the linear value model) + the softmax math.
  2. Behavior: an addressed Eugene almost always speaks at neutral mood;
     a sufficiently aversive state makes silence emerge; positive mood
     locks in speaking; removing the address removes the drive.
  3. Seeded determinism (bit-exact under a fixed seed) + different seeds
     can diverge near a marginal choice.
  4. Clamp-and-sample: hold mood fixed, sweep seeds, confirm the silence
     RATE rises monotonically as mood falls — the statistical contract.
  5. Loop integration: a neutral turn speaks (gate=speak + a speech event);
     a silence-forcing config makes the gate choose idle — no voice pass,
     no speech, but the bout still thought + evolved NT + flushed its trace.
"""

from __future__ import annotations

import math
import random

from eugene_plexus_gateway.bicameral.action import (
    Action,
    ActionPolicyParams,
    action_value,
    select_action,
)
from eugene_plexus_gateway.settings import Settings
from tests.conftest import (
    FakeHemisphereClient,
    build_loop_app,
    drive_message,
    make_message_event,
)


def _params(
    *,
    response_drive: float = 0.6,
    engagement_gain: float = 0.5,
    idle_floor: float = 0.0,
    selection_temperature: float = 0.15,
) -> ActionPolicyParams:
    # Defaults mirror the shipped config defaults so unit behavior reflects
    # production.
    return ActionPolicyParams(
        response_drive=response_drive,
        engagement_gain=engagement_gain,
        idle_floor=idle_floor,
        selection_temperature=selection_temperature,
    )


def _p_speak(valence: float, params: ActionPolicyParams, *, addressed: bool = True) -> float:
    """The closed-form softmax P(speak) for the two-action case, for
    cross-checking the sampler's empirical rates."""
    v_speak = action_value(Action.SPEAK, valence=valence, addressed=addressed, params=params)
    v_idle = action_value(Action.IDLE, valence=valence, addressed=addressed, params=params)
    return 1.0 / (1.0 + math.exp(-(v_speak - v_idle) / params.selection_temperature))


def _silence_rate(valence: float, params: ActionPolicyParams, *, n: int = 20_000) -> float:
    silent = 0
    for s in range(n):
        choice = select_action(valence=valence, addressed=True, params=params, rng=random.Random(s))
        if choice.action is Action.IDLE:
            silent += 1
    return silent / n


# --- 1. exact value + softmax arithmetic ---------------------------------


def test_speak_value_is_drive_plus_mood() -> None:
    p = _params()
    # addressed: response_drive + engagement_gain * valence
    assert action_value(Action.SPEAK, valence=0.0, addressed=True, params=p) == 0.6
    assert action_value(Action.SPEAK, valence=0.4, addressed=True, params=p) == 0.6 + 0.5 * 0.4
    assert action_value(Action.SPEAK, valence=-1.0, addressed=True, params=p) == 0.6 - 0.5


def test_not_addressed_drops_the_response_drive() -> None:
    p = _params()
    # No address → no innate drive; only the mood term remains.
    assert action_value(Action.SPEAK, valence=0.0, addressed=False, params=p) == 0.0
    assert action_value(Action.SPEAK, valence=0.4, addressed=False, params=p) == 0.5 * 0.4
    # So an unaddressed neutral Eugene is genuinely indifferent (≈ coin flip),
    # not driven to speak — the floor for the future mind-wandering increment.
    assert _p_speak(0.0, p, addressed=False) == 0.5


def test_idle_value_is_the_flat_floor() -> None:
    p = _params(idle_floor=0.3)
    assert action_value(Action.IDLE, valence=0.0, addressed=True, params=p) == 0.3
    assert action_value(Action.IDLE, valence=-1.0, addressed=False, params=p) == 0.3


def test_probabilities_are_a_normalized_softmax() -> None:
    choice = select_action(valence=0.0, addressed=True, params=_params(), rng=random.Random(0))
    assert abs(sum(choice.probabilities.values()) - 1.0) < 1e-9
    # At neutral mood SPEAK dominates (~0.98) but is not certain.
    assert 0.97 < choice.probabilities[Action.SPEAK] < 0.99
    assert choice.anticipated_valence == choice.values[choice.action]


# --- 2. behavior ----------------------------------------------------------


def test_neutral_mood_addressed_almost_always_speaks() -> None:
    assert _silence_rate(0.0, _params(), n=4_000) < 0.05


def test_stress_makes_silence_emerge() -> None:
    # A strongly aversive post-bout state (valence -1.0) drags SPEAK's value
    # toward the idle floor: silence stops being negligible. This is the
    # whole point — withdrawal emerges from NT state, it is not legislated.
    rate = _silence_rate(-1.0, _params(), n=20_000)
    assert 0.2 < rate < 0.45  # closed-form P(idle) ≈ 0.34


def test_positive_mood_locks_in_speaking() -> None:
    # Feeling good → engaging feels worth it → essentially always speaks.
    assert _silence_rate(0.5, _params(), n=4_000) < 0.01


def test_silence_rate_is_monotonic_in_mood() -> None:
    p = _params()
    rates = [_silence_rate(v, p) for v in (-1.0, -0.5, 0.0, 0.5)]
    # Worse mood → more silence, strictly. With N=20k the standard error on
    # each rate is < 0.004, far below the gaps between these points.
    assert rates[0] > rates[1] > rates[2] > rates[3]


# --- 3. seeded determinism ------------------------------------------------


def test_fixed_seed_is_bit_exact() -> None:
    a = select_action(valence=-1.0, addressed=True, params=_params(), rng=random.Random(7))
    b = select_action(valence=-1.0, addressed=True, params=_params(), rng=random.Random(7))
    assert a.action == b.action


def test_different_seeds_can_diverge_near_the_margin() -> None:
    # At a near-50/50 mood the choice genuinely depends on the draw, proving
    # the stochasticity is live (not a no-op argmax).
    p = _params()
    # Pick a valence where P(speak) ≈ 0.5: solve drive + gain*v = idle_floor →
    # 0.6 + 0.5*v = 0 → v = -1.2.
    chosen = {
        select_action(valence=-1.2, addressed=True, params=p, rng=random.Random(s)).action
        for s in range(40)
    }
    assert chosen == {Action.SPEAK, Action.IDLE}


def test_low_temperature_is_greedy() -> None:
    # Near-zero τ collapses the softmax to argmax: with a clear value gap the
    # higher-value action wins on every seed.
    p = _params(selection_temperature=0.01)
    chosen = {
        select_action(valence=0.0, addressed=True, params=p, rng=random.Random(s)).action
        for s in range(50)
    }
    assert chosen == {Action.SPEAK}


# --- 4. loop integration --------------------------------------------------


async def test_loop_speaks_on_a_neutral_turn(settings: Settings) -> None:
    # The seeded gate from build_loop_app (neutral NT, default action params)
    # elects SPEAK, so the turn emits a gate=speak decision and a speech event.
    left = FakeHemisphereClient(name="left")
    right = FakeHemisphereClient(name="right")
    left.responses = ["a settled thought"]
    right.responses = ["a settled thought"]
    app = build_loop_app(settings, [left, right])

    events = await drive_message(app, make_message_event("hello"))

    gates = [data for kind, data in events if kind == "gate_decision"]
    speeches = [data for kind, data in events if kind == "speech"]
    assert gates and gates[-1]["action"] == "speak"
    assert len(speeches) == 1


async def test_loop_stays_silent_when_the_gate_elects_idle(settings: Settings) -> None:
    # Force the silence branch deterministically (drive 0, high idle floor,
    # near-zero τ → IDLE wins for any seed). Eugene still THOUGHT (thoughts +
    # nt_update + tool trace), but emits no voice/speech.
    left = FakeHemisphereClient(name="left")
    right = FakeHemisphereClient(name="right")
    left.responses = ["left view"]
    right.responses = ["right view"]
    app = build_loop_app(settings, [left, right])
    store = app.state.config_store
    store._values["actionResponseDrive"] = 0.0
    store._values["actionIdleFloor"] = 1.0
    store._values["actionSelectionTemperature"] = 0.01

    events = await drive_message(app, make_message_event("hello"))

    kinds = [kind for kind, _ in events]
    gates = [data for kind, data in events if kind == "gate_decision"]
    assert gates and gates[-1]["action"] == "idle"
    assert "speech" not in kinds  # he chose not to reply
    assert "thought" in kinds  # but he did think
    assert "nt_update" in kinds  # and his NT evolved
    assert "tool_call" in kinds  # and the perception/NT trace was flushed
