"""Unit tests for the NT system: decay, observation impulses, and
parameter modulation. The end-to-end "NT evolves across chat turns"
test lives in test_nt_chat_evolution.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from eugene_plexus_gateway._generated.models import NTLevel, NTState
from eugene_plexus_gateway.bicameral.nt import (
    Observations,
    modulated_temperature,
    net_valence,
    neutral_state,
    tick,
)

# ---------------------------------------------------------------------------
# Decay toward baseline
# ---------------------------------------------------------------------------


def _state_with_dopamine_level(level: float, when: datetime) -> NTState:
    """Build an NT state with dopamine pinned to a custom level. Other
    NTs are neutral so they don't interfere with assertions."""
    base = neutral_state()
    return base.model_copy(
        update={
            "lastUpdated": when,
            "dopamine": NTLevel(level=level, baseline=0.5, decay=base.dopamine.decay),
        }
    )


def test_tick_with_no_observations_decays_toward_baseline() -> None:
    """A level offset from baseline must move TOWARD baseline (not
    away) when ticked without observations."""
    start = datetime.now(UTC) - timedelta(minutes=5)
    state = _state_with_dopamine_level(0.9, start)
    new_state = tick(state, observations=None, now=start + timedelta(minutes=5))
    assert 0.5 < new_state.dopamine.level < 0.9


def test_tick_preserves_baseline_when_level_already_at_baseline() -> None:
    """Level == baseline → decay is a no-op."""
    start = datetime.now(UTC) - timedelta(minutes=10)
    state = neutral_state().model_copy(update={"lastUpdated": start})
    new_state = tick(state, observations=None)
    for key in ("dopamine", "serotonin", "norepinephrine", "gaba", "cortisol", "acetylcholine"):
        triple: NTLevel = getattr(new_state, key)
        assert triple.level == pytest.approx(0.5)


def test_tick_negative_elapsed_treated_as_no_decay() -> None:
    """Clock skew → elapsed_seconds <= 0 → no decay. Defensive against
    weird host clocks; in practice this branch never fires in prod."""
    now = datetime.now(UTC)
    state = _state_with_dopamine_level(0.9, now)
    # `now` < state.lastUpdated would produce negative elapsed.
    new_state = tick(state, observations=None, now=now - timedelta(seconds=10))
    assert new_state.dopamine.level == 0.9


# ---------------------------------------------------------------------------
# Observation impulses
# ---------------------------------------------------------------------------


def _obs(
    *,
    final_agreement: float = 0.5,
    pass_count: int = 1,
    agreement_threshold: float = 0.5,
    avg_pass_latency_ms: float = 0.0,
) -> Observations:
    return Observations(
        final_agreement=final_agreement,
        pass_count=pass_count,
        agreement_threshold=agreement_threshold,
        avg_pass_latency_ms=avg_pass_latency_ms,
    )


def test_high_agreement_pushes_dopamine_up() -> None:
    state = neutral_state()
    new_state = tick(state, observations=_obs(final_agreement=1.0))
    assert new_state.dopamine.level > 0.5


def test_low_agreement_pulls_dopamine_down() -> None:
    state = neutral_state()
    new_state = tick(state, observations=_obs(final_agreement=0.0))
    assert new_state.dopamine.level < 0.5


def test_multi_pass_divergence_pushes_cortisol_up() -> None:
    state = neutral_state()
    new_state = tick(
        state,
        observations=_obs(final_agreement=0.2, pass_count=3),
    )
    assert new_state.cortisol.level > 0.5


def test_quick_single_pass_convergence_pushes_gaba_up() -> None:
    state = neutral_state()
    new_state = tick(
        state,
        observations=_obs(final_agreement=0.9, pass_count=1),
    )
    assert new_state.gaba.level > 0.5


def test_long_latency_pushes_norepinephrine_up() -> None:
    state = neutral_state()
    new_state = tick(state, observations=_obs(avg_pass_latency_ms=20_000.0))
    assert new_state.norepinephrine.level > 0.5


def test_short_latency_pulls_norepinephrine_down_slightly() -> None:
    """Very fast turns are calming for NE — Eugene doesn't need to be
    on high alert."""
    state = neutral_state()
    new_state = tick(state, observations=_obs(avg_pass_latency_ms=100.0))
    assert new_state.norepinephrine.level < 0.5


# ---------------------------------------------------------------------------
# Modulation: NT state → bicameral parameters
# ---------------------------------------------------------------------------


def _state_with(**levels: float) -> NTState:
    """Build a state where each kwarg-named NT is at the given level."""
    base = neutral_state()
    update: dict[str, object] = {}
    for name, lvl in levels.items():
        existing: NTLevel = getattr(base, name)
        update[name] = NTLevel(level=lvl, baseline=existing.baseline, decay=existing.decay)
    return base.model_copy(update=update)


def test_net_valence_at_neutral_is_exactly_zero() -> None:
    # Every level == baseline (0.5) → no deviation → zero felt valence.
    assert net_valence(neutral_state()) == 0.0


def test_net_valence_dopamine_is_appetitive() -> None:
    # dopamine weight +1.0; level 1.0 is +0.5 above baseline → +0.5.
    assert net_valence(_state_with(dopamine=1.0)) == pytest.approx(0.5)


def test_net_valence_cortisol_is_aversive() -> None:
    # cortisol weight -0.8; +0.5 deviation → -0.4. Stress feels bad.
    assert net_valence(_state_with(cortisol=1.0)) == pytest.approx(-0.4)


def test_net_valence_norepinephrine_is_mildly_aversive() -> None:
    # NE weight -0.3; +0.5 deviation → -0.15. Arousal/effort is a cost.
    assert net_valence(_state_with(norepinephrine=1.0)) == pytest.approx(-0.15)


def test_net_valence_gaba_is_mildly_appetitive() -> None:
    # GABA weight +0.4; +0.5 deviation → +0.2. Calm is mildly pleasant.
    assert net_valence(_state_with(gaba=1.0)) == pytest.approx(0.2)


def test_net_valence_ignores_decay_only_nts() -> None:
    # serotonin + acetylcholine carry weight 0.0 in v0.2 (they never leave
    # baseline), so sweeping them must not move valence.
    assert net_valence(_state_with(serotonin=1.0, acetylcholine=0.0)) == pytest.approx(0.0)
    assert net_valence(_state_with(serotonin=0.0, acetylcholine=1.0)) == pytest.approx(0.0)


def test_net_valence_combines_signs() -> None:
    # High dopamine + low cortisol → strongly positive; the reverse → negative.
    good = _state_with(dopamine=1.0, cortisol=0.0)  # +0.5 + (-0.8 * -0.5) = +0.9
    bad = _state_with(dopamine=0.0, cortisol=1.0)  # -0.5 + (-0.8 * +0.5) = -0.9
    assert net_valence(good) == pytest.approx(0.9)
    assert net_valence(bad) == pytest.approx(-0.9)
    assert net_valence(good) > net_valence(bad)


def test_modulated_temperature_at_neutral_equals_base() -> None:
    assert modulated_temperature(neutral_state(), 0.7) == pytest.approx(0.7)


def test_modulated_temperature_high_dopamine_raises_temperature() -> None:
    creative = _state_with(dopamine=1.0)
    t = modulated_temperature(creative, 0.5)
    assert t is not None
    assert t > 0.5


def test_modulated_temperature_high_gaba_lowers_temperature() -> None:
    determined = _state_with(gaba=1.0)
    t = modulated_temperature(determined, 0.5)
    assert t is not None
    assert t < 0.5


def test_modulated_temperature_returns_none_when_base_is_none() -> None:
    """Operator didn't configure a default temperature → NT must not
    fabricate one. Keep the "leave it unset" signal intact."""
    assert modulated_temperature(neutral_state(), None) is None


def test_modulated_temperature_clamped_to_valid_range() -> None:
    """Even at NT extremes, temperature can't exceed the [0, 2] range
    the config field documents."""
    crazy = _state_with(dopamine=1.0, gaba=0.0)
    t = modulated_temperature(crazy, 2.0)
    assert t is not None
    assert 0.0 <= t <= 2.0
