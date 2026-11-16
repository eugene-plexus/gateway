"""Row 14 (LD3, DI5): each runtime's peak prefill rate, calibrated from what it served.

The rate is the 90th percentile of uncached prompt tokens over time to first
token, per runtime and never pooled: replicas of one model run on unlike
machines. Short prompts say nothing about prefill (the fixed cost of a request
dominates), and a request that did not say how much it cached cannot be judged.
"""

from __future__ import annotations

import pytest

from eugene_plexus_gateway.prefill import (
    KEPT,
    MIN_SAMPLES,
    MIN_UNCACHED,
    PrefillModel,
    wait_seconds,
    worth_moving,
)

A = (None, "a")
B = ("tower", "b")


def feed(model: PrefillModel, runtime, rates, *, uncached: int = 4_000) -> None:  # type: ignore[no-untyped-def]
    for rate in rates:
        model.observe(
            runtime,
            prompt=uncached + 500,
            cached=500,
            first_ms=int(uncached / rate * 1000),
            elapsed_ms=int(uncached / rate * 1000) + 3_000,
        )


def test_nothing_is_known_before_enough_requests():
    model = PrefillModel()
    feed(model, A, [2_000] * (MIN_SAMPLES - 1))
    assert model.rate(A) is None
    assert model.prefill_seconds(A, 10_000) is None
    feed(model, A, [2_000])
    assert model.rate(A) == pytest.approx(2_000, rel=0.01)
    assert model.prefill_seconds(A, 10_000) == pytest.approx(5.0, rel=0.01)


def test_the_rate_is_the_peak_not_the_mean():
    """Requests that waited in the engine read slow. The peak is the rate the
    engine reads at when nothing stands in its way."""
    model = PrefillModel()
    feed(model, A, [3_000] * 8 + [400] * 2)
    assert model.rate(A) == pytest.approx(3_000, rel=0.01)


def test_each_runtime_has_its_own_rate():
    model = PrefillModel()
    feed(model, A, [4_000] * 10)
    feed(model, B, [800] * 10)
    assert model.rate(A) == pytest.approx(4_000, rel=0.01)
    assert model.rate(B) == pytest.approx(800, rel=0.01)
    assert model.rate((None, "never-served")) is None


def test_short_prompts_and_unjudged_requests_are_left_out():
    model = PrefillModel()
    feed(model, A, [50_000] * 20, uncached=MIN_UNCACHED - 1)  # too short to say anything
    for _ in range(20):
        model.observe(A, prompt=9_000, cached=None, first_ms=500, elapsed_ms=900)  # unjudged
        model.observe(A, prompt=9_000, cached=0, first_ms=0, elapsed_ms=900)  # no first token
        model.observe(A, prompt=9_000, cached=9_500, first_ms=500, elapsed_ms=900)  # more than read
    assert model.samples(A) == 0 and model.rate(A) is None


def test_only_the_last_samples_are_kept():
    model = PrefillModel()
    feed(model, A, [1_000] * KEPT)
    feed(model, A, [5_000] * KEPT)
    assert model.samples(A) == KEPT
    assert model.rate(A) == pytest.approx(5_000, rel=0.01)


def test_an_engine_that_left_ready_is_forgotten():
    model = PrefillModel()
    feed(model, A, [2_000] * 10)
    model.forget(A)
    assert model.rate(A) is None and model.service_seconds(A) is None and model.samples(A) == 0


def test_the_service_time_is_the_mean_of_whole_answers():
    model = PrefillModel()
    assert model.service_seconds(A) is None
    for ms in (2_000, 4_000, 6_000):
        model.observe(A, prompt=100, cached=0, first_ms=None, elapsed_ms=ms)
    assert model.service_seconds(A) == pytest.approx(4.0)
    assert model.samples(A) == 0, "a short non-streamed answer says nothing about prefill"


def test_seeding_from_retained_rows_is_the_same_as_having_seen_them():
    model = PrefillModel()
    rows = [(A, 4_500, 500, 2_000, 5_000)] * 10
    model.seed(rows)
    assert model.rate(A) == pytest.approx(2_000, rel=0.01)
    assert model.service_seconds(A) == pytest.approx(5.0)


def test_the_wait_for_a_slot_is_the_line_over_the_slots():
    # Four slots, an answer takes 20 s: one place frees every 5 s, so the
    # third in line waits 15 s.
    assert wait_seconds(waiting=3, parallel=4, service_seconds=20.0) == pytest.approx(15.0)
    # The one waiting is counted at least: it waits for a slot.
    assert wait_seconds(waiting=0, parallel=2, service_seconds=10.0) == pytest.approx(5.0)


def test_a_conversation_moves_only_when_the_other_replica_is_sooner_by_more_than_the_penalty():
    assert worth_moving(warm_ms=10_000, cold_ms=1_000, penalty_ms=2_000)
    assert not worth_moving(warm_ms=3_000, cold_ms=1_000, penalty_ms=2_000)  # exactly the penalty
    assert not worth_moving(warm_ms=10_000, cold_ms=9_000, penalty_ms=2_000)
    assert worth_moving(warm_ms=3_001, cold_ms=1_000, penalty_ms=2_000)
