"""Tests for the corpus-callosum scorers and blend function.

The `EmbeddingAgreementScorer` isn't exercised here — it would pull torch
into the test environment and lengthen CI by several seconds for model
load. Its semantics are covered by hand-tested traces against the
running gateway. The Jaccard scorer remains in the suite because
it's the test-time fallback every chat test runs through.
"""

from __future__ import annotations

import pytest

from eugene_plexus_gateway.bicameral.callosum import (
    JaccardAgreementScorer,
    blend,
    jaccard_word_agreement,
)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("hello world", "hello world", 1.0),
        ("hello", "world", 0.0),
        ("", "", 1.0),
        ("anything", "", 0.0),
        ("", "anything", 0.0),
    ],
)
def test_agreement_edge_cases(left: str, right: str, expected: float) -> None:
    assert jaccard_word_agreement(left, right) == expected


def test_agreement_partial_overlap() -> None:
    score = jaccard_word_agreement("the cat sat", "the cat slept")
    # 2 shared (the, cat) of 4 unique total
    assert score == pytest.approx(2 / 4)


def test_agreement_is_case_insensitive() -> None:
    assert jaccard_word_agreement("Hello World", "hello world") == 1.0


def test_jaccard_scorer_class_matches_function() -> None:
    """The class wrapper exists so the bicameral loop can stay agnostic
    of which scorer is loaded; its `score` method must produce
    bit-identical numbers to the underlying function the rest of the
    suite asserts against."""
    scorer = JaccardAgreementScorer()
    assert scorer.score("hello world", "hello world") == 1.0
    assert scorer.score("the cat sat", "the cat slept") == pytest.approx(2 / 4)
    assert scorer.score("", "") == 1.0
    assert scorer.score("only-left", "") == 0.0


def test_blend_picks_longer_with_left_tiebreak() -> None:
    assert blend("short", "much longer response") == "much longer response"
    assert blend("equal-len", "equally-l") == "equal-len"  # left wins ties
    assert blend("same", "same") == "same"
