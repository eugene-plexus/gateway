"""Corpus callosum: agreement signal + blend across hemisphere outputs.

The scorer is an injectable strategy because the right algorithm depends on
the runtime environment: production uses sentence-transformers cosine
similarity (semantic), tests use Jaccard word overlap (fast, no torch).

v0.1 shipped a single `jaccard_word_agreement` function — kept here as
`JaccardAgreementScorer`'s implementation. v0.2.x adds
`EmbeddingAgreementScorer` because word-overlap was triggering false
disagreements: two responses meaning the same thing in different words
were scoring 0.05-0.15 and the loop ran to `cap_reached` instead of
terminating at the actual point of agreement.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


def jaccard_word_agreement(left: str, right: str) -> float:
    """Return a 0..1 agreement score from the word-set overlap of two outputs.

    Both empty -> 1.0 (vacuously identical). One empty -> 0.0.
    """
    left_stripped = left.strip()
    right_stripped = right.strip()
    if not left_stripped and not right_stripped:
        return 1.0
    if not left_stripped or not right_stripped:
        return 0.0
    left_words = {w for w in left_stripped.lower().split() if w}
    right_words = {w for w in right_stripped.lower().split() if w}
    if not left_words or not right_words:
        return 0.0
    intersect = left_words & right_words
    union = left_words | right_words
    return len(intersect) / len(union)


def blend(left: str, right: str) -> str:
    """Pick a final assistant message from the two hemispheres' outputs.

    v0.2.x note: this is no longer the user-facing reply path — chat's
    voice pass is. `blend` survives as the fallback for paths that read
    `CallosumState.blendedMessage` (UI debug surface, future analyzers).
    Longer-of-the-two with left as tie-breaker; an honest placeholder.
    """
    if len(right) > len(left):
        return right
    return left


class AgreementScorer(Protocol):
    """Strategy interface for cross-hemisphere agreement scoring.

    `score` returns a 0..1 number where 1.0 means "the same response" and
    0.0 means "completely disjoint." The bicameral loop terminates when
    score >= the configured `agreementThreshold`, so the scale must be
    consistent across passes — implementations should be deterministic
    given the same inputs.
    """

    def score(self, left: str, right: str) -> float: ...


class JaccardAgreementScorer:
    """Word-overlap scorer. The pre-v0.2.x default; now the fallback.

    Used in tests (no torch dependency) and as the degraded-mode scorer
    when the embedding model fails to load. Acceptable for catching the
    trivial "identical text" agreement case; misses paraphrases.
    """

    def score(self, left: str, right: str) -> float:
        return jaccard_word_agreement(left, right)


class EmbeddingAgreementScorer:
    """Cosine similarity in a sentence-transformer embedding space.

    Default model `all-MiniLM-L6-v2`: 22M params, ~80MB on disk, ~5ms
    inference per text on CPU. Picks up paraphrases and equivalent
    intents that Jaccard misses entirely — two responses meaning the
    same thing in different words score ~0.7-0.85 here vs 0.05-0.15
    under Jaccard. The practical threshold for "substantively agreed"
    on this model is ~0.75 (vs Jaccard's 0.5).

    The model loads in `__init__` (eager) so failures surface at app
    startup, not on the first chat turn. Encoding is sync because
    sentence-transformers' `encode` is sync — callers can run the
    constructor through `asyncio.to_thread` to keep the event loop
    responsive during model load.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Imported lazily so test environments that never instantiate
        # this scorer don't have to install torch + transformers.
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model: SentenceTransformer = SentenceTransformer(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(self, left: str, right: str) -> float:
        left_stripped = left.strip()
        right_stripped = right.strip()
        # Match Jaccard's edge-case semantics so the agreementThreshold
        # field has a stable meaning across scorer swaps.
        if not left_stripped and not right_stripped:
            return 1.0
        if not left_stripped or not right_stripped:
            return 0.0

        embeddings = self._model.encode(
            [left_stripped, right_stripped],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # With normalized vectors the dot product IS cosine similarity.
        # `encode(list)` returns a 2D array; row 0 is left, row 1 is right.
        cosine = float((embeddings[0] * embeddings[1]).sum())
        # Cosine ranges [-1, 1]; we want [0, 1] so the agreementThreshold
        # config field's 0..1 range stays meaningful. Map via clamp +
        # shift: negatives (semantically opposed) collapse to 0, the rest
        # stays as-is because real sentence-pair similarities for the
        # MiniLM family are essentially always >= 0.
        if math.isnan(cosine):
            return 0.0
        if cosine < 0.0:
            return 0.0
        if cosine > 1.0:
            return 1.0
        return cosine


def load_default_scorer(model_name: str = "all-MiniLM-L6-v2") -> AgreementScorer:
    """Build the production scorer, falling back to Jaccard on failure.

    The gateway's lifespan calls this through `asyncio.to_thread`.
    Any exception during model load — missing torch, no network for the
    first-run model download, corrupted cache — degrades to Jaccard
    instead of failing startup. Logs the reason so the operator can fix
    it without parsing a stack trace.
    """
    try:
        scorer = EmbeddingAgreementScorer(model_name)
        log.info(
            "agreement scorer: %s (sentence-transformer cosine similarity)",
            model_name,
        )
        return scorer
    except Exception as e:
        log.warning(
            "could not load embedding scorer %r (%s); "
            "falling back to Jaccard word-overlap. Chat will still work, "
            "but the bicameral loop will run more passes than necessary "
            "for paraphrased agreements.",
            model_name,
            e,
        )
        return JaccardAgreementScorer()
