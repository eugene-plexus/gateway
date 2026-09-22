"""The decision door's own protocol checks — bounds before backend work.

The gateway's copy of the pinned TypeSafe System One rules
(docs.typesafe.ai/api, read 2026-09-22; the driver holds the same rules
in `engines/systemone_http.py`). A copy rather than an import because
components share schemas, not code — and the two copies serve different
moments: this one refuses before a backend is picked or a byte of state
leaves the gateway, the driver's is the backstop for callers that are
not this gateway.

The one rule stated once: **fields the protocol does not define are
refused, never dropped.** Pydantic sheds unknown fields before a
handler sees them, so these checks run against the RAW question
objects, exactly as the driver's do.
"""

from __future__ import annotations

from typing import Any

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

_QUESTION_FIELDS = frozenset({"type", "instructions", "criteria"})


def protocol_violations(raw_questions: Any, *, max_questions: int) -> list[str]:
    """Every way the raw `questions` map violates the pinned protocol.

    `max_questions` is the gateway's own ceiling (`decisionMaxQuestions`)
    — a request bound like the body-size limit, enforced here because
    every question multiplies the work a single-slot backend will hold
    capacity for.
    """
    problems: list[str] = []
    if not isinstance(raw_questions, dict) or not raw_questions:
        return ["`questions` must be a non-empty object of named questions"]
    if len(raw_questions) > max_questions:
        problems.append(
            f"{len(raw_questions)} questions exceed this gateway's ceiling of "
            f"{max_questions} per request (`decisionMaxQuestions`); split the batch"
        )
    for name, question in raw_questions.items():
        where = f"questions[{name!r}]"
        if not isinstance(question, dict):
            problems.append(f"{where} must be an object")
            continue
        unknown = sorted(set(question) - _QUESTION_FIELDS)
        if unknown:
            problems.append(
                f"{where} carries fields the pinned protocol does not define: "
                f"{', '.join(unknown)} — refused rather than dropped"
            )
        kind = question.get("type")
        if kind not in ("noul", "choice", "score"):
            problems.append(f"{where}.type must be one of noul, choice, score")
            continue
        if "instructions" not in question or question["instructions"] in (None, ""):
            problems.append(f"{where}.instructions is required")
        criteria = question.get("criteria")
        if kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict) or set(criteria) - {"true", "false"}
            ):
                problems.append(
                    f'{where}.criteria for noul is an optional object mapping "true" '
                    f'and "false" to outcome descriptions'
                )
        elif kind == "choice":
            if not isinstance(criteria, dict) or not criteria:
                problems.append(f"{where}.criteria for choice must map option names to rubrics")
            elif len(criteria) > MAX_CHOICE_OPTIONS:
                problems.append(
                    f"{where}.criteria has {len(criteria)} options; the protocol's "
                    f"ceiling is {MAX_CHOICE_OPTIONS}"
                )
        elif kind == "score":
            if not isinstance(criteria, list) or not all(isinstance(c, str) for c in criteria):
                problems.append(f"{where}.criteria for score must be an array of level strings")
            elif not (MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS):
                problems.append(
                    f"{where}.criteria has {len(criteria)} levels; the protocol takes "
                    f"{MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS}, ordered lowest first"
                )
    return problems
