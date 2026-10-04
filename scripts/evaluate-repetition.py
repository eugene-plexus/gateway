"""Replay a labelled corpus without calling a model or storing generated text.

Run with --output report.json. Optional --corpus input.jsonl replaces the small
bundled diagnostic corpus. Each input has id, text, label (loop or legitimate),
and optional channel (text, reasoning, tool), structured, override (off), and
variant (e.g. backend DRY on/off, when actually captured from that backend).
The report measures detection, not live GPU time saved or semantic quality.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import string
import time
from pathlib import Path
from typing import Any

from eugene_plexus_gateway.repetition import MAX_PERIOD, SCAN_EVERY, Detector, Policy

PASSAGE = (
    "One more thing I can offer beyond the tool itself: I can reason through problems, "
    "write and explain code, summarize, draft, and so on — just using my built-in "
    "abilities. Want to put any of that to work? 😊\n"
    "One small note: the exact toolset can vary depending on how the session is "
    "configured, so if you're testing me, this is an honest snapshot of what's "
    "available right now. 🙂\n"
)


def corpus() -> list[dict[str, Any]]:
    rng = random.Random(20261003)

    def block(size: int) -> str:
        return "".join(rng.choices(string.ascii_letters, k=size))

    return [
        {"id": "reported-qwen-cycle-replayed", "label": "loop", "text": PASSAGE * 30},
        {
            "id": "whitespace-variation",
            "label": "loop",
            "text": (PASSAGE + PASSAGE.replace(" ", "\t  ")) * 15,
        },
        {"id": "hundred-char-period", "label": "loop", "text": block(100) * 30},
        {"id": "maximum-period", "label": "loop", "text": block(MAX_PERIOD) * 10},
        {"id": "period-outside-window", "label": "loop", "text": block(MAX_PERIOD + 101) * 10},
        {"id": "short-unit-loop", "label": "loop", "text": "abc " * 10000},
        {
            "id": "changing-table-rows",
            "label": "legitimate",
            "text": "\n".join(f"| {i} | same value |" for i in range(2000)),
        },
        {
            "id": "changing-code",
            "label": "legitimate",
            "text": "\n".join(f"assert records[{i}].enabled is True" for i in range(2000)),
        },
        {
            "id": "math-derivation",
            "label": "legitimate",
            "text": "\n".join(f"x_{i + 1} = x_{i} + 1" for i in range(2000)),
        },
        {"id": "three-quoted-copies", "label": "legitimate", "text": PASSAGE * 3},
        {
            "id": "poem-short-refrain",
            "label": "legitimate",
            "text": "Sing it again, sing it again.\n" * 1000,
        },
        {"id": "intentional-long-quotation", "label": "legitimate", "text": PASSAGE * 10},
        {
            "id": "intentional-quotation-override",
            "label": "legitimate",
            "text": PASSAGE * 10,
            "override": "off",
        },
        {
            "id": "long-reasoning",
            "label": "legitimate",
            "text": PASSAGE * 10,
            "channel": "reasoning",
        },
        {
            "id": "repeated-tool-arguments",
            "label": "legitimate",
            "text": PASSAGE * 10,
            "channel": "tool",
        },
        {
            "id": "repeated-json-output",
            "label": "legitimate",
            "text": json.dumps([PASSAGE] * 10),
            "structured": True,
        },
    ]


def evaluate(cases: list[dict[str, Any]], policy: Policy) -> dict[str, Any]:
    results = []
    for case in cases:
        detector = Detector(policy)
        raw = case["text"]
        enabled = case.get("override") != "off"
        eligible = case.get("channel", "text") == "text" and not case.get("structured", False)
        streamed_at = None
        start = time.perf_counter()
        for offset in range(0, len(raw), 17):
            if enabled and detector.feed(raw[offset : offset + 17]) and streamed_at is None:
                streamed_at = min(len(raw), offset + 17)
        if enabled:
            detector.finish()
        elapsed_ms = (time.perf_counter() - start) * 1000
        found = detector.detection
        would_stop = eligible and streamed_at is not None
        results.append(
            {
                "id": case["id"],
                "label": case["label"],
                "variant": case.get("variant", "diagnostic-replay"),
                "channel": case.get("channel", "text"),
                "detected": found is not None,
                "would_stop": would_stop,
                "input_chars": len(raw),
                "delivered_chars_at_detection": streamed_at,
                "period_chars": found.period_chars if found else None,
                "normalized_chars_at_detection": found.normalized_chars if found else None,
                "replay_tail_chars": len(raw) - streamed_at if would_stop else 0,
                "detector_ms": round(elapsed_ms, 3),
            }
        )
    return {
        "min_chars": policy.min_chars,
        "repeats": policy.repeats,
        "detected_loops": sum(r["label"] == "loop" and r["would_stop"] for r in results),
        "missed_loops": sum(r["label"] == "loop" and not r["would_stop"] for r in results),
        "false_stops_if_enforced": sum(
            r["label"] == "legitimate" and r["would_stop"] for r in results
        ),
        "cases": results,
    }


def overhead() -> dict[str, Any]:
    raw = "\n".join(
        f"Step {i}: add one value and keep making useful progress." for i in range(2000)
    )
    elapsed = []
    for _ in range(7):
        detector = Detector()
        start = time.perf_counter()
        for offset in range(0, len(raw), 17):
            detector.feed(raw[offset : offset + 17])
        elapsed.append((time.perf_counter() - start) * 1000)
    return {
        "characters": len(raw),
        "chunk_chars": 17,
        "runs": 7,
        "median_ms": round(statistics.median(elapsed), 3),
        "max_ms": round(max(elapsed), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = (
        [
            json.loads(line)
            for line in args.corpus.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.corpus
        else corpus()
    )
    for case in cases:
        if case.get("label") not in ("loop", "legitimate") or not isinstance(case.get("text"), str):
            parser.error("Each corpus row needs text and label loop or legitimate")
    report = {
        "kind": "diagnostic replay; not a live model benchmark",
        "python": platform.python_version(),
        "system": platform.system(),
        "default_mode": "observe",
        "max_period_chars": MAX_PERIOD,
        "scan_interval_chars": SCAN_EVERY,
        "live_backend_comparison": "not measured",
        "token_and_deadline_comparison": (
            "not measured; replay has no tokenizer or generation timing"
        ),
        "overhead": overhead(),
        "policies": [
            evaluate(cases, Policy(min_chars=m, repeats=r))
            for m, r in [(100, 4), (160, 4), (100, 5)]
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "overhead": report["overhead"],
                "policies": [
                    {k: v for k, v in p.items() if k != "cases"} for p in report["policies"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
