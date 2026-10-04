"""Opt-in comparison against an already-running llama.cpp HTTP server.

Does not change the runtime/profile. Runs serial requests with one seed and
fixed resource backstops, comparing defaults, gateway stopping, and native DRY.
The forced-copy prompt is intentional repetition, not a reproduced model bug.
Saved reports contain counts, hashes and short previews, not full reasoning.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

from eugene_plexus_gateway.driver_client import StreamEvent
from eugene_plexus_gateway.repetition import Guard, Policy

PASSAGE = (
    "The patient observer records each measurement, checks the instrument, "
    "and writes the result in the notebook before beginning the next trial."
)


async def measure(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    variant: str,
    max_tokens: int,
    deadline: float,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "seed": 42,
    }
    if variant == "native-dry":
        body.update(dry_multiplier=0.8, dry_penalty_last_n=2048)
    guard = Guard(Policy(mode="stop" if variant == "gateway-stop" else "observe"))
    text, reasoning_chars, events = "", 0, 0
    usage = None
    finish = None
    outcome = "stream-ended"
    seen_at = None
    started = time.perf_counter()
    try:
        async with asyncio.timeout(deadline):
            async with client.stream("POST", "/v1/chat/completions", json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        outcome = "completed"
                        guard.take(StreamEvent(done=True))
                        break
                    chunk = json.loads(payload)
                    if chunk.get("error"):
                        outcome = "upstream-error"
                        break
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        visible = delta.get("content") or ""
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        finish = choice.get("finish_reason") or finish
                        text += visible
                        reasoning_chars += len(reasoning)
                        events += 1
                        stop = guard.take(StreamEvent(text=visible, reasoning=reasoning))
                        if seen_at is None and any(d.detection for d in guard.channels.values()):
                            seen_at = round((time.perf_counter() - started) * 1000, 3)
                        if stop:
                            outcome = "repetition-stopped"
                            break
                    if outcome == "repetition-stopped":
                        break
    except TimeoutError:
        outcome = "deadline"
    except httpx.HTTPError as exc:
        outcome = type(exc).__name__
    return {
        "variant": variant,
        "outcome": outcome,
        "finish_reason": finish,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "detected_at_ms": seen_at,
        "visible_chars": len(text),
        "reasoning_chars": reasoning_chars,
        "delta_events": events,
        "usage": usage,
        "visible_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "visible_preview": text[:200],
        "detections": {
            key: {
                "period_chars": d.detection.period_chars,
                "normalized_chars": d.detection.normalized_chars,
            }
            for key, d in guard.channels.items()
            if d.detection is not None
        },
        "sampling_override": {k: v for k, v in body.items() if k.startswith("dry_")},
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--deadline", type=float, default=90)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.deadline <= 0:
        parser.error("Positive token and deadline limits are required")
    async with httpx.AsyncClient(base_url=args.url, trust_env=False, timeout=30) as client:
        props_response = await client.get("/props")
        props_response.raise_for_status()
        props = props_response.json()
        params = props.get("default_generation_settings", {}).get("params", {})
        allowed = (
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "repeat_penalty",
            "repeat_last_n",
            "dry_multiplier",
            "dry_base",
            "dry_allowed_length",
            "dry_penalty_last_n",
        )
        report: dict[str, Any] = {
            "model": args.model,
            "engine_build": props.get("build_info"),
            "kind": "live local llama.cpp diagnostic; no original Workbench history or tools",
            "seed": 42,
            "max_tokens": args.max_tokens,
            "deadline_seconds": args.deadline,
            "runtime_defaults": {k: params[k] for k in allowed if k in params},
            "cases": [],
        }
        prompts = {
            "tools-question": "What tooling can you see right now?",
            "intentional-copies": (
                "For a formatting test, output exactly 20 copies of the following passage, "
                "separated by one blank line. Do not number the copies or add any other text.\n\n"
                + PASSAGE
            ),
        }
        for name, prompt in prompts.items():
            for variant in ("resource-limits-only", "gateway-stop", "native-dry"):
                print(f"Starting {name}: {variant}", flush=True)
                result = await measure(
                    client, args.model, prompt, variant, args.max_tokens, args.deadline
                )
                result["case"] = name
                report["cases"].append(result)
                args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(
                    json.dumps({k: v for k, v in result.items() if k not in ("visible_preview",)}),
                    flush=True,
                )


if __name__ == "__main__":
    asyncio.run(main())
