"""A stream says what the backend is doing, to a caller who asked.

2026-09-27. Until the first token a stream said nothing -- measured on
llama.cpp b11215, 21 s of prompt reading on the processor and not one
frame -- and a tester who saw nothing for minutes concluded his request
had failed. The driver now reports what each backend can observe (see
its `StreamProgress`); this is the gateway half:

* `stream_options.include_progress` asks, and only asking gets anything
  -- a progress chunk has no choices, which an OpenAI client that did
  not ask should never see.
* A progress chunk is `choices: []` with an `x_eugene_plexus` holding
  only `progress`, in this document's snake_case.
* **It is not output.** It is not the commit point (a backend that fails
  after reporting progress still fails over) and not the first token (the
  time-to-first-token a metric records is still the first word).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import GenerateRequest, Problem
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import (
    DriverError,
    FailoverDriverClient,
    HttpDriverClient,
    StreamEvent,
)
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

MODEL = "Qwen3-0.6B-Q4_K_M.gguf"

# Two batches of a 1,000-token prompt, then a Claude-style tool, in the
# driver's own camelCase.
READS = [
    {"stage": "prompt", "promptTokens": 1000, "cachedTokens": 0, "processedTokens": 0},
    {
        "stage": "prompt",
        "promptTokens": 1000,
        "cachedTokens": 0,
        "processedTokens": 512,
        "elapsedMs": 700,
    },
    {"stage": "tool", "tool": "Read"},
]


def _fake(**knobs: Any) -> FakeDriverClient:
    fake = FakeDriverClient(name="box", model_id=MODEL)
    fake.responses = ["East"]
    fake.progress = list(READS)
    for key, value in knobs.items():
        setattr(fake, key, value)
    return fake


def _client(settings: Settings, *fakes: FakeDriverClient) -> TestClient:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*fakes)
    return TestClient(app)


def _chat(**extra: Any) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Which way? One word."}],
        "stream": True,
        **extra,
    }


def _frames(raw: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[6:])
        for line in raw.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def test_asked_the_progress_arrives_before_the_answer(settings: Settings) -> None:
    fake = _fake()
    with _client(settings, fake) as client:
        r = client.post(
            "/v1/chat/completions", json=_chat(stream_options={"include_progress": True})
        )
    assert r.status_code == 200, r.text
    frames = _frames(r.text)
    progress = [f for f in frames if not f["choices"]]
    assert [f["x_eugene_plexus"] for f in progress] == [
        {
            "progress": {
                "stage": "prompt",
                "prompt_tokens": 1000,
                "cached_tokens": 0,
                "processed_tokens": 0,
            }
        },
        {
            "progress": {
                "stage": "prompt",
                "prompt_tokens": 1000,
                "cached_tokens": 0,
                "processed_tokens": 512,
                "elapsed_ms": 700,
            }
        },
        {"progress": {"stage": "tool", "tool": "Read"}},
    ]
    # Before the role chunk: progress is not the answer starting.
    assert frames[: len(progress)] == progress
    assert frames[len(progress)]["choices"][0]["delta"] == {"role": "assistant"}
    assert fake.calls[0].reportProgress is True
    text = "".join(f["choices"][0]["delta"].get("content") or "" for f in frames if f["choices"])
    assert text == "East"


def test_unasked_nothing_changes(settings: Settings) -> None:
    fake = _fake()
    with _client(settings, fake) as client:
        plain = client.post("/v1/chat/completions", json=_chat())
        usage_only = client.post(
            "/v1/chat/completions", json=_chat(stream_options={"include_usage": True})
        )
    for r in (plain, usage_only):
        assert r.status_code == 200, r.text
        assert all("progress" not in (f.get("x_eugene_plexus") or {}) for f in _frames(r.text)), (
            r.text
        )
    assert not fake.calls[0].reportProgress and not fake.calls[1].reportProgress


def test_progress_is_a_streaming_option_and_a_boolean(settings: Settings) -> None:
    fake = _fake()
    with _client(settings, fake) as client:
        batch = client.post(
            "/v1/chat/completions",
            json={**_chat(stream=False), "stream_options": {"include_progress": True}},
        )
        stringly = client.post(
            "/v1/chat/completions", json=_chat(stream_options={"include_progress": "true"})
        )
    assert batch.status_code == 400 and "requires stream true" in batch.text, batch.text
    assert stringly.status_code == 400, stringly.text
    assert "stream_options.include_progress" in stringly.text
    assert fake.calls == []


def _refusal() -> DriverError:
    return DriverError(
        driver_name="primary",
        driver_url="http://primary",
        status_code=502,
        problem=Problem(
            type="about:blank",
            title="refused before execution",
            status=502,
            retryDisposition="safe",
        ),
        raw_body="",
    )


async def test_progress_is_not_the_commit_point() -> None:
    # The primary read half the prompt and then refused, safely. Nothing of
    # its answer reached the caller, so the backup answers -- where the
    # same failure after a first word is a truncation (test_failover_safety).
    first, second = FakeDriverClient(name="primary"), FakeDriverClient(name="backup")
    second.responses = ["East"]

    async def half_read(request: GenerateRequest) -> Any:
        yield StreamEvent(progress=READS[1])
        raise _refusal()

    first.stream = half_read  # type: ignore[method-assign]
    events = [
        event
        async for event in FailoverDriverClient(name="alias", candidates=[first, second]).stream(
            GenerateRequest(messages=[], reportProgress=True)
        )
    ]
    assert events[0].progress == READS[1]
    assert "".join(e.text for e in events) == "East"
    assert len(second.calls) == 1


async def test_progress_is_not_the_first_token() -> None:
    # Time to first token is the first word, not the first frame: a CPU
    # box reading a long prompt would otherwise report a TTFT of zero.
    ended: dict[str, Any] = {}

    class Hooks:
        def on_attempt_start(self, driver: str, *, node: str | None = None) -> None:
            return None

        def on_attempt_end(self, driver: str, **facts: Any) -> None:
            ended.update(facts)

    fake = FakeDriverClient(name="box")
    fake.responses = ["East"]

    async def slow_reader(request: GenerateRequest) -> Any:
        yield StreamEvent(progress=READS[0])
        await asyncio.sleep(0.25)
        yield StreamEvent(text="East")
        async for event in FakeDriverClient.stream(fake, GenerateRequest(messages=[])):
            if event.done:
                yield event

    fake.stream = slow_reader  # type: ignore[method-assign]
    client = FailoverDriverClient(name="alias", candidates=[fake], hooks=Hooks())  # type: ignore[arg-type]
    _ = [e async for e in client.stream(GenerateRequest(messages=[], reportProgress=True))]
    assert ended["first_ms"] >= 200, ended


async def test_the_driver_client_reads_the_progress_event() -> None:
    body = (
        'event: progress\ndata: {"stage":"prompt","promptTokens":9,"processedTokens":4}\n\n'
        'event: progress\ndata: {"stage":"working"}\n\n'
        'event: token\ndata: {"text":"East"}\n\n'
        'event: done\ndata: {"content":"East","finishReason":"stop",'
        '"backend":"openai_compat_http"}\n\n'
    )

    def wire(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = HttpDriverClient(name="d", base_url="http://d.invalid")
    client._client = httpx.AsyncClient(
        base_url=client.base_url, transport=httpx.MockTransport(wire)
    )
    events = [e async for e in client.stream(GenerateRequest(messages=[], reportProgress=True))]
    assert [e.progress for e in events[:2]] == [
        {"stage": "prompt", "promptTokens": 9, "processedTokens": 4},
        {"stage": "working"},
    ]
    assert events[2].text == "East" and events[2].progress is None
    assert events[3].done


def test_a_driver_that_sends_progress_unasked_is_not_passed_on(settings: Settings) -> None:
    # The driver's contract is to send progress only when asked; a caller
    # who did not ask must not get a chunk with no choices even if one
    # does.
    fake = _fake()
    real = fake.stream

    async def eager(request: GenerateRequest) -> Any:
        yield StreamEvent(progress=READS[0])
        async for event in real(request):
            yield event

    fake.stream = eager  # type: ignore[method-assign]
    with _client(settings, fake) as client:
        r = client.post("/v1/chat/completions", json=_chat())
    assert r.status_code == 200, r.text
    assert all(f["choices"] for f in _frames(r.text)), r.text
