"""Serving regressions: evidence, chunk boundaries, cancellation and wire truth."""

from __future__ import annotations

import json
import logging
import random
import string
from collections.abc import AsyncGenerator

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import GenerateRequest
from eugene_plexus_gateway._generated.models import ConfigUpdateRequest
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.circuit import Circuit
from eugene_plexus_gateway.driver_client import (
    FailoverDriverClient,
    HttpDriverClient,
    RepetitionStopped,
    StreamEvent,
)
from eugene_plexus_gateway.repetition import HEADER, MAX_PERIOD, Detector, Guard, Policy
from eugene_plexus_gateway.routing import collect_attempts
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

# The user's reported two-paragraph cycle, used as one passage, not two
# unrelated repeated sentences. Emojis intentionally survive normalization.
PASSAGE = (
    "One more thing I can offer beyond the tool itself: I can reason through problems, "
    "write and explain code, summarize, draft, and so on — just using my built-in "
    "abilities. Want to put any of that to work? 😊\n"
    "One small note: the exact toolset can vary depending on how the session is "
    "configured, so if you're testing me, this is an honest snapshot of what's "
    "available right now. 🙂\n"
)


@pytest.mark.parametrize("chunk_size", [1, 7, 63, 64, 311, 100000])
@pytest.mark.parametrize("prefix", ["", "A useful introduction.\n", "先に説明します。\n"])
def test_reported_loop_is_independent_of_transport_chunks(chunk_size: int, prefix: str) -> None:
    text = prefix + PASSAGE * 10
    detector = Detector()
    for start in range(0, len(text), chunk_size):
        detector.feed(text[start : start + chunk_size])
    found = detector.detection
    assert found is not None
    period = len(" ".join(PASSAGE.split())) + 1
    assert found.period_chars == period
    assert period * 4 <= found.normalized_chars <= period * 4 + len(prefix) + 64
    assert len(detector.buffer) <= detector.window_chars


def test_normalization_across_chunks_and_four_complete_copies_at_eof() -> None:
    detector = Detector()
    for i in range(4):
        text = PASSAGE.replace(" ", "\t  ") if i % 2 else PASSAGE
        for char in text:
            detector.feed(char)
    detector.finish()
    assert detector.detection is not None


@pytest.mark.parametrize(
    "text",
    [
        PASSAGE * 3,
        (PASSAGE + "Substantive new information. ") * 3 + PASSAGE,
        "la " * 10000,
        "-" * 10000,
        "\n \t" * 10000,
        "\n".join(f"| {i} | same value |" for i in range(1000)),
        "\n".join(f"assert records[{i}].enabled is True" for i in range(1000)),
        "\n".join(f"x_{i + 1} = x_{i} + 1" for i in range(1000)),
    ],
    ids=[
        "three-copies",
        "new-information",
        "refrain",
        "separator",
        "whitespace",
        "table",
        "code",
        "math",
    ],
)
def test_repeated_formatting_and_progress_are_not_mechanical_loops(text: str) -> None:
    detector = Detector()
    detector.feed(text)
    detector.finish()
    assert detector.detection is None
    assert len(detector.buffer) <= detector.window_chars


@pytest.mark.parametrize("length", [100, 101, 256, 1024, MAX_PERIOD])
def test_period_range(length: int) -> None:
    rng = random.Random(length)
    passage = "".join(rng.choices(string.ascii_letters, k=length))
    detector = Detector()
    detector.feed(passage * 6)
    assert detector.detection is not None
    assert detector.detection.period_chars == length


def test_channels_and_structured_output_are_observed_without_stopping(caplog) -> None:
    caplog.set_level(logging.INFO)
    guard = Guard(Policy(mode="stop"), structured=True)
    assert not guard.take(StreamEvent(text=PASSAGE * 6))
    assert not guard.take(StreamEvent(reasoning=PASSAGE * 6))
    assert not guard.take(
        StreamEvent(tool_calls=[{"index": 0, "function": {"arguments": PASSAGE * 6}}])
    )
    assert len(guard.channels) == 3
    assert caplog.text.count("action=observe") == 3
    assert "One more thing" not in caplog.text

    separate = Guard(Policy(mode="stop"))
    separate.take(StreamEvent(text=PASSAGE * 2))
    separate.take(StreamEvent(reasoning=PASSAGE * 2))
    separate.take(StreamEvent(tool_calls=[{"index": 0, "function": {"arguments": PASSAGE * 2}}]))
    separate.take(StreamEvent(tool_calls=[{"index": 1, "function": {"arguments": PASSAGE * 2}}]))
    assert all(d.detection is None for d in separate.channels.values())
    for i in range(1000):
        separate.take(StreamEvent(tool_calls=[{"index": i, "function": {"arguments": "x"}}]))
    assert len(separate.channels) <= 10


def test_policy_defaults_model_selection_and_explicit_override() -> None:
    assert Policy.resolve({}.get, "qwen").mode == "observe"
    settings = {"repetitionMode": "stop", "repetitionStopModels": "qwen, another"}
    assert Policy.resolve(settings.get, "qwen").mode == "stop"
    assert Policy.resolve(settings.get, "third").mode == "observe"
    assert Policy.resolve(settings.get, "qwen", "off").mode == "off"
    assert Policy.resolve(settings.get, "third", "stop").mode == "stop"
    invalid = {"repetitionMode": "typo", "repetitionMinChars": True, "repetitionRepeats": 100000}
    assert Policy.resolve(invalid.get, "qwen") == Policy()


class LoopDriver(FakeDriverClient):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.closed = False
        self.produced = 0
        self.circuit = Circuit()

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        events = super().stream(request)
        try:
            async for event in events:
                self.produced += len(event.text)
                yield event
        finally:
            await events.aclose()
            self.closed = True


def test_stop_preserves_partial_text_closes_upstream_and_never_cascades(settings: Settings) -> None:
    first = LoopDriver(name="a-first", model_id="qwen")
    first.responses = [PASSAGE * 30]
    fallback = FakeDriverClient(name="z-fallback", model_id="qwen")
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(first, fallback)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={HEADER: "stop"},
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "tools?"}],
                "stream": True,
            },
        )
    assert response.status_code == 200
    frames = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert frames[-1]["error"]["code"] == "repetition_detected"
    assert "appears to be repeating" in frames[-1]["error"]["message"]
    partial = "".join(f.get("choices", [{}])[0].get("delta", {}).get("content", "") for f in frames)
    assert partial and partial in PASSAGE * 30
    assert first.closed and first.produced < len(PASSAGE * 6)
    assert len(first.calls) == 1 and not fallback.calls
    assert first.circuit.failures == 0
    assert not any(f.get("choices", [{}])[0].get("finish_reason") for f in frames)


@pytest.mark.parametrize("mode", [None, "observe", "off"])
def test_observation_and_off_leave_output_unchanged(
    settings: Settings, mode: str | None, caplog
) -> None:
    caplog.set_level(logging.INFO)
    fake = LoopDriver(name="first", model_id="qwen")
    fake.responses = [PASSAGE * 8]
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={HEADER: mode} if mode else {},
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "tools?"}],
                "stream": True,
            },
        )
    assert 'repetition_detected"' not in response.text
    assert '"finish_reason":"stop"' in response.text.replace(" ", "")
    assert fake.closed and fake.produced > len(PASSAGE * 7)
    assert ("repetition_detected channel=text" in caplog.text) == (mode != "off")


@pytest.mark.parametrize("door", ["/v1/messages", "/v1/responses"])
def test_other_streaming_doors_report_non_retryable_termination(
    settings: Settings, door: str
) -> None:
    fake = LoopDriver(name="first", model_id="qwen")
    fake.responses = [PASSAGE * 20]
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake)
    body = {"model": "qwen", "stream": True}
    if door.endswith("messages"):
        body.update(messages=[{"role": "user", "content": "tools?"}], max_tokens=20000)
    else:
        body["input"] = "tools?"
    with TestClient(app) as client:
        response = client.post(door, headers={HEADER: "stop"}, json=body)
    assert response.status_code == 200, response.text
    assert "appears to be repeating" in response.text
    assert (
        "invalid_prompt" if door.endswith("responses") else "invalid_request_error"
    ) in response.text
    assert first_output(response.text)
    assert fake.closed and fake.produced < len(PASSAGE * 6)


def first_output(text: str) -> bool:
    return "One" in text and "more" in text


async def test_http_response_is_closed_before_notifying_a_slow_client() -> None:
    class Wire(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            for _ in range(30):
                for char in PASSAGE:
                    yield ("event: token\ndata: " + json.dumps({"text": char}) + "\n\n").encode()

        async def aclose(self):
            self.closed = True

    wire = Wire()
    driver = HttpDriverClient(name="actual-http-client", base_url="http://driver.invalid")
    await driver._client.aclose()
    driver._client = httpx.AsyncClient(
        base_url="http://driver.invalid",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=wire)),
    )
    driver.circuit.failures = 1  # A recovery probe must be abandoned, not marked healthy.
    slot = FailoverDriverClient(name="qwen", candidates=[driver])
    slot.repetition_policy = Policy(mode="stop")
    notified_after_close = False
    try:
        with pytest.raises(RepetitionStopped):
            async for _ in slot.stream(GenerateRequest(messages=[])):
                notified_after_close = wire.closed
        assert notified_after_close and wire.closed
        assert not driver.circuit.probing and driver.circuit.failures == 1
    finally:
        await slot.aclose()


async def test_stopped_attempt_releases_accounting_and_records_terminal_reason() -> None:
    fake = LoopDriver(name="first", model_id="qwen")
    fake.responses = [PASSAGE * 10]
    table = make_routing_table(fake)
    slot = FailoverDriverClient(name="qwen", candidates=[fake], hooks=table)
    slot.repetition_policy = Policy(mode="stop")
    with collect_attempts() as rows, pytest.raises(RepetitionStopped):
        async for _ in slot.stream(GenerateRequest(messages=[])):
            assert table.inflight((None, fake.name)) == 1
    assert table.inflight((None, fake.name)) == 0
    assert len(rows) == 1
    assert not rows[0].served
    assert rows[0].error == "RepetitionStopped" and rows[0].retry_disposition == "terminal"


@pytest.mark.parametrize("structured", [False, True])
def test_reasoning_and_json_streams_finish_unchanged(settings: Settings, structured: bool) -> None:
    fake = LoopDriver(name="first", model_id="qwen")
    fake.reasoning = PASSAGE * 8
    fake.responses = [PASSAGE * 8 if structured else "An ordinary answer."]
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={HEADER: "stop"},
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "tools?"}],
                "stream": True,
                **({"response_format": {"type": "json_object"}} if structured else {}),
            },
        )
    assert response.status_code == 200 and '"error"' not in response.text
    assert fake.closed


def test_whole_response_observes_without_editing_completed_output(
    settings: Settings, caplog
) -> None:
    caplog.set_level(logging.INFO)
    fake = FakeDriverClient(name="first", model_id="qwen")
    fake.responses = [PASSAGE * 8]
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={HEADER: "stop"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "tools?"}]},
        )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == PASSAGE * 8
    assert "action=observe" in caplog.text


def test_bad_override_fails_before_generation_and_config_has_bounds(settings: Settings) -> None:
    fake = FakeDriverClient(name="first", model_id="qwen")
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={HEADER: "typo"},
            json={"model": "qwen", "messages": [{"role": "user", "content": "tools?"}]},
        )
        store = app.state.config_store
        result = store.apply_patch(
            ConfigUpdateRequest.model_validate({"repetitionMinChars": 10, "repetitionRepeats": 99})
        )
        assert len(result.rejected) == 2
    assert response.status_code == 400 and not fake.calls
