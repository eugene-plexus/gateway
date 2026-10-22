"""Serving regressions: evidence, chunk boundaries, cancellation and wire truth."""

from __future__ import annotations

import json
import logging
import random
import string
from collections.abc import AsyncGenerator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import GenerateRequest
from eugene_plexus_gateway._generated.models import ConfigUpdateRequest
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.circuit import Circuit
from eugene_plexus_gateway.config import ConfigStore
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


def assert_names_cause_and_remedy(message: str) -> None:
    """A stop says what it saw and how to turn it off, in every door's words."""
    period = len(" ".join(PASSAGE.split())) + 1
    assert f"repeated the same {period}-character passage 4 times in a row" in message
    assert "stopped because it repeated" in message
    assert "text already sent is kept" in message
    assert f"{HEADER}: off" in message
    assert "Repeated response protection" in message


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
    # Row 19's evaluation did not justify default enforcement.
    assert Policy().mode == "observe"
    assert Policy.resolve({}.get, "qwen").mode == "observe"
    assert Policy.resolve({"repetitionMode": "observe"}.get, "qwen").mode == "observe"
    assert Policy.resolve({"repetitionMode": "off"}.get, "qwen").mode == "off"
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
    assert frames[-1]["finish_reason"] == "repetition_detected"
    assert_names_cause_and_remedy(frames[-1]["error"]["message"])
    partial = "".join(f.get("choices", [{}])[0].get("delta", {}).get("content", "") for f in frames)
    assert partial and partial in PASSAGE * 30
    assert first.closed and first.produced < len(PASSAGE * 6)
    assert len(first.calls) == 1 and not fallback.calls
    assert first.circuit.failures == 0
    assert not any(f.get("choices", [{}])[0].get("finish_reason") for f in frames)


@pytest.mark.parametrize("mode", ["observe", "off"])
def test_observation_and_off_leave_output_unchanged(settings: Settings, mode: str, caplog) -> None:
    caplog.set_level(logging.INFO)
    fake = LoopDriver(name="first", model_id="qwen")
    fake.responses = [PASSAGE * 8]
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={HEADER: mode},
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


def _stream_chat(
    app, *, headers: dict[str, str] | None = None, mode: str | None = None
) -> list[dict]:
    with TestClient(app) as client:
        if mode is not None:  # The operator's choice, made in the console.
            app.state.config_store.apply_patch(
                ConfigUpdateRequest.model_validate({"repetitionMode": mode})
            )
        response = client.post(
            "/v1/chat/completions",
            headers=headers or {},
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "go"}],
                "stream": True,
            },
        )
    assert response.status_code == 200
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _said(frames: list[dict]) -> str:
    return "".join(f.get("choices", [{}])[0].get("delta", {}).get("content", "") for f in frames)


@pytest.mark.parametrize("mode", [None, "stop"])
def test_default_and_console_stop_settings_match_the_wire(
    settings: Settings, mode: str | None
) -> None:
    """Observe remains the default; the expert's Stop choice really stops."""
    first = LoopDriver(name="first", model_id="qwen")
    first.responses = [PASSAGE * 40]
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(first)
    frames = _stream_chat(app, mode=mode)
    assert first.closed
    if mode == "stop":
        assert frames[-1]["error"]["code"] == "repetition_detected"
        assert frames[-1]["finish_reason"] == "repetition_detected"
        assert first.produced < len(PASSAGE * 6)
        assert _said(frames) and _said(frames) in PASSAGE * 40
    else:
        assert _said(frames) == (PASSAGE * 40).rstrip()
        assert not any("error" in frame for frame in frames)
    with TestClient(app) as client:
        schema = client.get("/v1/config/schema").json()
        document = client.get("/v1/config").json()
    field = next(f for f in schema["fields"] if f["key"] == "repetitionMode")
    assert field["default"] == "observe"
    assert document["repetitionMode"] == (mode or "observe")


# All legitimate answers from the labelled specs evaluation set. Keep the fixture
# local so this repository's CI does not depend on a sibling checkout.
def _legitimate_repetition() -> dict[str, str]:
    return {
        case["id"]: case["text"]
        for line in (Path(__file__).parent / "fixtures/repetition-legitimate.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if (case := json.loads(line))
    }


@pytest.mark.parametrize("name", sorted(_legitimate_repetition()))
def test_default_settings_leave_legitimate_repetition_untouched(
    settings: Settings, name: str
) -> None:
    class VerbatimDriver(FakeDriverClient):
        async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
            result = await self.generate(request)
            content = result.content
            assert isinstance(content, str)
            for offset in range(0, len(content), 7):
                yield StreamEvent(text=content[offset : offset + 7])
            yield StreamEvent(done=True, result=result)

    text = _legitimate_repetition()[name]
    fake = VerbatimDriver(name="first", model_id="qwen")
    fake.responses = [text]
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake)
    frames = _stream_chat(app)
    assert not any("error" in f for f in frames), name
    assert _said(frames) == text, name
    assert frames[-1]["choices"][0]["finish_reason"] == "stop", name


def test_a_chosen_observe_or_off_is_kept(settings: Settings) -> None:
    """An explicit choice applies immediately and is visible in the console."""
    for mode in ("observe", "off"):
        fake = LoopDriver(name="first", model_id="qwen")
        fake.responses = [PASSAGE * 8]
        app = create_app(settings=settings)
        app.state.routing = make_routing_table(fake)
        frames = _stream_chat(app, mode=mode)
        assert not any("error" in f for f in frames), mode
        assert _said(frames) == (PASSAGE * 8).rstrip(), mode


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
    assert_names_cause_and_remedy(response.text)
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


def _store(path) -> ConfigStore:
    store = ConfigStore(path)
    store.load()
    return store


def test_a_stored_observe_is_preserved(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("logLevel: INFO\nrepetitionMode: observe\n", encoding="utf-8")
    assert _store(path).get("repetitionMode") == "observe"
    assert "repetitionMode: observe" in path.read_text(encoding="utf-8")
    store = _store(path)
    store.apply_patch(ConfigUpdateRequest.model_validate({"repetitionMode": "observe"}))
    assert _store(path).get("repetitionMode") == "observe"


def test_other_stored_modes_and_a_fresh_install(tmp_path) -> None:
    for stored in ("off", "stop"):
        path = tmp_path / f"{stored}.yaml"
        path.write_text(f"repetitionMode: {stored}\n", encoding="utf-8")
        assert _store(path).get("repetitionMode") == stored
    fresh = tmp_path / "new" / "config.yaml"
    assert _store(fresh).get("repetitionMode") == "observe"
    store = _store(fresh)
    store.apply_patch(ConfigUpdateRequest.model_validate({"repetitionMode": "observe"}))
    assert _store(fresh).get("repetitionMode") == "observe"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("repetitionMode: off\n", Policy(mode="off")),
        ("repetitionMode: typo\nrepetitionMinChars: true\nrepetitionRepeats: 99\n", Policy()),
        (
            "repetitionMode: stop\nrepetitionMinChars: 384\nrepetitionRepeats: 5\n",
            Policy(mode="stop", min_chars=384, repeats=5),
        ),
    ],
)
def test_loaded_repetition_settings_show_the_policy_in_effect(tmp_path, raw, expected) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(raw, encoding="utf-8")
    store = _store(path)
    assert store.get("repetitionMode") == expected.mode
    assert store.get("repetitionMinChars") == expected.min_chars
    assert store.get("repetitionRepeats") == expected.repeats
    assert Policy.resolve(store.get, "qwen") == expected


def test_legacy_completions_stop_names_the_finish_reason(settings: Settings) -> None:
    class LoopCompleter(LoopDriver):
        def describe(self):
            info = super().describe()
            for model in info.models or []:
                model.surfaces = ["chat", "completion"]
            return info

    fake = LoopCompleter(name="first", model_id="qwen")
    fake.responses = [PASSAGE * 20]
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake)
    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            headers={HEADER: "stop"},
            json={"model": "qwen", "prompt": "go", "stream": True},
        )
    assert response.status_code == 200, response.text
    frames = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert frames[-1]["finish_reason"] == "repetition_detected"
    assert frames[-1]["error"]["code"] == "repetition_detected"
    assert_names_cause_and_remedy(frames[-1]["error"]["message"])
    assert fake.closed
