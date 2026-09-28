"""A spoken answer through chat, routed only to a model that speaks (P2b).

Every test here fails against the gateway as it was before P2b, whose
chat request had no `modalities` or `audio` (400 at the schema, as
unknown fields) and whose responses and deltas had no `audio` to carry.
Measured behaviour behind the rules: `provider-accounts-measurement.md`
section 4 (2026-09-28).
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import (
    AudioOutputFormat,
    Capabilities,
    FinishReason,
    GeneratedAudio,
    GenerateRequest,
    GenerateResponse,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import StreamEvent
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(28)
MP3 = b"ID3\x03\x00\x00\x00\x00\x00\x00" + bytes(32)
PCM = bytes(range(256))


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


class Speaks(FakeDriverClient):
    """A fake whose model answers with audio, as a driver does since P2b."""

    def __init__(self, speaks: bool = True, *, clip: GeneratedAudio | None = None, **kw: Any):
        super().__init__(**kw)
        self.speaks = speaks
        self.clip = clip or GeneratedAudio(
            data=b64(WAV),
            format=AudioOutputFormat.wav,
            id="audio_1",
            transcript="Hello there",
            expiresAt=1790609721,
        )

    def describe(self):  # type: ignore[no-untyped-def]
        info = super().describe()
        for model in info.models or []:
            base = model.capabilities or Capabilities()
            model.capabilities = base.model_copy(update={"audioOutput": self.speaks})
        return info

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.calls.append(request)
        return GenerateResponse(
            content=None if request.audioOutput else "text",
            audio=self.clip if request.audioOutput else None,
            finishReason=FinishReason.stop,
            backend=self.backend,
            modelId=request.model or self.model_id,
            latencyMs=1,
        )

    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamEvent]:
        self.calls.append(request)
        yield StreamEvent(audio={"id": "audio_1", "transcript": "Hello"})
        yield StreamEvent(audio={"data": b64(PCM[:128]), "format": "pcm16", "expiresAt": 17906})
        yield StreamEvent(audio={"data": b64(PCM[128:]), "transcript": " there"})
        yield StreamEvent(
            done=True,
            result=GenerateResponse(
                finishReason=FinishReason.stop,
                backend=self.backend,
                modelId=request.model or self.model_id,
                latencyMs=1,
            ),
        )


def serve(settings: Settings, *drivers: FakeDriverClient, **table: Any) -> TestClient:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*drivers, **table)
    return TestClient(app)


def speak(model: str = "voice", fmt: str = "wav", **extra: Any) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello."}],
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": fmt},
        **extra,
    }


def test_models_say_which_one_speaks(settings: Settings) -> None:
    with serve(
        settings, Speaks(name="a", model_id="voice"), Speaks(False, name="b", model_id="text")
    ) as client:
        models = {m["id"]: m["x_eugene_plexus"] for m in client.get("/v1/models").json()["data"]}
    assert models["voice"]["audio_output"] is True
    assert models["text"]["audio_output"] is False


def test_a_spoken_answer_skips_a_tier_that_cannot_speak(settings: Settings) -> None:
    text = Speaks(False, name="text", model_id="text-only")
    voice = Speaks(name="voice", model_id="voice")
    slots = [{"model": "assistant", "targets": ["text-only", "voice"]}]
    with serve(settings, text, voice, slots=slots) as client:
        response = client.post("/v1/chat/completions", json=speak("assistant"))
    assert response.status_code == 200, response.text
    assert not text.calls, "a model that cannot speak was asked to"
    asked = voice.calls[-1].audioOutput
    assert asked is not None and asked.voice == "alloy" and asked.format.value == "wav"
    body = response.json()
    message = body["choices"][0]["message"]
    assert message["audio"] == {
        "id": "audio_1",
        "data": b64(WAV),
        "format": "wav",
        "transcript": "Hello there",
        "expires_at": 1790609721,
    }
    assert message["content"] is None
    assert body["x_eugene_plexus"]["tier"] == 2


def test_the_format_is_what_the_bytes_are_not_what_was_asked(settings: Settings) -> None:
    """P2-2: Lyria answers MP3 asked for WAV, and the driver says so."""
    lyria = Speaks(name="lyria", model_id="voice", clip=GeneratedAudio(data=b64(MP3), format="mp3"))
    with serve(settings, lyria) as client:
        audio = client.post("/v1/chat/completions", json=speak(fmt="wav")).json()["choices"][0][
            "message"
        ]["audio"]
    assert audio["format"] == "mp3" and base64.b64decode(audio["data"]) == MP3


def test_a_streamed_answer_carries_each_fragment_as_delta_audio(settings: Settings) -> None:
    voice = Speaks(name="voice", model_id="voice")
    with serve(settings, voice) as client:
        response = client.post("/v1/chat/completions", json=speak(fmt="pcm16", stream=True))
    assert response.status_code == 200, response.text
    frames = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")
    ]
    fragments = [
        c["delta"]["audio"] for f in frames for c in f.get("choices") or [] if "audio" in c["delta"]
    ]
    assert fragments[0] == {"id": "audio_1", "transcript": "Hello"}
    assert fragments[1]["format"] == "pcm16" and fragments[1]["expires_at"] == 17906
    assert b"".join(base64.b64decode(f["data"]) for f in fragments if "data" in f) == PCM
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"
    assert voice.calls[-1].audioOutput is not None


@pytest.mark.parametrize(
    ("body", "field", "said"),
    [
        (speak(fmt="mp3"), "audio.format", "wav or pcm16"),
        (speak(fmt="flac"), "audio.format", "wav or pcm16"),
        (speak(fmt="wav", stream=True), "audio.format", "pcm16"),
        (
            {
                "model": "voice",
                "messages": [{"role": "user", "content": "hi"}],
                "audio": {"voice": "alloy", "format": "wav"},
            },
            "audio",
            "modalities",
        ),
        (
            {
                "model": "voice",
                "messages": [{"role": "user", "content": "hi"}],
                "modalities": ["text", "audio"],
            },
            "audio",
            "voice and format",
        ),
        (
            {
                "model": "voice",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "audio": {"id": "audio_1"}},
                    {"role": "user", "content": "again"},
                ],
            },
            "messages[1].audio",
            "keeps none",
        ),
    ],
)
def test_what_cannot_be_served_is_refused_before_anything_is_sent(
    settings: Settings, body: dict, field: str, said: str
) -> None:
    voice = Speaks(name="voice", model_id="voice")
    with serve(settings, voice) as client:
        response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == field and said in error["message"], error
    assert not voice.calls


def test_with_no_model_that_speaks_it_is_a_400_naming_the_field(settings: Settings) -> None:
    text = Speaks(False, name="text", model_id="voice")
    with serve(settings, text) as client:
        response = client.post("/v1/chat/completions", json=speak())
    assert response.status_code == 400, response.text
    assert "x_eugene_plexus.audio_output" in response.json()["error"]["message"]
    assert not text.calls


def test_a_text_request_asks_for_no_audio(settings: Settings) -> None:
    voice = Speaks(name="voice", model_id="voice")
    with serve(settings, voice) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "voice", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200, response.text
    assert voice.calls[-1].audioOutput is None
    assert response.json()["choices"][0]["message"].get("audio") is None
