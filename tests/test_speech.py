"""POST /v1/audio/speech: same model only, streamed bytes (P3a).

Every test here fails against the gateway as it was before P3a, which had
no speech door and did not list a speech model at all. The request shapes
are the OpenAI SDK's, captured 2026-09-28
(`provider-accounts-measurement.md` section 8).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import (
    Capabilities,
    SpeakRequest,
    SpeechFormat,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import DriverError
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

MP3 = b"ID3\x04" + bytes(range(100))


class Speaker(FakeDriverClient):
    """A fake whose model speaks, as a driver does since P3a."""

    def __init__(
        self,
        *,
        formats: list[str] | None = None,
        voices: list[str] | None = None,
        fail: Exception | None = None,
        fail_after_first: bool = False,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.formats = formats or ["mp3", "pcm", "wav"]
        self.voices = voices
        self.fail = fail
        self.fail_after_first = fail_after_first
        self.spoken: list[SpeakRequest] = []

    def describe(self):  # type: ignore[no-untyped-def]
        info = super().describe()
        for model in info.models or []:
            model.surfaces = ["speech"]
            model.voices = self.voices
            base = model.capabilities or Capabilities()
            model.capabilities = base.model_copy(
                update={"speechFormats": [SpeechFormat(f) for f in self.formats]}
            )
        return info

    async def speak(self, request: SpeakRequest) -> AsyncIterator[str | bytes]:
        self.spoken.append(request)
        if self.fail is not None:
            raise self.fail
        yield "audio/mpeg"
        yield MP3[:50]
        if self.fail_after_first:
            raise httpx.ReadError("the backend went away")
        yield MP3[50:]


def serve(settings: Settings, *drivers: FakeDriverClient, **table: Any) -> TestClient:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*drivers, **table)
    return TestClient(app)


def speech(model: str = "narrator", **extra: Any) -> dict[str, Any]:
    return {"model": model, "input": "Hello there.", "voice": "af_heart", **extra}


def test_a_speech_model_is_listed_with_its_voices_and_formats(settings: Settings) -> None:
    voice = Speaker(name="a", model_id="narrator", voices=["af_heart", "af_bella"])
    with serve(settings, voice) as client:
        models = {m["id"]: m["x_eugene_plexus"] for m in client.get("/v1/models").json()["data"]}
    assert models["narrator"]["surfaces"] == ["speech"]
    assert models["narrator"]["voices"] == ["af_heart", "af_bella"]
    assert models["narrator"]["speech_formats"] == ["mp3", "wav", "pcm"]


def test_the_sdks_request_is_spoken_and_streamed_back(settings: Settings) -> None:
    voice = Speaker(name="a", model_id="narrator")
    with serve(settings, voice) as client:
        response = client.post(
            "/v1/audio/speech",
            json=speech(speed=1.1, instructions="cheerful", stream_format="audio"),
        )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "audio/mpeg" and response.content == MP3
    sent = voice.spoken[-1]
    assert (sent.input, sent.voice, sent.format.value) == ("Hello there.", "af_heart", "mp3")
    assert (sent.speed, sent.instructions, sent.localOnly) == (1.1, "cheerful", False)


def test_what_served_the_clip_rides_the_headers(settings: Settings) -> None:
    """The body is the audio, so the envelope rides response headers, as on
    /v1/messages (U4): the playground's report reads them."""
    with serve(settings, Speaker(name="kokoro-a", model_id="narrator")) as client:
        response = client.post("/v1/audio/speech", json=speech())
    assert response.status_code == 200, response.text
    assert response.headers["x-eugene-plexus-driver"] == "kokoro-a"
    assert response.headers["x-eugene-plexus-attempts"] == "1"
    assert response.headers["x-eugene-plexus-tier"] == "1"
    assert int(response.headers["x-eugene-plexus-latency-ms"]) >= 0


def test_the_format_is_always_sent_mp3_by_default(settings: Settings) -> None:
    voice = Speaker(name="a", model_id="narrator")
    with serve(settings, voice) as client:
        client.post("/v1/audio/speech", json=speech())
    assert voice.spoken[-1].format.value == "mp3"


@pytest.mark.parametrize(
    ("extra", "param"),
    [
        ({"stream_format": "sse"}, "stream_format"),
        ({"speaker": "x"}, "speaker"),
        ({"speed": 9}, "speed"),
        ({"response_format": "opus"}, "response_format"),
    ],
)
def test_what_this_door_cannot_serve_is_refused_naming_the_field(
    settings: Settings, extra: dict, param: str
) -> None:
    voice = Speaker(name="a", model_id="narrator")
    with serve(settings, voice) as client:
        response = client.post("/v1/audio/speech", json=speech(**extra))
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == param
    assert not voice.spoken


def test_a_refused_format_names_the_ones_the_model_can_make(settings: Settings) -> None:
    voice = Speaker(name="a", model_id="narrator")
    with serve(settings, voice) as client:
        response = client.post("/v1/audio/speech", json=speech(response_format="aac"))
    assert response.status_code == 400
    assert "mp3, wav, pcm" in response.json()["error"]["message"]


def test_a_voice_the_model_does_not_list_is_refused_before_sending(settings: Settings) -> None:
    """M10 (2026-10-08): OpenRouter's own refusal of an unknown voice said
    only "Provider returned 400". Every backend lists its voices here, so
    the gateway refuses, naming them, and nothing is sent."""
    voice = Speaker(name="a", model_id="narrator", voices=["af_heart", "af_bella"])
    with serve(settings, voice) as client:
        refused = client.post("/v1/audio/speech", json=speech(voice="no_such_voice"))
        taken = client.post("/v1/audio/speech", json=speech(voice="af_bella"))
    assert refused.status_code == 400, refused.text
    error = refused.json()["error"]
    assert error["param"] == "voice"
    assert "'no_such_voice'" in error["message"] and "af_heart, af_bella" in error["message"]
    assert "Nothing was sent" in error["message"]
    assert taken.status_code == 200
    assert [s.voice for s in voice.spoken] == ["af_bella"]


def test_a_voice_is_passed_through_when_any_backend_leaves_it_to_its_provider(
    settings: Settings,
) -> None:
    """One replica that lists nothing makes the voice its provider's to
    check (P3-3): the gateway cannot know the voice is wrong there."""
    listed = Speaker(name="a", model_id="narrator", voices=["af_heart"])
    unlisted = Speaker(name="b", model_id="narrator", voices=None)
    with serve(settings, listed, unlisted) as client:
        response = client.post("/v1/audio/speech", json=speech(voice="rachel"))
    assert response.status_code == 200, response.text
    with serve(settings, Speaker(name="c", model_id="narrator", voices=None)) as client:
        assert client.post("/v1/audio/speech", json=speech(voice="any")).status_code == 200


def test_a_slot_alias_checks_the_voices_of_its_first_model(settings: Settings) -> None:
    first = Speaker(name="a", model_id="narrator", voices=["af_heart"])
    second = Speaker(name="c", model_id="other-voice", voices=["rachel"])
    slots = [{"model": "voice", "targets": ["narrator", "other-voice"]}]
    with serve(settings, first, second, slots=slots) as client:
        response = client.post("/v1/audio/speech", json=speech(model="voice", voice="rachel"))
    assert response.status_code == 400 and response.json()["error"]["param"] == "voice"
    assert not first.spoken and not second.spoken


def test_speech_fails_over_between_replicas_and_never_to_another_model(settings: Settings) -> None:
    dead = Speaker(name="a", model_id="narrator", fail=httpx.ConnectError("refused"))
    alive = Speaker(name="b", model_id="narrator")
    other = Speaker(name="c", model_id="other-voice")
    slots = [{"model": "narrator", "targets": ["other-voice"]}]
    with serve(settings, dead, alive, other, slots=slots) as client:
        for _ in range(3):
            response = client.post("/v1/audio/speech", json=speech())
            assert response.status_code == 200, response.text
            assert response.content == MP3
    assert alive.spoken and not other.spoken, "a slot's other target is a different voice"


def test_with_every_replica_down_the_slot_target_is_still_not_used(settings: Settings) -> None:
    dead = Speaker(
        name="a",
        model_id="narrator",
        fail=DriverError(
            driver_name="a", driver_url="http://a", status_code=500, problem=None, raw_body="died"
        ),
    )
    other = Speaker(name="c", model_id="other-voice")
    slots = [{"model": "narrator", "targets": ["other-voice"]}]
    with serve(settings, dead, other, slots=slots) as client:
        response = client.post("/v1/audio/speech", json=speech())
    assert response.status_code >= 500, response.text
    assert not other.spoken


def test_a_slot_alias_speaks_with_its_first_model(settings: Settings) -> None:
    first = Speaker(name="a", model_id="narrator", voices=["af_heart"], formats=["mp3"])
    second = Speaker(name="c", model_id="other-voice", voices=["rachel"])
    slots = [{"model": "voice", "targets": ["narrator", "other-voice"]}]
    with serve(settings, first, second, slots=slots) as client:
        response = client.post("/v1/audio/speech", json=speech(model="voice"))
        listed = {m["id"]: m["x_eugene_plexus"] for m in client.get("/v1/models").json()["data"]}
    assert response.status_code == 200, response.text
    assert first.spoken and not second.spoken
    # What the alias is listed with is what it will be answered with.
    assert listed["voice"]["voices"] == ["af_heart"]
    assert listed["voice"]["speech_formats"] == ["mp3"]


def test_a_slot_aliass_first_model_down_is_not_replaced_by_its_second(
    settings: Settings,
) -> None:
    first = Speaker(name="a", model_id="narrator", fail=httpx.ConnectError("refused"))
    second = Speaker(name="c", model_id="other-voice")
    slots = [{"model": "voice", "targets": ["narrator", "other-voice"]}]
    with serve(settings, first, second, slots=slots) as client:
        response = client.post("/v1/audio/speech", json=speech(model="voice"))
    assert response.status_code >= 500, response.text
    assert first.spoken and not second.spoken


def test_after_the_first_byte_a_failure_ends_the_audio_and_does_not_cascade(
    settings: Settings,
) -> None:
    breaks = Speaker(name="a", model_id="narrator", fail_after_first=True)
    spare = Speaker(name="b", model_id="narrator")
    with serve(settings, breaks, spare) as client:
        responses = [client.post("/v1/audio/speech", json=speech()) for _ in range(4)]
    cut = [r for r in responses if r.content == MP3[:50]]
    assert cut, "the breaking replica was never chosen"
    assert all(r.status_code == 200 for r in cut)
    # A replica that failed after its first byte did not hand over to the other.
    assert len(breaks.spoken) + len(spare.spoken) == 4


def _rows(client: TestClient) -> list[dict[str, Any]]:
    for _ in range(50):
        rows = client.get("/v1/metrics/requests").json()["requests"]
        if rows:
            return rows
        time.sleep(0.02)
    return []


def test_a_served_clip_is_retained_as_served(settings: Settings) -> None:
    """Left to the admission middleware's fallback row, which knows only
    embeddings, every served clip was retained as an error with no served
    model (the P3a acceptance run found it)."""
    voice = Speaker(name="a", model_id="narrator")
    with serve(settings, voice) as client:
        assert client.post("/v1/audio/speech", json=speech()).status_code == 200
        [row] = _rows(client)
    assert (row["outcome"], row["servedModel"], row["streamed"]) == ("served", "narrator", True)
    assert [t["served"] for t in row["tries"]] == [True]
    # P3b: speech is billed in characters, and the row says so.
    assert (row["door"], row["characters"], row["audioSeconds"]) == (
        "speech",
        len("Hello there."),
        None,
    )


def test_a_failed_clip_is_retained_as_an_error_with_its_attempt(settings: Settings) -> None:
    dead = Speaker(name="a", model_id="narrator", fail=httpx.ConnectError("refused"))
    with serve(settings, dead) as client:
        assert client.post("/v1/audio/speech", json=speech()).status_code >= 500
        [row] = _rows(client)
    assert row["outcome"] == "error" and [t["served"] for t in row["tries"]] == [False]


def test_a_chat_model_is_refused_at_the_speech_door(settings: Settings) -> None:
    chat = FakeDriverClient(name="chat", model_id="narrator")
    with serve(settings, chat) as client:
        response = client.post("/v1/audio/speech", json=speech())
    assert response.status_code == 400, response.text
    assert "chat" in response.json()["error"]["message"]
