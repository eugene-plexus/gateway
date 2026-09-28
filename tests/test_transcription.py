"""POST /v1/audio/transcriptions: OpenAI's multipart form, tiers as chat (P3b).

The request shape is the OpenAI SDK's, captured 2026-09-28
(`provider-accounts-measurement.md` section 8): `file` with its filename and
type, `model`, and `language`, `prompt`, `response_format`, `temperature` and
`timestamp_granularities[]` when set. Every test here fails against the
gateway before P3b, which had no transcription door.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway import chat_contract
from eugene_plexus_gateway._generated.driver_models import (
    TranscribeRequest,
    TranscribeResponse,
    TranscriptionUsage,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

FOX = b"ID3\x04" + bytes(range(256)) * 4
SAID = "The quick brown fox jumps over the lazy dog."


class Scribe(FakeDriverClient):
    """A fake whose model transcribes, as a driver does since P3b -- and,
    with `translates`, translates too, as OpenAI's whisper does (P3-4)."""

    def __init__(
        self, *, fail: Exception | None = None, translates: bool = False, **kw: Any
    ) -> None:
        super().__init__(**kw)
        self.fail = fail
        self.translates = translates
        self.heard: list[TranscribeRequest] = []

    def describe(self):  # type: ignore[no-untyped-def]
        info = super().describe()
        for model in info.models or []:
            model.surfaces = (
                ["transcription", "translation"] if self.translates else ["transcription"]
            )
        return info

    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        self.heard.append(request)
        if self.fail is not None:
            raise self.fail
        return TranscribeResponse(
            text=SAID,
            language="english",
            duration=3.5,
            segments=[{"id": 0, "start": 0.0, "end": 3.5, "text": SAID}],
            words=[{"word": "The", "start": 0.0, "end": 0.2}],
            usage=TranscriptionUsage(seconds=3.5),
            modelId=request.model,
        )


def serve(settings: Settings, *drivers: FakeDriverClient, **table: Any) -> TestClient:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*drivers, **table)
    return TestClient(app)


def upload(client: TestClient, model: str = "scribe", **fields: Any) -> httpx.Response:
    files = fields.pop("files", {"file": ("fox.mp3", FOX, "audio/mpeg")})
    return client.post("/v1/audio/transcriptions", data={"model": model, **fields}, files=files)


def test_a_transcription_model_is_listed(settings: Settings) -> None:
    with serve(settings, Scribe(name="a", model_id="scribe")) as client:
        models = {m["id"]: m["x_eugene_plexus"] for m in client.get("/v1/models").json()["data"]}
    assert models["scribe"]["surfaces"] == ["transcription"]


def test_the_sdks_form_is_heard_and_answered_as_json(settings: Settings) -> None:
    scribe = Scribe(name="a", model_id="scribe")
    with serve(settings, scribe) as client:
        response = upload(client, language="en", prompt="Foxes.", temperature="0.2")
    assert response.status_code == 200, response.text
    assert response.json() == {"text": SAID, "usage": {"type": "duration", "seconds": 3.5}}
    [sent] = scribe.heard
    assert base64.b64decode(sent.audio.data) == FOX
    assert (sent.audio.filename, sent.audio.mediaType) == ("fox.mp3", "audio/mpeg")
    assert (sent.language, sent.prompt, sent.temperature, sent.verbose) == (
        "en",
        "Foxes.",
        0.2,
        False,
    )


def test_text_is_rendered_here(settings: Settings) -> None:
    with serve(settings, Scribe(name="a", model_id="scribe")) as client:
        response = upload(client, response_format="text")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/plain") and response.text == SAID


def test_verbose_json_asks_the_backend_and_carries_its_granularities(settings: Settings) -> None:
    scribe = Scribe(name="a", model_id="scribe")
    with serve(settings, scribe) as client:
        response = client.post(
            "/v1/audio/transcriptions",
            data={
                "model": "scribe",
                "response_format": "verbose_json",
                "timestamp_granularities[]": ["word", "segment"],
            },
            files={"file": ("fox.mp3", FOX, "audio/mpeg")},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["task"], body["language"], body["duration"]) == ("transcribe", "english", 3.5)
    assert body["segments"][0]["text"] == SAID
    [sent] = scribe.heard
    assert sent.verbose and [g.value for g in sent.timestampGranularities] == ["word", "segment"]


@pytest.mark.parametrize(
    ("fields", "param"),
    [
        ({"response_format": "srt"}, "response_format"),
        ({"response_format": "vtt"}, "response_format"),
        ({"stream": "true"}, "stream"),
        ({"chunking_strategy": "auto"}, "chunking_strategy"),
        ({"include[]": "logprobs"}, "include"),
        ({"speaker": "x"}, "speaker"),
        ({"timestamp_granularities[]": "word"}, "timestamp_granularities"),
        ({"temperature": "3"}, "temperature"),
        ({"files": None}, "file"),
    ],
)
def test_what_this_door_does_not_carry_is_refused_naming_the_field(
    settings: Settings, fields: dict[str, Any], param: str
) -> None:
    scribe = Scribe(name="a", model_id="scribe")
    with serve(settings, scribe) as client:
        response = upload(client, **fields)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == param
    assert not scribe.heard


def test_a_body_that_is_not_a_form_is_refused(settings: Settings) -> None:
    with serve(settings, Scribe(name="a", model_id="scribe")) as client:
        response = client.post("/v1/audio/transcriptions", json={"model": "scribe"})
    assert response.status_code == 400 and response.json()["error"]["param"] in ("body", "file")


def test_a_file_over_the_limit_is_a_413(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chat_contract, "MAX_UPLOAD_BYTES", 100)
    scribe = Scribe(name="a", model_id="scribe")
    with serve(settings, scribe) as client:
        response = upload(client)
    assert response.status_code == 413 and response.json()["error"]["param"] == "file"
    assert "25 MiB" in response.json()["error"]["message"] and not scribe.heard


def test_a_slot_cascades_past_a_dead_scribe_and_never_asks_a_chat_model(settings: Settings) -> None:
    chat = FakeDriverClient(name="c", model_id="chatty")
    dead = Scribe(name="a", model_id="primary", fail=httpx.ConnectError("refused"))
    backup = Scribe(name="b", model_id="backup")
    slots = [{"model": "scribe", "targets": ["chatty", "primary", "backup"]}]
    with serve(settings, chat, dead, backup, slots=slots) as client:
        response = upload(client)
        rows = _rows(client)
    assert response.status_code == 200, response.text
    assert dead.heard and backup.heard and not chat.calls
    assert rows[0]["servedModel"] == "backup" and rows[0]["tier"] == 3


def test_a_chat_model_is_sent_to_the_chat_door(settings: Settings) -> None:
    chat = FakeDriverClient(name="c", model_id="scribe")
    with serve(settings, chat) as client:
        response = upload(client)
    assert response.status_code == 400, response.text
    assert "/v1/chat/completions" in response.json()["error"]["message"]


# --------------------------------------------------------------------------- #
# /v1/audio/translations (P3-4)
# --------------------------------------------------------------------------- #


def translate(client: TestClient, model: str = "whisper", **fields: Any) -> httpx.Response:
    files = fields.pop("files", {"file": ("fr.mp3", FOX, "audio/mpeg")})
    return client.post("/v1/audio/translations", data={"model": model, **fields}, files=files)


def test_a_model_that_translates_is_listed_as_translating(settings: Settings) -> None:
    whisper = Scribe(name="a", model_id="whisper", translates=True)
    with serve(settings, whisper, Scribe(name="b", model_id="scribe")) as client:
        models = {m["id"]: m["x_eugene_plexus"] for m in client.get("/v1/models").json()["data"]}
    assert models["whisper"]["surfaces"] == ["transcription", "translation"]
    assert models["scribe"]["surfaces"] == ["transcription"]


def test_the_sdks_translation_form_is_sent_as_a_translation(settings: Settings) -> None:
    whisper = Scribe(name="a", model_id="whisper", translates=True)
    with serve(settings, whisper) as client:
        response = translate(client, prompt="Foxes.", temperature="0.1")
    assert response.status_code == 200, response.text
    assert response.json() == {"text": SAID, "usage": {"type": "duration", "seconds": 3.5}}
    [sent] = whisper.heard
    assert sent.translate and base64.b64decode(sent.audio.data) == FOX
    assert (sent.prompt, sent.temperature, sent.language, sent.timestampGranularities) == (
        "Foxes.",
        0.1,
        None,
        None,
    )


def test_a_verbose_translation_is_openais_translate_shape(settings: Settings) -> None:
    whisper = Scribe(name="a", model_id="whisper", translates=True)
    with serve(settings, whisper) as client:
        verbose = translate(client, response_format="verbose_json")
        text = translate(client, response_format="text")
    assert verbose.status_code == 200, verbose.text
    body = verbose.json()
    assert (body["task"], body["language"], body["duration"]) == ("translate", "english", 3.5)
    assert "words" not in body  # a translation carries segments, never words
    assert text.headers["content-type"].startswith("text/plain") and text.text == SAID


@pytest.mark.parametrize(
    ("fields", "param", "said"),
    [
        ({"language": "fr"}, "language", "always English"),
        ({"timestamp_granularities[]": "word"}, "timestamp_granularities", "translation"),
        ({"stream": "true"}, "stream", "one JSON document"),
        ({"chunking_strategy": "auto"}, "chunking_strategy", "not a field"),
        ({"response_format": "srt"}, "response_format", "srt"),
    ],
)
def test_what_a_translation_does_not_take_is_refused_naming_the_field(
    settings: Settings, fields: dict[str, Any], param: str, said: str
) -> None:
    whisper = Scribe(name="a", model_id="whisper", translates=True)
    with serve(settings, whisper) as client:
        response = translate(client, **fields)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == param and said in error["message"]
    assert not whisper.heard


def test_a_model_that_only_transcribes_is_sent_to_the_transcription_door(
    settings: Settings,
) -> None:
    scribe = Scribe(name="a", model_id="scribe")
    with serve(settings, scribe) as client:
        response = translate(client, "scribe")
    assert response.status_code == 400, response.text
    assert "/v1/audio/transcriptions" in response.json()["error"]["message"]
    assert not scribe.heard


def test_a_translation_slot_skips_a_model_that_would_answer_in_french(
    settings: Settings,
) -> None:
    """A transcript is not a translation: a fallback tier that only
    transcribes would answer in the language spoken, with a 200."""
    scribe = Scribe(name="a", model_id="scribe")
    dead = Scribe(name="b", model_id="primary", translates=True, fail=httpx.ConnectError("x"))
    backup = Scribe(name="c", model_id="backup", translates=True)
    slots = [{"model": "english", "targets": ["scribe", "primary", "backup"]}]
    with serve(settings, scribe, dead, backup, slots=slots) as client:
        response = translate(client, "english")
        rows = _rows(client)
    assert response.status_code == 200, response.text
    assert not scribe.heard and dead.heard and backup.heard
    assert rows[0]["servedModel"] == "backup" and rows[0]["tier"] == 3
    assert (rows[0]["door"], rows[0]["audioSeconds"]) == ("translation", 3.5)


def _rows(client: TestClient) -> list[dict[str, Any]]:
    for _ in range(50):
        rows = client.get("/v1/metrics/requests").json()["requests"]
        if rows:
            return rows
        time.sleep(0.02)
    return []


def test_a_transcription_is_retained_in_seconds_of_audio(settings: Settings) -> None:
    with serve(settings, Scribe(name="a", model_id="scribe")) as client:
        assert upload(client).status_code == 200
        [row] = _rows(client)
    assert (row["door"], row["audioSeconds"], row["outcome"]) == ("transcription", 3.5, "served")
    assert row["characters"] is None and row["servedModel"] == "scribe"


@pytest.mark.parametrize(
    ("door", "wanted"),
    [
        ("/v1/chat/completions", "/v1/audio/transcriptions"),
        ("/v1/audio/speech", "/v1/audio/transcriptions"),
    ],
)
def test_a_wrong_door_names_the_models_own(settings: Settings, door: str, wanted: str) -> None:
    """With five surfaces, one door per caller was wrong for most models: a
    transcription model sent to chat was told to use /v1/embeddings."""
    body = (
        {"model": "scribe", "messages": [{"role": "user", "content": "hi"}]}
        if door == "/v1/chat/completions"
        else {"model": "scribe", "input": "hi", "voice": "x"}
    )
    with serve(settings, Scribe(name="a", model_id="scribe")) as client:
        response = client.post(door, json=body)
    assert response.status_code == 400, response.text
    message = response.json()["error"]["message"]
    assert f"Send this request to {wanted} instead" in message, message
