"""Audio and PDFs at every door, routed only where they are confirmed (P2).

Each test here fails against the gateway as it was before 2026-09-28:
the chat door refused `input_audio` and `file` parts at the schema, the
Anthropic door refused every `document` block, the Responses door
refused `input_file` and `input_audio`, and routing knew one kind of
attachment, images.

The done-when this file carries: an audio question and a PDF question
are answered through chat, and a text-only backend is routed around,
never sent the audio.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway import images
from eugene_plexus_gateway._generated.driver_models import Capabilities
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import DriverError
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

MODEL = "gemini-2.5-flash-lite"
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(28)
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + bytes(32)
PDF = b"%PDF-1.4\n1 0 obj << >> endobj\ntrailer << >>\n%%EOF\n"
PDF_URL = "data:application/pdf;base64," + base64.b64encode(PDF).decode()


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


class Takes(FakeDriverClient):
    """A fake whose model confirms the given kinds of input."""

    def __init__(self, *kinds: str, **kwargs: Any) -> None:
        kwargs.setdefault("model_id", MODEL)
        kwargs.setdefault("supports_tools", True)
        super().__init__(**kwargs)
        self.kinds = set(kinds)

    def describe(self):  # type: ignore[no-untyped-def]
        info = super().describe()
        for model in info.models or []:
            base = model.capabilities or Capabilities()
            model.capabilities = base.model_copy(
                update={
                    "imageInput": "image" in self.kinds,
                    "audioInput": "audio" in self.kinds,
                    "fileInput": "file" in self.kinds,
                }
            )
        return info


def serve(settings: Settings, *drivers: FakeDriverClient, **table: Any) -> TestClient:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*drivers, **table)
    return TestClient(app)


def audio_part(raw: bytes = MP3, fmt: str = "mp3") -> dict[str, Any]:
    return {"type": "input_audio", "input_audio": {"data": b64(raw), "format": fmt}}


def file_part(data: str = PDF_URL, **extra: Any) -> dict[str, Any]:
    return {"type": "file", "file": {"filename": "note.pdf", "file_data": data, **extra}}


def chat(*parts: dict[str, Any], model: str = MODEL) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "What is in this?"}, *parts]}
        ],
    }


def sent_parts(fake: FakeDriverClient) -> list[dict[str, Any]]:
    return fake.calls[-1].model_dump(mode="json", exclude_none=True)["messages"][-1]["content"]


# --------------------------------------------------------------------------- #
# Chat: answered, carried unchanged, routed only where confirmed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("part", "kind"),
    [(audio_part(), "audio"), (audio_part(WAV, "wav"), "audio"), (file_part(), "file")],
)
def test_an_attachment_is_answered_by_the_backend_that_takes_it(
    settings: Settings, part: dict, kind: str
) -> None:
    deaf = Takes(name="deaf")
    hears = Takes(kind, name="hears")
    with serve(settings, deaf, hears) as client:
        # Several times, so a balancer that ignored the attachment would
        # land on the text-only backend at least once.
        for _ in range(4):
            r = client.post("/v1/chat/completions", json=chat(part))
            assert r.status_code == 200, r.text
    assert not deaf.calls, "a text-only backend was sent the attachment"
    assert len(hears.calls) == 4
    assert sent_parts(hears)[1] == part


@pytest.mark.parametrize(
    ("part", "field"), [(audio_part(), "audio_input"), (file_part(), "file_input")]
)
def test_with_no_backend_that_takes_it_the_request_is_refused_not_sent(
    settings: Settings, part: dict, field: str
) -> None:
    deaf = Takes("image", name="deaf")
    with serve(settings, deaf) as client:
        r = client.post("/v1/chat/completions", json=chat(part))
    assert r.status_code == 400, r.text
    assert f"x_eugene_plexus.{field}" in r.json()["error"]["message"]
    assert not deaf.calls


def test_a_fallback_tier_that_cannot_hear_is_never_handed_the_audio(settings: Settings) -> None:
    primary = Takes("audio", name="primary", model_id="voice")
    primary.generate_error = DriverError(
        driver_name="primary",
        driver_url="http://fake-driver",
        status_code=500,
        problem=None,
        raw_body="engine died",
    )
    backup = Takes(name="backup", model_id="text-only")
    slots = [{"model": "voice", "targets": ["text-only"]}]
    with serve(settings, primary, backup, slots=slots) as client:
        r = client.post("/v1/chat/completions", json=chat(audio_part(), model="voice"))
    assert primary.calls, "the confirmed primary is tried"
    assert not backup.calls, "the text-only fallback must not answer a recording it never heard"
    assert r.status_code >= 500, r.text


def test_audio_and_a_pdf_together_go_to_a_backend_that_takes_both(settings: Settings) -> None:
    hears = Takes("audio", name="hears")
    reads = Takes("file", name="reads")
    both = Takes("audio", "file", name="both")
    with serve(settings, hears, reads, both) as client:
        for _ in range(3):
            r = client.post("/v1/chat/completions", json=chat(audio_part(), file_part()))
            assert r.status_code == 200, r.text
    assert not hears.calls and not reads.calls
    assert len(both.calls) == 3


def test_kinds_confirmed_only_apart_are_named_together(settings: Settings) -> None:
    with serve(settings, Takes("audio", name="hears"), Takes("file", name="reads")) as client:
        r = client.post("/v1/chat/completions", json=chat(audio_part(), file_part()))
    assert r.status_code == 400
    assert "audio input and file input together" in r.json()["error"]["message"]


def test_bare_base64_reaches_the_driver_as_the_data_url(settings: Settings) -> None:
    reads = Takes("file", name="reads")
    with serve(settings, reads) as client:
        r = client.post("/v1/chat/completions", json=chat(file_part(b64(PDF))))
    assert r.status_code == 200, r.text
    assert sent_parts(reads)[1]["file"]["file_data"] == PDF_URL


@pytest.mark.parametrize(
    ("part", "said"),
    [
        (audio_part(WAV, "mp3"), "does not match its declared mp3"),
        (
            {"type": "input_audio", "input_audio": {"data": b64(MP3), "format": "flac"}},
            "messages.0.content.1.input_audio.format",
        ),
        (file_part(file_id="file-abc"), "no file store"),
        (file_part("data:text/plain;base64," + b64(b"hi")), "PDF data URL"),
        (file_part("https://example.com/a.pdf"), "URLs are not fetched"),
        (file_part(b64(b"not a pdf")), "is not a PDF"),
    ],
)
def test_a_bad_attachment_is_refused_naming_it(settings: Settings, part: dict, said: str) -> None:
    both = Takes("audio", "file", name="both")
    with serve(settings, both) as client:
        r = client.post("/v1/chat/completions", json=chat(part))
    assert r.status_code == 400, r.text
    assert said in r.json()["error"]["message"]
    # The schema names a field as `messages.0.content.1` and the attachment
    # check as `messages[0].content[1]`; either way it is the part sent.
    message = r.json()["error"]["message"]
    assert "messages[0].content[1]" in message or "messages.0.content.1" in message
    assert not both.calls


def test_every_kind_counts_toward_one_request_total(settings: Settings, monkeypatch) -> None:
    monkeypatch.setattr(images, "MAX_ATTACHMENTS_TOTAL", len(MP3) + len(PDF) - 1)
    both = Takes("audio", "file", name="both")
    with serve(settings, both) as client:
        r = client.post("/v1/chat/completions", json=chat(audio_part(), file_part()))
    assert r.status_code == 400
    assert "11 MiB" in r.json()["error"]["message"]


def test_the_model_list_says_who_hears_and_who_reads(settings: Settings) -> None:
    with serve(settings, Takes("audio", "file", name="both")) as client:
        listed = client.get("/v1/models").json()["data"]
    info = next(m for m in listed if m["id"] == MODEL)["x_eugene_plexus"]
    assert (info["audio_input"], info["file_input"], info["image_input"]) == (True, True, False)


# --------------------------------------------------------------------------- #
# Anthropic: a document block is carried, including from a tool
# --------------------------------------------------------------------------- #


def anthropic(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": content}]}


def pdf_document(**extra: Any) -> dict[str, Any]:
    return {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": b64(PDF)},
        **extra,
    }


def test_a_pdf_document_block_is_a_file_part(settings: Settings) -> None:
    reads = Takes("file", name="reads")
    with serve(settings, reads) as client:
        r = client.post(
            "/v1/messages",
            json=anthropic(
                [pdf_document(title="note.pdf"), {"type": "text", "text": "Summarise."}]
            ),
        )
    assert r.status_code == 200, r.text
    part = sent_parts(reads)[0]
    assert part == {"type": "file", "file": {"filename": "note.pdf", "file_data": PDF_URL}}


def test_a_document_reaches_only_a_model_that_reads_files(settings: Settings) -> None:
    deaf = Takes("image", name="deaf")
    with serve(settings, deaf) as client:
        r = client.post("/v1/messages", json=anthropic([pdf_document()]))
    assert r.status_code == 400
    assert "file input" in r.json()["error"]["message"]
    assert not deaf.calls


def test_a_text_document_is_carried_as_its_text(settings: Settings) -> None:
    fake = Takes(name="plain")
    source = {"type": "text", "media_type": "text/plain", "data": "The word is zebra."}
    with serve(settings, fake) as client:
        r = client.post(
            "/v1/messages", json=anthropic([{"type": "document", "source": source, "title": "N"}])
        )
    assert r.status_code == 200, r.text
    sent = fake.calls[-1].model_dump(mode="json", exclude_none=True)["messages"][-1]["content"]
    assert sent == "N\n\nThe word is zebra."


def test_citations_enabled_is_refused_rather_than_answered_uncited(settings: Settings) -> None:
    reads = Takes("file", name="reads")
    with serve(settings, reads) as client:
        r = client.post("/v1/messages", json=anthropic([pdf_document(citations={"enabled": True})]))
    assert r.status_code == 400
    assert "citations" in r.json()["error"]["message"]
    assert not reads.calls


def test_claude_codes_read_of_a_pdf_moves_to_the_next_user_message(settings: Settings) -> None:
    """Claude Code's `Read` of a PDF returns the document inside the
    `tool_result`, as its `Read` of an image does (captured 2026-09-23)."""
    reads = Takes("file", name="reads")
    request = {
        "model": MODEL,
        "max_tokens": 64,
        "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": "Read note.pdf"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [pdf_document()]}
                ],
            },
        ],
    }
    with serve(settings, reads) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 200, r.text
    messages = reads.calls[-1].model_dump(mode="json", exclude_none=True)["messages"]
    tool, after = messages[-2], messages[-1]
    assert tool["role"] == "tool"
    assert tool["content"] == (
        "[The tool returned a document; it is attached to the next user message.]"
    )
    assert after["content"][0]["text"] == "The document returned by tool call toolu_1:"
    assert after["content"][1]["file"]["file_data"] == PDF_URL


def test_count_tokens_with_a_document_says_it_cannot_count(settings: Settings) -> None:
    reads = Takes("file", name="reads")
    reads.prompt_tokens = 12
    with serve(settings, reads) as client:
        r = client.post(
            "/v1/messages/count_tokens",
            json={"model": MODEL, "messages": [{"role": "user", "content": [pdf_document()]}]},
        )
    assert r.status_code == 400
    assert "encoder" in r.json()["error"]["message"]
    assert not reads.count_calls


# --------------------------------------------------------------------------- #
# Responses: input_file and input_audio
# --------------------------------------------------------------------------- #


def responses(content: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": MODEL,
        "stream": False,
        "input": [{"type": "message", "role": "user", "content": content}],
    }


def test_responses_input_file_and_input_audio_are_carried(settings: Settings) -> None:
    both = Takes("audio", "file", name="both")
    both.responses = ["ok"]
    with serve(settings, both) as client:
        r = client.post(
            "/v1/responses",
            json=responses(
                [
                    {"type": "input_text", "text": "What do these say?"},
                    {"type": "input_file", "filename": "note.pdf", "file_data": PDF_URL},
                    audio_part(),
                ]
            ),
        )
    assert r.status_code == 200, r.text
    parts = sent_parts(both)
    assert parts[1] == {"type": "file", "file": {"filename": "note.pdf", "file_data": PDF_URL}}
    assert parts[2] == audio_part()


def test_a_file_a_tool_returned_moves_to_the_next_user_message(settings: Settings) -> None:
    reads = Takes("file", name="reads")
    reads.responses = ["ok"]
    request = {
        "model": MODEL,
        "stream": False,
        "input": [
            {"type": "message", "role": "user", "content": "read it"},
            {"type": "function_call", "call_id": "call_1", "name": "read", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": [{"type": "input_file", "file_data": PDF_URL}],
            },
        ],
    }
    with serve(settings, reads) as client:
        r = client.post("/v1/responses", json=request)
    assert r.status_code == 200, r.text
    messages = reads.calls[-1].model_dump(mode="json", exclude_none=True)["messages"]
    assert "a document; it is attached" in messages[-2]["content"]
    assert messages[-1]["content"][1]["file"]["file_data"] == PDF_URL
