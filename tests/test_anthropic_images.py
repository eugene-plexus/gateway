"""Images on the Anthropic door, 2026-09-23.

**The shapes here are captured, not read.** A real Claude Code 2.1.207
(agent-sdk 0.3.280) was answered with a `Read` of an image by
`specs/scripts/r4-capture.py --mode imageread`, and its next request
carried the picture as a `tool_result` whose content is a lone `image`
block -- no text beside it, `cache_control` on the result:

    {"type": "tool_result", "tool_use_id": "toolu_...",
     "content": [{"type": "image", "source": {"type": "base64",
                  "data": "iVBOR...", "media_type": "image/png"}}],
     "cache_control": {"type": "ephemeral"}}

That is the commonest way an image reaches this door, and it is the one
an OpenAI backend has no slot for: a `tool` message carries text, and
image parts are accepted on user messages only. Two more captured facts
decide the rest. A 36 MB 4000x3000 PNG arrived as a 490 KB **JPEG**, so
the client resizes before it sends and our 5 MiB/8192 px limits hold;
and a small `.webp` and `.gif` arrived **as themselves**, `image/webp`
and `image/gif`, so a door taking PNG/JPEG only would refuse `Read` of
either.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from eugene_plexus_gateway._generated.driver_models import Capabilities
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

MODEL = "gemma-4-e4b"


class VisionDriver(FakeDriverClient):
    def describe(self):
        info = super().describe()
        # Added to what the fake already advertises: replacing it would drop
        # `supportedSettings`, and admission would refuse `max_tokens`.
        base = info.capabilities or Capabilities()
        info.capabilities = base.model_copy(update={"imageInput": True})
        return info


def encoded(fmt: str = "PNG", colour: tuple[int, int, int] = (220, 20, 20), **save: Any) -> str:
    data = io.BytesIO()
    Image.new("RGB", (16, 16), colour).save(data, format=fmt, **save)
    return base64.b64encode(data.getvalue()).decode()


def image_block(data: str | None = None, media_type: str = "image/png") -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "data": data or encoded(), "media_type": media_type},
    }


def claude_code_read(result_content: list[dict[str, Any]], *, own: list | None = None) -> dict:
    """The captured follow-up to a `Read` of an image, trimmed but not tidied."""
    return {
        "model": MODEL,
        "max_tokens": 32000,
        "stream": False,
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {"effort": "high"},
        "system": [{"type": "text", "text": "You are a Claude agent."}],
        "tools": [
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {"file_path": {}}},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Look at red.png and tell me its colour"}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Looking."},
                    {
                        "type": "tool_use",
                        "id": "toolu_r4_capture_0001",
                        "name": "Read",
                        "input": {"file_path": "C:/work/red.png"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "tool_use_id": "toolu_r4_capture_0001",
                        "type": "tool_result",
                        "content": result_content,
                        "cache_control": {"type": "ephemeral"},
                    },
                    *(own or []),
                ],
            },
        ],
    }


def serve(settings: Settings, *drivers: FakeDriverClient) -> TestClient:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*drivers)
    return TestClient(app)


def sent(driver: FakeDriverClient) -> list[dict[str, Any]]:
    return driver.calls[0].model_dump(mode="json", exclude_none=True)["messages"]


def pixels(url: str) -> tuple[str, tuple[int, int, int]]:
    header, _, data = url.partition(",")
    with Image.open(io.BytesIO(base64.b64decode(data))) as picture:
        return header, picture.convert("RGB").getpixel((0, 0))


# --------------------------------------------------------------------------- #
# The reproduction: Claude Code's own Read of an image
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stream", [False, True])
def test_claude_codes_read_of_an_image_reaches_a_vision_backend(
    settings: Settings, stream: bool
) -> None:
    """Refused with a 400 until 2026-09-23, on every Claude Code session
    that read a screenshot -- the one image path that client has."""
    data = encoded()
    driver = VisionDriver(name="v", model_id=MODEL, supports_tools=True)
    request = claude_code_read([image_block(data)])
    request["stream"] = stream
    with serve(settings, driver) as client:
        r = client.post("/v1/messages?beta=true", json=request)
    assert r.status_code == 200, r.text
    messages = sent(driver)
    tool, after = messages[-2], messages[-1]

    # The tool message still answers the call it belongs to, and says
    # where its picture went: a tool message carries text only.
    assert tool["role"] == "tool"
    assert tool["toolCallId"] == "toolu_r4_capture_0001"
    assert "image" in tool["content"] and "next user message" in tool["content"]

    # The picture follows at once, on a user message, byte for byte.
    assert after["role"] == "user"
    images = [p for p in after["content"] if p["type"] == "image_url"]
    assert [p["image_url"]["url"] for p in images] == [f"data:image/png;base64,{data}"]
    label = after["content"][0]
    assert label["type"] == "text" and "toolu_r4_capture_0001" in label["text"]


def test_a_text_only_backend_is_refused_rather_than_answering_blind(settings: Settings) -> None:
    """The half that proves the image is carried rather than dropped: a
    model that cannot see it is never asked."""
    text_only = FakeDriverClient(name="t", model_id=MODEL, supports_tools=True)
    with serve(settings, text_only) as client:
        r = client.post("/v1/messages", json=claude_code_read([image_block()]))
    assert r.status_code == 400, r.text
    assert r.json()["type"] == "error"
    assert "image input" in r.json()["error"]["message"]
    assert not text_only.calls


def test_a_tool_result_with_text_and_an_image_keeps_both(settings: Settings) -> None:
    driver = VisionDriver(name="v", model_id=MODEL, supports_tools=True)
    request = claude_code_read(
        [{"type": "text", "text": "Screenshot of the login page"}, image_block()],
        own=[{"type": "text", "text": "Which button is blue?"}],
    )
    with serve(settings, driver) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 200, r.text
    tool, after = sent(driver)[-2:]
    assert tool["content"].startswith("Screenshot of the login page")
    kinds = [p["type"] for p in after["content"]]
    # The tool's picture first -- it answers the call above it -- then the
    # person's own words, in the order they wrote them.
    assert kinds == ["text", "image_url", "text"]
    assert after["content"][-1]["text"] == "Which button is blue?"


def test_an_errored_tool_result_with_an_image_still_says_it_failed(settings: Settings) -> None:
    driver = VisionDriver(name="v", model_id=MODEL, supports_tools=True)
    request = claude_code_read([image_block()])
    request["messages"][-1]["content"][0]["is_error"] = True
    with serve(settings, driver) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 200, r.text
    assert sent(driver)[-2]["content"].startswith("[tool error]")


def test_two_tool_results_with_images_keep_their_order(settings: Settings) -> None:
    red, blue = encoded(colour=(220, 20, 20)), encoded(colour=(20, 20, 220))
    driver = VisionDriver(name="v", model_id=MODEL, supports_tools=True)
    request = claude_code_read([image_block(red)])
    request["messages"][1]["content"].append(
        {"type": "tool_use", "id": "toolu_2", "name": "Read", "input": {"file_path": "b.png"}}
    )
    request["messages"][-1]["content"].append(
        {"type": "tool_result", "tool_use_id": "toolu_2", "content": [image_block(blue)]}
    )
    with serve(settings, driver) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 200, r.text
    messages = sent(driver)
    # Both tool messages directly after the assistant turn that asked for
    # them -- an OpenAI backend expects nothing between -- then one user
    # message carrying both pictures, each labelled with its call.
    assert [m["role"] for m in messages[-3:]] == ["tool", "tool", "user"]
    parts = messages[-1]["content"]
    urls = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
    assert urls == [f"data:image/png;base64,{red}", f"data:image/png;base64,{blue}"]
    labels = [p["text"] for p in parts if p["type"] == "text"]
    assert "toolu_r4_capture_0001" in labels[0] and "toolu_2" in labels[1]


# --------------------------------------------------------------------------- #
# A person's own image, in Anthropic's documented shape
# --------------------------------------------------------------------------- #


def test_a_users_image_keeps_its_place_among_the_text(settings: Settings) -> None:
    data = encoded("JPEG")
    driver = VisionDriver(name="v", model_id=MODEL)
    request = {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is a picture."},
                    {"type": "text", "text": "It is small."},
                    image_block(data, "image/jpeg"),
                    {"type": "text", "text": "What colour is it?"},
                ],
            }
        ],
    }
    with serve(settings, driver) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 200, r.text
    content = sent(driver)[0]["content"]
    assert content == [
        # Adjacent text blocks join as they always have on this door.
        {"type": "text", "text": "Here is a picture.\n\nIt is small."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}},
        {"type": "text", "text": "What colour is it?"},
    ]


def test_a_text_only_turn_is_still_a_plain_string(settings: Settings) -> None:
    """No image, no change: the shape every text request had before."""
    driver = VisionDriver(name="v", model_id=MODEL)
    request = {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            }
        ],
    }
    with serve(settings, driver) as client:
        assert client.post("/v1/messages", json=request).status_code == 200
    assert sent(driver)[0]["content"] == "a\n\nb"


@pytest.mark.parametrize(("fmt", "media_type"), [("WEBP", "image/webp"), ("GIF", "image/gif")])
def test_webp_and_gif_reach_the_backend_as_the_same_picture_in_png(
    settings: Settings, fmt: str, media_type: str
) -> None:
    """Claude Code sends a small `.webp` or `.gif` as itself (captured).
    Local engines and our internal contract take PNG and JPEG, so the door
    re-encodes it -- losslessly, and the pixels say so."""
    driver = VisionDriver(name="v", model_id=MODEL, supports_tools=True)
    source = encoded(fmt, colour=(20, 20, 220), **({"lossless": True} if fmt == "WEBP" else {}))
    with serve(settings, driver) as client:
        r = client.post("/v1/messages", json=claude_code_read([image_block(source, media_type)]))
    assert r.status_code == 200, r.text
    url = next(p for p in sent(driver)[-1]["content"] if p["type"] == "image_url")
    header, colour = pixels(url["image_url"]["url"])
    assert header == "data:image/png;base64"
    assert colour == (20, 20, 220)


# --------------------------------------------------------------------------- #
# What is still refused, in the caller's own coordinates
# --------------------------------------------------------------------------- #


def refused(settings: Settings, request: dict[str, Any]) -> str:
    driver = VisionDriver(name="v", model_id=MODEL, supports_tools=True)
    with serve(settings, driver) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 400, r.text
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert not driver.calls, "a refused request must not reach a backend"
    return r.json()["error"]["message"]


def test_an_image_url_is_never_fetched(settings: Settings) -> None:
    block = {"type": "image", "source": {"type": "url", "url": "http://169.254.169.254/x.png"}}
    message = refused(settings, claude_code_read([block]))
    assert "messages.2.content.0.content.0" in message
    assert "not fetched" in message
    assert "169.254" not in message, "a caller's URL is never reflected"


def test_a_files_api_reference_is_refused(settings: Settings) -> None:
    block = {"type": "image", "source": {"type": "file", "file_id": "file_011"}}
    assert "base64" in refused(settings, claude_code_read([block]))


def test_an_unsupported_media_type_is_refused(settings: Settings) -> None:
    message = refused(settings, claude_code_read([image_block(encoded("BMP"), "image/bmp")]))
    assert "PNG" in message and "WebP" in message


def test_an_animated_gif_is_refused(settings: Settings) -> None:
    frames = [Image.new("RGB", (8, 8), c) for c in ("red", "blue")]
    data = io.BytesIO()
    frames[0].save(data, format="GIF", save_all=True, append_images=frames[1:])
    block = image_block(base64.b64encode(data.getvalue()).decode(), "image/gif")
    assert "animated" in refused(settings, claude_code_read([block]))


def test_a_picture_that_is_not_what_it_claims_is_refused(settings: Settings) -> None:
    message = refused(settings, claude_code_read([image_block(encoded("JPEG"), "image/png")]))
    assert "messages.2.content.0.content.0" in message
    assert "invalid" in message


def test_five_screenshots_are_no_longer_refused(settings: Settings) -> None:
    """The reproduction. At a fixed four, a session that had read five
    screenshots was refused on every turn after the fifth."""
    request = claude_code_read(
        [image_block(), image_block(), image_block()],
        own=[image_block(), image_block()],
    )
    driver = VisionDriver(name="vision", model_id=MODEL, supports_tools=True)
    driver.responses = ["ok"]
    with serve(settings, driver) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 200, r.text


def test_the_per_request_image_limit_counts_both_places(settings: Settings) -> None:
    request = claude_code_read(
        [image_block(), image_block(), image_block()],
        own=[image_block(), image_block()],
    )
    driver = VisionDriver(name="vision", model_id=MODEL, supports_tools=True)
    with serve(settings, driver) as client:
        client.patch("/v1/config", json={"maxImagesPerRequest": 4})
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 400
    message = r.json()["error"]["message"]
    assert "at most 4 images" in message and "maxImagesPerRequest" in message
    assert "messages.2.content.2" in message
    assert not driver.calls


def test_an_image_on_an_assistant_turn_is_refused(settings: Settings) -> None:
    request = claude_code_read([{"type": "text", "text": "ok"}])
    request["messages"][1]["content"].insert(0, image_block())
    assert "messages.1.content.0" in refused(settings, request)


def test_a_document_block_is_still_refused(settings: Settings) -> None:
    block = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf"}}
    assert "document" in refused(settings, claude_code_read([block]))
