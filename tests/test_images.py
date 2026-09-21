"""Images survive routing/fallback, or fail before any driver is called."""

import base64
import io

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from eugene_plexus_gateway import body_limit, images
from eugene_plexus_gateway._generated.driver_models import Capabilities
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.chat_contract import Refusal, parse_request

from .conftest import FakeDriverClient, make_routing_table


def picture(fmt="PNG", size=(8, 8)):
    data = io.BytesIO()
    Image.new("RGB", size, "purple").save(data, format=fmt)
    mime = "jpeg" if fmt == "JPEG" else "png"
    return f"data:image/{mime};base64," + base64.b64encode(data.getvalue()).decode()


def request(url=None, *, count=1, role="user", stream=False):
    return {
        "model": "fixture",
        "stream": stream,
        "messages": [
            {
                "role": role,
                "content": [
                    {"type": "text", "text": "Describe it."},
                    *[
                        {"type": "image_url", "image_url": {"url": url or picture()}}
                        for _ in range(count)
                    ],
                    {"type": "text", "text": "Be concise."},
                ],
            }
        ],
    }


class VisionDriver(FakeDriverClient):
    def describe(self):
        info = super().describe()
        info.capabilities = Capabilities(imageInput=True)
        return info


@pytest.mark.parametrize("stream", [False, True])
def test_fallback_keeps_order_and_exact_image_and_skips_text_targets(settings, stream):
    first = VisionDriver(name="a", model_id="fixture")
    first.generate_error = httpx.ConnectError("disconnected")
    text = FakeDriverClient(name="b", model_id="fixture")
    fallback = VisionDriver(name="c", model_id="backup")
    app = create_app(settings)
    app.state.routing = make_routing_table(
        first, text, fallback, slots=[{"model": "fixture", "targets": ["backup"]}]
    )
    body = request(stream=stream)
    with TestClient(app) as client:
        model = client.get("/v1/models").json()["data"][0]
        assert model["x_eugene_plexus"]["image_input"] is True
        response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    assert not text.calls
    assert len(first.calls) == len(fallback.calls) == 1
    for driver in [first, fallback]:
        assert (
            driver.calls[0].model_dump(mode="json", exclude_none=True)["messages"][0]["content"]
            == body["messages"][0]["content"]
        )


@pytest.mark.parametrize("stream", [False, True])
def test_text_target_refuses_images(client, fake_driver, stream):
    body = request(stream=stream)
    body["model"] = fake_driver.model_id
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400, response.text
    assert "vision model" in response.text
    assert not fake_driver.calls


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/PRIVATE",
        "https://example.org/PRIVATE",
        "file:///PRIVATE",
        "data:image/svg+xml;base64,UFJJVkFURQ==",
        "data:image/png;base64,PRIVATE!",
        "data:image/jpeg;base64," + base64.b64encode(b"PRIVATE").decode(),
    ],
)
def test_invalid_image_is_not_fetched_or_echoed(client, fake_driver, url):
    body = request(url)
    body["model"] = fake_driver.model_id
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400
    assert "PRIVATE" not in response.text
    assert not fake_driver.calls


def test_limits_mime_and_role():
    for body in [
        request(count=5),
        request(role="assistant"),
        request(picture(size=(8193, 1))),
        request(picture().replace("image/png", "image/jpeg")),
    ]:
        with pytest.raises(Refusal):
            parse_request(body)
    parse_request(request(picture("JPEG")))


def test_aggregate_and_decoded_size_limits(monkeypatch):
    monkeypatch.setattr(images, "MAX_TOTAL_BYTES", 100)
    with pytest.raises(Refusal, match="10 MiB"):
        parse_request(request(count=2))
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 2)
    with pytest.raises(Refusal, match="5 MiB"):
        parse_request(request())


def test_text_arrays_normalize_without_erasing_images():
    body = request()
    body["messages"][0]["content"] = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    assert parse_request(body).messages[0].content == "firstsecond"
    body = request()
    body["messages"][0]["content"][1]["image_url"]["detail"] = "high"
    with pytest.raises(Refusal):
        parse_request(body)


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_chunked_body_bound(client, fake_driver, monkeypatch, path):
    monkeypatch.setattr(body_limit, "MAX_BODY_BYTES", 128)
    response = client.post(path, content=iter([b" " * 80, b" " * 80]))
    assert response.status_code == 413
    assert "16 MiB" in response.text
    assert not fake_driver.calls


def test_oversized_content_length_is_refused_without_integer_conversion(client):
    response = client.post(
        "/v1/chat/completions", content=b"{}", headers={"content-length": "9" * 5000}
    )
    assert response.status_code == 413
