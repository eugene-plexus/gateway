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
        request(count=images.DEFAULT_MAX_IMAGES + 1),
        request(role="assistant"),
        request(picture(size=(8193, 1))),
        request(picture().replace("image/png", "image/jpeg")),
    ]:
        with pytest.raises(Refusal):
            parse_request(body)
    parse_request(request(picture("JPEG")))


def test_five_images_pass_where_four_was_the_limit():
    """The reproduction. Fixed at four until 2026-09-23, which refused a
    Claude Code session on every turn after its fifth screenshot."""
    assert parse_request(request(count=5)).messages[0].content


def test_the_default_limit_is_twelve_and_the_refusal_names_the_setting():
    parse_request(request(count=12))
    with pytest.raises(Refusal, match=r"at most 12 images.*maxImagesPerRequest"):
        parse_request(request(count=13))


def test_the_limit_is_the_operators_setting(tmp_path):
    """Read from the live config on every request, as the door sees it."""
    from eugene_plexus_gateway.settings import Settings

    app = create_app(
        settings=Settings(
            config_file=tmp_path / "config.yaml", metrics_file=tmp_path / "metrics.sqlite3"
        )
    )
    driver = VisionDriver(name="vision", model_id="fixture")
    driver.responses = ["ok"]
    app.state.routing = make_routing_table(driver)
    with TestClient(app) as client:
        assert client.patch("/v1/config", json={"maxImagesPerRequest": 2}).json()["applied"]
        refused = client.post("/v1/chat/completions", json=request(count=3))
        served = client.post("/v1/chat/completions", json=request(count=2))
    assert refused.status_code == 400
    assert "at most 2 images" in refused.json()["error"]["message"]
    assert served.status_code == 200, served.text
    assert len(driver.calls) == 1


def test_the_setting_is_clamped_to_the_drivers_ceiling():
    class Store:
        def __init__(self, value):
            self.value = value

        def get(self, key):
            return self.value

    assert images.max_images(Store(1000)) == images.MAX_IMAGES_CEILING
    assert images.max_images(Store(0)) == 1
    assert images.max_images(Store(None)) == images.DEFAULT_MAX_IMAGES
    assert images.max_images(None) == images.DEFAULT_MAX_IMAGES


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
