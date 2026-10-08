"""POST /v1/images/generations and /v1/images/edits; variations refused (P4).

The request shapes are the OpenAI SDK's, captured 2026-09-28
(`provider-accounts-measurement.md` section 9): generation is JSON; an edit
is always multipart, `image` for one image, `image[]` for several, `mask`,
and a `BytesIO` arrives labelled `application/octet-stream`. The capabilities
are OpenRouter's `/images/models` as the driver reports them. Every test here
fails against the gateway before P4, which had no images door.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway import image_doors
from eugene_plexus_gateway._generated.driver_models import (
    GeneratedImage,
    ImageCapabilities,
    ImagePartial,
    ImageRequest,
    ImageResponse,
    ImageUsage,
    Problem,
    RetryDisposition,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import DriverError
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF" + bytes(64)
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + bytes(32)

#: flux.2-klein-4b as measured: one image, png/jpeg, up to four references,
#: no quality or background, no stream, no mask.
FLUX = ImageCapabilities(
    streaming=False,
    maxImages=1,
    minReferences=0,
    maxReferences=4,
    mask=False,
    qualities=[],
    backgrounds=[],
    outputFormats=["png", "jpeg"],
)
#: gpt-image-1-mini on OpenRouter: streams, up to ten, output_format unlisted.
MINI = ImageCapabilities(
    streaming=True,
    maxImages=10,
    minReferences=0,
    maxReferences=16,
    mask=False,
    qualities=["auto", "low", "medium", "high"],
    backgrounds=["auto", "transparent", "opaque"],
)
#: An image model on OpenAI's own API: its API checks, and it takes a mask.
OPENAI = ImageCapabilities(streaming=True, mask=True)


class Painter(FakeDriverClient):
    """A fake whose model makes images, as a driver does since P4."""

    def __init__(
        self,
        *,
        caps: ImageCapabilities | None = FLUX,
        answer: bytes = PNG,
        fail: Exception | None = None,
        fail_after_first: bool = False,
        images: int = 1,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.caps = caps
        self.answer = answer
        self.fail = fail
        self.fail_after_first = fail_after_first
        self.images = images
        self.asked: list[ImageRequest] = []

    def describe(self):  # type: ignore[no-untyped-def]
        info = super().describe()
        for model in info.models or []:
            model.surfaces = ["image"]
            assert model.capabilities is not None
            model.capabilities.image = self.caps
        return info

    def _result(self) -> ImageResponse:
        image = GeneratedImage(
            data=base64.b64encode(self.answer).decode(),
            mediaType=image_doors.sniff(self.answer) or "",
        )
        return ImageResponse(
            images=[image] * self.images,
            usage=ImageUsage(inputTokens=6, outputTokens=4096, totalTokens=4102, cost=0.014),
            modelId=self.model_id,
        )

    async def image(self, request: ImageRequest) -> ImageResponse:
        self.asked.append(request)
        if self.fail is not None:
            raise self.fail
        return self._result()

    async def image_stream(self, request: ImageRequest):  # type: ignore[no-untyped-def]
        self.asked.append(request)
        if self.fail is not None and not self.fail_after_first:
            raise self.fail
        partial = GeneratedImage(data=base64.b64encode(PNG).decode(), mediaType="image/png")
        yield ImagePartial(image=partial, index=0)
        if self.fail is not None:
            raise self.fail
        yield self._result()


class Chatter(FakeDriverClient):
    """A chat model, for the wrong-door refusals."""


def serve(settings: Settings, *drivers: FakeDriverClient, **table: Any) -> TestClient:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*drivers, **table)
    return TestClient(app)


def generate(client: TestClient, model: str = "flux", **fields: Any) -> httpx.Response:
    return client.post(
        "/v1/images/generations", json={"model": model, "prompt": "a blue square", **fields}
    )


def edit(client: TestClient, model: str = "flux", *, files: Any, **fields: Any) -> httpx.Response:
    return client.post(
        "/v1/images/edits", data={"model": model, "prompt": "make it red", **fields}, files=files
    )


def _refused(response: httpx.Response, param: str) -> str:
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == param, error
    return str(error["message"])


def _events(text: str) -> list[dict[str, Any]]:
    out = []
    for block in text.strip().split("\n\n"):
        data = next(line[6:] for line in block.split("\n") if line.startswith("data: "))
        out.append(json.loads(data))
    return out


# --------------------------------------------------------------------------- #
# Listed and answered
# --------------------------------------------------------------------------- #


def test_an_image_model_is_listed_with_what_it_takes(settings: Settings) -> None:
    drivers: tuple[FakeDriverClient, ...] = (
        Painter(name="a", model_id="flux", caps=FLUX),
        Painter(name="b", model_id="mini", caps=MINI),
        Painter(name="c", model_id="gpt-image-1", caps=OPENAI),
        Chatter(name="d", model_id="qwen"),
    )
    with serve(settings, *drivers) as client:
        models = {m["id"]: m["x_eugene_plexus"] for m in client.get("/v1/models").json()["data"]}
    assert models["flux"]["surfaces"] == ["image"]
    assert models["flux"]["image_edits"] is True
    assert models["flux"]["image_streaming"] is False and models["flux"]["image_mask"] is False
    assert models["mini"]["image_streaming"] is True
    assert models["gpt-image-1"]["image_mask"] is True
    # A chat model is not told it cannot stream images.
    assert models["qwen"].get("image_streaming") is None


def test_the_listing_names_the_settings_the_door_enforces(settings: Settings) -> None:
    """A form can offer exactly what the door would take (media screens,
    2026-10-08): the driver's ImageCapabilities, listed as the gateway
    routes on them. Null keeps meaning *the backend checks*."""
    drivers: tuple[FakeDriverClient, ...] = (
        Painter(name="a", model_id="flux", caps=FLUX),
        Painter(name="b", model_id="mini", caps=MINI),
        Painter(name="c", model_id="gpt-image-1", caps=OPENAI),
        Chatter(name="d", model_id="qwen"),
    )
    with serve(settings, *drivers) as client:
        models = {m["id"]: m["x_eugene_plexus"] for m in client.get("/v1/models").json()["data"]}
    flux, mini, oai = models["flux"], models["mini"], models["gpt-image-1"]
    assert (flux["image_max_images"], flux["image_qualities"], flux["image_backgrounds"]) == (
        1,
        [],
        [],
    )
    assert flux["image_output_formats"] == ["png", "jpeg"]
    assert (flux["image_min_references"], flux["image_max_references"]) == (0, 4)
    assert (mini["image_max_images"], mini["image_qualities"]) == (
        10,
        ["auto", "low", "medium", "high"],
    )
    assert mini["image_output_formats"] is None  # unlisted: carried, the backend decides
    for field in (
        "image_max_images",
        "image_qualities",
        "image_output_formats",
        "image_max_references",
    ):
        assert oai[field] is None, field
    # A chat model is told nothing about images, not "takes none".
    image_fields = [k for k in flux if k.startswith("image_") and k != "image_input"]
    assert len(image_fields) == 9, image_fields
    assert [k for k in image_fields if models["qwen"].get(k) is not None] == []


def test_a_slots_settings_are_what_some_backend_takes(settings: Settings) -> None:
    """A request routes to any backend that takes it, so a slot lists the
    largest limit and the union of choices -- and null once one backend
    leaves a field to its own API."""
    slots = [{"model": "pictures", "targets": ["flux", "mini"]}]
    flux, mini = (
        Painter(name="a", model_id="flux", caps=FLUX),
        Painter(name="b", model_id="mini", caps=MINI),
    )
    with serve(settings, flux, mini, slots=slots) as client:
        models = {m["id"]: m["x_eugene_plexus"] for m in client.get("/v1/models").json()["data"]}
        listed = models["pictures"]
        assert (listed["image_max_images"], listed["image_max_references"]) == (10, 16)
        assert listed["image_qualities"] == ["auto", "low", "medium", "high"]
        assert listed["image_output_formats"] is None
        # What the listing says is what the door does: 10 is taken, by mini.
        assert generate(client, "pictures", n=10).status_code == 200
    only_flux = [{"model": "pictures", "targets": ["flux"]}]
    flux = Painter(name="a", model_id="flux", caps=FLUX)
    with serve(settings, flux, slots=only_flux) as client:
        listed = {m["id"]: m["x_eugene_plexus"] for m in client.get("/v1/models").json()["data"]}[
            "pictures"
        ]
        assert (listed["image_max_images"], listed["image_qualities"]) == (1, [])
        assert "at most 1 " in _refused(generate(client, "pictures", n=2), "n")
    # One backend that checks its own fields makes the slot's limit unknown:
    # a larger request may route there, and its own refusal is relayed.
    mixed = [{"model": "pictures", "targets": ["flux", "gpt-image-1"]}]
    flux, oai = (
        Painter(name="a", model_id="flux", caps=FLUX),
        Painter(name="c", model_id="gpt-image-1", caps=OPENAI),
    )
    with serve(settings, flux, oai, slots=mixed) as client:
        listed = {m["id"]: m["x_eugene_plexus"] for m in client.get("/v1/models").json()["data"]}[
            "pictures"
        ]
        assert (listed["image_max_images"], listed["image_qualities"]) == (None, None)
        assert generate(client, "pictures", n=2).status_code == 200


def test_the_sdks_generation_is_answered_in_openais_shape(settings: Settings) -> None:
    painter = Painter(name="a", model_id="flux", answer=JPEG)
    before = int(time.time())
    with serve(settings, painter) as client:
        response = generate(
            client, size="1536x1024", output_format="png", response_format="url", user="u1"
        )
    assert response.status_code == 200, response.text
    body = response.json()
    # P4-1: url asked, b64_json answered.
    assert base64.b64decode(body["data"][0]["b64_json"]) == JPEG and "url" not in body["data"][0]
    # Labelled by the bytes, not by what was asked (P2-2).
    assert body["output_format"] == "jpeg"
    # The driver gave no `created` (flux says 0): this clock's.
    assert body["created"] >= before
    assert body["usage"] == {"input_tokens": 6, "output_tokens": 4096, "total_tokens": 4102}
    assert body["x_eugene_plexus"]["driver"] == "a"
    [sent] = painter.asked
    assert (sent.prompt, sent.size, sent.outputFormat, sent.user) == (
        "a blue square",
        "1536x1024",
        "png",
        "u1",
    )
    assert sent.references is None and sent.mask is None


def test_openrouters_own_fields_are_refused_by_name(settings: Settings) -> None:
    painter = Painter(name="a", model_id="flux")
    with serve(settings, painter) as client:
        said = _refused(generate(client, aspect_ratio="16:9"), "aspect_ratio")
        _refused(generate(client, colour="blue"), "colour")
    assert "size" in said
    assert not painter.asked


def test_variations_are_refused_saying_why(settings: Settings) -> None:
    with serve(settings, Painter(name="a", model_id="flux")) as client:
        response = client.post(
            "/v1/images/variations", files={"image": ("a.png", PNG, "image/png")}
        )
    assert response.status_code == 400, response.text
    assert "variations" in response.json()["error"]["message"]


def test_the_wrong_door_is_named_both_ways(settings: Settings) -> None:
    with serve(
        settings, Painter(name="a", model_id="flux"), Chatter(name="c", model_id="qwen")
    ) as client:
        at_images = generate(client, "qwen")
        at_chat = client.post(
            "/v1/chat/completions",
            json={"model": "flux", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert "/v1/chat/completions" in _refused(at_images, "model")
    assert "/v1/images/generations" in at_chat.json()["error"]["message"]


# --------------------------------------------------------------------------- #
# Settings route by the model's own listing
# --------------------------------------------------------------------------- #


def test_a_setting_a_model_does_not_take_routes_to_one_that_does(settings: Settings) -> None:
    flux = Painter(name="a", model_id="flux", caps=FLUX)
    mini = Painter(name="b", model_id="mini", caps=MINI)
    slots = [{"model": "pictures", "targets": ["flux", "mini"]}]
    with serve(settings, flux, mini, slots=slots) as client:
        response = generate(client, "pictures", quality="high", n=2)
    assert response.status_code == 200, response.text
    assert not flux.asked and mini.asked
    assert response.json()["x_eugene_plexus"]["tier"] == 2


def test_auto_asks_for_nothing_and_routes_nowhere_else(settings: Settings) -> None:
    flux = Painter(name="a", model_id="flux", caps=FLUX)
    with serve(settings, flux) as client:
        assert generate(client, quality="auto", background="auto").status_code == 200
    assert flux.asked


def test_a_setting_no_model_takes_is_refused_naming_it(settings: Settings) -> None:
    """flux answered an opaque JPEG with a 200 for `background: transparent`
    (measured): the silent drop A2 forbids."""
    flux = Painter(name="a", model_id="flux", caps=FLUX)
    with serve(settings, flux) as client:
        background = _refused(generate(client, background="transparent"), "background")
        fmt = _refused(generate(client, output_format="webp"), "output_format")
        _refused(generate(client, n=2), "n")
    assert "Nothing was sent" in background and "flux" in background
    assert "png, jpeg" in fmt
    assert not flux.asked


def test_an_unlisted_output_format_is_carried(settings: Settings) -> None:
    """gpt-image-1-mini honours an unlisted output_format (measured)."""
    mini = Painter(name="b", model_id="mini", caps=MINI, answer=WEBP)
    with serve(settings, mini) as client:
        response = generate(client, "mini", output_format="webp")
    assert response.status_code == 200, response.text
    assert mini.asked[0].outputFormat == "webp"
    assert response.json()["output_format"] == "webp"


def test_an_edit_only_model_makes_no_generation_and_a_generator_no_edit(settings: Settings) -> None:
    styles = Painter(
        name="a",
        model_id="styles",
        caps=ImageCapabilities(minReferences=1, maxReferences=1, mask=False),
    )
    plain = Painter(name="b", model_id="plain", caps=ImageCapabilities(maxReferences=0, mask=False))
    with serve(settings, styles, plain) as client:
        makes_nothing = _refused(generate(client, "styles"), "image")
        edits_nothing = _refused(
            edit(client, "plain", files={"image": ("a.png", PNG, "image/png")}), "image"
        )
    assert "only edits" in makes_nothing
    assert "does not edit" in edits_nothing
    assert not styles.asked and not plain.asked


# --------------------------------------------------------------------------- #
# Edits
# --------------------------------------------------------------------------- #


def test_the_sdks_multipart_edit_reaches_the_driver_typed_by_its_bytes(settings: Settings) -> None:
    painter = Painter(name="a", model_id="gpt-image-1", caps=OPENAI)
    files = [
        ("image[]", ("a.png", PNG, "image/png")),
        # A BytesIO arrives as `upload`, application/octet-stream (measured).
        ("image[]", ("upload", JPEG, "application/octet-stream")),
        ("mask", ("m.png", PNG, "image/png")),
    ]
    with serve(settings, painter) as client:
        response = edit(client, "gpt-image-1", files=files, input_fidelity="high", n="2")
    assert response.status_code == 200, response.text
    [sent] = painter.asked
    assert [(r.mediaType, base64.b64decode(r.data)) for r in sent.references or []] == [
        ("image/png", PNG),
        ("image/jpeg", JPEG),
    ]
    assert sent.mask is not None and sent.mask.mediaType == "image/png"
    assert (sent.inputFidelity, sent.n) == ("high", 2)


def test_one_image_is_the_part_named_image(settings: Settings) -> None:
    painter = Painter(name="a", model_id="flux")
    with serve(settings, painter) as client:
        assert edit(client, files={"image": ("a.png", PNG, "image/png")}).status_code == 200
    assert len(painter.asked[0].references or []) == 1


def test_a_mask_routes_only_where_it_is_honoured(settings: Settings) -> None:
    flux = Painter(name="a", model_id="flux", caps=FLUX)
    with serve(settings, flux) as client:
        said = _refused(
            edit(
                client,
                files=[
                    ("image", ("a.png", PNG, "image/png")),
                    ("mask", ("m.png", PNG, "image/png")),
                ],
            ),
            "mask",
        )
    assert "OpenAI" in said
    assert not flux.asked


def test_the_json_edit_takes_a_data_url_and_refuses_everything_else(settings: Settings) -> None:
    painter = Painter(name="a", model_id="flux")
    data = "data:image/png;base64," + base64.b64encode(PNG).decode()

    def ask(image: dict[str, str]) -> dict[str, Any]:
        return {"model": "flux", "prompt": "red", "images": [image]}

    with serve(settings, painter) as client:
        ok = client.post("/v1/images/edits", json=ask({"image_url": data}))
        by_url = client.post(
            "/v1/images/edits", json=ask({"image_url": "https://example.com/a.png"})
        )
        by_file = client.post("/v1/images/edits", json=ask({"file_id": "file-abc"}))
    assert ok.status_code == 200, ok.text
    assert "not fetched" in _refused(by_url, "images[0].image_url")
    assert "keeps none" in _refused(by_file, "images[0].file_id")
    assert len(painter.asked) == 1


def test_an_upload_that_is_not_an_image_is_refused(settings: Settings) -> None:
    painter = Painter(name="a", model_id="flux")
    with serve(settings, painter) as client:
        said = _refused(
            edit(client, files={"image": ("a.png", b"%PDF-1.7", "image/png")}), "image[0]"
        )
    assert "PNG, JPEG, WebP or GIF" in said
    assert not painter.asked


def test_images_over_25_mib_in_all_are_413(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(image_doors, "MAX_UPLOAD_BYTES", len(PNG) * 2 - 1)
    painter = Painter(name="a", model_id="flux")
    files = [("image[]", ("a.png", PNG, "image/png")), ("image[]", ("b.png", PNG, "image/png"))]
    with serve(settings, painter) as client:
        response = edit(client, files=files)
    assert response.status_code == 413, response.text
    assert not painter.asked


def test_seventeen_images_are_one_too_many(settings: Settings) -> None:
    painter = Painter(name="a", model_id="gpt-image-1", caps=OPENAI)
    files = [("image[]", (f"{i}.png", PNG, "image/png")) for i in range(17)]
    with serve(settings, painter) as client:
        response = edit(client, "gpt-image-1", files=files)
    assert response.status_code == 400, response.text
    assert not painter.asked


# --------------------------------------------------------------------------- #
# Failover: tiers, as chat
# --------------------------------------------------------------------------- #


def _safe(name: str) -> DriverError:
    return DriverError(
        driver_name=name,
        driver_url="http://x",
        status_code=503,
        problem=Problem(
            type="about:blank", title="down", status=503, retryDisposition=RetryDisposition.safe
        ),
        raw_body="down",
    )


def test_a_proven_failure_moves_to_the_next_tier(settings: Settings) -> None:
    dead = Painter(name="a", model_id="flux", fail=_safe("a"))
    spare = Painter(name="b", model_id="mini", caps=MINI)
    slots = [{"model": "pictures", "targets": ["flux", "mini"]}]
    with serve(settings, dead, spare, slots=slots) as client:
        response = generate(client, "pictures")
    assert response.status_code == 200, response.text
    assert dead.asked and spare.asked
    assert response.json()["x_eugene_plexus"]["attempts"] == 2


def test_a_providers_refusal_does_not_cascade(settings: Settings) -> None:
    refusal = DriverError(
        driver_name="a",
        driver_url="http://x",
        status_code=400,
        problem=Problem(
            type="about:blank",
            title="Backend rejected the request",
            status=400,
            detail="Black Forest Labs refused this prompt",
            retryDisposition=RetryDisposition.terminal,
        ),
        raw_body="refused",
    )
    filtered = Painter(name="a", model_id="flux", fail=refusal)
    spare = Painter(name="b", model_id="mini", caps=MINI)
    slots = [{"model": "pictures", "targets": ["flux", "mini"]}]
    with serve(settings, filtered, spare, slots=slots) as client:
        response = generate(client, "pictures")
    assert response.status_code == 400, response.text
    assert "Black Forest Labs" in response.text
    assert not spare.asked


# --------------------------------------------------------------------------- #
# Streaming (P4-3)
# --------------------------------------------------------------------------- #


def test_a_stream_to_a_model_that_cannot_is_refused_naming_stream(settings: Settings) -> None:
    flux = Painter(name="a", model_id="flux", caps=FLUX)
    with serve(settings, flux) as client:
        said = _refused(generate(client, stream=True), "stream")
    assert "image_streaming" in said
    assert not flux.asked


def test_a_stream_routes_past_a_model_that_cannot(settings: Settings) -> None:
    flux = Painter(name="a", model_id="flux", caps=FLUX)
    mini = Painter(name="b", model_id="mini", caps=MINI, images=2)
    slots = [{"model": "pictures", "targets": ["flux", "mini"]}]
    with serve(settings, flux, mini, slots=slots) as client:
        response = generate(client, "pictures", stream=True, partial_images=1, quality="low")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert not flux.asked and mini.asked[0].partialImages == 1
    events = _events(response.text)
    assert [e["type"] for e in events] == [
        "image_generation.partial_image",
        "image_generation.completed",
        "image_generation.completed",
    ]
    # Every field OpenAI's schema requires, which OpenRouter's omit.
    for event in events:
        for key in ("b64_json", "created_at", "size", "quality", "background", "output_format"):
            assert key in event, (key, event["type"])
    assert events[0]["partial_image_index"] == 0 and events[0]["quality"] == "low"
    assert events[-1]["usage"]["output_tokens"] == 4096
    assert events[-1]["x_eugene_plexus"]["tier"] == 2
    assert "x_eugene_plexus" not in events[1]


def test_a_streamed_edit_speaks_in_edit_events(settings: Settings) -> None:
    painter = Painter(name="a", model_id="gpt-image-1", caps=OPENAI)
    with serve(settings, painter) as client:
        response = edit(
            client, "gpt-image-1", files={"image": ("a.png", PNG, "image/png")}, stream="true"
        )
    assert [e["type"] for e in _events(response.text)] == [
        "image_edit.partial_image",
        "image_edit.completed",
    ]


def test_a_failure_before_the_first_event_cascades(settings: Settings) -> None:
    dead = Painter(name="a", model_id="mini", caps=MINI, fail=_safe("a"))
    spare = Painter(name="b", model_id="mini", caps=MINI)
    with serve(settings, dead, spare) as client:
        responses = [generate(client, "mini", stream=True) for _ in range(2)]
    assert all(r.status_code == 200 for r in responses)
    assert all(_events(r.text)[-1]["type"] == "image_generation.completed" for r in responses)


def test_after_the_first_event_a_failure_ends_the_stream_and_does_not_cascade(
    settings: Settings,
) -> None:
    breaks = Painter(name="a", model_id="mini", caps=MINI, fail=_safe("a"), fail_after_first=True)
    with serve(settings, breaks) as client:
        response = generate(client, "mini", stream=True)
    assert response.status_code == 200
    events = _events(response.text)
    assert [e["type"] for e in events] == ["image_generation.partial_image", "error"]


# --------------------------------------------------------------------------- #
# Retained as served
# --------------------------------------------------------------------------- #


def _rows(client: TestClient) -> list[dict[str, Any]]:
    for _ in range(50):
        rows = client.get("/v1/metrics/requests").json()["requests"]
        if rows:
            return rows
        time.sleep(0.02)
    return []


def test_a_served_image_is_retained_with_its_count_and_tokens(settings: Settings) -> None:
    painter = Painter(name="a", model_id="mini", caps=MINI, images=2)
    with serve(settings, painter) as client:
        assert generate(client, "mini", n=2).status_code == 200
        [row] = _rows(client)
    assert (row["outcome"], row["servedModel"], row["door"], row["images"]) == (
        "served",
        "mini",
        "images",
        2,
    )
    assert (row["promptTokens"], row["completionTokens"]) == (6, 4096)


def test_a_streamed_image_is_retained_as_streamed(settings: Settings) -> None:
    painter = Painter(name="a", model_id="mini", caps=MINI)
    with serve(settings, painter) as client:
        assert generate(client, "mini", stream=True).status_code == 200
        [row] = _rows(client)
    assert (row["outcome"], row["streamed"], row["images"]) == ("served", True, 1)
