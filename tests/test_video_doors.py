"""POST /v1/videos, GET /v1/videos/{video_id} and its content (P5).

The shapes are OpenAI's (the SDK's `client.videos`), and the capabilities are
OpenRouter's `/videos/models` as the driver reports them. The done-when: a job
submitted through one gateway is polled and downloaded through it, another
key's poll is refused, and a restart loses nothing. Every test here fails
against the gateway before P5, which had no videos door.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway import video_doors
from eugene_plexus_gateway._generated.driver_models import (
    Problem,
    RetryDisposition,
    VideoCapabilities,
    VideoJob,
    VideoJobStatus,
    VideoPrice,
    VideoPriceUnit,
    VideoRequest,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import DriverError
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table
from .test_admission import Authority

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
MP4 = b"\x00\x00\x00 ftypisom" + bytes(range(256)) * 8
#: grok-imagine-video's price list, as the driver reads OpenRouter's (2026-10-08).
GROK_PRICES = [
    VideoPrice(sku="cents_per_image_input", per=VideoPriceUnit.input_image, usd=0.002),
    VideoPrice(
        sku="cents_per_video_output_second_480p",
        per=VideoPriceUnit.second,
        usd=0.05,
        resolution="480p",
        sizes=["854x480"],
    ),
]
#: grok-imagine-video, as measured (trimmed).
GROK = VideoCapabilities(
    durations=[1, 2, 3, 4, 5], sizes=["854x480", "1280x720"], firstFrame=True, prices=GROK_PRICES
)
WIDE = VideoCapabilities(durations=list(range(1, 16)), sizes=None, firstFrame=False)


class Director(FakeDriverClient):
    """A fake whose model makes videos, as a driver does since P5."""

    def __init__(
        self,
        *,
        caps: VideoCapabilities | None = GROK,
        states: list[VideoJob] | None = None,
        fail: Exception | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.caps = caps
        self.states = states
        self.fail = fail
        self.submitted: list[VideoRequest] = []
        self.polled: list[str] = []

    def describe(self):  # type: ignore[no-untyped-def]
        info = super().describe()
        for model in info.models or []:
            model.surfaces = ["video"]
            assert model.capabilities is not None
            model.capabilities.video = self.caps
        return info

    async def video(self, request: VideoRequest) -> VideoJob:
        self.submitted.append(request)
        if self.fail is not None:
            raise self.fail
        return VideoJob(
            jobId=f"gen-vid-{self.name}", status=VideoJobStatus.queued, modelId=request.model
        )

    async def video_job(self, job_id: str) -> VideoJob:
        self.polled.append(job_id)
        if self.states:
            return self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return VideoJob(jobId=job_id, status=VideoJobStatus.completed, cost=0.05)

    async def video_content(self, job_id: str):  # type: ignore[no-untyped-def]
        yield MP4[:100]
        yield MP4[100:]


class Chatter(FakeDriverClient):
    """A chat model, for the wrong-door refusals."""


def serve(settings: Settings, *drivers: FakeDriverClient, **table: Any) -> TestClient:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*drivers, **table)
    return TestClient(app)


def submit(
    client: TestClient, model: str = "grok", headers: Any = None, **fields: Any
) -> httpx.Response:
    return client.post(
        "/v1/videos",
        json={"model": model, "prompt": "a red ball bouncing", **fields},
        headers=headers,
    )


def _refused(response: httpx.Response, param: str | None) -> str:
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == param, error
    return str(error["message"])


# --------------------------------------------------------------------------- #
# Listed, submitted, polled, downloaded
# --------------------------------------------------------------------------- #


def test_a_video_model_is_listed_with_what_it_takes(settings: Settings) -> None:
    with serve(
        settings, Director(name="a", model_id="grok"), Chatter(name="c", model_id="qwen")
    ) as client:
        models = {m["id"]: m["x_eugene_plexus"] for m in client.get("/v1/models").json()["data"]}
    assert models["grok"]["surfaces"] == ["video"]
    assert models["grok"]["video_durations"] == [1, 2, 3, 4, 5]
    assert models["grok"]["video_sizes"] == ["1280x720", "854x480"]
    assert models["grok"]["video_first_frame"] is True
    # The price list, in the gateway's own words (first_frame, not firstFrame);
    # the envelope sends an unset field as null, which reads as absent.
    prices = [
        {k: v for k, v in line.items() if v is not None} for line in models["grok"]["video_prices"]
    ]
    assert prices == [
        {"sku": "cents_per_image_input", "per": "input_image", "usd": 0.002},
        {
            "sku": "cents_per_video_output_second_480p",
            "per": "second",
            "usd": 0.05,
            "resolution": "480p",
            "sizes": ["854x480"],
        },
    ]
    assert models["qwen"].get("video_durations") is None
    assert models["qwen"].get("video_prices") is None


def test_a_price_lines_conditions_are_listed_in_the_gateways_words(settings: Settings) -> None:
    """Sound, the kind of start, and a minimum: each says when its line holds."""
    caps = VideoCapabilities(
        durations=[4, 8],
        sizes=None,
        firstFrame=True,
        prices=[
            VideoPrice(
                sku="duration_seconds_with_audio", per=VideoPriceUnit.second, usd=0.4, audio=True
            ),
            VideoPrice(
                sku="image_to_video_duration_seconds",
                per=VideoPriceUnit.second,
                usd=0.15,
                firstFrame=True,
            ),
            VideoPrice(sku="minimum_cents_per_generation", per=VideoPriceUnit.minimum, usd=0.56),
        ],
    )
    with serve(settings, Director(name="a", model_id="veo", caps=caps)) as client:
        info = client.get("/v1/models").json()["data"][0]["x_eugene_plexus"]
    lines = [{k: v for k, v in line.items() if v is not None} for line in info["video_prices"]]
    assert lines == [
        {"sku": "duration_seconds_with_audio", "per": "second", "usd": 0.4, "audio": True},
        {
            "sku": "image_to_video_duration_seconds",
            "per": "second",
            "usd": 0.15,
            "first_frame": True,
        },
        {"sku": "minimum_cents_per_generation", "per": "minimum", "usd": 0.56},
    ]


def test_a_price_is_listed_only_when_every_backend_lists_the_same(settings: Settings) -> None:
    """A request may land on any backend serving the name, so one figure is
    true only when they agree; a backend with no list makes it unknown."""
    dearer = VideoCapabilities.model_validate(
        {**GROK.model_dump(), "prices": [{**GROK_PRICES[1].model_dump(), "usd": 0.07}]}
    )
    unpriced = VideoCapabilities.model_validate({**GROK.model_dump(), "prices": None})
    for other, listed in ((GROK, True), (dearer, False), (unpriced, False)):
        with serve(
            settings,
            Director(name="a", model_id="grok"),
            Director(name="b", model_id="grok", caps=other),
        ) as client:
            info = client.get("/v1/models").json()["data"][0]["x_eugene_plexus"]
        assert (info.get("video_prices") is not None) is listed, (other, info)


def test_a_submit_answers_openais_resource_with_a_signed_handle(settings: Settings) -> None:
    director = Director(name="a", model_id="grok")
    with serve(settings, director) as client:
        response = submit(client, seconds="4", size="1280x720")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"].startswith("video_") and body["object"] == "video"
    assert (body["status"], body["progress"], body["seconds"], body["size"]) == (
        "queued",
        0,
        "4",
        "1280x720",
    )
    assert body["prompt"] is None and body["error"] is None and body["model"] == "grok"
    assert body["x_eugene_plexus"]["driver"] == "a"
    [sent] = director.submitted
    assert (sent.prompt, sent.seconds, sent.size, sent.firstFrame) == (
        "a red ball bouncing",
        4,
        "1280x720",
        None,
    )


def test_the_job_is_polled_and_downloaded_through_the_same_gateway(settings: Settings) -> None:
    director = Director(
        name="a",
        model_id="grok",
        states=[
            # One queued answer per poll before it completes: the first poll,
            # and the download's own check.
            VideoJob(jobId="gen-vid-a", status=VideoJobStatus.queued),
            VideoJob(jobId="gen-vid-a", status=VideoJobStatus.queued),
            VideoJob(jobId="gen-vid-a", status=VideoJobStatus.completed, cost=0.05),
        ],
    )
    with serve(settings, director) as client:
        handle = submit(client, seconds="1").json()["id"]
        queued = client.get(f"/v1/videos/{handle}").json()
        too_soon = client.get(f"/v1/videos/{handle}/content")
        done = client.get(f"/v1/videos/{handle}").json()
        video = client.get(f"/v1/videos/{handle}/content")
    assert queued["status"] == "queued" and queued["id"] == handle
    # What the provider billed, once the job has ended and it says (§6.4).
    assert "x_eugene_plexus" not in queued
    assert done["x_eugene_plexus"] == {"cost_usd": 0.05}
    assert too_soon.status_code == 400 and "no video yet" in too_soon.json()["error"]["message"]
    assert (done["status"], done["progress"]) == ("completed", 100)
    assert done["completed_at"] is not None and done["seconds"] == "1"
    assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"
    assert video.content == MP4
    assert set(director.polled) == {"gen-vid-a"}


def test_a_failed_job_says_why_in_openais_shape(settings: Settings) -> None:
    director = Director(
        name="a",
        model_id="grok",
        states=[
            VideoJob(
                jobId="gen-vid-a",
                status=VideoJobStatus.failed,
                error="Image dimensions 1x1 are too small.",
            )
        ],
    )
    with serve(settings, director) as client:
        handle = submit(client).json()["id"]
        body = client.get(f"/v1/videos/{handle}").json()
    assert body["status"] == "failed"
    assert body["error"] == {
        "code": "video_generation_failed",
        "message": "Image dimensions 1x1 are too small.",
    }


def test_a_restart_loses_nothing(settings: Settings) -> None:
    director = Director(name="a", model_id="grok")
    with serve(settings, director) as client:
        handle = submit(client).json()["id"]
    with serve(settings, director) as restarted:
        polled = restarted.get(f"/v1/videos/{handle}")
    assert polled.status_code == 200, polled.text
    assert (settings.config_file.parent / "video-handles.key").exists()


def test_a_forged_or_foreign_handle_reads_as_not_found(settings: Settings, tmp_path: Path) -> None:
    director = Director(name="a", model_id="grok")
    foreign = video_doors.Handles(tmp_path / "elsewhere" / "video-handles.key").issue(
        {"v": 1, "d": "a", "n": None, "m": "grok", "j": "gen-vid-a", "o": "install", "c": 1}
    )
    with serve(settings, director) as client:
        handle = submit(client).json()["id"]
        tampered = handle[:-4] + ("AAAA" if not handle.endswith("AAAA") else "BBBB")
        answers = [client.get(f"/v1/videos/{h}") for h in (tampered, foreign, "video_nonsense")]
    assert [a.status_code for a in answers] == [404, 404, 404]
    assert not director.polled


def test_the_backend_holding_a_job_being_gone_is_a_503_not_a_loss(settings: Settings) -> None:
    director = Director(name="a", model_id="grok")
    with serve(settings, director) as client:
        handle = submit(client).json()["id"]
    with serve(settings, Chatter(name="c", model_id="qwen")) as without:
        response = without.get(f"/v1/videos/{handle}")
    assert response.status_code == 503, response.text
    assert "not lost" in response.json()["error"]["message"]


def test_an_upstream_that_forgot_the_job_is_a_404(settings: Settings) -> None:
    class Forgetful(Director):
        async def video_job(self, job_id: str) -> VideoJob:
            raise DriverError(
                driver_name="a",
                driver_url="http://a",
                status_code=404,
                problem=Problem(type="about:blank", title="No such video job", status=404),
                raw_body="",
            )

    with serve(settings, Forgetful(name="a", model_id="grok")) as client:
        handle = submit(client).json()["id"]
        assert client.get(f"/v1/videos/{handle}").status_code == 404


def test_a_thumbnail_is_refused_rather_than_answered_with_the_video(settings: Settings) -> None:
    with serve(settings, Director(name="a", model_id="grok")) as client:
        handle = submit(client).json()["id"]
        response = client.get(f"/v1/videos/{handle}/content", params={"variant": "thumbnail"})
    _refused(response, "variant")


def test_the_list_is_refused_and_remix_is_not_routed(settings: Settings) -> None:
    with serve(settings, Director(name="a", model_id="grok")) as client:
        handle = submit(client).json()["id"]
        listed = client.get("/v1/videos")
        remix = client.post(f"/v1/videos/{handle}/remix", json={"prompt": "x"})
        deleted = client.delete(f"/v1/videos/{handle}")
    assert listed.status_code == 400 and "no store" in listed.json()["error"]["message"]
    assert remix.status_code == 404
    assert deleted.status_code == 405


# --------------------------------------------------------------------------- #
# Another key's poll is refused
# --------------------------------------------------------------------------- #


@pytest.fixture
def two_keys(settings: Settings, install: Any) -> Any:
    authority = Authority()
    authority.allowed = ["grok"]
    one = install.client_key(name="One", jti="key-1")
    two = install.client_key(name="Two", jti="key-2")
    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    app.state.routing = make_routing_table(Director(name="a", model_id="grok"))
    return app, {"Authorization": "Bearer " + one}, {"Authorization": "Bearer " + two}


def test_another_keys_poll_is_refused(two_keys: Any) -> None:
    app, one, two = two_keys
    with TestClient(app) as client:
        made = submit(client, headers=one)
        assert made.status_code == 200, made.text
        handle = made.json()["id"]
        mine = client.get(f"/v1/videos/{handle}", headers=one)
        theirs = client.get(f"/v1/videos/{handle}", headers=two)
        their_download = client.get(f"/v1/videos/{handle}/content", headers=two)
    assert mine.status_code == 200, mine.text
    assert theirs.status_code == 404 and their_download.status_code == 404


def test_an_operator_session_cannot_read_a_client_keys_job(two_keys: Any) -> None:
    app, one, _two = two_keys
    with TestClient(app) as client:
        handle = submit(client, headers=one).json()["id"]
        payload = app.state.video_handles.read(handle)
        assert payload["o"] == "key:key-1"


# --------------------------------------------------------------------------- #
# Settings route by the model's own listing
# --------------------------------------------------------------------------- #


def test_a_duration_a_model_does_not_list_routes_to_one_that_does(settings: Settings) -> None:
    grok = Director(name="a", model_id="grok", caps=GROK)
    wide = Director(name="b", model_id="wide", caps=WIDE)
    slots = [{"model": "clips", "targets": ["grok", "wide"]}]
    with serve(settings, grok, wide, slots=slots) as client:
        response = submit(client, "clips", seconds="10")
    assert response.status_code == 200, response.text
    assert not grok.submitted and wide.submitted
    assert response.json()["x_eugene_plexus"]["tier"] == 2


def test_a_setting_no_model_takes_is_refused_naming_it(settings: Settings) -> None:
    grok = Director(name="a", model_id="grok", caps=GROK)
    with serve(settings, grok) as client:
        seconds = _refused(submit(client, seconds="12"), "seconds")
        size = _refused(submit(client, size="720x1280"), "size")
        odd = _refused(submit(client, seconds="four"), "seconds")
    assert "1, 2, 3, 4, 5" in seconds and "Nothing was sent" in seconds
    assert "854x480" in size and odd
    assert not grok.submitted


def test_a_first_frame_routes_only_where_it_is_taken(settings: Settings) -> None:
    grok = Director(name="a", model_id="grok", caps=GROK)
    wide = Director(name="b", model_id="wide", caps=WIDE)
    data = "data:image/png;base64," + base64.b64encode(PNG).decode()
    with serve(settings, grok, wide) as client:
        refused = submit(client, "wide", input_reference={"image_url": data})
        served = submit(client, "grok", input_reference={"image_url": data})
    _refused(refused, "input_reference")
    assert served.status_code == 200, served.text
    frame = grok.submitted[0].firstFrame
    assert frame is not None and frame.mediaType == "image/png"
    assert not wide.submitted


def test_a_url_or_file_id_reference_is_refused_unsent(settings: Settings) -> None:
    grok = Director(name="a", model_id="grok", caps=GROK)
    with serve(settings, grok) as client:
        url = submit(client, input_reference={"image_url": "https://example.com/a.png"})
        file_id = submit(client, input_reference={"file_id": "file-abc"})
    assert "not fetched" in _refused(url, "input_reference.image_url")
    assert "keeps none" in _refused(file_id, "input_reference.file_id")
    assert not grok.submitted


def test_the_sdks_multipart_submit_carries_the_reference_file(settings: Settings) -> None:
    grok = Director(name="a", model_id="grok", caps=GROK)
    with serve(settings, grok) as client:
        response = client.post(
            "/v1/videos",
            data={
                "model": "grok",
                "prompt": "the square turns",
                "seconds": "4",
                "size": "1280x720",
            },
            files={"input_reference": ("frame.png", PNG, "application/octet-stream")},
        )
    assert response.status_code == 200, response.text
    [sent] = grok.submitted
    assert sent.firstFrame is not None and sent.firstFrame.mediaType == "image/png"
    assert (sent.seconds, sent.size) == (4, "1280x720")


def test_an_unknown_field_is_refused_by_name(settings: Settings) -> None:
    with serve(settings, Director(name="a", model_id="grok")) as client:
        _refused(submit(client, duration=4), "duration")


def test_the_wrong_door_is_named_both_ways(settings: Settings) -> None:
    with serve(
        settings, Director(name="a", model_id="grok"), Chatter(name="c", model_id="qwen")
    ) as client:
        at_videos = submit(client, "qwen")
        at_chat = client.post(
            "/v1/chat/completions",
            json={"model": "grok", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert "/v1/chat/completions" in _refused(at_videos, "model")
    assert "/v1/videos" in at_chat.json()["error"]["message"]


# --------------------------------------------------------------------------- #
# Failover at submit only; retained
# --------------------------------------------------------------------------- #


def test_a_proven_submit_failure_moves_to_the_next_tier(settings: Settings) -> None:
    dead = Director(
        name="a",
        model_id="grok",
        fail=DriverError(
            driver_name="a",
            driver_url="http://a",
            status_code=503,
            problem=Problem(
                type="about:blank", title="down", status=503, retryDisposition=RetryDisposition.safe
            ),
            raw_body="",
        ),
    )
    spare = Director(name="b", model_id="wide", caps=WIDE)
    slots = [{"model": "clips", "targets": ["grok", "wide"]}]
    with serve(settings, dead, spare, slots=slots) as client:
        response = submit(client, "clips", seconds="4")
        handle = response.json()["id"]
        polled = client.get(f"/v1/videos/{handle}")
    assert response.status_code == 200, response.text
    assert dead.submitted and spare.submitted
    # The job is the spare's for life: the poll goes there and nowhere else.
    assert spare.polled and not dead.polled
    assert polled.status_code == 200


def _rows(client: TestClient) -> list[dict[str, Any]]:
    import time

    for _ in range(50):
        rows = client.get("/v1/metrics/requests").json()["requests"]
        if rows:
            return rows
        time.sleep(0.02)
    return []


def test_a_submit_is_retained_with_its_seconds(settings: Settings) -> None:
    with serve(settings, Director(name="a", model_id="grok")) as client:
        handle = submit(client, seconds="4").json()["id"]
        client.get(f"/v1/videos/{handle}")
        rows = _rows(client)
    assert [(r["door"], r["videoSeconds"], r["outcome"], r["servedModel"]) for r in rows] == [
        ("videos", 4, "served", "grok")
    ]


def test_a_template_door_matches_one_segment_and_nothing_else() -> None:
    from eugene_plexus_gateway.door_paths import matches

    doors = {"/v1/videos", "/v1/videos/{video_id}", "/v1/videos/{video_id}/content"}
    assert matches("/v1/videos", doors)
    assert matches("/v1/videos/video_abc.def", doors)
    assert matches("/v1/videos/video_abc/content", doors)
    assert not matches("/v1/videos/video_abc/remix", doors)
    assert not matches("/v1/videos/", doors)
    assert not matches("/v1/videos/a/b/content", doors)
