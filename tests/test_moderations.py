"""POST /v1/moderations and GET /v1/models/{model} (P6).

The request shape is OpenAI's (the SDK's `client.moderations.create`), with
`model` optional (P6-1: the one moderation model this key may use) and a
verdict never taken from another model (P6-2). Every test here fails
against the gateway before P6, which had neither door.
"""

from __future__ import annotations

import base64
import io
import time
from typing import Any

import httpx
from fastapi.testclient import TestClient
from PIL import Image

from eugene_plexus_gateway._generated.driver_models import (
    ModerateRequest,
    ModerateResponse,
    ModerationPartType,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.door_paths import matches
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

RESULT = {
    "flagged": True,
    "categories": {"violence": True},
    "category_scores": {"violence": 0.87},
    "category_applied_input_types": {"violence": ["text"]},
}


def _png() -> str:
    out = io.BytesIO()
    Image.new("RGB", (16, 16), (200, 30, 30)).save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


class Moderator(FakeDriverClient):
    """A fake whose model moderates, as a driver's OpenAI account does."""

    def __init__(self, *, fail: Exception | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.fail = fail
        self.heard: list[ModerateRequest] = []

    def describe(self):  # type: ignore[no-untyped-def]
        info = super().describe()
        for model in info.models or []:
            model.surfaces = ["moderation"]
        return info

    async def moderate(self, request: ModerateRequest) -> ModerateResponse:
        self.heard.append(request)
        if self.fail is not None:
            raise self.fail
        count = len(request.texts) if request.texts else 1
        return ModerateResponse(
            id="modr-1", results=[RESULT] * count, modelId=request.model, latencyMs=5
        )


def serve(settings: Settings, *drivers: FakeDriverClient, **table: Any) -> TestClient:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*drivers, **table)
    return TestClient(app)


def moderate(client: TestClient, **body: Any) -> httpx.Response:
    return client.post("/v1/moderations", json=body)


def _rows(client: TestClient) -> list[dict[str, Any]]:
    for _ in range(50):
        rows = client.get("/v1/metrics/requests").json()["requests"]
        if rows:
            return rows
        time.sleep(0.02)
    return []


def test_a_moderation_model_is_listed_and_found_by_id(settings: Settings) -> None:
    with serve(settings, Moderator(name="a", model_id="omni")) as client:
        listed = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
        one = client.get("/v1/models/omni")
    assert listed["omni"]["x_eugene_plexus"]["surfaces"] == ["moderation"]
    assert one.status_code == 200 and one.json() == listed["omni"]


def test_the_sdks_shapes_are_heard_as_texts_or_parts(settings: Settings) -> None:
    moderator = Moderator(name="a", model_id="omni")
    image = _png()
    with serve(settings, moderator) as client:
        single = moderate(client, model="omni", input="I will hurt you.")
        batch = moderate(client, model="omni", input=["a", "b"])
        parts = moderate(
            client,
            model="omni",
            input=[
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": image}},
            ],
        )
    assert single.status_code == 200, single.text
    assert single.json() == {"id": "modr-1", "model": "omni", "results": [RESULT]}
    assert len(batch.json()["results"]) == 2
    assert parts.status_code == 200, parts.text
    said, listed, multimodal = moderator.heard
    assert [t.root for t in said.texts or []] == ["I will hurt you."] and said.parts is None
    assert [t.root for t in listed.texts or []] == ["a", "b"]
    assert multimodal.texts is None and [p.type for p in multimodal.parts or []] == [
        ModerationPartType.text,
        ModerationPartType.image,
    ]
    assert (multimodal.parts or [])[1].image == image


def test_left_out_model_is_the_only_moderation_model(settings: Settings) -> None:
    moderator = Moderator(name="a", model_id="omni")
    chat = FakeDriverClient(name="c", model_id="chatty")
    with serve(settings, moderator, chat) as client:
        response = moderate(client, input="hi")
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "omni" and moderator.heard


def test_a_slot_alias_is_not_a_second_moderation_model(settings: Settings) -> None:
    """Found by the acceptance run: `verdicts -> [omni]` made the one
    moderation model look like two, and `model` left out was refused."""
    moderator = Moderator(name="a", model_id="omni")
    slots = [{"model": "verdicts", "targets": ["omni"]}]
    with serve(settings, moderator, slots=slots) as client:
        response = moderate(client, input="hi")
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "omni"


def test_left_out_model_with_two_or_none_names_the_choices(settings: Settings) -> None:
    one, two = Moderator(name="a", model_id="omni"), Moderator(name="b", model_id="omni-2")
    with serve(settings, one, two) as client:
        several = moderate(client, input="hi")
    with serve(settings, FakeDriverClient(name="c", model_id="chatty")) as client:
        none = moderate(client, input="hi")
    assert several.status_code == 400 and several.json()["error"]["param"] == "model"
    assert "omni, omni-2" in several.json()["error"]["message"]
    assert none.status_code == 400 and "no moderation model" in none.json()["error"]["message"]
    assert not one.heard and not two.heard


def test_what_this_door_does_not_take_is_refused_naming_the_field(settings: Settings) -> None:
    moderator = Moderator(name="a", model_id="omni")
    with serve(settings, moderator) as client:
        answers = {
            "user": moderate(client, model="omni", input="hi", user="x"),
            "input": moderate(client, model="omni", input=[]),
            "mixed": moderate(client, model="omni", input=["a", {"type": "text", "text": "b"}]),
            "remote": moderate(
                client,
                model="omni",
                input=[{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}],
            ),
            "part type": moderate(client, model="omni", input=[{"type": "audio"}]),
        }
    for label, response in answers.items():
        assert response.status_code == 400, (label, response.text)
    assert answers["user"].json()["error"]["param"] == "user"
    assert "mixes strings and parts" in answers["mixed"].json()["error"]["message"]
    assert "URLs are not fetched" in answers["remote"].json()["error"]["message"]
    assert answers["part type"].json()["error"]["param"] == "input[0].type"
    assert not moderator.heard


def test_a_verdict_never_comes_from_another_model(settings: Settings) -> None:
    dead = Moderator(name="a", model_id="omni", fail=httpx.ConnectError("refused"))
    other = Moderator(name="b", model_id="omni-2")
    slots = [{"model": "mod", "targets": ["omni", "omni-2"]}]
    with serve(settings, dead, other, slots=slots) as client:
        response = moderate(client, model="mod", input="hi")
    assert response.status_code >= 500, response.text
    assert dead.heard and not other.heard


def test_a_replica_of_the_same_model_serves_when_one_fails(settings: Settings) -> None:
    dead = Moderator(name="a", model_id="omni", fail=httpx.ConnectError("refused"))
    replica = Moderator(name="b", model_id="omni")
    with serve(settings, dead, replica) as client:
        response = moderate(client, model="omni", input="hi")
        rows = _rows(client)
    assert response.status_code == 200, response.text
    assert replica.heard
    assert (rows[0]["door"], rows[0]["servedModel"], rows[0]["outcome"]) == (
        "moderation",
        "omni",
        "served",
    )


def test_a_chat_model_is_sent_to_the_chat_door(settings: Settings) -> None:
    with serve(settings, FakeDriverClient(name="c", model_id="chatty")) as client:
        response = moderate(client, model="chatty", input="hi")
    assert response.status_code == 400, response.text
    assert "/v1/chat/completions" in response.json()["error"]["message"]


def test_a_model_id_with_slashes_is_found_and_an_unknown_one_is_a_404(
    settings: Settings,
) -> None:
    account = FakeDriverClient(name="openrouter", models=["anthropic/claude-x"], account=True)
    with serve(settings, account) as client:
        found = client.get("/v1/models/openrouter/anthropic/claude-x")
        missing = client.get("/v1/models/openrouter/anthropic/nope")
    assert found.status_code == 200, found.text
    assert found.json()["id"] == "openrouter/anthropic/claude-x"
    assert missing.status_code == 404 and "does not exist" in missing.json()["error"]["message"]


def test_a_path_template_takes_the_rest_of_the_path() -> None:
    doors = {"/v1/models/{model:path}", "/v1/videos/{video_id}"}
    assert matches("/v1/models/a", doors) and matches("/v1/models/a/b/c", doors)
    assert not matches("/v1/models/", doors) and not matches("/v1/models/a//b", doors)
    assert matches("/v1/videos/x", doors) and not matches("/v1/videos/x/y", doors)
