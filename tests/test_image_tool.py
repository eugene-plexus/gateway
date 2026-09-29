"""P8e: OpenAI's `image_generation` tool, run by this install on /v1/responses.

A scripted chat model calls `image_generation`, and a fake image model
(P4's `Painter`) makes the image. What is pinned here:

- the loop offers the model an `image_generation` function and routes each
  call exactly as the images door would, in-process;
- the model is told in words (P8e-2) and never handed the image's bytes;
  the caller gets an `image_generation_call` item carrying them, streamed
  in_progress -> generating -> completed;
- which image model answers (P8e-1): the tool's own, else the gateway's
  `imageToolModel`, else the one there is;
- when it cannot run, a 400 naming why (P8e-3), never a silent removal;
- one budget for searches and images; a failed image is told to the model;
- a handed-back item becomes history, not bytes (P8e-4);
- metrics keep one row per image, no prompt, and the image backend's
  attempts are not counted as the request's.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

from fastapi.testclient import TestClient

from eugene_plexus_gateway import admission, server_tools
from eugene_plexus_gateway._generated.driver_models import FunctionCall, Problem, ToolCall
from eugene_plexus_gateway.driver_client import DriverError

from .conftest import make_routing_table
from .test_image_doors import PNG, Painter
from .test_server_tools import (
    MODEL,
    FakeSearch,
    ScriptedDriver,
    Turn,
    account,
    app_with,
    events,
    search_call,
)

B64 = base64.b64encode(PNG).decode()


def image_call(
    prompt: str = "a red fox in snow", call_id: str = "call_i1", **extra: Any
) -> ToolCall:
    return ToolCall(
        id=call_id,
        type="function",
        function=FunctionCall(
            name="image_generation", arguments=json.dumps({"prompt": prompt, **extra})
        ),
    )


def painter(name: str = "painter", model_id: str = "flux", **kw: Any) -> Painter:
    return Painter(name=name, model_id=model_id, **kw)


def ask(client: TestClient, *, tool: dict[str, Any] | None = None, **extra: Any) -> Any:
    return client.post(
        "/v1/responses",
        json={
            "model": MODEL,
            "input": "Draw me a fox.",
            "tools": [{"type": "image_generation", **(tool or {})}],
            **extra,
        },
    )


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def test_a_local_model_makes_an_image_and_the_caller_gets_it(settings) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[image_call()]), Turn("Here is your fox.")])
    art = painter()
    with TestClient(app_with(settings, model, art)) as client:
        response = ask(client)
    assert response.status_code == 200, response.text
    output = response.json()["output"]
    assert [i["type"] for i in output] == ["image_generation_call", "message"]
    item = output[0]
    assert item["status"] == "completed"
    assert item["result"] == B64
    assert item["revised_prompt"] == "a red fox in snow"
    assert item["output_format"] == "png"
    assert output[1]["content"][0]["text"] == "Here is your fox."
    # What the image model was asked, routed as the images door routes it.
    assert [r.prompt for r in art.asked] == ["a red fox in snow"]
    assert art.asked[0].n == 1
    # What the chat model was offered and told: words, never the bytes.
    first, second = model.calls
    assert [t.function.name for t in first.tools or []] == ["image_generation"]
    assert str(first.toolChoice) != "required", "an image tool does not force a call"
    told = second.messages[-1]
    assert told.role.value == "tool" and told.toolCallId == "call_i1"
    assert "An image was made" in told.content and "a red fox in snow" in told.content
    assert B64 not in json.dumps([m.model_dump(mode="json") for m in second.messages])
    assert response.headers["x-eugene-plexus-attempts"] == "1", (
        "the image backend's attempt is not one of the model's"
    )
    assert response.headers["x-eugene-plexus-driver"] == "local"


def test_the_image_streams_as_openai_streams_its_own(settings) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[image_call()]), Turn("Ready.")])
    with TestClient(app_with(settings, model, painter())) as client:
        response = ask(client, stream=True)
    assert response.status_code == 200, response.text
    names = [n for n, _ in events(response)]
    wanted = [
        "response.image_generation_call.in_progress",
        "response.image_generation_call.generating",
        "response.image_generation_call.completed",
    ]
    positions = [names.index(n) for n in wanted]
    assert positions == sorted(positions), names
    added = [d["item"] for n, d in events(response) if n == "response.output_item.added"]
    assert added[0]["type"] == "image_generation_call" and added[0]["result"] is None
    done = [d["item"] for n, d in events(response) if n == "response.output_item.done"]
    image = next(i for i in done if i["type"] == "image_generation_call")
    assert image["result"] == B64 and image["status"] == "completed"
    completed = next(d for n, d in events(response) if n == "response.completed")
    kinds = [i["type"] for i in completed["response"]["output"]]
    assert kinds[0] == "image_generation_call"
    assert not [i for i in added if i["type"] == "function_call"], (
        "our call never reaches the caller"
    )


def test_the_tools_settings_ride_on_the_image_and_the_unhonoured_ones_are_named(settings) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[image_call(size="512x512")]), Turn("Ready.")])
    art = painter()
    with TestClient(app_with(settings, model, art)) as client:
        response = ask(
            client,
            tool={"size": "1536x1024", "output_format": "png", "partial_images": 2},
        )
    assert response.status_code == 200, response.text
    assert art.asked[0].size == "1536x1024", "the caller's size is the caller's, not the model's"
    assert art.asked[0].outputFormat == "png"
    ignored = response.headers["x-eugene-plexus-ignored-settings"]
    assert "tools.image_generation.partial_images" in ignored
    offered = model.calls[0].tools[0].function.parameters  # type: ignore[index,union-attr]
    assert "size" not in offered["properties"], "a size the caller set is not offered"


# --------------------------------------------------------------------------- #
# Which image model (P8e-1)
# --------------------------------------------------------------------------- #


def test_the_one_image_model_answers_when_there_is_one(settings) -> None:
    table = make_routing_table(ScriptedDriver(turns=[]), painter())
    plan = server_tools.ImagePlan()
    assert server_tools.image_model(table, plan, None, None) == ("flux", "")


def test_several_image_models_ask_for_the_setting_and_the_setting_and_the_tool_choose(
    settings,
) -> None:
    table = make_routing_table(
        ScriptedDriver(turns=[]), painter(), painter("other", model_id="mini")
    )
    chosen, why = server_tools.image_model(table, server_tools.ImagePlan(), None, None)
    assert chosen is None and "flux" in why and "mini" in why and "Image model for tools" in why
    assert server_tools.image_model(table, server_tools.ImagePlan(), None, "mini") == ("mini", "")
    named = server_tools.ImagePlan(model="flux")
    assert server_tools.image_model(table, named, None, "mini") == ("flux", ""), (
        "the tool's own model wins over the setting"
    )
    chosen, why = server_tools.image_model(table, server_tools.ImagePlan(), None, "nowhere")
    assert chosen is None and "nowhere" in why


def test_the_setting_picks_the_model_through_the_route(settings) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[image_call()]), Turn("Ready.")])
    flux, mini = painter(), painter("other", model_id="mini")
    with TestClient(app_with(settings, model, flux, mini)) as client:
        refused = ask(client)
        assert refused.status_code == 400, refused.text
        assert client.patch("/v1/config", json={"imageToolModel": "mini"}).status_code == 200
        response = ask(client)
    assert response.status_code == 200, response.text
    assert flux.asked == [] and len(mini.asked) == 1


def test_a_key_that_may_not_make_images_is_told_why(settings) -> None:
    table = make_routing_table(ScriptedDriver(turns=[]), painter())
    for limits, words in (
        ({"allowedTools": ["web_search"]}, "tool scope"),
        ({"localOnly": True}, "local-only"),
        ({"allowedModels": [MODEL]}, "no image model"),
    ):
        context = admission.ClientRequest({})
        context.access = {"limits": limits}
        chosen, why = server_tools.image_model(table, server_tools.ImagePlan(), context, None)
        assert chosen is None and words in why, (limits, why)
    allowed = admission.ClientRequest({})
    allowed.access = {"limits": {"allowedTools": ["image_*"]}}
    assert server_tools.image_model(table, server_tools.ImagePlan(), allowed, None)[0] == "flux"


# --------------------------------------------------------------------------- #
# When it cannot run (P8e-3)
# --------------------------------------------------------------------------- #


def test_no_image_model_is_a_400_naming_why_and_nothing_is_forwarded(settings) -> None:
    model = ScriptedDriver(turns=[Turn("I cannot draw.")])
    with TestClient(app_with(settings, model)) as client:
        response = ask(client)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert "image generation could not run" in error["message"]
    assert "no image model is served here" in error["message"]
    assert error["param"] == "tools"
    assert model.calls == []


def test_a_model_that_cannot_call_tools_is_refused_saying_so(settings) -> None:
    model = ScriptedDriver(turns=[Turn("hi")])
    model.supports_tools = False
    with TestClient(app_with(settings, model, painter())) as client:
        response = ask(client)
    assert response.status_code == 400, response.text
    assert "can make images" in response.json()["error"]["message"]
    assert model.calls == []


def test_a_failed_image_is_told_to_the_model_and_its_item_is_failed(settings) -> None:
    broken = painter(
        fail=DriverError(
            driver_name="painter",
            driver_url="http://painter.invalid",
            status_code=400,
            problem=Problem(
                type="about:blank", title="Refused", status=400, detail="the prompt was refused"
            ),
            raw_body="",
        )
    )
    model = ScriptedDriver(turns=[Turn(calls=[image_call()]), Turn("It failed, sorry.")])
    with TestClient(app_with(settings, model, broken)) as client:
        response = ask(client)
    assert response.status_code == 200, response.text
    item = response.json()["output"][0]
    assert item["type"] == "image_generation_call"
    assert item["status"] == "failed" and item["result"] is None
    told = model.calls[1].messages[-1].content
    assert "could not be made" in told and "the prompt was refused" in told


def test_one_budget_covers_searches_and_images(settings) -> None:
    search = FakeSearch()
    model = ScriptedDriver(
        turns=[
            Turn(calls=[search_call()]),
            Turn(calls=[image_call()]),
            Turn("Done."),
        ]
    )
    art = painter()
    app = app_with(settings, model, art, searches=[account(search)])
    with TestClient(app) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": MODEL,
                "input": "look it up and draw it",
                "tools": [{"type": "web_search"}, {"type": "image_generation"}],
                "max_tool_calls": 1,
            },
        )
    assert response.status_code == 200, response.text
    assert len(search.asked) == 1 and art.asked == [], "the image was past the shared limit"
    kinds = [i["type"] for i in response.json()["output"]]
    assert kinds == ["web_search_call", "image_generation_call", "message"]
    assert response.json()["output"][1]["status"] == "failed"
    assert "No more images or searches" in model.calls[2].messages[-1].content
    offered = {t.function.name for t in model.calls[0].tools or []}
    assert offered == {"web_search", "image_generation"}


def test_a_handed_back_image_becomes_history_not_bytes(settings) -> None:
    model = ScriptedDriver(turns=[Turn("A fox, as asked.")])
    with TestClient(app_with(settings, model, painter())) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": MODEL,
                "input": [
                    {"role": "user", "content": "Draw me a fox."},
                    {
                        "type": "image_generation_call",
                        "id": "ig_1",
                        "status": "completed",
                        "result": B64,
                        "revised_prompt": "a red fox in snow",
                    },
                    {"role": "user", "content": "What did you draw?"},
                ],
                "tools": [{"type": "image_generation"}],
            },
        )
    assert response.status_code == 200, response.text
    sent = json.dumps([m.model_dump(mode="json") for m in model.calls[0].messages])
    assert 'Made an image for the prompt \\"a red fox in snow\\"' in sent
    assert B64 not in sent


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_metrics_keep_each_image_and_no_prompt(settings) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[image_call("secret words")]), Turn("Ready.")])
    with TestClient(app_with(settings, model, painter())) as client:
        assert ask(client).status_code == 200
        rows: list[Any] = []
        for _ in range(100):
            rows = client.get("/v1/metrics/requests").json()["requests"]
            if rows:
                break
            time.sleep(0.05)
    row = rows[0]
    assert row["attempts"] == 1
    assert len(row["tries"]) == 2, "the model's two turns, not the image backend's attempt"
    assert all(t["driver"] != "painter" for t in row["tries"])
    assert not row.get("webSearches"), "an image is not a search"
    [image] = row["imageGenerations"]
    assert image["tool"] == "image_generation"
    assert image["driver"] == "painter" and image["provider"] == "flux"
    assert image["outcome"] == "ok" and image["results"] == 1
    assert "secret words" not in json.dumps(row)
