"""Tool calls across the OpenAI-compatible front door.

Install-paths §9 step 6. The thesis work: before this, `tools` was not a
field the gateway accepted, so a harness's definitions were dropped by
FastAPI before any code saw them, the model never learned a tool
existed, and there was no return path for a call even if a backend had
made one. Claude Code, OpenCode and every other agent harness could not
work against this project -- not badly, at all.

Wire fidelity is what matters here, as in `test_inference.py`: the
assertions are on snake_case names, OpenAI's error envelope and OpenAI's
SSE framing, because an unmodified SDK is the audience.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import FunctionCall, ToolCall
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

MODEL = "Qwen3-30B-A3B-Q4_K_M"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the current weather for a place.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


def _app_with(settings: Settings, *fakes: FakeDriverClient) -> FastAPI:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*fakes)
    return app


def _tool_driver(name: str = "gpu-a", **kwargs: Any) -> FakeDriverClient:
    return FakeDriverClient(
        name=name, base_url=f"http://{name}", model_id=MODEL, supports_tools=True, **kwargs
    )


def _call(name: str = "get_weather", args: str = '{"location": "Oslo"}') -> ToolCall:
    return ToolCall(
        id="call_abc", type="function", function=FunctionCall(name=name, arguments=args)
    )


def _chat(**extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "weather in Oslo?"}],
    }
    body.update(extra)
    return body


# --------------------------------------------------------------------------- #
# down
# --------------------------------------------------------------------------- #


def test_tools_reach_the_driver(settings: Settings) -> None:
    """The field existing at all is the milestone.

    Asserted on what the driver was handed, not on a 200: the request
    used to succeed *and* drop the tools, which is exactly the failure
    mode -- a plain answer that looks like the model declining.
    """
    driver = _tool_driver()
    with TestClient(_app_with(settings, driver)) as c:
        assert c.post("/v1/chat/completions", json=_chat(tools=[WEATHER_TOOL])).status_code == 200

    sent = driver.calls[0]
    assert sent.tools is not None
    assert sent.tools[0].function.name == "get_weather"
    # The JSON Schema crosses untouched. We are a control plane, not a
    # schema validator, and a dialect we rewrote would fail somewhere
    # the caller cannot see.
    assert sent.tools[0].function.parameters == WEATHER_TOOL["function"]["parameters"]  # type: ignore[index]


def test_tool_choice_and_response_format_reach_the_driver(settings: Settings) -> None:
    driver = _tool_driver()
    with TestClient(_app_with(settings, driver)) as c:
        response = c.post(
            "/v1/chat/completions",
            json=_chat(
                tools=[WEATHER_TOOL],
                tool_choice={"type": "function", "function": {"name": "get_weather"}},
                response_format={"type": "json_object"},
            ),
        )
        assert response.status_code == 200

    sent = driver.calls[0]
    assert sent.toolChoice is not None
    assert sent.toolChoice.function.name == "get_weather"  # type: ignore[union-attr]
    assert sent.responseFormat is not None
    assert sent.responseFormat.type.value == "json_object"


def test_a_tool_result_survives_the_round_trip(settings: Settings) -> None:
    """The second half of an agent loop, and the half with no second chance.

    The harness replays the assistant turn that asked for the call plus a
    `tool` message carrying the result. If either is dropped or reshaped,
    the model cannot see that its call was answered -- and the visible
    symptom is a model that calls the same tool again, which reads as the
    model being stupid rather than as us losing a message.
    """
    driver = _tool_driver()
    with TestClient(_app_with(settings, driver)) as c:
        response = c.post(
            "/v1/chat/completions",
            json=_chat(
                messages=[
                    {"role": "user", "content": "weather in Oslo?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Oslo"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "content": '{"tempC": 4}', "tool_call_id": "call_abc"},
                ],
                tools=[WEATHER_TOOL],
            ),
        )
        assert response.status_code == 200

    sent = driver.calls[0]
    assistant, tool = sent.messages[1], sent.messages[2]
    assert assistant.role.value == "assistant"
    assert assistant.content is None
    assert assistant.toolCalls is not None
    assert assistant.toolCalls[0]["function"]["name"] == "get_weather"
    assert tool.role.value == "tool"
    assert tool.toolCallId == "call_abc"
    assert tool.content == '{"tempC": 4}'


def test_an_ordinary_request_is_unchanged(settings: Settings) -> None:
    """Every non-agent caller must be unaffected.

    A `tools: null` we invented would be a behaviour change for the
    overwhelmingly common request, and some backends treat the presence
    of the key as meaningful.
    """
    driver = _tool_driver()
    with TestClient(_app_with(settings, driver)) as c:
        assert c.post("/v1/chat/completions", json=_chat()).status_code == 200

    sent = driver.calls[0]
    assert sent.tools is None
    assert sent.toolChoice is None
    assert sent.responseFormat is None


# --------------------------------------------------------------------------- #
# up
# --------------------------------------------------------------------------- #


def test_a_tool_call_comes_back_in_openai_shape(settings: Settings) -> None:
    """`finish_reason` is the field an agent loop branches on.

    `stop` ends the turn; `tool_calls` dispatches and comes back. The
    driver used to flatten one into the other, so a correct model looked
    like a model that answered instead of using its tools.
    """
    driver = _tool_driver()
    driver.tool_calls = [_call()]
    with TestClient(_app_with(settings, driver)) as c:
        body = c.post("/v1/chat/completions", json=_chat(tools=[WEATHER_TOOL])).json()

    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call == {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"location": "Oslo"}'},
    }
    # A string, unparsed: a model can emit invalid JSON and OpenAI's
    # contract preserves what it said.
    assert isinstance(call["function"]["arguments"], str)


def test_streamed_tool_calls_arrive_as_deltas_with_an_index(settings: Settings) -> None:
    """Fragments, not calls -- and `index` is the load-bearing field.

    `arguments` arrives split at arbitrary points, so no single frame is
    parseable JSON, and a client reassembles by index. Losing the index
    would silently merge two calls into a third that was never made.
    """
    driver = _tool_driver()
    driver.tool_calls = [_call()]
    with (
        TestClient(_app_with(settings, driver)) as c,
        c.stream(
            "POST", "/v1/chat/completions", json=_chat(tools=[WEATHER_TOOL], stream=True)
        ) as response,
    ):
        assert response.status_code == 200
        frames = _frames(response.iter_lines())

    fragments = [
        d
        for f in frames
        for d in [f["choices"][0]["delta"]]
        if isinstance(d, dict) and d.get("tool_calls")
    ]
    assert len(fragments) >= 2, "the call must arrive in pieces, or this proves nothing"
    assert all(c["index"] == 0 for d in fragments for c in d["tool_calls"])

    # Reassembled the way a client does it.
    arguments = "".join(
        c["function"].get("arguments", "")
        for d in fragments
        for c in d["tool_calls"]
        if c.get("function")
    )
    assert json.loads(arguments) == {"location": "Oslo"}
    assert frames[-1]["choices"][0]["finish_reason"] == "tool_calls"


# --------------------------------------------------------------------------- #
# the refusal
# --------------------------------------------------------------------------- #


def test_a_backend_that_cannot_carry_tools_refuses(settings: Settings) -> None:
    """Never strip and answer anyway.

    This is the milestone's whole reason for existing stated as a test: a
    harness cannot distinguish a plain answer from "nobody offered the
    tools", so it re-prompts and loops -- the exact reported symptom this
    project set out to route around. Producing it ourselves would be
    worse than not shipping tools.
    """
    driver = FakeDriverClient(name="cli", base_url="http://cli", model_id=MODEL)
    with TestClient(_app_with(settings, driver)) as c:
        response = c.post("/v1/chat/completions", json=_chat(tools=[WEATHER_TOOL]))

    assert response.status_code == 400
    # OpenAI's envelope, so an SDK raises a real exception rather than
    # handing the caller a dict it cannot interpret.
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "tools"
    assert "tool_calling" in error["message"]
    # And nothing was sent: refusing after dispatch would bill the caller
    # for a request that was never going to work.
    assert driver.calls == []


def test_the_same_backend_still_serves_a_request_without_tools(settings: Settings) -> None:
    """The refusal is about the request, not the model.

    A CLI-subscription backend is a perfectly good chat backend and must
    stay one -- `claude_code_cli` is a contracted fallback tier.
    """
    driver = FakeDriverClient(name="cli", base_url="http://cli", model_id=MODEL)
    with TestClient(_app_with(settings, driver)) as c:
        assert c.post("/v1/chat/completions", json=_chat()).status_code == 200
    assert len(driver.calls) == 1


def test_models_reports_tool_calling_only_when_every_backend_can(settings: Settings) -> None:
    """The honest number, the same rule `context_length` follows.

    A request may land on any replica, so a model that is tool-capable on
    two of three is not tool-capable. Advertising otherwise would make
    the capability intermittent, which is the worst way to learn about
    it.
    """
    capable = _tool_driver("gpu-a")
    plain = FakeDriverClient(name="gpu-b", base_url="http://b", model_id=MODEL)

    with TestClient(_app_with(settings, capable)) as c:
        assert c.get("/v1/models").json()["data"][0]["x_eugene_plexus"]["tool_calling"] is True

    with TestClient(_app_with(settings, capable, plain)) as c:
        assert c.get("/v1/models").json()["data"][0]["x_eugene_plexus"]["tool_calling"] is False


def test_one_capable_backend_is_enough_to_accept_the_request(settings: Settings) -> None:
    """`/v1/models` says `all`, the refusal asks `any`, and that is not a
    contradiction: one advertises what a caller can rely on across every
    replica, the other decides whether to fail a request outright.
    Refusing while a capable backend exists would fail a request that
    would have worked."""
    capable = _tool_driver("gpu-a")
    plain = FakeDriverClient(name="gpu-b", base_url="http://b", model_id=MODEL)
    with TestClient(_app_with(settings, capable, plain)) as c:
        assert c.post("/v1/chat/completions", json=_chat(tools=[WEATHER_TOOL])).status_code == 200


def _frames(lines: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        out.append(json.loads(payload))
    return out
