"""The Responses door, slice 4 (2026-09-23).

**Every fixture here is shaped by a capture of a real Codex CLI 0.130**,
not by the documentation: `specs/docs/acceptance/responses-measurement.md`.
The request below keeps what arrived -- the `developer` message with two
parts, two user messages in a row, nine function tools with `strict:
false`, a `web_search` tool on every request, `store: false`,
`prompt_cache_key`, `client_metadata`, and no `max_output_tokens` at all.
A fixture that tidied any of them away would test a request nobody sends,
which is how `/v1/messages` came to refuse every real Claude Code request
while its tests stayed green.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import secrets
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from eugene_plexus_gateway import responses
from eugene_plexus_gateway._generated.driver_models import (
    Capabilities,
    FinishReason,
    FunctionCall,
    Problem,
    ToolCall,
    Usage,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.auth_state import AuthState
from eugene_plexus_gateway.driver_client import DriverError
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

MODEL = "qwen3-0.6b"


def codex_request(**overrides: Any) -> dict[str, Any]:
    """Codex CLI 0.130's first request, trimmed but not tidied."""
    request: dict[str, Any] = {
        "model": MODEL,
        "instructions": "You are a coding agent running in the Codex CLI.",
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [
                    {"type": "input_text", "text": "<permissions instructions>read-only"},
                    {"type": "input_text", "text": "<skills_instructions>none"},
                ],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "<environment_context>cwd"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "say ok"}],
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "shell",
                "description": "Runs a Powershell command (Windows) and returns its output.",
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "array", "items": {"type": "string"}}},
                    "required": ["command"],
                },
            },
            {
                "type": "function",
                "name": "view_image",
                "description": "View a local image.",
                "strict": False,
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
            {"type": "web_search", "external_web_access": False},
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "reasoning": None,
        "store": False,
        "stream": True,
        "include": [],
        "prompt_cache_key": "01a0cf68-2278-74d0-8dc5-5aefa90662ca",
        "client_metadata": {"x-codex-installation-id": "a40c9e44"},
    }
    request.update(overrides)
    return request


def turn_two(output: Any = "hi\r\n") -> dict[str, Any]:
    """The follow-up to a `function_call`, as Codex resends it: our items
    appended with their `id` and `status` dropped (captured)."""
    request = codex_request()
    request["input"] += [
        {
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "I should run echo."}],
            "encrypted_content": None,
        },
        {
            "type": "function_call",
            "name": "shell",
            "arguments": '{"command": ["cmd", "/c", "echo hi"]}',
            "call_id": "call_1",
        },
        {"type": "function_call_output", "call_id": "call_1", "output": output},
    ]
    return request


class VisionDriver(FakeDriverClient):
    def describe(self):  # type: ignore[no-untyped-def]
        info = super().describe()
        base = info.capabilities or Capabilities()
        info.capabilities = base.model_copy(update={"imageInput": True})
        return info


def png(colour: tuple[int, int, int] = (20, 180, 60)) -> str:
    data = io.BytesIO()
    Image.new("RGB", (16, 16), colour).save(data, format="PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


def serve(settings: Settings, *drivers: FakeDriverClient) -> TestClient:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*drivers)
    return TestClient(app)


def driver(**kwargs: Any) -> FakeDriverClient:
    fake = FakeDriverClient(name="d1", model_id=MODEL, supports_tools=True, **kwargs)
    fake.responses = ["ok"]
    return fake


def events(response: httpx.Response) -> list[tuple[str, dict[str, Any]]]:
    """Every SSE frame as (event name, data)."""
    out: list[tuple[str, dict[str, Any]]] = []
    name = None
    for line in response.iter_lines():
        if line.startswith("event: "):
            name = line[len("event: ") :]
        elif line.startswith("data: "):
            out.append((str(name), json.loads(line[len("data: ") :])))
            name = None
    return out


def stream(client: TestClient, body: dict[str, Any]) -> tuple[httpx.Response, list]:
    with client.stream("POST", "/v1/responses", json=body) as r:
        return r, events(r)


def sent(fake: FakeDriverClient) -> list[dict[str, Any]]:
    return fake.calls[-1].model_dump(mode="json", exclude_none=True)["messages"]


# --------------------------------------------------------------------------- #
# The door exists, and takes what Codex sends
# --------------------------------------------------------------------------- #


def test_codex_request_is_answered_rather_than_404(settings: Settings) -> None:
    """The reproduction: before slice 4 this path did not exist, and Codex
    0.130 has no other wire to try."""
    fake = driver()
    with serve(settings, fake) as client:
        r = client.post("/v1/responses", json=codex_request(stream=False))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["model"] == MODEL
    assert body["id"].startswith("resp_")
    [item] = body["output"]
    assert item["type"] == "message" and item["role"] == "assistant"
    assert item["content"][0] == {
        "type": "output_text",
        "text": "ok",
        "annotations": [],
        "logprobs": [],
    }
    assert body["store"] is False


def test_the_stream_is_the_responses_event_stream(settings: Settings) -> None:
    fake = driver()
    fake.responses = ["hello there"]
    with serve(settings, fake) as client:
        r, frames = stream(client, codex_request())
    assert r.status_code == 200
    names = [name for name, _ in frames]
    assert names[:2] == ["response.created", "response.in_progress"]
    assert names[2:4] == ["response.output_item.added", "response.content_part.added"]
    assert names.count("response.output_text.delta") == 2
    assert names[-4:] == [
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    # The two channels always agree, and every frame is numbered in order.
    assert all(name == data["type"] for name, data in frames)
    assert [data["sequence_number"] for _, data in frames] == list(range(len(frames)))
    deltas = "".join(d["delta"] for n, d in frames if n == "response.output_text.delta")
    assert deltas == "hello there"
    final = frames[-1][1]["response"]
    assert final["status"] == "completed"
    assert final["output"][0]["content"][0]["text"] == "hello there"


def test_the_stream_has_no_done_sentinel(settings: Settings) -> None:
    """`[DONE]` is the chat wire's framing; the Responses stream ends at its
    terminal event."""
    with (
        serve(settings, driver()) as client,
        client.stream("POST", "/v1/responses", json=codex_request()) as r,
    ):
        text = r.read().decode()
    assert "[DONE]" not in text


def test_web_search_is_removed_and_named_not_refused(settings: Settings) -> None:
    """On EVERY Codex request (captured). Refusing it would 400 the first
    request of every session -- `output_config` on `/v1/messages`, again."""
    fake = driver()
    with serve(settings, fake) as client:
        r = client.post("/v1/responses", json=codex_request(stream=False))
    assert r.status_code == 200, r.text
    names = [t.function.name for t in fake.calls[-1].tools or []]
    assert names == ["shell", "view_image"]
    ignored = r.headers["x-eugene-plexus-ignored-settings"]
    assert "tools.web_search" in ignored and "prompt_cache_key" in ignored


@pytest.mark.parametrize(
    "tool", [{"type": "file_search"}, {"type": "code_interpreter"}, {"type": "custom"}]
)
def test_any_other_server_side_tool_is_refused(settings: Settings, tool: dict) -> None:
    fake = driver()
    request = codex_request(stream=False)
    request["tools"].append(tool)
    with serve(settings, fake) as client:
        r = client.post("/v1/responses", json=request)
    assert r.status_code == 400
    assert tool["type"] in r.json()["error"]["message"]
    assert not fake.calls


def test_the_developer_and_two_user_messages_become_one_of_each(settings: Settings) -> None:
    """Codex sends two user messages in a row on every request (captured),
    and a template that requires alternating roles refuses them. They join,
    in order; `instructions` and the developer message are one system
    prompt."""
    fake = driver()
    with serve(settings, fake) as client:
        client.post("/v1/responses", json=codex_request(stream=False))
    messages = sent(fake)
    assert [m["role"] for m in messages] == ["system", "user"]
    system = messages[0]["content"]
    assert system.startswith("You are a coding agent")
    assert system.index("<permissions") < system.index("<skills")
    assert messages[1]["content"] == "<environment_context>cwd\n\nsay ok"


def test_a_developer_message_later_on_stays_where_it_was_put(settings: Settings) -> None:
    fake = driver()
    request = turn_two()
    request["input"].append(
        {"type": "message", "role": "developer", "content": "Approval policy changed."}
    )
    with serve(settings, fake) as client:
        client.post("/v1/responses", json={**request, "stream": False})
    roles = [m["role"] for m in sent(fake)]
    assert roles == ["system", "user", "assistant", "tool", "system"]


# --------------------------------------------------------------------------- #
# A tool loop, both directions
# --------------------------------------------------------------------------- #


def test_the_follow_up_carries_the_call_and_its_output(settings: Settings) -> None:
    fake = driver()
    with serve(settings, fake) as client:
        r = client.post("/v1/responses", json={**turn_two(), "stream": False})
    assert r.status_code == 200, r.text
    assistant, tool = sent(fake)[-2:]
    assert assistant["role"] == "assistant"
    assert assistant["toolCalls"][0]["id"] == "call_1"
    assert assistant["toolCalls"][0]["function"]["name"] == "shell"
    assert json.loads(assistant["toolCalls"][0]["function"]["arguments"]) == {
        "command": ["cmd", "/c", "echo hi"]
    }
    # The reasoning item before the call is that turn's reasoning.
    assert assistant["reasoning"] == "I should run echo."
    assert tool == {"role": "tool", "content": "hi\r\n", "toolCallId": "call_1"}


@pytest.mark.parametrize("streamed", [False, True])
def test_a_tool_call_comes_back_as_a_function_call_item(settings: Settings, streamed: bool) -> None:
    fake = driver()
    fake.tool_calls = [
        ToolCall(
            id="call_abc",
            type="function",
            function=FunctionCall(name="shell", arguments='{"command": ["echo", "hi"]}'),
        )
    ]
    with serve(settings, fake) as client:
        if streamed:
            r, frames = stream(client, codex_request())
            final = frames[-1][1]["response"]
            names = [n for n, _ in frames]
            added = next(d for n, d in frames if n == "response.output_item.added")
            assert added["item"]["type"] == "function_call"
            assert added["item"]["call_id"] == "call_abc" and added["item"]["name"] == "shell"
            # The fake splits the arguments so that no fragment parses alone.
            assert names.count("response.function_call_arguments.delta") == 2
            done = next(d for n, d in frames if n == "response.function_call_arguments.done")
            assert json.loads(done["arguments"]) == {"command": ["echo", "hi"]}
        else:
            r = client.post("/v1/responses", json=codex_request(stream=False))
            final = r.json()
    assert final["status"] == "completed"
    [item] = final["output"]
    assert item["type"] == "function_call" and item["call_id"] == "call_abc"
    assert json.loads(item["arguments"]) == {"command": ["echo", "hi"]}


# --------------------------------------------------------------------------- #
# Reasoning, out and back
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("streamed", [False, True])
def test_reasoning_is_a_reasoning_text_item_ahead_of_the_answer(
    settings: Settings, streamed: bool
) -> None:
    fake = driver()
    fake.reasoning = "The user wants ok."
    with serve(settings, fake) as client:
        if streamed:
            _, frames = stream(client, codex_request())
            final = frames[-1][1]["response"]
            text = "".join(d["delta"] for n, d in frames if n == "response.reasoning_text.delta")
            assert text == "The user wants ok."
        else:
            final = client.post("/v1/responses", json=codex_request(stream=False)).json()
    reasoning, message = final["output"]
    assert reasoning["type"] == "reasoning"
    assert reasoning["summary"] == []
    assert reasoning["content"] == [{"type": "reasoning_text", "text": "The user wants ok."}]
    # Not asked for: `include` was empty.
    assert "encrypted_content" not in reasoning
    assert message["type"] == "message"


def test_encrypted_content_carries_the_reasoning_when_asked(settings: Settings) -> None:
    fake = driver()
    fake.reasoning = "Think first."
    request = codex_request(stream=False, include=["reasoning.encrypted_content"])
    with serve(settings, fake) as client:
        reasoning = client.post("/v1/responses", json=request).json()["output"][0]
    assert reasoning["encrypted_content"].startswith("eugene-plexus-reasoning-v1:")
    assert responses.reasoning_text({"encrypted_content": reasoning["encrypted_content"]}) == (
        "Think first."
    )


def test_reasoning_sent_back_as_encrypted_content_alone_reaches_the_model(
    settings: Settings,
) -> None:
    fake = driver()
    request = turn_two()
    request["input"][3] = {
        "type": "reasoning",
        "summary": [],
        "content": None,
        "encrypted_content": responses.encode_signature("From the carrier."),
    }
    with serve(settings, fake) as client:
        client.post("/v1/responses", json={**request, "stream": False})
    assert sent(fake)[-2]["reasoning"] == "From the carrier."


def test_someone_elses_encrypted_reasoning_is_ignored(settings: Settings) -> None:
    fake = driver()
    request = turn_two()
    request["input"][3] = {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "OpenAI's summary"}],
        "encrypted_content": "gAAAAB-opaque-openai-blob",
    }
    with serve(settings, fake) as client:
        r = client.post("/v1/responses", json={**request, "stream": False})
    assert r.status_code == 200, r.text
    assert "reasoning" not in sent(fake)[-2]


def test_reasoning_effort_and_summary_are_named_not_honoured(settings: Settings) -> None:
    request = codex_request(
        stream=False,
        reasoning={"effort": "high", "summary": "auto"},
        include=["reasoning.encrypted_content"],
    )
    with serve(settings, driver()) as client:
        r = client.post("/v1/responses", json=request)
    assert r.status_code == 200, r.text
    ignored = r.headers["x-eugene-plexus-ignored-settings"]
    assert "reasoning.effort" in ignored and "reasoning.summary" in ignored


# --------------------------------------------------------------------------- #
# Images: `detail: "high"` on every one, and view_image's list output
# --------------------------------------------------------------------------- #


def test_an_attached_image_with_detail_high_is_carried(settings: Settings) -> None:
    """`codex exec -i` sends the picture between two marker texts, with
    `detail: "high"` (captured). The chat door refuses anything but `auto`;
    here that would 400 every image Codex sends."""
    fake = VisionDriver(name="v", model_id=MODEL, supports_tools=True)
    fake.responses = ["green"]
    request = codex_request(stream=False)
    request["input"][-1]["content"] = [
        {"type": "input_text", "text": "<image name=[Image #1]>"},
        {"type": "input_image", "image_url": png(), "detail": "high"},
        {"type": "input_text", "text": "</image>"},
        {"type": "input_text", "text": "Name this colour."},
    ]
    with serve(settings, fake) as client:
        r = client.post("/v1/responses", json=request)
    assert r.status_code == 200, r.text
    assert "input_image.detail" in r.headers["x-eugene-plexus-ignored-settings"]
    before, picture, after = sent(fake)[-1]["content"]
    # The picture stays between Codex's own markers; adjacent text joins.
    assert before["type"] == "text" and before["text"].endswith("<image name=[Image #1]>")
    assert picture == {"type": "image_url", "image_url": {"url": png()}}
    assert after == {"type": "text", "text": "</image>\n\nName this colour."}


def test_view_images_result_moves_to_the_next_user_message(settings: Settings) -> None:
    """`view_image` answers with a LIST holding an `input_image` (captured).
    A `tool` message carries text only, so the picture moves, labelled."""
    fake = VisionDriver(name="v", model_id=MODEL, supports_tools=True)
    fake.responses = ["green"]
    request = turn_two(output=[{"type": "input_image", "image_url": png(), "detail": "high"}])
    with serve(settings, fake) as client:
        r = client.post("/v1/responses", json={**request, "stream": False})
    assert r.status_code == 200, r.text
    tool, after = sent(fake)[-2:]
    assert tool["role"] == "tool" and tool["toolCallId"] == "call_1"
    assert "next user message" in tool["content"]
    assert after["role"] == "user"
    assert "call_1" in after["content"][0]["text"]
    assert after["content"][1]["image_url"]["url"] == png()


def test_a_text_only_backend_is_refused_rather_than_answering_blind(settings: Settings) -> None:
    fake = driver()
    request = codex_request(stream=False)
    request["input"][-1]["content"] = [{"type": "input_image", "image_url": png()}]
    with serve(settings, fake) as client:
        r = client.post("/v1/responses", json=request)
    assert r.status_code == 400
    assert "image input" in r.json()["error"]["message"]
    assert not fake.calls


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_image", "image_url": "https://example.com/cat.png"},
        {"type": "input_image", "file_id": "file-abc"},
        {"type": "input_file", "file_data": "JVBERi0="},
    ],
)
def test_what_cannot_be_carried_is_refused(settings: Settings, part: dict) -> None:
    fake = VisionDriver(name="v", model_id=MODEL, supports_tools=True)
    request = codex_request(stream=False)
    request["input"][-1]["content"] = [part]
    with serve(settings, fake) as client:
        r = client.post("/v1/responses", json=request)
    assert r.status_code == 400, r.text
    assert "input[2].content[0]" in r.json()["error"]["message"]
    assert not fake.calls


def test_the_image_limit_is_the_gateways_setting(settings: Settings) -> None:
    fake = VisionDriver(name="v", model_id=MODEL, supports_tools=True)
    request = codex_request(stream=False)
    request["input"][-1]["content"] = [
        {"type": "input_image", "image_url": png()} for _ in range(3)
    ]
    with serve(settings, fake) as client:
        client.patch("/v1/config", json={"maxImagesPerRequest": 2})
        r = client.post("/v1/responses", json=request)
    assert r.status_code == 400
    assert "maxImagesPerRequest" in r.json()["error"]["message"]


# --------------------------------------------------------------------------- #
# The output cap: none from the install default
# --------------------------------------------------------------------------- #


def test_no_max_output_tokens_means_no_install_cap(settings: Settings) -> None:
    """Codex sends none, and regenerates an answer that ends `incomplete`
    five times before failing (measured). So the install's
    `defaultMaxTokens` is not applied here -- while the chat door, given the
    same request, still gets it. The pair is what tells the fix from an
    accident."""
    responses_fake = driver()
    chat_fake = FakeDriverClient(name="d2", model_id="chat-model")
    chat_fake.responses = ["ok"]
    with serve(settings, responses_fake, chat_fake) as client:
        client.post("/v1/responses", json=codex_request(stream=False))
        client.post(
            "/v1/chat/completions",
            json={"model": "chat-model", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert responses_fake.calls[-1].maxTokens is None
    assert chat_fake.calls[-1].maxTokens == 2048


def test_a_caller_cap_is_carried(settings: Settings) -> None:
    fake = driver()
    with serve(settings, fake) as client:
        client.post("/v1/responses", json=codex_request(stream=False, max_output_tokens=77))
    assert fake.calls[-1].maxTokens == 77


@pytest.mark.parametrize("streamed", [False, True])
def test_an_answer_cut_at_the_cap_is_incomplete(settings: Settings, streamed: bool) -> None:
    fake = driver()
    fake.finish_reason = FinishReason.length
    with serve(settings, fake) as client:
        if streamed:
            _, frames = stream(client, codex_request(max_output_tokens=5))
            name, data = frames[-1]
            assert name == "response.incomplete"
            final = data["response"]
        else:
            final = client.post(
                "/v1/responses", json=codex_request(stream=False, max_output_tokens=5)
            ).json()
    assert final["status"] == "incomplete"
    assert final["incomplete_details"] == {"reason": "max_output_tokens"}
    assert final["output"][0]["status"] == "incomplete"


# --------------------------------------------------------------------------- #
# Refusals, and the statuses Codex acts on
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("previous_response_id", "resp_abc"),
        ("conversation", "conv_abc"),
        ("prompt", {"id": "pmpt_abc"}),
        ("background", True),
    ],
)
def test_stored_state_is_refused_by_name(settings: Settings, field: str, value: Any) -> None:
    fake = driver()
    with serve(settings, fake) as client:
        r = client.post("/v1/responses", json=codex_request(stream=False, **{field: value}))
    assert r.status_code == 400
    error = r.json()["error"]
    assert error["param"] == field and field in error["message"]
    assert not fake.calls


def test_an_item_reference_is_refused(settings: Settings) -> None:
    fake = driver()
    request = codex_request(stream=False)
    request["input"].append({"type": "item_reference", "id": "msg_abc"})
    with serve(settings, fake) as client:
        r = client.post("/v1/responses", json=request)
    assert r.status_code == 400
    assert "item_reference" in r.json()["error"]["message"]


def test_an_unknown_top_level_field_is_refused_and_the_opaque_ones_are_not(
    settings: Settings,
) -> None:
    fake = driver()
    with serve(settings, fake) as client:
        refused = client.post("/v1/responses", json=codex_request(stream=False, colour="teal"))
        served = client.post(
            "/v1/responses",
            json=codex_request(
                stream=False,
                metadata={"a": "b"},
                user="u",
                safety_identifier="s",
                service_tier="auto",
                max_tool_calls=3,
                truncation="disabled",
            ),
        )
    assert refused.status_code == 400 and refused.json()["error"]["param"] == "colour"
    assert served.status_code == 200, served.text


def test_store_true_is_accepted_and_named_but_nothing_is_kept(settings: Settings) -> None:
    with serve(settings, driver()) as client:
        r = client.post("/v1/responses", json=codex_request(stream=False, store=True))
        stored = client.get(f"/v1/responses/{r.json()['id']}")
    assert r.status_code == 200
    assert r.json()["store"] is False
    assert "store" in r.headers["x-eugene-plexus-ignored-settings"].split(", ")
    assert stored.status_code == 404
    assert "no response store" in stored.json()["error"]["message"]


def test_a_model_nothing_serves_is_400_here_and_404_on_the_chat_door(settings: Settings) -> None:
    """Codex retries a 404 five times before showing it, and a 400 never
    (record §3). The chat door keeps its 404: the pair stops anyone
    'harmonising' the two."""
    with serve(settings, driver()) as client:
        here = client.post("/v1/responses", json=codex_request(stream=False, model="nope"))
        chat = client.post(
            "/v1/chat/completions",
            json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert here.status_code == 400
    assert here.json()["error"]["code"] == "model_not_found"
    assert MODEL in here.json()["error"]["message"]
    assert chat.status_code == 404


def test_a_rejected_key_is_401_in_openais_envelope(settings: Settings) -> None:
    signing_key = secrets.token_bytes(32)
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(driver())
    app.state.auth_state = AuthState(signing_key=signing_key, service_token=None, master_key=None)
    now = int(time.time())
    good = jwt.encode(
        {"sub": "troy", "aud": "operator", "iat": now, "exp": now + 60}, signing_key, "HS256"
    )
    with TestClient(app) as client:
        missing = client.post("/v1/responses", json=codex_request(stream=False))
        wrong = client.post(
            "/v1/responses",
            json=codex_request(stream=False),
            headers={"Authorization": "Bearer not-a-token"},
        )
        right = client.post(
            "/v1/responses",
            json=codex_request(stream=False),
            headers={"Authorization": f"Bearer {good}"},
        )
    for r in (missing, wrong):
        assert r.status_code == 401
        assert r.json()["error"]["code"] == "invalid_api_key"
        assert r.json()["error"]["message"]
    assert right.status_code == 200, right.text


# --------------------------------------------------------------------------- #
# Failures once the stream is open: `response.failed`, with the right code
# --------------------------------------------------------------------------- #


def _driver_error(status: int, detail: str) -> DriverError:
    return DriverError(
        driver_name="d1",
        driver_url="http://fake-driver",
        status_code=status,
        problem=Problem(type="about:blank", title="failed", status=status, detail=detail),
        raw_body="",
    )


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (
            _driver_error(
                400,
                'openai_compat_http returned 400: {"error":{"type":"exceed_context_size_error",'
                '"message":"the request exceeds the available context size"}}',
            ),
            "context_length_exceeded",
        ),
        (_driver_error(400, "the backend rejected the request"), "invalid_prompt"),
        (httpx.ReadTimeout("slow"), "invalid_prompt"),
        (httpx.ConnectError("gone"), "server_error"),
        (_driver_error(503, "still loading"), "server_error"),
    ],
)
def test_a_failure_is_response_failed_with_a_code_codex_acts_on(
    settings: Settings, error: Exception, code: str
) -> None:
    """Codex stops on `context_length_exceeded` and `invalid_prompt` and
    retries `server_error` five times (record §4). A deadline that fired is
    `invalid_prompt`: the next attempt would compute the same prompt for as
    long, which is R2.5's finding in this client's vocabulary."""
    fake = driver()
    fake.stream_error_after = 0
    fake.stream_error = error
    with serve(settings, fake) as client:
        r, frames = stream(client, codex_request())
    assert r.status_code == 200
    names = [n for n, _ in frames]
    assert "error" not in names, "a bare error event loses the message in Codex"
    name, data = frames[-1]
    assert name == "response.failed"
    assert data["response"]["status"] == "failed"
    assert data["response"]["error"]["code"] == code
    assert data["response"]["error"]["message"]


def test_a_stream_that_breaks_mid_answer_keeps_what_arrived(settings: Settings) -> None:
    fake = driver()
    fake.responses = ["one two three"]
    fake.stream_error_after = 1
    fake.stream_error = httpx.RemoteProtocolError("peer closed")
    with serve(settings, fake) as client:
        _, frames = stream(client, codex_request())
    name, data = frames[-1]
    assert name == "response.failed"
    [item] = data["response"]["output"]
    assert item["status"] == "incomplete"
    assert item["content"][0]["text"] == "one "


# --------------------------------------------------------------------------- #
# The keepalive: the stream opens early and stays open
# --------------------------------------------------------------------------- #


class SlowStartDriver(FakeDriverClient):
    """A backend that takes a while before its first token -- the prefill a
    processor spends on Codex's ~10k-token first prompt."""

    delay = 0.35

    async def stream(self, request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(self.delay)
        async for event in super().stream(request):
            yield event


def test_the_stream_says_it_is_working_while_the_backend_is_silent(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex drops a stream that is silent for five minutes -- before the
    first event too -- and an SSE comment does not reset that timer while a
    `response.in_progress` does (record §5). So until output arrives the
    stream repeats `response.in_progress`."""
    monkeypatch.setattr(responses, "KEEPALIVE_SECONDS", 0.05)
    fake = SlowStartDriver(name="d1", model_id=MODEL, supports_tools=True)
    fake.responses = ["ok"]
    with serve(settings, fake) as client:
        _, frames = stream(client, codex_request())
    names = [n for n, _ in frames]
    first_output = names.index("response.output_item.added")
    assert names[0] == "response.created"
    assert names[:first_output].count("response.in_progress") >= 3
    assert names[-1] == "response.completed"


def test_the_stream_opens_before_the_first_token(settings: Settings) -> None:
    """`response.created` goes out as soon as a backend is chosen, and names
    the model that was asked for."""
    fake = SlowStartDriver(name="d1", model_id=MODEL, supports_tools=True)
    fake.delay = 0.0
    fake.responses = ["ok"]
    with serve(settings, fake) as client:
        _, frames = stream(client, codex_request())
    assert frames[0][1]["response"]["model"] == MODEL
    assert frames[0][1]["response"]["status"] == "in_progress"


async def test_keepalive_closes_the_backend_stream_when_the_client_leaves() -> None:
    """The consumer going away cancels the pump and closes the backend
    stream, so an engine is not left computing for nobody."""
    closed = asyncio.Event()

    async def backend() -> AsyncGenerator[int, None]:
        try:
            yield 1
            await asyncio.sleep(3600)
            yield 2
        finally:
            closed.set()

    # Well above Windows' 15.6 ms timer grid, or the first wait can expire
    # before the pump has run at all.
    iterator = responses.with_keepalive(backend(), 0.2)
    assert await iterator.__anext__() == 1
    assert await iterator.__anext__() is None  # the quiet period
    await iterator.aclose()
    await asyncio.wait_for(closed.wait(), 1)


async def test_keepalive_hands_a_backend_failure_to_the_consumer() -> None:
    async def backend() -> AsyncGenerator[int, None]:
        yield 1
        raise httpx.ConnectError("gone")

    seen = []
    with pytest.raises(httpx.ConnectError):
        async for value in responses.with_keepalive(backend(), 1):
            seen.append(value)
    assert seen == [1]


# --------------------------------------------------------------------------- #
# The shared path: admission, metrics, CORS, the body limit
# --------------------------------------------------------------------------- #


def test_the_door_is_under_client_admission() -> None:
    """A door missing from CLIENT_ADMISSION_PATHS has no client admission at
    all -- how /v1/systemone shipped for a few hours."""
    from eugene_plexus_gateway.admission import CLIENT_ADMISSION_PATHS

    assert "/v1/responses" in CLIENT_ADMISSION_PATHS


def test_a_response_is_recorded(settings: Settings) -> None:
    fake = driver()
    fake.usage = Usage(promptTokens=10, completionTokens=4, totalTokens=14)
    with serve(settings, fake) as client:
        _, frames = stream(client, codex_request())
        for _ in range(100):
            groups = client.get("/v1/metrics").json()["groups"]
            if groups:
                break
            time.sleep(0.05)
    assert groups and groups[0]["requests"] == 1
    usage = frames[-1][1]["response"]["usage"]
    assert usage == {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}


def test_usage_details_appear_only_when_reported(settings: Settings) -> None:
    fake = driver()
    fake.usage = Usage(
        promptTokens=10,
        completionTokens=4,
        totalTokens=14,
        cachedPromptTokens=6,
        reasoningTokens=2,
    )
    with serve(settings, fake) as client:
        usage = client.post("/v1/responses", json=codex_request(stream=False)).json()["usage"]
    # Unlike Anthropic's, OpenAI's input_tokens includes the cached part.
    assert usage["input_tokens"] == 10
    assert usage["input_tokens_details"] == {"cached_tokens": 6}
    assert usage["output_tokens_details"] == {"reasoning_tokens": 2}


def test_a_browser_preflight_is_answered(settings: Settings) -> None:
    with serve(settings, driver()) as client:
        r = client.options(
            "/v1/responses",
            headers={
                "Origin": "http://example.test",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization, content-type",
            },
        )
    assert r.status_code in (200, 204)
    assert r.headers["access-control-allow-origin"]


def test_the_body_limit_covers_the_door(settings: Settings) -> None:
    with serve(settings, driver()) as client:
        r = client.post("/v1/responses", content=b"{}", headers={"content-length": "999999999"})
    assert r.status_code == 413


def test_a_deadline_that_fires_mid_stream_is_response_failed(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The admission middleware owns the request deadline and ends a stream
    it did not write. On this door that end is `response.failed` with
    `invalid_prompt` -- not the chat door's bare `data:` error, which Codex
    reports without our message, and not a code Codex retries: the next
    attempt would compute the same prompt for as long."""
    monkeypatch.setattr(responses, "KEEPALIVE_SECONDS", 0.05)
    fake = SlowStartDriver(name="d1", model_id=MODEL, supports_tools=True)
    fake.delay = 2.0
    fake.responses = ["too late"]
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake)
    with TestClient(app) as client:
        store = app.state.config_store  # made at startup
        original = store.get
        monkeypatch.setattr(
            store, "get", lambda key: 0.4 if key == "requestTimeoutSeconds" else original(key)
        )
        _, frames = stream(client, codex_request())
    names = [n for n, _ in frames]
    assert names[0] == "response.created"
    assert "response.output_item.added" not in names
    name, data = frames[-1]
    assert name == "response.failed"
    assert data["response"]["error"]["code"] == "invalid_prompt"
    assert "deadline" in data["response"]["error"]["message"]
