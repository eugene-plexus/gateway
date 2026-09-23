"""The Anthropic Messages door, R4.

**Every fixture in this file is shaped by a capture, not by reading the
docs.** `specs/docs/acceptance/anthropic-messages-measurement.md` records
nine runs of a real Claude Code 2.1.207 against a throwaway listener, and
the request shape below -- the `?beta=true` query, the billing header
smuggled in as `system[0]`, `thinking` on every request, `cache_control`
on a `tool_result`, `x-api-key` with no `Authorization` beside it -- is
what actually arrived. That matters because the failure this endpoint is
most likely to have is *passing its own tests and 400ing the only client
it exists for*, which is exactly what the pre-measurement refusal list
would have done.

Wire fidelity is the point, as in `test_tool_calling.py`: the assertions
are on Anthropic's names, Anthropic's error envelope and Anthropic's
event stream, because an unmodified SDK is the audience.
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import Iterator
from typing import Any

import httpx
import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import FunctionCall, ToolCall
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.auth_state import AuthState
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

MODEL = "Qwen3-30B-A3B-Q4_K_M"
_JWT_ALG = "HS256"


def _app_with(settings: Settings, *fakes: FakeDriverClient) -> FastAPI:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*fakes)
    return app


def _client(app: FastAPI) -> TestClient:
    return TestClient(app)


def body(**overrides: Any) -> dict[str, Any]:
    """A Claude Code request, trimmed but not tidied.

    Kept deliberately faithful: `system[0]` really is a billing header
    rather than a prompt, `thinking` really is present on every request,
    and `context_management`/`metadata` really do arrive with no
    equivalent on our side. A fixture that cleaned these up would test a
    request no client sends.
    """
    request: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": 32000,
        "stream": False,
        "thinking": {"type": "adaptive", "display": "omitted"},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        # On every request since agent-sdk 0.3.280 (captured 2026-09-23).
        # Absent from this fixture until then, which is exactly how A2's
        # unknown-field refusal came to 400 every real Claude Code request
        # while every test here stayed green.
        "output_config": {"effort": "high"},
        "metadata": {"user_id": '{"device_id":"abc","account_uuid":"","session_id":"def"}'},
        "system": [
            {
                "type": "text",
                "text": "x-anthropic-billing-header: cc_version=2.1.207; cc_entrypoint=cli;",
            },
            {
                "type": "text",
                "text": "You are a Claude agent.",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<system-reminder>context</system-reminder>"},
                    {
                        "type": "text",
                        "text": "reply with the single word ok",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            }
        ],
    }
    request.update(overrides)
    return request


# --------------------------------------------------------------------------- #
# The door exists at all
# --------------------------------------------------------------------------- #


def test_the_door_answers_a_real_claude_code_request(settings: Settings) -> None:
    """The reproduction. Before R4 this path did not exist and every
    request from the most capable client in the field got a 404."""
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    fake.responses = ["ok"]
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages?beta=true", json=body())

    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["type"] == "message"
    assert payload["role"] == "assistant"
    assert payload["content"] == [{"type": "text", "text": "ok"}]
    assert payload["stop_reason"] == "end_turn"
    assert payload["model"] == MODEL
    assert payload["usage"]["input_tokens"] >= 0


def test_the_query_parameter_claude_code_always_sends_is_accepted(settings: Settings) -> None:
    """`?beta=true` rides on every real request. A route that 404s or
    422s on it fails 100% of real traffic and 0% of invented traffic."""
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    fake.responses = ["ok"]
    with _client(_app_with(settings, fake)) as client:
        with_param = client.post("/v1/messages?beta=true", json=body())
        without = client.post("/v1/messages", json=body())
    assert with_param.status_code == 200
    assert without.status_code == 200


def test_an_unknown_anthropic_beta_header_is_tolerated(settings: Settings) -> None:
    """The list changes with the client's own version and login method;
    validating it would refuse tomorrow's Claude Code."""
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    fake.responses = ["ok"]
    with _client(_app_with(settings, fake)) as client:
        r = client.post(
            "/v1/messages?beta=true",
            json=body(),
            headers={
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "claude-code-20250219,something-we-have-never-heard-of",
            },
        )
    assert r.status_code == 200, r.text


# --------------------------------------------------------------------------- #
# What is dropped -- the half the measurement rewrote
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "thinking",
    [
        {"budget_tokens": 31999, "type": "enabled", "display": "omitted"},
        {"type": "adaptive", "display": "omitted"},
        None,
    ],
    ids=["known-claude-id", "local-model-id", "MAX_THINKING_TOKENS=0"],
)
def test_thinking_is_dropped_in_all_three_observed_shapes(
    settings: Settings, thinking: Any
) -> None:
    """R4 as designed refused `thinking` with a 400 naming the field.

    All three shapes are real: a known Claude id, an arbitrary local id
    -- which is the entire point of this door -- and `null`, which is
    what a caller who set `MAX_THINKING_TOKENS=0` sends **with the key
    still present**. So a refusal fails everybody and a bare presence
    check fails the one caller genuinely asking for no thinking.
    """
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    fake.responses = ["ok"]
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages?beta=true", json=body(thinking=thinking))
    assert r.status_code == 200, r.text


def test_cache_control_is_dropped_wherever_it_appears(settings: Settings) -> None:
    """Including on a `tool_result`, which the scope did not name."""
    fake = FakeDriverClient(name="d1", model_id=MODEL, supports_tools=True)
    fake.responses = ["ok"]
    request = body(
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Glob", "input": {"p": "*.py"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "alpha.py",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
        ]
    )
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 200, r.text


def test_metadata_and_context_management_are_dropped(settings: Settings) -> None:
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    fake.responses = ["ok"]
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=body())
    assert r.status_code == 200, r.text
    # And none of it reached the backend, which is the half a status
    # code cannot show.
    sent = fake.calls[0].model_dump()
    assert "thinking" not in sent
    assert "metadata" not in sent
    assert "context_management" not in sent


# --------------------------------------------------------------------------- #
# What is refused, with the field named
# --------------------------------------------------------------------------- #


def test_an_image_that_will_not_decode_is_refused_naming_the_field(settings: Settings) -> None:
    """Amended 2026-09-23. This asserted that EVERY image block was
    refused; images are carried now (`test_anthropic_images.py`), and
    the payload here, four characters of a PNG header, is refused for
    being no picture at all -- which the old assertion, `"image" in
    message`, would have accepted as the same thing. It names the block
    now, in the caller's coordinates, so the two cannot be confused."""
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    request = body(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is in this screenshot"},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "iVBO"},
                    },
                ],
            }
        ]
    )
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 400, r.text
    payload = r.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] == "invalid_request_error"
    assert "messages.0.content.1.source" in payload["error"]["message"]
    assert "invalid" in payload["error"]["message"]
    assert not fake.calls, "a refused request must not reach a backend"


def test_a_document_block_is_refused(settings: Settings) -> None:
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    request = body(
        messages=[
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "text", "data": "x"}}],
            }
        ]
    )
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 400
    assert "document" in r.json()["error"]["message"]


def test_an_image_inside_a_tool_result_is_checked_too(settings: Settings) -> None:
    """The nested case, amended 2026-09-23 like the one above: a check
    that only looked at top-level blocks would pass its own test and then
    carry a picture it never validated. This one has no `media_type` and
    is refused for that, at its nested coordinate."""
    fake = FakeDriverClient(name="d1", model_id=MODEL, supports_tools=True)
    request = body(
        messages=[
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Shot", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "data": "iVBO"}}
                        ],
                    }
                ],
            },
        ]
    )
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 400
    assert "messages.2.content.0.content.0.source" in r.json()["error"]["message"]
    assert "media_type" in r.json()["error"]["message"]
    assert not fake.calls


def test_a_server_side_tool_is_refused(settings: Settings) -> None:
    """We have no web search to run and no sandbox to run code in."""
    fake = FakeDriverClient(name="d1", model_id=MODEL, supports_tools=True)
    request = body(tools=[{"type": "web_search_20250305", "name": "web_search"}])
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 400
    assert "web_search" in r.json()["error"]["message"]


def test_mcp_servers_is_refused(settings: Settings) -> None:
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    request = body(mcp_servers=[{"type": "url", "url": "https://example.test/mcp"}])
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 400
    assert "mcp_servers" in r.json()["error"]["message"]


def test_a_fifth_stop_sequence_is_refused_rather_than_truncated(settings: Settings) -> None:
    """Dropping one changes where the answer ends."""
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    request = body(stop_sequences=["a", "b", "c", "d", "e"])
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 400
    assert "stop_sequences" in r.json()["error"]["message"]


def test_four_stop_sequences_are_carried(settings: Settings) -> None:
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    fake.responses = ["ok"]
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=body(stop_sequences=["a", "b", "c", "d"]))
    assert r.status_code == 200, r.text
    assert fake.calls[0].stop == ["a", "b", "c", "d"]


def test_a_missing_max_tokens_is_400_naming_the_field(settings: Settings) -> None:
    """A 400 in OUR envelope, not FastAPI's 422 `detail` shape, which an
    Anthropic SDK cannot read."""
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    request = body()
    del request["max_tokens"]
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 400, r.text
    payload = r.json()
    assert payload["type"] == "error"
    assert "max_tokens" in payload["error"]["message"]
    assert "detail" not in payload


def test_an_unknown_model_is_400_not_404(settings: Settings) -> None:
    """Measured: Claude Code discards a 404's body and shows a generic
    message blaming the model, so a 404 here throws away the
    explanation -- including the sealed-control-root diagnosis."""
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=body(model="nobody-serves-this"))
    assert r.status_code == 400, r.text
    payload = r.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert "nobody-serves-this" in payload["error"]["message"]
    # The hint that only a surviving body can deliver.
    assert MODEL in payload["error"]["message"]


def test_a_tool_request_against_a_toolless_backend_is_refused(settings: Settings) -> None:
    """The OpenAI door's rule, inherited rather than re-implemented: a
    harness cannot tell "the model chose not to" from "nobody offered
    it the tools"."""
    fake = FakeDriverClient(name="d1", model_id=MODEL, supports_tools=False)
    request = body(tools=[{"name": "Glob", "input_schema": {"type": "object"}}])
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 400
    assert "tool" in r.json()["error"]["message"].lower()


# --------------------------------------------------------------------------- #
# Translation
# --------------------------------------------------------------------------- #


def test_system_blocks_are_concatenated_in_order(settings: Settings) -> None:
    """Including the billing header, which is `system[0]` on every real
    request and is not a prompt at all. Any code that treats `system[0]`
    as the instruction is wrong about the commonest client."""
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    fake.responses = ["ok"]
    with _client(_app_with(settings, fake)) as client:
        client.post("/v1/messages", json=body())
    sent = fake.calls[0]
    system = [m for m in sent.messages if m.role.value == "system"]
    assert len(system) == 1
    assert "x-anthropic-billing-header" in system[0].content
    assert "You are a Claude agent." in system[0].content
    assert system[0].content.index("x-anthropic-billing") < system[0].content.index("You are")


def test_a_plain_string_system_prompt_also_works(settings: Settings) -> None:
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    fake.responses = ["ok"]
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=body(system="be terse"))
    assert r.status_code == 200, r.text
    system = [m for m in fake.calls[0].messages if m.role.value == "system"]
    assert system[0].content == "be terse"


def test_several_text_blocks_in_one_user_message_are_joined(settings: Settings) -> None:
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    fake.responses = ["ok"]
    with _client(_app_with(settings, fake)) as client:
        client.post("/v1/messages", json=body())
    user = [m for m in fake.calls[0].messages if m.role.value == "user"]
    assert len(user) == 1
    assert "<system-reminder>context</system-reminder>" in user[0].content
    assert "reply with the single word ok" in user[0].content


def test_a_tool_result_in_a_user_message_becomes_a_tool_message(settings: Settings) -> None:
    """Anthropic has no `tool` role: a result is a block inside a USER
    message, several to a message. Getting this wrong turns the tool's
    output into the human speaking, which is step 6's exact defect one
    protocol over."""
    fake = FakeDriverClient(name="d1", model_id=MODEL, supports_tools=True)
    fake.responses = ["done"]
    request = body(
        messages=[
            {"role": "user", "content": "list the python files"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Looking."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Glob",
                        "input": {"pattern": "*.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "alpha.py"},
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "beta.py"},
                ],
            },
        ]
    )
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 200, r.text

    roles = [m.role.value for m in fake.calls[0].messages]
    # The fixture carries a system prompt, so it leads.
    assert roles == ["system", "user", "assistant", "tool", "tool"]
    tools = [m for m in fake.calls[0].messages if m.role.value == "tool"]
    assert tools[0].toolCallId == "toolu_1"
    assert tools[0].content == "alpha.py"
    assert tools[1].toolCallId == "toolu_2"


def test_an_assistant_tool_use_block_becomes_tool_calls(settings: Settings) -> None:
    """The inbound half the scope did not name. Without it a tool loop
    cannot continue past its first turn, because the assistant turn we
    are handed back carries `tool_use` blocks rather than text."""
    fake = FakeDriverClient(name="d1", model_id=MODEL, supports_tools=True)
    fake.responses = ["done"]
    request = body(
        messages=[
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Looking."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Glob",
                        "input": {"pattern": "*.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "alpha.py"}
                ],
            },
        ]
    )
    with _client(_app_with(settings, fake)) as client:
        client.post("/v1/messages", json=request)

    assistant = next(m for m in fake.calls[0].messages if m.role.value == "assistant")
    assert assistant.content == "Looking."
    assert assistant.toolCalls is not None
    # Plain dicts, not models: `common.yaml` types `toolCalls` loosely on
    # purpose -- the shared schema declines to be a third definition of
    # OpenAI's object -- so the driver re-shapes them for its backend.
    call = assistant.toolCalls[0]
    assert call["id"] == "toolu_1"
    assert call["function"]["name"] == "Glob"
    assert json.loads(call["function"]["arguments"]) == {"pattern": "*.py"}


def test_a_failed_tool_result_says_so(settings: Settings) -> None:
    """`is_error` carried rather than dropped: a harness that cannot see
    its own tool failed will call it again."""
    fake = FakeDriverClient(name="d1", model_id=MODEL, supports_tools=True)
    fake.responses = ["done"]
    request = body(
        messages=[
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "Glob", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "no such directory",
                        "is_error": True,
                    }
                ],
            },
        ]
    )
    with _client(_app_with(settings, fake)) as client:
        client.post("/v1/messages", json=request)
    tool = next(m for m in fake.calls[0].messages if m.role.value == "tool")
    assert "no such directory" in tool.content
    assert "error" in tool.content.lower()


def test_tools_are_translated_into_the_openai_function_shape(settings: Settings) -> None:
    fake = FakeDriverClient(name="d1", model_id=MODEL, supports_tools=True)
    fake.responses = ["ok"]
    schema = {"type": "object", "properties": {"pattern": {"type": "string"}}}
    request = body(
        tools=[{"name": "Glob", "description": "Find files.", "input_schema": schema}],
        tool_choice={"type": "any"},
    )
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=request)
    assert r.status_code == 200, r.text
    sent = fake.calls[0]
    assert sent.tools is not None
    assert sent.tools[0].function.name == "Glob"
    assert sent.tools[0].function.description == "Find files."
    assert sent.tools[0].function.parameters == schema
    # `any` is Anthropic for "you must call one".
    assert sent.toolChoice == "required"


def test_a_tool_call_comes_back_as_a_tool_use_block(settings: Settings) -> None:
    fake = FakeDriverClient(name="d1", model_id=MODEL, supports_tools=True)
    fake.tool_calls = [
        ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="Glob", arguments='{"pattern": "*.py"}'),
        )
    ]
    request = body(tools=[{"name": "Glob", "input_schema": {"type": "object"}}])
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=request)
    payload = r.json()
    assert payload["stop_reason"] == "tool_use"
    block = next(b for b in payload["content"] if b["type"] == "tool_use")
    assert block["id"] == "call_1"
    assert block["name"] == "Glob"
    assert block["input"] == {"pattern": "*.py"}


def test_the_envelope_rides_on_headers_and_not_in_the_body(settings: Settings) -> None:
    """Their wire is typed events and a strict client is who this door
    is for, so an unknown top-level key is a risk with no upside."""
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    fake.responses = ["ok"]
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=body())
    assert "x_eugene_plexus" not in r.json()
    assert r.headers["x-eugene-plexus-driver"] == "d1"
    assert r.headers["x-eugene-plexus-attempts"] == "1"


# --------------------------------------------------------------------------- #
# Streaming
# --------------------------------------------------------------------------- #


def _events(raw: str) -> list[dict[str, Any]]:
    """Parse the SSE body into the `data:` objects, in order."""
    out = []
    for block in raw.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


def test_the_stream_is_anthropic_shaped_and_carries_no_done_sentinel(
    settings: Settings,
) -> None:
    """`data: [DONE]` is OpenAI's framing. A strict Anthropic client
    rejects a stream that carries it, and the rejection happens INSIDE a
    200 where no status code can report it."""
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    fake.responses = ["hello there friend"]
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=body(stream=True))
    assert r.status_code == 200
    assert "[DONE]" not in r.text

    events = _events(r.text)
    types = [e["type"] for e in events]
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"
    assert "content_block_start" in types
    assert "content_block_stop" in types
    assert types[-2] == "message_delta"

    # Both channels carry the name, because a client may read either.
    assert "event: message_start" in r.text
    assert "event: message_stop" in r.text

    text = "".join(e["delta"]["text"] for e in events if e["type"] == "content_block_delta")
    assert text == "hello there friend"

    stop = next(e for e in events if e["type"] == "message_delta")
    assert stop["delta"]["stop_reason"] == "end_turn"


def test_message_start_names_the_model_that_actually_answered(settings: Settings) -> None:
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    fake.responses = ["ok"]
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=body(stream=True))
    start = _events(r.text)[0]
    assert start["message"]["role"] == "assistant"
    assert start["message"]["type"] == "message"
    assert start["message"]["model"] == MODEL
    assert start["message"]["content"] == []


def test_stream_block_indices_are_assigned_statefully(settings: Settings) -> None:
    """The one piece of real work. Our internal stream numbers tool
    fragments per call and gives text no index at all, so the translator
    holds the open text block and a map from call index to block index.
    Getting it wrong means a strict SDK rejects the stream inside a
    200 -- which reads to the user as the model producing nothing."""
    fake = FakeDriverClient(name="d1", model_id=MODEL, supports_tools=True)
    fake.tool_calls = [
        ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="Glob", arguments='{"pattern": "*.py"}'),
        ),
        ToolCall(
            id="call_2",
            type="function",
            function=FunctionCall(name="Read", arguments='{"path": "a.py"}'),
        ),
    ]
    request = body(stream=True, tools=[{"name": "Glob", "input_schema": {"type": "object"}}])
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=request)

    events = _events(r.text)
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert [s["index"] for s in starts] == [0, 1]
    assert [s["content_block"]["type"] for s in starts] == ["tool_use", "tool_use"]
    assert starts[0]["content_block"]["id"] == "call_1"
    assert starts[1]["content_block"]["id"] == "call_2"
    # Empty input on start; the arguments arrive as deltas.
    assert starts[0]["content_block"]["input"] == {}

    # Every block that opened also closed, in order.
    stops = [e["index"] for e in events if e["type"] == "content_block_stop"]
    assert stops == [0, 1]

    first = "".join(
        e["delta"]["partial_json"]
        for e in events
        if e["type"] == "content_block_delta" and e["index"] == 0
    )
    assert json.loads(first) == {"pattern": "*.py"}

    stop = next(e for e in events if e["type"] == "message_delta")
    assert stop["delta"]["stop_reason"] == "tool_use"


def test_a_text_block_is_closed_before_a_tool_block_opens(settings: Settings) -> None:
    """Anthropic allows one open block at a time. A translator that left
    text open would produce a stream a strict client rejects."""
    fake = FakeDriverClient(name="d1", model_id=MODEL, supports_tools=True)
    fake.responses = ["Looking."]
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=body(stream=True))
    events = _events(r.text)
    order = [e["type"] for e in events if e["type"].startswith("content_block")]
    # **This assertion is load-bearing and was added because the test
    # passed before the endpoint existed**: an empty `order` satisfies
    # every nesting rule below, so without it the check could not fail.
    # Same family as M10's check 7 and the tree slice's "a driver sits
    # under its machine" -- a structural assertion that never meets its
    # subject.
    assert order, "no content block events at all"
    # start ... delta ... stop, and never two starts without a stop.
    depth = 0
    for kind in order:
        if kind == "content_block_start":
            assert depth == 0, "a block opened while another was open"
            depth = 1
        elif kind == "content_block_stop":
            assert depth == 1
            depth = 0
    assert depth == 0, "a content block was left open"


def test_a_backend_that_dies_mid_stream_reports_an_error_event(settings: Settings) -> None:
    """Past the first token the slot is committed, so the stream
    truncates. The 200 is long gone, so the report is an `error` event
    -- followed by `message_stop`, so a client's state machine does not
    hang waiting for one."""
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    fake.responses = ["one two three"]
    fake.stream_error_after = 1
    # A transport error rather than the fake's default RuntimeError: the
    # translator catches what the tiered client actually re-raises past
    # the commit point, and matching the OpenAI door here is deliberate
    # -- an unexpected exception type should surface as a real failure
    # rather than be laundered into a well-formed stream event.
    fake.stream_error = httpx.ConnectError("connection refused")
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages", json=body(stream=True))
    events = _events(r.text)
    types = [e["type"] for e in events]
    assert "error" in types
    assert types[-1] == "message_stop"
    assert "[DONE]" not in r.text


# --------------------------------------------------------------------------- #
# Auth -- both headers, and the status that is not the obvious one
# --------------------------------------------------------------------------- #


def _issue(
    *,
    signing_key: bytes,
    sub: str,
    aud: str,
    ttl_seconds: int = 60,
    jti: str | None = "key-1",
) -> str:
    """Mint a JWT exactly the way the agent would.

    `jti` is not decoration. S4's rule is that a client-audience token
    **without** one is refused, because it could never be matched
    against the revocation list and a credential that cannot be turned
    off is not one to accept on a signature alone. The first draft of
    these tests omitted it and read the resulting 403 as a bug in the
    new door; it was the old rule, working.
    """
    now = int(time.time())
    claims: dict[str, Any] = {"sub": sub, "aud": aud, "iat": now, "exp": now + ttl_seconds}
    if jti is not None:
        claims["jti"] = jti
    return jwt.encode(claims, signing_key, algorithm=_JWT_ALG)


@pytest.fixture
def signing_key() -> bytes:
    return secrets.token_bytes(32)


@pytest.fixture
def authed_client(settings: Settings, signing_key: bytes) -> Iterator[TestClient]:
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    fake.responses = ["ok"]
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake)
    app.state.auth_state = AuthState(signing_key=signing_key, service_token=None, master_key=None)
    from tests.test_client_keys import FakeAgent

    app.state.client_key_guard = FakeAgent().as_guard()
    with TestClient(app) as client:
        client.fake = fake  # type: ignore[attr-defined]
        yield client


def test_x_api_key_is_a_credential_here(authed_client: TestClient, signing_key: bytes) -> None:
    """The reproduction of the auth half. Measured: `ANTHROPIC_API_KEY`
    sends this header and **no `Authorization` header at all**, so a
    door on the existing bearer scheme sees no credential rather than
    the wrong one."""
    token = _issue(signing_key=signing_key, sub="app", aud="client")
    r = authed_client.post("/v1/messages", json=body(), headers={"x-api-key": token})
    assert r.status_code == 200, r.text


def test_authorization_bearer_still_works(authed_client: TestClient, signing_key: bytes) -> None:
    """`ANTHROPIC_AUTH_TOKEN` sends this one, and no `x-api-key`."""
    token = _issue(signing_key=signing_key, sub="app", aud="client")
    r = authed_client.post(
        "/v1/messages", json=body(), headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200, r.text


def test_an_operator_token_works_too(authed_client: TestClient, signing_key: bytes) -> None:
    token = _issue(signing_key=signing_key, sub="troy", aud="operator")
    r = authed_client.post("/v1/messages", json=body(), headers={"x-api-key": token})
    assert r.status_code == 200, r.text


def test_no_credential_is_403_and_never_401(authed_client: TestClient) -> None:
    """**The divergence, and it must not be tidied away.** Measured: a
    401 from this endpoint makes Claude Code retry without bound -- nine
    attempts in 79 seconds, still climbing -- while showing its user
    nothing at all. A 403 is reported on the first attempt, verbatim."""
    r = authed_client.post("/v1/messages", json=body())
    assert r.status_code == 403, r.text
    payload = r.json()
    assert payload["type"] == "error"
    assert payload["error"]["type"] in {"authentication_error", "permission_error"}
    assert "detail" not in payload


def test_a_bad_token_is_403(authed_client: TestClient) -> None:
    r = authed_client.post("/v1/messages", json=body(), headers={"x-api-key": "not-a-jwt"})
    assert r.status_code == 403, r.text
    assert r.json()["type"] == "error"


def test_an_expired_token_is_403(authed_client: TestClient, signing_key: bytes) -> None:
    token = _issue(signing_key=signing_key, sub="app", aud="client", ttl_seconds=-3600)
    r = authed_client.post("/v1/messages", json=body(), headers={"x-api-key": token})
    assert r.status_code == 403


def test_the_openai_door_still_answers_401(authed_client: TestClient) -> None:
    """The other half of the divergence: an OpenAI SDK reports a 401
    properly, and a 403 there would read as "this key exists but may
    not do this". Asserting both is what stops the pair being
    "harmonised" by someone who sees only one of them."""
    r = authed_client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 401


def test_a_wrong_audience_is_refused(authed_client: TestClient, signing_key: bytes) -> None:
    """`aud: client` is accepted on the front door; an audience that is
    neither operator, service nor client is not a credential here."""
    token = _issue(signing_key=signing_key, sub="x", aud="nonsense")
    r = authed_client.post("/v1/messages", json=body(), headers={"x-api-key": token})
    assert r.status_code == 403


def test_a_client_key_with_no_jti_is_refused_here_too(
    authed_client: TestClient, signing_key: bytes
) -> None:
    """S4's rule, asserted on the new door rather than assumed to carry
    over: a client token that cannot be matched against the revocation
    list is a credential that can never be turned off."""
    token = _issue(signing_key=signing_key, sub="app", aud="client", jti=None)
    r = authed_client.post("/v1/messages", json=body(), headers={"x-api-key": token})
    assert r.status_code == 403


def test_the_anthropic_door_answers_cors_with_x_api_key_allowed(settings: Settings) -> None:
    """A preflight that omits `x-api-key` is a door that answers the
    preflight and then fails the request -- and the browser reports only
    `Failed to fetch` while the server sees nothing at all."""
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    with _client(_app_with(settings, fake)) as client:
        r = client.options(
            "/v1/messages",
            headers={
                "Origin": "https://example.test",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-api-key, anthropic-version",
            },
        )
    assert r.status_code < 400, r.text
    assert r.headers["access-control-allow-origin"] == "*"
    allowed = r.headers["access-control-allow-headers"].lower()
    assert "x-api-key" in allowed
    assert "anthropic-version" in allowed


def test_an_operator_path_still_refuses_another_origin(settings: Settings) -> None:
    """The new door widened the front door, not the gateway."""
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    with _client(_app_with(settings, fake)) as client:
        r = client.get("/v1/config", headers={"Origin": "https://example.test"})
    assert "access-control-allow-origin" not in r.headers


def test_a_system_role_inside_messages_is_carried_in_place(settings: Settings) -> None:
    """**The live run's finding, and the published contract says this
    cannot happen.**

    Anthropic documents `user` and `assistant` as the only message roles
    and this schema said so, until a real Claude Code driving a tool
    loop was refused on its first request with
    `messages.1.role: Input should be 'user' or 'assistant'`. It sends
    the documented top-level `system` AND a separate `system`-role
    message inside `messages`, several kilobytes of it, after the first
    user turn.

    Carried **in place**: the client put it there deliberately, and
    hoisting it into the leading system prompt would change what the
    model sees for the sake of tidiness on a wire we do not own.

    This shape appears only once tools are in play, which is why every
    unit fixture above -- all built from a capture of a simple
    request -- missed it. A live run is not a formality.
    """
    fake = FakeDriverClient(name="d1", model_id=MODEL, supports_tools=True)
    fake.responses = ["ok"]
    request = body(
        messages=[
            {"role": "user", "content": [{"type": "text", "text": "list the python files"}]},
            {"role": "system", "content": "<env>cwd: /work</env>"},
        ]
    )
    with _client(_app_with(settings, fake)) as client:
        r = client.post("/v1/messages?beta=true", json=request)
    assert r.status_code == 200, r.text

    sent = [(m.role.value, m.content) for m in fake.calls[0].messages]
    roles = [role for role, _ in sent]
    # The fixture's own top-level system leads; the in-message one keeps
    # its position after the user turn rather than being merged into it.
    assert roles == ["system", "user", "system"]
    assert sent[2][1] == "<env>cwd: /work</env>"
    assert "x-anthropic-billing-header" in sent[0][1]


def test_text_then_a_tool_call_closes_the_text_block_first(settings: Settings) -> None:
    """**The transition, which neither streaming test above exercised.**

    One test streams only text and the other only tool calls, so the
    nesting assertion in the first never met a second block and the
    index assertion in the second never met an open text block. A
    sabotage that left the text block open escaped both of them: each
    was correct about its own case and neither covered the seam.

    Anthropic allows one open content block at a time, and this is the
    only shape where a translator can get that wrong.
    """
    fake = FakeDriverClient(name="d1", model_id=MODEL, supports_tools=True)
    fake.responses = ["Looking."]
    fake.tool_calls = [
        ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="Glob", arguments='{"pattern": "*.py"}'),
        )
    ]
    # The fake answers with tool calls OR text, never both, so the text
    # block has to come from a second driver in the same conversation.
    # Drive the seam directly instead: the translator is the subject.
    from eugene_plexus_gateway.anthropic import StreamTranslator

    translator = StreamTranslator(MODEL)
    frames = [translator.start()]
    frames += translator.text("Looking.")
    frames += translator.tool_fragments(
        [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "Glob", "arguments": "{}"},
            }
        ]
    )
    frames += translator.finish(reason="tool_use")

    events = _events("".join(frames))
    order = [(e["type"], e.get("index")) for e in events if e["type"].startswith("content_block")]
    assert order, "no content block events at all"

    # The text block opened at 0 and MUST close before the tool opens at 1.
    assert ("content_block_start", 0) in order
    assert order.index(("content_block_stop", 0)) < order.index(("content_block_start", 1))

    depth = 0
    for kind, _ in order:
        if kind == "content_block_start":
            assert depth == 0, "a block opened while another was open"
            depth = 1
        elif kind == "content_block_stop":
            assert depth == 1
            depth = 0
    assert depth == 0, "a content block was left open"


def test_a_preflight_that_names_no_headers_still_allows_x_api_key(settings: Settings) -> None:
    """The fallback list, which is **not** the mechanism.

    R4's first draft claimed a default list without `x-api-key` would
    answer the preflight and then fail the request. It would not: the
    reply echoes `access-control-request-headers` whenever a preflight
    sends one, and a real browser always does. The sabotage pass found
    that claim by removing `x-api-key` from the default and watching
    every check still pass.

    The fallback is still asserted, because a fallback that describes a
    different door from the one the echo opens is a trap for the next
    reader.
    """
    fake = FakeDriverClient(name="d1", model_id=MODEL)
    with _client(_app_with(settings, fake)) as client:
        r = client.options(
            "/v1/messages",
            headers={
                "Origin": "https://example.test",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert r.status_code < 400, r.text
    allowed = r.headers["access-control-allow-headers"].lower()
    assert "x-api-key" in allowed
    assert "anthropic-version" in allowed
