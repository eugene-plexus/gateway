"""Reasoning reaches both doors, and the local samplers reach the driver.

2026-09-23. Until today a reasoning model's thinking was discarded one
hop down (the driver read only `content`), so nothing here ever saw it;
and `top_k`, `min_p`, the two penalties, `parallel_tool_calls` and the
`developer` role were each refused with a 400 at the OpenAI door --
`top_k` at the Anthropic door too -- because the internal request had
nowhere to carry them. `tool_choice.disable_parallel_tool_use` was worse:
accepted and silently dropped.

Wire fidelity is the point, as in `test_anthropic_messages.py`: every
assertion is on the name a real client parses -- `reasoning_content` on
the OpenAI wire, `thinking` blocks and `thinking_delta` on Anthropic's.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import (
    FinishReason,
    GenerateRequest,
    Usage,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import FailoverDriverClient
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table
from .test_anthropic_messages import body as claude_code_body

MODEL = "Qwen3-30B-A3B-Q4_K_M"
THOUGHT = "CANARYTHOUGHT the user wants a number, so 2+2 is 4"
SIGNATURE_PREFIX = "eugene-plexus-reasoning-v1:"


def _app(settings: Settings, *fakes: FakeDriverClient, **table: Any) -> FastAPI:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*fakes, **table)
    return app


def _fake(**knobs: Any) -> FakeDriverClient:
    fake = FakeDriverClient(name="qwen-box", model_id=MODEL, supports_tools=True)
    for key, value in knobs.items():
        setattr(fake, key, value)
    return fake


def _chat(**extra: Any) -> dict[str, Any]:
    return {"model": MODEL, "messages": [{"role": "user", "content": "What is 2+2?"}], **extra}


def _openai_frames(raw: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[6:])
        for line in raw.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _anthropic_events(raw: str) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")]


# --------------------------------------------------------------------------- #
# OpenAI door: the samplers
# --------------------------------------------------------------------------- #


def test_the_local_samplers_reach_the_driver_as_explicit_settings(settings: Settings) -> None:
    fake = _fake(responses=["4"])
    with TestClient(_app(settings, fake)) as client:
        r = client.post(
            "/v1/chat/completions",
            json=_chat(
                top_k=20,
                min_p=0.05,
                frequency_penalty=0.1,
                presence_penalty=-0.2,
                parallel_tool_calls=True,
            ),
        )

    assert r.status_code == 200, r.text
    sent = fake.calls[-1]
    assert (sent.topK, sent.minP, sent.frequencyPenalty, sent.presencePenalty) == (
        20,
        0.05,
        0.1,
        -0.2,
    )
    assert sent.parallelToolCalls is True
    assert {"topK", "minP", "frequencyPenalty", "presencePenalty", "parallelToolCalls"} <= set(
        sent.callerSettings or []
    )


def test_falsy_samplers_are_still_explicit(settings: Settings) -> None:
    """`top_k: 0` disables the cut, `parallel_tool_calls: false` asks for
    one call a turn. Both falsy, both the caller's -- the seed=0 rule."""
    fake = _fake(responses=["4"])
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/chat/completions", json=_chat(top_k=0, parallel_tool_calls=False))

    assert r.status_code == 200, r.text
    sent = fake.calls[-1]
    assert sent.topK == 0
    assert sent.parallelToolCalls is False
    assert {"topK", "parallelToolCalls"} <= set(sent.callerSettings or [])


def test_unset_samplers_are_neither_sent_nor_claimed(settings: Settings) -> None:
    fake = _fake(responses=["4"])
    with TestClient(_app(settings, fake)) as client:
        client.post("/v1/chat/completions", json=_chat())

    sent = fake.calls[-1]
    assert (sent.topK, sent.minP, sent.frequencyPenalty, sent.presencePenalty) == (
        None,
        None,
        None,
        None,
    )
    assert sent.parallelToolCalls is None
    assert not {"topK", "minP", "frequencyPenalty", "presencePenalty", "parallelToolCalls"} & set(
        sent.callerSettings or []
    )


def test_a_backend_that_cannot_carry_top_k_is_routed_around_not_tried(
    settings: Settings,
) -> None:
    """OpenAI's own endpoint refuses `top_k`; its driver leaves it out of
    `supportedSettings`. With nothing else serving the model, the request
    is refused before anything is forwarded -- and the pair: the same
    backend still serves a request that does not ask for it."""
    fake = _fake(responses=["4", "4"])
    fake.supported_settings = [s for s in fake.supported_settings if s != "topK"]
    with TestClient(_app(settings, fake)) as client:
        refused = client.post("/v1/chat/completions", json=_chat(top_k=20))
        served = client.post("/v1/chat/completions", json=_chat(temperature=0.2))

    assert refused.status_code == 400, refused.text
    assert served.status_code == 200, served.text
    assert len(fake.calls) == 1


# --------------------------------------------------------------------------- #
# OpenAI door: the developer role, and reasoning handed back
# --------------------------------------------------------------------------- #


def test_a_developer_message_reaches_the_backend_as_system_in_place(settings: Settings) -> None:
    fake = _fake(responses=["Paris"])
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "developer", "content": "Answer in one word."},
        {"role": "user", "content": "Capital of France?"},
    ]
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/chat/completions", json={"model": MODEL, "messages": messages})

    assert r.status_code == 200, r.text
    roles = [m.role.value for m in fake.calls[-1].messages]
    assert roles == ["user", "system", "user"]
    assert fake.calls[-1].messages[1].content == "Answer in one word."


def test_an_assistant_turn_hands_its_reasoning_back_to_the_driver(settings: Settings) -> None:
    fake = _fake(responses=["It is snowing."])
    messages = [
        {"role": "user", "content": "Weather in Oslo?"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": THOUGHT,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "snow"},
    ]
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/chat/completions", json={"model": MODEL, "messages": messages})

    assert r.status_code == 200, r.text
    sent = fake.calls[-1].messages
    assert sent[1].reasoning == THOUGHT
    assert sent[0].reasoning is None and sent[2].reasoning is None


def test_reasoning_content_on_a_user_message_is_refused_by_name(settings: Settings) -> None:
    """Only a model's own turn has reasoning. Anywhere else it would be
    dropped, and a dropped field that changes the prompt is the A2 rule."""
    fake = _fake(responses=["4"])
    messages = [{"role": "user", "content": "hi", "reasoning_content": "SECRET"}]
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/chat/completions", json={"model": MODEL, "messages": messages})

    assert r.status_code == 400, r.text
    assert r.json()["error"]["param"] == "messages[0].reasoning_content"
    assert "SECRET" not in r.text
    assert not fake.calls


# --------------------------------------------------------------------------- #
# OpenAI door: reasoning comes back
# --------------------------------------------------------------------------- #


def test_the_batch_response_carries_reasoning_content(settings: Settings) -> None:
    fake = _fake(responses=["4"], reasoning=THOUGHT)
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/chat/completions", json=_chat())

    message = r.json()["choices"][0]["message"]
    assert message["reasoning_content"] == THOUGHT
    assert message["content"] == "4"


def test_no_reasoning_means_no_key_and_no_unreported_usage_detail(settings: Settings) -> None:
    """Absent, not null: a non-reasoning model's reply is byte-for-byte
    what it was, and a detail the backend did not report is not claimed."""
    fake = _fake(responses=["4"], usage=Usage(promptTokens=11, completionTokens=4, totalTokens=15))
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/chat/completions", json=_chat())

    payload = r.json()
    assert "reasoning_content" not in payload["choices"][0]["message"]
    assert payload["usage"] == {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}


def test_reported_usage_details_use_openais_names(settings: Settings) -> None:
    usage = Usage(
        promptTokens=100,
        completionTokens=40,
        totalTokens=140,
        cachedPromptTokens=80,
        reasoningTokens=30,
    )
    fake = _fake(responses=["4", "4"], usage=usage)
    with TestClient(_app(settings, fake)) as client:
        batch = client.post("/v1/chat/completions", json=_chat())
        streamed = client.post(
            "/v1/chat/completions",
            json=_chat(stream=True, stream_options={"include_usage": True}),
        )

    assert batch.json()["usage"]["prompt_tokens_details"] == {"cached_tokens": 80}
    assert batch.json()["usage"]["completion_tokens_details"] == {"reasoning_tokens": 30}
    final = [f for f in _openai_frames(streamed.text) if f.get("usage")][-1]
    assert final["usage"]["prompt_tokens_details"] == {"cached_tokens": 80}
    assert final["usage"]["completion_tokens_details"] == {"reasoning_tokens": 30}


def test_the_stream_carries_reasoning_deltas_before_the_answer(settings: Settings) -> None:
    fake = _fake(responses=["the answer is 4"], reasoning=THOUGHT)
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/chat/completions", json=_chat(stream=True))

    deltas = [f["choices"][0]["delta"] for f in _openai_frames(r.text) if f.get("choices")]
    reasoning = [d["reasoning_content"] for d in deltas if d.get("reasoning_content")]
    content_at = [i for i, d in enumerate(deltas) if d.get("content")]
    reasoning_at = [i for i, d in enumerate(deltas) if d.get("reasoning_content")]
    assert "".join(reasoning) == THOUGHT
    assert len(reasoning) == 2
    assert max(reasoning_at) < min(content_at)
    assert "".join(d.get("content") or "" for d in deltas) == "the answer is 4"
    # A reasoning frame is not also a content frame.
    assert not [d for d in deltas if d.get("reasoning_content") and d.get("content")]


async def test_reasoning_is_the_commit_point_like_any_other_output() -> None:
    """Once a caller has been shown the model thinking, another model's
    answer cannot be spliced on after it. A backend that dies after its
    reasoning truncates; the backup is never asked. And the pair: dying
    before ANY output still cascades."""
    for fail_after_reasoning, expect_backup in ((True, False), (False, True)):
        primary = _fake(reasoning=THOUGHT, stream_error=httpx.ConnectError("gone"))
        if fail_after_reasoning:
            primary.stream_error_after_reasoning = True
        else:
            primary.generate_error = httpx.ConnectError("gone")
        backup = _fake(responses=["backup reply"])
        slot = FailoverDriverClient(name="qwen-box", candidates=[primary, backup])
        request = GenerateRequest(messages=[{"role": "user", "content": "hi"}])

        if expect_backup:
            events = [e async for e in slot.stream(request)]
            assert "".join(e.text for e in events if not e.done) == "backup reply"
        else:
            seen: list[str] = []
            with pytest.raises(httpx.ConnectError):
                async for event in slot.stream(request):
                    seen.append(event.reasoning)
            assert "".join(seen) == THOUGHT
        assert bool(backup.calls) is expect_backup


# --------------------------------------------------------------------------- #
# Anthropic door
# --------------------------------------------------------------------------- #


def test_top_k_is_carried_on_the_anthropic_door(settings: Settings) -> None:
    fake = _fake(responses=["ok"])
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/messages", json=claude_code_body(top_k=40))

    assert r.status_code == 200, r.text
    assert fake.calls[-1].topK == 40
    assert "topK" in (fake.calls[-1].callerSettings or [])


@pytest.mark.parametrize("disable,expected", [(True, False), (False, True), (None, None)])
def test_disable_parallel_tool_use_is_carried_with_its_sense_inverted(
    settings: Settings, disable: bool | None, expected: bool | None
) -> None:
    fake = _fake(responses=["ok"])
    choice: dict[str, Any] = {"type": "auto"}
    if disable is not None:
        choice["disable_parallel_tool_use"] = disable
    tools = [{"name": "get_weather", "input_schema": {"type": "object"}}]
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/messages", json=claude_code_body(tools=tools, tool_choice=choice))

    assert r.status_code == 200, r.text
    assert fake.calls[-1].parallelToolCalls is expected


def _thinking_response(settings: Settings, thinking: Any, *, stream: bool = False) -> Any:
    fake = _fake(responses=["the answer is 4"], reasoning=THOUGHT)
    with TestClient(_app(settings, fake)) as client:
        return client.post("/v1/messages", json=claude_code_body(thinking=thinking, stream=stream))


def test_enabled_thinking_returns_a_thinking_block_first(settings: Settings) -> None:
    r = _thinking_response(settings, {"type": "adaptive"})

    content = r.json()["content"]
    assert content[0] == {"type": "thinking", "thinking": THOUGHT, "signature": ""}
    assert content[1] == {"type": "text", "text": "the answer is 4"}


def test_display_omitted_empties_the_text_and_carries_it_in_the_signature(
    settings: Settings,
) -> None:
    """Claude Code sends `display: "omitted"` on every request (measured).
    Anthropic's own answer is a block with the text empty and the
    continuity in the opaque signature -- so ours is too."""
    r = _thinking_response(settings, {"type": "adaptive", "display": "omitted"})

    block = r.json()["content"][0]
    assert block["type"] == "thinking"
    assert block["thinking"] == ""
    assert block["signature"].startswith(SIGNATURE_PREFIX)
    decoded = base64.b64decode(block["signature"][len(SIGNATURE_PREFIX) :]).decode()
    assert decoded == THOUGHT
    assert r.json()["content"][1]["text"] == "the answer is 4"


@pytest.mark.parametrize(
    "thinking", [None, {"type": "disabled"}, "absent"], ids=["null", "disabled", "absent"]
)
def test_a_client_that_did_not_enable_thinking_gets_no_thinking_block(
    settings: Settings, thinking: Any
) -> None:
    """Anthropic's own rule, and the one that keeps `content[0].text`
    pointing at the answer for every client that never asked."""
    fake = _fake(responses=["the answer is 4"], reasoning=THOUGHT)
    request = claude_code_body()
    if thinking == "absent":
        request.pop("thinking")
    else:
        request["thinking"] = thinking
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/messages", json=request)

    assert r.json()["content"] == [{"type": "text", "text": "the answer is 4"}]


def test_the_stream_opens_and_closes_a_thinking_block_before_the_text(
    settings: Settings,
) -> None:
    r = _thinking_response(settings, {"type": "adaptive"}, stream=True)

    events = _anthropic_events(r.text)
    types = [e["type"] for e in events]
    assert types[0] == "message_start"
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert [s["content_block"]["type"] for s in starts] == ["thinking", "text"]
    assert [s["index"] for s in starts] == [0, 1]
    thinking = [
        e["delta"]["thinking"]
        for e in events
        if e["type"] == "content_block_delta" and e["delta"]["type"] == "thinking_delta"
    ]
    assert "".join(thinking) == THOUGHT and len(thinking) == 2
    # The thinking block closes before the text block opens.
    stop_0 = next(
        i for i, e in enumerate(events) if e == {"type": "content_block_stop", "index": 0}
    )
    start_1 = next(i for i, e in enumerate(events) if e is starts[1])
    assert stop_0 < start_1
    text = "".join(
        e["delta"]["text"]
        for e in events
        if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"
    )
    assert text == "the answer is 4"


def test_an_omitted_stream_sends_no_thinking_text_and_a_signature_at_the_close(
    settings: Settings,
) -> None:
    r = _thinking_response(settings, {"type": "adaptive", "display": "omitted"}, stream=True)

    events = _anthropic_events(r.text)
    deltas = [e["delta"] for e in events if e["type"] == "content_block_delta"]
    assert not [d for d in deltas if d["type"] == "thinking_delta"]
    signatures = [d["signature"] for d in deltas if d["type"] == "signature_delta"]
    assert len(signatures) == 1
    assert base64.b64decode(signatures[0][len(SIGNATURE_PREFIX) :]).decode() == THOUGHT
    assert THOUGHT not in r.text.replace(signatures[0], "")
    signature_at = next(
        i for i, e in enumerate(events) if e.get("delta", {}).get("type") == "signature_delta"
    )
    assert events[signature_at + 1] == {"type": "content_block_stop", "index": 0}


def test_hidden_reasoning_still_starts_the_message_and_numbers_text_from_zero(
    settings: Settings,
) -> None:
    r = _thinking_response(settings, None, stream=True)

    events = _anthropic_events(r.text)
    assert events[0]["type"] == "message_start"
    starts = [e for e in events if e["type"] == "content_block_start"]
    assert [(s["index"], s["content_block"]["type"]) for s in starts] == [(0, "text")]
    assert THOUGHT not in r.text


def _assistant_turn(block: dict[str, Any]) -> dict[str, Any]:
    return claude_code_body(
        messages=[
            {"role": "user", "content": "Weather in Oslo?"},
            {
                "role": "assistant",
                "content": [
                    block,
                    {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "snow"}],
            },
        ],
        tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
    )


@pytest.mark.parametrize(
    "block,expected",
    [
        ({"type": "thinking", "thinking": THOUGHT, "signature": ""}, THOUGHT),
        (
            {
                "type": "thinking",
                "thinking": "",
                "signature": SIGNATURE_PREFIX + base64.b64encode(THOUGHT.encode()).decode(),
            },
            THOUGHT,
        ),
        # Anthropic-shaped: exactly as long as our prefix, then base64 that
        # DOES decode -- so only the prefix check can tell it is not ours.
        (
            {
                "type": "thinking",
                "thinking": "",
                "signature": "EqQBCkYIBxgCKkDzMWq4ZvLtNpR"
                + base64.b64encode(b"FOREIGN opaque").decode(),
            },
            None,
        ),
        # Ours, with junk a lenient decoder would skip past to "CORRUPT".
        (
            {
                "type": "thinking",
                "thinking": "",
                "signature": SIGNATURE_PREFIX + "@@@@" + base64.b64encode(b"CORRUPT").decode(),
            },
            None,
        ),
        ({"type": "redacted_thinking", "data": "EmwKAhgBEgy3va"}, None),
    ],
    ids=["text", "our-signature", "foreign-signature", "corrupt-signature", "redacted"],
)
def test_a_thinking_block_sent_back_becomes_that_turns_reasoning(
    settings: Settings, block: dict[str, Any], expected: str | None
) -> None:
    fake = _fake(responses=["It is snowing."])
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/messages", json=_assistant_turn(block))

    assert r.status_code == 200, r.text
    assistant = [m for m in fake.calls[-1].messages if m.role.value == "assistant"]
    assert len(assistant) == 1
    assert assistant[0].reasoning == expected
    assert assistant[0].toolCalls  # the call beside it survived


def test_the_matched_stop_sequence_is_named_on_both_paths(settings: Settings) -> None:
    fake = _fake(
        responses=["1, 2", "1, 2"],
        finish_reason=FinishReason.stop_sequence,
        stop_sequence="END",
    )
    with TestClient(_app(settings, fake)) as client:
        batch = client.post("/v1/messages", json=claude_code_body(stop_sequences=["END"]))
        streamed = client.post(
            "/v1/messages", json=claude_code_body(stop_sequences=["END"], stream=True)
        )

    assert batch.json()["stop_reason"] == "stop_sequence"
    assert batch.json()["stop_sequence"] == "END"
    delta = next(e for e in _anthropic_events(streamed.text) if e["type"] == "message_delta")
    assert delta["delta"] == {"stop_reason": "stop_sequence", "stop_sequence": "END"}


def test_cached_input_is_split_out_the_way_anthropic_counts_it(settings: Settings) -> None:
    usage = Usage(promptTokens=100, completionTokens=9, totalTokens=109, cachedPromptTokens=80)
    fake = _fake(responses=["ok"], usage=usage)
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/messages", json=claude_code_body())

    assert r.json()["usage"] == {
        "input_tokens": 20,
        "output_tokens": 9,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 80,
    }


@pytest.mark.parametrize(
    "thinking,listed",
    [
        ({"type": "adaptive", "display": "omitted"}, False),
        ({"type": "adaptive"}, False),
        ({"type": "enabled", "budget_tokens": 31999}, True),
        ({"type": "disabled"}, True),
        (None, False),
    ],
)
def test_thinking_is_named_as_ignored_only_for_what_is_not_honoured(
    settings: Settings, thinking: Any, listed: bool
) -> None:
    fake = _fake(responses=["ok"])
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/messages", json=claude_code_body(thinking=thinking))

    ignored = r.headers.get("x-eugene-plexus-ignored-settings", "")
    assert ("thinking" in ignored.split(", ")) is listed


# --------------------------------------------------------------------------- #
# The real driver client, not the fake
# --------------------------------------------------------------------------- #


async def test_the_http_driver_client_parses_reasoning_off_the_drivers_own_frames() -> None:
    """Every test above hands the gateway a fake that yields reasoning
    events directly, so none of them reads the wire. This one feeds
    `HttpDriverClient` the frames the driver's route actually writes --
    `event: token` with `{"reasoning": ...}` -- and the terminal `done`."""
    from eugene_plexus_gateway.driver_client import HttpDriverClient

    done = {
        "content": "4",
        "reasoning": THOUGHT,
        "finishReason": "stop_sequence",
        "stopSequence": "END",
        "backend": "openai_compat_http",
        "usage": {
            "promptTokens": 10,
            "completionTokens": 5,
            "totalTokens": 15,
            "cachedPromptTokens": 8,
        },
    }
    wire = (
        'event: token\ndata: {"reasoning": "CANARY"}\n\n'
        'event: token\ndata: {"reasoning": "THOUGHT"}\n\n'
        'event: token\ndata: {"text": "4"}\n\n'
        f"event: done\ndata: {json.dumps(done)}\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=wire, headers={"content-type": "text/event-stream"})

    client = HttpDriverClient(name="qwen-box", base_url="http://driver.invalid")
    client._client = httpx.AsyncClient(
        base_url=client.base_url, transport=httpx.MockTransport(handler)
    )
    events = [
        e async for e in client.stream(GenerateRequest(messages=[{"role": "user", "content": "q"}]))
    ]
    await client.aclose()

    assert [e.reasoning for e in events if e.reasoning] == ["CANARY", "THOUGHT"]
    assert [e.text for e in events if e.text] == ["4"]
    final = events[-1].result
    assert final is not None
    assert final.reasoning == THOUGHT
    assert final.stopSequence == "END"
    assert final.usage is not None and final.usage.cachedPromptTokens == 8


# --------------------------------------------------------------------------- #
# output_config: what today's Claude Code sends
# --------------------------------------------------------------------------- #


def test_output_config_effort_is_accepted_and_disclosed(settings: Settings) -> None:
    """Captured 2026-09-23 on every Claude Code request. A2's unknown-field
    refusal answered it with a 400, so the client this door exists for
    could not complete one request; the live acceptance run is what saw it."""
    fake = _fake(responses=["ok"])
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/messages", json=claude_code_body(output_config={"effort": "high"}))

    assert r.status_code == 200, r.text
    ignored = r.headers.get("x-eugene-plexus-ignored-settings", "").split(", ")
    assert "output_config" in ignored


@pytest.mark.parametrize(
    "config",
    [{"format": {"type": "json_schema", "schema": {"type": "object"}}}, {"task_budget": {}}],
    ids=["structured-output", "unknown-key"],
)
def test_any_other_output_config_key_is_refused_by_name(
    settings: Settings, config: dict[str, Any]
) -> None:
    fake = _fake(responses=["ok"])
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/messages", json=claude_code_body(output_config=config))

    assert r.status_code == 400, r.text
    assert f"output_config.{next(iter(config))}" in r.json()["error"]["message"]
    assert not fake.calls


def test_a_null_output_config_is_neither_refused_nor_disclosed(settings: Settings) -> None:
    fake = _fake(responses=["ok"])
    request = claude_code_body(output_config=None)
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/messages", json=request)

    assert r.status_code == 200, r.text
    assert "output_config" not in r.headers.get("x-eugene-plexus-ignored-settings", "")
