"""The chat fields P2c carries, where they go, and what comes back.

Every test here fails against the gateway as it was before P2c, whose chat
request refused `logprobs: true`, `logit_bias`, `reasoning_effort` and the
rest as unsupported, whose answers had no `logprobs` or `annotations`, and
which knew neither the deprecated `functions` nor
`/v1/responses/input_tokens`. Measured behaviour behind the rules:
`provider-accounts-measurement.md` section 7 (2026-09-28).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import (
    ChatAnnotation,
    ChatLogprobs,
    FinishReason,
    FunctionCall,
    GenerateRequest,
    GenerateResponse,
    ToolCall,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import StreamEvent
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

P2C = ["logprobs", "logitBias", "reasoningEffort", "verbosity", "prediction", "webSearchOptions"]
LOGPROBS = {
    "content": [
        {
            "token": "Red",
            "logprob": -0.01,
            "bytes": [82, 101, 100],
            "top_logprobs": [{"token": "Red", "logprob": -0.01, "bytes": [82, 101, 100]}],
        }
    ],
}
CITATION = {
    "type": "url_citation",
    "url_citation": {
        "url": "https://example.org/c",
        "title": "C",
        "start_index": 0,
        "end_index": 0,
    },
}


class Rich(FakeDriverClient):
    """A fake that takes P2c's settings and answers with logprobs and citations."""

    def __init__(self, takes: bool = True, **kw: Any) -> None:
        super().__init__(**kw)
        if takes:
            self.supported_settings = [*self.supported_settings, *P2C]

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        response = await super().generate(request)
        return response.model_copy(
            update={
                "logprobs": ChatLogprobs.model_validate(LOGPROBS) if request.logprobs else None,
                "annotations": [ChatAnnotation.model_validate(CITATION)]
                if request.webSearchOptions
                else None,
            }
        )

    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamEvent]:
        if self.tool_calls is not None:
            async for event in super().stream(request):
                yield event
            return
        self.calls.append(request)
        yield StreamEvent(text="Red", logprobs=LOGPROBS)
        yield StreamEvent(annotations=[CITATION])
        yield StreamEvent(text=" sky", logprobs={"content": [{"token": " sky", "logprob": -0.5}]})
        yield StreamEvent(
            done=True,
            result=GenerateResponse(
                content="Red sky",
                finishReason=FinishReason.stop,
                backend=self.backend,
                modelId=request.model or self.model_id,
                latencyMs=1,
            ),
        )


def serve(settings: Settings, *drivers: FakeDriverClient, **table: Any) -> TestClient:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*drivers, **table)
    return TestClient(app)


def chat(model: str = "m", **extra: Any) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": "A colour?"}], **extra}


FIELDS = {
    "logprobs": True,
    "top_logprobs": 2,
    "logit_bias": {"50256": -100},
    "reasoning_effort": "low",
    "verbosity": "low",
    "prediction": {"type": "content", "content": "Red"},
    "web_search_options": {"search_context_size": "low"},
}


# --------------------------------------------------------------------------- #
# Settings: carried, and routed only where a backend takes them
# --------------------------------------------------------------------------- #


def test_each_setting_reaches_the_driver_and_is_named_as_explicit(settings: Settings) -> None:
    rich = Rich(name="rich", model_id="m")
    with serve(settings, rich) as client:
        response = client.post("/v1/chat/completions", json=chat(**FIELDS))
    assert response.status_code == 200, response.text
    sent = rich.calls[-1]
    assert set(P2C) <= set(sent.callerSettings or [])
    assert sent.logprobs is True and sent.topLogprobs == 2
    assert sent.logitBias == {"50256": -100}
    assert sent.reasoningEffort.value == "low" and sent.verbosity.value == "low"
    assert sent.prediction.model_dump(mode="json") == {"type": "content", "content": "Red"}
    assert sent.webSearchOptions.model_dump(mode="json", exclude_none=True) == {
        "search_context_size": "low"
    }


def test_logprobs_false_asks_for_nothing(settings: Settings) -> None:
    plain = Rich(False, name="plain", model_id="m")
    with serve(settings, plain) as client:
        response = client.post("/v1/chat/completions", json=chat(logprobs=False))
    assert response.status_code == 200, response.text
    assert plain.calls[-1].logprobs is None
    assert "logprobs" not in (plain.calls[-1].callerSettings or [])


def test_a_setting_routes_past_a_backend_that_does_not_take_it(settings: Settings) -> None:
    plain = Rich(False, name="plain", model_id="plain")
    rich = Rich(name="rich", model_id="rich")
    slots = [{"model": "assistant", "targets": ["plain", "rich"]}]
    with serve(settings, plain, rich, slots=slots) as client:
        response = client.post("/v1/chat/completions", json=chat("assistant", logit_bias={"1": 5}))
    assert response.status_code == 200, response.text
    assert not plain.calls and rich.calls
    assert response.json()["x_eugene_plexus"]["tier"] == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("logit_bias", {"1": 5}),
        ("reasoning_effort", "high"),
        ("verbosity", "low"),
        ("prediction", {"type": "content", "content": "x"}),
        ("web_search_options", {}),
        ("logprobs", True),
    ],
)
def test_with_no_backend_that_takes_it_the_refusal_names_it(
    settings: Settings, field: str, value: Any
) -> None:
    plain = Rich(False, name="plain", model_id="m")
    with serve(settings, plain) as client:
        response = client.post("/v1/chat/completions", json=chat(**{field: value}))
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == field and field in error["message"], error
    assert not plain.calls


def test_top_logprobs_needs_logprobs(settings: Settings) -> None:
    rich = Rich(name="rich", model_id="m")
    with serve(settings, rich) as client:
        response = client.post("/v1/chat/completions", json=chat(top_logprobs=3))
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "top_logprobs"


def test_hints_are_carried_and_never_restrict_routing(settings: Settings) -> None:
    plain = Rich(False, name="plain", model_id="m")
    hints = {
        "prompt_cache_key": "k",
        "prompt_cache_retention": "24h",
        "service_tier": "flex",
        "safety_identifier": "u",
    }
    with serve(settings, plain) as client:
        response = client.post("/v1/chat/completions", json=chat(**hints))
    assert response.status_code == 200, response.text
    sent = plain.calls[-1]
    assert (sent.promptCacheKey, sent.promptCacheRetention.value) == ("k", "24h")
    assert (sent.serviceTier.value, sent.safetyIdentifier) == ("flex", "u")
    assert not {"promptCacheKey", "serviceTier", "safetyIdentifier"} & set(
        sent.callerSettings or []
    )


# --------------------------------------------------------------------------- #
# What comes back
# --------------------------------------------------------------------------- #


def test_a_batch_answer_carries_logprobs_and_citations(settings: Settings) -> None:
    rich = Rich(name="rich", model_id="m")
    with serve(settings, rich) as client:
        body = client.post("/v1/chat/completions", json=chat(**FIELDS)).json()
    choice = body["choices"][0]
    assert choice["logprobs"]["content"] == LOGPROBS["content"]
    assert choice["message"]["annotations"] == [CITATION]


def test_a_stream_carries_logprobs_on_the_choice_and_citations_in_the_delta(
    settings: Settings,
) -> None:
    rich = Rich(name="rich", model_id="m")
    with serve(settings, rich) as client:
        response = client.post("/v1/chat/completions", json=chat(stream=True, **FIELDS))
    frames = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")
    ]
    choices = [c for f in frames for c in f.get("choices") or []]
    with_logprobs = [c for c in choices if c.get("logprobs")]
    assert [c["delta"].get("content") for c in with_logprobs] == ["Red", " sky"]
    assert with_logprobs[0]["logprobs"]["content"] == LOGPROBS["content"]
    assert [c["delta"]["annotations"] for c in choices if c["delta"].get("annotations")] == [
        [CITATION]
    ]


# --------------------------------------------------------------------------- #
# The deprecated functions
# --------------------------------------------------------------------------- #

WEATHER = {
    "name": "get_weather",
    "description": "Weather.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def _calling(**kw: Any) -> Rich:
    fake = Rich(name="rich", model_id="m", supports_tools=True, **kw)
    fake.tool_calls = [
        ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="get_weather", arguments='{"city": "Oslo"}'),
        )
    ]
    return fake


def test_functions_are_carried_as_tools_and_answered_as_a_function_call(settings: Settings) -> None:
    fake = _calling()
    with serve(settings, fake) as client:
        response = client.post(
            "/v1/chat/completions",
            json=chat(functions=[WEATHER], function_call={"name": "get_weather"}),
        )
    assert response.status_code == 200, response.text
    sent = fake.calls[-1]
    assert [t.function.name for t in sent.tools] == ["get_weather"]
    assert sent.toolChoice.model_dump(mode="json") == {
        "type": "function",
        "function": {"name": "get_weather"},
    }
    assert sent.parallelToolCalls is False
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "function_call"
    assert choice["message"]["function_call"] == {
        "name": "get_weather",
        "arguments": '{"city": "Oslo"}',
    }
    assert choice["message"].get("tool_calls") is None


def test_a_streamed_function_call_is_function_call_fragments(settings: Settings) -> None:
    fake = _calling()
    with serve(settings, fake) as client:
        response = client.post("/v1/chat/completions", json=chat(stream=True, functions=[WEATHER]))
    frames = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")
    ]
    choices = [c for f in frames for c in f.get("choices") or []]
    fragments = [c["delta"]["function_call"] for c in choices if c["delta"].get("function_call")]
    assert fragments[0]["name"] == "get_weather"
    assert "".join(f.get("arguments") or "" for f in fragments) == '{"city": "Oslo"}'
    assert not any(c["delta"].get("tool_calls") for c in choices)
    assert choices[-1]["finish_reason"] == "function_call"


def test_a_function_history_is_carried_as_one_tool_call_and_its_result(settings: Settings) -> None:
    fake = Rich(name="rich", model_id="m", supports_tools=True)
    history = [
        {"role": "user", "content": "Weather in Oslo?"},
        {
            "role": "assistant",
            "content": None,
            "function_call": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
        },
        {"role": "function", "name": "get_weather", "content": '{"temp": -3}'},
    ]
    with serve(settings, fake) as client:
        response = client.post(
            "/v1/chat/completions", json={"model": "m", "messages": history, "functions": [WEATHER]}
        )
    assert response.status_code == 200, response.text
    messages = fake.calls[-1].messages
    assert messages[1].toolCalls[0]["id"] == "call_function_1"
    assert messages[1].toolCalls[0]["function"]["name"] == "get_weather"
    assert messages[2].role.value == "tool" and messages[2].toolCallId == "call_function_1"


@pytest.mark.parametrize(
    ("extra", "param"),
    [
        (
            {"functions": [WEATHER], "tools": [{"type": "function", "function": WEATHER}]},
            "functions",
        ),
        ({"functions": [WEATHER], "tool_choice": "auto"}, "tool_choice"),
        ({"function_call": "auto"}, "function_call"),
        ({"messages": [{"role": "function", "name": "f", "content": "x"}]}, "messages[0]"),
    ],
)
def test_what_the_old_shape_cannot_mean_is_refused(
    settings: Settings, extra: dict, param: str
) -> None:
    fake = Rich(name="rich", model_id="m")
    with serve(settings, fake) as client:
        response = client.post("/v1/chat/completions", json=chat(**extra))
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == param
    assert not fake.calls


# --------------------------------------------------------------------------- #
# /v1/responses/input_tokens
# --------------------------------------------------------------------------- #


def test_input_tokens_are_counted_by_the_backend(settings: Settings) -> None:
    fake = Rich(name="rich", model_id="m")
    fake.prompt_tokens = 42
    with serve(settings, fake) as client:
        response = client.post("/v1/responses/input_tokens", json={"model": "m", "input": "Hello"})
    assert response.status_code == 200, response.text
    assert response.json() == {"object": "response.input_tokens", "input_tokens": 42}
    assert fake.count_calls and not fake.calls


def test_a_count_that_cannot_be_given_is_a_400_saying_why(settings: Settings) -> None:
    fake = Rich(name="rich", model_id="m")
    with serve(settings, fake) as client:
        response = client.post("/v1/responses/input_tokens", json={"model": "m", "input": "Hello"})
    assert response.status_code == 400, response.text
    assert "without generating" in response.json()["error"]["message"]
    assert not fake.calls
