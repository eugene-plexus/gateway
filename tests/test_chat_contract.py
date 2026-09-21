"""A2: validate the public request before defaults, routing or generation."""

import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway.driver_client import HttpDriverClient

from .conftest import FakeDriverClient
from .test_profile_defaults import Library, setup_routes


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_limit_on_driver_wire(app: FastAPI, field: str, stream: bool, fallback: bool) -> None:
    library = Library()
    library.values["/models/a.gguf"]["maxTokens"] = 2048
    library.values["/models/b.gguf"]["maxTokens"] = 4096
    first = FakeDriverClient(name="a", model_id="first", runtime="a")
    second = FakeDriverClient(name="b", model_id="second", runtime="b")
    profiles = setup_routes(app, library, first, second)
    table = app.state.routing
    table._slots = lambda: [{"model": "alias", "targets": ["first", "second"]}]
    captured: list[tuple[str, dict[str, Any]]] = []

    async def wire(request: httpx.Request) -> httpx.Response:
        captured.append((request.url.host, json.loads(request.content)))
        if fallback and request.url.host == "first.invalid":
            return httpx.Response(503, json={"error": "unavailable"})
        result = {"content": "ok", "finishReason": "stop", "backend": "openai_compat_http"}
        if request.url.path.endswith("/stream"):
            return httpx.Response(
                200,
                text='event: token\ndata: {"text":"ok"}\n\nevent: done\ndata: '
                + json.dumps(result)
                + "\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=result)

    clients = []
    for backend in table.resolve("alias").backends():
        client = HttpDriverClient(
            name=backend.name, base_url=f"http://{backend.info.modelId}.invalid"
        )
        client._client = httpx.AsyncClient(
            base_url=client.base_url, transport=httpx.MockTransport(wire)
        )
        backend.client = client
        clients.append(client)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "alias",
                "messages": [{"role": "user", "content": "hello"}],
                field: 25,
                "stream": stream,
            },
        )
        assert response.status_code == 200, response.text
        assert "ok" in response.text
        assert len(captured) == (2 if fallback else 1)
        assert all(payload["maxTokens"] == 25 for _, payload in captured), captured
        assert all(payload["callerSettings"] == ["maxTokens"] for _, payload in captured)
        client.portal.call(profiles.aclose)
        for driver in clients:
            client.portal.call(driver.aclose)


def body(**extra: Any) -> dict[str, Any]:
    return {
        "model": "Qwen3-30B-A3B-Q4_K_M",
        "messages": [{"role": "user", "content": "PRIVATE PROMPT SENTINEL"}],
        **extra,
    }


@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("value", [0, -1, True, 2.5, 25.0, "25", {}, []])
def test_invalid_limit_is_safe_400(
    client: TestClient, fake_driver: FakeDriverClient, field: str, value: Any
) -> None:
    response = client.post("/v1/chat/completions", json=body(**{field: value}))
    assert response.status_code == 400
    assert response.json()["error"]["param"] == field
    assert "PRIVATE PROMPT" not in response.text
    assert not fake_driver.calls


@pytest.mark.parametrize(
    "limits,expected",
    [
        ({"max_tokens": 25, "max_completion_tokens": 25}, 25),
        ({"max_tokens": None, "max_completion_tokens": 25}, 25),
        ({"max_tokens": 25, "max_completion_tokens": None}, 25),
        ({"max_tokens": None, "max_completion_tokens": None}, 2048),
        ({}, 2048),
    ],
)
def test_limit_resolution(
    client: TestClient, fake_driver: FakeDriverClient, limits: dict[str, Any], expected: int
) -> None:
    response = client.post("/v1/chat/completions", json=body(**limits))
    assert response.status_code == 200, response.text
    assert fake_driver.calls[-1].maxTokens == expected


def test_conflicting_limits(client: TestClient, fake_driver: FakeDriverClient) -> None:
    response = client.post(
        "/v1/chat/completions", json=body(max_tokens=25, max_completion_tokens=26)
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "max_completion_tokens"
    assert not fake_driver.calls


@pytest.mark.parametrize(
    "extra,param",
    [
        ({"reasoning_effort": "high"}, "reasoning_effort"),
        ({"frequency_penalty": 1}, "frequency_penalty"),
        ({"logit_bias": {"1": -100}}, "logit_bias"),
        ({"parallel_tool_calls": False}, "parallel_tool_calls"),
        ({"n": 2}, "n"),
        ({"logprobs": True}, "logprobs"),
        ({"store": True}, "store"),
        ({"model_misspelled": None}, "model_misspelled"),
        ({"response_format": {"type": "json_schema"}}, "response_format.json_schema"),
        ({"response_format": {"type": "text", "extra": "SECRET"}}, "response_format.extra"),
        ({"messages": [{"role": "user", "content": "SECRET", "audio": {}}]}, "messages[0].audio"),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "https://SECRET/image"}}
                        ],
                    }
                ]
            },
            "messages[0].content[0].image_url",
        ),
    ],
)
def test_consequential_unsupported_settings(
    client: TestClient, fake_driver: FakeDriverClient, extra: dict[str, Any], param: str
) -> None:
    response = client.post("/v1/chat/completions", json=body(**extra))
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"].startswith(param)
    assert "SECRET" not in response.text
    assert "PRIVATE PROMPT" not in response.text
    assert not fake_driver.calls


def test_metadata_neutral_defaults_and_string_stop(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json=body(
            user="user-1",
            metadata={"client": "test"},
            safety_identifier="opaque",
            n=1,
            logprobs=False,
            store=False,
            stop="STOP",
        ),
    )
    assert response.status_code == 200, response.text
    forwarded = fake_driver.calls[-1].model_dump(exclude_none=True)
    assert forwarded["stop"] == ["STOP"]
    assert not ({"user", "metadata", "safety_identifier"} & forwarded.keys())


@pytest.mark.parametrize("content", [b'{"messages": "PRIVATE PROMPT",', b"[]", b"null", b"\xff"])
def test_malformed_body_does_not_echo_input(client: TestClient, content: bytes) -> None:
    response = client.post(
        "/v1/chat/completions", content=content, headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "body"
    assert "PRIVATE PROMPT" not in response.text


def test_structured_output_schema_is_preserved(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"reasoning_effort": {"type": "string"}},
        "required": ["reasoning_effort"],
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "answer", "strict": True, "schema": schema},
    }
    response = client.post("/v1/chat/completions", json=body(response_format=response_format))
    assert response.status_code == 200, response.text
    assert (
        fake_driver.calls[-1].responseFormat.model_dump(by_alias=True, exclude_none=True)
        == response_format
    )


@pytest.mark.parametrize("include", [True, False])
def test_stream_usage_option(
    client: TestClient, fake_driver: FakeDriverClient, include: bool
) -> None:
    from eugene_plexus_gateway._generated.driver_models import Usage

    fake_driver.usage = Usage(promptTokens=3, completionTokens=2, totalTokens=5)
    response = client.post(
        "/v1/chat/completions", json=body(stream=True, stream_options={"include_usage": include})
    )
    assert response.status_code == 200, response.text
    frames = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    usage_frames = [frame for frame in frames if "usage" in frame]
    assert len(usage_frames) == int(include)
    if include:
        assert usage_frames[0]["choices"] == []
        assert usage_frames[0]["usage"]["total_tokens"] == 5
    assert frames[-2 if include else -1]["choices"][0]["finish_reason"] == "stop"


def test_request_schema_is_available(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()["paths"]["/v1/chat/completions"]["post"][
        "requestBody"
    ]["content"]["application/json"]["schema"]
    assert "max_completion_tokens" in schema["properties"]


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_compatibility_hints_are_disclosed(client: TestClient, stream: bool) -> None:
    response = client.post(
        "/v1/messages",
        json=body(
            max_tokens=25,
            stream=stream,
            thinking={"type": "adaptive"},
            context_management={"edits": []},
            system=[{"type": "text", "text": "SECRET", "cache_control": {"type": "ephemeral"}}],
        ),
    )
    assert response.status_code == 200, response.text
    assert (
        response.headers["x-eugene-plexus-ignored-settings"]
        == "thinking, context_management, cache_control"
    )


@pytest.mark.parametrize(
    "extra", [{"top_k": 5}, {"unknown_setting": "SECRET"}, {"max_tokens": True}]
)
def test_anthropic_unsupported_settings_are_explicit(
    client: TestClient, extra: dict[str, Any]
) -> None:
    response = client.post("/v1/messages", json=body(**{"max_tokens": 25, **extra}))
    assert response.status_code == 400
    assert next(iter(extra)) in response.json()["error"]["message"]
    assert "SECRET" not in response.text
