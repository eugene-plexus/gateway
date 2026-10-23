from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import FunctionCall, ToolCall, Usage
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings
from tests.test_admission import Authority

from .conftest import FakeDriverClient, FakeInstall, make_routing_table


def ask(**extra: Any) -> dict[str, Any]:
    return {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}], **extra}


@pytest.fixture
def door(settings: Settings):
    driver = FakeDriverClient(name="fixture", model_id="fixture/chat", supports_tools=True)
    driver.prompt_tokens = 17
    driver.supported_settings.append("reasoningEffort")
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(
        driver, slots=[{"model": "role:chat", "targets": ["fixture/chat"]}]
    )
    with TestClient(app) as client:
        yield client, driver


@pytest.mark.parametrize("version", ["v1beta", "v1"])
def test_generate_content_round_trips_text_and_usage(door, version: str) -> None:
    client, driver = door
    driver.responses = ["Hello back"]
    driver.usage = Usage(
        promptTokens=10, completionTokens=7, totalTokens=17, reasoningTokens=3, cachedPromptTokens=2
    )
    response = client.post(f"/{version}/models/fixture/chat:generateContent", json=ask())
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["candidates"][0] == {
        "index": 0,
        "content": {"role": "model", "parts": [{"text": "Hello back"}]},
        "finishReason": "STOP",
    }
    assert result["usageMetadata"] == {
        "promptTokenCount": 10,
        "candidatesTokenCount": 4,
        "thoughtsTokenCount": 3,
        "cachedContentTokenCount": 2,
        "totalTokenCount": 17,
    }
    assert result["modelVersion"] == "fixture/chat"
    assert response.headers["x-eugene-plexus-model"] == "fixture/chat"
    assert driver.calls[0].messages[0].content == "Hello"
    assert len(client.get("/v1/metrics/requests").json()["requests"]) == 1


def test_unknown_names_use_chat_role_and_mapping_is_live_in_gui(door) -> None:
    client, driver = door
    response = client.post("/v1beta/models/gemini-3.1-pro-preview:generateContent", json=ask())
    assert response.status_code == 200, response.text
    assert response.json()["modelVersion"] == "fixture/chat"
    value = [{"model": "gemini-special", "targets": ["fixture/chat"]}]
    patched = client.patch("/v1/config", json={"geminiModelNames": value})
    assert patched.status_code == 200, patched.text
    schema = client.get("/v1/config/schema").json()
    field = next(f for f in schema["fields"] if f["key"] == "geminiModelNames")
    assert field["label"] == "Gemini model names" and field["valueType"] == "model_slots"
    assert client.get("/v1/config").json()["geminiModelNames"] == value
    assert (
        client.post("/v1beta/models/gemini-special:generateContent", json=ask()).status_code == 200
    )
    assert len(driver.calls) == 2
    bad = client.patch(
        "/v1/config", json={"geminiModelNames": [{"model": "x", "targets": ["a", "b"]}]}
    )
    assert bad.json()["rejected"][0]["key"] == "geminiModelNames"


def test_function_calling_round_trip(door) -> None:
    client, driver = door
    driver.tool_calls = [
        ToolCall(
            id="call-1",
            type="function",
            function=FunctionCall(name="weather", arguments='{"city":"Eugene"}'),
        )
    ]
    tools = [
        {
            "functionDeclarations": [
                {
                    "name": "weather",
                    "parameters_json_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ]
        }
    ]
    first = client.post("/v1beta/models/fixture/chat:generateContent", json=ask(tools=tools))
    assert first.status_code == 200, first.text
    called = first.json()["candidates"][0]["content"]
    assert called["parts"][0]["functionCall"] == {
        "id": "call-1",
        "name": "weather",
        "args": {"city": "Eugene"},
    }
    driver.tool_calls = None
    response = client.post(
        "/v1beta/models/fixture/chat:generateContent",
        json={
            "tools": tools,
            "contents": [
                *ask()["contents"],
                called,
                {
                    "role": "user",
                    "parts": [
                        {"functionResponse": {"name": "weather", "response": {"temperature": 18}}}
                    ],
                },
            ],
        },
    )
    assert response.status_code == 200, response.text
    result = driver.calls[-1].messages[-1]
    assert result.toolCallId == "call-1" and json.loads(result.content) == {"temperature": 18}


def test_stream_is_sse_of_responses(door) -> None:
    client, driver = door
    driver.responses = ["Hi from stream"]
    response = client.post("/v1beta/models/fixture/chat:streamGenerateContent?alt=sse", json=ask())
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert (
        "".join(
            p["text"]
            for f in frames
            for c in f.get("candidates", [])
            for p in c.get("content", {}).get("parts", [])
        )
        == "Hi from stream"
    )
    assert frames[-1]["candidates"][0]["finishReason"] == "STOP"
    assert "event:" not in response.text and "[DONE]" not in response.text
    assert len(client.get("/v1/metrics/requests").json()["requests"]) == 1


def test_streamed_function_arguments_are_complete_json(door) -> None:
    client, driver = door
    driver.tool_calls = [
        ToolCall(
            id="c", type="function", function=FunctionCall(name="f", arguments='{"value": 42}')
        )
    ]
    response = client.post("/v1beta/models/fixture/chat:streamGenerateContent?alt=sse", json=ask())
    frames = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    calls = [
        p["functionCall"]
        for f in frames
        for c in f.get("candidates", [])
        for p in c.get("content", {}).get("parts", [])
        if "functionCall" in p
    ]
    assert calls == [{"id": "c", "name": "f", "args": {"value": 42}}]


def test_candidate_count_two_is_refused_naming_it(door) -> None:
    client, driver = door
    response = client.post(
        "/v1beta/models/fixture/chat:generateContent",
        json=ask(generationConfig={"candidateCount": 2}),
    )
    assert response.status_code == 400
    assert "candidateCount" in response.json()["error"]["message"]
    assert response.json()["error"]["status"] == "INVALID_ARGUMENT"
    assert not driver.calls


@pytest.mark.parametrize(
    "thinking",
    [
        {"thinkingLevel": "HIGH", "includeThoughts": True},
        {"thinking_level": "HIGH", "include_thoughts": True},
        {"thinkingBudget": 24000, "includeThoughts": True},
    ],
)
def test_structured_output_thinking_and_safety_are_accounted_for(door, thinking: dict) -> None:
    client, driver = door
    driver.reasoning = "Let me think"
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    response = client.post(
        "/v1beta/models/fixture/chat:generateContent",
        json=ask(
            systemInstruction={"parts": [{"text": "Be brief"}]},
            safetySettings=[{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"}],
            generationConfig={
                "thinkingConfig": thinking,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
                "maxOutputTokens": 32,
                "topK": 4,
            },
        ),
    )
    assert response.status_code == 200, response.text
    call = driver.calls[0]
    assert (
        call.reasoningEffort.value == "high" and call.responseFormat.json_schema.schema_ == schema
    )
    assert call.messages[0].content == "Be brief" and call.maxTokens == 32 and call.topK == 4
    assert response.json()["candidates"][0]["content"]["parts"][0] == {
        "text": "Let me think",
        "thought": True,
    }
    assert "safetySettings" in response.headers["x-eugene-plexus-ignored-settings"]
    if "thinkingBudget" in thinking:
        assert (
            "thinkingConfig.thinkingBudget" in response.headers["x-eugene-plexus-ignored-settings"]
        )


def test_count_tokens_and_model_discovery(door) -> None:
    client, driver = door
    for version in ("v1beta", "v1"):
        counted = client.post(f"/{version}/models/fixture/chat:countTokens", json=ask())
        assert counted.status_code == 200, counted.text
        assert counted.json() == {"totalTokens": 17}
        assert counted.headers["x-eugene-plexus-driver"] == "fixture"
        listed = client.get(f"/{version}/models").json()
        assert listed["models"][0]["name"] == "models/fixture/chat"
        found = client.get(f"/{version}/models/fixture/chat").json()
        assert found["name"] == listed["models"][0]["name"]
    assert len(driver.count_calls) == 2 and not driver.calls
    listed = client.get("/v1/models").json()
    assert listed["object"] == "list" and listed["data"][0]["id"] == "fixture/chat"


def test_count_tokens_refuses_a_non_string_nested_model(door) -> None:
    client, driver = door
    response = client.post(
        "/v1beta/models/fixture/chat:countTokens",
        json={"generateContentRequest": {"model": 42, **ask()}},
    )
    assert response.status_code == 400, response.text
    assert "generateContentRequest.model" in response.json()["error"]["message"]
    assert not driver.count_calls


def test_empty_query_key_on_shared_catalogue_uses_google_error(
    settings: Settings, install: FakeInstall
) -> None:
    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    app.state.routing = make_routing_table(FakeDriverClient(name="d", model_id="chat"))
    with TestClient(app) as client:
        response = client.get("/v1/models?key=")
        assert response.status_code == 401
        assert response.json()["error"]["status"] == "UNAUTHENTICATED"


def test_inline_attachments_translate_and_gemini_can_count_them(settings: Settings) -> None:
    import base64
    import io

    from PIL import Image

    from eugene_plexus_gateway._generated.driver_models import BackendKind
    from tests.test_attachments import PDF, WAV, Takes

    picture = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(picture, format="PNG")
    driver = Takes("image", "audio", "file", name="gem", model_id="fixture/chat")
    driver.backend = BackendKind.gemini_api
    driver.prompt_tokens = 99
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(driver)
    parts = [
        {"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode()}}
        for mime, data in (
            ("image/png", picture.getvalue()),
            ("audio/wav", WAV),
            ("application/pdf", PDF),
        )
    ]
    body = {"contents": [{"role": "user", "parts": [{"text": "Read these"}, *parts]}]}
    with TestClient(app) as client:
        response = client.post("/v1beta/models/fixture/chat:generateContent", json=body)
        assert response.status_code == 200, response.text
        counted = client.post("/v1beta/models/fixture/chat:countTokens", json=body)
        assert counted.status_code == 200 and counted.json()["totalTokens"] == 99, counted.text
        bad = client.post(
            "/v1beta/models/fixture/chat:generateContent",
            json={
                "contents": [
                    {"parts": [{"inlineData": {"mimeType": "image/png", "data": "not-base64"}}]}
                ]
            },
        )
        assert bad.status_code == 400
    sent = driver.calls[0].messages[0].content
    assert [p.type for p in sent.root] == ["text", "image_url", "input_audio", "file"]
    assert len(driver.count_calls) == 1


def test_invalid_model_mapping_file_uses_and_shows_the_default(settings: Settings) -> None:
    settings.config_file.write_text("geminiModelNames: broken\n", encoding="utf-8")
    app = create_app(settings=settings)
    driver = FakeDriverClient(name="d", model_id="fixture/chat")
    app.state.routing = make_routing_table(
        driver, slots=[{"model": "role:chat", "targets": ["fixture/chat"]}]
    )
    with TestClient(app) as client:
        assert client.get("/v1/config").json()["geminiModelNames"] == []
        response = client.post("/v1beta/models/gemini-name:generateContent", json=ask())
        assert response.status_code == 200, response.text


def test_scoped_key_cannot_reach_an_excluded_model_through_a_gemini_alias(
    settings: Settings, install: FakeInstall
) -> None:
    authority = Authority()
    authority.allowed = ["allowed"]
    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    allowed = FakeDriverClient(name="yes", model_id="allowed")
    excluded = FakeDriverClient(name="no", model_id="excluded")
    app.state.routing = make_routing_table(
        allowed, excluded, slots=[{"model": "role:chat", "targets": ["excluded"]}]
    )
    headers = {"x-goog-api-key": install.client_key(name="app", jti="key-1")}
    with TestClient(app) as client:
        for method in ("generateContent", "countTokens"):
            response = client.post(
                f"/v1beta/models/gemini-unknown:{method}", json=ask(), headers=headers
            )
            assert response.status_code in (400, 403, 404), response.text
            assert response.json()["error"]["status"] in (
                "INVALID_ARGUMENT",
                "PERMISSION_DENIED",
                "NOT_FOUND",
            )
    assert not excluded.calls and not excluded.count_calls


def test_count_tokens_uses_a_key_check_without_acquiring_a_reservation(
    settings: Settings, install: FakeInstall
) -> None:
    authority = Authority()
    authority.allowed = ["allowed"]
    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    driver = FakeDriverClient(name="yes", model_id="allowed")
    driver.prompt_tokens = 9
    app.state.routing = make_routing_table(driver)
    with TestClient(app) as client:
        response = client.post(
            "/v1beta/models/allowed:countTokens",
            json=ask(),
            headers={"x-goog-api-key": install.client_key(name="app", jti="key-1")},
        )
        assert response.status_code == 200, response.text
    assert [c["action"] for c in authority.calls] == ["check"]


def test_stream_failure_names_observed_cause_without_a_finish_reason(door) -> None:
    import httpx

    client, driver = door
    driver.stream_error_after = 1
    driver.stream_error = httpx.ReadError("fixture stream broke")
    response = client.post("/v1beta/models/fixture/chat:streamGenerateContent?alt=sse", json=ask())
    frames = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert "fixture stream broke" in frames[-1]["error"]["message"]
    assert frames[-1]["error"]["status"] == "INTERNAL"
    assert all("finishReason" not in c for f in frames for c in f.get("candidates", []))


def test_key_in_query_is_checked_and_never_logged(
    settings: Settings, install: FakeInstall, caplog
) -> None:
    caplog.set_level(logging.INFO)
    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    authority = Authority()
    authority.allowed = ["fixture/chat"]
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    driver = FakeDriverClient(name="d", model_id="fixture/chat")
    app.state.routing = make_routing_table(driver)
    key = install.client_key(name="app", jti="key-1")
    with TestClient(app) as client:
        response = client.post(f"/v1beta/models/fixture/chat:generateContent?key={key}", json=ask())
        assert response.status_code == 200, response.text
        rows = client.get(
            "/v1/metrics/requests", headers={"authorization": f"Bearer {install.session()}"}
        ).text
        denied = client.post(
            "/v1beta/models/fixture/chat:generateContent?key=invalid-key-secret", json=ask()
        )
        assert denied.status_code == 401 and denied.json()["error"]["status"] == "UNAUTHENTICATED"
    assert key not in caplog.text and "invalid-key-secret" not in caplog.text and key not in rows
    assert any(c["action"] == "acquire" for c in authority.calls)


def test_models_pages_and_v1_projections_are_the_same(door) -> None:
    client, _ = door
    for version in ("v1beta", "v1"):
        first = client.get(f"/{version}/models?pageSize=1")
        assert first.status_code == 200, first.text
        page = first.json()
        assert len(page["models"]) == 1 and page["nextPageToken"]
        second = client.get(
            f"/{version}/models", params={"pageSize": 1, "pageToken": page["nextPageToken"]}
        ).json()
        assert len(second["models"]) == 1 and not second.get("nextPageToken")
        assert page["models"][0]["name"] != second["models"][0]["name"]
        if version == "v1":
            assert page["models"][0]["name"] == page["data"][0]["name"]
    bad = client.get("/v1beta/models?pageToken=invalid")
    assert bad.status_code == 400 and "pageToken" in bad.json()["error"]["message"]


def test_shared_catalogue_keeps_openai_missing_model_errors(door) -> None:
    client, _ = door
    response = client.get("/v1/models/not-here")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_parallel_function_results_match_in_call_order(door) -> None:
    client, driver = door
    contents = [
        {
            "role": "model",
            "parts": [
                {"functionCall": {"id": f"c{i}", "name": "f", "args": {"i": i}}} for i in range(2)
            ],
        },
        {
            "role": "user",
            "parts": [{"functionResponse": {"name": "f", "response": {"i": i}}} for i in range(2)],
        },
    ]
    response = client.post(
        "/v1beta/models/fixture/chat:generateContent", json={"contents": contents}
    )
    assert response.status_code == 200, response.text
    assert [m.toolCallId for m in driver.calls[-1].messages if m.role.value == "tool"] == [
        "c0",
        "c1",
    ]


def test_body_limit_and_key_scope_refusals_use_google_errors(door, monkeypatch) -> None:
    from eugene_plexus_gateway import body_limit

    client, driver = door
    monkeypatch.setattr(body_limit, "MAX_BODY_BYTES", 8)
    response = client.post("/v1beta/models/fixture/chat:generateContent", json=ask())
    assert response.status_code == 413 and response.json()["error"]["status"] == "INVALID_ARGUMENT"
    assert not driver.calls


def test_stream_reports_actual_model_in_headers(door) -> None:
    client, _ = door
    response = client.post(
        "/v1beta/models/gemini-3.1-pro-preview:streamGenerateContent?alt=sse", json=ask()
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-eugene-plexus-model"] == "fixture/chat"


def test_json_schema_property_names_keep_their_case(door) -> None:
    client, driver = door
    schema = {"type": "OBJECT", "properties": {"type": {"type": "STRING", "enum": ["HIGH"]}}}
    response = client.post(
        "/v1beta/models/fixture/chat:generateContent",
        json=ask(
            generationConfig={"responseMimeType": "application/json", "responseSchema": schema}
        ),
    )
    assert response.status_code == 200, response.text
    assert driver.calls[-1].responseFormat.json_schema.schema_ == {
        "type": "object",
        "properties": {"type": {"type": "string", "enum": ["HIGH"]}},
    }


@pytest.mark.parametrize(
    "extra,field",
    [
        ({"cachedContent": "x"}, "cachedContent"),
        (
            {
                "generationConfig": {
                    "thinkingConfig": {"thinkingLevel": "HIGH", "thinking_level": "LOW"}
                }
            },
            "thinking_level",
        ),
        (
            {"contents": [{"parts": [{"functionResponse": {"name": "missing", "response": {}}}]}]},
            "functionResponse",
        ),
    ],
)
def test_refusals_name_the_field(door, extra: dict, field: str) -> None:
    client, driver = door
    response = client.post("/v1beta/models/fixture/chat:generateContent", json=ask(**extra))
    assert response.status_code == 400, response.text
    assert field in response.json()["error"]["message"]
    assert not driver.calls
