"""The OpenAI-compatible front door.

What matters here is wire fidelity: an unmodified OpenAI SDK has to be
able to point `base_url` at us and work. So these tests assert on
snake_case field names, OpenAI's error envelope, and OpenAI's SSE
framing — not on whatever shape happened to be convenient.
"""

from __future__ import annotations

import json

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import BackendKind, Usage
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import DriverError
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

MODEL = "Qwen3-30B-A3B-Q4_K_M"


def _app_with(settings: Settings, *fakes: FakeDriverClient) -> FastAPI:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*fakes)
    return app


def _chat(model: str = MODEL, **extra: object) -> dict[str, object]:
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    }
    body.update(extra)
    return body


# --------------------------------------------------------------------------- #
# /v1/models
# --------------------------------------------------------------------------- #


def test_models_reports_openai_shape(client: TestClient, fake_driver: FakeDriverClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1

    model = body["data"][0]
    assert model["id"] == fake_driver.model_id
    assert model["object"] == "model"
    # snake_case, because this is OpenAI's field name, not ours.
    assert "owned_by" in model


def test_two_drivers_serving_one_model_produce_one_entry(settings: Settings) -> None:
    """Replicas are a routing detail. A client should not have to know
    how many GPUs are behind a name."""
    a = FakeDriverClient(name="gpu-a", base_url="http://a", model_id=MODEL)
    b = FakeDriverClient(name="gpu-b", base_url="http://b", model_id=MODEL)
    with TestClient(_app_with(settings, a, b)) as c:
        body = c.get("/v1/models").json()

    assert [m["id"] for m in body["data"]] == [MODEL]
    # ...but the extension says both are behind it, so an operator can see
    # the replication that a client can't.
    assert body["data"][0]["x_eugene_plexus"]["drivers"] == ["gpu-a", "gpu-b"]


def test_model_list_reports_the_smallest_context_across_replicas(
    settings: Settings,
) -> None:
    """The honest number: a request may land on either, so promising the
    larger window would let a prompt that "fits" still be rejected."""
    a = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL, max_context_tokens=8192)
    b = FakeDriverClient(name="b", base_url="http://b", model_id=MODEL, max_context_tokens=4096)
    with TestClient(_app_with(settings, a, b)) as c:
        body = c.get("/v1/models").json()
    assert body["data"][0]["x_eugene_plexus"]["context_length"] == 4096


def test_a_driver_with_no_model_id_is_not_routable(settings: Settings) -> None:
    """No model id means no key to route on. It still shows up on the
    admin surface, but it cannot appear in the model list."""
    anon = FakeDriverClient(name="degraded", base_url="http://x", model_id=None)
    with TestClient(_app_with(settings, anon)) as c:
        assert c.get("/v1/models").json()["data"] == []
        assert c.get("/v1/admin/drivers").json()["drivers"][0]["name"] == "degraded"


# --------------------------------------------------------------------------- #
# /v1/chat/completions — the happy path
# --------------------------------------------------------------------------- #


def test_chat_completion_returns_openai_shape(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    fake_driver.responses = ["the model's answer"]
    fake_driver.usage = Usage(promptTokens=11, completionTokens=4, totalTokens=15)

    response = client.post("/v1/chat/completions", json=_chat())
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["model"] == MODEL
    assert body["choices"][0]["index"] == 0
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "the model's answer"
    assert body["choices"][0]["finish_reason"] == "stop"
    # snake_case usage keys, per OpenAI.
    assert body["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
    }


def test_gateway_fills_in_the_generation_params(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    """The gateway owns every output-affecting parameter. A driver never
    substitutes a default of its own, so an omitted value must be filled
    in here and sent explicitly — not left unset for the backend to
    decide."""
    client.post("/v1/chat/completions", json=_chat())

    sent = fake_driver.calls[-1]
    assert sent.temperature == 0.7
    assert sent.maxTokens == 2048


def test_caller_params_win_over_the_defaults(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    client.post(
        "/v1/chat/completions",
        json=_chat(temperature=0.1, max_tokens=7, stop=["\n\n"]),
    )
    sent = fake_driver.calls[-1]
    assert sent.temperature == 0.1
    assert sent.maxTokens == 7
    assert sent.stop == ["\n\n"]


def test_messages_pass_through_untouched(client: TestClient, fake_driver: FakeDriverClient) -> None:
    """The gateway is not a prompt injector. Whatever system message the
    caller wants is already in the request; nothing gets prepended."""
    client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
        },
    )
    sent = fake_driver.calls[-1]
    assert [(m.role.value, m.content) for m in sent.messages] == [
        ("system", "be terse"),
        ("user", "hi"),
    ]


def test_response_reports_which_backend_served_it(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    """The failure mode of a routing layer is opacity."""
    response = client.post("/v1/chat/completions", json=_chat())
    info = response.json()["x_eugene_plexus"]
    assert info["driver"] == fake_driver.name
    assert info["backend"] == BackendKind.openai_compat_http.value
    assert info["attempts"] == 1
    assert info["latency_ms"] >= 0


# --------------------------------------------------------------------------- #
# Failover
# --------------------------------------------------------------------------- #


def test_a_dead_replica_cascades_to_the_next(settings: Settings) -> None:
    """Two drivers serving one model are automatically a priority list —
    failover falls out of the topology with nothing configured."""
    dead = FakeDriverClient(name="a-dead", base_url="http://a", model_id=MODEL)
    dead.generate_error = httpx.ConnectError("connection refused")
    alive = FakeDriverClient(name="b-alive", base_url="http://b", model_id=MODEL)
    alive.responses = ["from the backup"]

    with TestClient(_app_with(settings, dead, alive)) as c:
        response = c.post("/v1/chat/completions", json=_chat())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "from the backup"
    # attempts > 1 is the visible evidence the cascade fired.
    assert body["x_eugene_plexus"]["attempts"] == 2
    assert body["x_eugene_plexus"]["driver"] == "b-alive"


def test_a_4xx_does_not_cascade(settings: Settings) -> None:
    """A 4xx is the same bad request everywhere. Cascading past it would
    bury the real problem — an expired token reading as "all backends
    down" instead of "fix your token"."""
    first = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL)
    first.generate_error = DriverError(
        driver_name="a",
        driver_url="http://a",
        status_code=401,
        problem=None,
        raw_body="bad token",
    )
    second = FakeDriverClient(name="b", base_url="http://b", model_id=MODEL)
    second.responses = ["should never be reached"]

    with TestClient(_app_with(settings, first, second)) as c:
        response = c.post("/v1/chat/completions", json=_chat())

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert second.calls == [], "a 4xx must not fall through to the next backend"


def test_every_backend_failing_is_a_502(settings: Settings) -> None:
    a = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL)
    a.generate_error = httpx.ConnectError("refused")
    b = FakeDriverClient(name="b", base_url="http://b", model_id=MODEL)
    b.generate_error = httpx.ConnectError("refused")

    with TestClient(_app_with(settings, a, b)) as c:
        response = c.post("/v1/chat/completions", json=_chat())

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"


# --------------------------------------------------------------------------- #
# Errors — OpenAI's envelope, not problem+json
# --------------------------------------------------------------------------- #


def test_unknown_model_is_a_404_naming_what_is_available(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json=_chat(model="not-a-model"))
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "model"
    # The person reading this configured the thing — tell them what IS here.
    assert MODEL in error["message"]


def test_no_models_at_all_says_where_to_look(settings: Settings) -> None:
    with TestClient(_app_with(settings)) as c:
        response = c.post("/v1/chat/completions", json=_chat())
    assert response.status_code == 404
    assert "/v1/runtimes" in response.json()["error"]["message"]


def test_errors_use_openai_envelope_not_problem_json(client: TestClient) -> None:
    """OpenAI SDKs parse this shape to build their exceptions. A
    problem+json body reads to them as an unhelpful generic failure."""
    response = client.post("/v1/chat/completions", json=_chat(model="nope"))
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"message", "type", "param", "code"}
    # problem+json's discriminators must NOT be here.
    assert "title" not in body
    assert "detail" not in body


def test_a_backend_still_loading_is_a_retryable_503(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    """Deliberately not folded into the 502: "still loading its weights"
    clears on its own, and a client should retry rather than give up."""
    fake_driver.generate_error = DriverError(
        driver_name=fake_driver.name,
        driver_url="http://a",
        status_code=503,
        problem=None,
        raw_body="engine loading",
    )
    response = client.post("/v1/chat/completions", json=_chat())
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "service_unavailable"


def test_a_malformed_request_is_rejected_before_routing(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json={"model": MODEL})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Streaming — OpenAI's SSE framing
# --------------------------------------------------------------------------- #


def _sse_frames(text: str) -> list[str]:
    return [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]


def test_streaming_uses_openai_framing(client: TestClient, fake_driver: FakeDriverClient) -> None:
    fake_driver.responses = ["streamed answer"]
    response = client.post("/v1/chat/completions", json=_chat(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = _sse_frames(response.text)
    assert frames[-1] == "[DONE]", "OpenAI clients terminate on this exact sentinel"

    chunks = [json.loads(f) for f in frames[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    # First chunk carries the role, per OpenAI.
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "streamed answer"
    # Terminal chunk carries the finish reason.
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_streaming_ids_are_stable_across_chunks(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    fake_driver.responses = ["abc"]
    response = client.post("/v1/chat/completions", json=_chat(stream=True))
    chunks = [json.loads(f) for f in _sse_frames(response.text)[:-1]]
    assert len({c["id"] for c in chunks}) == 1


def test_a_failure_mid_stream_emits_an_error_frame(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    """The 200 is already sent, so this cannot become an HTTP status. An
    error frame followed by [DONE] is what OpenAI does."""
    fake_driver.generate_error = httpx.ConnectError("refused")
    response = client.post("/v1/chat/completions", json=_chat(stream=True))

    assert response.status_code == 200
    frames = _sse_frames(response.text)
    assert frames[-1] == "[DONE]"
    assert json.loads(frames[0])["error"]["type"] == "upstream_error"
