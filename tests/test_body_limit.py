"""Every request-body door of the front door is bounded, not only chat.

`InferenceBodyLimit` reads at most 16 MiB before FastAPI parses a byte,
and until 2026-09-22 it was mounted on `/v1/chat/completions` and
`/v1/messages` alone. `/v1/embeddings` takes a batch of inputs and
`/v1/systemone` a state plus questions -- and the decision door reads
the body a second time, raw, to check the questions -- so both were
read to the end and parsed whole however large, on the one component a
client key can reach from another machine.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway import body_limit

from .conftest import FakeDriverClient

OPENAI_SHAPED = [
    "/v1/chat/completions",
    "/v1/embeddings",
    "/v1/rerank",
    "/v1/systemone",
    "/v1/audio/speech",
    # The counts: each reads a whole JSON body like its parent door, and a key holder
    # could post unbounded JSON to both. `/v1/messages/count_tokens` is answered in the
    # OpenAI-shaped envelope, because the middleware is vendored from specs/platform and
    # names `/v1/messages` alone for Anthropic's; the message is `error.message` in both.
    "/v1/responses/input_tokens",
    "/v1/messages/count_tokens",
]
#: The Messages door answers in Anthropic's envelope.
ANTHROPIC_SHAPED = ["/v1/messages"]
BOUNDED = [*OPENAI_SHAPED, *ANTHROPIC_SHAPED]


@pytest.mark.parametrize("path", OPENAI_SHAPED)
def test_a_chunked_body_past_the_limit_is_refused_in_the_openai_shape(
    client: TestClient,
    fake_driver: FakeDriverClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    monkeypatch.setattr(body_limit, "MAX_BODY_BYTES", 128)
    response = client.post(path, content=iter([b" " * 80, b" " * 80]))

    assert response.status_code == 413, response.text
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "body"
    assert "16 MiB" in error["message"]
    assert not fake_driver.calls


@pytest.mark.parametrize("path", ANTHROPIC_SHAPED)
def test_a_chunked_body_past_the_limit_is_refused_in_anthropics_shape(
    client: TestClient,
    fake_driver: FakeDriverClient,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    monkeypatch.setattr(body_limit, "MAX_BODY_BYTES", 128)
    response = client.post(path, content=iter([b" " * 80, b" " * 80]))

    assert response.status_code == 413, response.text
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "request_too_large"
    assert "16 MiB" in body["error"]["message"]
    assert not fake_driver.calls and not fake_driver.count_calls


@pytest.mark.parametrize("path", BOUNDED)
def test_a_declared_length_past_the_limit_is_refused_unread(client: TestClient, path: str) -> None:
    response = client.post(path, content=b"{}", headers={"content-length": "9" * 5000})
    assert response.status_code == 413, response.text
