"""Row 18: each conversational door names a nested invalid field."""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.parametrize(
    "door,field",
    [
        ("/v1/chat/completions", "messages"),
        ("/v1/messages", "messages"),
        ("/v1/messages/count_tokens", "messages"),
        ("/v1/responses", "input"),
        ("/v1/responses/input_tokens", "input"),
    ],
)
def test_nested_400_names_the_callers_field(client: TestClient, door: str, field: str) -> None:
    content_type = "input_text" if field == "input" else "text"
    messages = [{"role": "user", "content": "hello"} for _ in range(3)]
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": content_type, "text": "ok"},
                {"type": content_type, "text": {"private": "do not echo this"}},
            ],
        }
    )
    body = {"model": "fake-model", field: messages}
    if field == "messages" and "chat" not in door:
        body["max_tokens"] = 32
    response = client.post(door, json=body)
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    normalized = message.replace("[", ".").replace("]", "")
    assert f"{field}.3.content.1.text" in normalized
    assert "do not echo this" not in message
    assert "TextBlock" not in message and ".str" not in message
