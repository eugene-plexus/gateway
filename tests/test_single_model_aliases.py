"""A wrong first target is a model mismatch, not temporary unavailability."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table
from .test_moderations import Moderator
from .test_rerank import Reranker
from .test_speech import Speaker


@pytest.mark.parametrize(
    ("path", "surface", "body", "kind"),
    [
        ("/v1/embeddings", "embeddings", {"input": "text"}, "embed"),
        ("/v1/rerank", "rerank", {"query": "a", "documents": ["b"]}, "rerank"),
        ("/v1/audio/speech", "speech", {"input": "text", "voice": "a"}, "speech"),
        ("/v1/moderations", "moderation", {"input": "text"}, "moderate"),
        (
            "/v1/systemone",
            "decisions",
            {"state": "text", "questions": {"answer": {"type": "noul", "instructions": "Yes?"}}},
            "decide",
        ),
    ],
)
def test_alias_names_its_incompatible_first_target(
    settings: Settings, path: str, surface: str, body: dict[str, Any], kind: str
) -> None:
    wrong = FakeDriverClient(name="chat", model_id="chat-only")
    kw: dict[str, Any] = {"name": "other", "model_id": "right-model"}
    if kind == "embed":
        right = FakeDriverClient(**kw, supports_embeddings=True)
    elif kind == "decide":
        right = FakeDriverClient(**kw, supports_decisions=True)
    else:
        right = {"rerank": Reranker, "speech": Speaker, "moderate": Moderator}[kind](**kw)
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(
        wrong, right, slots=[{"model": "chosen", "targets": ["chat-only", "right-model"]}]
    )
    with TestClient(app) as client:
        response = client.post(path, json={"model": "chosen", **body})
    assert response.status_code == 400, response.text
    message = response.json()["error"]["message"]
    for word in ("chosen", "first target", "chat-only", surface):
        assert word in message
    assert wrong.embed_calls == 0 and right.embed_calls == 0
