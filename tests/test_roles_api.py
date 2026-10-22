"""Role assignments and advisory capability findings."""

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.models import ModelRole
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table


def test_roles_list_every_role_with_its_models_and_capability(settings: Settings) -> None:
    yes = FakeDriverClient(name="yes", model_id="tools", supports_tools=True)
    no = FakeDriverClient(name="no", model_id="plain", supports_tools=False)
    unknown = FakeDriverClient(name="unknown", model_id="old", capabilities_reported=False)
    slots = [
        {"model": "role:agentic", "targets": ["tools", "plain", "old", "missing"]},
        {"model": "coder", "targets": ["plain"]},
    ]
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(yes, no, unknown, slots=slots)
    with TestClient(app) as client:
        # Findings never prevent assigning even an untested or explicitly negative model.
        assert client.patch("/v1/config", json={"modelSlots": slots}).json()["rejected"] == []
        response = client.get("/v1/roles")
    assert response.status_code == 200, response.text
    roles = response.json()["roles"]
    assert [r["role"] for r in roles] == [r.value for r in ModelRole]
    agentic = next(r for r in roles if r["role"] == "agentic")
    assert agentic["label"] == "Agentic (tools and code)"
    assert agentic["requiredCapability"] == "toolCalling"
    assert [m["model"] for m in agentic["models"]] == ["tools", "plain", "old", "missing"]
    assert [m["capability"] for m in agentic["models"]] == ["yes", "no", "unknown", "unknown"]
    assert agentic["models"][0]["reason"] is None
    assert "toolCalling" in agentic["models"][1]["reason"]
    assert all(m["reason"] for m in agentic["models"][1:])
    assert all(r["models"] == [] for r in roles if r["role"] != "agentic")


@pytest.mark.parametrize(
    ("role", "surface", "capability"),
    [
        ("chat", "chat", None),
        ("agentic", "chat", "toolCalling"),
        ("vision", "chat", "imageInput"),
        ("reads-pages", "chat", "imageInput"),
        ("image", "image", None),
        ("speech-to-text", "transcription", None),
        ("text-to-speech", "speech", None),
        ("music", "audio", "audioOutput"),
        ("embeddings", "embeddings", None),
        ("rerank", "rerank", None),
        ("decision", "decisions", None),
    ],
)
def test_each_role_reads_its_required_capability(
    settings: Settings, role: str, surface: str, capability: str | None
) -> None:
    fake = FakeDriverClient(name="model", model_id="model", surfaces=[surface])
    table = make_routing_table(fake, slots=[{"model": f"role:{role}", "targets": ["model"]}])
    backend = table.resolve("model").backends()[0]
    if capability:
        assert backend.caps is not None
        setattr(backend.caps, capability, True)
    app = create_app(settings=settings)
    app.state.routing = table
    with TestClient(app) as client:
        roles = client.get("/v1/roles").json()["roles"]
    assigned = next(r for r in roles if r["role"] == role)
    assert assigned["models"] == [{"model": "model", "capability": "yes", "reason": None}]
