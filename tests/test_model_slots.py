"""Reserved model-role slots use the existing routing and config protocol."""

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.models import ModelRole
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table


def test_a_role_slot_routes_like_any_slot(settings: Settings) -> None:
    first = FakeDriverClient(name="first", model_id="one")
    first.generate_error = httpx.ConnectError("first backend is down")
    second = FakeDriverClient(name="second", model_id="two")
    slots = [{"model": "role:chat", "targets": ["one", "two"]}]
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(first, second, slots=slots)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "role:chat", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200, response.text
        assert len(first.calls) == len(second.calls) == 1
        models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
        assert models["role:chat"]["x_eugene_plexus"]["role"] == {"role": "chat", "label": "Chat"}
        assert "role:reads-pages" not in models


@pytest.mark.parametrize("name", ["role:typo", "role:", "role:Chat", "role:chat "])
def test_an_unknown_role_name_is_refused_with_the_list(client: TestClient, name: str) -> None:
    before = client.get("/v1/config").json()["modelSlots"]
    response = client.patch(
        "/v1/config", json={"modelSlots": [{"model": name, "targets": ["anything"]}]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == []
    assert body["rejected"][0]["key"] == "modelSlots"
    message = body["rejected"][0]["message"]
    assert name in message
    assert all(f"role:{role.value}" in message for role in ModelRole)
    assert client.get("/v1/config").json()["modelSlots"] == before
