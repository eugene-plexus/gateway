"""Response provenance identifies the serving replica, including after failover."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import BackendKind
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.routing import _RuntimeFacts
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table


def _runtime(node: str) -> _RuntimeFacts:
    return _RuntimeFacts(
        name="r",
        alias="m",
        status="ready",
        node=node,
        url=f"http://{node}/engine",
        context_length=8192,
        parallel_slots=2,
        idle_unload_seconds=None,
        start_on_demand=False,
        stop_reason=None,
    )


@pytest.mark.parametrize(
    "door,stream",
    [
        ("chat/completions", False),
        ("chat/completions", True),
        ("completions", False),
        ("completions", True),
        ("messages", False),
        ("responses", False),
        ("embeddings", False),
        ("systemone", False),
    ],
)
def test_the_node_is_the_replica_that_answered(settings: Settings, door: str, stream: bool) -> None:
    drivers = [
        FakeDriverClient(
            name="driver",
            node=node,
            runtime="r",
            model_id="m",
            surfaces=["chat", "completion", "embeddings", "decisions"],
            supports_decisions=True,
        )
        for node in ("a-tower", "b-studio")
    ]
    dead = httpx.ConnectError("fixture driver refused the connection")
    drivers[0].generate_error = drivers[0].embed_error = drivers[0].decide_error = dead
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(
        *drivers, runtimes=[_runtime(node) for node in ("a-tower", "b-studio")]
    )
    bodies = {
        "chat/completions": {"messages": [{"role": "user", "content": "hi"}]},
        "completions": {"prompt": "hi"},
        "messages": {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
        "responses": {"input": "hi"},
        "embeddings": {"input": "hi"},
        "systemone": {
            "state": "hi",
            "questions": {"ok": {"type": "noul", "instructions": "Is this OK?"}},
        },
    }
    body = {"model": "m", **bodies[door]}
    if door not in ("embeddings", "systemone"):
        body["stream"] = stream
    with TestClient(app) as client:
        response = client.post("/v1/" + door, json=body, headers={"Origin": "http://console"})
    assert response.status_code == 200, response.text
    if door in ("messages", "responses"):
        assert response.headers["x-eugene-plexus-node"] == "b-studio"
        assert response.headers["x-eugene-plexus-runtime"] == "r"
        exposed = response.headers["access-control-expose-headers"].lower().split(", ")
        assert "x-eugene-plexus-node" in exposed
    else:
        if stream:
            frames = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            routing = next(f["x_eugene_plexus"] for f in frames if f.get("x_eugene_plexus"))
        else:
            routing = response.json()["x_eugene_plexus"]
        assert routing["node"] == "b-studio"
        assert routing["runtime"] == "r"
        assert routing["attempts"] == 2


@pytest.mark.parametrize(
    "backend",
    [BackendKind.openai_compat_http, BackendKind.claude_code_cli, BackendKind.codex_cli],
)
def test_a_backend_without_a_runtime_has_no_serving_node(
    settings: Settings, backend: BackendKind
) -> None:
    driver = FakeDriverClient(name="driver", node="tower", model_id="m", backend=backend)
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(driver)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200, response.text
    routing = response.json()["x_eugene_plexus"]
    assert routing.get("runtime") is None
    assert routing.get("node") is None
