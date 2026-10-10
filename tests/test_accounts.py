"""One driver, many models: provider accounts in the routing table (P1).

A driver's `/v1/info` reports `models[]` now. Every entry is a routing
candidate keyed `(node, driver, model)`; an account's are published as
`<driver name>/<id>`, with the gateway adding the prefix because a driver
does not know its own name (call P1-1). Each test here fails against the
gateway as it was: one `modelId` per driver, and nothing on a request
saying which model it was for.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import GenerateRequest, Message, Role
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import BoundClient, TieredClient
from eugene_plexus_gateway.model_patterns import matches, permits
from eugene_plexus_gateway.routing import RoutingTable
from tests.conftest import FakeDriverClient, install_snapshot, make_routing_table
from tests.test_admission import Authority
from tests.test_routing import _components, _driver_entry, route_http  # noqa: F401


def _chat(model: str) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


def _account(name: str = "openrouter", **kwargs: Any) -> FakeDriverClient:
    return FakeDriverClient(
        name=name,
        base_url=f"http://{name}-driver",
        models=["mistralai/mistral-nemo", "openai/gpt-oss-20b"],
        account=True,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# ids and the wire
# --------------------------------------------------------------------------- #


def test_an_accounts_models_are_published_under_its_name(settings: Any) -> None:
    app = create_app(settings=settings)
    account = _account()
    single = FakeDriverClient(name="qwen-driver", model_id="qwen3-8b")
    app.state.routing = make_routing_table(account, single)
    with TestClient(app) as client:
        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert ids == [
        "openrouter/mistralai/mistral-nemo",
        "openrouter/openai/gpt-oss-20b",
        # A single-model driver keeps its bare id: no installed client's
        # model name changes under it (call #1's second rule).
        "qwen3-8b",
    ]


def test_the_driver_is_asked_for_its_own_id_and_the_caller_sees_the_public_one(
    settings: Any,
) -> None:
    app = create_app(settings=settings)
    account = _account()
    account.responses = ["one", "two"]
    app.state.routing = make_routing_table(account)
    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions", json=_chat("openrouter/mistralai/mistral-nemo")
        ).json()
        second = client.post(
            "/v1/chat/completions", json=_chat("openrouter/openai/gpt-oss-20b")
        ).json()
    # One driver process, two models, and each request reached it naming
    # the model it was for -- unprefixed, as the driver knows it.
    assert [c.model for c in account.calls] == ["mistralai/mistral-nemo", "openai/gpt-oss-20b"]
    assert first["model"] == "openrouter/mistralai/mistral-nemo"
    assert second["model"] == "openrouter/openai/gpt-oss-20b"
    assert first["x_eugene_plexus"]["driver"] == "openrouter"


def test_a_streamed_answer_is_published_under_the_public_id(settings: Any) -> None:
    app = create_app(settings=settings)
    account = _account()
    account.responses = ["streamed"]
    app.state.routing = make_routing_table(account)
    with (
        TestClient(app) as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={**_chat("openrouter/openai/gpt-oss-20b"), "stream": True},
        ) as response,
    ):
        frames = [line for line in response.iter_lines() if line.startswith("data: {")]
    models = {json.loads(f[6:]).get("model") for f in frames}
    assert models == {"openrouter/openai/gpt-oss-20b"}
    assert account.calls[-1].model == "openai/gpt-oss-20b"


def test_the_same_account_name_on_two_machines_is_one_account(settings: Any) -> None:
    """P1-1's consequence: the name is the account, so two machines'
    `openrouter` drivers are replicas of each model, as a model alias on
    two machines already is."""
    table = make_routing_table(_account(node="node-a"), _account(node="node-b"))
    resolution = table.resolve("openrouter/mistralai/mistral-nemo")
    assert sorted(b.node for b in resolution.backends()) == ["node-a", "node-b"]


# --------------------------------------------------------------------------- #
# per-model facts
# --------------------------------------------------------------------------- #


def test_a_model_with_no_door_yet_is_not_listed(settings: Any) -> None:
    """P1-4: an account's rerank model is on the driver's list and not on
    `/v1/models` until a rerank door exists. This used a speech model until
    P3a gave speech its door, an image model until P4 and a video model until
    P5 (2026-09-28)."""
    app = create_app(settings=settings)

    class _WithRerank(FakeDriverClient):
        def describe(self):  # type: ignore[no-untyped-def]
            info = super().describe()
            assert info.models is not None
            info.models[1].surfaces = ["rerank"]
            return info

    account = _WithRerank(
        name="openrouter",
        models=["mistralai/mistral-nemo", "cohere/rerank-v3.5"],
        account=True,
    )
    app.state.routing = make_routing_table(account)
    with TestClient(app) as client:
        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
        refused = client.post("/v1/chat/completions", json=_chat("openrouter/cohere/rerank-v3.5"))
    assert ids == ["openrouter/mistralai/mistral-nemo"]
    assert refused.status_code >= 400
    assert not account.calls


def test_one_failing_model_cools_down_alone() -> None:
    """Each candidate owns its circuit: a model that keeps failing must not
    cool down the other models of its account."""
    table = make_routing_table(_account())
    nemo = table.pick(table.resolve("openrouter/mistralai/mistral-nemo"))
    oss = table.pick(table.resolve("openrouter/openai/gpt-oss-20b"))
    assert nemo is not None and oss is not None
    (nemo_client,) = nemo.candidates
    (oss_client,) = oss.candidates
    assert isinstance(nemo_client, BoundClient) and isinstance(oss_client, BoundClient)
    assert nemo_client.circuit is not oss_client.circuit
    assert nemo_client._inner is oss_client._inner  # one HTTP client per driver


def test_a_circuit_survives_a_refresh() -> None:
    """The candidate -- and so its cooldown -- is kept across refreshes
    for as long as its driver's client is, rather than rebuilt every 15 s."""
    same = _account()
    table = make_routing_table(same)
    first = table.pick(table.resolve("openrouter/mistralai/mistral-nemo"))
    install_snapshot(table, same)
    second = table.pick(table.resolve("openrouter/mistralai/mistral-nemo"))
    assert first is not None and second is not None
    assert first.candidates[0] is second.candidates[0]


# --------------------------------------------------------------------------- #
# keys scoped by pattern
# --------------------------------------------------------------------------- #


@pytest.fixture
def scoped(settings: Any, install: Any) -> Any:
    authority = Authority()
    token = install.client_key(name="App", jti="key-1")
    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    account = _account()
    other = _account(name="work")
    local = FakeDriverClient(name="qwen-driver", model_id="qwen3-8b")
    app.state.routing = make_routing_table(account, other, local)
    return app, authority, account, other, {"Authorization": "Bearer " + token}


def test_a_key_scoped_to_one_account_sees_that_account_and_nothing_else(scoped: Any) -> None:
    app, authority, account, other, headers = scoped
    authority.allowed = ["openrouter/*"]
    account.responses = ["ok"]
    with TestClient(app) as client:
        ids = [m["id"] for m in client.get("/v1/models", headers=headers).json()["data"]]
        served = client.post(
            "/v1/chat/completions", json=_chat("openrouter/openai/gpt-oss-20b"), headers=headers
        )
        refused = client.post(
            "/v1/chat/completions", json=_chat("work/openai/gpt-oss-20b"), headers=headers
        )
    assert ids == ["openrouter/mistralai/mistral-nemo", "openrouter/openai/gpt-oss-20b"]
    assert served.status_code == 200, served.text
    assert refused.status_code == 404
    assert not other.calls


def test_a_narrower_pattern_narrows(scoped: Any) -> None:
    app, authority, *_rest, headers = scoped
    authority.allowed = ["openrouter/openai/*", "qwen3-8b"]
    with TestClient(app) as client:
        ids = [m["id"] for m in client.get("/v1/models", headers=headers).json()["data"]]
    assert ids == ["openrouter/openai/gpt-oss-20b", "qwen3-8b"]


def test_star_is_the_only_wildcard_and_it_crosses_slashes() -> None:
    assert matches("openrouter/*", "openrouter/anthropic/claude-opus-5.5")
    assert not matches("openrouter/*", "work/anthropic/claude-opus-5.5")
    assert matches("*:free", "openrouter/respan/span-01-lite:free")
    assert matches("a?b", "a?b") and not matches("a?b", "axb")
    assert permits(None, "anything")
    assert not permits([], "anything")


def test_an_accounts_health_says_how_many_models_it_serves(settings: Any) -> None:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(_account())
    with TestClient(app) as client:
        (health,) = client.get("/v1/admin/drivers").json()["drivers"]
    assert health["account"] is True
    assert health["modelCount"] == 2
    assert "modelId" not in health or health["modelId"] is None


# --------------------------------------------------------------------------- #
# through a real refresh
# --------------------------------------------------------------------------- #


async def test_a_refresh_prefixes_an_accounts_models(route_http: Any) -> None:  # noqa: F811
    info = {
        "backend": "openai_compat_http",
        "version": "0.2.0",
        "models": [
            {"id": "mistralai/mistral-nemo", "surfaces": ["chat"]},
            # A rerank model: no door yet (a speech model until P3a, an image
            # model until P4, a video model until P5).
            {"id": "cohere/rerank-v3.5", "surfaces": ["rerank"]},
        ],
        "catalogue": {"source": "openrouter", "total": 2, "exposed": 2},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/components":
            return httpx.Response(200, json=_components(_driver_entry("openrouter", 8081)))
        return httpx.Response(200, json=info)

    route_http(handler)
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()
    assert table.known_models() == [
        "openrouter/cohere/rerank-v3.5",
        "openrouter/mistralai/mistral-nemo",
    ]
    assert [m.id for m in table.as_model_list()] == ["openrouter/mistralai/mistral-nemo"]
    await table.aclose()


def test_the_per_request_recheck_asks_for_one_model_not_the_whole_list(scoped: Any) -> None:
    """Admission re-checks a candidate's settings before it sends anything.
    For an account that must be one entry, not six hundred."""
    app, authority, account, _other, headers = scoped
    authority.allowed = None
    account.responses = ["ok"]
    account.info_calls.clear()
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={**_chat("openrouter/openai/gpt-oss-20b"), "temperature": 0.2},
            headers=headers,
        )
    assert response.status_code == 200, response.text
    assert {"models": True, "model": "openai/gpt-oss-20b"} in account.info_calls


async def test_each_attempt_records_the_model_it_asked_for() -> None:
    table = make_routing_table(_account())
    rows: list[Any] = []
    table._note_attempt = rows.append  # type: ignore[method-assign]
    client = table.pick(table.resolve("openrouter/openai/gpt-oss-20b"))
    assert isinstance(client, TieredClient)
    await client.generate(GenerateRequest(messages=[Message(role=Role.user, content="x")]))
    assert [r.model for r in rows] == ["openrouter/openai/gpt-oss-20b"]
    assert client.served_model == "openrouter/openai/gpt-oss-20b"
