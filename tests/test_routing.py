"""The routing table's refresh: topology in, model->drivers out.

The other tests inject a pre-built table. These exercise the part that
actually talks HTTP — reading the agent topology and each driver's
`/v1/info` — because that is where "adding a model is not a config edit"
either works or doesn't.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from eugene_plexus_gateway.driver_client import HttpDriverClient
from eugene_plexus_gateway.routing import RoutingTable


def _components(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"components": list(entries)}


def _driver_entry(name: str, port: int) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "inference-driver",
        "url": f"http://127.0.0.1:{port}",
        "status": "running",
    }


def _info(model_id: str | None, *, context: int | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"backend": "openai_compat_http", "version": "0.1.0"}
    if model_id is not None:
        body["modelId"] = model_id
    if context is not None:
        body["capabilities"] = {"maxContextTokens": context}
    return body


def _runtime_entry(name: str, alias: str, *, context: int | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "engine": "llama_cpp",
        "modelPath": f"/models/{alias}.gguf",
        "modelAlias": alias,
        "status": "ready",
    }
    if context is not None:
        body["capabilities"] = {"contextLength": context}
    return body


@pytest.fixture
def route_http(monkeypatch: pytest.MonkeyPatch):
    """Route every httpx.AsyncClient in the process through one handler.

    `/v1/runtimes` is answered by the fixture, not the test's handler:
    every refresh reads it, and threading an empty list through a dozen
    handlers that don't care about runtimes would bury the thing each
    test is actually about. Tests that DO care pass `runtimes=[...]`, or
    `handle_runtimes=True` to take the path over entirely (which is the
    only way to make it fail).
    """

    def install(
        handler: Any,
        *,
        runtimes: list[dict[str, Any]] | None = None,
        handle_runtimes: bool = False,
    ) -> None:
        def wrapped(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/runtimes" and not handle_runtimes:
                return httpx.Response(200, json={"runtimes": runtimes or []})
            return handler(request)

        real_init = httpx.AsyncClient.__init__

        def init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = httpx.MockTransport(wrapped)
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", init)

    return install


async def test_refresh_groups_drivers_by_the_model_they_serve(route_http: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/components":
            return httpx.Response(
                200, json=_components(_driver_entry("a", 8081), _driver_entry("b", 8082))
            )
        if request.url.port == 8081:
            return httpx.Response(200, json=_info("qwen"))
        return httpx.Response(200, json=_info("llama"))

    route_http(handler)
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()

    assert table.known_models() == ["llama", "qwen"]
    assert [b.name for b in table.backends_for("qwen")] == ["a"]
    await table.aclose()


async def test_two_drivers_on_one_model_become_a_priority_list(route_http: Any) -> None:
    """This is where failover comes from: nothing is configured, the
    topology just happens to have two drivers serving the same thing."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/components":
            return httpx.Response(
                200, json=_components(_driver_entry("b", 8082), _driver_entry("a", 8081))
            )
        return httpx.Response(200, json=_info("qwen"))

    route_http(handler)
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()

    # Sorted by name, so repeated requests hit the same backend first.
    # Round-robin is load balancing and deliberately isn't here yet.
    assert [b.name for b in table.backends_for("qwen")] == ["a", "b"]

    client = table.resolve("qwen")
    assert client is not None
    assert type(client).__name__ == "FailoverDriverClient"
    await table.aclose()


async def test_one_driver_resolves_without_a_failover_wrapper(route_http: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/components":
            return httpx.Response(200, json=_components(_driver_entry("a", 8081)))
        return httpx.Response(200, json=_info("qwen"))

    route_http(handler)
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()

    assert isinstance(table.resolve("qwen"), HttpDriverClient)
    await table.aclose()


async def test_non_driver_components_are_ignored(route_http: Any) -> None:
    """The topology also holds the gateway itself and (later) the library.
    Probing those for `/v1/info` would be nonsense."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/components":
            return httpx.Response(
                200,
                json=_components(
                    {
                        "name": "gateway",
                        "kind": "gateway",
                        "url": "http://127.0.0.1:8080",
                        "status": "running",
                    },
                    _driver_entry("a", 8081),
                ),
            )
        assert request.url.port == 8081, "only the driver should be probed"
        return httpx.Response(200, json=_info("qwen"))

    route_http(handler)
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()

    assert [b.name for b in table.backends_for("qwen")] == ["a"]
    await table.aclose()


async def test_an_unreachable_driver_is_reported_not_routed_to(route_http: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/components":
            return httpx.Response(
                200, json=_components(_driver_entry("dead", 8081), _driver_entry("ok", 8082))
            )
        if request.url.port == 8081:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json=_info("qwen"))

    route_http(handler)
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()

    assert [b.name for b in table.backends_for("qwen")] == ["ok"]
    healths = {h.name: h for h in table.as_driver_health()}
    assert healths["ok"].reachable is True
    assert healths["dead"].reachable is False
    assert healths["dead"].error
    await table.aclose()


async def test_an_unreachable_agent_leaves_nothing_routable(route_http: Any) -> None:
    """Degrades rather than crashing: config endpoints stay up so the
    operator can fix whatever is wrong."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no agent")

    route_http(handler)
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()

    assert table.is_empty()
    assert table.resolve("anything") is None
    await table.aclose()


async def test_a_refresh_replaces_the_table_wholesale(route_http: Any) -> None:
    """A model that went away must stop being routable, and a new one
    must start — that is what makes starting an engine enough."""
    state = {"model": "qwen"}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/components":
            return httpx.Response(200, json=_components(_driver_entry("a", 8081)))
        return httpx.Response(200, json=_info(state["model"]))

    route_http(handler)
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()
    assert table.known_models() == ["qwen"]

    state["model"] = "llama"
    await table.refresh()
    assert table.known_models() == ["llama"]
    assert table.resolve("qwen") is None
    await table.aclose()


async def test_clients_are_reused_across_refreshes(route_http: Any) -> None:
    """Rebuilding an httpx client every refresh would throw away the
    connection pool and leak sockets."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/components":
            return httpx.Response(200, json=_components(_driver_entry("a", 8081)))
        return httpx.Response(200, json=_info("qwen"))

    route_http(handler)
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()
    first = table.backends_for("qwen")[0].client
    await table.refresh()
    assert table.backends_for("qwen")[0].client is first
    await table.aclose()


async def test_a_driver_leaving_the_topology_closes_its_client(route_http: Any) -> None:
    entries = [_driver_entry("a", 8081), _driver_entry("b", 8082)]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/components":
            return httpx.Response(200, json=_components(*entries))
        return httpx.Response(200, json=_info("qwen"))

    route_http(handler)
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()
    assert len(table.backends_for("qwen")) == 2

    entries.pop()
    await table.refresh()
    assert [b.name for b in table.backends_for("qwen")] == ["a"]
    await table.aclose()


async def test_smallest_context_wins_across_replicas(route_http: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/components":
            return httpx.Response(
                200, json=_components(_driver_entry("a", 8081), _driver_entry("b", 8082))
            )
        context = 8192 if request.url.port == 8081 else 4096
        return httpx.Response(200, json=_info("qwen", context=context))

    route_http(handler)
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()

    models = table.as_model_list()
    assert models[0].x_eugene_plexus is not None
    assert models[0].x_eugene_plexus.context_length == 4096
    await table.aclose()


async def test_a_agent_error_response_is_not_a_crash(route_http: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "nope"})

    route_http(handler)
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()
    assert table.is_empty()
    await table.aclose()


async def test_runtime_attribution_matches_alias_to_what_the_driver_serves(
    route_http: Any,
) -> None:
    """A runtime's modelAlias IS what a client asks the gateway for, so
    an alias equal to a driver's modelId identifies the engine process
    behind that driver — no new field on any contract."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/components":
            return httpx.Response(200, json=_components(_driver_entry("a", 8081)))
        return httpx.Response(200, json=_info("qwen"))

    route_http(handler, runtimes=[_runtime_entry("qwen3-27b", "qwen", context=4096)])
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()

    assert table.runtime_for("qwen") == "qwen3-27b"
    # The driver reported no capabilities of its own — an
    # openai_compat_http driver only sees an HTTP endpoint — so without
    # the runtime this would be None.
    assert table.as_model_list()[0].x_eugene_plexus.context_length == 4096
    await table.aclose()


async def test_runtime_attribution_is_absent_for_a_model_with_replicas(
    route_http: Any,
) -> None:
    """Which of two identical runtimes served a request is knowable, but
    not from here. Report nothing rather than a coin flip."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/components":
            return httpx.Response(200, json=_components(_driver_entry("a", 8081)))
        return httpx.Response(200, json=_info("qwen"))

    route_http(
        handler,
        runtimes=[
            _runtime_entry("qwen-gpu0", "qwen", context=4096),
            _runtime_entry("qwen-gpu1", "qwen", context=8192),
        ],
    )
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()

    assert table.runtime_for("qwen") is None
    assert table.as_model_list()[0].x_eugene_plexus.context_length is None
    await table.aclose()


async def test_runtime_attribution_is_absent_when_no_runtime_serves_the_model(
    route_http: Any,
) -> None:
    """A hosted or CLI backend has no runtime of ours. Routing is
    unaffected — the field is just absent."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/components":
            return httpx.Response(200, json=_components(_driver_entry("a", 8081)))
        return httpx.Response(200, json=_info("claude-opus-4-7"))

    route_http(handler)
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()

    assert table.runtime_for("claude-opus-4-7") is None
    assert table.resolve("claude-opus-4-7") is not None
    await table.aclose()


async def test_unreadable_runtimes_endpoint_does_not_stop_routing(route_http: Any) -> None:
    """The agent can 500 on /v1/runtimes and everything still routes;
    attribution is additive, never load-bearing."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/components":
            return httpx.Response(200, json=_components(_driver_entry("a", 8081)))
        return httpx.Response(200, json=_info("qwen"))

    seen: list[str] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/runtimes":
            seen.append("runtimes")
            return httpx.Response(500, text="boom")
        return handler(request)

    # Take over /v1/runtimes so the failure actually reaches the table;
    # the fixture would otherwise answer it with an empty list and this
    # test would pass without exercising anything.
    route_http(wrapped, handle_runtimes=True)
    table = RoutingTable(agent_url="http://agent")
    await table.refresh()

    assert seen == ["runtimes"], "the 500 must actually reach the table"
    assert table.known_models() == ["qwen"]
    assert table.runtime_for("qwen") is None
    await table.aclose()
