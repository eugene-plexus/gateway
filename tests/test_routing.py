"""The routing table's refresh: topology in, model->drivers out.

The other tests inject a pre-built table. These exercise the part that
actually talks HTTP — reading the watchdog topology and each driver's
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


@pytest.fixture
def route_http(monkeypatch: pytest.MonkeyPatch):
    """Route every httpx.AsyncClient in the process through one handler."""

    def install(handler: Any) -> None:
        real_init = httpx.AsyncClient.__init__

        def init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = httpx.MockTransport(handler)
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
    table = RoutingTable(watchdog_url="http://watchdog")
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
    table = RoutingTable(watchdog_url="http://watchdog")
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
    table = RoutingTable(watchdog_url="http://watchdog")
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
    table = RoutingTable(watchdog_url="http://watchdog")
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
    table = RoutingTable(watchdog_url="http://watchdog")
    await table.refresh()

    assert [b.name for b in table.backends_for("qwen")] == ["ok"]
    healths = {h.name: h for h in table.as_driver_health()}
    assert healths["ok"].reachable is True
    assert healths["dead"].reachable is False
    assert healths["dead"].error
    await table.aclose()


async def test_an_unreachable_watchdog_leaves_nothing_routable(route_http: Any) -> None:
    """Degrades rather than crashing: config endpoints stay up so the
    operator can fix whatever is wrong."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no watchdog")

    route_http(handler)
    table = RoutingTable(watchdog_url="http://watchdog")
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
    table = RoutingTable(watchdog_url="http://watchdog")
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
    table = RoutingTable(watchdog_url="http://watchdog")
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
    table = RoutingTable(watchdog_url="http://watchdog")
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
    table = RoutingTable(watchdog_url="http://watchdog")
    await table.refresh()

    models = table.as_model_list()
    assert models[0].x_eugene_plexus is not None
    assert models[0].x_eugene_plexus.context_length == 4096
    await table.aclose()


async def test_a_watchdog_error_response_is_not_a_crash(route_http: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "nope"})

    route_http(handler)
    table = RoutingTable(watchdog_url="http://watchdog")
    await table.refresh()
    assert table.is_empty()
    await table.aclose()
