"""A sealed control root must not make one machine's runtimes routable on faith.

Found by A4 on a GitHub macOS runner (2026-09-30, specs
docs/acceptance/a4-macos-runner-run.md): a one-machine install, enrolled to
its own control root, whose root was sealed (`GET /v1/nodes` answered 503).
The gateway fell back to its own agent, keyed as `None`, so that agent's
drivers were `(None, name)`. The same agent's runtimes were keyed by the
node the agent REPORTS on each runtime, `box`, because the two were
assumed to be "the same thing". They are not, while the root is sealed:
nothing joined, every driver was "routable on faith", and a stopped, a
loading or a crashed engine was sent requests. A container restarted with
no keyring comes back in exactly this state.

A driver and its runtime come from one agent read, so they share the key
the gateway asked by, whatever the agent calls itself.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from eugene_plexus_gateway.routing import RoutingTable

AGENT = "http://agent-a:8079"
CONTROL = "http://control:8083"
DRIVER_URL = "http://127.0.0.1:8090"
MODEL = "qwen3-1.7b"
DRIVER = "qwen-driver"
RUNTIME = "qwen"
NODE = "box"


class SealedInstall:
    def __init__(self) -> None:
        self.root_status = 503
        self.runtime_status = "stopped"

    def handler(self, request: httpx.Request) -> httpx.Response:
        url, path = (
            f"{request.url.scheme}://{request.url.host}:{request.url.port}",
            request.url.path,
        )
        if url == CONTROL and path == "/v1/nodes":
            if self.root_status != 200:
                return httpx.Response(
                    self.root_status, json={"detail": {"title": "Locked", "detail": "sealed"}}
                )
            return httpx.Response(200, json={"nodes": [{"name": NODE, "url": AGENT}]})
        if url == AGENT:
            if path == "/v1/node":
                return httpx.Response(
                    200, json={"enrolled": True, "name": NODE, "controlUrl": CONTROL}
                )
            if path == "/v1/components":
                return httpx.Response(
                    200,
                    json={
                        "components": [
                            {
                                "name": DRIVER,
                                "kind": "inference-driver",
                                "url": DRIVER_URL,
                                "status": "running",
                            }
                        ]
                    },
                )
            if path == "/v1/runtimes":
                # An enrolled agent stamps its own name on each runtime.
                runtime: dict[str, Any] = {
                    "name": RUNTIME,
                    "engine": "llama_cpp",
                    "modelPath": "/m/qwen.gguf",
                    "modelAlias": MODEL,
                    "status": self.runtime_status,
                    "url": DRIVER_URL,
                    "node": NODE,
                }
                return httpx.Response(200, json={"runtimes": [runtime]})
        if path == "/v1/info":
            return httpx.Response(
                200,
                json={
                    "backend": "openai_compat_http",
                    "version": "0.1.0",
                    "models": [{"id": MODEL, "surfaces": ["chat"]}],
                    "runtime": RUNTIME,
                },
            )
        return httpx.Response(404, json={"detail": f"{url}{path}"})


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch) -> SealedInstall:
    fake = SealedInstall()
    real_init = httpx.AsyncClient.__init__

    def init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = httpx.MockTransport(fake.handler)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
    return fake


async def _table() -> RoutingTable:
    table = RoutingTable(agent_url=AGENT, refresh_seconds=3600)
    await table.refresh()
    return table


async def test_a_sealed_root_still_joins_this_machines_drivers_to_their_runtimes(
    install: SealedInstall,
) -> None:
    table = await _table()
    [backend] = table.backends_for(MODEL)
    assert backend.runtime is not None, "joined on faith: the sealed-root fallback"
    assert backend.runtime.name == RUNTIME
    # The point of the join: a stopped engine is not sent requests.
    assert backend.runtime.status == "stopped"
    assert not backend.eligible


async def test_a_ready_runtime_behind_a_sealed_root_is_routable(install: SealedInstall) -> None:
    install.runtime_status = "ready"
    table = await _table()
    [backend] = table.backends_for(MODEL)
    assert backend.runtime is not None and backend.eligible


async def test_an_answering_root_keys_by_the_node_it_lists(install: SealedInstall) -> None:
    """The guard: the ordinary enrolled case keeps its node names."""
    install.root_status = 200
    table = await _table()
    [backend] = table.backends_for(MODEL)
    assert backend.node == NODE
    assert backend.runtime is not None and backend.runtime.node == NODE
    assert not backend.eligible
