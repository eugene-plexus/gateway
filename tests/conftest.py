"""Pytest fixtures shared across the gateway test suite.

The routing table is injected on `app.state.routing` before the FastAPI
lifespan runs, so tests never reach a real agent or a real driver.
Each test scripts its fake drivers' responses by mutating them.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import (
    BackendKind,
    Capabilities,
    DriverInfo,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    Usage,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.routing import (
    RoutingTable,
    _Backend,
    _RuntimeFacts,
    _Snapshot,
    _Unreachable,
)
from eugene_plexus_gateway.settings import Settings


class FakeDriverClient:
    """In-memory test double implementing the DriverClient protocol."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str = "http://fake-driver",
        backend: BackendKind = BackendKind.openai_compat_http,
        model_id: str | None = "fake-model",
        provider: str | None = None,
        max_context_tokens: int | None = None,
        runtime: str | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.backend = backend
        self.model_id = model_id
        self.provider = provider
        self.max_context_tokens = max_context_tokens
        self.runtime = runtime

        # Mirrors the real clients' surface so the route can read these
        # off either without asking which kind it holds.
        self.attempts = 1
        self.served_by: str | None = name

        self.responses: list[str] = []
        """FIFO queue of canned responses; tests assign before calling."""
        self.calls: list[GenerateRequest] = []
        self.info_error: Exception | None = None
        self.generate_error: Exception | None = None
        """If set, `generate()` raises this instead of returning."""
        self.usage: Usage | None = None

    def describe(self) -> DriverInfo:
        """The same answer `info()` gives, without needing a loop."""
        capabilities = (
            Capabilities(maxContextTokens=self.max_context_tokens)
            if self.max_context_tokens is not None
            else None
        )
        return DriverInfo(
            backend=self.backend,
            provider=self.provider,
            modelId=self.model_id,
            runtime=self.runtime,
            capabilities=capabilities,
            version="0.0.0-fake",
        )

    async def info(self) -> DriverInfo:
        if self.info_error is not None:
            raise self.info_error
        return self.describe()

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.calls.append(request)
        if self.generate_error is not None:
            raise self.generate_error
        text = self.responses.pop(0) if self.responses else f"<{self.name} default response>"
        return GenerateResponse(
            content=text,
            finishReason=FinishReason.stop,
            backend=self.backend,
            modelId=self.model_id,
            usage=self.usage,
            latencyMs=1,
        )

    async def aclose(self) -> None:
        return None


def runtime_facts(
    name: str,
    *,
    alias: str | None = None,
    status: str = "ready",
    node: str | None = None,
    context_length: int | None = None,
    parallel_slots: int | None = None,
    idle_unload_seconds: int | None = None,
    start_on_demand: bool = False,
    stop_reason: str | None = None,
) -> _RuntimeFacts:
    """What the agent would report about one runtime, shaped by the test."""
    return _RuntimeFacts(
        name=name,
        alias=alias,
        status=status,
        node=node,
        url=f"http://127.0.0.1:809{len(name) % 10}",
        context_length=context_length,
        parallel_slots=parallel_slots,
        idle_unload_seconds=idle_unload_seconds,
        start_on_demand=start_on_demand,
        stop_reason=stop_reason,
        spec={"name": name, "engine": "llama_cpp", "modelPath": f"/models/{name}.gguf"},
    )


def make_routing_table(
    *fakes: FakeDriverClient,
    unreachable: dict[str, str] | None = None,
    runtimes: list[_RuntimeFacts] | None = None,
    slots: list[dict[str, Any]] | None = None,
    strategy: str = "least_busy",
    agent_url: str = "http://fake-agent",
) -> RoutingTable:
    """A RoutingTable pre-loaded with fakes, without any HTTP.

    Reaches into `_snapshot` deliberately. The alternative — a public
    "install this snapshot" method — would be production API existing
    only for tests, and the snapshot dataclasses are the honest seam:
    the real `refresh()` builds exactly this and assigns exactly here.

    A fake whose `runtime` names one of `runtimes` is joined to it, the
    way the real refresh joins by `DriverInfo.runtime`.
    """
    table = RoutingTable(
        agent_url=agent_url,
        slots=lambda: list(slots or []),
        strategy=lambda: strategy,
    )
    install_snapshot(table, *fakes, unreachable=unreachable, runtimes=runtimes)
    return table


def install_snapshot(
    table: RoutingTable,
    *fakes: FakeDriverClient,
    unreachable: dict[str, str] | None = None,
    runtimes: list[_RuntimeFacts] | None = None,
) -> None:
    """Replace a table's snapshot — what a refresh would have produced."""
    facts = {r.name: r for r in (runtimes or [])}
    snapshot = _Snapshot(runtimes=facts, agents={None: table._agent_url})
    for fake in fakes:
        backend = _Backend(
            name=fake.name,
            url=fake.base_url,
            client=fake,  # type: ignore[arg-type]
            info=fake.describe(),
            runtime=facts.get(fake.runtime) if fake.runtime else None,
        )
        snapshot.reachable.append(backend)
        if fake.model_id:
            snapshot.by_model.setdefault(fake.model_id, []).append(backend)
    for name, error in (unreachable or {}).items():
        snapshot.unreachable.append(_Unreachable(name=name, url=f"http://{name}.fake", error=error))
    for backends in snapshot.by_model.values():
        backends.sort(key=lambda b: b.name)
    table._snapshot = snapshot


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(config_file=tmp_path / "config.yaml")


@pytest.fixture
def fake_driver() -> FakeDriverClient:
    return FakeDriverClient(
        name="qwen-box",
        base_url="http://fake-driver-1",
        model_id="Qwen3-30B-A3B-Q4_K_M",
    )


@pytest.fixture
def app(settings: Settings, fake_driver: FakeDriverClient) -> FastAPI:
    """Default app: one reachable driver serving one model."""
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake_driver)
    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c
