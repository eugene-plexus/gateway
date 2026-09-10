"""Pytest fixtures shared across the gateway test suite.

The routing table is injected on `app.state.routing` before the FastAPI
lifespan runs, so tests never reach a real agent or a real driver.
Each test scripts its fake drivers' responses by mutating them.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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
from eugene_plexus_gateway.routing import RoutingTable, _Backend, _Snapshot, _Unreachable
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


def make_routing_table(
    *fakes: FakeDriverClient,
    unreachable: dict[str, str] | None = None,
) -> RoutingTable:
    """A RoutingTable pre-loaded with fakes, without any HTTP.

    Reaches into `_snapshot` deliberately. The alternative — a public
    "install this snapshot" method — would be production API existing
    only for tests, and the snapshot dataclasses are the honest seam:
    the real `refresh()` builds exactly this and assigns exactly here.
    """
    table = RoutingTable(agent_url="http://fake-agent")
    snapshot = _Snapshot()
    for fake in fakes:
        backend = _Backend(
            name=fake.name,
            url=fake.base_url,
            client=fake,  # type: ignore[arg-type]
            info=fake.describe(),
        )
        snapshot.reachable.append(backend)
        if fake.model_id:
            snapshot.by_model.setdefault(fake.model_id, []).append(backend)
    for name, error in (unreachable or {}).items():
        snapshot.unreachable.append(_Unreachable(name=name, url=f"http://{name}.fake", error=error))
    for backends in snapshot.by_model.values():
        backends.sort(key=lambda b: b.name)
    table._snapshot = snapshot
    return table


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
