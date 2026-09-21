"""Pytest fixtures shared across the gateway test suite.

The routing table is injected on `app.state.routing` before the FastAPI
lifespan runs, so tests never reach a real agent or a real driver.
Each test scripts its fake drivers' responses by mutating them.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator
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
    ToolCall,
    Usage,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import StreamEvent
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
        node: str | None = None,
        supports_tools: bool = False,
        supports_embeddings: bool = False,
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.backend = backend
        self.model_id = model_id
        self.provider = provider
        self.max_context_tokens = max_context_tokens
        self.runtime = runtime
        self.node = node
        """Which machine's agent reported this driver. `None` is a
        single-host install, and it is half of every key in `routing.py`
        since R1.6 -- a driver is joined only to a runtime on its OWN
        node, because two replicas of one model share a runtime name."""
        self.supports_tools = supports_tools
        self.supports_embeddings = supports_embeddings
        self.tool_calls: list[ToolCall] | None = None
        """When set, `generate`/`stream` answer with these instead of
        text -- the tool-call-only turn, whose `content` is None."""
        self.finish_reason: FinishReason = FinishReason.stop
        """The terminal reason both paths report. A knob because
        `content_filter` has to be observable on BOTH -- a filtered
        answer that survives the batch path and is flattened on the
        streaming one is exactly the shape of defect this project keeps
        producing, and one path's assertion says nothing about the
        other's."""

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
        self.stream_error_after: int | None = None
        """If set, `stream()` raises after this many token events. The
        knob the commit-point tests turn: 0 means "fail before the first
        token" (still cascadable) and 1 means "fail after it" (not)."""
        self.stream_error: Exception = RuntimeError("stream died")
        self.usage: Usage | None = None
        self.embed_error: Exception | None = None
        """If set, `embed()` raises this instead of returning."""
        self.embed_calls = 0
        self.generate_hook: Callable[[], Awaitable[Any]] | None = None
        self.embed_hook: Callable[[], Awaitable[Any]] | None = None
        """Awaited before the canned answer. The disconnect tests hand
        in something that never returns, so the thing under test is what
        happens to an in-flight backend call rather than to a fast one."""

    def describe(self) -> DriverInfo:
        """The same answer `info()` gives, without needing a loop."""
        capabilities = Capabilities(
            supportedSettings=[
                "maxTokens",
                "temperature",
                "topP",
                "seed",
                "stop",
                "tools",
                "toolChoice",
                "responseFormat",
            ],
            maxContextTokens=self.max_context_tokens,
            toolCalling=self.supports_tools,
            embeddings=self.supports_embeddings or None,
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
        if self.generate_hook is not None:
            await self.generate_hook()
        if self.generate_error is not None:
            raise self.generate_error
        if self.tool_calls is not None:
            return GenerateResponse(
                content=None,
                toolCalls=self.tool_calls,
                finishReason=FinishReason.tool_calls,
                backend=self.backend,
                modelId=self.model_id,
                usage=self.usage,
                latencyMs=1,
            )
        text = self.responses.pop(0) if self.responses else f"<{self.name} default response>"
        return GenerateResponse(
            content=text,
            finishReason=self.finish_reason,
            backend=self.backend,
            modelId=self.model_id,
            usage=self.usage,
            latencyMs=1,
        )

    async def embed(self, request: Any) -> Any:
        """Vectors, or the scripted failure. Values derive from the
        driver name so a test can tell WHICH backend answered from the
        numbers alone -- which is the only way to catch a cascade that
        crossed models."""
        from eugene_plexus_gateway._generated.driver_models import EmbedResponse

        self.embed_calls += 1
        if self.embed_hook is not None:
            await self.embed_hook()
        if self.embed_error is not None:
            raise self.embed_error
        seed = float(len(self.name))
        return EmbedResponse(
            embeddings=[[seed + i for i in range(4)] for _ in request.input],
            modelId=self.model_id,
            backend=self.backend,
            usage=self.usage,
            latencyMs=1,
        )

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        """Mirror `generate`, but a word at a time.

        Chunked rather than one-shot on purpose: a fake that yielded the
        whole answer in a single event would pass every assertion the
        M0-to-M9 implementation already passed, and so would prove
        nothing about the thing M10 changed.

        `stream_error_after` raises once that many events are out --
        which is how the commit point gets tested, since failing before
        the first event and failing after it must do opposite things.
        """
        self.calls.append(request)
        if self.generate_error is not None:
            raise self.generate_error
        if self.tool_calls is not None:
            # Split `arguments` in half so the test double reproduces
            # the property that actually breaks readers: no single
            # fragment is parseable JSON.
            for index, call in enumerate(self.tool_calls):
                args = call.function.arguments
                half = len(args) // 2
                if self.stream_error_after is not None and index >= self.stream_error_after:
                    raise self.stream_error
                yield StreamEvent(
                    tool_calls=[
                        {
                            "index": index,
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.function.name, "arguments": args[:half]},
                        }
                    ]
                )
                yield StreamEvent(
                    tool_calls=[{"index": index, "function": {"arguments": args[half:]}}]
                )
            yield StreamEvent(
                done=True,
                result=GenerateResponse(
                    content=None,
                    toolCalls=self.tool_calls,
                    finishReason=FinishReason.tool_calls,
                    backend=self.backend,
                    modelId=self.model_id,
                    usage=self.usage,
                    latencyMs=1,
                ),
            )
            return
        text = self.responses.pop(0) if self.responses else f"<{self.name} default response>"
        pieces = [w + " " for w in text.split(" ")]
        if pieces:
            pieces[-1] = pieces[-1].rstrip()
        for index, piece in enumerate(pieces):
            if self.stream_error_after is not None and index >= self.stream_error_after:
                raise self.stream_error
            yield StreamEvent(text=piece)
        yield StreamEvent(
            done=True,
            result=GenerateResponse(
                content=text,
                finishReason=self.finish_reason,
                backend=self.backend,
                modelId=self.model_id,
                usage=self.usage,
                latencyMs=1,
            ),
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
    """Replace a table's snapshot — what a refresh would have produced.

    **Keyed by `(node, name)`, because that is what a refresh produces.**
    This helper kept a bare-name map for three weeks after R1.6 moved
    production to the tuple, and nothing failed, because no test that
    uses this helper does a keyed lookup. That is the R1.6 lesson exactly
    — *the test written for this scenario had been passing on a shape no
    install can produce* — so the shape is corrected here rather than
    left for whichever check reaches for the key next.
    """
    facts = {(r.node, r.name): r for r in (runtimes or [])}
    snapshot = _Snapshot(runtimes=facts, agents={None: table._agent_url})
    for fake in fakes:
        backend = _Backend(
            name=fake.name,
            url=fake.base_url,
            client=fake,  # type: ignore[arg-type]
            info=fake.describe(),
            node=fake.node,
            runtime=facts.get((fake.node, fake.runtime)) if fake.runtime else None,
        )
        snapshot.reachable.append(backend)
        if fake.model_id:
            snapshot.by_model.setdefault(fake.model_id, []).append(backend)
    for name, error in (unreachable or {}).items():
        snapshot.unreachable.append(_Unreachable(name=name, url=f"http://{name}.fake", error=error))
    # `(node, name)`, as the real refresh sorts: two replicas of one
    # model on two machines have equal names, and sorting on the name
    # alone would leave their order to dict insertion.
    for backends in snapshot.by_model.values():
        backends.sort(key=lambda b: (b.node or "", b.name))
    table._snapshot = snapshot


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # `metrics_file` is relative by default, so leaving it unset writes a
    # metrics database into the checkout on every test run - which then
    # sits there waiting for a `git add -A`. Both paths belong in tmp.
    return Settings(
        config_file=tmp_path / "config.yaml",
        metrics_file=tmp_path / "metrics.sqlite3",
    )


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
