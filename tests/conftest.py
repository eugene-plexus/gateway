"""Pytest fixtures shared across the test suite.

We stub the hemisphere clients before the FastAPI lifespan runs so tests
never try to reach real URLs. Each test can replace the per-driver
script of canned responses by mutating the fake clients on the app state.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.hemisphere_models import (
    BackendKind,
    DriverInfo,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
)
from eugene_plexus_gateway._generated.models import (
    AfferentEvent,
    IncomingMessage,
    MessageSource,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.bicameral.callosum import JaccardAgreementScorer
from eugene_plexus_gateway.bicameral.nt import neutral_state
from eugene_plexus_gateway.config import ConfigStore
from eugene_plexus_gateway.identity import IdentityClient, InProcessIdentity
from eugene_plexus_gateway.memory import NIL_PERSON_ID, InProcessMemory, MemoryClient
from eugene_plexus_gateway.runtime.loop import ConsciousnessLoop
from eugene_plexus_gateway.runtime.stream import ConsciousnessBroker
from eugene_plexus_gateway.settings import Settings
from eugene_plexus_gateway.tools import build_tool_runner


class FakeHemisphereClient:
    """In-memory test double implementing DriverClient."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str = "http://fake-driver",
        backend: BackendKind = BackendKind.claude_code_cli,
        model_id: str = "fake-model",
    ):
        self.name = name
        self.base_url = base_url
        self.backend = backend
        self.model_id = model_id
        self.responses: list[str] = []
        """FIFO queue of canned responses; tests assign before calling the route."""
        self._last_response: str | None = None
        self.calls: list[GenerateRequest] = []
        self.info_error: Exception | None = None
        self.generate_error: Exception | None = None
        """If set, `generate()` raises this instead of returning a response."""

    async def info(self) -> DriverInfo:
        if self.info_error is not None:
            raise self.info_error
        return DriverInfo(
            backend=self.backend,
            modelId=self.model_id,
            version="0.0.0-fake",
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.calls.append(request)
        if self.generate_error is not None:
            raise self.generate_error
        # Repeat the last scripted response once the queue drains, rather
        # than emitting a distinct sentinel. Under the plateau-stop a bout
        # may run more passes than a test scripted; repeating keeps the
        # cross-hemisphere agreement trajectory stable (the hemisphere
        # "keeps saying the same thing") instead of injecting a spurious
        # agreement swing on the extra passes.
        if self.responses:
            self._last_response = self.responses.pop(0)
        text = (
            self._last_response
            if self._last_response is not None
            else f"<{self.name} default response>"
        )
        return GenerateResponse(
            content=text,
            finishReason=FinishReason.stop,
            backend=self.backend,
            modelId=self.model_id,
            latencyMs=1,
        )

    async def aclose(self) -> None:
        return None


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    # Keep torch + sentence-transformers out of the test environment by
    # routing every test app through the Jaccard fallback scorer.
    return Settings(
        config_file=tmp_path / "config.yaml",
        disable_embedding_scorer=True,
    )


@pytest.fixture
def left_fake() -> FakeHemisphereClient:
    return FakeHemisphereClient(name="left", base_url="http://fake-left")


@pytest.fixture
def right_fake() -> FakeHemisphereClient:
    return FakeHemisphereClient(
        name="right",
        base_url="http://fake-right",
        backend=BackendKind.codex_cli,
        model_id="fake-gpt",
    )


@pytest.fixture
def app(
    settings: Settings,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> FastAPI:
    """Default app: drivers + memory wired, identity OFF.

    Tests that need identity wired in build their own app via
    `make_app_with_identity` or set `app.state.identity` directly.
    """
    app = create_app(settings=settings)
    app.state.drivers = [left_fake, right_fake]
    app.state.memory = InProcessMemory()
    app.state.memory_url = "in-process"
    app.state.identity = None
    app.state.identity_url = ""
    return app


@pytest.fixture
def in_process_identity() -> InProcessIdentity:
    return InProcessIdentity()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# Continuous-loop test helpers
#
# The loop is exercised by `await`ing `_handle_message` directly on a fresh
# `ConsciousnessLoop` over a fully-populated `app.state` — fast, and it
# sidesteps the friction between the sync TestClient and the async
# fire-and-forget loop. (`asyncio_mode = "auto"`, so `async def test_*`
# just works.)
# --------------------------------------------------------------------------- #


def build_loop_app(
    settings: Settings,
    drivers: list[FakeHemisphereClient],
    *,
    memory: MemoryClient | None = None,
    identity: IdentityClient | None = None,
) -> FastAPI:
    """An app with all of `app.state` populated for direct loop testing.

    Mirrors the non-loop parts of the lifespan (config, NT, scorer, tool
    runner) so a test can construct a `ConsciousnessLoop` and `await`
    `_handle_message` without running the lifespan or a TestClient.
    """
    app = create_app(settings=settings)
    store = ConfigStore(settings.config_file)
    store.load()
    # Pin both gate RNGs so loop-integration tests are reproducible while
    # still exercising the real (noise-on) code path — "seed the RNG, don't
    # disable the noise." Tests that need a specific stop / silence behavior
    # override the plateau* / action* values; the dedicated BoutGate /
    # action-selector tests construct their own seeded RNGs. With the default
    # action params at neutral NT, P(speak)≈0.98, so a pinned-seed turn
    # speaks — existing loop tests still see a `speech` event.
    store._values["plateauSeed"] = 12345
    store._values["actionSeed"] = 12345
    app.state.config_store = store
    app.state.safe_mode = False
    app.state.nt_state = neutral_state()
    app.state.drivers = list(drivers)
    app.state.memory = memory if memory is not None else InProcessMemory()
    app.state.memory_url = "in-process"
    app.state.identity = identity
    app.state.identity_url = "in-process" if identity is not None else ""
    app.state.scorer = JaccardAgreementScorer()
    app.state.tool_runner = build_tool_runner(memory=app.state.memory, identity=identity)
    return app


def make_message_event(
    content: str,
    *,
    person_id: UUID = NIL_PERSON_ID,
    conversation_id: UUID | None = None,
    platform: str = "ui",
) -> AfferentEvent:
    """Build a `kind=message` AfferentEvent for the loop."""
    source = MessageSource(platform=platform)
    return AfferentEvent(
        eventId=uuid4(),
        kind="message",
        source=source,
        timestamp=datetime.now(UTC),
        message=IncomingMessage(
            personId=person_id,
            content=content,
            conversationId=conversation_id,
            source=source,
        ),
    )


async def drive_message(app: FastAPI, event: AfferentEvent) -> list[tuple[str, dict]]:
    """Run one message through a fresh loop on `app`; return published events.

    Each element is `(event_type, data)` — the consciousness-stream events
    the loop emitted for this turn, in order.
    """
    broker = ConsciousnessBroker()
    loop = ConsciousnessLoop(app, broker)
    queue = broker.subscribe()
    await loop._handle_message(event)
    events: list[tuple[str, dict]] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events
