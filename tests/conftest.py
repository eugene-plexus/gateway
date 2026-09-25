"""Pytest fixtures shared across the gateway test suite.

The routing table is injected on `app.state.routing` before the FastAPI
lifespan runs, so tests never reach a real agent or a real driver.
Each test scripts its fake drivers' responses by mutating them.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway import tokens
from eugene_plexus_gateway._generated.driver_models import (
    BackendKind,
    Capabilities,
    DecisionCapability,
    DriverInfo,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    Problem,
    ToolCall,
    Usage,
)
from eugene_plexus_gateway._generated.driver_models import (
    Kind as DecisionKindEnum,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.auth_state import AuthState, load_auth_state
from eugene_plexus_gateway.driver_client import DriverError, StreamEvent
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
        supports_decisions: bool = False,
        decision_max_concurrent: int | None = None,
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
        self.supports_decisions = supports_decisions
        self.decision_max_concurrent = decision_max_concurrent
        self.decide_calls = 0
        self.decide_hook = None
        self.decide_error: Exception | None = None
        self.tool_calls: list[ToolCall] | None = None
        """When set, `generate`/`stream` answer with these instead of
        text -- the tool-call-only turn, whose `content` is None."""
        self.reasoning: str | None = None
        """When set, both paths report this as the model's reasoning:
        `generate` on the response, `stream` as two reasoning events
        ahead of the text -- two, so a translator that only handles the
        first fragment of a block is caught."""
        self.stream_error_after_reasoning = False
        """Raise once the reasoning is out and before any text: the
        commit-point case for a model that thinks first."""
        self.stop_sequence: str | None = None
        """`stopSequence` on both paths' result, for a backend (vLLM)
        that names the stop string it matched."""
        self.supported_settings: list[str] = [
            "maxTokens",
            "temperature",
            "topP",
            "seed",
            "stop",
            "tools",
            "toolChoice",
            "responseFormat",
            "topK",
            "minP",
            "frequencyPenalty",
            "presencePenalty",
            "parallelToolCalls",
        ]
        """What `describe()` advertises as `supportedSettings`. A knob so
        a test can hand the gateway a backend that cannot carry one."""
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
        self.prompt_tokens: int | None = None
        """What `count_tokens` answers; None answers the driver's 501."""
        self.count_calls: list[GenerateRequest] = []
        self.count_error: Exception | None = None
        self.generate_hook: Callable[[], Awaitable[Any]] | None = None
        self.embed_hook: Callable[[], Awaitable[Any]] | None = None
        """Awaited before the canned answer. The disconnect tests hand
        in something that never returns, so the thing under test is what
        happens to an in-flight backend call rather than to a fast one."""

    def describe(self) -> DriverInfo:
        """The same answer `info()` gives, without needing a loop."""
        capabilities = Capabilities(
            supportedSettings=list(self.supported_settings),
            maxContextTokens=self.max_context_tokens,
            toolCalling=self.supports_tools,
            embeddings=self.supports_embeddings or None,
            chatCapable=not self.supports_decisions,
            decision=(
                DecisionCapability(
                    kinds=[DecisionKindEnum.noul, DecisionKindEnum.choice, DecisionKindEnum.score],
                    maxConcurrent=self.decision_max_concurrent,
                )
                if self.supports_decisions
                else None
            ),
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
            reasoning=self.reasoning,
            finishReason=self.finish_reason,
            stopSequence=self.stop_sequence,
            backend=self.backend,
            modelId=self.model_id,
            usage=self.usage,
            latencyMs=1,
        )

    async def count_tokens(self, request: GenerateRequest) -> int:
        """The driver's `/v1/generate/count`: `prompt_tokens`, or its 501.

        None -- the default -- is what every backend but llama.cpp answers,
        so a fake nobody configured cannot count, exactly as a real
        vLLM or Ollama cannot. The 501's words are the real driver's.
        """
        self.count_calls.append(request)
        if self.count_error is not None:
            raise self.count_error
        if self.prompt_tokens is None:
            raise DriverError(
                driver_name=self.name,
                driver_url=self.base_url,
                status_code=501,
                problem=Problem(
                    type="https://github.com/eugene-plexus/inference-driver#token-count-unsupported",
                    title="This backend cannot count this prompt without generating",
                    status=501,
                    detail="Cannot count exactly: this backend has no /apply-template (HTTP "
                    "404); only llama.cpp's llama-server can count a chat prompt without "
                    "generating. Nothing was sent to the model.",
                ),
                raw_body="",
            )
        return self.prompt_tokens

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

    async def decide(self, request: Any) -> Any:
        """Canned decisions per question kind. `choice` always picks the
        FIRST option of the request's criteria, so a test can tell the
        answer derives from the request and not from a hardcoded body."""
        from eugene_plexus_gateway._generated.driver_models import (
            DecisionAnswer,
            DecisionResponse,
        )

        self.decide_calls += 1
        if self.decide_hook is not None:
            await self.decide_hook()
        if self.decide_error is not None:
            raise self.decide_error
        answers: dict[str, DecisionAnswer] = {}
        for name, question in request.questions.items():
            kind = getattr(question.type, "value", question.type)
            if kind == "noul":
                answers[name] = DecisionAnswer.model_validate({"type": "noul", "noul": 0.9})
            elif kind == "choice":
                options = list(question.criteria or {"only": None})
                probabilities = {o: 0.0 for o in options}
                probabilities[options[0]] = 1.0
                answers[name] = DecisionAnswer.model_validate(
                    {
                        "type": "choice",
                        "choice": options[0],
                        "probabilities": probabilities,
                        "confidence": 0.8,
                    }
                )
            else:
                levels = list(question.criteria or ["a", "b"])
                answers[name] = DecisionAnswer.model_validate(
                    {
                        "type": "score",
                        "score": 1.0,
                        "legend": {str(i): level for i, level in enumerate(levels)},
                        "probabilities": {
                            str(i): (1.0 if i == 1 else 0.0) for i in range(len(levels))
                        },
                        "confidence": 0.7,
                    }
                )
        return DecisionResponse(
            answers=answers,
            modelId=self.model_id,
            reportedModel=f"<{self.name} reported>",
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
        if self.reasoning:
            half = len(self.reasoning) // 2
            yield StreamEvent(reasoning=self.reasoning[:half])
            yield StreamEvent(reasoning=self.reasoning[half:])
            if self.stream_error_after_reasoning:
                raise self.stream_error
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
                reasoning=self.reasoning,
                stopSequence=self.stop_sequence,
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


# --------------------------------------------------------------------------- #
# Trust (per-node token keys, 2026-09-25)
# --------------------------------------------------------------------------- #


@dataclass
class FakeInstall:
    """A trust bundle as this gateway's agent would keep it on disk.

    The gateway runs on node `gw`. The root's token key signs sessions
    and client keys; `gw`'s own key signs the tokens its agent hands its
    children; `far` is another machine of the install.
    """

    directory: Path
    name: str = "gw"
    grants: tuple[str, ...] = ()
    far_grants: tuple[str, ...] = ()
    identity: Ed25519PrivateKey = field(default_factory=tokens.generate_private_key)
    root: tokens.Signer = field(
        default_factory=lambda: tokens.Signer(key=tokens.generate_private_key(), issuer="control")
    )
    node: tokens.Signer | None = None
    far: tokens.Signer = field(
        default_factory=lambda: tokens.Signer(key=tokens.generate_private_key(), issuer="node:far")
    )
    version: int = 0

    def __post_init__(self) -> None:
        if self.node is None:
            self.node = tokens.Signer(key=tokens.generate_private_key(), issuer=self.recipient)
        self.publish()

    @property
    def recipient(self) -> str:
        return tokens.node_recipient(self.name)

    @property
    def authority(self) -> str:
        return tokens.public_b64(self.identity)

    @property
    def bundle_path(self) -> Path:
        return self.directory / "trust_bundle.json"

    def publish(self, *, revoked: tuple[tuple[str, int], ...] = ()) -> tokens.TrustBundle:
        assert self.node is not None
        self.version += 1
        bundle = tokens.build_bundle(
            authority=self.identity,
            version=self.version,
            epoch=1,
            keys=[
                self.root.trust_key(["authority"]),
                self.node.trust_key(["node", *self.grants]),
                self.far.trust_key(["node", *self.far_grants]),
            ],
            revoked_sessions=revoked,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        tokens.write_bundle_file(self.bundle_path, bundle)
        return bundle

    def auth_state(self, *, master_key_b64: str | None = None) -> AuthState:
        return load_auth_state(
            trust_bundle_file=str(self.bundle_path),
            trust_authority=self.authority,
            auth_recipient=self.recipient,
            service_token=self.service("gateway", ttl=365 * 24 * 3600),
            master_key_b64=master_key_b64,
        )

    def session(
        self,
        *,
        sub: str = "operator",
        ttl: int = 3600,
        aud: list[str] | None = None,
        now: int | None = None,
    ) -> str:
        token, _ = self.root.mint(
            typ=tokens.TYP_SESSION,
            sub=sub,
            aud=aud or [self.recipient, "control"],
            ttl_seconds=ttl,
            now=now,
        )
        return token

    def client_key(
        self, *, name: str = "app", jti: str = "key-1", ttl: int = 3600, now: int | None = None
    ) -> str:
        token, _ = self.root.mint(
            typ=tokens.TYP_CLIENT, sub=name, aud=["gateway"], ttl_seconds=ttl, now=now, jti=jti
        )
        return token

    def service(self, sub: str = "gateway", *, ttl: int = 3600) -> str:
        """A token this machine's agent minted for one of its children."""
        assert self.node is not None
        token, _ = self.node.mint(
            typ=tokens.TYP_SERVICE, sub=sub, aud=[self.recipient], ttl_seconds=ttl
        )
        return token

    def foreign_service(self, sub: str = "agent", *, ttl: int = 600) -> str:
        """Another machine's token, addressed here and correctly signed."""
        token, _ = self.far.mint(
            typ=tokens.TYP_SERVICE, sub=sub, aud=[self.recipient], ttl_seconds=ttl
        )
        return token

    def raw(self, signer: tokens.Signer, typ: str, **claims: Any) -> str:
        """A token with exactly these claims, for the shapes `mint` will not make."""
        now = int(time.time())
        body: dict[str, Any] = {"iss": signer.issuer, "iat": now, "exp": now + 60, **claims}
        body = {k: v for k, v in body.items() if v is not None}
        return jwt.encode(
            body, signer.key, algorithm="EdDSA", headers={"typ": typ, "kid": signer.kid}
        )


@pytest.fixture
def install(tmp_path: Path) -> FakeInstall:
    return FakeInstall(tmp_path / "node")
