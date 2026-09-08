"""In-process tool dispatch — the Phase-1 retrofit of v0.2 operations
onto the tool-calling wire format.

Every operation the gateway performs is being re-expressed as a
`ToolCall` dispatched through a `ToolRunner` to a channel-tagged
executor, so the wire format is the genuine spine of all operations
(not a bolt-on for "real" external tools).

Phase 1 is **behavior-preserving plumbing**:
  - The gateway CONSTRUCTS the calls for ops it already does; the
    model does not yet emit tool calls.
  - Tool RESULTS do not yet cross into the model's view — they stay as
    in-process typed payloads on `ToolInvocation.payload`. So Phase 1
    deliberately does NOT serialize results into
    `ToolResult.structuredContent`; that's Phase 2's job, when results
    actually reach the model.
  - Executors PROPAGATE exceptions (they do not swallow them into
    `isError` results) so the chat handler's existing 404 / 502 mapping
    is preserved exactly. The `isError` path on `ToolResult` exists for
    Phase 2, when the model sees the failure and reacts.
  - Phase-1 calls carry their inputs on `ToolContext` (live typed
    objects already in scope), not in `ToolCall.arguments` — the
    gateway invokes known ops with real objects rather than
    model-style JSON. `arguments` becomes load-bearing in Phase 2.

`build_tool_runner()` is reload-capable: call it again to rebuild the
registry (e.g. a future config change that adds/removes tools — the
operator-facing reload Troy flagged). No per-turn cost; built once at
lifespan startup and stored on `app.state.tool_runner`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ._generated.models import (
    ToolCall,
    ToolChannel,
    ToolDefinition,
    ToolEffect,
    ToolInvocationRecord,
    ToolResult,
)
from .identity import IdentityClient
from .memory import MemoryClient

log = logging.getLogger(__name__)

# Stable tool identifiers (match each registered ToolDefinition.name).
# Underscore-separated, not dotted: the spec's `name` pattern
# (^[a-zA-Z0-9_-]{1,64}$) mirrors Anthropic/OpenAI tool-name validation,
# which forbids dots — these names get passed to native tool calling in
# Phase 2, so they must already be vendor-legal.
TOOL_MEMORY_PERSON_RECENT = "memory_person_recent"
TOOL_MEMORY_APPEND_ENTRY = "memory_append_entry"
TOOL_NT_OBSERVE = "nt_observe"
TOOL_IDENTITY_GET_CONSTITUTION = "identity_get_constitution"
TOOL_IDENTITY_QUERY_SELF_MODEL = "identity_query_self_model"
TOOL_IDENTITY_GET_RELATIONSHIP = "identity_get_relationship"
TOOL_IDENTITY_LIST_PERSONS = "identity_list_persons"


@dataclass
class ToolContext:
    """Per-invocation live context for Phase-1 executors.

    Holds in-scope typed objects (UUIDs, domain models) the gateway
    passes when it constructs a call. Phase-2 model-driven calls will
    mostly carry their inputs in `ToolCall.arguments` instead; this bag
    is the transitional seam for objects that never leave the process
    this phase and so don't need serializing.
    """

    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolInvocation:
    """Outcome of running one tool.

    `result` is the canonical `ToolResult` envelope — the thing that
    becomes a `role: tool` message once results cross to the model in
    Phase 2. `payload` is the live typed object for in-process Phase-1
    consumers, so domain models that never leave the process this phase
    aren't pointlessly serialized and reparsed.
    """

    result: ToolResult
    payload: Any = None


ToolExecutor = Callable[[ToolCall, ToolContext], Awaitable[ToolInvocation]]


# Per-turn tool-invocation trace. `begin_tool_trace()` is called once at
# the start of a chat turn (before any tool runs); ToolRunner.run appends
# a ToolInvocationRecord per call. Task-local (ContextVar), so concurrent
# requests never share a trace; a fresh list is set at the top of every
# turn, so no stale data can bleed across sequential requests that reuse a
# task — hence no explicit reset is required.
_tool_trace: ContextVar[list[ToolInvocationRecord] | None] = ContextVar(
    "eugene_plexus_tool_trace", default=None
)


def begin_tool_trace() -> list[ToolInvocationRecord]:
    """Start capturing tool invocations for the current turn.

    Returns the list ToolRunner.run will append to; attach it to
    `ChatResponse.toolInvocations` once the turn completes.
    """
    trace: list[ToolInvocationRecord] = []
    _tool_trace.set(trace)
    return trace


def new_call(name: str, **arguments: Any) -> ToolCall:
    """Construct a `ToolCall` with a fresh id.

    Phase-1 callers usually pass no `arguments` (inputs ride on
    `ToolContext`); the kwargs path is here for the Phase-2 model-style
    invocation where `arguments` is the JSON the model emitted.
    """
    return ToolCall(id=str(uuid4()), name=name, arguments=dict(arguments))


class ToolRunner:
    """Channel-tagged tool registry + dispatcher. One per process.

    Rebuild by calling `build_tool_runner()` again — there's no mutable
    per-turn state, so a fresh runner can replace the old one wholesale
    on a config reload.
    """

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolDefinition, ToolExecutor]] = {}

    def register(self, definition: ToolDefinition, executor: ToolExecutor) -> None:
        self._tools[definition.name] = (definition, executor)

    def has(self, name: str) -> bool:
        return name in self._tools

    def definitions(self) -> list[ToolDefinition]:
        """The tool catalog — channel + effect per tool. Introspectable
        for a future operator-facing config / UI surface."""
        return [definition for (definition, _) in self._tools.values()]

    async def run(self, call: ToolCall, ctx: ToolContext | None = None) -> ToolInvocation:
        entry = self._tools.get(call.name)
        if entry is None:
            # Unknown tool in Phase 1 is a programming error (the
            # gateway only invokes tools it registered), not a model
            # misfire — fail loud rather than returning an isError result.
            raise KeyError(f"no tool registered under {call.name!r}")
        definition, executor = entry
        started = time.monotonic()
        # Exceptions propagate (see module docstring) — this is what keeps
        # the chat handler's existing error mapping behavior-identical. A
        # raising tool is therefore not traced; the turn fails and returns
        # no response anyway.
        invocation = await executor(call, ctx or ToolContext())
        trace = _tool_trace.get()
        if trace is not None:
            trace.append(
                ToolInvocationRecord(
                    name=definition.name,
                    channel=definition.channel,
                    effect=definition.effect,
                    summary=invocation.result.content,
                    isError=bool(invocation.result.isError),
                    latencyMs=int((time.monotonic() - started) * 1000),
                )
            )
        return invocation


# --------------------------------------------------------------------------- #
# Executors. Each factory binds the stable client/dependency at registration;
# per-turn inputs arrive on ToolContext.inputs.
# --------------------------------------------------------------------------- #


def _memory_person_recent_executor(memory: MemoryClient) -> ToolExecutor:
    """afferent / read_only — recall recent turns with a person."""

    async def execute(call: ToolCall, ctx: ToolContext) -> ToolInvocation:
        person_id = ctx.inputs["person_id"]
        limit = ctx.inputs["limit"]
        entries = await memory.person_recent(person_id, limit=limit)
        return ToolInvocation(
            result=ToolResult(callId=call.id, content=f"{len(entries)} recent entries"),
            payload=entries,
        )

    return execute


def _memory_append_entry_executor(memory: MemoryClient) -> ToolExecutor:
    """efferent / reversible — persist one conversation turn (append-only)."""

    async def execute(call: ToolCall, ctx: ToolContext) -> ToolInvocation:
        conversation_id = ctx.inputs["conversation_id"]
        entry = ctx.inputs["entry"]
        stored = await memory.append_entry(conversation_id, entry)
        return ToolInvocation(
            result=ToolResult(
                callId=call.id,
                content="entry appended" if stored is not None else "conversation not found",
            ),
            payload=stored,
        )

    return execute


def _nt_observe_executor() -> ToolExecutor:
    """internal — advance NT state from a turn's observations.

    The seam where a future emotion-read (an `internal` LLM call feeding
    cortisol etc.) plugs in; v0.2's observation→impulse map is the
    current executor body.
    """
    # Imported inside the factory so importing `tools` doesn't drag in the
    # bicameral package at module load.
    from .bicameral.nt import Observations, tick

    async def execute(call: ToolCall, ctx: ToolContext) -> ToolInvocation:
        state = ctx.inputs["state"]
        observations: Observations = ctx.inputs["observations"]
        new_state = tick(state, observations=observations)
        return ToolInvocation(
            result=ToolResult(callId=call.id, content="nt state advanced"),
            payload=new_state,
        )

    return execute


def _identity_get_constitution_executor(identity: IdentityClient) -> ToolExecutor:
    """afferent / read_only — read Eugene's constitution."""

    async def execute(call: ToolCall, ctx: ToolContext) -> ToolInvocation:
        constitution = await identity.get_constitution()
        return ToolInvocation(
            result=ToolResult(callId=call.id, content="constitution"),
            payload=constitution,
        )

    return execute


def _identity_query_self_model_executor(identity: IdentityClient) -> ToolExecutor:
    """afferent / read_only — query self-model entries by topic / person."""

    async def execute(call: ToolCall, ctx: ToolContext) -> ToolInvocation:
        entries = await identity.query_self_model(
            topic=ctx.inputs.get("topic"),
            person_id=ctx.inputs.get("person_id"),
            limit=ctx.inputs.get("limit", 5),
        )
        return ToolInvocation(
            result=ToolResult(callId=call.id, content=f"{len(entries)} self-model entries"),
            payload=entries,
        )

    return execute


def _identity_get_relationship_executor(identity: IdentityClient) -> ToolExecutor:
    """afferent / read_only — read the relationship summary for a person."""

    async def execute(call: ToolCall, ctx: ToolContext) -> ToolInvocation:
        relationship = await identity.get_relationship(ctx.inputs["person_id"])
        return ToolInvocation(
            result=ToolResult(
                callId=call.id,
                content="relationship" if relationship is not None else "no relationship",
            ),
            payload=relationship,
        )

    return execute


def _identity_list_persons_executor(identity: IdentityClient) -> ToolExecutor:
    """afferent / read_only — list known persons."""

    async def execute(call: ToolCall, ctx: ToolContext) -> ToolInvocation:
        persons = await identity.list_persons()
        return ToolInvocation(
            result=ToolResult(callId=call.id, content=f"{len(persons)} persons"),
            payload=persons,
        )

    return execute


def _definition(
    name: str,
    *,
    channel: ToolChannel,
    effect: ToolEffect,
    description: str,
) -> ToolDefinition:
    # Phase-1 inputSchema is a placeholder — the model never reads it this
    # phase. Real per-tool JSON Schemas land in Phase 2, when the model
    # picks tools and supplies arguments.
    return ToolDefinition(
        name=name,
        description=description,
        inputSchema={"type": "object"},
        channel=channel,
        effect=effect,
    )


def build_tool_runner(
    *,
    memory: MemoryClient | None,
    identity: IdentityClient | None = None,
) -> ToolRunner:
    """Build the gateway's tool registry.

    Reload-capable and idempotent: returns a fresh runner each call, so a
    config reload can swap `app.state.tool_runner` wholesale. Tools whose
    backing client is absent (e.g. memory in safe mode) are simply not
    registered.
    """
    runner = ToolRunner()

    if memory is not None:
        runner.register(
            _definition(
                TOOL_MEMORY_PERSON_RECENT,
                channel=ToolChannel.afferent,
                effect=ToolEffect.read_only,
                description="Recall recent conversation turns with a person.",
            ),
            _memory_person_recent_executor(memory),
        )
        runner.register(
            _definition(
                TOOL_MEMORY_APPEND_ENTRY,
                channel=ToolChannel.efferent,
                effect=ToolEffect.reversible,
                description="Persist one conversation turn to memory (append-only).",
            ),
            _memory_append_entry_executor(memory),
        )

    if identity is not None:
        runner.register(
            _definition(
                TOOL_IDENTITY_GET_CONSTITUTION,
                channel=ToolChannel.afferent,
                effect=ToolEffect.read_only,
                description="Read Eugene's constitution (declarative identity).",
            ),
            _identity_get_constitution_executor(identity),
        )
        runner.register(
            _definition(
                TOOL_IDENTITY_QUERY_SELF_MODEL,
                channel=ToolChannel.afferent,
                effect=ToolEffect.read_only,
                description="Query Eugene's self-model entries by topic / person.",
            ),
            _identity_query_self_model_executor(identity),
        )
        runner.register(
            _definition(
                TOOL_IDENTITY_GET_RELATIONSHIP,
                channel=ToolChannel.afferent,
                effect=ToolEffect.read_only,
                description="Read the relationship summary for a person.",
            ),
            _identity_get_relationship_executor(identity),
        )
        runner.register(
            _definition(
                TOOL_IDENTITY_LIST_PERSONS,
                channel=ToolChannel.afferent,
                effect=ToolEffect.read_only,
                description="List known persons.",
            ),
            _identity_list_persons_executor(identity),
        )

    runner.register(
        _definition(
            TOOL_NT_OBSERVE,
            channel=ToolChannel.internal,
            effect=ToolEffect.read_only,
            description="Advance neurotransmitter state from a turn's observations.",
        ),
        _nt_observe_executor(),
    )

    # Remaining Phase-1 ops — the agreement scorer (internal) and speech
    # (efferent) — live inside the linear deliberation pipeline
    # (loop.py / voice.py) that Eugene's continuous-loop direction is
    # slated to reshape (speaking becomes a decision, not an automatic
    # terminal step; the scorer stops being a response-forcing gate).
    # Retrofitting them into the current pipeline would be throwaway, so
    # they take their real shape as part of that reshape rather than here.
    # See project_functional_region_architecture.

    return runner
