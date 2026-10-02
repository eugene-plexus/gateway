"""A turn waits for room in its replica's shared context (CB3).

**llama-server's slots share one KV pool by default** (`-c`, automatic
slots), and when the prompts in flight on a replica together outgrow it the
engine refuses the next request or cuts every stream it is decoding.
Measured 2026-10-02 (`specs/docs/acceptance/cache-aware-balancing-measurement.md`
§5, §8): an 8B at `-c 65536` with four agents a replica overflowed 22-42
times a run; on eight 1B replicas a budget like this one took the failed
turns from 44 to 0 and halved the p99.

**The budget.** Per runtime whose agent reports
`capabilities.contextPoolTokens`, the prompt tokens in flight may not pass
90% of the pool. Each turn counts as its conversation's last prompt tokens,
plus its new text at ~3.5 characters a token (the whole request's, for a new
conversation), plus its `max_tokens`. The 10% spare is because characters
over 3.5 let two overflows through on a pool that holds two prompts.

**The wait.** A turn that does not fit waits for a request on that runtime
to finish: on its own replica while it holds the conversation (going
elsewhere would read its whole history again), else on whichever replica of
the tier has room first. After `swapWaitSeconds` it is sent in the usual
order, and a refusal then is the driver's `#backend-capacity`, which
cascades and is not counted against the replica. A turn alone on its
runtime always fits, and a request bigger than the whole pool is sent at
once: the engine's own refusal is the honest answer to it.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .affinity import MAX_KEYS, TTL_SECONDS

#: Characters a token, for text the engine has not counted yet. Measured on
#: the Llama 3 tokenizer over Claude Code and Codex requests (record §5).
CHARS_PER_TOKEN = 3.5
#: The share of a pool the budget keeps free for its own estimate's error.
SPARE = 0.10

#: A runtime, as the routing table keys it: `(node, name)`.
Key = tuple[str | None, str]


def request_chars(request: Any) -> int:
    """How much text a request carries, as the engine will see it.

    The messages and the tool definitions, as JSON: the template renders
    both into the prompt, and the measured estimate counted the same.
    """
    parts: list[Any] = []
    for name in ("messages", "tools"):
        value = getattr(request, name, None)
        if value:
            parts.append(
                [
                    v.model_dump(mode="json", exclude_none=True) if hasattr(v, "model_dump") else v
                    for v in value
                ]
            )
    completion = getattr(request, "completion", None)
    prompt = getattr(completion, "prompt", None) if completion is not None else None
    if prompt:
        parts.append(prompt)
    return len(json.dumps(parts, default=str))


@dataclass(frozen=True)
class Conversation:
    """The conversation a turn continues: where its sizes are kept, and, on
    an affinity hit, the replica it went home to and the prompt that replica
    held after the last turn (CB5)."""

    sizes: ConversationSizes
    target: str
    key: str
    home: Any = None
    previous_prompt: int | None = None


@dataclass(frozen=True)
class _Seen:
    prompt_tokens: int
    chars: int
    at: float


class ConversationSizes:
    """Model and conversation key -> its last prompt, as the engine counted it.

    Bounded and expiring like the affinity table: a conversation idle past
    its TTL is estimated from its size alone, which only overcounts.
    """

    def __init__(self, *, ttl: float = TTL_SECONDS, size: int = MAX_KEYS) -> None:
        self._ttl = ttl
        self._size = size
        self._entries: OrderedDict[tuple[str, str], _Seen] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, target: str, key: str) -> _Seen | None:
        with self._lock:
            seen = self._entries.get((target, key))
            if seen is None:
                return None
            if time.perf_counter() - seen.at > self._ttl:
                del self._entries[(target, key)]
                return None
            return seen

    def put(self, target: str, key: str, prompt_tokens: int, chars: int) -> None:
        with self._lock:
            self._entries[(target, key)] = _Seen(prompt_tokens, chars, time.perf_counter())
            self._entries.move_to_end((target, key))
            while len(self._entries) > self._size:
                self._entries.popitem(last=False)


def estimate(chars: int, max_tokens: int | None, seen: _Seen | None) -> tuple[int, int]:
    """`(prompt, total)`: the tokens one turn's prompt is, and what the turn
    will hold in its runtime's pool once it has answered."""
    if seen is not None and chars >= seen.chars:
        prompt = seen.prompt_tokens + (chars - seen.chars) / CHARS_PER_TOKEN
    else:
        prompt = chars / CHARS_PER_TOKEN
    tokens = math.ceil(prompt)
    return tokens, tokens + max(0, max_tokens or 0)


class PoolLedger:
    """The prompt tokens in flight on each runtime, install-wide.

    One per routing table. Synchronous bookkeeping on the event loop, so a
    check and the reservation that follows it cannot be split by another
    request; `room` wakes every waiter when anything is released.
    """

    def __init__(self) -> None:
        self._tokens: dict[Key, int] = {}
        self._room: asyncio.Event | None = None

    def in_flight(self, runtime: Key) -> int:
        return self._tokens.get(runtime, 0)

    def fits(self, runtime: Key, pool: int, tokens: int, prompt: int) -> bool:
        """Whether a turn holding `tokens` (its prompt `prompt`) fits now.

        Alone it always goes. A PROMPT bigger than the pool goes at once,
        because the engine refuses it outright and so takes nothing from
        anyone; judged on the prompt and not on `tokens`, since a turn's
        `max_tokens` (Claude Code sends 32,000) can make a prompt that runs
        fine look bigger than the pool, and sent at once it would overflow
        the turns already there.
        """
        held = self._tokens.get(runtime, 0)
        if held == 0 or prompt > pool:
            return True
        return held + tokens <= int(pool * (1 - SPARE))

    def take(self, runtime: Key, tokens: int) -> None:
        self._tokens[runtime] = self._tokens.get(runtime, 0) + tokens

    def give(self, runtime: Key, tokens: int) -> None:
        left = self._tokens.get(runtime, 0) - tokens
        if left > 0:
            self._tokens[runtime] = left
        else:
            self._tokens.pop(runtime, None)
        if self._room is not None:
            self._room.set()
            self._room = None

    async def changed(self, seconds: float) -> None:
        """Until something is released, or `seconds` pass."""
        if self._room is None:
            self._room = asyncio.Event()
        room = self._room
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(room.wait(), seconds)


class TurnBudget:
    """One turn's place in the ledger, across the attempts it makes.

    Built by `RoutingTable.pick` when any backend in the slot reports a
    shared pool; held by the `TieredClient`. `admit` waits and reorders the
    first tier; `enter` moves the reservation to whichever backend an
    attempt goes to; `close` gives it back, on every path.
    """

    def __init__(
        self,
        *,
        ledger: PoolLedger,
        sizes: ConversationSizes,
        pools: dict[int, tuple[Key, int]],
        target: str,
        conversation: str | None,
        held: bool,
        wait_seconds: Callable[[], float],
    ) -> None:
        self._ledger = ledger
        self._sizes = sizes
        self._pools = pools
        self._target = target
        self._conversation = conversation
        self._held = held
        self._wait_seconds = wait_seconds
        self._chars = 0
        self.prompt = 0
        self.tokens = 0
        self._lease: tuple[int, Key] | None = None
        #: Milliseconds this turn waited for room; 0 when it did not.
        self.waited_ms = 0

    def _pool_of(self, candidate: Any) -> tuple[Key, int] | None:
        return self._pools.get(id(candidate))

    def _reserve(self, candidate: Any) -> None:
        pool = self._pool_of(candidate)
        if pool is None:
            return
        self._ledger.take(pool[0], self.tokens)
        self._lease = (id(candidate), pool[0])

    async def admit(self, request: Any, tier: Sequence[Any]) -> list[Any]:
        """Wait for room, and answer the tier with the backend to try first."""
        self._chars = request_chars(request)
        seen = self._sizes.get(self._target, self._conversation) if self._conversation else None
        self.prompt, self.tokens = estimate(self._chars, getattr(request, "maxTokens", None), seen)
        ordered = list(tier)
        if not ordered:
            return ordered
        # Only the conversation's own replica counts while it holds it.
        waiting_on = ordered[:1] if self._held else ordered
        started = time.perf_counter()
        deadline = started + max(0.0, float(self._wait_seconds()))
        while True:
            for candidate in waiting_on:
                pool = self._pool_of(candidate)
                if pool is None or self._ledger.fits(pool[0], pool[1], self.tokens, self.prompt):
                    self._reserve(candidate)
                    self.waited_ms = int((time.perf_counter() - started) * 1000)
                    return [candidate, *(c for c in ordered if c is not candidate)]
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                # Waited long enough: the usual order, and the engine decides.
                self.waited_ms = int((time.perf_counter() - started) * 1000)
                return ordered
            await self._ledger.changed(remaining)

    def enter(self, candidate: Any) -> None:
        """An attempt is going to `candidate`: hold its room, and only its."""
        if self._lease is not None and self._lease[0] == id(candidate):
            return
        self._release()
        self._reserve(candidate)

    @property
    def chars(self) -> int:
        """The request's size as `admit` measured it."""
        return self._chars

    def _release(self) -> None:
        if self._lease is not None:
            self._ledger.give(self._lease[1], self.tokens)
            self._lease = None

    def close(self) -> None:
        self._release()
