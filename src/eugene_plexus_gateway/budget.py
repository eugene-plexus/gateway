"""A turn holds room in its replica's shared context (CB3, CB-C2).

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
conversation), plus room for its answer. The 10% spare is because characters
over 3.5 let two overflows through on a pool that holds two prompts.

**The answer's room (CB-C2, Choice 3, Troy 2026-10-21).** Not the whole
`max_tokens` (Claude Code sends 32,000 and writes a few hundred): the lower of
`max_tokens` and the length 95 of 100 of its conversation's last 16 answers
were shorter than, or 4,096 tokens for a new conversation, and never less than
the owner's floor (`answerReserveFloorTokens`, default 1,024). An answer longer
than its room is counted (`outranReserve`); the 10% spare absorbs some of it.

**The wait is the fair queue's (row 11).** A turn that does not fit waits in
`fair_queue`, by band, person and arrival, for room on its own replica while
it holds the conversation (going elsewhere would read its whole history
again), else on whichever replica of the tier has room first; the ledger tells
the queue each time room is given back. A turn alone on its runtime always
fits, and a request bigger than the whole pool is sent at once: the engine's
own refusal is the honest answer to it.
"""

from __future__ import annotations

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
#: CB-C2: the room a new conversation holds for its answer.
NEW_CONVERSATION_ANSWER = 4096
#: CB-C2: how many of a conversation's answers its room is judged from.
ANSWERS_KEPT = 16
#: CB-C2: the least room an answer holds when the gateway's setting does not say.
DEFAULT_ANSWER_FLOOR = 1024

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
    #: Its last answers' lengths in tokens, oldest first (CB-C2).
    answers: tuple[int, ...] = ()


class ConversationSizes:
    """Model and conversation key -> its last prompt, as the engine counted it,
    and its last `ANSWERS_KEPT` answers.

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

    def put(
        self, target: str, key: str, prompt_tokens: int, chars: int, answer: int | None = None
    ) -> None:
        with self._lock:
            before = self._entries.get((target, key))
            answers = before.answers if before is not None else ()
            if answer is not None and answer >= 0:
                answers = (*answers, answer)[-ANSWERS_KEPT:]
            self._entries[(target, key)] = _Seen(prompt_tokens, chars, time.perf_counter(), answers)
            self._entries.move_to_end((target, key))
            while len(self._entries) > self._size:
                self._entries.popitem(last=False)


def long_answer(answers: Sequence[int]) -> int | None:
    """The length 95 of 100 of `answers` are no longer than (nearest rank)."""
    if not answers:
        return None
    ordered = sorted(answers)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))]


def answer_room(
    max_tokens: int | None, seen: _Seen | None, floor: int = DEFAULT_ANSWER_FLOOR
) -> int:
    """CB-C2: the room a turn holds for its answer."""
    typical = long_answer(seen.answers) if seen is not None else None
    room = max(max(0, floor), typical if typical is not None else NEW_CONVERSATION_ANSWER)
    if max_tokens is not None and max_tokens > 0:
        room = min(room, max_tokens)
    return room


def estimate(
    chars: int, max_tokens: int | None, seen: _Seen | None, *, floor: int = DEFAULT_ANSWER_FLOOR
) -> tuple[int, int]:
    """`(prompt, total)`: the tokens one turn's prompt is, and what the turn
    holds in its runtime's pool while it answers (its prompt and its room)."""
    if seen is not None and chars >= seen.chars:
        prompt = seen.prompt_tokens + (chars - seen.chars) / CHARS_PER_TOKEN
    else:
        prompt = chars / CHARS_PER_TOKEN
    tokens = math.ceil(prompt)
    return tokens, tokens + answer_room(max_tokens, seen, floor)


class PoolLedger:
    """The prompt tokens in flight on each runtime, install-wide.

    One per routing table. Synchronous bookkeeping on the event loop, so a
    check and the reservation that follows it cannot be split by another
    request. `listeners` are told whenever room is given back: the fair
    queue's dispatch (row 11), which is where a turn waits for it.
    """

    def __init__(self) -> None:
        self._tokens: dict[Key, int] = {}
        self.listeners: list[Callable[[], None]] = []

    def in_flight(self, runtime: Key) -> int:
        return self._tokens.get(runtime, 0)

    def fits(self, runtime: Key, pool: int, tokens: int, prompt: int) -> bool:
        """Whether a turn holding `tokens` (its prompt `prompt`) fits now.

        Alone it always goes. A PROMPT bigger than the pool goes at once,
        because the engine refuses it outright and so takes nothing from
        anyone; judged on the prompt and not on `tokens`, since a turn's
        answer room can make a prompt that runs fine look bigger than the
        pool, and sent at once it would overflow the turns already there.
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
        for listener in self.listeners:
            listener()


class TurnBudget:
    """One turn's place in the ledger, across the attempts it makes.

    Built by `RoutingTable.pick` when any backend in the slot reports a
    shared pool; held by the `TieredClient`. `size` measures the turn;
    the fair queue reserves its room where it leaves for (`reserve_on`);
    `enter` moves the reservation to whichever backend an attempt goes to;
    `close` gives it back, on every path. `admit` places a turn that no
    queue holds (a client built by hand): first where it fits now, waiting
    for nothing.
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
        floor: int = DEFAULT_ANSWER_FLOOR,
    ) -> None:
        self._ledger = ledger
        self._sizes = sizes
        self._pools = pools
        self._target = target
        self._conversation = conversation
        self._held = held
        #: The least room an answer holds (`answerReserveFloorTokens`).
        self.floor = floor
        self._chars = 0
        self.prompt = 0
        self.tokens = 0
        self._lease: tuple[int, Key] | None = None

    @property
    def held(self) -> bool:
        """Whether the turn's conversation holds its first replica (an
        affinity hit), so it waits only on that one."""
        return self._held

    @property
    def room(self) -> int:
        """The tokens this turn holds for its answer."""
        return self.tokens - self.prompt

    def pool_of(self, candidate: Any) -> tuple[Key, int] | None:
        return self._pools.get(id(candidate))

    def _reserve(self, candidate: Any) -> None:
        pool = self.pool_of(candidate)
        if pool is None:
            return
        self._ledger.take(pool[0], self.tokens)
        self._lease = (id(candidate), pool[0])

    def _sized(self, request: Any) -> tuple[int, int, int]:
        """`(chars, prompt, tokens)` of one turn, as the estimate reads it."""
        chars = request_chars(request)
        seen = self._sizes.get(self._target, self._conversation) if self._conversation else None
        prompt, tokens = estimate(
            chars, getattr(request, "maxTokens", None), seen, floor=self.floor
        )
        return chars, prompt, tokens

    def size(self, request: Any) -> None:
        """Measure this turn: what it reserves where it goes."""
        self._chars, self.prompt, self.tokens = self._sized(request)

    def waiting_on(self, tier: Sequence[Any]) -> list[Any]:
        """The replicas this turn waits for room on: its own while it holds
        the conversation, else any of the tier."""
        return list(tier[:1] if self._held else tier)

    def _with_room(self, waiting_on: Sequence[Any], prompt: int, tokens: int) -> Any | None:
        """The first of `waiting_on` a turn this size fits on now, if any."""
        for candidate in waiting_on:
            pool = self.pool_of(candidate)
            if pool is None or self._ledger.fits(pool[0], pool[1], tokens, prompt):
                return candidate
        return None

    def first(self, request: Any, tier: Sequence[Any]) -> Any | None:
        """The backend with room for this turn this instant, or None.

        **A read**: nothing is reserved and nothing waits, so a stream's head can
        ask it before anything is sent. None when nothing has room now, which is
        when the turn waits in the queue, and who gets room first is not known
        until it has: the tier's usual order is then the best answer.
        """
        if not tier:
            return None
        _chars, prompt, tokens = self._sized(request)
        return self._with_room(self.waiting_on(tier), prompt, tokens)

    async def admit(self, request: Any, tier: Sequence[Any]) -> list[Any]:
        """A turn no queue holds: sized, put first where it fits now and its
        room reserved there; the usual order when nothing has room."""
        self.size(request)
        ordered = list(tier)
        candidate = self._with_room(self.waiting_on(ordered), self.prompt, self.tokens)
        if candidate is None:
            return ordered
        self._reserve(candidate)
        return [candidate, *(c for c in ordered if c is not candidate)]

    def reserve_on(self, candidate: Any) -> None:
        """The queue let this turn go to `candidate`: hold its room there."""
        self._release()
        self._reserve(candidate)

    def unreserve(self) -> None:
        """The turn went back in line: give its room back."""
        self._release()

    def enter(self, candidate: Any) -> None:
        """An attempt is going to `candidate`: hold its room, and only its."""
        if self._lease is not None and self._lease[0] == id(candidate):
            return
        self._release()
        self._reserve(candidate)

    @property
    def chars(self) -> int:
        """The request's size as `size` measured it."""
        return self._chars

    @property
    def reserved(self) -> bool:
        return self._lease is not None

    def _release(self) -> None:
        if self._lease is not None:
            self._ledger.give(self._lease[1], self.tokens)
            self._lease = None

    def close(self) -> None:
        self._release()
