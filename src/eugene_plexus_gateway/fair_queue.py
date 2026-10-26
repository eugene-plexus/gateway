"""One fair queue in the gateway (row 11, LD1).

Three waits used to be three mechanisms, or none: a client key over its
`maxConcurrentRequests` was refused with a 429, a turn with no room in its
runtime's shared context woke on any change and raced the others for it
(CB3), and a request that met a runtime still starting got a 503. They are
one queue now, one per routing table, and a request leaves it when two
things hold:

- **(a) its key** holds fewer leases on this gateway than its limit, and the
  root did not just refuse it for `concurrency` (another gateway holds the
  key's leases: it asks again at the next local release, or after a second);
- **(b) one of its seats** has room: a free slot of the `parallelSlots` its
  runtime's agent reports, and room in the `contextPoolTokens` it reports
  (`PoolLedger.fits`). A seat whose runtime reports neither always has room,
  and a request with no seats (every door but chat, and a cloud backend)
  needs only (a).

**The order.** By band (`interactive`, `background`, `sheddable`), then by
person in turns (the flow served longest ago goes first), then by arrival
within a person. One pass over the queue, run when a request arrives, when
an attempt ends, when a lease or room is given back, and a second after the
root refused a key.

**No overtaking on a runtime.** The first entry in that order that waits on
runtime R, with room under its key, claims R: if it does not fit R's pool,
nothing after it leaves on R, so a big turn is never starved by small ones.
An entry held back by its key claims nothing, so a key at its limit never
holds back another key.

**Bounds.** At most `maxQueuedRequests` wait for each key and person, and at
most `queueMaxRequests` in the gateway; past that a request of a higher band
takes the place of the newest one of the lowest band, which is refused as set
aside. After `queueWaitSeconds` an entry leaves with `Waited`, and the caller
applies `MOVE_RULES` (today: move to the slot's next tier) or refuses it.

Nothing here talks to the root or to a driver: `admission.take_turn` acquires
the key's lease when an entry leaves, and the routing table's clients send.
Every map is bounded by what waits or runs now, except `_turns`, which is
pruned to the flows that wait once it passes `TURNS_KEPT`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: A runtime, as the routing table keys it: `(node, name)`.
Key = tuple[str | None, str]

#: Highest first. `QueueBand` in `client-keys.yaml`.
BANDS: tuple[str, ...] = ("interactive", "background", "sheddable")
DEFAULT_BAND = "interactive"

#: The gateway settings' defaults (`config.py`), for a queue no request configured.
DEFAULT_WAIT_SECONDS = 60.0
DEFAULT_MAX_REQUESTS = 256
#: A key's `maxQueuedRequests` when its limits do not say.
DEFAULT_MAX_QUEUED = 8
#: How long a key the root refused for `concurrency` waits before it asks again,
#: unless a request of that key ends here first.
ROOT_RETRY_SECONDS = 1.0
#: The flows whose last turn is remembered, past which only those that wait are.
TURNS_KEPT = 4096


def rank(band: str | None) -> int:
    """0 for `interactive`, the highest; an unknown band is the highest."""
    return BANDS.index(band) if band in BANDS else 0


def lower(band: str | None, asked: str | None) -> str:
    """The band in effect: `band`, or `asked` when that is lower. A request may
    lower its own band and never raise it."""
    mine = band if band in BANDS else DEFAULT_BAND
    if asked in BANDS and rank(asked) > rank(mine):
        return str(asked)
    return mine


# --- refusals ---------------------------------------------------------------


@dataclass(frozen=True)
class Refusal:
    """Why the queue turned a request away: its `QueueRefusalCode`, the status
    and `Retry-After` it is answered with, and its words."""

    code: str
    status: int
    retry: int
    words: str


class Refused(Exception):
    def __init__(self, refusal: Refusal) -> None:
        self.refusal = refusal
        super().__init__(refusal.words)


def key_queue_full(running: int, waiting: int) -> Refusal:
    return Refusal(
        "key_queue_full",
        429,
        5,
        f"This key has {running} running and {waiting} waiting for you, its most. "
        "Retry when one finishes.",
    )


def queue_full(waiting: int) -> Refusal:
    return Refusal(
        "queue_full",
        503,
        5,
        f"Eugene has {waiting} requests waiting, its most. Retry in a few seconds.",
    )


def set_aside() -> Refusal:
    return Refusal("set_aside", 503, 10, "Set aside for a more urgent request. Retry later.")


def key_queue_timeout(seconds: float, limit: int | None) -> Refusal:
    at_once = f" ({limit} at once)" if limit is not None else ""
    return Refusal(
        "key_queue_timeout",
        429,
        5,
        f"Waited {seconds:g} s for this key's turn{at_once}. Retry, or raise the key's limit.",
    )


def queue_timeout(seconds: float, what: str, ahead: int) -> Refusal:
    return Refusal(
        "queue_timeout",
        503,
        10,
        f"Waited {seconds:g} s for room on {what}, {ahead} ahead. Retry shortly.",
    )


def still_starting(what: str, seconds: float) -> Refusal:
    return Refusal(
        "still_starting", 503, 10, f"{what} is still starting after {seconds:g} s. Retry shortly."
    )


def shutting_down() -> Refusal:
    return Refusal(
        "shutting_down", 503, 5, "Eugene's gateway is restarting. Retry in a few seconds."
    )


def rate_limited(per_minute: int | None, retry: int) -> Refusal:
    used = f"{per_minute} requests a minute" if per_minute else "requests a minute"
    return Refusal("rate_limited", 429, retry, f"This key's {used} are used. Retry in {retry} s.")


def runtime_words(model: str | None, node: str | None) -> str:
    """ "qwen3-8b on tower": what a person calls a runtime."""
    what = model or "this model"
    return f"{what} on {node}" if node else what


# --- moves (TIER-SPILL, built as rules) --------------------------------------


@dataclass(frozen=True)
class MoveRule:
    """What happens to an entry that waited its `queueWaitSeconds`: a
    condition and an action, kept as data so an operator can be given more of
    them later. The Metrics page counts every move (`tierMoves`)."""

    when: str
    then: str


#: TIER-SPILL, Troy 2026-10-21: waited for room past the queue's wait, so move
#: to the slot's next tier (which may be a cloud account the operator put
#: there as the fallback). The only rule today.
MOVE_RULES: tuple[MoveRule, ...] = (MoveRule(when="waited_for_room", then="next_tier"),)


def move_for(waited: Waited, *, has_next_tier: bool) -> str | None:
    """The action of the first rule whose condition holds, or None (refuse)."""
    conditions = {"waited_for_room": waited.for_room}
    possible = {"next_tier": has_next_tier}
    for rule in MOVE_RULES:
        if conditions.get(rule.when) and possible.get(rule.then):
            return rule.then
    return None


# --- what waits -------------------------------------------------------------


@dataclass(eq=False)
class Seat:
    """One place an entry may go: a candidate backend, and what gates it."""

    candidate: Any = None
    #: Its runtime, whose slots this gateway counts. None: a cloud backend.
    runtime: Key | None = None
    #: The runtime's `parallelSlots`, when its agent reports them.
    slots: int | None = None
    #: The runtime's shared pool, `(pool key, tokens)`, when its agent reports one.
    pool: tuple[Key, int] | None = None
    #: The model and node, for the words of a refusal and the Metrics page.
    model: str | None = None
    node: str | None = None


@dataclass(eq=False)
class Entry:
    """One request's place in line."""

    band: str = DEFAULT_BAND
    #: `person:<id>`, `key:<id>`, `session:<sub>` or `anonymous` (LD-C3, KEY-OWNER).
    fairness: str = "anonymous"
    #: The key's bucket (its lineage), for gate (a). None: no key, no gate.
    key: str | None = None
    #: The key's `maxConcurrentRequests`; None for a key with no limits.
    limit: int | None = None
    #: The key's `maxQueuedRequests`; None for no per-key bound.
    max_queued: int | None = None
    seats: list[Seat] = field(default_factory=list)
    prompt: int = 0
    tokens: int = 0
    wait_seconds: float = DEFAULT_WAIT_SECONDS
    max_requests: int = DEFAULT_MAX_REQUESTS
    #: The request already holds its key's lease (a later turn of a search loop,
    #: or a turn after a wake): gate (a) does not apply and nothing is counted.
    holds_key: bool = False
    #: Called when the entry leaves on a seat, to take that seat's pool room,
    #: and to give it back when the request returns to the line.
    reserve: Callable[[Seat], None] | None = None
    unreserve: Callable[[], None] | None = None
    #: The model asked for, for the words of a refusal with no seat.
    model: str | None = None
    arrived: float = field(default_factory=time.perf_counter)
    # Set by the queue.
    seq: int = 0
    joined: float | None = None
    queued_s: float = 0.0
    future: asyncio.Future[Ticket] | None = None
    blocked_by_room: bool = False
    ahead: int = 0

    @property
    def what(self) -> str:
        seat = next((s for s in self.seats if s.runtime is not None), None)
        if seat is not None:
            return runtime_words(seat.model or self.model, seat.node)
        return runtime_words(self.model, None)

    @property
    def queued_ms(self) -> int:
        return int(self.queued_s * 1000)


class Waited(Exception):
    """An entry waited its `queueWaitSeconds` and did not leave."""

    def __init__(self, entry: Entry) -> None:
        self.entry = entry
        self.for_room = entry.blocked_by_room
        super().__init__(self.refusal.words)

    @property
    def refusal(self) -> Refusal:
        entry = self.entry
        if self.for_room:
            return queue_timeout(entry.wait_seconds, entry.what, entry.ahead)
        return key_queue_timeout(entry.wait_seconds, entry.limit)


class Ticket:
    """What a request holds once it leaves the line: its key's place on this
    gateway, and the slot of the runtime it went to. Each part is given back
    once, on every path; the pool room is the turn budget's (`Entry.reserve`).
    """

    def __init__(self, queue: FairQueue, seat: Seat | None, key: str | None) -> None:
        self._queue = queue
        self.seat = seat
        self._key = key
        self._runtime = seat.runtime if seat is not None else None

    @property
    def runtime(self) -> Key | None:
        return self._runtime

    def move(self, runtime: Key | None) -> None:
        """An attempt went to another runtime (a cascade): hold its slot, and
        only its."""
        if runtime == self._runtime:
            return
        self.release_seat()
        if runtime is not None:
            self._queue._take_seat(runtime)
            self._runtime = runtime

    def release_seat(self) -> None:
        if self._runtime is not None:
            runtime, self._runtime = self._runtime, None
            self._queue._give_seat(runtime)

    def release_key(self) -> None:
        if self._key is not None:
            key, self._key = self._key, None
            self._queue._give_key(key)

    def release(self) -> None:
        self.release_seat()
        self.release_key()


class FairQueue:
    """One per routing table (`of`). All bookkeeping is synchronous on the
    event loop, so a check and the reservation that follows it cannot be split
    by another request."""

    def __init__(self) -> None:
        self._waiting: list[Entry] = []
        self._key_running: dict[str, int] = {}
        self._seat_running: dict[Key, int] = {}
        #: Keys the root refused for `concurrency`, until when.
        self._root_full: dict[str, float] = {}
        #: `(band, fairness)` -> the number of the turn it last had.
        self._turns: dict[tuple[str, str], int] = {}
        self._turn = 0
        self._seq = 0
        self._closed = False
        self._kicked = False
        #: The timer for the earliest key the root refused, re-armed by every pass.
        self._timer: asyncio.TimerHandle | None = None
        self._timer_at = 0.0
        self._starts: dict[Key, asyncio.Task[str | None]] = {}
        #: The pool ledger whose room the turns hold, when one is attached: what
        #: gate (b) asks `fits`.
        self.ledger: Any = None

    # --- reading ----------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    def waiting(self) -> int:
        return len(self._waiting)

    def running(self, runtime: Key) -> int:
        return self._seat_running.get(runtime, 0)

    def key_running(self, key: str) -> int:
        return self._key_running.get(key, 0)

    def snapshot(self) -> list[dict[str, Any]]:
        """What waits now, by runtime: `MetricsSummary.queue`."""
        groups: dict[Any, dict[str, Any]] = {}
        for entry in self._waiting:
            seat = next((s for s in entry.seats if s.runtime is not None), None)
            ident: Any = seat.runtime if seat is not None else ("model", entry.model)
            group = groups.get(ident)
            if group is None:
                group = groups[ident] = {
                    "model": (seat.model if seat is not None else None) or entry.model,
                    "runtime": seat.runtime[1] if seat is not None and seat.runtime else None,
                    "node": seat.node if seat is not None else None,
                    "waiting": dict.fromkeys(BANDS, 0),
                    "running": self.running(seat.runtime) if seat and seat.runtime else 0,
                    "slots": seat.slots if seat is not None else None,
                }
            band = entry.band if entry.band in BANDS else DEFAULT_BAND
            group["waiting"][band] += 1
        return list(groups.values())

    # --- waiting ----------------------------------------------------------

    async def wait(self, entry: Entry) -> Ticket:
        """Wait until `entry` may leave, and answer what it holds then.

        Raises `Refused` (the line is full, the request was set aside, the
        gateway is stopping) or `Waited` (it waited its `wait_seconds`). A
        cancelled wait leaves nothing behind: an entry still in line is
        removed, and one that was let go in the same instant gives back what
        it was given.
        """
        if self._closed:
            raise Refused(shutting_down())
        loop = asyncio.get_running_loop()
        entry.future = loop.create_future()
        first = entry.joined is None
        if first:
            self._seq += 1
            entry.seq = self._seq
            entry.joined = time.perf_counter()
        started = time.perf_counter()
        self._waiting.append(entry)
        try:
            self._dispatch()
            if first and not entry.future.done():
                self._bound(entry)
            joined = entry.joined if entry.joined is not None else started
            deadline = joined + entry.wait_seconds
            while not entry.future.done():
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                await asyncio.wait({entry.future}, timeout=remaining)
        except BaseException:
            entry.queued_s += time.perf_counter() - started
            self._withdraw(entry)
            raise
        entry.queued_s += time.perf_counter() - started
        if not entry.future.done():
            self._remove(entry)
            self.kick()
            raise Waited(entry)
        return entry.future.result()

    def undo(self, entry: Entry, ticket: Ticket) -> None:
        """An entry that was let go returns what it was given (its lease was
        refused, or it is going back in line)."""
        ticket.release()
        if entry.unreserve is not None:
            entry.unreserve()

    def root_full(self, key: str, retry_seconds: float = ROOT_RETRY_SECONDS) -> None:
        """The root refused `key` for `concurrency`: its entries keep their
        places and wait for a request of that key to end here, or for
        `retry_seconds`."""
        self._root_full[key] = time.perf_counter() + retry_seconds
        self._arm(self._root_full[key])

    def _arm(self, at: float) -> None:
        """Run a pass at `at` (a `perf_counter` time), or earlier. The loop's
        timer runs on another clock, so the pass it starts may come a little
        early: each pass arms it again for what is still held back."""
        if self._timer is not None and not self._timer.cancelled() and self._timer_at <= at:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._timer is not None:
            self._timer.cancel()
        self._timer_at = at
        delay = max(0.0, at - time.perf_counter()) + 0.02
        self._timer = loop.call_later(delay, self._fire)

    def _fire(self) -> None:
        self._timer = None
        self._dispatch()

    def close(self) -> None:
        """Refuse every waiter `shutting_down`, and every later arrival."""
        self._closed = True
        for entry in list(self._waiting):
            self._remove(entry)
            if entry.future is not None and not entry.future.done():
                entry.future.set_exception(Refused(shutting_down()))
        for task in list(self._starts.values()):
            task.cancel()
        self._starts.clear()
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def kick(self) -> None:
        """Run a dispatch pass soon: something was given back."""
        if self._kicked or not self._waiting:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._kicked = True
        loop.call_soon(self._dispatch)

    # --- a runtime that is starting (WAKE) ---------------------------------

    def starting(
        self, runtime: Key, poll: Callable[[], Awaitable[str | None]], seconds: float
    ) -> Awaitable[str | None]:
        """Wait for a runtime that is coming up: one poll for each runtime,
        shared by every request that meets it, until `poll` answers a status
        that is not coming up or `seconds` pass. Answers the last status."""
        task = self._starts.get(runtime)
        if task is None or task.done():
            task = asyncio.create_task(self._poll_start(poll, seconds), name=f"starting:{runtime}")
            self._starts[runtime] = task

            def forget(done: asyncio.Task[str | None], key: Key = runtime) -> None:
                if self._starts.get(key) is done:
                    self._starts.pop(key, None)

            task.add_done_callback(forget)
        return asyncio.shield(task)

    @staticmethod
    async def _poll_start(poll: Callable[[], Awaitable[str | None]], seconds: float) -> str | None:
        from .routing import COMING_UP

        deadline = time.perf_counter() + max(1.0, seconds)
        status: str | None = None
        while True:
            with contextlib.suppress(Exception):
                status = await poll()
            if status not in COMING_UP or time.perf_counter() >= deadline:
                return status
            await asyncio.sleep(min(1.0, max(0.0, deadline - time.perf_counter())))

    # --- the pass ---------------------------------------------------------

    def _ordered(self) -> list[Entry]:
        """By band; within a band, persons in turns (served longest ago
        first); within a person, by arrival."""
        flows: dict[tuple[str, str], list[Entry]] = {}
        for entry in self._waiting:
            flows.setdefault((entry.band, entry.fairness), []).append(entry)
        order: list[Entry] = []
        for band in sorted({flow[0] for flow in flows}, key=rank):
            mine = [(flow, entries) for flow, entries in flows.items() if flow[0] == band]
            for _flow, entries in mine:
                entries.sort(key=lambda e: (e.arrived, e.seq))
            mine.sort(key=lambda f: (self._turns.get(f[0], -1), f[1][0].arrived, f[1][0].seq))
            depth = max(len(entries) for _flow, entries in mine)
            for index in range(depth):
                for _flow, entries in mine:
                    if index < len(entries):
                        order.append(entries[index])
        return order

    def _key_ok(self, entry: Entry, now: float) -> bool:
        if entry.key is None or entry.holds_key:
            return True
        until = self._root_full.get(entry.key)
        if until is not None:
            if now < until:
                return False
            del self._root_full[entry.key]
        return entry.limit is None or self._key_running.get(entry.key, 0) < entry.limit

    def _has_room(self, seat: Seat, entry: Entry) -> bool:
        if (
            seat.slots is not None
            and seat.runtime is not None
            and self._seat_running.get(seat.runtime, 0) >= seat.slots
        ):
            return False
        if seat.pool is not None and self.ledger is not None:
            pool_key, pool = seat.pool
            if not self.ledger.fits(pool_key, pool, entry.tokens, entry.prompt):
                return False
        return True

    def _dispatch(self) -> None:
        self._kicked = False
        if not self._waiting:
            return
        now = time.perf_counter()
        claimed: set[Key] = set()
        waiting_on: dict[Key, int] = {}
        held_back: list[float] = []
        for entry in self._ordered():
            if entry.future is None or entry.future.done():
                continue
            if not self._key_ok(entry, now):
                entry.blocked_by_room = False
                if entry.key is not None and entry.key in self._root_full:
                    held_back.append(self._root_full[entry.key])
                continue
            seat = None
            for candidate in entry.seats:
                if candidate.runtime is not None and candidate.runtime in claimed:
                    continue
                if self._has_room(candidate, entry):
                    seat = candidate
                    break
            if seat is not None or not entry.seats:
                self._go(entry, seat)
                continue
            # No room on any seat: it claims them, so nothing after it in the
            # order leaves on them (no overtaking on a runtime).
            entry.blocked_by_room = True
            gated = [s.runtime for s in entry.seats if s.runtime is not None]
            entry.ahead = max((waiting_on.get(r, 0) for r in gated), default=0)
            for runtime in gated:
                claimed.add(runtime)
                waiting_on[runtime] = waiting_on.get(runtime, 0) + 1
        if held_back:
            self._arm(min(held_back))
        if len(self._turns) > TURNS_KEPT:
            keep = {(e.band, e.fairness) for e in self._waiting}
            self._turns = {flow: n for flow, n in self._turns.items() if flow in keep}

    def _go(self, entry: Entry, seat: Seat | None) -> None:
        self._remove(entry)
        key = None
        if entry.key is not None and not entry.holds_key:
            key = entry.key
            self._key_running[key] = self._key_running.get(key, 0) + 1
        if seat is not None and seat.runtime is not None:
            self._take_seat(seat.runtime)
        if seat is not None and entry.reserve is not None:
            entry.reserve(seat)
        self._turn += 1
        self._turns[(entry.band, entry.fairness)] = self._turn
        assert entry.future is not None
        entry.future.set_result(Ticket(self, seat, key))

    def _bound(self, entry: Entry) -> None:
        """The two bounds, for an entry that could not leave at once."""
        if entry.key is not None and entry.max_queued is not None:
            mine = [e for e in self._waiting if e.key == entry.key and e.fairness == entry.fairness]
            if len(mine) > entry.max_queued:
                self._remove(entry)
                raise Refused(key_queue_full(self._key_running.get(entry.key, 0), len(mine) - 1))
        if len(self._waiting) > entry.max_requests:
            others = [e for e in self._waiting if e is not entry]
            lowest = max(rank(e.band) for e in others)
            if rank(entry.band) < lowest:
                victim = max(
                    (e for e in others if rank(e.band) == lowest), key=lambda e: (e.arrived, e.seq)
                )
                self._remove(victim)
                if victim.future is not None and not victim.future.done():
                    victim.future.set_exception(Refused(set_aside()))
                self.kick()
            else:
                self._remove(entry)
                raise Refused(queue_full(len(others)))

    def _remove(self, entry: Entry) -> None:
        with contextlib.suppress(ValueError):
            self._waiting.remove(entry)

    def _withdraw(self, entry: Entry) -> None:
        if entry in self._waiting:
            self._remove(entry)
            self.kick()
            return
        future = entry.future
        if future is None or not future.done() or future.cancelled():
            return
        if future.exception() is None:
            self.undo(entry, future.result())

    def _take_seat(self, runtime: Key) -> None:
        self._seat_running[runtime] = self._seat_running.get(runtime, 0) + 1

    def _give_seat(self, runtime: Key) -> None:
        left = self._seat_running.get(runtime, 0) - 1
        if left > 0:
            self._seat_running[runtime] = left
        else:
            self._seat_running.pop(runtime, None)
        self.kick()

    def _give_key(self, key: str) -> None:
        left = self._key_running.get(key, 0) - 1
        if left > 0:
            self._key_running[key] = left
        else:
            self._key_running.pop(key, None)
        # A request of this key ended here, so a key the root refused asks again.
        self._root_full.pop(key, None)
        self.kick()


def of(owner: Any) -> FairQueue:
    """The queue of a routing table (or of anything else that holds one), made
    on first use. Kept on the owner, so it lives and closes with it, and its
    pool ledger is the owner's when it has one."""
    queue = getattr(owner, "fair_queue", None)
    if not isinstance(queue, FairQueue):
        queue = FairQueue()
        ledger = getattr(owner, "_ledger", None)
        if ledger is not None:
            queue.ledger = ledger
            listeners = getattr(ledger, "listeners", None)
            if isinstance(listeners, list):
                listeners.append(queue.kick)
        owner.fair_queue = queue
    return queue


__all__ = [
    "BANDS",
    "DEFAULT_BAND",
    "MOVE_RULES",
    "Entry",
    "FairQueue",
    "MoveRule",
    "Refusal",
    "Refused",
    "Seat",
    "Ticket",
    "Waited",
    "lower",
    "move_for",
    "of",
    "rank",
]
