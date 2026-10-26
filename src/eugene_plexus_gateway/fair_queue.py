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
from collections.abc import Awaitable, Callable, Iterator
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

    @property
    def gated(self) -> bool:
        """Whether anything here is counted: slots or a pool."""
        return self.slots is not None or self.pool is not None


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
    #: The runtime it waits to finish starting (WAKE): no pass lets it go.
    starting: Key | None = None

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
    by another request.

    What waits is kept in arrival order with counters beside it (by key, by
    key and person, by runtime), so joining, leaving and the bounds cost the
    same however long the line is, and a request that competes with nobody
    leaves without a pass over it (`_free_now`).
    """

    def __init__(self) -> None:
        self._waiting: dict[Entry, None] = {}
        #: Waiting entries by key, by key and person, and by the runtimes they wait on.
        self._waiting_keys: dict[str, int] = {}
        self._waiting_flows: dict[tuple[str, str], int] = {}
        self._waiting_runtimes: dict[Key, int] = {}
        self._key_running: dict[str, int] = {}
        self._seat_running: dict[Key, int] = {}
        #: Keys the root refused for `concurrency`, until when. Only while some
        #: entry of the key waits: forgotten when the last one leaves.
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

    def _join(self, entry: Entry) -> asyncio.Future[Ticket]:
        """Put `entry` in line, under both bounds. Raises `Refused`."""
        if self._closed:
            raise Refused(shutting_down())
        future: asyncio.Future[Ticket] = asyncio.get_running_loop().create_future()
        entry.future = future
        if entry.joined is None:
            self._seq += 1
            entry.seq = self._seq
            entry.joined = time.perf_counter()
        self._add(entry)
        # Every join is bounded, a return after a root refusal too: five that
        # left together and came back together are five waiting.
        self._bound(entry)
        return future

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
        started = time.perf_counter()
        if entry.joined is None and self._free_now(entry):
            # It competes with nobody waiting: no pass over the line.
            entry.future = asyncio.get_running_loop().create_future()
            self._seq += 1
            entry.seq, entry.joined = self._seq, started
            self._go(entry, self._room_now(entry))
            return entry.future.result()
        future = self._join(entry)
        try:
            self._dispatch()
            joined = entry.joined if entry.joined is not None else started
            deadline = joined + entry.wait_seconds
            while not future.done():
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                await asyncio.wait({future}, timeout=remaining)
        except BaseException:
            entry.queued_s += time.perf_counter() - started
            self._withdraw(entry)
            raise
        entry.queued_s += time.perf_counter() - started
        if not future.done():
            self._remove(entry)
            self.kick()
            raise Waited(entry)
        return future.result()

    async def wait_for_start(
        self,
        entry: Entry,
        runtime: Key,
        poll: Callable[[], Awaitable[str | None]],
        seconds: float,
    ) -> str | None:
        """Wait in line for a runtime that is coming up (WAKE).

        The entry is counted by both bounds and by the Metrics page, and a
        stop refuses it `shutting_down`; no pass lets it go, because what it
        waits for is the one poll this runtime shares (`starting`). Answers the
        runtime's last status. Raises `Refused`.
        """
        entry.starting = runtime
        started = time.perf_counter()
        future = self._join(entry)
        polled = self.starting(runtime, poll, seconds)
        try:
            either: set[asyncio.Future[Any]] = {future, polled}
            await asyncio.wait(either, return_when=asyncio.FIRST_COMPLETED)
            if future.done():
                future.result()  # set aside, or the gateway stopping: raises `Refused`
            return await polled
        finally:
            entry.queued_s += time.perf_counter() - started
            if entry in self._waiting:
                self._remove(entry)
            if not polled.done():
                polled.cancel()  # the shield: the shared poll goes on for the others

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
        """Refuse every waiter `shutting_down`, those waiting for a starting
        runtime too, and every later arrival."""
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
    ) -> asyncio.Future[str | None]:
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

    # --- the line's bookkeeping ---------------------------------------------

    @staticmethod
    def _gated(entry: Entry) -> list[Key]:
        return [s.runtime for s in entry.seats if s.runtime is not None and s.gated]

    def _add(self, entry: Entry) -> None:
        self._waiting[entry] = None
        if entry.key is not None:
            self._waiting_keys[entry.key] = self._waiting_keys.get(entry.key, 0) + 1
            flow = (entry.key, entry.fairness)
            self._waiting_flows[flow] = self._waiting_flows.get(flow, 0) + 1
        for runtime in self._gated(entry):
            self._waiting_runtimes[runtime] = self._waiting_runtimes.get(runtime, 0) + 1

    def _remove(self, entry: Entry) -> None:
        if entry not in self._waiting:
            return
        del self._waiting[entry]
        if entry.key is not None:
            _drop(self._waiting_keys, entry.key)
            _drop(self._waiting_flows, (entry.key, entry.fairness))
            if entry.key not in self._waiting_keys:
                # Nobody of this key waits: what the root said of it is moot.
                self._root_full.pop(entry.key, None)
        for runtime in self._gated(entry):
            _drop(self._waiting_runtimes, runtime)

    def _free_now(self, entry: Entry) -> bool:
        """Whether `entry` may leave at once with no pass: nothing waiting
        competes with it, for its key or for any runtime it may use."""
        if self._closed or entry.starting is not None:
            return False
        competes = entry.key is not None and not entry.holds_key
        if competes and (
            self._waiting_keys.get(str(entry.key)) or not self._key_ok(entry, time.perf_counter())
        ):
            return False
        seat = self._room_now(entry)
        if entry.seats and seat is None:
            return False
        return seat is None or seat.runtime is None or not self._waiting_runtimes.get(seat.runtime)

    def _room_now(self, entry: Entry) -> Seat | None:
        """The first seat of `entry` with room now that nobody waiting claims."""
        for seat in entry.seats:
            if seat.runtime is not None and seat.gated and self._waiting_runtimes.get(seat.runtime):
                continue
            if self._has_room(seat, entry):
                return seat
        return None

    # --- the pass ---------------------------------------------------------

    def _ordered(self) -> list[Entry]:
        """By band; within a band, persons in turns (served longest ago
        first); within a person, by arrival. Work follows the entries: each
        round takes one entry from each person who still has one."""
        flows: dict[tuple[str, str], list[Entry]] = {}
        for entry in sorted(self._waiting, key=lambda e: (e.arrived, e.seq)):
            flows.setdefault((entry.band, entry.fairness), []).append(entry)
        order: list[Entry] = []
        for band in sorted({flow[0] for flow in flows}, key=rank):
            mine = [(flow, entries) for flow, entries in flows.items() if flow[0] == band]
            mine.sort(key=lambda f: (self._turns.get(f[0], -1), f[1][0].arrived, f[1][0].seq))
            active = [iter(entries) for _flow, entries in mine]
            while active:
                still: list[Iterator[Entry]] = []
                for line in active:
                    nxt = next(line, None)
                    if nxt is not None:
                        order.append(nxt)
                        still.append(line)
                active = still
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
            if entry.future is None or entry.future.done() or entry.starting is not None:
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
        # What the root refused is kept only for keys still waiting, until due.
        for key in [k for k, until in self._root_full.items() if k not in self._waiting_keys]:
            del self._root_full[key]
        if held_back:
            self._arm(min(held_back))

    def _go(self, entry: Entry, seat: Seat | None) -> None:
        """Let `entry` leave on `seat`: count its key and its slot, take its
        room, and hand it the ticket. A reservation that fails gives back what
        was taken, and the entry's wait ends with that failure."""
        self._remove(entry)
        assert entry.future is not None
        key = None
        took_seat = False
        try:
            if entry.key is not None and not entry.holds_key:
                self._key_running[entry.key] = self._key_running.get(entry.key, 0) + 1
                key = entry.key
            if seat is not None and seat.runtime is not None:
                self._take_seat(seat.runtime)
                took_seat = True
            if seat is not None and entry.reserve is not None:
                entry.reserve(seat)
        except Exception as exc:
            if took_seat and seat is not None and seat.runtime is not None:
                self._give_seat(seat.runtime)
            if key is not None:
                self._give_key(key)
            if not entry.future.done():
                entry.future.set_exception(exc)
            return
        self._turn += 1
        # Least recently served first in the dict, so the history keeps its bound
        # on every path, the fast one too, by dropping who was served longest ago
        # (who, waiting again, still goes first: never served sorts first).
        flow = (entry.band, entry.fairness)
        self._turns.pop(flow, None)
        self._turns[flow] = self._turn
        while len(self._turns) > TURNS_KEPT:
            del self._turns[next(iter(self._turns))]
        entry.future.set_result(Ticket(self, seat, key))

    def _bound(self, entry: Entry) -> None:
        """The two bounds, for an entry that has just joined the line."""
        if entry.key is not None and entry.max_queued is not None:
            mine = self._waiting_flows.get((entry.key, entry.fairness), 0)
            if mine > entry.max_queued:
                self._remove(entry)
                raise Refused(key_queue_full(self._key_running.get(entry.key, 0), mine - 1))
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
        _drop(self._seat_running, runtime)
        self.kick()

    def _give_key(self, key: str) -> None:
        _drop(self._key_running, key)
        # A request of this key ended here, so a key the root refused asks again.
        self._root_full.pop(key, None)
        self.kick()


def _drop[T](counts: dict[T, int], key: T) -> None:
    left = counts.get(key, 0) - 1
    if left > 0:
        counts[key] = left
    else:
        counts.pop(key, None)


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
