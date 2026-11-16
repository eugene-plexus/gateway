"""What each runtime reads a prompt at, and how long an answer takes (row 14, LD3, DI5).

**Waiting for a warm replica or moving.** A conversation's own replica holds its
prompt, so a turn that goes back reads only its new text. When that replica has
no room, the turn can wait, or go to a replica with room and read the whole
prompt again. A 2,000-token chat turn should go at once; a 195,000-token agent
turn should wait a long time. Which is which is a time to first token on each
side:

    warm  = the wait for room on the home replica + the new text / its rate
    cold  = the whole prompt / the other replica's rate

and the turn moves only when `warm` is more than a penalty above `cold`
(`moveFirstTokenPenaltyMs`), as llm-d's "sticky until saturated" does. The
estimate needs no trained model:

* **The prefill rate** is each runtime's own peak: the 90th percentile of
  uncached prompt tokens over time to first token across its last `KEPT`
  requests that read at least `MIN_UNCACHED` uncached tokens. A request that
  waited in the engine reads slow, which the peak leaves out. A short prompt
  says nothing about prefill, because a request's fixed cost dominates it.
* **The wait** is the number waiting divided by the places there are, times the
  mean time an answer takes on that runtime (`wait_seconds`).

Per runtime, never pooled, because one model's replicas run on unlike machines.
Samples come from the retained metrics at start (`MetricsStore.prefill_samples`)
and from every answer after, and a runtime that leaves `ready` loses them: its
next engine may read at another speed. With either number missing, nothing is
predicted and a turn keeps waiting on its own replica, as CB3 had it.

**Not built:** DI5's fitted `a + b·uncached + c·uncached·cached`. LD3 says the
peak rate needs no predictor, and the first measurement is what decides whether
the quadratic term earns its keep.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

#: A runtime, as the routing table keys it: `(node, name)`.
Key = tuple[str | None, str]

#: A prompt reads its prefill rate only past this many uncached tokens.
MIN_UNCACHED = 256
#: The samples a runtime needs before its rate is said.
MIN_SAMPLES = 8
#: The samples kept for each runtime.
KEPT = 256
#: Which share of the rates is the peak: the 90th percentile.
PEAK = 0.90


class PrefillModel:
    """Per runtime: its prefill rates, and how long its answers take."""

    def __init__(self, *, kept: int = KEPT) -> None:
        self._kept = kept
        self._rates: dict[Key, deque[float]] = {}
        self._service: dict[Key, deque[float]] = {}

    def observe(
        self,
        runtime: Key,
        *,
        prompt: int | None,
        cached: int | None,
        first_ms: int | None,
        elapsed_ms: int | None,
    ) -> None:
        """One served request. Each part is kept only when it says something:
        the answer's time always, the prefill rate when the request read enough
        uncached tokens and said how many it cached."""
        if isinstance(elapsed_ms, int) and elapsed_ms > 0:
            self._service.setdefault(runtime, deque(maxlen=self._kept)).append(elapsed_ms / 1000.0)
        if (
            isinstance(prompt, int)
            and isinstance(cached, int)
            and isinstance(first_ms, int)
            and first_ms > 0
            and 0 <= cached <= prompt
            and prompt - cached >= MIN_UNCACHED
        ):
            self._rates.setdefault(runtime, deque(maxlen=self._kept)).append(
                (prompt - cached) / (first_ms / 1000.0)
            )

    def seed(
        self, rows: Iterable[tuple[Key, int | None, int | None, int | None, int | None]]
    ) -> None:
        """Retained requests, oldest first: `(runtime, prompt, cached, first_ms, elapsed_ms)`."""
        for runtime, prompt, cached, first_ms, elapsed_ms in rows:
            self.observe(
                runtime, prompt=prompt, cached=cached, first_ms=first_ms, elapsed_ms=elapsed_ms
            )

    def forget(self, runtime: Key) -> None:
        self._rates.pop(runtime, None)
        self._service.pop(runtime, None)

    def samples(self, runtime: Key) -> int:
        """How many prefill rates this runtime's calibration holds."""
        return len(self._rates.get(runtime, ()))

    def rate(self, runtime: Key) -> float | None:
        """Tokens a second, the peak. None until `MIN_SAMPLES` requests have said."""
        rates = self._rates.get(runtime)
        if rates is None or len(rates) < MIN_SAMPLES:
            return None
        ordered = sorted(rates)
        return ordered[min(len(ordered) - 1, max(0, math.ceil(PEAK * len(ordered)) - 1))]

    def service_seconds(self, runtime: Key) -> float | None:
        """The mean time an answer took on this runtime, or None before one has."""
        times = self._service.get(runtime)
        return sum(times) / len(times) if times else None

    def prefill_seconds(self, runtime: Key, uncached: int) -> float | None:
        """How long reading `uncached` tokens takes there, or None when unknown."""
        rate = self.rate(runtime)
        if rate is None or rate <= 0:
            return None
        return max(0, uncached) / rate


def wait_seconds(*, waiting: int, parallel: int, service_seconds: float) -> float:
    """How long the last of `waiting` turns waits for a place: a place frees
    about every `service_seconds / parallel`. The turn itself counts."""
    return max(1, waiting) * service_seconds / max(1, parallel)


def worth_moving(*, warm_ms: float, cold_ms: float, penalty_ms: float) -> bool:
    """llm-d's rule: leave the replica that holds the prompt only when waiting
    for it is predicted to take more than `penalty_ms` longer than reading the
    whole prompt elsewhere."""
    return warm_ms > cold_ms + penalty_ms


@dataclass
class WaitOrMove:
    """One held turn's choice, asked at every pass of the fair queue (row 14).

    Built by `RoutingTable.wait_or_move` only when it can predict: the home
    replica has a calibrated rate and an answer time, and the turn knows what
    its conversation holds there. `queue` is the fair queue, whose line this
    turn waits in; `home` the seat that holds the conversation's prompt.
    """

    model: PrefillModel
    queue: Any
    home: Any
    #: The whole prompt, in tokens, as the budget estimates it.
    prompt: int
    #: What the home replica holds of it: the conversation's last prompt.
    held: int
    penalty_ms: float

    def warm_ms(self) -> float | None:
        """Time to first token if the turn waits for its own replica: the wait
        for a place, then reading only the text it does not hold there."""
        runtime = self.home.runtime
        service = self.model.service_seconds(runtime)
        read = self.model.prefill_seconds(runtime, self.prompt - self.held)
        if service is None or read is None:
            return None
        waited = wait_seconds(
            waiting=self.queue.waiting_for(runtime),
            parallel=self.home.slots or max(1, self.queue.running(runtime)),
            service_seconds=service,
        )
        return (waited + read) * 1000.0

    def cold_ms(self, seat: Any) -> float | None:
        """Time to first token on `seat`, which has room now: reading the whole
        prompt. A replica not measured yet is read at the home replica's rate."""
        read = self.model.prefill_seconds(seat.runtime, self.prompt)
        if read is None:
            read = self.model.prefill_seconds(self.home.runtime, self.prompt)
        return None if read is None else read * 1000.0

    def allows(self, seat: Any) -> bool:
        warm = self.warm_ms()
        cold = self.cold_ms(seat)
        if warm is None or cold is None:
            return False
        return worth_moving(warm_ms=warm, cold_ms=cold, penalty_ms=self.penalty_ms)
