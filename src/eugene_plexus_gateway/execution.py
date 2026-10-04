"""Request execution shared by every driver operation.

The routing table supplies eligible tiers. This module owns attempt lifetimes,
circuits, cancellation, retry boundaries and accounting; protocol adapters own
payloads and stream semantics. An attempt is recorded exactly once.
"""

from __future__ import annotations

import sys
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._generated.driver_models import GenerateRequest
    from .driver_client import DriverClient, TieredClient


@dataclass(frozen=True)
class StreamPolicy:
    progress: Callable[[Any], bool] = lambda event: False
    complete: Callable[[Any], bool] = lambda event: False
    usage: Callable[[Any], Any] = lambda event: None
    eof_is_success: bool = True


DEFAULT_STREAM_POLICY = StreamPolicy()


class Attempt:
    def __init__(self, owner: TieredClient, candidate: DriverClient, tier: int) -> None:
        self.owner, self.candidate, self.tier = owner, candidate, tier
        self.driver = getattr(candidate, "name", None)
        self.node = getattr(candidate, "node", None)
        self.started = time.perf_counter()
        circuit = getattr(candidate, "circuit", None)
        self.probe_epoch = circuit.epoch if circuit is not None and circuit.probing else None
        self.runtime = (
            owner._hooks.on_attempt_start(self.driver, node=self.node)
            if owner._hooks is not None and self.driver
            else None
        )
        self.finished = False
        self.committed = False
        self.first_ms: int | None = None

    def commit(self) -> None:
        if not self.committed:
            self.committed = True
            self.first_ms = int((time.perf_counter() - self.started) * 1000)

    def publish(self) -> None:
        self.owner.served_by = self.driver
        self.owner.served_by_node = self.node
        self.owner.served_model = getattr(self.candidate, "public_model", None)
        self.owner.served_candidate = self.candidate
        self.owner.tier = self.tier

    def finish(
        self,
        *,
        served: bool = False,
        usage: Any = None,
        error: BaseException | None = None,
        incomplete: bool = False,
    ) -> None:
        if self.finished:
            return
        self.finished = True
        from .driver_client import RepetitionStopped, retry_disposition

        self.owner._finish_circuit(
            self.candidate,
            error if error else (RuntimeError("incomplete") if incomplete else None),
            probe_epoch=self.probe_epoch,
        )
        if self.owner._hooks is not None and self.driver:
            self.owner._hooks.on_attempt_end(
                self.driver,
                node=self.node,
                model=getattr(self.candidate, "public_model", None),
                runtime=self.runtime,
                served=served,
                usage=usage,
                elapsed_ms=int((time.perf_counter() - self.started) * 1000),
                first_ms=self.first_ms,
                error=type(error).__name__
                if error
                else ("IncompleteStream" if incomplete else None),
                retry_disposition=(
                    "terminal"
                    if isinstance(error, RepetitionStopped)
                    else "indeterminate"
                    if incomplete or (error and self.committed)
                    else retry_disposition(error)
                    if error
                    else None
                ),
            )


class Executor:
    def __init__(self, owner: TieredClient) -> None:
        self.owner = owner
        self.last_error: Exception | None = None

    async def attempts(self) -> AsyncIterator[Attempt]:
        from .driver_client import _cooling_error

        count = 0
        for tier, candidates in enumerate(self.owner._tiers, 1):
            for candidate in candidates:
                if self.owner.authorize_attempt is not None:
                    await self.owner.authorize_attempt()
                circuit = getattr(candidate, "circuit", None)
                if circuit is not None and not circuit.acquire():
                    if self.last_error is None:
                        self.last_error = _cooling_error(circuit.until - time.perf_counter())
                    continue
                count += 1
                self.owner.attempts = count
                yield Attempt(self.owner, candidate, tier)

    def retry(self, attempt: Attempt, label: str, error: BaseException) -> bool:
        from .driver_client import _is_cascade_eligible

        if attempt.committed or not isinstance(error, Exception) or not _is_cascade_eligible(error):
            return False
        self.last_error = error
        self.owner._log_cascade(
            label,
            self.owner.attempts - 1,
            attempt.candidate,
            error,
            total=len(self.owner.candidates),
        )
        return True

    async def whole[T](
        self,
        label: str,
        call: Callable[[DriverClient], Awaitable[T]],
        *,
        generation: GenerateRequest | None = None,
    ) -> T:
        async for attempt in self.attempts():
            try:
                if generation is not None:
                    self.owner._enter(attempt.candidate)
                result = await call(attempt.candidate)
            except BaseException as exc:
                attempt.finish(error=exc)
                if not self.retry(attempt, label, exc):
                    raise
            else:
                usage = getattr(result, "usage", None)
                attempt.finish(served=True, usage=usage)
                if generation is not None:
                    self.owner._served(usage, attempt.candidate, generation)
                attempt.publish()
                return result
        assert self.last_error is not None
        raise self.last_error

    async def stream[T](
        self,
        label: str,
        call: Callable[[DriverClient], AsyncGenerator[T, None]],
        *,
        policy: StreamPolicy = DEFAULT_STREAM_POLICY,
        generation: GenerateRequest | None = None,
    ) -> AsyncGenerator[T, None]:
        async for attempt in self.attempts():
            events = None
            complete = False
            usage = None
            try:
                if generation is not None:
                    self.owner._enter(attempt.candidate)
                events = call(attempt.candidate)
                async for event in events:
                    if not policy.progress(event):
                        attempt.commit()
                        attempt.publish()
                    if policy.complete(event):
                        complete = True
                        usage = policy.usage(event)
                    yield event
                served = complete or policy.eof_is_success
                attempt.finish(served=served, usage=usage, incomplete=not served)
                if served and generation is not None:
                    self.owner._served(usage, attempt.candidate, generation)
                attempt.publish()
                return
            except BaseException as exc:
                attempt.finish(error=exc)
                if not self.retry(attempt, label, exc):
                    raise
            finally:
                if not attempt.finished:
                    attempt.finish(error=sys.exc_info()[1] or RuntimeError("abandoned"))
                if events is not None:
                    await events.aclose()
        assert self.last_error is not None
        raise self.last_error
