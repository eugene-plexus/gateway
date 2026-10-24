"""Saved Library answers and observations, independent of routing."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)
WRITE_SECONDS = 60.0


def is_outage(exc: Exception) -> bool:
    return isinstance(exc, httpx.TransportError) or (
        isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {502, 503, 504}
    )


def failure_cause(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "the Library check timed out"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"the Library proxy answered HTTP {exc.response.status_code}"
    return str(exc) or type(exc).__name__


class LibraryCache:
    def __init__(
        self, path: Path | None = None, *, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        self.path = path
        self._clock = clock
        self.profiles: dict[str, Any] = {}
        self.special_tokens: dict[str, Any] = {}
        self.reachable: bool | None = None
        self.since: datetime | None = None
        self.cause: str | None = None
        self._dirty = False
        self._written: float | None = None
        self._writer: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._writing = False
        self._closing = False
        if path is not None:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self.profiles = self._section(raw.get("profiles"))
                    self.special_tokens = self._section(raw.get("specialTokens"))
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as exc:
                log.warning("could not load saved Library settings from %s: %s", path, exc)

    @staticmethod
    def _section(value: Any) -> dict[str, Any]:
        return dict(list(value.items())[-256:]) if isinstance(value, dict) else {}

    def observe(self, exc: Exception | None = None) -> None:
        if exc is not None and is_outage(exc):
            if self.reachable is not False:
                self.since = datetime.now(UTC)
            self.reachable = False
            self.cause = failure_cause(exc)
        else:
            self.reachable = True
            self.since = None
            self.cause = None

    def update(self, section: str, path: str, value: Any) -> None:
        entries = self.profiles if section == "profiles" else self.special_tokens
        held = entries.get(path)
        unchanged = isinstance(held, dict) and held.get("value") == value
        entries.pop(path, None)
        entries[path] = {"value": value, "fetchedAt": time.time()}
        if unchanged:
            return
        while len(entries) > 256:
            entries.pop(next(iter(entries)))
        self._dirty = True
        if self.path is not None and self._writer is None:
            self._writer = asyncio.create_task(self._write_later())

    async def _write_later(self) -> None:
        try:
            while self._dirty and not self._closing:
                delay = (
                    0.0
                    if self._written is None
                    else WRITE_SECONDS - (self._clock() - self._written)
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                await self.flush()
        finally:
            self._writer = None

    async def flush(self, *, force: bool = False) -> None:
        async with self._lock:
            await self._flush(force=force)

    async def _flush(self, *, force: bool = False) -> None:
        if self.path is None or not self._dirty:
            return
        if (
            not force
            and self._written is not None
            and self._clock() - self._written < WRITE_SECONDS
        ):
            return
        # Snapshot before yielding; changes during the write remain dirty.
        payload = json.dumps({"profiles": self.profiles, "specialTokens": self.special_tokens})
        self._dirty = False
        self._written = self._clock()
        self._writing = True
        try:
            await asyncio.to_thread(self._write, payload)
        except OSError as exc:
            self._dirty = True
            log.warning("could not save Library settings to %s: %s", self.path, exc)
        finally:
            self._writing = False

    def _write(self, payload: str) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(".json.tmp")
        with partial.open("w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, self.path)

    async def aclose(self) -> None:
        self._closing = True
        if self._writer is not None:
            # A write already in progress must complete before another starts.
            if self._writing:
                await self._writer
            else:
                self._writer.cancel()
                await asyncio.gather(self._writer, return_exceptions=True)
        # A stop keeps what changed in the last minute too.
        await self.flush(force=True)
