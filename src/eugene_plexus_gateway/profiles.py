"""Bounded, last-known-good Library profile reads for generation defaults.

Paths are Library paths from runtime declarations, never aliases or local
model copies. The agent resolves the Library location; only our service
credential crosses that edge. A missing profile is a successful empty result.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from ._generated.library_models import ModelProfileList
from ._http import internal_client

log = logging.getLogger(__name__)
_MAX_ENTRIES = 256
_RETRY_SECONDS = 5.0


@dataclass(frozen=True)
class _Entry:
    values: dict[str, Any]
    fetched_at: float | None
    retry_at: float = 0.0
    failed: bool = False


class ProfileDefaults:
    def __init__(
        self,
        agent_url: str,
        service_token: str | None,
        get_config: Callable[[str], Any],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._base = agent_url.rstrip("/") + "/api/proxy/library"
        self._config = get_config
        self._clock = clock
        headers = {"Authorization": f"Bearer {service_token}"} if service_token else {}
        self._client = internal_client(
            headers=headers, timeout=httpx.Timeout(2.0, connect=1.0), transport=transport
        )
        self._cache: OrderedDict[str, _Entry] = OrderedDict()
        self._pending: dict[str, asyncio.Task[_Entry]] = {}

    def _seconds(self, key: str, default: float) -> float:
        value = self._config(key)
        return default if value is None else max(0.0, float(value))

    def _usable(self, entry: _Entry) -> dict[str, Any]:
        if entry.fetched_at is None:
            return {}
        age = self._clock() - entry.fetched_at
        fresh = self._seconds("profileCacheSeconds", 30)
        stale = self._seconds("profileMaxStaleSeconds", 300)
        return dict(entry.values) if age < fresh + stale else {}

    async def get(self, model_path: str | None) -> dict[str, Any]:
        if not model_path:
            return {}
        now = self._clock()
        previous = self._cache.get(model_path)
        if previous is not None:
            self._cache.move_to_end(model_path)
            if previous.failed and now < previous.retry_at:
                return self._usable(previous)
            if (
                not previous.failed
                and previous.fetched_at is not None
                and now - previous.fetched_at < self._seconds("profileCacheSeconds", 30)
            ):
                return dict(previous.values)
        task = self._pending.get(model_path)
        if task is None:
            task = asyncio.create_task(self._refresh(model_path, previous))
            self._pending[model_path] = task
            task.add_done_callback(lambda done: self._pending.pop(model_path, None))
        # One cancelled inference must not cancel the lookup another awaits.
        entry = await asyncio.shield(task)
        return self._usable(entry) if entry.failed else dict(entry.values)

    async def _refresh(self, path: str, previous: _Entry | None) -> _Entry:
        try:
            values = await self._read(path)
            entry = _Entry(values, self._clock())
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            entry = _Entry(
                previous.values if previous else {},
                previous.fetched_at if previous else None,
                retry_at=self._clock() + _RETRY_SECONDS,
                failed=True,
            )
            log.warning(
                "Library profile lookup failed for %s (%s); using %s",
                path,
                type(exc).__name__,
                "cached profile defaults" if self._usable(entry) else "gateway defaults",
            )
        self._cache[path] = entry
        self._cache.move_to_end(path)
        while len(self._cache) > _MAX_ENTRIES:
            self._cache.popitem(last=False)
        return entry

    async def _read(self, path: str) -> dict[str, Any]:
        response = await self._client.get(f"{self._base}/v1/models", params={"path": path})
        response.raise_for_status()
        models = response.json()["models"]
        if not isinstance(models, list) or len(models) > 1:
            raise ValueError("Library reverse lookup did not identify one model")
        if not models:
            return {}
        model_id = models[0]["id"]
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("Library model id is absent")
        response = await self._client.get(
            f"{self._base}/v1/models/{quote(model_id, safe='')}/profiles"
        )
        response.raise_for_status()
        profiles = ModelProfileList.model_validate(response.json()).profiles
        defaults = [profile for profile in profiles if profile.default]
        if len(defaults) > 1:
            raise ValueError("Library returned more than one default profile")
        if not defaults:
            return {}
        return defaults[0].model_dump(
            include={"maxTokens", "temperature", "topP"}, exclude_none=True
        )

    async def aclose(self) -> None:
        pending = list(self._pending.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await self._client.aclose()
