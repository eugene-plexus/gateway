"""A3: bounded, persistent client-key policy; no signing material or fail-open list."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Literal

import httpx

from ._generated.client_key_models import ClientKeyPolicy
from ._http import internal_client

log = logging.getLogger(__name__)
DEFAULT_TTL_SECONDS = 15.0
DEFAULT_TIMEOUT_SECONDS = 4.0
DEFAULT_MAX_AGE_SECONDS = 60.0
CLOCK_TOLERANCE_SECONDS = 5.0
# How often a key the policy does not list may make the guard ask again.
# A key the root minted a moment ago is signed by the root and missing
# from a policy up to a refresh old; one re-read settles it. Bounded, so a
# stream of unknown keys cannot turn into a stream of reads.
MISS_REFRESH_SECONDS = 1.0
type Decision = Literal["allowed", "revoked", "unregistered", "unavailable"]


class ClientKeyGuard:
    def __init__(
        self,
        *,
        agent_url: str,
        service_token: str | None = None,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
        retry_seconds: float = 1.0,
        cache_file: Path | None = None,
        authority: str | None = None,
    ) -> None:
        self._agent_url = agent_url.rstrip("/")
        self._service_token = service_token
        self._ttl = ttl_seconds
        self._timeout = timeout_seconds
        self._max_age = max_age_seconds
        self._retry = retry_seconds
        self._cache_file = cache_file
        # A cached policy is only ever read back by the same authority
        # through the same agent: a re-enrolled node must not trust the
        # last install's list of who was turned off.
        self._scope = hashlib.sha256(
            (authority or "").encode() + b"\0" + self._agent_url.encode()
        ).hexdigest()
        self._policy: ClientKeyPolicy | None = None
        self._observed_revoked: set[str] = set()
        self._fetched_at: float | None = None
        self._initial_age = 0.0
        self._next_attempt = 0.0
        self._failures = 0
        self._task: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None
        self._last_warning = -math.inf
        self._last_miss_refresh = -math.inf
        # When the most recent policy read began: a request is answered by
        # a read that began after it arrived, never by an older one.
        self._last_fetch_started = -math.inf
        self._load()

    @property
    def revision(self) -> int | None:
        return self._policy.revision if self._policy else None

    @property
    def _revoked(self) -> frozenset[str]:
        stored = (
            {k.id for k in self._policy.keys if k.revokedAt is not None} if self._policy else set()
        )
        return frozenset(stored | self._observed_revoked)

    def _age(self) -> float:
        if self._policy is None:
            return math.inf
        wall_age = time.time() - self._policy.generatedAt
        if wall_age < -CLOCK_TOLERANCE_SECONDS:
            return math.inf
        elapsed = 0.0 if self._fetched_at is None else time.perf_counter() - self._fetched_at
        return max(wall_age, self._initial_age + elapsed)

    async def decision(self, key_id: str | None) -> Decision:
        arrived = time.perf_counter()
        if not key_id:
            return "unregistered"
        if key_id in self._revoked:
            return "revoked"
        await self._refresh_if_stale()
        if key_id in self._revoked:
            return "revoked"
        if self._policy is None or self._age() >= self._max_age:
            return "unavailable"
        if self._registered(key_id):
            return "allowed"
        # Not listed: most likely made a moment ago, after the policy this
        # gateway holds. Ask once before refusing; the 401 says "make a new
        # one", which would only be refused the same way.
        if not await self._refresh_for_miss(arrived):
            return "unregistered"
        if key_id in self._revoked:
            return "revoked"
        if self._policy is None or self._age() >= self._max_age:
            return "unavailable"
        return "allowed" if self._registered(key_id) else "unregistered"

    def _registered(self, key_id: str) -> bool:
        assert self._policy is not None
        return any(
            k.id == key_id and k.expiresAt.timestamp() > time.time() for k in self._policy.keys
        )

    async def _refresh_for_miss(self, arrived: float) -> bool:
        """Re-read the policy for a key it does not list; at most one miss read a second.

        The answer must come from a read that began after this request
        arrived -- a key made a moment ago is exactly what an older read
        cannot know. A request inside the window waits out the rest of it
        rather than being refused: a person who makes two keys in a row and
        uses both at once is the common case, and the first key's re-read
        must not cost the second one a 401. Concurrent misses share a read,
        so the bound holds, and an unknown key costs a second at most.
        """
        while True:
            if self._task is not None and not self._task.done():
                await asyncio.shield(self._task)
                continue
            if self._last_fetch_started >= arrived:
                return True
            now = time.perf_counter()
            if self._failures and now < self._next_attempt:
                return False
            remaining = MISS_REFRESH_SECONDS - (now - self._last_miss_refresh)
            if remaining > 0:
                await asyncio.sleep(remaining)
                continue
            self._last_miss_refresh = now
            self._start_fetch()

    def _start_fetch(self) -> None:
        self._last_fetch_started = time.perf_counter()
        self._task = asyncio.create_task(self._fetch())

    async def is_revoked(self, key_id: str | None) -> bool:
        """Legacy query helper; authorization uses the four-way decision."""
        return await self.decision(key_id) in ("revoked", "unregistered")

    async def _refresh_if_stale(self) -> None:
        now = time.perf_counter()
        if self._task is not None and not self._task.done():
            await asyncio.shield(self._task)
            return
        if now < self._next_attempt:
            return
        self._start_fetch()
        assert self._task is not None
        await asyncio.shield(self._task)

    @staticmethod
    def _parse(raw: object, *, check_clock: bool = True) -> ClientKeyPolicy:
        if not isinstance(raw, dict) or type(raw.get("revision")) is not int:
            raise ValueError("policy revision must be an integer")
        if type(raw.get("generatedAt")) not in (int, float):
            raise ValueError("policy timestamp must be numeric")
        policy = ClientKeyPolicy.model_validate(raw)
        if not math.isfinite(policy.generatedAt):
            raise ValueError("non-finite policy timestamp")
        if check_clock and policy.generatedAt > time.time() + CLOCK_TOLERANCE_SECONDS:
            raise ValueError("policy timestamp is in the future")
        ids = [key.id for key in policy.keys]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate policy identifier")
        for key in policy.keys:
            if key.expiresAt.utcoffset() is None or (
                key.revokedAt is not None and key.revokedAt.utcoffset() is None
            ):
                raise ValueError("policy timestamps need a timezone")
        return policy

    def _load(self) -> None:
        if self._cache_file is None or not self._cache_file.exists():
            return
        try:
            saved = json.loads(self._cache_file.read_text(encoding="utf-8"))
            if saved["scope"] != self._scope:
                raise ValueError("policy belongs to another trust key or agent")
            if type(saved["savedAt"]) not in (int, float) or not math.isfinite(saved["savedAt"]):
                raise ValueError("invalid cache timestamp")
            self._policy = self._parse(saved["policy"], check_clock=False)
            self._initial_age = (
                math.inf
                if saved["savedAt"] > time.time() + CLOCK_TOLERANCE_SECONDS
                else max(0.0, time.time() - self._policy.generatedAt)
            )
            self._fetched_at = time.perf_counter()
        except (OSError, ValueError, KeyError, TypeError):
            log.warning(
                "Client-key policy cache is unavailable or invalid; "
                "client access requires a fresh policy"
            )
            self._policy = None

    def _save(self, policy: ClientKeyPolicy) -> None:
        if self._cache_file is None:
            return
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._cache_file.with_suffix(self._cache_file.suffix + ".tmp")
        data = {
            "scope": self._scope,
            "savedAt": time.time(),
            "policy": policy.model_dump(mode="json", exclude_none=True),
        }
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self._cache_file)

    async def _fetch(self) -> None:
        headers = {"Authorization": f"Bearer {self._service_token}"} if self._service_token else {}
        try:
            async with asyncio.timeout(self._timeout):
                response = await self._ensure_client().get(
                    f"{self._agent_url}/v1/auth/client-keys/policy",
                    headers=headers,
                    timeout=self._timeout,
                )
            response.raise_for_status()
            fresh = self._parse(response.json())
            if time.time() - fresh.generatedAt > self._max_age:
                raise ValueError("authority policy is already stale")
            if self._policy is not None and fresh.authority == self._policy.authority:
                if fresh.revision < self._policy.revision:
                    raise ValueError("policy revision regressed")
                # A replay/rollback must never turn a known revoked identifier on.
                old = {
                    k.id: k
                    for k in self._policy.keys
                    if k.revokedAt is not None and k.expiresAt.timestamp() > time.time()
                }
                entries = {k.id: k for k in fresh.keys}
                entries.update(old)
                fresh = fresh.model_copy(update={"keys": list(entries.values())})
            self._observed_revoked.update(k.id for k in fresh.keys if k.revokedAt is not None)
            self._save(fresh)
        except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError, TimeoutError):
            self._failures += 1
            self._next_attempt = time.perf_counter() + min(
                15.0, self._retry * 2 ** min(self._failures - 1, 4)
            )
            if time.perf_counter() - self._last_warning >= 60:
                self._last_warning = time.perf_counter()
                log.warning(
                    "Client-key policy refresh failed; cached permission expires after %.1fs",
                    self._max_age,
                )
            return
        self._policy = fresh
        self._initial_age = max(0.0, time.time() - fresh.generatedAt)
        self._fetched_at = time.perf_counter()
        self._next_attempt = self._fetched_at + self._ttl
        self._failures = 0

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = internal_client()
        return self._client

    async def admission(self, body: dict[str, object]) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._service_token}"} if self._service_token else {}
        return await self._ensure_client().post(
            f"{self._agent_url}/v1/auth/client-keys/admission",
            json=body,
            headers=headers,
            timeout=4.0,
        )

    async def aclose(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._client is not None:
            await self._client.aclose()
