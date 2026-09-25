"""Which token this gateway presents, and to whom (per-node token keys, D8).

The gateway holds **no key of any kind** (2026-09-25). It is spawned
with one token, addressed to its own machine and good nowhere else, and
it presents that to everything on this machine: its own agent, the
drivers beside it, the library through the agent's proxy.

For another machine it asks its own agent, with that local token, for a
fifteen-minute one addressed to that machine alone (`POST
/v1/auth/service-token`). The agent signs it with this node's key, and
only if the bundle grants this node `gateway`; the far side checks the
same grant again. So a gateway that is taken over reaches other machines
for fifteen minutes at a time and only while the agent beside it keeps
agreeing -- and the token it was handed at spawn is worth nothing off
this machine, where every one of these requests could be observed.

**A request whose recipient is not known carries nothing.** The one
thing never done is to send the local token somewhere that is not this
machine: it is long-lived, and whoever reads it could replay it here.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from typing import Any

import httpx

from . import tokens
from ._http import internal_client

log = logging.getLogger(__name__)

RECIPIENT_CONTROL = tokens.RECIPIENT_CONTROL

_FETCH_TIMEOUT_SECONDS = 5.0
# A failed fetch is not retried on every request for that recipient.
_RETRY_AFTER_SECONDS = 5.0
_WARN_INTERVAL_SECONDS = 60.0


class Outbound:
    """Tokens for this gateway's outbound calls, one recipient at a time."""

    def __init__(
        self,
        *,
        recipient: str | None,
        local_token: str | None,
        agent_url: str,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.recipient = recipient
        self._local = local_token
        self._agent_url = agent_url.rstrip("/")
        self._transport = transport
        self._clock = clock
        # recipient -> (token, refresh at, expires at)
        self._cache: dict[str, tuple[str, float, float]] = {}
        self._retry_at: dict[str, float] = {}
        self._warned_at: dict[str, float] = {}
        # Why the agent last refused a recipient, until it next agrees:
        # a far side's bare 401 says nothing about a grant it never saw.
        self._refused: dict[str, str] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return self._local is not None and self.recipient is not None

    @property
    def local_token(self) -> str | None:
        return self._local

    def for_node(self, node: str | None) -> str | None:
        """The recipient for a machine by name; `None` is this one."""
        if self.recipient is None:
            return None
        return self.recipient if node is None else tokens.node_recipient(node)

    async def token(self, recipient: str | None) -> str | None:
        """The bearer for `recipient`, or None to send nothing."""
        if not self.enabled or recipient is None:
            return None
        if recipient == self.recipient:
            return self._local
        now = self._clock()
        held = self._cache.get(recipient)
        if held is not None and now < held[1]:
            return held[0]
        if now < self._retry_at.get(recipient, 0.0):
            return held[0] if held is not None and now < held[2] else None
        lock = self._locks.setdefault(recipient, asyncio.Lock())
        async with lock:
            held = self._cache.get(recipient)
            now = self._clock()
            if held is not None and now < held[1]:
                return held[0]
            fetched = await self._fetch(recipient)
            if fetched is not None:
                return fetched
            # Keep using a token that has not expired yet: the agent being
            # briefly unreachable is not a reason to stop reaching others.
            return held[0] if held is not None and now < held[2] else None

    async def headers(self, recipient: str | None) -> dict[str, str]:
        token = await self.token(recipient)
        return {"Authorization": f"Bearer {token}"} if token else {}

    def refusal(self, recipient: str | None) -> str | None:
        """Why this node's agent last refused a token for `recipient`, if it did."""
        return self._refused.get(recipient) if recipient is not None else None

    def auth(self, recipient: str | None) -> httpx.Auth:
        """An `httpx.Auth` presenting this recipient's token on every request."""
        return _OutboundAuth(self, recipient)

    async def _fetch(self, recipient: str) -> str | None:
        assert self._local is not None
        try:
            response = await self._ensure_client().post(
                f"{self._agent_url}/v1/auth/service-token",
                json={"audience": recipient},
                headers={"Authorization": f"Bearer {self._local}"},
            )
            response.raise_for_status()
            body = response.json()
            token = str(body["token"])
            expires = _unix(body["expiresAt"])
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            now = self._clock()
            self._retry_at[recipient] = now + _RETRY_AFTER_SECONDS
            self._refused[recipient] = _describe(exc)
            if now - self._warned_at.get(recipient, float("-inf")) >= _WARN_INTERVAL_SECONDS:
                self._warned_at[recipient] = now
                log.warning(
                    "this node's agent would not give the gateway a token for %s (%s); calls "
                    "there go without one until it does",
                    recipient,
                    _describe(exc),
                )
            return None
        issued = self._clock()
        # Replaced at half its life, so a request started now finishes
        # long before the token it carries lapses.
        self._cache[recipient] = (token, issued + max(0.0, expires - issued) / 2, expires)
        self._retry_at.pop(recipient, None)
        self._refused.pop(recipient, None)
        return token

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = internal_client(
                timeout=_FETCH_TIMEOUT_SECONDS, transport=self._transport
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class _OutboundAuth(httpx.Auth):
    def __init__(self, outbound: Outbound, recipient: str | None) -> None:
        self._outbound = outbound
        self._recipient = recipient

    def sync_auth_flow(self, request: httpx.Request) -> Any:  # pragma: no cover
        raise RuntimeError("the gateway's outbound calls are async")

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        token = await self._outbound.token(self._recipient)
        if token:
            request.headers["Authorization"] = f"Bearer {token}"
        yield request


def _unix(value: Any) -> float:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def _describe(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        try:
            body = response.json()
            problem = body.get("detail", body) if isinstance(body, dict) else {}
            said = (
                problem.get("detail") or problem.get("title") if isinstance(problem, dict) else None
            )
        except ValueError:
            said = None
        return f"{response.status_code}: {said}" if said else f"HTTP {response.status_code}"
    return str(exc) or type(exc).__name__
