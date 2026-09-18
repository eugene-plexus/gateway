"""Which client keys this gateway must refuse (hobbyist UX S4, 2026-09-15).

A **client key** is the third audience: a long-lived named bearer the
agent mints for an app outside the install -- Continue, Cline, Open
WebUI, SillyTavern, the OpenAI SDK, `curl`. It carries `aud: client`,
which this gateway accepts on `/v1/models`, `/v1/chat/completions` and
`/v1/embeddings` and nothing anywhere accepts otherwise.

An operator who turns one off expects it to stop working. Nothing about
a JWT stops working on its own before `exp`, so something has to say
"not that one" -- and this is the something.

## Where the list comes from

The agent on **this gateway's own node**, at
`GET /v1/auth/client-keys/revoked`, with the gateway's service token.
The same agent this gateway already reads `/v1/node` and
`/v1/components` from on every routing refresh, so it is an address the
process already has and a peer it already trusts.

That is also why the contract tells an operator to mint against the
node the gateway runs on: the whole install shares one signing key, so a
key minted anywhere *verifies* everywhere, but its **record** lives on
one agent, and this is the one that gets asked.

## Three properties, each deliberate

**Cached, not asked per request.** A round trip on the hot path for
every completion would put the agent in the inference path, which the
two-layer split exists to avoid. The list is re-read when this copy is
older than `ttl_seconds` -- the routing refresh interval -- so revoking
takes effect within about that long. The contract says so in those
words rather than promising instant.

**Fail-open on the list, never on the token.** If the agent cannot be
reached the previous answer keeps being used and requests keep being
served. Refusing every client key because the local agent restarted
would take every harness in an install down for the length of a
restart; a revoked key living fifteen seconds longer is the smaller
failure, and the token still has to carry a valid signature and an
unexpired `exp`. The total, instant revocation has existed since M7 and
is unchanged: rotate the install's signing key.

**Nothing is fetched until a client key arrives.** An install where
nobody has minted one never makes the call -- `require_authorized` only
consults the guard for `aud: client`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from ._http import internal_client

log = logging.getLogger(__name__)

# How often the list is re-read, and how long a revocation may take to
# bite here. The routing refresh interval, because it is the number an
# operator already has in mind for "how long until the gateway notices".
DEFAULT_TTL_SECONDS = 15.0

# A local call to a peer that is almost always on loopback. Short: this
# runs inside a request, and a hung agent must not hold a completion.
DEFAULT_TIMEOUT_SECONDS = 3.0

_WARN_INTERVAL_SECONDS = 60.0


class ClientKeyGuard:
    """The revoked set, cached, refreshed on a TTL, shared by all callers."""

    def __init__(
        self,
        *,
        agent_url: str,
        service_token: str | None = None,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._agent_url = agent_url.rstrip("/")
        self._service_token = service_token
        self._ttl = ttl_seconds
        self._timeout = timeout_seconds
        self._revoked: frozenset[str] = frozenset()
        self._revision: int | None = None
        self._fetched_at: float | None = None
        # One refresh at a time. Under load every concurrent request
        # would otherwise open its own connection to the agent the
        # moment the copy goes stale.
        self._lock = asyncio.Lock()
        self._client: httpx.AsyncClient | None = None
        self._last_warning = 0.0

    @property
    def revision(self) -> int | None:
        return self._revision

    def _stale(self, now: float) -> bool:
        return self._fetched_at is None or (now - self._fetched_at) >= self._ttl

    async def is_revoked(self, key_id: str | None) -> bool:
        """Whether this `jti` has been turned off.

        A client token with **no** `jti` cannot be matched against the
        list, and is refused: every token this endpoint mints carries
        one, so a client-audience token without it was not minted by
        this install's agent as a client key -- and a credential that
        can never be revoked is not one to accept on the strength of a
        signature alone.
        """
        if key_id is None:
            return True
        await self._refresh_if_stale()
        return key_id in self._revoked

    async def _refresh_if_stale(self) -> None:
        if not self._stale(time.perf_counter()):
            return
        async with self._lock:
            # Another caller may have refreshed while this one waited.
            if not self._stale(time.perf_counter()):
                return
            await self._fetch()

    async def _fetch(self) -> None:
        url = f"{self._agent_url}/v1/auth/client-keys/revoked"
        headers = {"Authorization": f"Bearer {self._service_token}"} if self._service_token else {}
        try:
            client = self._ensure_client()
            response = await client.get(url, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            body: Any = response.json()
            ids = body.get("ids") or []
            revision = body.get("revision")
            fresh = frozenset(str(i) for i in ids)
        except Exception as e:
            # Keep the previous answer, keep serving, and say so at most
            # once a minute so an agent that is down for an hour costs
            # one line rather than two hundred and forty.
            now = time.perf_counter()
            if now - self._last_warning >= _WARN_INTERVAL_SECONDS:
                self._last_warning = now
                log.warning(
                    "could not read the revoked client-key list from %s (%s); serving with the "
                    "list this gateway already has (%d revoked). A key revoked while the agent "
                    "is unreachable keeps working until it answers.",
                    url,
                    e,
                    len(self._revoked),
                )
            # The timestamp is NOT advanced: a failure must not make the
            # copy look fresh, or one blip would buy a whole TTL of
            # silence. It does mean a persistently dead agent is asked
            # once per client request, which the timeout bounds.
            return
        if self._revision is not None and revision != self._revision:
            log.info(
                "revoked client-key list changed (revision %s -> %s): %d key(s) refused",
                self._revision,
                revision,
                len(fresh),
            )
        self._revoked = fresh
        self._revision = revision if isinstance(revision, int) else self._revision
        self._fetched_at = time.perf_counter()

    def _ensure_client(self) -> httpx.AsyncClient:
        # One client for the life of the guard, built on first poll.
        # `internal_client` rather than `httpx.AsyncClient()`: the bare
        # constructor parses certifi's PEM bundle, ~104 ms of synchronous
        # CPU on the event loop, and this polls its own node's agent
        # every routing-refresh interval. Each request carries its own
        # `timeout=`, so none lives on the client. It dials this
        # install's own agent, so it also declines the user's proxy.
        if self._client is None:
            self._client = internal_client()
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            await client.aclose()
