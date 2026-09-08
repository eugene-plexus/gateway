"""Conversation memory: protocol + in-process and HTTP implementations.

Production path is `HttpMemory`, which talks to the
[eugene-plexus/memory](https://github.com/eugene-plexus/memory) service
over its OpenAPI surface. `InProcessMemory` is retained as a test double
and dev fallback — tests inject it via `app.state.memory` before the
FastAPI lifespan runs, mirroring how the hemisphere clients are
injected.

Both implementations satisfy the `MemoryClient` Protocol so routes can
type against the contract without caring which backend is in play.

v0.2 surfaces:
  - `append_entry()` writes a full `MemoryEntry` (with `personId` and
    optional NT snapshot / hemisphere attribution). The gateway
    uses this on every chat turn.
  - `person_recent()` reads back recent entries for a `personId` —
    feeds the relationship-context section of per-hemisphere prompts.
  - The legacy `append()` (bare `Message`) stays for backward compat
    and for callers without a personId in scope.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

import httpx

from ._generated.models import Conversation, MemoryEntry, Message, Role

# When the gateway can't resolve a personId (no identity component
# wired, or identity unreachable, and body.personId not supplied),
# memory entries land under this sentinel. Matches memory component's
# NIL_PERSON_ID — same byte pattern, same intent.
NIL_PERSON_ID: UUID = UUID("00000000-0000-0000-0000-000000000000")


class MemoryClient(Protocol):
    """Contract every memory backend implements (real or fake-for-tests).

    `append` / `append_entry` return `None` when the conversation does
    not exist; routes map that to a 404 response. Other transport-level
    failures (the memory service is down, a 5xx, etc.) raise
    `httpx.HTTPError` and the route's existing 502 handler picks it up.
    """

    async def create(self) -> UUID: ...
    async def get(self, conversation_id: UUID) -> Conversation | None: ...
    async def append(self, conversation_id: UUID, message: Message) -> Message | None: ...
    async def append_entry(
        self, conversation_id: UUID, entry: MemoryEntry
    ) -> MemoryEntry | None: ...
    async def person_recent(
        self,
        person_id: UUID,
        *,
        limit: int = 50,
        conversation_id: UUID | None = None,
    ) -> list[MemoryEntry]: ...
    async def delete(self, conversation_id: UUID) -> bool: ...
    async def aclose(self) -> None: ...


class InProcessMemory:
    """Test double / dev fallback. Single-process, thread-unaware — fine for
    FastAPI's single-event-loop runtime, not appropriate for production.
    The eugene-plexus/memory service is the production path.

    Stores MemoryEntry rows internally so the v0.2 `person_recent` /
    `append_entry` paths behave the same way the HTTP backend does.
    """

    def __init__(self) -> None:
        self._conversations: dict[UUID, list[MemoryEntry]] = {}

    async def create(self) -> UUID:
        cid = uuid4()
        self._conversations[cid] = []
        return cid

    async def get(self, conversation_id: UUID) -> Conversation | None:
        entries = self._conversations.get(conversation_id)
        if entries is None:
            return None
        # Project back to v0.1 Message shape so callers reading
        # conversations see the same payload as the real backend.
        return Conversation(
            id=conversation_id,
            messages=[
                Message(role=e.role, content=e.content, timestamp=e.timestamp) for e in entries
            ],
        )

    async def append(self, conversation_id: UUID, message: Message) -> Message | None:
        entry = MemoryEntry(
            entryId=uuid4(),
            personId=NIL_PERSON_ID,
            conversationId=conversation_id,
            role=message.role,
            content=message.content,
            timestamp=message.timestamp or datetime.now(UTC),
        )
        stored = await self.append_entry(conversation_id, entry)
        if stored is None:
            return None
        return Message(role=stored.role, content=stored.content, timestamp=stored.timestamp)

    async def append_entry(self, conversation_id: UUID, entry: MemoryEntry) -> MemoryEntry | None:
        if conversation_id not in self._conversations:
            return None
        stored = entry.model_copy(update={"conversationId": conversation_id})
        self._conversations[conversation_id].append(stored)
        return stored

    async def person_recent(
        self,
        person_id: UUID,
        *,
        limit: int = 50,
        conversation_id: UUID | None = None,
    ) -> list[MemoryEntry]:
        matches: list[MemoryEntry] = []
        for cid, entries in self._conversations.items():
            if conversation_id is not None and cid != conversation_id:
                continue
            for e in entries:
                if e.personId == person_id:
                    matches.append(e)
        matches.sort(key=lambda e: e.timestamp, reverse=True)
        return matches[:limit]

    async def delete(self, conversation_id: UUID) -> bool:
        return self._conversations.pop(conversation_id, None) is not None

    async def aclose(self) -> None:
        return None


class HttpMemory:
    """HTTP-backed memory client. Talks to eugene-plexus/memory."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 30.0,
        service_token: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        # See HttpDriverClient — same pattern: thread the
        # gateway's service-audience bearer onto every outbound
        # request to the memory service.
        headers = {"Authorization": f"Bearer {service_token}"} if service_token else None
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            headers=headers,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    async def create(self) -> UUID:
        response = await self._client.post("/v1/conversations")
        response.raise_for_status()
        conversation = Conversation.model_validate(response.json())
        # The spec marks Conversation.id optional (server-assigned) so the
        # codegen'd type is UUID | None. The memory service always sets it
        # on the create-201 response — assert that contract here so a
        # spec-violating backend surfaces as a loud error rather than a
        # silent None bouncing further down the stack.
        if conversation.id is None:
            raise ValueError(
                "memory service returned a conversation without an id "
                "(violates POST /v1/conversations contract)"
            )
        return conversation.id

    async def get(self, conversation_id: UUID) -> Conversation | None:
        response = await self._client.get(f"/v1/conversations/{conversation_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return Conversation.model_validate(response.json())

    async def append(self, conversation_id: UUID, message: Message) -> Message | None:
        """Append a bare v0.1 Message. The memory component upgrades it
        server-side to a MemoryEntry with `personId = NIL_PERSON_ID`.

        Kept for backward compat; v0.2 callers should use `append_entry`
        with an explicit `personId`.
        """
        payload = message.model_dump(mode="json", exclude_none=True)
        response = await self._client.post(
            f"/v1/conversations/{conversation_id}/messages",
            json=payload,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        # The memory service returns a MemoryEntry on success; project
        # back to Message for this legacy path's callers.
        body = response.json()
        return Message(
            role=Role(body["role"]),
            content=body["content"],
            timestamp=datetime.fromisoformat(body["timestamp"]),
        )

    async def append_entry(self, conversation_id: UUID, entry: MemoryEntry) -> MemoryEntry | None:
        payload = entry.model_dump(mode="json", exclude_none=True)
        response = await self._client.post(
            f"/v1/conversations/{conversation_id}/messages",
            json=payload,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return MemoryEntry.model_validate(response.json())

    async def person_recent(
        self,
        person_id: UUID,
        *,
        limit: int = 50,
        conversation_id: UUID | None = None,
    ) -> list[MemoryEntry]:
        params: dict[str, str | int] = {"limit": limit}
        if conversation_id is not None:
            params["conversationId"] = str(conversation_id)
        response = await self._client.get(
            f"/v1/memory/persons/{person_id}/recent",
            params=params,
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        body = response.json()
        return [MemoryEntry.model_validate(e) for e in body.get("entries", [])]

    async def delete(self, conversation_id: UUID) -> bool:
        response = await self._client.delete(f"/v1/conversations/{conversation_id}")
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    async def aclose(self) -> None:
        await self._client.aclose()
