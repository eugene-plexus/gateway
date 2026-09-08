"""Identity client: Protocol + HTTP and in-process implementations.

The gateway pulls constitution + relevant self-model entries +
per-person relationship context from the identity component every chat
turn, and assembles per-hemisphere system prompts from them. The HTTP
path talks to `eugene-plexus/identity`; the in-process variant is the
test double / "no identity configured" fallback.

Both implementations satisfy the `IdentityClient` Protocol so callers
type against the contract.

When `identityUrl` is unset in config, the gateway skips identity
entirely and falls back to the v0.1 single-shared-system-prompt path —
that keeps existing installs working until the operator adds an
identity component to their topology.
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

import httpx

from ._generated.models import (
    Constitution,
    Person,
    RelationshipSummary,
    SelfModelEntry,
)


class IdentityClient(Protocol):
    """Contract every identity backend (real or fake-for-tests) implements.

    All methods raise `httpx.HTTPError` on transport-level failures; the
    chat handler maps those to 502s. `get_relationship` returns `None`
    when the person doesn't exist (404 from identity).
    """

    async def get_constitution(self) -> Constitution: ...
    async def query_self_model(
        self,
        *,
        topic: str | None = None,
        person_id: UUID | None = None,
        limit: int = 5,
    ) -> list[SelfModelEntry]: ...
    async def get_relationship(self, person_id: UUID) -> RelationshipSummary | None: ...
    async def list_persons(self) -> list[Person]: ...
    async def aclose(self) -> None: ...


class HttpIdentity:
    """HTTP-backed identity client. Talks to eugene-plexus/identity."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 30.0,
        service_token: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {service_token}"} if service_token else None
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds, connect=10.0),
            headers=headers,
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    async def get_constitution(self) -> Constitution:
        response = await self._client.get("/v1/identity/constitution")
        response.raise_for_status()
        return Constitution.model_validate(response.json())

    async def query_self_model(
        self,
        *,
        topic: str | None = None,
        person_id: UUID | None = None,
        limit: int = 5,
    ) -> list[SelfModelEntry]:
        params: dict[str, str | int] = {"limit": limit}
        if topic is not None:
            params["topic"] = topic
        if person_id is not None:
            params["personId"] = str(person_id)
        response = await self._client.get("/v1/identity/self-model", params=params)
        response.raise_for_status()
        body = response.json()
        return [SelfModelEntry.model_validate(e) for e in body.get("entries", [])]

    async def get_relationship(self, person_id: UUID) -> RelationshipSummary | None:
        response = await self._client.get(f"/v1/identity/persons/{person_id}/relationship")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return RelationshipSummary.model_validate(response.json())

    async def list_persons(self) -> list[Person]:
        response = await self._client.get("/v1/identity/persons")
        response.raise_for_status()
        body = response.json()
        return [Person.model_validate(p) for p in body.get("persons", [])]

    async def aclose(self) -> None:
        await self._client.aclose()


class InProcessIdentity:
    """Test double / dev fallback. Holds constitution + entries in memory."""

    def __init__(
        self,
        *,
        constitution: Constitution | None = None,
        self_model_entries: list[SelfModelEntry] | None = None,
        persons: list[Person] | None = None,
        relationships: dict[UUID, RelationshipSummary] | None = None,
    ) -> None:
        self._constitution = constitution or Constitution(name="Eugene")
        self._entries = list(self_model_entries or [])
        self._persons = list(persons or [])
        self._relationships = dict(relationships or {})

    async def get_constitution(self) -> Constitution:
        return self._constitution

    async def query_self_model(
        self,
        *,
        topic: str | None = None,
        person_id: UUID | None = None,
        limit: int = 5,
    ) -> list[SelfModelEntry]:
        out: list[SelfModelEntry] = []
        for entry in self._entries:
            if person_id is not None and (
                not entry.relatedPersonIds or person_id not in entry.relatedPersonIds
            ):
                continue
            out.append(entry)
        if topic is not None:
            out.sort(key=lambda e: 0 if e.topic == topic else 1)
        return out[:limit]

    async def get_relationship(self, person_id: UUID) -> RelationshipSummary | None:
        return self._relationships.get(person_id)

    async def list_persons(self) -> list[Person]:
        return list(self._persons)

    async def aclose(self) -> None:
        return None
