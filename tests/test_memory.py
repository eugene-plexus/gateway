"""Tests for the memory clients.

InProcessMemory is exercised end-to-end. HttpMemory is exercised against
an httpx MockTransport — no respx dependency, no live memory service
required.
"""

from __future__ import annotations

from collections.abc import Iterable
from uuid import UUID, uuid4

import httpx
import pytest

from eugene_plexus_gateway._generated.models import Conversation, Message, Role
from eugene_plexus_gateway.memory import HttpMemory, InProcessMemory

# ---------------------------------------------------------------------------
# InProcessMemory
# ---------------------------------------------------------------------------


async def test_in_process_create_then_get_returns_empty_conversation() -> None:
    mem = InProcessMemory()
    cid = await mem.create()
    convo = await mem.get(cid)
    assert convo is not None
    assert convo.id == cid
    assert convo.messages == []


async def test_in_process_get_unknown_returns_none() -> None:
    mem = InProcessMemory()
    assert await mem.get(uuid4()) is None


async def test_in_process_append_persists_and_stamps_timestamp() -> None:
    mem = InProcessMemory()
    cid = await mem.create()
    written = await mem.append(cid, Message(role=Role.user, content="hi"))
    assert written is not None
    assert written.timestamp is not None
    convo = await mem.get(cid)
    assert convo is not None and convo.messages[0].content == "hi"


async def test_in_process_append_to_unknown_returns_none() -> None:
    mem = InProcessMemory()
    written = await mem.append(uuid4(), Message(role=Role.user, content="ghost"))
    assert written is None


async def test_in_process_delete_removes() -> None:
    mem = InProcessMemory()
    cid = await mem.create()
    assert await mem.delete(cid) is True
    assert await mem.get(cid) is None
    assert await mem.delete(cid) is False


# ---------------------------------------------------------------------------
# HttpMemory (against an httpx MockTransport)
# ---------------------------------------------------------------------------


def _http_memory_with(handler: httpx.MockTransport) -> HttpMemory:
    """Build an HttpMemory whose underlying client uses the given mock transport."""
    mem = HttpMemory(base_url="http://fake-memory")
    mem._client._transport = handler
    return mem


def _route(method: str, path: str, status_code: int, json_body: object) -> httpx.MockTransport:
    """A MockTransport that responds to one (method, path) pair and 405s
    everything else — useful for asserting the client makes exactly one
    call to exactly one endpoint."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == method and request.url.path == path:
            return httpx.Response(status_code, json=json_body)
        return httpx.Response(405)

    return httpx.MockTransport(handler)


def _multi_route(
    routes: Iterable[tuple[str, str, int, object]],
) -> httpx.MockTransport:
    routes = list(routes)

    def handler(request: httpx.Request) -> httpx.Response:
        for method, path, status_code, json_body in routes:
            if request.method == method and request.url.path == path:
                return httpx.Response(status_code, json=json_body)
        return httpx.Response(405)

    return httpx.MockTransport(handler)


async def test_http_create_extracts_id_from_response() -> None:
    cid = uuid4()
    fake = _route(
        "POST",
        "/v1/conversations",
        201,
        Conversation(id=cid, messages=[]).model_dump(mode="json"),
    )
    mem = _http_memory_with(fake)
    try:
        assert await mem.create() == cid
    finally:
        await mem.aclose()


async def test_http_get_404_returns_none() -> None:
    cid = uuid4()
    fake = _route("GET", f"/v1/conversations/{cid}", 404, {})
    mem = _http_memory_with(fake)
    try:
        assert await mem.get(cid) is None
    finally:
        await mem.aclose()


async def test_http_get_200_returns_conversation() -> None:
    cid = uuid4()
    body = Conversation(
        id=cid,
        messages=[Message(role=Role.user, content="hello")],
    ).model_dump(mode="json")
    fake = _route("GET", f"/v1/conversations/{cid}", 200, body)
    mem = _http_memory_with(fake)
    try:
        convo = await mem.get(cid)
        assert convo is not None
        assert convo.id == cid
        assert len(convo.messages) == 1
        assert convo.messages[0].content == "hello"
    finally:
        await mem.aclose()


async def test_http_append_404_returns_none() -> None:
    cid = uuid4()
    fake = _route("POST", f"/v1/conversations/{cid}/messages", 404, {})
    mem = _http_memory_with(fake)
    try:
        result = await mem.append(cid, Message(role=Role.user, content="ghost"))
        assert result is None
    finally:
        await mem.aclose()


async def test_http_append_201_returns_message() -> None:
    """v0.2 memory always returns a MemoryEntry on success — the legacy
    `append` method projects it back to a Message for backward compat."""
    from datetime import UTC, datetime

    cid = uuid4()
    server_response = {
        "entryId": str(uuid4()),
        "personId": "00000000-0000-0000-0000-000000000000",
        "conversationId": str(cid),
        "role": "user",
        "content": "hi",
        "timestamp": datetime.now(UTC).isoformat(),
    }
    fake = _route("POST", f"/v1/conversations/{cid}/messages", 201, server_response)
    mem = _http_memory_with(fake)
    try:
        written = await mem.append(cid, Message(role=Role.user, content="hi"))
        assert written is not None
        assert written.content == "hi"
    finally:
        await mem.aclose()


async def test_http_delete_404_returns_false() -> None:
    cid = uuid4()
    fake = _route("DELETE", f"/v1/conversations/{cid}", 404, {})
    mem = _http_memory_with(fake)
    try:
        assert await mem.delete(cid) is False
    finally:
        await mem.aclose()


async def test_http_delete_204_returns_true() -> None:
    cid = uuid4()
    # MockTransport responds with json={} → empty body. 204 with empty body works.
    fake = _route("DELETE", f"/v1/conversations/{cid}", 204, None)
    mem = _http_memory_with(fake)
    try:
        assert await mem.delete(cid) is True
    finally:
        await mem.aclose()


async def test_http_5xx_propagates_as_httpx_error() -> None:
    cid = uuid4()
    fake = _route("GET", f"/v1/conversations/{cid}", 500, {"error": "boom"})
    mem = _http_memory_with(fake)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await mem.get(cid)
    finally:
        await mem.aclose()


# ---------------------------------------------------------------------------
# Sanity: UUID round-trip across the wire
# ---------------------------------------------------------------------------


async def test_http_create_then_get_roundtrip() -> None:
    """Wire test: create returns an id, subsequent get returns a conversation
    with that exact id."""
    cid = UUID("11111111-2222-3333-4444-555555555555")
    fake = _multi_route(
        [
            (
                "POST",
                "/v1/conversations",
                201,
                Conversation(id=cid, messages=[]).model_dump(mode="json"),
            ),
            (
                "GET",
                f"/v1/conversations/{cid}",
                200,
                Conversation(id=cid, messages=[]).model_dump(mode="json"),
            ),
        ]
    )
    mem = _http_memory_with(fake)
    try:
        created = await mem.create()
        assert created == cid
        fetched = await mem.get(created)
        assert fetched is not None and fetched.id == cid
    finally:
        await mem.aclose()
