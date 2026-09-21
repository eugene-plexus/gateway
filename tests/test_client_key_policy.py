"""Bounded authorization under failed reads, restarts, rollback and contention."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from tests.test_client_keys import FakeAgent


@pytest.mark.asyncio
async def test_persisted_revocation_and_permission_age(tmp_path: Path, monkeypatch) -> None:
    # A 50 ms allowance must measure the injected clock, not the CI runner's
    # disk/fsync speed. Advance both clocks explicitly across the age boundary.
    wall = time.time()
    elapsed = [0.0]
    monkeypatch.setattr(time, "time", lambda: wall + elapsed[0])
    monkeypatch.setattr(time, "perf_counter", lambda: elapsed[0])
    path = tmp_path / "policy.json"
    agent = FakeAgent()
    agent.revoke("key-1")
    guard = agent.as_guard(cache_file=path, ttl_seconds=0, max_age_seconds=0.05)
    assert await guard.decision("key-1") == "revoked"
    assert await guard.decision("key-2") == "allowed"
    await guard.aclose()
    agent.fail = True
    restarted = agent.as_guard(cache_file=path, max_age_seconds=0.05)
    assert await restarted.decision("key-1") == "revoked"
    assert await restarted.decision("key-2") == "allowed"
    elapsed[0] += 0.07
    assert await restarted.decision("key-2") == "unavailable"
    assert await restarted.decision("key-1") == "revoked"
    await restarted.aclose()


@pytest.mark.asyncio
async def test_concurrent_timeout_is_shared_and_backed_off() -> None:
    agent = FakeAgent()
    guard = agent.as_guard(ttl_seconds=0, retry_seconds=0.03)
    reads = 0

    async def outage(request: httpx.Request) -> httpx.Response:
        nonlocal reads
        reads += 1
        await asyncio.sleep(0.05)
        raise httpx.ReadTimeout("unavailable", request=request)

    await guard._client.aclose()
    guard._client = httpx.AsyncClient(transport=httpx.MockTransport(outage))
    started = time.perf_counter()
    results = await asyncio.gather(*(guard.decision("key-1") for _ in range(40)))
    assert set(results) == {"unavailable"}
    assert reads == 1 and time.perf_counter() - started < 0.5
    assert await guard.decision("key-1") == "unavailable"
    assert reads == 1
    await asyncio.sleep(0.04)
    await guard.decision("key-1")
    assert reads == 2
    await guard.aclose()


@pytest.mark.asyncio
async def test_cancelled_request_does_not_cancel_shared_refresh() -> None:
    agent = FakeAgent()
    guard = agent.as_guard()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(request: httpx.Request) -> httpx.Response:
        started.set()
        await release.wait()
        return agent._handle(request)

    await guard._client.aclose()
    guard._client = httpx.AsyncClient(transport=httpx.MockTransport(slow))
    first = asyncio.create_task(guard.decision("key-1"))
    await started.wait()
    second = asyncio.create_task(guard.decision("key-2"))
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    release.set()
    assert await second == "allowed" and agent.reads == 1
    await guard.aclose()


@pytest.mark.asyncio
async def test_revision_rollback_cannot_extend_permission(tmp_path: Path) -> None:
    agent = FakeAgent()
    agent.revision = 5
    guard = agent.as_guard(ttl_seconds=0, max_age_seconds=0.03)
    assert await guard.decision("key-1") == "allowed"
    agent.revision = 4
    await asyncio.sleep(0.04)
    assert await guard.decision("key-1") == "unavailable"
    assert guard.revision == 5
    await guard.aclose()


@pytest.mark.asyncio
async def test_failed_cache_write_cannot_authorize_new_keys(tmp_path: Path, monkeypatch) -> None:
    agent = FakeAgent()
    guard = agent.as_guard(cache_file=tmp_path / "policy.json")

    def fail(_policy):
        raise OSError("disk full")

    monkeypatch.setattr(guard, "_save", fail)
    assert await guard.decision("key-1") == "unavailable"
    await guard.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["stale", "scope", "future", "corrupt", "nan"])
async def test_unusable_persisted_cache_fails_closed(tmp_path: Path, damage: str) -> None:
    path = tmp_path / "policy.json"
    agent = FakeAgent()
    guard = agent.as_guard(cache_file=path)
    assert await guard.decision("key-1") == "allowed"
    await guard.aclose()
    saved = json.loads(path.read_text())
    if damage == "stale":
        saved["policy"]["generatedAt"] -= 120
    elif damage == "scope":
        saved["scope"] = "different-key-or-agent"
    elif damage == "future":
        saved["savedAt"] += 60
    elif damage == "nan":
        saved["savedAt"] = float("nan")
    path.write_text("broken" if damage == "corrupt" else json.dumps(saved))
    agent.fail = True
    restarted = agent.as_guard(cache_file=path)
    assert await restarted.decision("key-1") == "unavailable"
    await restarted.aclose()


@pytest.mark.asyncio
async def test_clock_rollback_preserves_known_revocation(tmp_path):
    path = tmp_path / "policy.json"
    agent = FakeAgent()
    agent.revoke("key-1")
    guard = agent.as_guard(cache_file=path)
    assert await guard.decision("key-1") == "revoked"
    await guard.aclose()
    saved = json.loads(path.read_text())
    saved["savedAt"] += 60
    saved["policy"]["generatedAt"] += 60
    path.write_text(json.dumps(saved))
    agent.fail = True
    restarted = agent.as_guard(cache_file=path)
    assert await restarted.decision("key-1") == "revoked"
    assert await restarted.decision("key-2") == "unavailable"
    await restarted.aclose()
