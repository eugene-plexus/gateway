"""Switching does not cut off answers or reuse the old model's identity."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from eugene_plexus_gateway.lifecycle import LifecycleManager

from .conftest import FakeDriverClient, make_routing_table, runtime_facts


class Table:
    def runtime_node(self, node):
        return node

    def __init__(self):
        self.active = 0
        self.held = set()
        self.refresh = AsyncMock()
        self.record_stopped = AsyncMock()
        self.old = SimpleNamespace(
            name="old",
            node=None,
            key=(None, "old"),
            alias="model-a",
            status="ready",
            spec={"engine": "strata"},
            start_on_demand=False,
        )
        self.new = SimpleNamespace(
            name="new",
            node=None,
            key=(None, "new"),
            alias="model-b",
            status="stopped",
            spec={"engine": "strata"},
            start_on_demand=False,
        )

    def runtimes(self):
        return [self.old, self.new]

    def agent_url_for(self, facts):
        return "http://agent"

    def runtime_inflight(self, key):
        return self.active if key == self.old.key else 0

    @contextmanager
    def stopping(self, key):
        self.held.add(key)
        try:
            yield
        finally:
            self.held.remove(key)


@pytest.fixture
def setup():
    table = Table()
    client = SimpleNamespace(
        admission=AsyncMock(return_value={"decision": "admit", "fit": "unknown"}),
        stop=AsyncMock(return_value=True),
        runtime=AsyncMock(return_value={"status": "stopped"}),
        start=AsyncMock(return_value=(202, "loading")),
    )
    manager = LifecycleManager(
        table,
        client=client,
        swap_wait_seconds=lambda: 5,
        idle_check_seconds=lambda: 30,
        poll_seconds=0.001,
    )
    return table, client, manager


@pytest.mark.anyio
async def test_waits_for_active_answer_and_serializes_switches(setup):
    table, client, manager = setup
    table.active = 1
    work = asyncio.create_task(manager.switch(None, "old", "new"))
    await asyncio.sleep(0.02)
    assert table.held == {(None, "old"), (None, "new")}
    client.stop.assert_not_awaited()
    with pytest.raises(ValueError, match="already running"):
        await manager.switch(None, "old", "new")
    table.active = 0
    await work
    client.stop.assert_awaited_once_with("http://agent", "old", reason="operator", node=None)
    client.start.assert_awaited_once_with("http://agent", "new", node=None)
    assert table.old.alias == "model-a" and table.new.alias == "model-b"
    assert not table.held


@pytest.mark.anyio
async def test_failure_reports_recovery_without_replaying_or_relabeling(setup):
    table, client, manager = setup
    client.start.return_value = (422, "missing prepared config")
    with pytest.raises(ValueError, match="Start old to restore"):
        await manager.switch(None, "old", "new")
    assert client.start.await_count == 1 and not table.held


@pytest.mark.anyio
async def test_missing_target_assets_leave_current_model_running(setup):
    table, client, manager = setup
    client.admission.return_value = {"decision": "refuse", "reason": "missing MTP assets"}
    with pytest.raises(ValueError, match=r"missing MTP assets.*Nothing was stopped"):
        await manager.switch(None, "old", "new")
    client.stop.assert_not_awaited()
    client.start.assert_not_awaited()
    assert not table.held


@pytest.mark.anyio
async def test_unconfirmed_stop_does_not_start_another_model(setup):
    table, client, manager = setup
    client.runtime.return_value = {"status": "ready"}
    with pytest.raises(ValueError, match="shutdown was not confirmed"):
        await manager.switch(None, "old", "new")
    client.start.assert_not_awaited()
    table.record_stopped.assert_not_awaited()
    assert not table.held


@pytest.mark.anyio
async def test_failed_target_and_unavailable_agent_cannot_restore_old_ready_state(
    setup, monkeypatch
):
    _, client, _ = setup
    old = runtime_facts("old", alias="model-a")
    new = runtime_facts("new", alias="model-b", status="stopped")
    driver = FakeDriverClient(
        name="old-driver", base_url="http://driver", model_id="model-a", runtime="old"
    )
    table = make_routing_table(driver, runtimes=[old, new])
    monkeypatch.setattr(table, "refresh", AsyncMock())
    client.start.return_value = (422, "target load refused")
    manager = LifecycleManager(
        table, client=client, swap_wait_seconds=lambda: 5, idle_check_seconds=lambda: 30
    )
    assert table.resolve("model-a").eligible_backends()
    try:
        with pytest.raises(ValueError, match="Start old to restore"):
            await manager.switch(None, "old", "new")
        assert not table.resolve("model-a").eligible_backends()
        assert not table.is_stopping(old.key)
        # The next failed poll must retain the confirmed stop, not a cached
        # ready status from before the operator switched models.
        monkeypatch.setattr(table, "_fetch_driver_entries", AsyncMock(return_value=([], False)))
        monkeypatch.setattr(table, "_fetch_runtime_facts", AsyncMock(return_value=({}, False)))
        _, retained = await table._fetch_agent(None, "http://agent")
        assert retained[old.key].status == "stopped"
    finally:
        await table.aclose()
