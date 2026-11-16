"""Row 38: what the control root reads here to evaluate alerts and SLOs.

`specs/docs/design/rows/row-38.md`. The active root reads the metrics
summary once a minute (alert rules, SLOs) and the per-key usage (an SLO set
on a client key) with its own service token. No other service token reads
them, and the root's token opens nothing else. The rollups keep the wake
wait (`waitedMsSum`, metrics v16), so the wake cost adds up over 90 days.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway import tokens
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.metrics import _DDL, SCHEMA_VERSION, AttemptRow, MetricsStore, RequestRow
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, FakeInstall, make_routing_table


@pytest.fixture
def client(
    settings: Settings, install: FakeInstall, fake_driver: FakeDriverClient
) -> Iterator[TestClient]:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake_driver)
    app.state.auth_state = install.auth_state()
    with TestClient(app) as c:
        yield c


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _root_service(install: FakeInstall, *, machine: str | None = None) -> str:
    token, _ = install.root.mint(
        typ=tokens.TYP_SERVICE,
        sub=tokens.SUB_CONTROL,
        aud=[tokens.node_recipient(machine or install.name)],
        ttl_seconds=300,
    )
    return token


def test_the_control_root_reads_the_metrics_summary_and_per_key_usage(
    client: TestClient, install: FakeInstall
) -> None:
    root = _bearer(_root_service(install))
    assert client.get("/v1/metrics", headers=root).status_code == 200
    assert client.get("/v1/metrics?groupBy=total&bucket=hour", headers=root).status_code == 200
    assert client.get("/v1/metrics/clients", headers=root).status_code == 200


def test_the_control_root_token_opens_nothing_else(
    client: TestClient, install: FakeInstall
) -> None:
    root = _bearer(_root_service(install))
    assert client.get("/v1/metrics/requests", headers=root).status_code in (401, 403)
    assert client.get("/v1/config", headers=root).status_code in (401, 403)
    assert client.patch("/v1/config", json={"values": {}}, headers=root).status_code in (401, 403)
    assert client.get("/v1/admin/routing", headers=root).status_code in (401, 403)
    # Addressed to another machine: not here.
    elsewhere = _bearer(_root_service(install, machine="far"))
    assert client.get("/v1/metrics", headers=elsewhere).status_code == 401


def test_no_other_service_token_reads_metrics(client: TestClient, install: FakeInstall) -> None:
    assert install.node is not None
    for signer, sub in (
        (install.far, tokens.SUB_AGENT),
        (install.node, "library"),
        (install.node, tokens.SUB_CONTROL),
    ):
        token, _ = signer.mint(
            typ=tokens.TYP_SERVICE, sub=sub, aud=[install.recipient], ttl_seconds=300
        )
        for path in ("/v1/metrics", "/v1/metrics/clients"):
            got = client.get(path, headers=_bearer(token))
            assert got.status_code == 401, (signer.issuer, sub, path, got.text)


def _woke(when: datetime, waited: int) -> RequestRow:
    return RequestRow(
        started_at=when,
        requested_model="qwen",
        served_model="qwen",
        attempts=1,
        tier=1,
        total_ms=waited + 500,
        waited_ms=waited,
        swapped_in=True,
        outcome="served",
        tries=[AttemptRow(driver="qwen-box", runtime="qwen-rt", elapsed_ms=500, served=True)],
    )


def test_the_wake_wait_is_summed_live_and_kept_in_the_rollups(tmp_path: Path) -> None:
    async def run() -> tuple[list[dict], list[dict]]:
        store = MetricsStore(tmp_path / "m.sqlite3", retention_days=1)
        await store.start()
        try:
            ten_days = (datetime.now(UTC) - timedelta(days=10)).replace(minute=20, second=0)
            now = datetime.now(UTC)
            store._insert(
                [
                    _woke(ten_days, 4000),
                    _woke(ten_days + timedelta(minutes=5), 6000),
                    _woke(now, 3000),
                    _woke(now, 1000),
                ]
            )
            store.maintain()
            hourly = store.summary(
                since=now - timedelta(days=11),
                until=now + timedelta(hours=1),
                bucket="hour",
                group_by="model",
            )
            live = store.summary(
                since=now - timedelta(hours=1), until=now + timedelta(hours=1), group_by="total"
            )
        finally:
            await store.aclose()
        return hourly, live

    hourly, live = asyncio.run(run())
    old = next(g for g in hourly if g["source"] == "rollup")
    assert (old["swappedIn"], old["waitedMsSum"]) == (2, 10_000)
    assert live[0]["swappedIn"] == 2 and live[0]["waitedMsSum"] == 4000


def test_a_v15_store_gains_the_wake_wait_in_place(tmp_path: Path) -> None:
    path = tmp_path / "m.sqlite3"
    v15 = _DDL.replace("    waited_ms_sum     INTEGER NOT NULL DEFAULT 0,\n", "")
    assert v15 != _DDL
    with sqlite3.connect(path) as conn:
        conn.executescript(v15)
        conn.execute("INSERT INTO meta VALUES ('schema_version', '15')")
        conn.execute(
            "INSERT INTO rollup (bucket_start, requested_model, driver, requests, swapped_in)"
            " VALUES ('2026-08-01T10:00:00Z', 'qwen', 'qwen-box', 3, 2)"
        )

    async def reopen() -> list[dict]:
        store = MetricsStore(path)
        await store.start()
        try:
            return store.summary(
                since=datetime(2026, 8, 1, tzinfo=UTC),
                until=datetime(2026, 8, 2, tzinfo=UTC),
                bucket="hour",
                group_by="model",
            )
        finally:
            await store.aclose()

    groups = asyncio.run(reopen())
    assert not list(tmp_path.glob("*.bak")), "the store was set aside, not moved forward"
    # The hour kept its counts; the wait it never stored reads as 0.
    assert [(g["requests"], g["swappedIn"], g["waitedMsSum"]) for g in groups] == [(3, 2, 0)]
    with sqlite3.connect(path) as conn:
        version = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        assert version[0] == str(SCHEMA_VERSION) == "16"
