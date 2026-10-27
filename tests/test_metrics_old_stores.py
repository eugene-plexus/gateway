"""Released v12 history migrates in place; older alpha and future files are kept aside.

Schema 13 adds person attribution to the released schema 12 without losing
dashboard history, and schema 14 (row 11) adds the queue's columns to 12 and 13
the same way. Other schemas retain the prior backup-and-reset behavior
(`metrics.v<N>.bak`, history readable in it).

**The fixtures are real files of the old formats.** Schema 2 is the DDL of the
build that introduced it (gateway `7cf5640`, 2026-09-10), copied from that
commit, with rows written as that build wrote them. Schema 11 (`601f4c7`) has
the same DDL as 12 (`2a2be9c`), which added a value to a column and no column.
Removing the two attribution columns and the queue's five from today's `_DDL`
reproduces it; removing the queue's five alone reproduces 13. Schema 15
represents a build newer than this one.

The cases: each such file is set aside whole and its history is readable
there; nothing old is read back; a second start opens the fresh file in place
and does not set anything aside again; an earlier kept file is never replaced;
and a file that cannot be set aside costs the gateway its metrics and
nothing else, naming why, and is set aside at the next start.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.metrics import _DDL, SCHEMA_VERSION, MetricsStore, RequestRow
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table, runtime_facts

#: `_DDL` of gateway 7cf5640, verbatim.
V2_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS request (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at        TEXT    NOT NULL,
    requested_model   TEXT    NOT NULL,
    served_model      TEXT,
    attempts          INTEGER NOT NULL,
    tier              INTEGER,
    total_ms          INTEGER NOT NULL,
    waited_ms         INTEGER NOT NULL DEFAULT 0,
    swapped_in        INTEGER NOT NULL DEFAULT 0,
    streamed          INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    outcome           TEXT    NOT NULL,
    routing_ms        INTEGER,
    refreshed         INTEGER NOT NULL DEFAULT 0,
    strategy          TEXT
);

CREATE INDEX IF NOT EXISTS request_started_at ON request (started_at);

CREATE TABLE IF NOT EXISTS attempt (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL REFERENCES request (id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    driver     TEXT    NOT NULL,
    runtime    TEXT,
    node       TEXT,
    backend    TEXT,
    elapsed_ms INTEGER NOT NULL,
    served     INTEGER NOT NULL,
    error      TEXT,
    backend_ms INTEGER
);

CREATE INDEX IF NOT EXISTS attempt_request ON attempt (request_id);

-- What the balancer considered, when there was a choice. A separate
-- table rather than JSON on the request, because the whole point is to
-- be able to ask questions across it - "which backend keeps being
-- rejected, and why" - and that is a GROUP BY, not a scan of blobs.
CREATE TABLE IF NOT EXISTS candidate (
    request_id INTEGER NOT NULL REFERENCES request (id) ON DELETE CASCADE,
    seq        INTEGER NOT NULL,
    driver     TEXT    NOT NULL,
    tier       INTEGER NOT NULL,
    eligible   INTEGER NOT NULL,
    reason     TEXT,
    in_flight  INTEGER,
    slots      INTEGER
);

CREATE INDEX IF NOT EXISTS candidate_request ON candidate (request_id);

-- Hourly aggregates, kept indefinitely: a few hundred rows a day, and
-- the only thing that can answer "what did last night look like" once
-- raw rows age out. `served_ms_*` accumulate the SERVING attempt's
-- elapsed time, never the request total, so throughput survives
-- compaction with the same meaning it has in a live query.
CREATE TABLE IF NOT EXISTS rollup (
    bucket_start      TEXT NOT NULL,
    requested_model   TEXT NOT NULL,
    driver            TEXT NOT NULL,
    runtime           TEXT,
    node              TEXT,
    backend           TEXT,
    requests          INTEGER NOT NULL DEFAULT 0,
    errors            INTEGER NOT NULL DEFAULT 0,
    cascaded          INTEGER NOT NULL DEFAULT 0,
    swapped_in        INTEGER NOT NULL DEFAULT 0,
    total_ms_sum      INTEGER NOT NULL DEFAULT 0,
    total_ms_max      INTEGER NOT NULL DEFAULT 0,
    served_ms_sum     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    token_samples     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (bucket_start, requested_model, driver, runtime, node, backend)
);
"""


V13_DDL = _DDL.replace(
    "    affinity          TEXT,\n    queued_ms         INTEGER,\n    band              TEXT,\n"
    "    refusal           TEXT,\n    tier_moves        INTEGER,\n"
    "    outran_reserve    INTEGER\n",
    "    affinity          TEXT\n",
)
assert V13_DDL != _DDL
V12_DDL = V13_DDL.replace("    person_id        TEXT,\n", "").replace(
    "    app_client_id    TEXT,\n", ""
)


def build_old_store(path: Path, version: str) -> None:
    """A store as the build of that schema left it, with one served request."""
    conn = sqlite3.connect(path)
    if version == "2":
        conn.executescript(V2_DDL)
        conn.execute(
            "INSERT INTO request (started_at, requested_model, served_model, attempts, tier,"
            " total_ms, waited_ms, swapped_in, streamed, prompt_tokens, completion_tokens,"
            " outcome, routing_ms, refreshed, strategy)"
            " VALUES ('2026-09-12T10:00:00Z', 'qwen', 'qwen', 1, 1, 812, 0, 0, 1, 12, 40,"
            " 'served', 3, 0, 'least_loaded')"
        )
        conn.execute(
            "INSERT INTO attempt (request_id, seq, driver, runtime, node, backend, elapsed_ms,"
            " served, error, backend_ms) VALUES (1, 0, 'qwen-box', 'qwen-rt', 'box', 'llama_cpp',"
            " 800, 1, NULL, 790)"
        )
    else:
        conn.executescript(V12_DDL if version in ("11", "12") else _DDL)
        conn.execute(
            "INSERT INTO request (started_at, requested_model, served_model, attempts, tier,"
            " total_ms, outcome, affinity) VALUES ('2026-10-01T10:00:00Z', 'qwen', 'qwen', 1,"
            " 1, 812, 'served', 'hit')"
        )
    conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (version,))
    conn.commit()
    conn.close()


def schema_of(path: Path) -> str:
    conn = sqlite3.connect(path)
    try:
        return str(conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0])
    finally:
        conn.close()


def requests_in(path: Path) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT requested_model, outcome FROM request").fetchall()
    finally:
        conn.close()


def names_matching(directory: Path, pattern: str) -> list[str]:
    return sorted(p.name for p in directory.glob(pattern))


def a_request() -> RequestRow:
    return RequestRow(
        started_at=datetime.now(UTC),
        requested_model="new",
        attempts=1,
        total_ms=5,
        outcome="served",
    )


async def test_a_v13_store_gains_the_queue_columns_in_place(tmp_path):
    """Row 11, acceptance check 10: a v13 store keeps its history and gains
    `queued_ms`, `band`, `refusal`, `tier_moves` and `outran_reserve`."""
    path = tmp_path / "metrics.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript(V13_DDL)
        conn.execute("INSERT INTO meta VALUES ('schema_version', '13')")
        conn.execute(
            "INSERT INTO request (started_at, requested_model, attempts, total_ms, outcome,"
            " person_id) VALUES (?, 'old-history', 1, 9, 'served', 'ada')",
            (datetime.now(UTC).isoformat(),),
        )
    store = MetricsStore(path)
    await store.start()
    row = a_request()
    row.queued_ms, row.band, row.tier_moves, row.outran_reserve = 420, "background", 1, True
    store.record(row)
    refused = a_request()
    refused.outcome, refused.refusal, refused.band = "refused", "queue_full", "interactive"
    store.record(refused)
    await store.aclose()
    assert schema_of(path) == "14"
    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT requested_model, person_id, queued_ms, band, refusal, tier_moves,"
            " outran_reserve, outcome FROM request ORDER BY id"
        ).fetchall() == [
            ("old-history", "ada", None, None, None, None, None, "served"),
            ("new", None, 420, "background", None, 1, 1, "served"),
            ("new", None, None, "interactive", "queue_full", None, None, "refused"),
        ]
    assert not list(tmp_path.glob("*.bak"))
    again = MetricsStore(path)
    await again.start()
    try:
        rows, _ = again.requests(limit=10)
        assert [(r["queuedMs"], r["band"], r["refusal"]) for r in rows] == [
            (None, "interactive", "queue_full"),
            (420, "background", None),
            (None, None, None),
        ]
        assert rows[1]["outranReserve"] is True and rows[1]["tierMoves"] == 1
    finally:
        await again.aclose()


async def test_released_v12_history_is_preserved_when_person_attribution_is_added(tmp_path):
    path = tmp_path / "metrics.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript(V12_DDL)
        conn.execute("INSERT INTO meta VALUES ('schema_version', '12')")
        conn.execute(
            "INSERT INTO request (started_at, requested_model, attempts, total_ms, outcome) "
            "VALUES (?, 'old-history', 1, 9, 'served')",
            (datetime.now(UTC).isoformat(),),
        )
    store = MetricsStore(path)
    await store.start()
    row = a_request()
    row.person_id, row.app_client_id = "ada", "workbench"
    store.record(row)
    await store.aclose()
    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT requested_model, person_id, app_client_id FROM request ORDER BY id"
        ).fetchall() == [
            ("old-history", None, None),
            ("new", "ada", "workbench"),
        ]
    assert not list(tmp_path.glob("*.bak"))


@pytest.mark.parametrize("version", ["2", "11", "15"])
async def test_a_store_of_another_schema_is_kept_aside_whole(
    tmp_path: Path, version: str, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "metrics.sqlite3"
    build_old_store(path, version)
    assert version != str(SCHEMA_VERSION)
    store = MetricsStore(path)
    with caplog.at_level(logging.WARNING):
        await store.start()
    try:
        # A fresh store at this build's schema, holding none of the old rows.
        rows, _ = store.requests(limit=10)
        assert rows == []
        assert schema_of(path) == str(SCHEMA_VERSION)
        # The old one is kept whole: its stamp, and its history, readable.
        aside = tmp_path / f"metrics.v{version}.bak"
        assert aside.is_file()
        assert schema_of(aside) == version
        assert requests_in(aside) == [("qwen", "served")]
        # And the warning names the file, the versions and where it went.
        warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warned) == 1
        assert f"schema version {version}, expected {SCHEMA_VERSION}" in warned[0]
        assert f"moving it to metrics.v{version}.bak" in warned[0]
        store.record(a_request())
    finally:
        await store.aclose()

    # A second start reads the fresh file where it is, and sets nothing aside.
    again = MetricsStore(path)
    await again.start()
    try:
        rows, _ = again.requests(limit=10)
        assert [r["requestedModel"] for r in rows] == ["new"]
    finally:
        await again.aclose()
    assert names_matching(tmp_path, "metrics*.bak") == [f"metrics.v{version}.bak"]
    assert requests_in(tmp_path / f"metrics.v{version}.bak") == [("qwen", "served")]


async def test_a_store_of_the_current_schema_is_read_where_it_is(tmp_path: Path) -> None:
    """The v0.1.0 case, and every start since: nothing is set aside."""
    path = tmp_path / "metrics.sqlite3"
    build_old_store(path, str(SCHEMA_VERSION))
    store = MetricsStore(path)
    await store.start()
    try:
        rows, _ = store.requests(limit=10)
        assert [(r["requestedModel"], r["affinity"]) for r in rows] == [("qwen", "hit")]
    finally:
        await store.aclose()
    assert names_matching(tmp_path, "*.bak") == []


async def test_a_file_kept_earlier_is_not_replaced(tmp_path: Path) -> None:
    """Setting a file aside must not destroy the history an earlier setting
    aside kept: the second goes beside it under the next name."""
    path = tmp_path / "metrics.sqlite3"
    earlier = tmp_path / "metrics.v2.bak"
    earlier.write_bytes(b"history kept by an earlier start")
    build_old_store(path, "2")
    store = MetricsStore(path)
    await store.start()
    await store.aclose()

    assert earlier.read_bytes() == b"history kept by an earlier start"
    assert requests_in(tmp_path / "metrics.v2.1.bak") == [("qwen", "served")]
    assert schema_of(path) == str(SCHEMA_VERSION)


def test_a_store_that_cannot_be_set_aside_costs_the_metrics_and_nothing_else(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = settings.metrics_file
    build_old_store(path, "2")

    def build() -> Any:
        app = create_app(settings=settings)
        fake = FakeDriverClient(name="qwen-box", model_id="qwen", runtime="qwen-rt")
        app.state.routing = make_routing_table(fake, runtimes=[runtime_facts("qwen-rt")])
        return app

    real_replace = Path.replace

    def refuse(self: Path, target: Any) -> Any:
        if self == path:
            raise PermissionError(13, "Access is denied", str(self))
        return real_replace(self, target)

    with monkeypatch.context() as m, caplog.at_level(logging.WARNING):
        m.setattr(Path, "replace", refuse)
        with TestClient(build()) as client:
            # Inference is unaffected, and the metrics say they are off.
            done = client.post(
                "/v1/chat/completions",
                json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert done.status_code == 200
            off = client.get("/v1/metrics")
            assert off.status_code == 503 and "could not be opened" in off.text
    named = [
        r.getMessage()
        for r in caplog.records
        if "could not open the metrics store" in r.getMessage()
    ]
    assert len(named) == 1
    assert str(path) in named[0] and "Access is denied" in named[0]
    # The old file is where it was.
    assert schema_of(path) == "2" and requests_in(path) == [("qwen", "served")]

    # The next start, with the file movable, sets it aside and records.
    with TestClient(build()) as client:
        client.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
        )
        for _ in range(100):
            if client.get("/v1/metrics").json()["groups"]:
                break
            import time

            time.sleep(0.05)
        assert client.get("/v1/metrics").status_code == 200
    assert requests_in(path.with_suffix(".v2.bak")) == [("qwen", "served")]
    assert schema_of(path) == str(SCHEMA_VERSION)


async def test_an_interrupted_migration_is_finished_at_the_next_start(tmp_path):
    """Row 11 review: a v13 store whose move to v14 stopped after its first new
    column (a crash, a killed process) is finished at the next start, its
    history kept, rather than failing on the column it already has and leaving
    metrics off."""
    path = tmp_path / "metrics.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript(V13_DDL)
        conn.execute("INSERT INTO meta VALUES ('schema_version', '13')")
        conn.execute(
            "INSERT INTO request (started_at, requested_model, attempts, total_ms, outcome)"
            " VALUES (?, 'old-history', 1, 9, 'served')",
            (datetime.now(UTC).isoformat(),),
        )
        conn.execute("ALTER TABLE request ADD COLUMN queued_ms INTEGER")  # then it stopped
    store = MetricsStore(path)
    await store.start()
    row = a_request()
    row.queued_ms, row.band, row.refusal, row.tier_moves, row.outran_reserve = (
        5,
        "background",
        None,
        1,
        False,
    )
    store.record(row)
    await store.aclose()
    assert schema_of(path) == "14"
    with sqlite3.connect(path) as conn:
        columns = {c[1] for c in conn.execute("PRAGMA table_info(request)")}
        assert {"queued_ms", "band", "refusal", "tier_moves", "outran_reserve"} <= columns
        assert conn.execute(
            "SELECT requested_model, queued_ms, band FROM request ORDER BY id"
        ).fetchall() == [("old-history", None, None), ("new", 5, "background")]
    assert not list(tmp_path.glob("*.bak"))
