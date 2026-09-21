"""Keep pre-A5 history; report known per-key usage without inventing tokens."""

import sqlite3
from datetime import UTC, datetime, timedelta

from eugene_plexus_gateway.metrics import _DDL, MetricsStore, RequestRow


async def test_v2_migration_keeps_history_and_attributed_totals(tmp_path):
    path = tmp_path / "metrics.sqlite3"
    conn = sqlite3.connect(path)
    old = (
        _DDL.replace("    client_key_id     TEXT,\n", "")
        .replace("    client_key_name   TEXT\n", "")
        .replace("    strategy          TEXT,\n", "    strategy          TEXT\n")
    )
    conn.executescript(old)
    now = datetime.now(UTC)
    conn.execute("INSERT INTO meta VALUES ('schema_version', '2')")
    conn.execute(
        "INSERT INTO request (started_at, requested_model, attempts, total_ms, outcome) VALUES (?, ?, ?, ?, ?)",
        (now.isoformat(), "old", 1, 5, "served"),
    )
    conn.commit()
    conn.close()
    store = MetricsStore(path)
    await store.start()
    for key, attempts, outcome, prompt, completion in [
        ("a", 2, "served", 10, 4),
        ("a", 0, "error", None, None),
        ("b", 1, "served", 7, 0),
    ]:
        store.record(
            RequestRow(
                started_at=now,
                requested_model="m",
                attempts=attempts,
                outcome=outcome,
                total_ms=10,
                client_key_id=key,
                client_key_name="App " + key,
                prompt_tokens=prompt,
                completion_tokens=completion,
            )
        )
    await store.aclose()
    store = MetricsStore(path)
    await store.start()
    try:
        rows, _ = store.requests(limit=20)
        assert len(rows) == 4 and any(
            r["requestedModel"] == "old" and r["clientKeyId"] is None for r in rows
        )
        usage = {
            r["clientKeyId"]: r
            for r in store.client_usage(
                since=now - timedelta(seconds=1), until=now + timedelta(seconds=1)
            )
        }
        assert usage["a"]["requests"] == 2 and usage["a"]["failed"] == 1
        assert usage["a"]["promptTokens"] == 10 and usage["a"]["completionTokens"] == 4
        assert usage["a"]["incompleteUsageRequests"] == 2
        assert usage["b"]["incompleteUsageRequests"] == 0
        assert not list(tmp_path.glob("*.bak"))
    finally:
        await store.aclose()
