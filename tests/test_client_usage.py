"""Report known per-key usage without inventing tokens."""

from datetime import UTC, datetime, timedelta

from eugene_plexus_gateway.metrics import AttemptRow, MetricsStore, RequestRow


async def test_attributed_totals_count_known_usage_and_say_how_many_requests_lacked_it(tmp_path):
    path = tmp_path / "metrics.sqlite3"
    now = datetime.now(UTC)
    store = MetricsStore(path)
    await store.start()
    # A request no key is attributed to, as an anonymous one is.
    store.record(
        RequestRow(started_at=now, requested_model="old", attempts=1, total_ms=5, outcome="served")
    )
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


async def test_an_attempt_that_reported_no_usage_reads_as_unknown_not_zero(tmp_path):
    path = tmp_path / "metrics.sqlite3"
    store = MetricsStore(path)
    await store.start()
    store.record(
        RequestRow(
            started_at=datetime.now(UTC),
            requested_model="old",
            attempts=1,
            total_ms=5,
            outcome="error",
            tries=[AttemptRow(driver="old-driver", elapsed_ms=5, served=False)],
        )
    )
    await store.aclose()
    store = MetricsStore(path)
    await store.start()
    try:
        rows, _ = store.requests()
        assert len(rows) == 1 and rows[0]["requestId"] is None
        assert rows[0]["tries"][0]["usageKnown"] is False
        assert rows[0]["tries"][0]["promptTokens"] is None
        assert not list(tmp_path.glob("*.bak"))
    finally:
        await store.aclose()
