"""TTFT, decode throughput and groupBy — the metrics-dashboard slice.

The two headline numbers exist because the whole-attempt rate includes
prefill: 40 tokens over a 1200 ms attempt whose first event landed at
200 ms is 33 tok/s by the old arithmetic and 40 tok/s of actual decode.
The pair differing is not noise to reconcile — it is the prefill cost
made visible, and the property the first test pins.

`groupBy` is a server parameter because percentiles do not recombine:
the install-wide p50 is not computable from per-backend p50s, so a UI
that needs the coarse grain must ask for it rather than merge what does
not merge.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from .test_metrics import (  # noqa: F401  (fixtures resolve by name)
    _drain,
    _row,
    _store,
    metrics_app,
    metrics_client,
)


def _window() -> dict[str, datetime]:
    return {
        "since": datetime.now(UTC) - timedelta(hours=1),
        "until": datetime.now(UTC) + timedelta(hours=1),
    }


def test_ttft_and_decode_rate_come_from_the_first_streamed_event(tmp_path: Path) -> None:
    """A fixture where the two rates came out equal would prove the field
    was populated and nothing else; this one makes decode strictly
    higher, which is what prefill exclusion means."""

    async def run() -> None:
        store = await _store(tmp_path / "m.sqlite3")
        try:
            store._insert([_row(elapsed=1200, tokens=40, first_ms=200)])
            g = store.summary(**_window())[0]
            assert g["ttftMs"] == {"p50": 200, "p90": 200, "p99": 200, "max": 200}
            assert g["decodeTokensPerSecond"] == {"p50": 40.0, "p90": 40.0, "samples": 1}
            assert g["tokensPerSecond"]["p50"] == 33.33
            assert g["decodeTokensPerSecond"]["p50"] > g["tokensPerSecond"]["p50"]
        finally:
            await store.aclose()

    asyncio.run(run())


def test_ttft_is_null_when_nothing_streamed(tmp_path: Path) -> None:
    """A non-streamed request has no first token to time — the response
    arrives whole. Null, never zero: a zero TTFT says "instant" where
    the truth is "unmeasured"."""

    async def run() -> None:
        store = await _store(tmp_path / "m.sqlite3")
        try:
            store._insert([_row(first_ms=None)])
            g = store.summary(**_window())[0]
            assert g["ttftMs"] is None
            assert g["decodeTokensPerSecond"] is None
            # The whole-attempt rate never needed a first-token time.
            assert g["tokensPerSecond"] is not None
        finally:
            await store.aclose()

    asyncio.run(run())


def test_a_decode_rate_needs_a_window(tmp_path: Path) -> None:
    """A 50 ms decode window reports quantization, not the GPU. Excluded
    rather than rounded — the guard is the playground badge's, kept
    equal so two surfaces cannot disagree about one request. TTFT still
    counts: the guard is about the division, and the first token's
    timing needs none."""

    async def run() -> None:
        store = await _store(tmp_path / "m.sqlite3")
        try:
            # Window = 1200 - 1150 = 50 ms < 250: excluded.
            store._insert([_row(elapsed=1200, tokens=40, first_ms=1150)])
            g = store.summary(**_window())[0]
            assert g["ttftMs"] is not None
            assert g["decodeTokensPerSecond"] is None
        finally:
            await store.aclose()

    asyncio.run(run())


def test_one_token_never_makes_a_decode_rate(tmp_path: Path) -> None:
    async def run() -> None:
        store = await _store(tmp_path / "m.sqlite3")
        try:
            store._insert([_row(elapsed=1200, tokens=1, first_ms=200)])
            g = store.summary(**_window())[0]
            assert g["decodeTokensPerSecond"] is None
        finally:
            await store.aclose()

    asyncio.run(run())


def test_a_failed_attempts_ttft_does_not_pollute_the_group(tmp_path: Path) -> None:
    """A stream that died mid-answer has a first_ms too — kept on the
    attempt row as evidence — but a group's ttftMs is the SERVING
    attempts' number. Mixing in the failure's timing would report the
    failure's latency as the backend's speed: the same mistake
    latency-from-request-total was, one field over."""

    async def run() -> None:
        store = await _store(tmp_path / "m.sqlite3")
        try:
            store._insert(
                [
                    _row(elapsed=1000, tokens=40, first_ms=100),
                    _row(elapsed=9000, tokens=None, first_ms=8000, served=False),
                ]
            )
            g = store.summary(**_window())[0]
            assert g["requests"] == 2
            assert g["ttftMs"]["max"] == 100, "a failed attempt's TTFT leaked into the group"
        finally:
            await store.aclose()

    asyncio.run(run())


def test_group_by_total_collapses_dimensions_with_real_percentiles(tmp_path: Path) -> None:
    """Two backends at 100 ms x3 and 1000 ms x1 have per-group p50s of
    100 and 1000, and no arithmetic over those two numbers yields the
    install-wide p50 of 100. The coarse grain is computed over the raw
    rows, so it can."""

    async def run() -> None:
        store = await _store(tmp_path / "m.sqlite3")
        try:
            store._insert(
                [
                    _row(driver="a", elapsed=100),
                    _row(driver="a", elapsed=100),
                    _row(driver="a", elapsed=100),
                    _row(driver="b", elapsed=1000),
                ]
            )
            assert len(store.summary(**_window())) == 2, "the default grain moved"

            total = store.summary(**_window(), group_by="total")
            assert len(total) == 1
            g = total[0]
            assert g["requests"] == 4
            assert g["latencyMs"]["p50"] == 100
            assert g["latencyMs"]["max"] == 1000
            # A collapsed grain omits what it collapsed — a `total` group
            # naming one driver would be a lie told by iteration order.
            assert g["model"] is None
            assert g["driver"] is None
        finally:
            await store.aclose()

    asyncio.run(run())


def test_group_by_model_keeps_the_model_and_drops_the_backend(tmp_path: Path) -> None:
    async def run() -> None:
        store = await _store(tmp_path / "m.sqlite3")
        try:
            store._insert(
                [
                    _row(driver="a", model="qwen"),
                    _row(driver="b", model="qwen"),
                    _row(driver="a", model="gemma"),
                ]
            )
            groups = store.summary(**_window(), group_by="model")
            by_model = {g["model"]: g for g in groups}
            assert set(by_model) == {"qwen", "gemma"}
            assert by_model["qwen"]["requests"] == 2
            assert by_model["qwen"]["driver"] is None
        finally:
            await store.aclose()

    asyncio.run(run())


def test_a_v4_store_migrates_in_place(tmp_path: Path) -> None:
    """The upgrade an alpha install performs on its next boot. In place,
    keeping the rows — renaming aside is for versions this build does
    not know, and v4 is not one of those."""
    path = tmp_path / "m.sqlite3"

    async def build() -> None:
        store = await _store(path)
        try:
            store._insert([_row()])
        finally:
            await store.aclose()

    asyncio.run(build())

    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE attempt DROP COLUMN first_ms")
    conn.execute("UPDATE meta SET value = '4' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    async def reopen() -> None:
        store = await _store(path)
        try:
            groups = store.summary(**_window())
            assert groups[0]["requests"] == 1, "migration lost the v4 rows"
            assert groups[0]["ttftMs"] is None
            with store._reader() as conn:
                assert conn is not None
                version = conn.execute(
                    "SELECT value FROM meta WHERE key = 'schema_version'"
                ).fetchone()
                columns = {r[1] for r in conn.execute("PRAGMA table_info(attempt)")}
            assert version == ("5",)
            assert "first_ms" in columns
        finally:
            await store.aclose()

    asyncio.run(reopen())
    assert not path.with_suffix(".v4.bak").exists(), "an in-place upgrade renamed the file aside"


def test_a_streamed_completion_records_its_first_token_time(
    metrics_client: TestClient,  # noqa: F811  (pytest fixture, imported above)
) -> None:
    """End to end through the app: the stamp in `TieredClient.stream`,
    the hook kwarg, the AttemptRow field, the insert and both reads.
    The fake streams instantly, so the assertion is presence, not
    magnitude — magnitude belongs to the store tests, where the clock
    is synthetic."""
    with metrics_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        for _ in response.iter_lines():
            pass

    _drain(metrics_client)
    page = metrics_client.get("/v1/metrics/requests").json()
    tries = page["requests"][0]["tries"]
    assert tries[0]["firstMs"] is not None
    assert tries[0]["firstMs"] >= 0

    body = metrics_client.get("/v1/metrics").json()
    assert body["groups"][0]["ttftMs"] is not None

    # And the route's groupBy plumbing, against the same rows.
    total = metrics_client.get("/v1/metrics", params={"groupBy": "total"}).json()
    assert len(total["groups"]) == 1
    assert total["groups"][0]["model"] is None


def test_a_non_streamed_completion_has_no_first_token_time(
    metrics_client: TestClient,  # noqa: F811  (pytest fixture, imported above)
) -> None:
    """`generate()` returns a completed response — there is no first
    token to time, and inventing one from the whole-attempt elapsed
    would be exactly the confidently wrong number this slice refuses."""
    metrics_client.post(
        "/v1/chat/completions",
        json={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
    )
    _drain(metrics_client)
    page = metrics_client.get("/v1/metrics/requests").json()
    assert page["requests"][0]["tries"][0]["firstMs"] is None
    body = metrics_client.get("/v1/metrics").json()
    assert body["groups"][0]["ttftMs"] is None
