"""Retained per-request metrics.

Every completion already reported how it was served, on
`x_eugene_plexus`. None of it was kept, so "is vLLM faster than
llama.cpp for this model on this box" was not a question the system
could answer about itself. This module keeps it.

Design in `specs/docs/design/m8-retained-request-metrics.md`. Four
things here are not arbitrary:

**Two row shapes, not one.** A `request` row is what the client asked
for and got; an `attempt` row is one backend touched. They are separate
because `latency_ms` on a request includes *failed* attempts — a
cascade whose primary timed out for 30 s before a secondary answered in
800 ms would otherwise report that secondary as forty times slower than
it is. Throughput is computed from the serving attempt's own elapsed
time, and that is only possible if the attempt is a row.

**Writes never happen on the request path.** Rows go onto a queue and a
single background task drains it in batched transactions. A completion
must not wait on a disk flush or a write lock. When the queue is full
rows are **dropped and counted**, and the count is reported on every
summary — recording degrades before inference does, and it says so
rather than lying by omission.

**Errors are stored as an exception class name, never a message.**
Driver error text can carry a provider's response body, and a metrics
table is the kind of thing that gets pasted into an issue. This must
not become an accidental credential store.

**Not replicated.** M5 settled that liveness is never in the log; these
are per-request measurements of one process's traffic, which is further
from install-wide truth than liveness. The control root never sees this
file.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA_VERSION = 5

# Bounded because the alternative to dropping rows is stalling
# completions, and that trade is never worth making. Sized so a burst
# lasting several seconds is absorbed rather than sampled: a drop is a
# reported degradation, not a normal condition.
QUEUE_MAX = 4096

# One transaction per drain, so a batch of rows costs one commit. The
# batch is bounded so a huge backlog still commits incrementally rather
# than holding a write lock for an unbounded stretch.
BATCH_MAX = 256

# How long the writer waits for more rows before committing what it has.
# A completion takes seconds; a second of write latency is invisible and
# buys much better batching than a tight loop.
DRAIN_INTERVAL_SECONDS = 1.0

# Retention and rollup run on the writer's own loop rather than a second
# timer, because a second timer is a second thing to fail silently.
PRUNE_INTERVAL_SECONDS = 3600.0

# A decode rate qualifies only past a minimum window: a two-token answer
# reports a rate dominated by quantization, not by the GPU. The same
# guard the playground's badge uses (`clientTokPerSec`), kept equal on
# purpose so the two surfaces cannot disagree about one request.
DECODE_MIN_TOKENS = 2
DECODE_MIN_WINDOW_MS = 250


@dataclass(slots=True)
class CandidateRow:
    """One backend the balancer considered, and what it saw."""

    driver: str
    tier: int
    eligible: bool
    reason: str | None = None
    in_flight: int | None = None
    slots: int | None = None


@dataclass(slots=True)
class AttemptRow:
    """One backend touched by one request, in the order tried."""

    driver: str
    elapsed_ms: int
    served: bool
    runtime: str | None = None
    node: str | None = None
    backend: str | None = None
    error: str | None = None
    # The DRIVER's own measurement of its backend call. `elapsed_ms`
    # minus this is the control plane's overhead on the attempt - the
    # local hop plus the driver's work - which the design has asserted
    # is negligible since M0 without measuring it.
    backend_ms: int | None = None
    retry_disposition: str | None = None
    usage_known: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Time to this attempt's first streamed event — TTFT, gateway-side.
    # None for non-streamed attempts, which have no first token to time.
    first_ms: int | None = None


@dataclass(slots=True)
class RequestRow:
    """One completion the gateway accepted."""

    started_at: datetime
    requested_model: str
    total_ms: int
    attempts: int
    outcome: str
    tries: list[AttemptRow] = field(default_factory=list)
    served_model: str | None = None
    tier: int | None = None
    waited_ms: int = 0
    swapped_in: bool = False
    streamed: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Deciding where to send it: resolve, pick, and any routing-table
    # refresh. Measured before `total_ms` starts and excluding the wake.
    routing_ms: int | None = None
    refreshed: bool = False
    strategy: str | None = None
    # What the balancer saw, when there was a choice to make. Empty when
    # there was one eligible backend - that is not an audit trail.
    candidates: list[CandidateRow] = field(default_factory=list)
    client_key_id: str | None = None
    client_key_name: str | None = None
    request_id: str | None = None
    elapsed_ms: int | None = None


_DDL = """
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
    strategy          TEXT,
    client_key_id     TEXT,
    client_key_name   TEXT,
    correlation_id    TEXT,
    elapsed_ms        INTEGER
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
    backend_ms INTEGER,
    retry_disposition TEXT,
    usage_known INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    first_ms   INTEGER
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


def _iso(value: datetime) -> str:
    """UTC, second-resolution ISO 8601 with a `Z`.

    Text rather than a numeric epoch because these rows get read by a
    human with the sqlite3 CLI more often than they get read by anything
    else, and because lexical order matches chronological order for this
    format, which is what makes the `started_at` index useful for
    windows.
    """
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile on an already-sorted list.

    Deliberately not interpolated: these are millisecond observations
    and reporting a latency no request actually had, to make a chart
    smoother, is a worse answer.
    """
    if not values:
        return 0
    index = min(len(values) - 1, max(0, round(fraction * (len(values) - 1))))
    return values[index]


class MetricsStore:
    """SQLite-backed store with an off-path writer.

    Open with `start()`, close with `aclose()`. `record()` is safe to
    call from the request path and never blocks on I/O — it puts a row
    on a queue and returns, or drops it and counts the drop.
    """

    def __init__(
        self,
        path: Path,
        *,
        retention_days: int = 7,
        rollup_enabled: bool = True,
        queue_max: int = QUEUE_MAX,
    ) -> None:
        self._path = path
        self._retention_days = retention_days
        self._rollup_enabled = rollup_enabled
        self._queue: asyncio.Queue[RequestRow] = asyncio.Queue(maxsize=queue_max)
        self._task: asyncio.Task[None] | None = None
        self._conn: sqlite3.Connection | None = None
        # Held by everything that touches `_conn`. A cancelled writer task
        # does NOT stop the worker thread its `to_thread` call is running
        # on, so without this, closing on shutdown freed the connection
        # under a live INSERT - which on Windows is an access violation,
        # not an exception. Found by the existing suite on the first run.
        self._write_lock = threading.Lock()
        self._rows_dropped = 0
        self._started_at = datetime.now(UTC)
        self._last_prune = time.perf_counter()

    # --- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        self._conn = await asyncio.to_thread(self._open)
        self._task = asyncio.create_task(self._writer(), name="metrics-writer")

    def _open(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, check_same_thread=False)
        # WAL so a long read for the metrics endpoint cannot block a
        # write on the request path. NORMAL because losing the last few
        # measurements to a power cut is a fine trade for not fsyncing
        # per completion — this is not the replicated log.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(_DDL)
        conn.commit()

        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is not None and int(row[0]) in (2, 3, 4):
            columns = {r[1] for r in conn.execute("PRAGMA table_info(request)")}
            for column in ("client_key_id", "client_key_name"):
                if column not in columns:
                    conn.execute(f"ALTER TABLE request ADD COLUMN {column} TEXT")
            for table, additions in {
                "request": {"correlation_id": "TEXT", "elapsed_ms": "INTEGER"},
                "attempt": {
                    "retry_disposition": "TEXT",
                    "usage_known": "INTEGER NOT NULL DEFAULT 0",
                    "prompt_tokens": "INTEGER",
                    "completion_tokens": "INTEGER",
                    # v5: TTFT. Additive, so every older version migrates
                    # in place — the existence check makes this idempotent
                    # whatever version the file starts at.
                    "first_ms": "INTEGER",
                },
            }.items():
                existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
                for column, kind in additions.items():
                    if column not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
            conn.execute(
                "UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),)
            )
            conn.commit()
            row = (str(SCHEMA_VERSION),)
        if row is None:
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            conn.commit()
        elif int(row[0]) != SCHEMA_VERSION:
            # A version this build does not know: rename the file aside
            # and start fresh rather than refuse to boot. Metrics are
            # never worth a degraded control plane, and this is the one
            # place where discarding data is the right answer.
            found = row[0]
            conn.close()
            aside = self._path.with_suffix(f".v{found}.bak")
            log.warning(
                "metrics store at %s has schema version %s, expected %d; moving it to %s "
                "and starting fresh. History before now is in that file.",
                self._path,
                found,
                SCHEMA_VERSION,
                aside.name,
            )
            aside.unlink(missing_ok=True)
            self._path.replace(aside)
            return self._open()
        conn.execute(
            "CREATE INDEX IF NOT EXISTS request_client_key ON request (client_key_id, started_at)"
        )
        conn.commit()
        return conn

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        # Flush whatever is still queued, so a shutdown does not silently
        # discard the requests that prompted it, then close. Both under
        # the lock and in one thread hop, so an INSERT that was already
        # in flight finishes before the connection goes away.
        pending: list[RequestRow] = []
        while not self._queue.empty():
            pending.append(self._queue.get_nowait())
        await asyncio.to_thread(self._flush_and_close, pending)

    def _flush_and_close(self, pending: list[RequestRow]) -> None:
        with self._write_lock:
            conn = self._conn
            self._conn = None
            if conn is None:
                return
            if pending:
                with contextlib.suppress(sqlite3.Error):
                    self._insert_locked(conn, pending)
            with contextlib.suppress(sqlite3.Error):
                conn.close()

    # --- recording ----------------------------------------------------------

    @property
    def rows_dropped(self) -> int:
        return self._rows_dropped

    @property
    def started_at(self) -> datetime:
        return self._started_at

    def oldest_retained(self) -> datetime:
        """The earliest instant raw rows could still describe.

        Computed from the retention window rather than from the oldest
        row on disk, deliberately: an install that has only been running
        an hour has no rows from yesterday, but a window covering
        yesterday is not *truncated* — nothing was lost, there was
        nothing to lose. Retention is the only thing that discards.
        """
        return datetime.now(UTC) - timedelta(days=max(0, self._retention_days))

    def record(self, row: RequestRow) -> None:
        """Queue one request. Never raises, never blocks."""
        try:
            self._queue.put_nowait(row)
        except asyncio.QueueFull:
            self._rows_dropped += 1
            if self._rows_dropped == 1 or self._rows_dropped % 1000 == 0:
                log.warning(
                    "metrics write queue full; %d measurement(s) dropped so far. "
                    "Inference is unaffected and GET /v1/metrics reports the count.",
                    self._rows_dropped,
                )

    async def _writer(self) -> None:
        while True:
            batch: list[RequestRow] = []
            # A bounded wait rather than a plain get(), so the loop also
            # wakes on a quiet install — otherwise maintenance would only
            # ever run behind an arriving request, and the one install
            # that most needs pruning is the one that stopped serving.
            with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                batch.append(await asyncio.wait_for(self._queue.get(), DRAIN_INTERVAL_SECONDS))
            # Take whatever else is already waiting, up to the batch cap,
            # so a burst costs one commit rather than one per request.
            while len(batch) < BATCH_MAX and not self._queue.empty():
                batch.append(self._queue.get_nowait())

            if batch:
                try:
                    await asyncio.to_thread(self._insert, batch)
                except sqlite3.Error as e:
                    # A failed write must not kill the writer, or the
                    # first transient error would silently stop all
                    # recording for the life of the process.
                    log.warning("metrics write failed, dropping %d row(s): %s", len(batch), e)
                    self._rows_dropped += len(batch)

            if time.perf_counter() - self._last_prune >= PRUNE_INTERVAL_SECONDS:
                self._last_prune = time.perf_counter()
                try:
                    await asyncio.to_thread(self.maintain)
                except sqlite3.Error as e:
                    log.warning("metrics maintenance failed: %s", e)

    def _insert(self, batch: Iterable[RequestRow]) -> None:
        with self._write_lock:
            conn = self._conn
            if conn is None:
                return
            self._insert_locked(conn, batch)

    def _insert_locked(self, conn: sqlite3.Connection, batch: Iterable[RequestRow]) -> None:
        with conn:
            for row in batch:
                cursor = conn.execute(
                    "INSERT INTO request (started_at, requested_model, served_model, attempts,"
                    " tier, total_ms, waited_ms, swapped_in, streamed, prompt_tokens,"
                    " completion_tokens, outcome, routing_ms, refreshed, strategy,"
                    " client_key_id, client_key_name, correlation_id, elapsed_ms)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        _iso(row.started_at),
                        row.requested_model,
                        row.served_model,
                        row.attempts,
                        row.tier,
                        row.total_ms,
                        row.waited_ms,
                        int(row.swapped_in),
                        int(row.streamed),
                        row.prompt_tokens,
                        row.completion_tokens,
                        row.outcome,
                        row.routing_ms,
                        int(row.refreshed),
                        row.strategy,
                        row.client_key_id,
                        row.client_key_name,
                        row.request_id,
                        row.elapsed_ms,
                    ),
                )
                request_id = cursor.lastrowid
                conn.executemany(
                    "INSERT INTO attempt (request_id, seq, driver, runtime, node, backend,"
                    " elapsed_ms, served, error, backend_ms, retry_disposition, usage_known,"
                    " prompt_tokens, completion_tokens, first_ms)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            request_id,
                            seq,
                            a.driver,
                            a.runtime,
                            a.node,
                            a.backend,
                            a.elapsed_ms,
                            int(a.served),
                            a.error,
                            a.backend_ms,
                            a.retry_disposition,
                            int(a.usage_known),
                            a.prompt_tokens,
                            a.completion_tokens,
                            a.first_ms,
                        )
                        for seq, a in enumerate(row.tries)
                    ],
                )
                conn.executemany(
                    "INSERT INTO candidate (request_id, seq, driver, tier, eligible, reason,"
                    " in_flight, slots) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            request_id,
                            seq,
                            c.driver,
                            c.tier,
                            int(c.eligible),
                            c.reason,
                            c.in_flight,
                            c.slots,
                        )
                        for seq, c in enumerate(row.candidates)
                    ],
                )

    # --- retention ----------------------------------------------------------

    def maintain(self, *, now: datetime | None = None) -> None:
        """Roll up completed hours, then drop raw rows past retention.

        Rollup before prune, always: the other order loses the traffic it
        was about to summarise. Public so the acceptance script can
        advance it by hand rather than waiting an hour.
        """
        with self._write_lock:
            conn = self._conn
            if conn is None:
                return
            now = now or datetime.now(UTC)
            if self._rollup_enabled:
                self._rollup(conn, now)
            if self._retention_days >= 0:
                cutoff = _iso(now - timedelta(days=self._retention_days))
                with conn:
                    conn.execute("DELETE FROM request WHERE started_at < ?", (cutoff,))

    def _rollup(self, conn: sqlite3.Connection, now: datetime) -> None:
        """Aggregate every fully-elapsed hour not already rolled up.

        Only complete hours: rolling up the current one would produce a
        bucket that a later pass would have to correct, and an
        `INSERT OR REPLACE` that overwrites a partial bucket with a
        second partial bucket is how double-counting starts.
        """
        boundary = _iso(now.replace(minute=0, second=0, microsecond=0))
        done = conn.execute("SELECT value FROM meta WHERE key = 'rollup_through'").fetchone()
        since = done[0] if done else "0000-01-01T00:00:00Z"
        if since >= boundary:
            return
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO rollup (
                    bucket_start, requested_model, driver, runtime, node, backend,
                    requests, errors, cascaded, swapped_in,
                    total_ms_sum, total_ms_max, served_ms_sum,
                    completion_tokens, token_samples
                )
                SELECT
                    substr(r.started_at, 1, 13) || ':00:00Z',
                    r.requested_model,
                    COALESCE(a.driver, ''),
                    COALESCE(a.runtime, ''),
                    COALESCE(a.node, ''),
                    COALESCE(a.backend, ''),
                    COUNT(*),
                    SUM(CASE WHEN r.outcome = 'error' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN r.attempts > 1 THEN 1 ELSE 0 END),
                    SUM(r.swapped_in),
                    SUM(r.total_ms),
                    MAX(r.total_ms),
                    SUM(COALESCE(a.elapsed_ms, r.total_ms)),
                    SUM(COALESCE(r.completion_tokens, 0)),
                    SUM(CASE WHEN r.completion_tokens IS NOT NULL THEN 1 ELSE 0 END)
                FROM request r
                LEFT JOIN attempt a ON a.request_id = r.id AND a.served = 1
                WHERE r.started_at >= ? AND r.started_at < ?
                GROUP BY 1, 2, 3, 4, 5, 6
                """,
                (since, boundary),
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('rollup_through', ?)",
                (boundary,),
            )

    # --- reading ------------------------------------------------------------

    @contextlib.contextmanager
    def _reader(self) -> Iterator[sqlite3.Connection | None]:
        """A short-lived read connection of its own.

        Not the writer's connection. Sharing one across threads needs a
        lock around every use, which would serialize reads behind writes
        and throw away the reason WAL mode was chosen. A separate
        connection is what makes a long report harmless to the request
        path. Opening one is microseconds.

        Yields None once the store is closed, so a read racing shutdown
        answers empty rather than raising.
        """
        if self._conn is None:
            yield None
            return
        conn = sqlite3.connect(self._path, check_same_thread=False)
        try:
            yield conn
        finally:
            with contextlib.suppress(sqlite3.Error):
                conn.close()

    def summary(
        self,
        *,
        since: datetime,
        until: datetime,
        model: str | None = None,
        driver: str | None = None,
        bucket: str = "none",
        group_by: str = "backend",
    ) -> list[dict[str, Any]]:
        """Aggregate raw rows over a window, grouped as the contract says.

        Aggregated in Python rather than SQL because the percentiles and
        the "throughput from the serving attempt" rule are both clearer
        expressed once than as window functions, and the row counts here
        are small: a window is bounded by retention, and retention is
        bounded by a config key with a documented size.

        `group_by` collapses the dimension tuple — `model` drops the
        backend dimensions, `total` drops the model too — and it is a
        server-side parameter rather than client-side math because
        percentiles do not recombine: the install-wide p90 is not
        computable from per-backend p90s, so the coarse grain must be
        computed here, over the raw rows.
        """
        where = ["r.started_at >= ?", "r.started_at < ?"]
        params: list[Any] = [_iso(since), _iso(until)]
        if model:
            where.append("r.requested_model = ?")
            params.append(model)
        if driver:
            where.append("served.driver = ?")
            params.append(driver)

        with self._reader() as conn:
            if conn is None:
                return []
            rows = conn.execute(
                f"""
            SELECT r.started_at, r.requested_model, r.outcome, r.attempts, r.tier,
                   r.total_ms, r.waited_ms, r.swapped_in, r.completion_tokens,
                   served.driver, served.runtime, served.node, served.backend,
                   served.elapsed_ms, r.routing_ms, served.backend_ms,
                   served.first_ms, served.served
            FROM request r
            -- The attempt a request is ATTRIBUTED to: the one that served
            -- it, or, when none did, the last one tried.
            --
            -- Joining on `served = 1` alone loses a failed request
            -- entirely: it lands in a group with a null driver and a null
            -- backend, so "is anything failing" - the question this is
            -- most for - can be answered with a count but not with a
            -- name. Found live the first time a backend really failed,
            -- with a Codex CLI whose token needed refreshing: the row was
            -- there, correct, and attributed to nobody.
            --
            -- `ORDER BY served DESC` puts the serving attempt first when
            -- there is one; `seq DESC` otherwise takes the last backend
            -- tried, which is the one whose failure ended the request.
            LEFT JOIN attempt served
                   ON served.id = (
                       SELECT a2.id FROM attempt a2
                       WHERE a2.request_id = r.id
                       ORDER BY a2.served DESC, a2.seq DESC
                       LIMIT 1
                   )
            WHERE {" AND ".join(where)}
            """,
                params,
            ).fetchall()

        groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        for (
            started_at,
            requested_model,
            outcome,
            attempts,
            tier,
            total_ms,
            waited_ms,
            swapped_in,
            completion_tokens,
            s_driver,
            s_runtime,
            s_node,
            s_backend,
            s_elapsed,
            routing_ms,
            s_backend_ms,
            s_first_ms,
            s_served,
        ) in rows:
            bucket_start = started_at[:13] + ":00:00Z" if bucket == "hour" else None
            key: tuple[Any, ...]
            if group_by == "total":
                key = (bucket_start,)
            elif group_by == "model":
                key = (bucket_start, requested_model)
            else:
                key = (bucket_start, requested_model, s_driver, s_runtime, s_node, s_backend)
            g = groups.get(key)
            if g is None:
                g = groups[key] = {
                    "bucketStart": bucket_start,
                    # A collapsed grain omits the dimensions it collapsed
                    # rather than carrying whichever value arrived first —
                    # a `total` group naming one model would be a lie told
                    # by iteration order.
                    "model": requested_model if group_by != "total" else None,
                    "driver": s_driver if group_by == "backend" else None,
                    "runtime": s_runtime if group_by == "backend" else None,
                    "node": s_node if group_by == "backend" else None,
                    "backend": s_backend if group_by == "backend" else None,
                    "requests": 0,
                    "errors": 0,
                    "cascaded": 0,
                    "swappedIn": 0,
                    "_latency": [],
                    "_waited": [],
                    "_tps": [],
                    "_ttft": [],
                    "_decode": [],
                    "_routing": [],
                    "_overhead": [],
                    "tierCounts": {},
                }
            g["requests"] += 1
            if outcome == "error":
                g["errors"] += 1
            if attempts and attempts > 1:
                g["cascaded"] += 1
            if swapped_in:
                g["swappedIn"] += 1
                g["_waited"].append(int(waited_ms or 0))
            g["_latency"].append(int(total_ms or 0))
            if outcome != "error" and tier:
                key_t = str(tier)
                g["tierCounts"][key_t] = g["tierCounts"].get(key_t, 0) + 1
            # Throughput from the SERVING attempt's elapsed time. Using
            # total_ms here is the mistake this whole two-table shape
            # exists to prevent.
            if completion_tokens and s_elapsed and s_elapsed > 0:
                g["_tps"].append(completion_tokens * 1000.0 / s_elapsed)
            # TTFT and decode rate, from attempts that SERVED only. The
            # attributed attempt for a failed request is the last one
            # tried, and a stream that died mid-answer has a first_ms
            # too — mixing its timing into a group's TTFT would report
            # the failure's latency as the backend's speed.
            if s_served and s_first_ms is not None:
                g["_ttft"].append(int(s_first_ms))
                window = int(s_elapsed or 0) - int(s_first_ms)
                if (
                    completion_tokens
                    and completion_tokens >= DECODE_MIN_TOKENS
                    and window >= DECODE_MIN_WINDOW_MS
                ):
                    g["_decode"].append(completion_tokens * 1000.0 / window)
            if routing_ms is not None:
                g["_routing"].append(int(routing_ms))
            # The control plane's own cost. Clamped at zero because the
            # two clocks are different processes' and a sub-millisecond
            # hop can round to the driver measuring marginally MORE than
            # the gateway did - a negative overhead is a rounding
            # artefact, not a discovery.
            if s_elapsed is not None and s_backend_ms is not None:
                g["_overhead"].append(max(0, int(s_elapsed) - int(s_backend_ms)))

        out: list[dict[str, Any]] = []
        for g in groups.values():
            latency = sorted(g.pop("_latency"))
            waited = sorted(g.pop("_waited"))
            tps = sorted(g.pop("_tps"))
            ttft = sorted(g.pop("_ttft"))
            decode = sorted(g.pop("_decode"))
            routing = sorted(g.pop("_routing"))
            overhead = sorted(g.pop("_overhead"))
            g["routingMs"] = _spread(routing)
            g["overheadMs"] = _spread(overhead)
            g["latencyMs"] = {
                "p50": _percentile(latency, 0.50),
                "p90": _percentile(latency, 0.90),
                "p99": _percentile(latency, 0.99),
                "max": latency[-1] if latency else 0,
            }
            g["waitedMs"] = (
                {
                    "p50": _percentile(waited, 0.50),
                    "p90": _percentile(waited, 0.90),
                    "p99": _percentile(waited, 0.99),
                    "max": waited[-1],
                }
                if waited
                else None
            )
            # Null rather than a zeroed object: "no backend here reports
            # token usage" and "these backends produced zero tokens per
            # second" are different facts, and the CLI subscription
            # backends are permanently the former.
            g["tokensPerSecond"] = (
                {
                    "p50": round(_pct_float(tps, 0.50), 2),
                    "p90": round(_pct_float(tps, 0.90), 2),
                    "samples": len(tps),
                }
                if tps
                else None
            )
            # Null, not zeroed, for the same reason as tokensPerSecond:
            # a group of non-streamed requests has no first token to
            # time, and "unmeasured" must stay distinguishable from
            # "instant".
            g["ttftMs"] = (
                {
                    "p50": _percentile(ttft, 0.50),
                    "p90": _percentile(ttft, 0.90),
                    "p99": _percentile(ttft, 0.99),
                    "max": ttft[-1],
                }
                if ttft
                else None
            )
            g["decodeTokensPerSecond"] = (
                {
                    "p50": round(_pct_float(decode, 0.50), 2),
                    "p90": round(_pct_float(decode, 0.90), 2),
                    "samples": len(decode),
                }
                if decode
                else None
            )
            out.append(g)
        out.sort(key=lambda x: (x["bucketStart"] or "", -x["requests"]))
        return out

    def client_usage(self, *, since: datetime, until: datetime) -> list[dict[str, Any]]:
        with self._reader() as conn:
            if conn is None:
                return []
            rows = conn.execute(
                """
                SELECT r.client_key_id,
                       COALESCE((SELECT n.client_key_name FROM request n
                         WHERE n.client_key_id = r.client_key_id AND n.client_key_name IS NOT NULL
                         ORDER BY n.id DESC LIMIT 1), 'Name unavailable'),
                       COUNT(*), SUM(CASE WHEN outcome = 'served' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN outcome != 'served' THEN 1 ELSE 0 END), SUM(attempts),
                       SUM(COALESCE(prompt_tokens, 0)), SUM(COALESCE(completion_tokens, 0)),
                       SUM(CASE WHEN prompt_tokens IS NULL OR completion_tokens IS NULL
                                OR attempts > 1 OR outcome != 'served' THEN 1 ELSE 0 END)
                FROM request r WHERE r.client_key_id IS NOT NULL
                  AND started_at >= ? AND started_at < ?
                GROUP BY r.client_key_id ORDER BY r.client_key_id
            """,
                (_iso(since), _iso(until)),
            ).fetchall()
        names = (
            "clientKeyId",
            "clientKeyName",
            "requests",
            "served",
            "failed",
            "attempts",
            "promptTokens",
            "completionTokens",
            "incompleteUsageRequests",
        )
        return [dict(zip(names, row, strict=True)) for row in rows]

    def requests(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
        model: str | None = None,
        outcome: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """One page of raw requests, newest first, each with its attempts.

        Keyset-paged on the autoincrement id rather than OFFSET, so a
        page stays stable while new requests arrive underneath it.
        """
        where: list[str] = []
        params: list[Any] = []
        if cursor:
            where.append("r.id < ?")
            params.append(int(cursor))
        if model:
            where.append("r.requested_model = ?")
            params.append(model)
        if outcome == "cascaded":
            where.append("r.attempts > 1")
        elif outcome in {"served", "error"}:
            where.append("r.outcome = ?")
            params.append(outcome)
        clause = f"WHERE {' AND '.join(where)}" if where else ""

        with self._reader() as conn:
            if conn is None:
                return [], None
            # limit + 1 so the presence of another page is known without a
            # second COUNT query over the same predicate.
            rows = conn.execute(
                f"""
                SELECT id, started_at, requested_model, served_model, attempts, tier,
                       total_ms, waited_ms, swapped_in, streamed, prompt_tokens,
                       completion_tokens, outcome, routing_ms, refreshed, strategy,
                       client_key_id, client_key_name, correlation_id, elapsed_ms
                FROM request r {clause}
                ORDER BY id DESC LIMIT ?
                """,
                [*params, limit + 1],
            ).fetchall()

            more = len(rows) > limit
            rows = rows[:limit]
            if not rows:
                return [], None

            # One query for every attempt on the page, rather than one per
            # request: a page of 50 cascading requests would otherwise be
            # 51 round trips to answer one question.
            ids = [r[0] for r in rows]
            placeholders = ",".join("?" * len(ids))
            tries: dict[int, list[dict[str, Any]]] = {i: [] for i in ids}
            for (
                request_id,
                driver,
                runtime,
                node,
                backend,
                elapsed_ms,
                served,
                error,
                backend_ms,
                retry_disposition,
                usage_known,
                prompt_tokens,
                completion_tokens,
                first_ms,
            ) in conn.execute(
                f"""
                SELECT request_id, driver, runtime, node, backend, elapsed_ms, served,
                       error, backend_ms, retry_disposition, usage_known,
                       prompt_tokens, completion_tokens, first_ms
                FROM attempt WHERE request_id IN ({placeholders}) ORDER BY request_id, seq
                """,
                ids,
            ).fetchall():
                tries[request_id].append(
                    {
                        "driver": driver,
                        "runtime": runtime,
                        "node": node,
                        "backend": backend,
                        "elapsedMs": elapsed_ms,
                        "served": bool(served),
                        "error": error,
                        "backendMs": backend_ms,
                        "retryDisposition": retry_disposition,
                        "usageKnown": bool(usage_known),
                        "promptTokens": prompt_tokens,
                        "completionTokens": completion_tokens,
                        "firstMs": first_ms,
                    }
                )

            # Same one-query-per-page shape as the attempts above.
            considered: dict[int, list[dict[str, Any]]] = {i: [] for i in ids}
            for (
                request_id,
                driver,
                tier,
                eligible,
                reason,
                in_flight,
                slots,
            ) in conn.execute(
                f"""
                SELECT request_id, driver, tier, eligible, reason, in_flight, slots
                FROM candidate WHERE request_id IN ({placeholders}) ORDER BY request_id, seq
                """,
                ids,
            ).fetchall():
                considered[request_id].append(
                    {
                        "driver": driver,
                        "tier": tier,
                        "eligible": bool(eligible),
                        "reason": reason,
                        "inFlight": in_flight,
                        "slots": slots,
                    }
                )

        out = [
            {
                "startedAt": _parse(r[1]),
                "requestedModel": r[2],
                "servedModel": r[3],
                "attempts": r[4],
                "tier": r[5],
                "totalMs": r[6],
                "waitedMs": r[7],
                "swappedIn": bool(r[8]),
                "streamed": bool(r[9]),
                "promptTokens": r[10],
                "completionTokens": r[11],
                "outcome": r[12],
                "routingMs": r[13],
                "refreshed": bool(r[14]),
                "strategy": r[15],
                "clientKeyId": r[16],
                "clientKeyName": r[17],
                "requestId": r[18],
                "elapsedMs": r[19],
                "tries": tries[r[0]],
                "candidates": considered[r[0]],
            }
            for r in rows
        ]
        return out, (str(rows[-1][0]) if more else None)


def _spread(values: list[int]) -> dict[str, int] | None:
    """Percentiles for a sorted list, or None when there is nothing.

    None rather than a zeroed object, for the same reason throughput is:
    "not measured" and "measured as zero" are different facts and a UI
    cannot recover the difference.
    """
    if not values:
        return None
    return {
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "max": values[-1],
    }


def _pct_float(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round(fraction * (len(values) - 1))))
    return values[index]
