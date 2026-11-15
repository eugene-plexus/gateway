"""The gateway's numbers for a scraper and a collector (row 12).

What these show: the page a Prometheus scraper reads is made from the requests
the gateway served, under names that start with `eugene_`; a made-up model name
makes no series; the page is read by an operator or by a key with `readMetrics`
and by nobody else; a decision is a request row; hours past retention come from
the stored rollups; an exported byte carries no prompt; and a collector that is
down costs nothing but its own state.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway import telemetry
from eugene_plexus_gateway._generated.driver_models import Usage
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.metrics import AttemptRow, MetricsStore, RequestRow
from eugene_plexus_gateway.routes import inference
from eugene_plexus_gateway.routes import telemetry as telemetry_route
from eugene_plexus_gateway.settings import Settings

from .conftest import (
    FakeDriverClient,
    FakeInstall,
    install_snapshot,
    make_routing_table,
    runtime_facts,
)
from .test_client_keys import FakeAgent

# --------------------------------------------------------------------- #
# reading the page
# --------------------------------------------------------------------- #

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})? (\S+)$")


def samples(page: str) -> list[tuple[str, dict[str, str], float]]:
    out = []
    for line in page.splitlines():
        if not line or line.startswith("#"):
            continue
        found = _LINE.match(line)
        assert found, f"not Prometheus text: {line!r}"
        name, labels, value = found.groups()
        pairs = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', labels or ""))
        out.append((name, pairs, float(value)))
    return out


def value(page: str, name: str, **labels: str) -> float | None:
    """The one sample of `name` whose labels include `labels`, or None."""
    found = [
        v
        for n, pairs, v in samples(page)
        if n == name and all(pairs.get(k) == want for k, want in labels.items())
    ]
    assert len(found) <= 1, f"{name}{labels} matched {len(found)} series"
    return found[0] if found else None


class SlowFake(FakeDriverClient):
    """A fake that takes measurable time, so timings are not all zero."""

    async def generate(self, request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0.02)
        return await super().generate(request)

    async def stream(self, request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0.02)
        async for event in super().stream(request):
            yield event


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    app = create_app(settings=settings)
    fake = SlowFake(name="qwen-box", model_id="qwen", runtime="qwen-rt")
    fake.usage = Usage(promptTokens=10, completionTokens=40, totalTokens=50)
    app.state.routing = make_routing_table(fake, runtimes=[runtime_facts("qwen-rt")])
    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def _chat(client: TestClient, content: str = "hi", *, model: str = "qwen", stream: bool = False):
    body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": content}]}
    if stream:
        body["stream"] = True
        with client.stream("POST", "/v1/chat/completions", json=body) as response:
            assert response.status_code == 200
            return [line for line in response.iter_lines() if line.startswith("data: ")]
    return client.post("/v1/chat/completions", json=body)


# --------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------- #


def test_metrics_shows_ttft_and_requests_after_a_stream(client: TestClient) -> None:
    _chat(client, stream=True)
    _chat(client)
    page = client.get("/metrics")
    assert page.status_code == 200, page.text
    assert page.headers["content-type"].startswith("text/plain")
    text = page.text
    assert value(text, "eugene_requests_total", door="chat", outcome="served", model="qwen") == 2
    assert value(text, "eugene_time_to_first_token_seconds_count", model="qwen") == 1
    assert value(text, "eugene_request_duration_seconds_count", door="chat", model="qwen") == 2
    assert value(text, "eugene_tokens_total", type="input", model="qwen", runtime="qwen-rt") == 20
    assert value(text, "eugene_tokens_total", type="output", model="qwen", runtime="qwen-rt") == 80
    assert (
        value(
            text,
            "eugene_attempts_total",
            driver="qwen-box",
            runtime="qwen-rt",
            served="true",
            model="qwen",
        )
        == 2
    )
    # Gauges are read when the scrape arrives, and the build says what answered.
    assert value(text, "eugene_in_flight", runtime="qwen-rt") == 0
    assert value(text, "eugene_build_info", component="gateway") == 1
    assert value(text, "eugene_metrics_enabled") == 1
    # Histogram buckets are cumulative and fixed.
    assert (
        value(text, "eugene_request_duration_seconds_bucket", model="qwen", le="+Inf", door="chat")
        == 2
    )
    assert "# TYPE eugene_requests_total counter" in text


def test_a_made_up_model_name_makes_no_series(client: TestClient) -> None:
    """The row of a request for a model nobody serves is counted under `other`:
    the name a caller types is theirs, and a series per name is a leak and a
    memory cost."""
    store = client.app.state.metrics
    for number in range(50):
        store.record(
            RequestRow(
                started_at=datetime.now(UTC),
                requested_model=f"zz-made-up-{number}",
                total_ms=5,
                attempts=1,
                outcome="error",
                tries=[AttemptRow(driver="qwen-box", elapsed_ms=5, served=False)],
            )
        )
    page = client.get("/metrics").text
    assert "zz-made-up" not in page
    assert value(page, "eugene_requests_total", model="other", outcome="error") == 50


def test_label_sets_past_the_cap_are_counted_and_not_kept() -> None:
    numbers = telemetry.Telemetry()
    for number in range(telemetry.MAX_SERIES + 25):
        numbers.observe(
            RequestRow(
                started_at=datetime.now(UTC),
                requested_model="m",
                total_ms=1,
                attempts=1,
                outcome="served",
                client_key_id=f"key-{number}",
            )
        )
    page = numbers.render({})
    kept = [s for s in samples(page) if s[0] == "eugene_requests_total"]
    assert len(kept) == telemetry.MAX_SERIES
    assert (
        value(page, "eugene_telemetry_series_dropped_total", family="eugene_requests_total") == 25
    )


def test_a_failed_attempt_and_a_cascade_are_counted(client: TestClient) -> None:
    store = client.app.state.metrics
    store.record(
        RequestRow(
            started_at=datetime.now(UTC),
            requested_model="qwen",
            served_model="qwen",
            total_ms=900,
            attempts=2,
            tier=1,
            outcome="served",
            affinity="hit",
            tries=[
                AttemptRow(
                    driver="a-box",
                    runtime="a-rt",
                    elapsed_ms=600,
                    served=False,
                    retry_disposition="safe_to_retry",
                    error="ConnectError",
                ),
                AttemptRow(driver="b-box", runtime="b-rt", elapsed_ms=300, served=True),
            ],
        )
    )
    page = client.get("/metrics").text
    assert (
        value(
            page,
            "eugene_attempts_total",
            driver="a-box",
            served="false",
            retry_disposition="safe_to_retry",
        )
        == 1
    )
    assert value(page, "eugene_cascades_total", driver="a-box", model="qwen") == 1
    assert value(page, "eugene_affinity_total", model="qwen", outcome="hit") == 1


def test_a_refused_request_is_counted_with_its_reason_and_band(client: TestClient) -> None:
    client.app.state.metrics.record(
        RequestRow(
            started_at=datetime.now(UTC),
            requested_model="qwen",
            total_ms=2,
            attempts=0,
            outcome="refused",
            refusal="queue_full",
            band="batch",
            queued_ms=0,
        )
    )
    page = client.get("/metrics").text
    assert value(page, "eugene_queue_refused_total", reason="queue_full", band="batch") == 1
    assert value(page, "eugene_requests_total", outcome="refused", model="qwen") == 1
    # A refusal is not a request that took time.
    assert value(page, "eugene_request_duration_seconds_count", outcome="refused") is None


# --------------------------------------------------------------------- #
# who may read it
# --------------------------------------------------------------------- #


class ScrapeAgent(FakeAgent):
    """The admission authority for keys with their own limits, as control answers."""

    def __init__(self) -> None:
        super().__init__()
        self.limits: dict[str, dict[str, Any]] = {}

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/auth/client-keys/admission":
            return super()._handle(request)
        body = json.loads(request.content)
        key = body["keyId"]
        if key in self.revoked and body["action"] != "release":
            return httpx.Response(401)
        limits = self.limits.get(key)
        purpose = body.get("purpose", "inference")
        if purpose == "metrics":
            if body["action"] != "check" or body.get("model"):
                return httpx.Response(403)
            if not (limits or {}).get("readMetrics"):
                return httpx.Response(403)
        elif body.get("model") and limits is not None:
            allowed = limits.get("allowedModels")
            if allowed is not None and body["model"] not in allowed:
                return httpx.Response(403)
        answer: dict[str, Any] = {"keyId": key, "keyName": f"name of {key}", "leaseSeconds": 30}
        if limits is not None:
            answer["limits"] = limits
        return httpx.Response(200, json=answer)


@pytest.fixture
def agent() -> ScrapeAgent:
    found = ScrapeAgent()
    found.limits = {
        # What the console's "Make a scrape key" makes: numbers, no model, no tool.
        "key-scrape": {"readMetrics": True, "allowedModels": [], "allowedTools": []},
        "key-chat": {"allowedModels": None},
    }
    found.registered = ["key-scrape", "key-chat"]
    return found


@pytest.fixture
def authed(
    settings: Settings, install: FakeInstall, agent: ScrapeAgent
) -> Iterator[tuple[TestClient, FakeInstall]]:
    app = create_app(settings=settings)
    fake = SlowFake(name="qwen-box", model_id="qwen", runtime="qwen-rt")
    app.state.routing = make_routing_table(fake, runtimes=[runtime_facts("qwen-rt")])
    app.state.auth_state = install.auth_state()
    app.state.client_key_guard = agent.as_guard(ttl_seconds=0.0)
    with TestClient(app) as c:
        yield c, install


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_metrics_refuses_no_key_and_a_key_without_read_metrics(
    authed: tuple[TestClient, FakeInstall],
) -> None:
    client, install = authed
    assert client.get("/metrics").status_code == 401
    inference_key = install.client_key(name="chat", jti="key-chat")
    refused = client.get("/metrics", headers=_bearer(inference_key))
    assert refused.status_code == 403
    assert "May read metrics" in refused.text
    assert client.get("/v1/metrics/export", headers=_bearer(inference_key)).status_code == 403
    # A service token is no operator and no scraper.
    service = install.service("gateway")
    assert client.get("/metrics", headers=_bearer(service)).status_code == 401


def test_a_scrape_key_reads_metrics_and_the_operator_does_too(
    authed: tuple[TestClient, FakeInstall],
) -> None:
    client, install = authed
    scrape = install.client_key(name="prometheus", jti="key-scrape")
    page = client.get("/metrics", headers=_bearer(scrape))
    assert page.status_code == 200, page.text
    assert value(page.text, "eugene_build_info", component="gateway") == 1
    assert client.get("/metrics", headers=_bearer(install.session())).status_code == 200
    export = client.get("/v1/metrics/export", headers=_bearer(scrape)).json()
    assert export["prometheus"]["path"] == "/metrics"
    assert export["prometheus"]["lastScrapeKeyName"] == "operator"
    assert export["otlp"]["state"] == "off"


def test_the_export_says_which_key_scraped_last(
    authed: tuple[TestClient, FakeInstall],
) -> None:
    client, install = authed
    scrape = install.client_key(name="prometheus", jti="key-scrape")
    client.get("/metrics", headers=_bearer(scrape))
    export = client.get("/v1/metrics/export", headers=_bearer(scrape)).json()
    assert export["prometheus"]["lastScrapeKeyName"] == "name of key-scrape"
    assert export["prometheus"]["lastScrapeAt"] is not None


def test_a_scrape_key_runs_no_inference_and_an_inference_key_reads_no_metrics(
    authed: tuple[TestClient, FakeInstall],
) -> None:
    client, install = authed
    scrape = install.client_key(name="prometheus", jti="key-scrape")
    chat = install.client_key(name="chat", jti="key-chat")
    body = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
    assert (
        client.post("/v1/chat/completions", json=body, headers=_bearer(scrape)).status_code == 403
    )
    # Nor a web search: absent `allowedTools` would permit every tool the install runs.
    searched = client.post("/v1/web_search", json={"query": "anything"}, headers=_bearer(scrape))
    assert searched.status_code == 403, searched.text
    assert client.post("/v1/chat/completions", json=body, headers=_bearer(chat)).status_code == 200
    assert client.get("/metrics", headers=_bearer(chat)).status_code == 403
    # And the scrape key opens no operator path.
    assert client.get("/v1/metrics", headers=_bearer(scrape)).status_code == 401
    assert client.get("/v1/config", headers=_bearer(scrape)).status_code == 401


def test_a_revoked_scrape_key_is_refused(
    authed: tuple[TestClient, FakeInstall], agent: ScrapeAgent
) -> None:
    client, install = authed
    scrape = install.client_key(name="prometheus", jti="key-scrape")
    assert client.get("/metrics", headers=_bearer(scrape)).status_code == 200
    agent.revoke("key-scrape")
    # The kept standing is not cleared: the gateway's own key policy refuses a
    # revoked key at once, before the standing is read.
    assert client.get("/metrics", headers=_bearer(scrape)).status_code == 401


def test_a_key_that_lost_the_grant_is_refused_once_the_standing_expires(
    authed: tuple[TestClient, FakeInstall],
    agent: ScrapeAgent,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, install = authed
    scrape = install.client_key(name="prometheus", jti="key-scrape")
    assert client.get("/metrics", headers=_bearer(scrape)).status_code == 200
    agent.limits["key-scrape"] = {"allowedModels": []}
    # Held, the standing still says yes: it is kept for a few seconds so a scraper
    # every fifteen is not a question to the root each time.
    assert client.get("/metrics", headers=_bearer(scrape)).status_code == 200
    # Expired at once, the same cache asks again each time, and the answer is the
    # authority's.
    monkeypatch.setattr(telemetry_route, "_STANDING_SECONDS", 0.0)
    client.app.state.metrics_keys.clear()
    agent.limits["key-scrape"] = {"readMetrics": True, "allowedModels": []}
    assert client.get("/metrics", headers=_bearer(scrape)).status_code == 200
    agent.limits["key-scrape"] = {"allowedModels": []}
    assert client.get("/metrics", headers=_bearer(scrape)).status_code == 403


def test_a_wider_answer_than_the_grant_is_not_taken_for_it(
    authed: tuple[TestClient, FakeInstall], agent: ScrapeAgent
) -> None:
    """An authority that answers 200 for a metrics check does not grant it: the
    gateway reads `limits.readMetrics` itself."""
    client, install = authed
    chat = install.client_key(name="chat", jti="key-chat")

    original = agent._handle

    def lenient(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/auth/client-keys/admission":
            return httpx.Response(
                200,
                json={"keyId": "key-chat", "keyName": "chat", "limits": {"allowedModels": None}},
            )
        return original(request)

    client.app.state.client_key_guard = agent.as_guard(ttl_seconds=0.0)
    client.app.state.client_key_guard._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lenient)
    )
    assert client.get("/metrics", headers=_bearer(chat)).status_code == 403


# --------------------------------------------------------------------- #
# a decision is a request row (B2-MET)
# --------------------------------------------------------------------- #


def _decider() -> FakeDriverClient:
    return FakeDriverClient(
        name="kev-driver", model_id="tickets", supports_decisions=True, decision_max_concurrent=1
    )


def _decision_app(settings: Settings, driver: FakeDriverClient) -> FastAPI:
    app = create_app(settings=settings)
    table = make_routing_table(slots=[])
    install_snapshot(table, driver)
    app.state.routing = table
    return app


_QUESTIONS = {"refunded": {"type": "noul", "instructions": "Was a refund issued?"}}


def _rows(client: TestClient) -> list[dict[str, Any]]:
    for _ in range(100):
        page = client.get("/v1/metrics/requests").json()
        if page["requests"]:
            return page["requests"]
        time.sleep(0.05)
    return []


def test_a_decision_writes_a_row(settings: Settings) -> None:
    driver = _decider()
    driver.usage = Usage(promptTokens=89, completionTokens=156, totalTokens=245)
    with TestClient(_decision_app(settings, driver)) as client:
        response = client.post(
            "/v1/systemone", json={"model": "tickets", "state": "x", "questions": _QUESTIONS}
        )
        assert response.status_code == 200, response.text
        rows = _rows(client)
        page = client.get("/metrics").text
    assert len(rows) == 1
    assert rows[0]["door"] == "decisions"
    assert rows[0]["outcome"] == "served"
    assert rows[0]["requestedModel"] == "tickets"
    assert rows[0]["promptTokens"] == 89
    assert rows[0]["tries"][0]["driver"] == "kev-driver"
    assert value(page, "eugene_requests_total", door="decisions", outcome="served") == 1


def test_a_failed_decision_writes_an_error_row(settings: Settings) -> None:
    driver = _decider()
    driver.decide_error = httpx.ConnectError("refused")
    with TestClient(_decision_app(settings, driver)) as client:
        response = client.post(
            "/v1/systemone", json={"model": "tickets", "state": "x", "questions": _QUESTIONS}
        )
        assert response.status_code >= 400
        rows = _rows(client)
    assert rows and rows[0]["door"] == "decisions"
    assert rows[0]["outcome"] == "error"


# --------------------------------------------------------------------- #
# the affinity key's kind (gateway#19)
# --------------------------------------------------------------------- #


def test_the_row_says_which_key_placed_the_request(client: TestClient) -> None:
    _chat(client, "opening words")
    client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen",
            "messages": [{"role": "user", "content": "x"}],
            "prompt_cache_key": "my-own-session",
        },
    )
    rows = _rows(client)
    time.sleep(1.2)
    rows = client.get("/v1/metrics/requests").json()["requests"]
    kinds = sorted(r["affinityKey"] for r in rows)
    assert kinds == ["client", "start"]
    # Never the key itself.
    assert "my-own-session" not in json.dumps(rows)


def test_the_kind_comes_from_the_key_that_was_made() -> None:
    assert inference._affinity_kind("c:resp_1", None) == "chain"
    assert inference._affinity_kind("k:abc", None) == "client"
    assert inference._affinity_kind("k:abc", "abc") == "session"
    assert inference._affinity_kind("f:0123", None) == "start"
    assert inference._affinity_kind(None, None) is None


# --------------------------------------------------------------------- #
# the store: v14 moves to v15, and the rollups are served
# --------------------------------------------------------------------- #


def _old_row(when: datetime, *, tokens: int | None = 40, elapsed: int = 1000) -> RequestRow:
    return RequestRow(
        started_at=when,
        requested_model="qwen",
        served_model="qwen",
        attempts=1,
        tier=1,
        total_ms=elapsed,
        outcome="served",
        completion_tokens=tokens,
        tries=[AttemptRow(driver="qwen-box", runtime="qwen-rt", elapsed_ms=elapsed, served=True)],
    )


def test_hour_buckets_past_retention_come_from_rollups(tmp_path: Path) -> None:
    async def run() -> None:
        store = MetricsStore(tmp_path / "m.sqlite3", retention_days=1)
        await store.start()
        try:
            ten_days = (datetime.now(UTC) - timedelta(days=10)).replace(minute=20, second=0)
            store._insert(
                [
                    _old_row(ten_days, elapsed=1000),
                    _old_row(ten_days + timedelta(minutes=10), elapsed=3000),
                    _old_row(datetime.now(UTC), elapsed=500),
                ]
            )
            store.maintain()
            groups = store.summary(
                since=datetime.now(UTC) - timedelta(days=11),
                until=datetime.now(UTC) + timedelta(hours=1),
                bucket="hour",
                group_by="model",
            )
        finally:
            await store.aclose()
        sources = {g["source"] for g in groups}
        assert sources == {"rollup", "raw"}
        old = next(g for g in groups if g["source"] == "rollup")
        assert old["requests"] == 2
        assert old["latencyMeanMs"] == 2000
        assert old["latencyMaxMs"] == 3000
        assert old["tokensPerSecondMean"] == 20.0  # 80 tokens over 4 s
        assert "latencyMs" not in old, "a rollup has no percentiles to give"
        recent = next(g for g in groups if g["source"] == "raw")
        assert recent["latencyMs"]["max"] == 500

    asyncio.run(run())


def test_a_window_past_retention_is_not_truncated_when_rollups_cover_it(
    app: FastAPI, client: TestClient
) -> None:
    store = client.app.state.metrics
    since = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    plain = client.get("/v1/metrics", params={"since": since}).json()
    assert plain["truncated"] is True
    hourly = client.get("/v1/metrics", params={"since": since, "bucket": "hour"}).json()
    assert hourly["truncated"] is False
    assert store.rollup_enabled


def test_a_v14_store_moves_to_v15_in_place_and_keeps_its_history(tmp_path: Path) -> None:
    path = tmp_path / "m.sqlite3"

    async def make() -> None:
        store = MetricsStore(path)
        await store.start()
        store._insert([_old_row(datetime.now(UTC) - timedelta(days=3))])
        store.maintain()
        await store.aclose()

    asyncio.run(make())
    # Take it back to what a v14 build wrote: no affinity_key, a rollup of seven fewer
    # columns, and the version.
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE request DROP COLUMN affinity_key")
    for column in (
        "first_ms_sum",
        "first_ms_samples",
        "prompt_tokens",
        "cached_tokens",
        "queued_ms_sum",
        "refused",
        "token_ms_sum",
    ):
        conn.execute(f"ALTER TABLE rollup DROP COLUMN {column}")
    conn.execute("UPDATE meta SET value = '14' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    async def reopen() -> list[dict[str, Any]]:
        store = MetricsStore(path)
        await store.start()
        try:
            rows, _ = store.requests()
            store._insert([_old_row(datetime.now(UTC))])
            return rows
        finally:
            await store.aclose()

    rows = asyncio.run(reopen())
    assert len(rows) == 1 and rows[0]["affinityKey"] is None
    assert not list(tmp_path.glob("*.bak")), "the store was set aside, not moved forward"
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0] == "15"
    assert "token_ms_sum" in {c[1] for c in conn.execute("PRAGMA table_info(rollup)")}
    conn.close()


# --------------------------------------------------------------------- #
# OTLP
# --------------------------------------------------------------------- #

CANARY = "CANARY-7f3a91-do-not-export"


def _walk(node: Any) -> Iterator[Any]:
    yield node
    if isinstance(node, dict):
        for child in node.values():
            yield from _walk(child)
    elif isinstance(node, list):
        for child in node:
            yield from _walk(child)


def test_otlp_payload_has_no_message_text(client: TestClient) -> None:
    store = client.app.state.metrics
    settings = telemetry.OtlpSettings(endpoint="http://collector:4318")
    client.app.state.telemetry._settings = lambda: settings
    _chat(client, f"please keep this secret {CANARY}")
    _chat(client, f"and this one {CANARY}", stream=True)
    numbers = client.app.state.telemetry
    gauges = telemetry.read_gauges(client.app.state.routing, rows_dropped=0, enabled=True)
    metrics = json.dumps(numbers.metrics_payload(gauges))
    spans = numbers.take_spans()
    traces = json.dumps(numbers.trace_payload(spans))
    assert spans, "two requests made no spans"
    for payload in (metrics, traces):
        assert CANARY not in payload
        assert "secret" not in payload
        assert "person" not in payload
    # The one thing a span says of what was asked is the model, as upstream names it.
    root = next(s for s in spans if s.parent_id is None)
    assert root.attributes["gen_ai.request.model"] == "qwen"
    assert root.attributes["gen_ai.usage.input_tokens"] == 10
    assert any(s.name == "eugene.attempt" for s in spans)
    assert store is not None
    # A key's name is for the Prometheus page, never for a collector.
    assert "eugene_client_key_info" not in metrics


def test_a_span_continues_the_callers_trace_and_respects_its_sampling() -> None:
    numbers = telemetry.Telemetry(settings=lambda: telemetry.OtlpSettings(endpoint="http://c:4318"))
    trace = "0af7651916cd43dd8448eb211c80319c"

    def row(parent: str | None) -> RequestRow:
        return RequestRow(
            started_at=datetime.now(UTC),
            requested_model="m",
            total_ms=10,
            attempts=1,
            outcome="served",
            tries=[AttemptRow(driver="d", elapsed_ms=5, served=True)],
            trace_parent=parent,
        )

    numbers.observe(row(f"00-{trace}-b7ad6b7169203331-01"))
    numbers.observe(row(f"00-{trace}-b7ad6b7169203331-00"))  # the caller did not sample it
    numbers.observe(row("garbage"))
    spans = numbers.take_spans()
    roots = [s for s in spans if s.name.endswith(" m")]
    assert len(roots) == 2  # the sampled one and the one with no parent
    continued = next(s for s in roots if s.trace_id == trace)
    assert continued.parent_id == "b7ad6b7169203331"


def test_no_spans_are_kept_while_traces_are_off() -> None:
    numbers = telemetry.Telemetry()  # no endpoint
    numbers.observe(
        RequestRow(
            started_at=datetime.now(UTC),
            requested_model="m",
            total_ms=1,
            attempts=1,
            outcome="served",
        )
    )
    assert numbers.spans_waiting == 0


def test_a_full_span_buffer_drops_and_counts() -> None:
    numbers = telemetry.Telemetry(settings=lambda: telemetry.OtlpSettings(endpoint="http://c:4318"))
    for _ in range(telemetry.MAX_SPANS + 10):
        numbers.observe(
            RequestRow(
                started_at=datetime.now(UTC),
                requested_model="m",
                total_ms=1,
                attempts=0,
                outcome="served",
            )
        )
    assert numbers.spans_waiting == telemetry.MAX_SPANS
    page = numbers.render({})
    assert value(page, "eugene_telemetry_spans_dropped_total") == 10


def _exporter(
    numbers: telemetry.Telemetry, handler: Any, settings: telemetry.OtlpSettings
) -> telemetry.OtlpExporter:
    return telemetry.OtlpExporter(
        numbers,
        settings=lambda: settings,
        gauges=lambda: {},
        client_for=lambda url: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_the_exporter_posts_metrics_and_traces_with_the_credential() -> None:
    seen: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.headers), json.loads(request.content)))
        return httpx.Response(200, json={})

    settings = telemetry.OtlpSettings(
        endpoint="http://collector:4318/", authorization="Bearer secret-token"
    )
    numbers = telemetry.Telemetry(settings=lambda: settings)
    numbers.observe(
        RequestRow(
            started_at=datetime.now(UTC),
            requested_model="m",
            total_ms=10,
            attempts=1,
            outcome="served",
            tries=[AttemptRow(driver="d", elapsed_ms=5, served=True)],
        )
    )

    async def run() -> telemetry.OtlpExporter:
        exporter = _exporter(numbers, handler, settings)
        await exporter.push_once()
        await exporter.aclose()
        return exporter

    exporter = asyncio.run(run())
    assert [path for path, _, _ in seen] == ["/v1/metrics", "/v1/traces"]
    assert all(h["authorization"] == "Bearer secret-token" for _, h, _ in seen)
    names = {m["name"] for m in seen[0][2]["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]}
    assert "eugene_requests_total" in names
    assert exporter.snapshot()["state"] == "ok"
    assert exporter.snapshot()["endpoint"] == "http://collector:4318"
    assert exporter.snapshot()["lastSuccessAt"] is not None


def test_a_collector_that_is_down_makes_the_export_failing_and_says_why() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    settings = telemetry.OtlpSettings(
        endpoint="http://user:hunter2@collector:4318", authorization="Bearer secret-token"
    )
    numbers = telemetry.Telemetry(settings=lambda: settings)
    numbers.observe(
        RequestRow(
            started_at=datetime.now(UTC),
            requested_model="m",
            total_ms=10,
            attempts=1,
            outcome="served",
            tries=[AttemptRow(driver="d", elapsed_ms=5, served=True)],
        )
    )

    async def run() -> dict[str, Any]:
        exporter = _exporter(numbers, handler, settings)
        await exporter.push_once()
        await exporter.aclose()
        return exporter.snapshot()

    snapshot = asyncio.run(run())
    assert snapshot["state"] == "failing"
    assert "connection refused" in snapshot["lastError"]
    assert "collector:4318" in snapshot["lastError"]
    # Nothing secret is in what the console is shown.
    shown = json.dumps(snapshot, default=str)
    assert "hunter2" not in shown and "secret-token" not in shown


def test_a_failing_collector_does_not_stop_requests_being_served(
    app: FastAPI, client: TestClient
) -> None:
    app.state.config_store._values["telemetryOtlpEndpoint"] = "http://127.0.0.1:9"
    assert _chat(client).status_code == 200
    assert _chat(client, stream=True)
    export = client.get("/v1/metrics/export").json()
    assert export["otlp"]["endpoint"] == "http://127.0.0.1:9"
    assert export["otlp"]["state"] in ("ok", "failing")


def test_an_endpoint_with_no_value_is_off() -> None:
    exporter = _exporter(
        telemetry.Telemetry(), lambda r: httpx.Response(200), telemetry.OtlpSettings()
    )
    asyncio.run(exporter.push_once())
    assert exporter.snapshot() == {
        "state": "off",
        "endpoint": None,
        "lastSuccessAt": None,
        "lastError": None,
    }


def test_the_config_says_what_it_sends_and_keeps_the_credential_secret(client: TestClient) -> None:
    schema = client.get("/v1/config/schema").json()
    keys = {f["key"]: f for f in schema["fields"]}
    for name in (
        "telemetryOtlpEndpoint",
        "telemetryOtlpAuthorization",
        "telemetryOtlpSignals",
        "telemetryTraceSampleRatio",
        "telemetryOtlpIntervalSeconds",
    ):
        assert keys[name]["category"] == "telemetry"
        assert keys[name].get("requiresRestart") in (None, False)
    assert keys["telemetryOtlpAuthorization"]["valueType"] == "secret"
    assert "telemetryExportPersonLabels" not in keys, "no person is a label (MET-PEOPLE)"
    patched = client.patch(
        "/v1/config",
        json={
            "telemetryOtlpEndpoint": "http://collector:4318",
            "telemetryOtlpAuthorization": "Bearer abc",
        },
    )
    assert patched.status_code == 200, patched.text
    document = client.get("/v1/config").json()
    assert document["telemetryOtlpEndpoint"] == "http://collector:4318"
    assert document["telemetryOtlpAuthorization"] == "<redacted>"
    bad = client.patch("/v1/config", json={"telemetryOtlpEndpoint": "ftp://nope"})
    assert bad.json()["rejected"], bad.text


# --------------------------------------------------------------------- #
# review: a credential is never said, a bad address is named, gauges say their machine
# --------------------------------------------------------------------- #

SECRET = "r12-SECRET-TOKEN-123"


class _Collector:
    """A real HTTP listener, so the real HTTP library judges the header."""

    def __init__(self) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        owner = self
        self.seen: list[dict[str, str]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("content-length") or 0))
                owner.seen.append({k.lower(): v for k, v in self.headers.items()})
                self.send_response(200)
                self.send_header("content-length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args: Any) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _real_exporter(settings: telemetry.OtlpSettings) -> telemetry.OtlpExporter:
    from eugene_plexus_gateway._http import client_for

    numbers = telemetry.Telemetry(settings=lambda: settings)
    return telemetry.OtlpExporter(
        numbers,
        settings=lambda: settings,
        gauges=lambda: {},
        client_for=client_for,
        timeout_seconds=3.0,
    )


def test_describe_says_only_the_type_of_an_error_it_does_not_know() -> None:
    cause = telemetry.describe(
        httpx.LocalProtocolError(f"Illegal header value b'Bearer {SECRET}'"), "http://c:4318"
    )
    assert cause == "LocalProtocolError"
    assert SECRET not in telemetry.describe(
        httpx.UnsupportedProtocol(f"Request URL has an unsupported protocol {SECRET}"),
        "http://c:4318",
    )


def test_a_credential_a_header_cannot_hold_is_never_said(
    caplog: pytest.LogCaptureFixture,
) -> None:
    collector = _Collector()
    try:
        # A NUL inside: h11 refuses the header and quotes it in its own error.
        settings = telemetry.OtlpSettings(
            endpoint=collector.url, authorization=f"Bearer {SECRET}\x00more"
        )

        async def run() -> dict[str, Any]:
            exporter = _real_exporter(settings)
            with caplog.at_level("DEBUG"):
                await exporter.push_once()
                await exporter.push_once()
            snapshot = exporter.snapshot()
            await exporter.aclose()
            return snapshot

        snapshot = asyncio.run(run())
    finally:
        collector.stop()
    assert snapshot["state"] == "failing"
    assert "header cannot hold" in snapshot["lastError"]
    assert collector.seen == [], "a refused credential was still sent"
    said = json.dumps(snapshot, default=str) + "\n".join(r.getMessage() for r in caplog.records)
    assert SECRET not in said
    # Said once, not at every interval.
    warned = [r for r in caplog.records if "OTLP export" in r.getMessage()]
    assert len(warned) == 1


def test_a_credential_with_a_trailing_space_is_trimmed_and_sent() -> None:
    collector = _Collector()
    try:
        settings = telemetry.OtlpSettings(
            endpoint=collector.url, authorization=f"Bearer {SECRET} \n"
        )

        async def run() -> dict[str, Any]:
            exporter = _real_exporter(settings)
            await exporter.push_once()
            snapshot = exporter.snapshot()
            await exporter.aclose()
            return snapshot

        snapshot = asyncio.run(run())
    finally:
        collector.stop()
    assert snapshot["state"] == "ok", snapshot
    assert collector.seen and all(h["authorization"] == f"Bearer {SECRET}" for h in collector.seen)


def test_the_gateway_trims_the_credential_it_reads_from_the_config(client: TestClient) -> None:
    patched = client.patch("/v1/config", json={"telemetryOtlpAuthorization": f"Bearer {SECRET} \n"})
    assert patched.status_code == 200, patched.text
    read = client.app.state.otlp_exporter._settings()
    assert read.authorization == f"Bearer {SECRET}"


def test_an_address_that_is_not_a_url_is_named_and_not_taken_for_working() -> None:
    settings = telemetry.OtlpSettings(endpoint="http://[bad")

    async def run() -> dict[str, Any]:
        exporter = _real_exporter(settings)
        await exporter.push_once()
        snapshot = exporter.snapshot()
        await exporter.aclose()
        return snapshot

    snapshot = asyncio.run(run())
    assert snapshot["state"] == "failing"
    assert snapshot["lastError"] == "the collector address is not a valid URL"


def test_the_exporter_sends_again_when_any_export_setting_changes() -> None:
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.headers))
        return httpx.Response(200, json={})

    box = {"settings": telemetry.OtlpSettings(endpoint="http://c:4318", authorization="Bearer one")}
    numbers = telemetry.Telemetry(settings=lambda: box["settings"])
    exporter = telemetry.OtlpExporter(
        numbers,
        settings=lambda: box["settings"],
        gauges=lambda: {},
        client_for=lambda url: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    async def run() -> None:
        exporter.start()
        for _ in range(60):
            if seen:
                break
            await asyncio.sleep(0.1)
        first = len(seen)
        assert first, "nothing was sent"
        # The interval is a minute: only the change can make this send now.
        box["settings"] = telemetry.OtlpSettings(
            endpoint="http://c:4318", authorization="Bearer two"
        )
        for _ in range(60):
            if len(seen) > first:
                break
            await asyncio.sleep(0.1)
        await exporter.aclose()

    asyncio.run(run())
    assert seen[0]["authorization"] == "Bearer one"
    assert seen[-1]["authorization"] == "Bearer two"


def test_a_budget_gauge_and_the_queue_say_which_machine_they_are_about() -> None:
    from eugene_plexus_gateway import fair_queue

    table = make_routing_table(
        runtimes=[
            runtime_facts("rt", node="a", context_pool=1000),
            runtime_facts("rt", node="b", context_pool=2000),
        ]
    )
    fair_queue.of(table).ledger.take(("a", "rt"), 50)
    fair_queue.of(table).snapshot = lambda: [  # type: ignore[method-assign]
        {
            "model": "m",
            "runtime": "rt",
            "node": "b",
            "waiting": {"interactive": 2, "background": 0, "sheddable": 0},
            "running": 0,
            "slots": None,
        }
    ]
    page = telemetry.Telemetry().render(telemetry.read_gauges(table, rows_dropped=0, enabled=True))
    assert value(page, "eugene_budget_pool_tokens", runtime="rt", node="a") == 1000
    assert value(page, "eugene_budget_pool_tokens", runtime="rt", node="b") == 2000
    assert value(page, "eugene_budget_reserved_tokens", runtime="rt", node="a") == 50
    assert value(page, "eugene_queue_waiting", runtime="rt", node="b", band="interactive") == 2
