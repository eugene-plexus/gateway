"""The prompt cache, made visible (PC5): cached tokens and the affinity outcome."""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import Usage
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table, runtime_facts


def _drain(client: TestClient, requests: int) -> dict:  # type: ignore[type-arg]
    """Until the rows are written AND the summary's window, which ends at
    the current second, has passed the second they were stamped in."""
    for _ in range(100):
        body = client.get("/v1/metrics/requests").json()
        groups = client.get("/v1/metrics", params={"groupBy": "total"}).json()["groups"]
        if len(body["requests"]) >= requests and groups and groups[0]["requests"] >= requests:
            return body
        time.sleep(0.05)
    raise AssertionError("the metrics writer never caught up")


@pytest.fixture
def two_backends(
    settings: Settings,
) -> Iterator[tuple[TestClient, FakeDriverClient, FakeDriverClient]]:
    """Two replicas under `conversation`, one reporting a cached count and
    one reporting none, so the share can be checked to count only the
    requests that said."""
    app = create_app(settings=settings)
    reports = FakeDriverClient(name="a-driver", base_url="http://a", model_id="qwen", runtime="a")
    reports.usage = Usage(
        promptTokens=1000, completionTokens=4, totalTokens=1004, cachedPromptTokens=900
    )
    silent = FakeDriverClient(name="b-driver", base_url="http://b", model_id="qwen", runtime="b")
    silent.usage = Usage(promptTokens=1000, completionTokens=4, totalTokens=1004)
    app.state.routing = make_routing_table(
        reports, silent, runtimes=[runtime_facts("a"), runtime_facts("b")], strategy="conversation"
    )
    with TestClient(app) as client:
        yield client, reports, silent


def _ask(client: TestClient, first: str, *more: str) -> None:
    messages = [{"role": "user", "content": first}]
    for text in more:
        messages += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": text}]
    r = client.post("/v1/chat/completions", json={"model": "qwen", "messages": messages})
    assert r.status_code == 200, r.text


def test_cached_tokens_and_the_affinity_outcome_are_recorded(two_backends) -> None:  # type: ignore[no-untyped-def]
    client, reports, silent = two_backends
    _ask(client, "one conversation")
    _ask(client, "one conversation", "its second turn")
    body = _drain(client, 2)
    rows = list(reversed(body["requests"]))  # served newest first
    home = reports if reports.calls else silent
    # The reporting replica reuses 900 of every 1,000-token prompt, so its
    # second turn reused less than the first turn's whole prompt: evicted
    # (CB5). The silent one cannot be judged, and its hit stays a hit.
    assert [r["affinity"] for r in rows] == ["new", "evicted" if home is reports else "hit"]
    expected = 900 if home is reports else None
    assert [r["cachedTokens"] for r in rows] == [expected, expected]
    served = rows[0]["tries"][0]
    assert served.get("cachedTokens") == expected


def test_the_share_counts_only_requests_that_reported(two_backends) -> None:  # type: ignore[no-untyped-def]
    """A backend that says nothing must not read as one that reused nothing."""
    client, reports, silent = two_backends
    for i in range(6):
        _ask(client, f"conversation {i}")
    _drain(client, 6)
    total = client.get("/v1/metrics", params={"groupBy": "total"}).json()["groups"][0]
    reported = len(reports.calls)
    assert reported and len(silent.calls), "both replicas should have been used"
    assert total["promptCache"] == {
        "requests": reported,
        "promptTokens": 1000 * reported,
        "cachedTokens": 900 * reported,
    }
    assert total["affinity"] == {"hit": 0, "new": 6, "moved": 0, "evicted": 0}


def test_nothing_reported_is_null_not_zero(settings: Settings) -> None:
    app = create_app(settings=settings)
    fake = FakeDriverClient(name="d", model_id="qwen", runtime="r")
    fake.usage = Usage(promptTokens=10, completionTokens=1, totalTokens=11)
    app.state.routing = make_routing_table(fake, runtimes=[runtime_facts("r")])
    with TestClient(app) as client:
        _ask(client, "hi")
        _drain(client, 1)
        group = client.get("/v1/metrics").json()["groups"][0]
    assert group["promptCache"] is None
    # One backend: there was no choice for affinity to make.
    assert group["affinity"] is None
