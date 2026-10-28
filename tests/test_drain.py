"""Row 13: the gateway's own drain, and its reports on a runtime's drain.

A restart used to end the process 500 ms after it was asked and cut every
stream; now the answers in flight finish (up to `shutdownDrainSeconds`), new
ones are refused `shutting_down`, and only then does the server exit.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from eugene_plexus_gateway import drain as drain_module
from eugene_plexus_gateway import fair_queue
from eugene_plexus_gateway.__main__ import _Server
from eugene_plexus_gateway._generated.driver_models import GenerateRequest
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import StreamEvent
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, FakeInstall, make_routing_table, runtime_facts

MODEL = "slow-model"


class SlowStream(FakeDriverClient):
    """Streams eight words over `seconds`, and counts the streams it holds."""

    def __init__(self, seconds: float = 2.0, **kwargs: Any) -> None:
        kwargs.setdefault("name", "slow")
        kwargs.setdefault("model_id", MODEL)
        super().__init__(**kwargs)
        self.seconds = seconds
        self.now = 0

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        self.now += 1
        try:
            async for event in super().stream(request):
                if event.text:
                    await asyncio.sleep(self.seconds / 8)
                yield event
        finally:
            self.now -= 1


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _chat(stream: bool, text: str = "one two three four five six seven eight") -> dict[str, Any]:
    return {
        "model": MODEL,
        "stream": stream,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": text}],
    }


async def _read_stream(client: httpx.AsyncClient, url: str) -> tuple[str, float]:
    async with client.stream("POST", url + "/v1/chat/completions", json=_chat(True)) as response:
        assert response.status_code == 200, await response.aread()
        body = "".join([chunk async for chunk in response.aiter_text()])
    return body, time.perf_counter()


async def test_restart_finishes_open_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three 2-second streams are open when a restart arrives. All three finish
    whole, a new request gets 503 `shutting_down`, and then the server exits
    (its `run` returns, which is the process exiting). Before row 13 the
    process was ended 500 ms after the restart, cutting all three."""
    exits: list[float] = []

    def hard_exit(code: int) -> None:  # the old restart's `os._exit`
        exits.append(time.perf_counter())
        server.should_exit = True

    monkeypatch.setattr(os, "_exit", hard_exit)
    settings = Settings(config_file=tmp_path / "gateway.yaml", metrics_file=tmp_path / "m.sqlite3")
    app = create_app(settings=settings)
    driver = SlowStream(seconds=2.0)
    driver.responses = ["one two three four five six seven eight"] * 3
    app.state.routing = make_routing_table(driver)
    port = _free_port()
    server = _Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on"), app
    )
    original = getattr(app.state, "request_exit", None)
    if callable(original):

        def recorded() -> None:
            exits.append(time.perf_counter())
            original()

        app.state.request_exit = recorded
    serving = asyncio.create_task(server.serve())
    url = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
            for _ in range(200):
                if server.started:
                    break
                await asyncio.sleep(0.05)
            streams = [asyncio.create_task(_read_stream(client, url)) for _ in range(3)]
            for _ in range(100):
                if driver.now == 3:
                    break
                await asyncio.sleep(0.02)
            assert driver.now == 3, "three streams are open before the restart"

            restart = await client.post(url + "/v1/admin/restart")
            assert restart.status_code == 202, restart.text

            refused = await client.post(url + "/v1/chat/completions", json=_chat(False))
            assert refused.status_code == 503, refused.text
            assert refused.json()["error"]["code"] == "shutting_down"
            assert refused.headers["Retry-After"] == "5"

            done = await asyncio.gather(*streams)
        for body, _ended in done:
            assert "data: [DONE]" in body, body[-300:]
            assert '"error"' not in body, body[-300:]
            words = "".join(
                (json.loads(line[6:])["choices"] or [{}])[0].get("delta", {}).get("content") or ""
                for line in body.splitlines()
                if line.startswith("data: {")
            )
            assert words == "one two three four five six seven eight"
        await asyncio.wait_for(serving, 10)
        assert exits, "the server was asked to exit"
        assert exits[0] >= max(ended for _body, ended in done) - 0.05, (
            "the exit came only after every stream had finished"
        )
    finally:
        server.should_exit = True
        await asyncio.gather(serving, return_exceptions=True)


async def test_stream_still_open_at_the_limit_is_cut(tmp_path: Path) -> None:
    """Past `shutdownDrainSeconds` a stream gets its door's error frame, its
    metrics row says `cut`, and the process exits anyway."""
    (tmp_path / "gateway.yaml").write_text("shutdownDrainSeconds: 0\n", encoding="utf-8")
    settings = Settings(config_file=tmp_path / "gateway.yaml", metrics_file=tmp_path / "m.sqlite3")
    app = create_app(settings=settings)
    driver = SlowStream(seconds=8.0)
    app.state.routing = make_routing_table(driver)
    exits: list[float] = []
    app.state.request_exit = lambda: exits.append(time.perf_counter())
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://gw") as client,
    ):
        reading = asyncio.create_task(
            client.post("/v1/chat/completions", json=_chat(True), timeout=30)
        )
        for _ in range(100):
            if driver.now == 1:
                break
            await asyncio.sleep(0.02)
        began = time.perf_counter()
        assert (await client.post("/v1/admin/drain")).status_code == 202
        answer = await reading
        assert time.perf_counter() - began < 5
        assert answer.status_code == 200
        assert "Cut: Eugene's gateway restarted after waiting 0 s" in answer.text
        for _ in range(50):
            if exits:
                break
            await asyncio.sleep(0.05)
        view = (await client.get("/v1/admin/drain")).json()
        assert view["state"] == "exiting" and view["cut"] == 1, view
        assert exits, "it exits after the cut"
        rows: list[dict[str, Any]] = []
        for _ in range(100):
            rows = (await client.get("/v1/metrics/requests")).json()["requests"]
            if rows:
                break
            await asyncio.sleep(0.05)
        assert [r["outcome"] for r in rows] == ["cut"], rows


def test_only_local_agent_token_may_drain_gateway(settings: Settings, install: FakeInstall) -> None:
    """`/v1/admin/drain` takes an operator session, or this machine's own
    agent's token for itself. Not a gateway's or a driver's token, not
    another machine's agent, not a client key."""
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(FakeDriverClient(name="d"))
    app.state.auth_state = install.auth_state()
    app.state.request_exit = lambda: None

    def bearer(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    with TestClient(app) as c:
        for refused in (
            install.service("gateway"),
            install.service("inference-driver"),
            install.foreign_service("agent"),
            install.client_key(),
        ):
            assert c.get("/v1/admin/drain", headers=bearer(refused)).status_code == 401
            assert c.post("/v1/admin/drain", headers=bearer(refused)).status_code == 401
        assert c.get("/v1/admin/drain").status_code == 401
        assert c.get("/v1/admin/drain", headers=bearer(install.session())).json()["state"] == (
            "serving"
        )
        started = c.post("/v1/admin/drain", headers=bearer(install.service("agent")))
        assert started.status_code == 202, started.text
        assert started.json()["state"] in ("draining", "exiting")
        # Restart stays an operator's: the agent drains, it does not restart.
        assert c.post(
            "/v1/admin/restart", headers=bearer(install.service("agent"))
        ).status_code == (401)


def test_config_and_health_answer_while_draining(settings: Settings) -> None:
    app = create_app(settings=settings)
    driver = SlowStream(seconds=4.0)
    app.state.routing = make_routing_table(driver)
    app.state.request_exit = lambda: None
    with TestClient(app) as c:
        c.post("/v1/admin/drain")
        assert c.get("/healthz").status_code == 200
        assert c.get("/v1/config").status_code == 200
        assert c.get("/v1/models").status_code == 200
        refused = c.post("/v1/chat/completions", json=_chat(False))
        assert refused.status_code == 503
        assert refused.json()["error"]["code"] == "shutting_down"


# --- a runtime's drain, as this gateway sees it ------------------------------------


def _drain(state: str, then: str = "restart") -> dict[str, Any]:
    return {"state": state, "then": then, "reason": "update", "when": "covered", "running": 0}


def test_a_draining_runtime_is_not_routed_and_says_why() -> None:
    from dataclasses import replace

    a = replace(
        runtime_facts("r", alias=MODEL, node="a"), status="draining", drain=_drain("draining")
    )
    driver = FakeDriverClient(name="d", model_id=MODEL, runtime="r", node="a")
    table = make_routing_table(driver, runtimes=[a])
    backend = table.resolve(MODEL).backends()[0]
    assert not backend.eligible
    assert backend.ineligible_reason == "runtime 'r' is draining for an update"
    # Its only runtime restarting counts as coming up: the request waits in line.
    assert [r.name for r in table.resolve(MODEL).waking()] == ["r"]
    stopping = replace(a, drain=_drain("draining", then="stop"))
    table = make_routing_table(driver, runtimes=[stopping])
    assert table.resolve(MODEL).waking() == []


def test_covered_only_when_the_other_copies_have_room() -> None:
    """DI-C3: one copy leaves only while the others carry the current demand,
    measured from the queue and the slots in use."""
    from dataclasses import replace

    leaving = replace(
        runtime_facts("r", alias=MODEL, node="a", parallel_slots=2), drain=_drain("waiting_covered")
    )
    other = runtime_facts("r", alias=MODEL, node="b", parallel_slots=2)
    drivers = [FakeDriverClient(name="d", model_id=MODEL, runtime="r", node=node) for node in "ab"]
    table = make_routing_table(*drivers, runtimes=[leaving, other])
    report = drain_module.report_for(table, leaving)
    assert (report.covered, report.other_copies, report.running) == (True, 1, 0)

    for _ in range(3):
        table.on_attempt_start("d", node="b")
    assert not drain_module.report_for(table, leaving).covered, "3 running, 2 slots"

    alone = make_routing_table(runtimes=[leaving])
    report = drain_module.report_for(alone, leaving)
    assert (report.other_copies, report.covered) == (0, False), "one copy: the agent goes at once"

    loading = replace(other, status="loading")
    report = drain_module.report_for(make_routing_table(runtimes=[leaving, loading]), leaving)
    assert (report.other_copies, report.covered) == (1, False), "a copy coming back is waited for"


def test_two_copies_waiting_for_cover_leave_one_at_a_time() -> None:
    from dataclasses import replace

    first = replace(runtime_facts("r", alias=MODEL, node="a"), drain=_drain("waiting_covered"))
    second = replace(runtime_facts("r", alias=MODEL, node="b"), drain=_drain("waiting_covered"))
    table = make_routing_table(runtimes=[first, second])
    assert drain_module.report_for(table, first).covered
    assert not drain_module.report_for(table, second).covered


async def test_reporter_reports_open_drains_only() -> None:
    from dataclasses import replace

    open_ = replace(
        runtime_facts("r", alias=MODEL, node="a"), status="draining", drain=_drain("draining")
    )
    closed = runtime_facts("s", alias="other", node="a")
    table = make_routing_table(runtimes=[open_, closed])
    sent: list[tuple[str, str, dict[str, Any], str | None]] = []
    hooked: list[str] = []

    async def report(agent: str, name: str, body: dict[str, Any], node: str | None) -> bool:
        sent.append((agent, name, body, node))
        return True

    reporter = drain_module.DrainReporter(table, report=report)

    async def moved(facts: Any) -> None:
        hooked.append(facts.name)

    reporter.on_draining = moved
    assert await reporter.report_once() == [("a", "r")]
    assert [(name, body["running"], node) for _agent, name, body, node in sent] == [("r", 0, "a")]
    assert hooked == ["r"], "row 44's hook runs before a 0 is reported"


async def test_refresh_every_two_seconds_while_a_drain_is_open() -> None:
    from dataclasses import replace

    from eugene_plexus_gateway.routing import RoutingTable

    table = RoutingTable(agent_url="http://agent", refresh_seconds=15.0)
    assert table.refresh_interval() == 15.0
    open_ = replace(runtime_facts("r", alias=MODEL), drain=_drain("waiting_covered"))
    from .conftest import install_snapshot

    install_snapshot(table, runtimes=[open_])
    assert table.refresh_interval() == 2.0


def test_queue_close_is_what_refuses_waiters() -> None:
    """Row 11's `FairQueue.close()` is the drain's first act on the line."""
    table = make_routing_table(FakeDriverClient(name="d"))
    app = create_app(settings=Settings(config_file=Path("unused.yaml")))
    app.state.routing = table
    drain = drain_module.GatewayDrain(app, seconds=lambda: 0.0, exit_process=lambda: None)

    async def go() -> None:
        drain.start()
        await asyncio.sleep(0.05)

    asyncio.run(go())
    assert fair_queue.of(table).closed
