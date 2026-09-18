"""One client, one context, one clock — the gateway's half.

The routing table reads five to seven URLs per refresh, every 15 s, and
again inside whatever request called `refresh_if_stale`. Each of those
reads used to build its own `httpx.AsyncClient()`, and constructing one
parses certifi's PEM bundle: **104-136 ms of synchronous CPU** measured
in this venv on the Python both installers provision. That is ~0.6 s of
blocked event loop per refresh, landing on the loop that is streaming
every in-flight completion.

And every `*_ms` this component reports was measured with
`time.monotonic()`, which on Windows/CPython 3.12 is `GetTickCount64` —
**20 distinct values in 300 ms**, a 15.6 ms grid. The developer box runs
3.14, where CPython fixed it, so the defect is invisible exactly where
it is written and present on every shipped install.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from eugene_plexus_gateway import _http
from eugene_plexus_gateway.client_keys import ClientKeyGuard
from eugene_plexus_gateway.driver_client import HttpDriverClient
from eugene_plexus_gateway.lifecycle import AgentLifecycleClient
from eugene_plexus_gateway.routing import RoutingTable


def _count_constructions(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]
    real = httpx.AsyncClient.__init__

    def counting(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        calls[0] += 1
        real(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", counting)
    return calls


# --------------------------------------------------------------------------- #
# one client
# --------------------------------------------------------------------------- #


async def test_a_routing_refresh_builds_no_clients_for_its_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**The refresh is the hot path.** Seven topology reads used to be
    seven clients — ~0.7 s of synchronous CPU on the loop serving every
    stream. The table owns one, built in `__init__`.
    """
    table = RoutingTable(agent_url="http://127.0.0.1:8079", service_token=None)
    built = _count_constructions(monkeypatch)
    for i in range(7):
        await table._get_json(f"http://127.0.0.1:8079/v1/does-not-exist-{i}")
    assert built[0] == 0, f"{built[0]} clients built during seven topology reads"
    await table.aclose()


async def test_the_topology_client_is_closed_with_the_table() -> None:
    """A client that outlives its table is a leaked pool; `aclose()`
    already closed the per-driver clients and not this one."""
    table = RoutingTable(agent_url="http://127.0.0.1:8079", service_token=None)
    client = table._json_client
    await table.aclose()
    assert client.is_closed


async def test_a_failed_read_does_not_poison_the_shared_client() -> None:
    """Every topology read is a connection refusal on this box. The
    client has to stay usable, or one unreachable node would take the
    whole refresh down with it for the life of the process."""
    table = RoutingTable(agent_url="http://127.0.0.1:8079", service_token=None)
    body, error = await table._get_json_or_error("http://127.0.0.1:9/v1/nodes")
    assert body is None and error
    body2, error2 = await table._get_json_or_error("http://127.0.0.1:9/v1/components")
    assert body2 is None and error2
    assert not table._json_client.is_closed
    await table.aclose()


# --------------------------------------------------------------------------- #
# no ambient proxy between this install's own processes
# --------------------------------------------------------------------------- #


def test_the_topology_client_declines_an_ambient_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """**The Windows symptom this defends is silent.** The logon task
    inherits the user environment, so `HTTP_PROXY` reaches the gateway;
    a corporate proxy cannot dial 127.0.0.1, every topology read fails,
    and the install reports nothing routable while every process is up.
    """
    monkeypatch.setenv("HTTP_PROXY", "http://corp.proxy:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://corp.proxy:3128")
    table = RoutingTable(agent_url="http://127.0.0.1:8079", service_token=None)
    assert table._json_client._mounts == {}


def test_the_driver_client_declines_an_ambient_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://corp.proxy:3128")
    client = HttpDriverClient(name="d", base_url="http://127.0.0.1:8081", service_token=None)
    assert client._client._mounts == {}


def test_the_lifecycle_client_declines_an_ambient_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://corp.proxy:3128")
    assert AgentLifecycleClient(service_token=None)._client._mounts == {}


def test_the_client_key_guard_declines_an_ambient_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://corp.proxy:3128")
    guard = ClientKeyGuard(agent_url="http://127.0.0.1:8079", service_token=None)
    assert guard._ensure_client()._mounts == {}


# --------------------------------------------------------------------------- #
# one clock
# --------------------------------------------------------------------------- #


def test_no_duration_in_this_component_is_measured_with_monotonic() -> None:
    """Every `latencyMs`, `waitedMs`, `routingMs` and `elapsedMs` this
    component reports comes off one of these clocks. On the shipped
    Python, `monotonic()` cannot resolve anything under 15.6 ms — which
    is how a 104 ms per-request cost was obtained by subtracting two
    spans and came out as an unexplained 116 ms.
    """
    root = Path(__file__).resolve().parents[1] / "src" / "eugene_plexus_gateway"
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "_generated" not in path.parts and "time.monotonic()" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"these measure with monotonic(): {offenders}"


def test_the_clocks_differ_the_way_this_rule_claims() -> None:
    """The rule is only worth enforcing if the grid is real here.
    Asserts the *ratio*, so it holds on Linux (where both are fine) and
    on Windows (where one is not)."""
    mono = time.get_clock_info("monotonic").resolution
    perf = time.get_clock_info("perf_counter").resolution
    assert perf <= mono
    assert perf <= 1e-6


# --------------------------------------------------------------------------- #
# the context itself
# --------------------------------------------------------------------------- #


def test_the_ssl_context_is_built_once_per_process() -> None:
    assert _http.ssl_context() is _http.ssl_context()


async def test_five_clients_cost_nothing_once_the_context_exists() -> None:
    _http.ssl_context()
    started = time.perf_counter()
    clients = [_http.internal_client() for _ in range(5)]
    elapsed_ms = (time.perf_counter() - started) * 1000
    await asyncio.gather(*(c.aclose() for c in clients))
    assert elapsed_ms < 50, (
        f"five clients cost {elapsed_ms:.0f} ms; a certifi parse is ~104 ms each"
    )
