"""build_clients resolves driver-slot backends (topology names) to URLs.

v0.2.1 item 2: a slot's `backends` are watchdog-topology inference-driver
entry NAMES; `build_clients` resolves them against a name->url map fetched
once at startup. These tests pin the resolution + degraded-mode rules.
"""

from __future__ import annotations

from typing import Any

from eugene_plexus_gateway.app import (
    _resolve_backend_url,
    build_clients,
    driver_topology,
)
from eugene_plexus_gateway.driver_client import FailoverDriverClient


class _FakeStore:
    def __init__(self, drivers: list[dict[str, Any]]) -> None:
        self._values: dict[str, Any] = {"drivers": drivers, "requestTimeoutSeconds": 30}

    def get(self, key: str) -> Any:
        return self._values.get(key)


class _FakeAuth:
    service_token: str | None = None


def _build(drivers: list[dict[str, Any]], topology: dict[str, str]) -> list[FailoverDriverClient]:
    return build_clients(_FakeStore(drivers), _FakeAuth(), topology)  # type: ignore[arg-type]


def test_resolves_names_to_urls() -> None:
    clients = _build(
        [
            {"name": "left", "backends": ["claude-local"]},
            {"name": "right", "backends": ["gpt-local"]},
        ],
        {"claude-local": "http://h:8081", "gpt-local": "http://h:8082/"},
    )
    assert [c.name for c in clients] == ["left", "right"]
    # base_url is the resolved primary, trailing slash stripped.
    assert [c.base_url for c in clients] == ["http://h:8081", "http://h:8082"]


def test_unknown_backend_name_skips_slot() -> None:
    clients = _build(
        [
            {"name": "left", "backends": ["good"]},
            {"name": "right", "backends": ["nope"]},
        ],
        {"good": "http://h:8081"},
    )
    # The slot whose only backend doesn't resolve is not built.
    assert [c.name for c in clients] == ["left"]


def test_url_shaped_backend_used_directly() -> None:
    """Migration tolerance: a backend that looks like a URL (a value
    carried over from the old `urls` shape) is used directly."""
    clients = _build(
        [{"name": "left", "backends": ["http://10.0.0.1:8081"]}],
        {},  # empty topology
    )
    assert len(clients) == 1
    assert clients[0].base_url == "http://10.0.0.1:8081"


def test_failover_backend_unresolvable_is_dropped_from_slot() -> None:
    """A slot keeps its resolvable backends and drops the rest (the slot
    is still built as long as >=1 resolves)."""
    clients = _build(
        [{"name": "left", "backends": ["primary", "missing"]}],
        {"primary": "http://h:8081"},
    )
    assert len(clients) == 1
    assert clients[0]._candidates and len(clients[0]._candidates) == 1  # missing dropped


def test_empty_topology_builds_nothing() -> None:
    clients = _build(
        [
            {"name": "left", "backends": ["left"]},
            {"name": "right", "backends": ["right"]},
        ],
        {},
    )
    assert clients == []


def test_resolve_backend_url_precedence() -> None:
    topo = {"left": "http://h:8081/"}
    # 1. topology name wins (trailing slash stripped)
    assert _resolve_backend_url("left", topo) == "http://h:8081"
    # 2. URL-shaped fallback
    assert _resolve_backend_url("https://x:1/", topo) == "https://x:1"
    # 3. unresolvable
    assert _resolve_backend_url("unknown", topo) is None


def test_driver_topology_filters_kind() -> None:
    components = [
        {"name": "left", "kind": "inference-driver", "url": "http://h:8081"},
        {"name": "mem", "kind": "memory", "url": "http://h:8083"},
        {"name": "nourl", "kind": "inference-driver"},  # dropped (no url)
    ]
    assert driver_topology(components) == {"left": "http://h:8081"}
