"""Row 39: operations rights for people who are not the owner, on the gateway.

`specs/docs/design/rows/row-39.md`. A person with any operations power
reads the roles and the metrics summary; nothing else here takes their
token. Routing, backends, client keys and the config stay the owner's.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from eugene_plexus_gateway import operations, tokens
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, FakeInstall, make_routing_table

EXPECTED_READS = {("GET", "/v1/roles"), ("GET", "/v1/metrics")}


def _operations(install: FakeInstall, *powers: str, machine: str | None = None) -> str:
    token, _ = install.root.mint(
        typ=tokens.TYP_OPERATIONS,
        sub="p-dana",
        aud=[tokens.node_recipient(machine or install.name)],
        ttl_seconds=tokens.OPERATIONS_TTL_SECONDS,
        extra={"ops": list(powers), "name": "Dana", "sid": "admin-1", "administration_version": 1},
    )
    return token


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client(
    settings: Settings, install: FakeInstall, fake_driver: FakeDriverClient
) -> Iterator[TestClient]:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake_driver)
    app.state.auth_state = install.auth_state()
    with TestClient(app) as c:
        yield c


def _every_route(app: FastAPI) -> list[tuple[str, str]]:
    found: set[tuple[str, str]] = set()

    def walk(routes: list[Any], prefix: str) -> None:
        for route in routes:
            included = getattr(route, "original_router", None)
            if included is not None:
                walk(included.routes, prefix + (route.include_context.prefix or ""))
                continue
            if isinstance(route, APIRoute):
                for method in route.methods or ():
                    found.add((method, prefix + route.path))

    walk(app.router.routes, "")
    return sorted(found)


def test_the_table_is_the_designs() -> None:
    assert operations.OPERATIONS_READS == EXPECTED_READS


def test_any_power_reads_roles_and_metrics(client: TestClient, install: FakeInstall) -> None:
    for power in tokens.POWERS:
        dana = _bearer(_operations(install, power))
        assert client.get("/v1/roles", headers=dana).status_code == 200
        assert client.get("/v1/metrics", headers=dana).status_code == 200
    # Another machine's token, and the request log, are refused.
    far = _bearer(_operations(install, *tokens.POWERS, machine="far"))
    assert client.get("/v1/roles", headers=far).status_code == 401
    log = client.get("/v1/metrics/requests", headers=_bearer(_operations(install, *tokens.POWERS)))
    assert log.status_code == 403
    assert "Only the owner" in log.json()["detail"]["detail"]


def test_config_patch_stays_owner_only(client: TestClient, install: FakeInstall) -> None:
    everything = _bearer(_operations(install, *tokens.POWERS))
    for answer in (
        client.patch("/v1/config", json={}, headers=everything),
        client.get("/v1/config", headers=everything),
    ):
        assert answer.status_code == 403
        assert "Only the owner" in answer.json()["detail"]["detail"]


def test_nothing_else_takes_an_operations_token(client: TestClient, install: FakeInstall) -> None:
    """Every route that refuses a request with no token refuses one holding
    every power too, unless the table names it."""
    everything = _bearer(_operations(install, *tokens.POWERS))
    tried: list[tuple[str, str]] = []
    opened: list[tuple[str, str, int]] = []
    for method, path in _every_route(client.app):  # type: ignore[arg-type]
        if (method, path) in EXPECTED_READS or method == "HEAD":
            continue
        concrete = re.sub(r"\{[^}]+\}", "x-none", path)
        body = {} if method in ("POST", "PUT", "PATCH") else None
        if client.request(method, concrete, json=body).status_code not in (401, 403):
            continue
        tried.append((method, path))
        answer = client.request(method, concrete, headers=everything, json=body)
        if answer.status_code not in (401, 403):
            opened.append((method, path, answer.status_code))
    assert ("POST", "/v1/chat/completions") in tried
    assert len(tried) > 20, tried
    assert opened == []
