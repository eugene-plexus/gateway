"""Admin (drivers, probe, restart) and the config protocol."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table


def _app_with(settings: Settings, table: object) -> FastAPI:
    app = create_app(settings=settings)
    app.state.routing = table
    return app


# --------------------------------------------------------------------------- #
# /v1/admin/drivers
# --------------------------------------------------------------------------- #


def test_admin_drivers_reports_what_the_routing_table_saw(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    response = client.get("/v1/admin/drivers")
    assert response.status_code == 200
    drivers = response.json()["drivers"]
    assert len(drivers) == 1
    assert drivers[0]["name"] == fake_driver.name
    assert drivers[0]["reachable"] is True
    assert drivers[0]["modelId"] == fake_driver.model_id


def test_admin_drivers_carries_the_runtime_a_driver_follows(settings: Settings) -> None:
    """Straight off the driver's /v1/info. A reachable driver serving
    nothing, next to a `ready` runtime routed to by nobody, is the visible
    shape of a mis-wired install — and this is the field that shows it.
    Absent for a backend that is not a runtime this install supervises."""
    following = FakeDriverClient(name="local", runtime="qwen3-8b")
    cloud = FakeDriverClient(name="cloud", provider="openai")
    app = _app_with(settings, make_routing_table(following, cloud))
    with TestClient(app) as c:
        drivers = {d["name"]: d for d in c.get("/v1/admin/drivers").json()["drivers"]}

    assert drivers["local"]["runtime"] == "qwen3-8b"
    assert drivers["cloud"].get("runtime") is None


def test_admin_drivers_lists_unreachable_ones_too(
    settings: Settings, fake_driver: FakeDriverClient
) -> None:
    """A driver that stopped answering has to stay visible: hiding it
    would make the dashboard say everything is fine."""
    app = _app_with(
        settings,
        make_routing_table(fake_driver, unreachable={"dead-box": "connection refused"}),
    )
    with TestClient(app) as c:
        drivers = c.get("/v1/admin/drivers").json()["drivers"]

    by_name = {d["name"]: d for d in drivers}
    assert by_name[fake_driver.name]["reachable"] is True
    assert by_name["dead-box"]["reachable"] is False
    assert "connection refused" in by_name["dead-box"]["error"]


def test_admin_drivers_503_when_none_reachable(settings: Settings) -> None:
    app = _app_with(settings, make_routing_table(unreachable={"dead-box": "connection refused"}))
    with TestClient(app) as c:
        response = c.get("/v1/admin/drivers")
    assert response.status_code == 503
    assert "No driver in the topology is reachable" in response.json()["detail"]["detail"]


def test_admin_drivers_503_when_topology_has_none(settings: Settings) -> None:
    """Distinct message from "all unreachable": an empty topology is an
    operator setup step, not a failure to diagnose."""
    app = _app_with(settings, make_routing_table())
    with TestClient(app) as c:
        response = c.get("/v1/admin/drivers")
    assert response.status_code == 503
    assert "no inference-driver entries" in response.json()["detail"]["detail"].lower()


# --------------------------------------------------------------------------- #
# /v1/admin/drivers/probe
# --------------------------------------------------------------------------- #


def test_probe_reports_an_unreachable_url_without_erroring(client: TestClient) -> None:
    """The UI's Test button wants a yes/no, not an exception — so an
    unreachable URL is a 200 with `reachable: false`."""
    response = client.post(
        "/v1/admin/drivers/probe",
        json={"url": "http://127.0.0.1:1/", "name": "candidate"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["reachable"] is False
    assert body["error"]


def test_probe_rejects_an_invalid_url(client: TestClient) -> None:
    response = client.post("/v1/admin/drivers/probe", json={"url": "not-a-url"})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# /v1/admin/restart
# --------------------------------------------------------------------------- #


def test_restart_returns_202_and_schedules_an_exit(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[float] = []

    class _Loop:
        def call_later(self, delay: float, _fn: object) -> None:
            calls.append(delay)

    monkeypatch.setattr(asyncio, "get_event_loop", lambda: _Loop())
    response = client.post("/v1/admin/restart")

    assert response.status_code == 202
    body = response.json()
    assert body["scheduled"] is True
    assert body["delayMs"] > 0
    # Scheduled, not executed — the response has to flush first.
    assert calls == [body["delayMs"] / 1000.0]


# --------------------------------------------------------------------------- #
# Config protocol
# --------------------------------------------------------------------------- #


def test_config_schema_lists_the_gateway_fields(client: TestClient) -> None:
    response = client.get("/v1/config/schema")
    assert response.status_code == 200
    schema = response.json()
    assert schema["component"] == "gateway"

    keys = {f["key"] for f in schema["fields"]}
    assert keys == {
        "defaultTemperature",
        "defaultMaxTokens",
        "requestTimeoutSeconds",
        "routingRefreshSeconds",
        "logLevel",
    }
    # No backend URLs and no model list: routing is derived, so there is
    # nothing here to get out of step with reality.
    assert not any("url" in k.lower() for k in keys)
    assert "drivers" not in keys

    for field in schema["fields"]:
        assert field["label"]
        assert field["description"], f"{field['key']} needs help text"
        assert field["category"] in schema["categories"]


def test_config_get_then_patch_round_trip(client: TestClient) -> None:
    before = client.get("/v1/config").json()
    assert before["logLevel"] == "INFO"

    patched = client.patch("/v1/config", json={"logLevel": "DEBUG"})
    assert patched.status_code == 200
    result = patched.json()
    assert result["applied"] == ["logLevel"]
    # logLevel is read once at startup, so the UI has to be told.
    assert result["requiresRestart"] is True

    assert client.get("/v1/config").json()["logLevel"] == "DEBUG"


def test_config_patch_rejects_an_unknown_field(client: TestClient) -> None:
    response = client.patch("/v1/config", json={"nonsense": 1})
    assert response.status_code == 200
    rejected = response.json()["rejected"]
    assert [r["key"] for r in rejected] == ["nonsense"]


def test_config_patch_validates_ranges(client: TestClient) -> None:
    response = client.patch("/v1/config", json={"defaultTemperature": 5.0})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == []
    assert body["rejected"][0]["key"] == "defaultTemperature"
    assert "<= 2" in body["rejected"][0]["message"]


def test_config_test_reports_no_drivers_when_the_topology_is_empty(
    client: TestClient,
) -> None:
    """The Test button reads the topology fresh, and the fake agent
    URL isn't reachable — which is exactly the "nothing to route to"
    answer an operator needs before they've wired anything up."""
    response = client.post("/v1/config/test")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["component"] == "gateway"
    assert "nothing to route to" in body["error"]


def test_config_test_accepts_overrides_without_persisting_them(
    client: TestClient,
) -> None:
    response = client.post("/v1/config/test", json={"overrides": {"requestTimeoutSeconds": 7}})
    assert response.status_code == 200
    # The override must not have been written.
    assert client.get("/v1/config").json()["requestTimeoutSeconds"] == 180


def test_healthz_is_ok_and_unauthenticated(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["component"] == "gateway"
    assert body["safeMode"] is False
