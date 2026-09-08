"""Tests for admin endpoints and the config protocol."""

from __future__ import annotations

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.conftest import FakeHemisphereClient, make_message_event


def test_admin_drivers_reports_list(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    response = client.get("/v1/admin/drivers")
    assert response.status_code == 200
    body = response.json()
    drivers = body["drivers"]
    assert [d["name"] for d in drivers] == ["left", "right"]
    assert drivers[0]["reachable"] is True
    assert drivers[0]["backend"] == "claude_code_cli"
    assert drivers[1]["reachable"] is True
    assert drivers[1]["backend"] == "codex_cli"


def test_admin_drivers_reports_individual_unreachability(
    client: TestClient,
    left_fake: FakeHemisphereClient,
) -> None:
    left_fake.info_error = httpx.ConnectError("connection refused")
    response = client.get("/v1/admin/drivers")
    assert response.status_code == 200
    body = response.json()
    drivers = {d["name"]: d for d in body["drivers"]}
    assert drivers["left"]["reachable"] is False
    assert "connection refused" in drivers["left"]["error"]
    assert drivers["right"]["reachable"] is True


def test_admin_drivers_503_when_all_down(
    client: TestClient,
    left_fake: FakeHemisphereClient,
    right_fake: FakeHemisphereClient,
) -> None:
    left_fake.info_error = httpx.ConnectError("dead")
    right_fake.info_error = httpx.ConnectError("dead")
    response = client.get("/v1/admin/drivers")
    assert response.status_code == 503


def test_admin_drivers_probe_reports_unreachable_url(client: TestClient) -> None:
    """Probe endpoint hits a real URL — pointing at a port nothing
    is listening on returns ok=200 with `reachable: false`, not a 5xx.
    The UI's Test button needs structured failure to render the red
    banner with the error reason."""
    response = client.post(
        "/v1/admin/drivers/probe",
        json={"name": "candidate", "url": "http://127.0.0.1:1"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["name"] == "candidate"
    assert body["reachable"] is False
    assert body["error"]


def test_admin_drivers_probe_rejects_invalid_url(client: TestClient) -> None:
    response = client.post(
        "/v1/admin/drivers/probe",
        json={"name": "bad", "url": "not-a-url"},
    )
    # DriverProbeRequest.url is `format: uri` -> 422 from Pydantic before reaching the route.
    assert response.status_code == 422


def test_admin_nt_state_returns_neutral_baseline(client: TestClient) -> None:
    """v0.2 NT shape: per-NT {level, baseline, decay} triple plus
    `lastUpdated`. Six NTs — cortisol replaces v0.1's glutamate.

    Levels and baselines start at 0.5 (neutral). Decay rates are
    per-NT (dopamine fast, serotonin slow, etc.) so we assert
    structure + initial level/baseline but not the specific decay
    constant — that's a tuning knob.
    """
    response = client.get("/v1/admin/nt-state")
    assert response.status_code == 200
    body = response.json()
    assert "lastUpdated" in body
    for key in ("serotonin", "dopamine", "norepinephrine", "acetylcholine", "gaba", "cortisol"):
        triple = body[key]
        assert triple["level"] == 0.5
        assert triple["baseline"] == 0.5
        assert triple["decay"] > 0.0  # every NT has a non-zero decay back to baseline
    # v0.1's glutamate is gone in v0.2.
    assert "glutamate" not in body


def test_admin_restart_returns_202_and_schedules_exit(
    client: TestClient, monkeypatch: object
) -> None:
    """Verify /v1/admin/restart returns the right shape and schedules a
    delayed exit. We intercept the asyncio loop's call_later so the test
    process doesn't actually exit."""
    captured: dict[str, object] = {}

    class _FakeLoop:
        def call_later(self, delay: float, callback: object) -> None:
            captured["delay"] = delay
            captured["callback"] = callback

    import asyncio as _asyncio

    real_get_event_loop = _asyncio.get_event_loop
    monkeypatch.setattr(_asyncio, "get_event_loop", lambda: _FakeLoop())  # type: ignore[attr-defined]

    try:
        response = client.post("/v1/admin/restart")
    finally:
        monkeypatch.setattr(_asyncio, "get_event_loop", real_get_event_loop)  # type: ignore[attr-defined]

    assert response.status_code == 202
    body = response.json()
    assert body["scheduled"] is True
    assert body["delayMs"] >= 0
    assert "message" in body
    assert captured["delay"] == body["delayMs"] / 1000.0
    assert callable(captured["callback"])


def test_config_schema_lists_orchestrator_fields(client: TestClient) -> None:
    response = client.get("/v1/config/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "gateway"
    keys = {f["key"] for f in body["fields"]}
    expected = {
        "drivers",
        "memoryUrl",
        "logLevel",
        "defaultMaxPasses",
        "agreementThreshold",
        "defaultSystemPrompt",
        "defaultTemperature",
        "defaultMaxTokens",
        "requestTimeoutSeconds",
    }
    assert expected.issubset(keys)
    # The legacy left/right URL fields are gone — replaced by `drivers`.
    assert "leftDriverUrl" not in keys
    assert "rightDriverUrl" not in keys
    # `port` is no longer a config field — owned by the watchdog topology.
    assert "port" not in keys

    drivers_field = next(f for f in body["fields"] if f["key"] == "drivers")
    assert drivers_field["valueType"] == "driver_list"
    assert drivers_field["requiresRestart"] is True


def test_config_get_then_patch_round_trip(client: TestClient) -> None:
    initial = client.get("/v1/config").json()
    # v0.2.x bumped the default 0.5 → 0.75 when the scorer switched from
    # Jaccard word-overlap to embedding cosine similarity. The new scale
    # makes 0.5 mean "same topic" instead of "substantively agreed."
    assert initial["agreementThreshold"] == 0.75
    # `drivers` ships with the canonical bicameral pair on local ports.
    assert [d["name"] for d in initial["drivers"]] == ["left", "right"]

    patch = client.patch(
        "/v1/config",
        json={"agreementThreshold": 0.6, "defaultMaxPasses": 5, "bogus": True},
    )
    assert patch.status_code == 200
    body = patch.json()
    assert "agreementThreshold" in body["applied"]
    assert "defaultMaxPasses" in body["applied"]
    rejected_keys = {r["key"] for r in body["rejected"]}
    assert "bogus" in rejected_keys
    # Both applied fields are read live at request time — no restart needed.
    assert body["requiresRestart"] is False

    follow = client.get("/v1/config").json()
    assert follow["agreementThreshold"] == 0.6
    assert follow["defaultMaxPasses"] == 5


def test_config_patch_drivers_validates_shape(client: TestClient) -> None:
    """Canonical v0.2.1-item-2 shape: each slot carries a `backends`
    priority list of watchdog-topology entry names."""
    response = client.patch(
        "/v1/config",
        json={
            "drivers": [
                {"name": "left", "backends": ["claude-local", "claude-openrouter"]},
                {"name": "right", "backends": ["gpt-local"]},
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "drivers" in body["applied"]
    assert body["requiresRestart"] is True

    follow = client.get("/v1/config").json()
    assert [d["name"] for d in follow["drivers"]] == ["left", "right"]
    # The priority list (including the fallback name) round-trips intact.
    assert follow["drivers"][0]["backends"] == ["claude-local", "claude-openrouter"]
    assert follow["drivers"][1]["backends"] == ["gpt-local"]


def test_config_patch_drivers_migrates_legacy_urls(client: TestClient) -> None:
    """A PATCH using the v0.2.1-item-1 `urls` shape is accepted and
    renamed to `backends` (values preserved as URL-shaped backends),
    so configs written between item 1 and item 2 keep working."""
    response = client.patch(
        "/v1/config",
        json={
            "drivers": [
                {"name": "left", "urls": ["http://10.0.0.1:8081", "http://10.0.0.9:8081"]},
                {"name": "right", "urls": ["http://10.0.0.2:8081"]},
            ]
        },
    )
    assert response.status_code == 200
    assert "drivers" in response.json()["applied"]

    follow = client.get("/v1/config").json()
    assert follow["drivers"][0]["backends"] == ["http://10.0.0.1:8081", "http://10.0.0.9:8081"]
    assert "urls" not in follow["drivers"][0]
    assert follow["drivers"][1]["backends"] == ["http://10.0.0.2:8081"]


def test_config_patch_drivers_migrates_oldest_url(client: TestClient) -> None:
    """The pre-v0.2.1 single-`url` shape upgrades straight to `backends`."""
    response = client.patch(
        "/v1/config",
        json={"drivers": [{"name": "left", "url": "http://10.0.0.1:8081"}]},
    )
    assert response.status_code == 200
    follow = client.get("/v1/config").json()
    assert follow["drivers"][0]["backends"] == ["http://10.0.0.1:8081"]
    assert "url" not in follow["drivers"][0]


def test_config_patch_drivers_rejects_malformed(client: TestClient) -> None:
    response = client.patch(
        "/v1/config",
        json={
            "drivers": [
                {"name": "ok", "backends": ["left"]},
                {"name": "", "backends": ["right"]},  # empty name
            ]
        },
    )
    assert response.status_code == 200
    rejected = {r["key"]: r["message"] for r in response.json()["rejected"]}
    assert "drivers" in rejected
    assert "name" in rejected["drivers"]


def test_config_patch_drivers_rejects_empty_backends(client: TestClient) -> None:
    """A slot with no backends is rejected — a priority list must have at
    least one entry to be reachable."""
    response = client.patch(
        "/v1/config",
        json={"drivers": [{"name": "left", "backends": []}]},
    )
    assert response.status_code == 200
    rejected = {r["key"]: r["message"] for r in response.json()["rejected"]}
    assert "drivers" in rejected
    assert "backends" in rejected["drivers"]


def test_config_patch_drivers_rejects_duplicate_names(client: TestClient) -> None:
    response = client.patch(
        "/v1/config",
        json={
            "drivers": [
                {"name": "twin", "backends": ["a"]},
                {"name": "twin", "backends": ["b"]},
            ]
        },
    )
    assert response.status_code == 200
    rejected = {r["key"]: r["message"] for r in response.json()["rejected"]}
    assert "duplicate" in rejected["drivers"].lower()


def test_config_schema_voicedriver_is_strict_enum(client: TestClient) -> None:
    """voiceDriver renders as a strict enum of the configured slot names
    (plus a leading '(first driver)' default) — but validation stays
    lenient (see test below)."""
    client.patch(
        "/v1/config",
        json={
            "drivers": [
                {"name": "left", "backends": ["a"]},
                {"name": "right", "backends": ["b"]},
            ]
        },
    )
    schema = client.get("/v1/config/schema").json()
    vd = next(f for f in schema["fields"] if f["key"] == "voiceDriver")
    assert vd["valueType"] == "enum"
    assert vd["enumValues"] == ["", "left", "right"]
    assert vd["enumLabels"][0] == "(first driver)"


def test_config_patch_voicedriver_stays_lenient(client: TestClient) -> None:
    """Despite the strict enum schema, validation accepts an unknown
    voiceDriver name — the chat handler falls back to the first driver,
    so we never hard-reject a name the operator is about to create."""
    response = client.patch("/v1/config", json={"voiceDriver": "not-a-slot"})
    assert response.status_code == 200
    assert "voiceDriver" in response.json()["applied"]
    assert client.get("/v1/config").json()["voiceDriver"] == "not-a-slot"


def test_events_503_when_fewer_than_two_drivers(
    app: FastAPI, left_fake: FakeHemisphereClient
) -> None:
    """v0.2.1 item 2: slot backends resolve against the watchdog topology
    at startup. If the topology was unreachable / missing the named
    drivers, fewer than two slots get built — POST /v1/events must 503
    cleanly rather than enqueue an event the loop can't act on."""
    app.state.drivers = [left_fake]  # only one slot resolved
    with TestClient(app) as c:
        response = c.post(
            "/v1/events",
            json=make_message_event("hi").model_dump(mode="json", exclude_none=True),
        )
    assert response.status_code == 503
    assert "driver" in response.json()["detail"]["detail"].lower()
