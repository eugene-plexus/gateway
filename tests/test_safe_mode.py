"""Tests for the watchdog safe-mode contract on the gateway.

Per specs/openapi/gateway.yaml: when started with
`EUGENE_PLEXUS_GATEWAY_SAFE_MODE=1` the gateway must

  - skip loading its persisted config file (defaults only)
  - still expose /v1/config endpoints (operator can repair via UI)
  - report /healthz as `degraded` with `safeMode: true`
  - return 503 from /v1/events (no drivers configured)
  - allow PATCH /v1/config to write to the on-disk file as normal
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings


@pytest.fixture
def safe_mode_settings(tmp_path: Path) -> Settings:
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "drivers": [
                    {"name": "left", "url": "http://127.0.0.1:8081"},
                    {"name": "right", "url": "http://127.0.0.1:8082"},
                ],
                "memoryUrl": "http://memory.persisted.example:8083",
                "logLevel": "DEBUG",
            }
        ),
        encoding="utf-8",
    )
    return Settings(config_file=config, safe_mode=True)


@pytest.fixture
def safe_mode_app(safe_mode_settings: Settings) -> FastAPI:
    return create_app(settings=safe_mode_settings)


@pytest.fixture
def safe_mode_client(safe_mode_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(safe_mode_app) as c:
        yield c


def test_healthz_reports_safe_mode_and_degraded(safe_mode_client: TestClient) -> None:
    response = safe_mode_client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["safeMode"] is True


def test_config_get_returns_defaults_not_disk_values(safe_mode_client: TestClient) -> None:
    """Disk had a custom memoryUrl + DEBUG log level; safe mode must ignore
    the file and serve the built-in defaults instead. Drivers in safe
    mode fall back to the canonical left/right localhost pair (the field
    default), not to the persisted disk values."""
    response = safe_mode_client.get("/v1/config")
    assert response.status_code == 200
    body = response.json()
    assert body["logLevel"] == "INFO"
    assert "memory.persisted.example" not in str(body.get("memoryUrl", "")), (
        "safe mode loaded memoryUrl from disk"
    )


def test_events_return_503_in_safe_mode(safe_mode_client: TestClient) -> None:
    from tests.conftest import make_message_event

    response = safe_mode_client.post(
        "/v1/events",
        json=make_message_event("hi").model_dump(mode="json", exclude_none=True),
    )
    assert response.status_code == 503


def test_patch_config_writes_to_disk_in_safe_mode(
    safe_mode_client: TestClient, safe_mode_settings: Settings
) -> None:
    """Operator's repair must persist so the next clean boot picks it up."""
    response = safe_mode_client.patch("/v1/config", json={"logLevel": "WARNING"})
    assert response.status_code == 200
    body = response.json()
    assert "logLevel" in body["applied"]

    on_disk = yaml.safe_load(safe_mode_settings.config_file.read_text(encoding="utf-8"))
    assert on_disk["logLevel"] == "WARNING"
