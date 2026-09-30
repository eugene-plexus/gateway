"""Settings never lie (Troy, 2026-09-29: fundamental).

No widget may show a value other than the one in effect. For the gateway's
config trio that means:

- **A null in the file is the default.** `metricsEnabled: null` turned
  metrics off while GET said null and the schema said the default was on.
- **A restart is pending only for this PATCH's keys**, and only while the
  saved value differs from what the process runs on -- which the schema
  reports per field (`pendingRestart`, `inEffect`). It was every restart key
  saved since the process started, so a later save of a live field made the
  UI restart the gateway.
- **An unset value says what it does**: no output cap, the image tool's
  fallback, the control root found through the agent, and an empty origin
  list admitting any website -- which read "Default: none".
- **NaN is not a number here.** It passed the range check, and a timeout of
  NaN ended every request at once.
- **0 days of metrics is 0**, not the week `or 7` made it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from eugene_plexus_gateway._generated.models import ConfigUpdateRequest
from eugene_plexus_gateway.config import ConfigStore, as_schema


def _store(tmp_path: Path, content: dict[str, Any] | None = None) -> ConfigStore:
    path = tmp_path / "gateway.yaml"
    if content is not None:
        path.write_text(yaml.safe_dump(content), encoding="utf-8")
    store = ConfigStore(path)
    store.load()
    return store


def _patch(store: ConfigStore, body: dict[str, Any]):  # type: ignore[no-untyped-def]
    return store.apply_patch(ConfigUpdateRequest.model_validate(body))


def _field(key: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    return next(f for f in as_schema(**kwargs).fields if f.key == key)


def test_a_null_in_the_file_is_the_default(tmp_path: Path) -> None:
    store = _store(tmp_path, {"metricsEnabled": None, "defaultTemperature": None})
    doc = store.as_document().model_dump()
    assert doc["metricsEnabled"] is True and doc["defaultTemperature"] == 0.7


def test_a_restart_is_pending_only_while_the_value_differs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = _patch(store, {"logLevel": "DEBUG"})
    assert first.requiresRestart is True and first.pendingRestart == ["logLevel"]
    assert store.pending_restart() == {"logLevel": "INFO"}
    # A live field saved afterwards does not ask for a restart.
    second = _patch(store, {"maxToolCalls": 7})
    assert second.requiresRestart is False and second.pendingRestart == []
    # Put back to what the process runs on: nothing is pending.
    third = _patch(store, {"logLevel": "INFO"})
    assert third.requiresRestart is False and store.pending_restart() == {}


def test_the_schema_says_which_value_is_in_effect(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _patch(store, {"metricsRetentionDays": 30})
    field = _field("metricsRetentionDays", pending=store.pending_restart())
    assert field.pendingRestart is True and field.inEffect == 7
    assert _field("metricsRetentionDays").pendingRestart in (None, False)


def test_an_unset_value_says_what_it_does() -> None:
    assert "No cap" in (_field("defaultMaxTokens").unsetMeans or "")
    assert "any website" in (_field("corsAllowedOrigins").unsetMeans or "")
    assert "image model" in (_field("imageToolModel").unsetMeans or "")
    derived = _field("controlUrl", derived_control_url="http://192.168.16.252:8283")
    assert derived.unsetResolvesTo == "http://192.168.16.252:8283"
    assert "192.168.16.252:8283" in (derived.unsetMeans or "")
    assert _field("controlUrl").unsetResolvesTo is None


def test_nan_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = _patch(store, {"requestTimeoutSeconds": float("nan")})
    assert result.applied == [] and "finite" in result.rejected[0].message
    assert store.get("requestTimeoutSeconds") == 600.0
