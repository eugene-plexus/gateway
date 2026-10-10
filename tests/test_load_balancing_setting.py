"""Affinity is its own setting; `loadBalancing` says where a new conversation goes (CB1).

Until 2026-10-02 `loadBalancing: conversation` (PC4) both kept conversations
on their replicas and placed new ones by least busy, and the other values
turned affinity off. Every install's file holds `conversation`, because the
store writes each default out on first start. So `conversation` is read as
the default placement with affinity, which is what it did, and a marker beside
the file says the value there was written as a default rather than chosen, so
the next default (CB2's) reaches it too.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eugene_plexus_gateway import config
from eugene_plexus_gateway._generated.models import ConfigUpdateRequest
from eugene_plexus_gateway.config import ConfigStore, as_schema

MARKER = ".load-balancing-default"


def store_at(path: Path) -> ConfigStore:
    store = ConfigStore(path)
    store.load()
    return store


def _field(key: str):  # type: ignore[no-untyped-def]
    return next(f for f in as_schema().fields if f.key == key)


def test_affinity_is_on_by_default_and_can_be_turned_off():
    field = _field("conversationAffinity")
    assert field.default is True and field.valueType.value == "boolean"
    assert "benchmark" in field.description


def test_conversation_is_no_longer_a_placement():
    assert "conversation" not in _field("loadBalancing").enumValues


def test_spread_is_the_default_placement():
    """CB2: measured +20 points of prompt reuse on eight replicas, noise on
    three, and nothing measurable lost anywhere."""
    field = _field("loadBalancing")
    assert field.default == "spread" and field.enumValues[0] == "spread"


def test_a_new_install_marks_its_value_as_the_default(tmp_path: Path) -> None:
    store_at(tmp_path / "gateway.yaml")
    assert (tmp_path / MARKER).exists()


def test_a_value_written_as_the_default_follows_the_next_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "gateway.yaml"
    store_at(path)  # writes today's default and marks it
    later = config._FIELDS_BY_KEY["loadBalancing"].model_copy(update={"default": "round_robin"})
    monkeypatch.setitem(config._FIELDS_BY_KEY, "loadBalancing", later)
    assert store_at(path).get("loadBalancing") == "round_robin"


def test_a_chosen_value_stays_whatever_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "gateway.yaml"
    store = store_at(path)
    store.apply_patch(ConfigUpdateRequest.model_validate({"loadBalancing": "least_busy"}))
    assert not (tmp_path / MARKER).exists()
    later = config._FIELDS_BY_KEY["loadBalancing"].model_copy(update={"default": "round_robin"})
    monkeypatch.setitem(config._FIELDS_BY_KEY, "loadBalancing", later)
    assert store_at(path).get("loadBalancing") == "least_busy"


def test_an_older_file_holding_a_choice_keeps_it(tmp_path: Path) -> None:
    path = tmp_path / "gateway.yaml"
    path.write_text(yaml.safe_dump({"loadBalancing": "round_robin"}))
    assert store_at(path).get("loadBalancing") == "round_robin"
    assert not (tmp_path / MARKER).exists()


def test_setting_it_back_to_unset_is_the_default_again(tmp_path: Path) -> None:
    path = tmp_path / "gateway.yaml"
    store = store_at(path)
    store.apply_patch(ConfigUpdateRequest.model_validate({"loadBalancing": "round_robin"}))
    store.apply_patch(ConfigUpdateRequest.model_validate({"loadBalancing": None}))
    assert store.get("loadBalancing") == _field("loadBalancing").default
    assert (tmp_path / MARKER).exists()


def test_a_patch_of_conversation_is_refused(tmp_path: Path) -> None:
    store = store_at(tmp_path / "gateway.yaml")
    result = store.apply_patch(
        ConfigUpdateRequest.model_validate({"loadBalancing": "conversation"})
    )
    assert [r.key for r in result.rejected] == ["loadBalancing"]


def test_the_setting_reaches_the_routing_table(settings) -> None:  # type: ignore[no-untyped-def]
    """Read live, like loadBalancing: a PATCH takes effect on the next request."""
    from fastapi.testclient import TestClient

    from eugene_plexus_gateway.app import create_app

    app = create_app(settings=settings)
    with TestClient(app) as client:
        table = app.state.routing
        assert table.affinity_on() is True
        r = client.patch("/v1/config", json={"conversationAffinity": False})
        assert r.status_code == 200, r.text
        assert table.affinity_on() is False
        client.patch("/v1/config", json={"loadBalancing": "round_robin"})
        assert table.placement() == "round_robin"
