"""The install sets no output cap by default (2026-09-28).

2048 cut a reasoning model off while it was still thinking, so it thought
and never answered (a tester, on Qwen). And because the store writes every
default into its file on first start, every older install holds
`defaultMaxTokens: 2048` as though someone chose it: the old value is
cleared once, and a marker keeps a 2048 the operator sets afterwards.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from eugene_plexus_gateway._generated.models import ConfigUpdateRequest
from eugene_plexus_gateway.config import ConfigStore, as_schema

MARKER = ".default-max-tokens-cleared"


def store_at(path: Path) -> ConfigStore:
    store = ConfigStore(path)
    store.load()
    return store


def test_a_new_install_sets_no_cap(tmp_path: Path) -> None:
    store = store_at(tmp_path / "gateway.yaml")
    assert store.get("defaultMaxTokens") is None
    field = next(f for f in as_schema().fields if f.key == "defaultMaxTokens")
    assert field.default is None
    assert "defaultMaxTokens" not in yaml.safe_load((tmp_path / "gateway.yaml").read_text())
    assert (tmp_path / MARKER).exists()


def test_the_old_default_an_older_install_wrote_is_cleared_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "gateway.yaml"
    path.write_text(yaml.safe_dump({"defaultMaxTokens": 2048, "defaultTemperature": 0.7}))
    with caplog.at_level(logging.WARNING):
        store = store_at(path)
    assert store.get("defaultMaxTokens") is None
    assert "defaultMaxTokens" not in yaml.safe_load(path.read_text())
    assert (tmp_path / MARKER).exists()
    assert "old shipped default" in caplog.text


def test_a_2048_set_after_the_clearing_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "gateway.yaml"
    path.write_text(yaml.safe_dump({"defaultMaxTokens": 2048}))
    store = store_at(path)
    store.apply_patch(ConfigUpdateRequest.model_validate({"defaultMaxTokens": 2048}))
    assert store_at(path).get("defaultMaxTokens") == 2048


def test_a_cap_the_operator_chose_is_never_touched(tmp_path: Path) -> None:
    path = tmp_path / "gateway.yaml"
    path.write_text(yaml.safe_dump({"defaultMaxTokens": 4096}))
    assert store_at(path).get("defaultMaxTokens") == 4096
    assert (tmp_path / MARKER).exists()


def test_clearing_the_field_returns_to_no_cap(tmp_path: Path) -> None:
    path = tmp_path / "gateway.yaml"
    store = store_at(path)
    store.apply_patch(ConfigUpdateRequest.model_validate({"defaultMaxTokens": 1000}))
    assert store.get("defaultMaxTokens") == 1000
    store.apply_patch(ConfigUpdateRequest.model_validate({"defaultMaxTokens": None}))
    assert store.get("defaultMaxTokens") is None
