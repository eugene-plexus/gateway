"""`x_eugene_plexus.locality` on `GET /v1/models` (2026-10-08).

A client says *runs on <account>* before it sends (Workbench's media
screens, call M6). It is the serving drivers' own classification, joined
the honest way: local only when every backend is, external when any is.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import DriverInfo, Locality
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table


class Placed(FakeDriverClient):
    """A chat driver that says where its engine runs."""

    def __init__(self, *, locality: Locality | None, **kw: Any) -> None:
        super().__init__(**kw)
        self.locality = locality

    def describe(self) -> DriverInfo:
        info = super().describe()
        if self.locality is not None:
            info.locality = self.locality
            info.localOnlyEnforced = self.locality == Locality.local
        return info


def listed(settings: Settings, *drivers: FakeDriverClient, **table: Any) -> dict[str, Any]:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*drivers, **table)
    with TestClient(app) as client:
        return {m["id"]: m["x_eugene_plexus"] for m in client.get("/v1/models").json()["data"]}


def test_each_model_says_where_it_runs(settings: Settings) -> None:
    models = listed(
        settings,
        Placed(name="a", model_id="qwen", locality=Locality.local),
        Placed(name="b", model_id="gpt", locality=Locality.external),
        Placed(name="c", model_id="custom", locality=None),
    )
    assert models["qwen"]["locality"] == "local"
    assert models["gpt"]["locality"] == "external"
    # A driver that never said is unknown, never assumed local.
    assert models["custom"]["locality"] == "unknown"


def test_a_slot_is_local_only_when_every_backend_is(settings: Settings) -> None:
    drivers = (
        Placed(name="a", model_id="qwen", locality=Locality.local),
        Placed(name="b", model_id="gpt", locality=Locality.external),
        Placed(name="c", model_id="custom", locality=None),
        Placed(name="d", model_id="qwen-2", locality=Locality.local),
    )
    slots = [
        {"model": "mixed", "targets": ["qwen", "gpt"]},
        {"model": "home", "targets": ["qwen", "qwen-2"]},
        {"model": "unsure", "targets": ["qwen", "custom"]},
    ]
    models = listed(settings, *drivers, slots=slots)
    assert models["mixed"]["locality"] == "external"
    assert models["home"]["locality"] == "local"
    assert models["unsure"]["locality"] == "unknown"
