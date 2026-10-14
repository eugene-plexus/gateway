"""The model list cannot call outside or unknown inference local."""

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import Locality
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table


@pytest.mark.parametrize(
    ("locality", "providers", "owner"),
    [
        ("local", ["openai_compat_custom"], "local"),
        ("external", ["openrouter"], "openrouter"),
        ("external", ["openrouter", "openai"], "mixed"),
        ("unknown", [None], "unknown"),
    ],
)
def test_owned_by_agrees_with_where_the_model_runs(
    settings: Settings, locality: str, providers: list[str | None], owner: str
) -> None:
    drivers = []
    for i, provider in enumerate(providers):
        driver = FakeDriverClient(name=f"driver-{i}", model_id="model")
        info = driver.describe().model_copy(
            update={"locality": Locality(locality), "provider": provider}
        )
        driver.describe = lambda info=info: info  # type: ignore[method-assign]
        drivers.append(driver)
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*drivers)
    with TestClient(app) as client:
        model = client.get("/v1/models").json()["data"][0]
    assert model["owned_by"] == owner
    assert model["x_eugene_plexus"]["locality"] == locality
