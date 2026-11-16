"""Row 40: how this gateway sees a model's price, for the root that counts it."""

from fastapi.testclient import TestClient

from eugene_plexus_gateway import prices
from tests.conftest import FakeDriverClient, make_routing_table


def _table(*drivers: FakeDriverClient):
    return make_routing_table(*drivers)


def test_an_unknown_price_is_unknown_not_zero():
    cloud = FakeDriverClient(name="cloud", model_id="m", locality="external")
    table = _table(cloud)
    assert prices.hint_for(table, "m") == {"locality": "external"}
    assert prices.hint_for(table, "nowhere") == {"locality": "unknown"}
    assert prices.hint_for(None, "m") == {"locality": "unknown"}
    (model,) = table.as_model_list()
    assert model.x_eugene_plexus.provider_price is None, "no price listed, never free"


def test_a_local_model_is_said_local_with_no_price():
    local = FakeDriverClient(name="engine", model_id="m", locality="local")
    assert prices.hint_for(_table(local), "m") == {"locality": "local"}


def test_two_backends_list_the_higher_price_and_none_when_one_has_none():
    cheap = FakeDriverClient(name="a", model_id="m", locality="external", price=(1, 9))
    dear = FakeDriverClient(name="b", model_id="m", locality="external", price=(5, 2))
    hint = prices.hint_for(_table(cheap, dear), "m")
    assert hint["providerPrice"] == {"inputPerMillionMicros": 5, "outputPerMillionMicros": 9}
    (model,) = _table(cheap, dear).as_model_list()
    listed = model.x_eugene_plexus.provider_price
    assert (listed.input_per_million_micros, listed.output_per_million_micros) == (5, 9)
    assert listed.source == "openrouter" and str(listed.as_of) == "2026-11-01"
    unpriced = FakeDriverClient(name="c", model_id="m", locality="external")
    assert "providerPrice" not in prices.hint_for(_table(cheap, unpriced), "m")


def test_the_model_list_carries_the_provider_price(settings, install):
    from eugene_plexus_gateway.app import create_app

    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    app.state.routing = _table(
        FakeDriverClient(name="cloud", model_id="m", locality="external", price=(3, 15))
    )
    operator = {"Authorization": "Bearer " + install.session(sub="operator")}
    with TestClient(app) as c:
        listed = c.get("/v1/models", headers=operator).json()["data"][0]
    assert listed["x_eugene_plexus"]["provider_price"]["input_per_million_micros"] == 3
