"""Bearer auth on the gateway, against this machine's trust bundle.

The gateway is verify-only: it holds no key (per-node token keys,
2026-09-25). These tests stand in for the agent and the control root with
`FakeInstall` -- a real bundle on disk, signed by a real authority key,
listing the root's token key, this machine's (`node:gw`) and another
machine's (`node:far`) -- and assert the dependencies accept and refuse
the right shapes.

Auth posture is selected by whether `app.state.auth_state` is
pre-populated before the lifespan runs:

  * default fixtures (`client`) leave it unset -> lifespan reads env
    vars (empty in tests) -> `auth_disabled=True`, the dev path.
  * `authed_app` injects the fake install's `AuthState`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway import tokens
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.auth_state import load_auth_state
from eugene_plexus_gateway.settings import Settings
from tests.conftest import FakeDriverClient, FakeInstall, make_routing_table

CHAT = {"messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def authed_app(settings: Settings, install: FakeInstall, fake_driver: FakeDriverClient) -> FastAPI:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake_driver)
    # The lifespan leaves a pre-populated auth_state alone.
    app.state.auth_state = install.auth_state()
    return app


@pytest.fixture
def authed_client(authed_app: FastAPI) -> Iterator[TestClient]:
    with TestClient(authed_app) as c:
        yield c


@pytest.fixture
def operator_token(install: FakeInstall) -> str:
    return install.session()


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _chat(client: TestClient, driver: FakeDriverClient, token: str) -> int:
    driver.responses = ["hello"]
    return client.post(
        "/v1/chat/completions", json={"model": driver.model_id, **CHAT}, headers=_bearer(token)
    ).status_code


# --------------------------------------------------------------------------- #
# Auth-disabled path (default `client` fixture)
# --------------------------------------------------------------------------- #


def test_auth_disabled_lets_everything_through(client: TestClient) -> None:
    """With no trust bundle wired in, every route answers without a
    bearer. The dev / standalone posture; the agent always supplies one."""
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/admin/drivers").status_code == 200
    assert client.get("/v1/config").status_code == 200


def test_healthz_is_always_open(authed_client: TestClient) -> None:
    assert authed_client.get("/healthz").status_code == 200


# --------------------------------------------------------------------------- #
# Missing / malformed / foreign tokens
# --------------------------------------------------------------------------- #


def test_missing_bearer_rejects_with_401(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/config")
    assert response.status_code == 401
    assert response.json()["detail"]["component"] == "gateway"


def test_a_session_signed_by_a_key_the_bundle_does_not_list_is_refused(
    authed_client: TestClient, install: FakeInstall
) -> None:
    stranger = tokens.Signer(key=tokens.generate_private_key(), issuer="control")
    token, _ = stranger.mint(
        typ=tokens.TYP_SESSION, sub="operator", aud=[install.recipient], ttl_seconds=60
    )
    assert authed_client.get("/v1/config", headers=_bearer(token)).status_code == 401


def test_garbage_bearer_rejects(authed_client: TestClient) -> None:
    response = authed_client.get("/v1/config", headers=_bearer("not.a.real.jwt"))
    assert response.status_code == 401


def test_expired_token_rejects(authed_client: TestClient, install: FakeInstall) -> None:
    # Past the 300 s clock-skew leeway, not merely past exp.
    expired = install.session(ttl=60, now=int(time.time()) - 1000)
    assert authed_client.get("/v1/config", headers=_bearer(expired)).status_code == 401


def test_a_session_for_another_machine_is_refused(
    authed_client: TestClient, install: FakeInstall, fake_driver: FakeDriverClient
) -> None:
    """A session is addressed to the machines it may be used on. One the
    operator holds for `far` opens nothing here."""
    elsewhere = install.session(aud=["node:far"])
    assert authed_client.get("/v1/config", headers=_bearer(elsewhere)).status_code == 401
    assert _chat(authed_client, fake_driver, elsewhere) == 401


# --------------------------------------------------------------------------- #
# The operator's session -- operator routes and the front door
# --------------------------------------------------------------------------- #


def test_operator_token_accepted_on_config(authed_client: TestClient, operator_token: str) -> None:
    assert authed_client.get("/v1/config", headers=_bearer(operator_token)).status_code == 200


def test_operator_token_accepted_on_admin(authed_client: TestClient, operator_token: str) -> None:
    response = authed_client.get("/v1/admin/drivers", headers=_bearer(operator_token))
    assert response.status_code == 200


def test_operator_token_accepted_on_chat(
    authed_client: TestClient, operator_token: str, fake_driver: FakeDriverClient
) -> None:
    assert _chat(authed_client, fake_driver, operator_token) == 200


def test_a_session_exchanged_for_this_machine_is_accepted(
    authed_client: TestClient, install: FakeInstall
) -> None:
    """What another console's agent presents after exchanging its
    operator's session at the root: five minutes, this machine alone."""
    exchanged = install.root.mint(
        typ=tokens.TYP_SESSION,
        sub="operator",
        aud=[install.recipient],
        ttl_seconds=300,
        extra={"act": {"sub": "node:far"}, "sid": "s-1"},
    )[0]
    assert authed_client.get("/v1/config", headers=_bearer(exchanged)).status_code == 200


def test_a_signed_out_session_is_refused(
    authed_client: TestClient, install: FakeInstall, operator_token: str
) -> None:
    """A sign-out anywhere reaches this gateway through the bundle."""
    assert authed_client.get("/v1/config", headers=_bearer(operator_token)).status_code == 200
    claims = install.auth_state().verify(operator_token, classes=(tokens.TYP_SESSION,))
    install.publish(revoked=((claims.jti, claims.exp),))
    # The bundle file is re-read at most once a second.
    time.sleep(1.1)
    assert authed_client.get("/v1/config", headers=_bearer(operator_token)).status_code == 401


# --------------------------------------------------------------------------- #
# Service tokens -- the front door takes this machine's own, and only those
# --------------------------------------------------------------------------- #


def test_service_token_rejected_on_config(authed_client: TestClient, install: FakeInstall) -> None:
    """Config is operator-only: a component has no business editing it."""
    token = install.service("library")
    assert authed_client.get("/v1/config", headers=_bearer(token)).status_code == 401


def test_service_token_rejected_on_admin_restart(
    authed_client: TestClient, install: FakeInstall
) -> None:
    token = install.service("library")
    assert authed_client.post("/v1/admin/restart", headers=_bearer(token)).status_code == 401


def test_this_machines_service_token_is_accepted_on_chat(
    authed_client: TestClient, install: FakeInstall, fake_driver: FakeDriverClient
) -> None:
    """A component beside this gateway calling the one OpenAI-compatible
    endpoint is a legitimate caller."""
    assert _chat(authed_client, fake_driver, install.service("library")) == 200


def test_this_machines_service_token_is_accepted_on_models(
    authed_client: TestClient, install: FakeInstall
) -> None:
    response = authed_client.get("/v1/models", headers=_bearer(install.service("library")))
    assert response.status_code == 200


def test_another_machines_service_token_is_refused_at_the_front_door(
    authed_client: TestClient, install: FakeInstall, fake_driver: FakeDriverClient
) -> None:
    """The case per-node keys exist for: a token another machine signed,
    correctly, and addressed here. A leaked worker key must not buy
    inference on every machine in the install."""
    assert _chat(authed_client, fake_driver, install.foreign_service("agent")) == 401
    response = authed_client.get("/v1/models", headers=_bearer(install.foreign_service()))
    assert response.status_code == 401


def test_a_node_key_cannot_mint_a_session_or_a_client_key(
    authed_client: TestClient, install: FakeInstall, fake_driver: FakeDriverClient
) -> None:
    assert install.node is not None
    for typ, aud in ((tokens.TYP_SESSION, [install.recipient]), (tokens.TYP_CLIENT, ["gateway"])):
        forged, _ = install.node.mint(typ=typ, sub="operator", aud=aud, ttl_seconds=60)
        assert _chat(authed_client, fake_driver, forged) == 401
        assert authed_client.get("/v1/config", headers=_bearer(forged)).status_code == 401


# --------------------------------------------------------------------------- #
# Loading the environment the agent supplies
# --------------------------------------------------------------------------- #


def _env(install: FakeInstall) -> dict[str, str | None]:
    return {
        "trust_bundle_file": str(install.bundle_path),
        "trust_authority": install.authority,
        "auth_recipient": install.recipient,
        "service_token": install.service(),
        "master_key_b64": None,
    }


def test_load_auth_state_disabled_when_nothing_is_supplied() -> None:
    state = load_auth_state(
        trust_bundle_file=None,
        trust_authority=None,
        auth_recipient=None,
        service_token=None,
        master_key_b64=None,
    )
    assert state.auth_disabled is True


@pytest.mark.parametrize(
    "missing", ["trust_bundle_file", "trust_authority", "auth_recipient", "service_token"]
)
def test_load_auth_state_refuses_a_partial_environment(install: FakeInstall, missing: str) -> None:
    """Some but not all is a wiring bug: fail loudly rather than run half-authenticated."""
    env = _env(install)
    env[missing] = None
    with pytest.raises(ValueError, match="missing"):
        load_auth_state(**env)  # type: ignore[arg-type]


def test_load_auth_state_refuses_an_authority_that_is_not_a_key(install: FakeInstall) -> None:
    env = _env(install)
    env["trust_authority"] = "not-a-key"
    with pytest.raises(ValueError, match="TRUST_AUTHORITY"):
        load_auth_state(**env)  # type: ignore[arg-type]


def test_a_bundle_signed_by_another_authority_verifies_nothing(
    tmp_path: Path, install: FakeInstall
) -> None:
    """The file is the agent's, but the authority is pinned: a bundle
    anyone else wrote there is refused and nothing is trusted from it."""
    impostor = FakeInstall(tmp_path / "impostor")
    env = _env(install)
    env["trust_bundle_file"] = str(impostor.bundle_path)
    state = load_auth_state(**env)  # type: ignore[arg-type]
    with pytest.raises(tokens.TokenError):
        state.verify(impostor.session(aud=[install.recipient]), classes=(tokens.TYP_SESSION,))
