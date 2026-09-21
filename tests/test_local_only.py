"""Privacy policy across both chat APIs, discovery, embeddings and wake."""

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import Locality
from tests.conftest import install_snapshot, runtime_facts
from tests.test_admission import body
from tests.test_admission import setup as _setup

setup = _setup


def describe_as(driver, locality, enforced=True):
    original = driver.describe()
    info = original.model_copy(
        update={
            "locality": Locality(locality),
            "localOnlyEnforced": enforced,
        }
    )
    driver.describe = lambda: info
    return info


@pytest.mark.parametrize(
    "path,extra",
    [
        ("/v1/chat/completions", {}),
        ("/v1/chat/completions", {"stream": True}),
        ("/v1/messages", {"max_tokens": 10}),
        ("/v1/messages", {"max_tokens": 10, "stream": True}),
        ("/v1/embeddings", {"input": "PRIVATE_INPUT"}),
    ],
)
def test_protected_requests_carry_policy_to_local_driver(setup, path, extra):
    app, authority, local, cloud, headers, _ = setup
    authority.local_only = True
    local.supports_embeddings = path.endswith("embeddings")
    describe_as(local, "local")
    describe_as(cloud, "external")
    install_snapshot(app.state.routing, local, cloud)
    embeds = []
    original_embed = local.embed

    async def capture_embed(request):
        embeds.append(request)
        return await original_embed(request)

    local.embed = capture_embed
    with TestClient(app) as client:
        payload = {"model": "allowed", **extra} if path.endswith("embeddings") else body(**extra)
        response = client.post(path, json=payload, headers=headers)
        assert response.status_code == 200, response.text
        requests = embeds if path.endswith("embeddings") else local.calls
        assert requests and requests[0].localOnly is True
        assert not cloud.calls


@pytest.mark.parametrize(
    "locality,enforced",
    [
        ("external", True),
        ("unknown", True),
        ("local", False),
    ],
)
def test_no_wake_or_content_when_policy_cannot_be_confirmed(setup, locality, enforced):
    app, authority, driver, cloud, headers, _ = setup
    authority.local_only = True
    driver.runtime = "sleeping"
    describe_as(driver, locality, enforced)
    install_snapshot(
        app.state.routing,
        driver,
        cloud,
        runtimes=[runtime_facts("sleeping", status="stopped", start_on_demand=True)],
    )

    class NoWake:
        async def wake(self, resolution):
            raise AssertionError("policy-ineligible backend was woken")

    app.state.lifecycle = NoWake()
    with TestClient(app) as client:
        assert client.get("/v1/models", headers=headers).json()["data"] == []
        response = client.post("/v1/chat/completions", json=body(), headers=headers)
        assert response.status_code == 403, response.text
        assert "local-only" in response.text
        assert "PRIVATE_PROMPT" not in response.text
        assert not driver.calls and not cloud.calls


def test_local_outage_never_falls_back_to_cloud_even_when_model_is_allowed(setup):
    app, authority, local, cloud, headers, operator = setup
    authority.local_only = True
    authority.allowed = None
    describe_as(local, "local")
    describe_as(cloud, "external")
    install_snapshot(app.state.routing, local, cloud)
    local.generate_error = httpx.ConnectError("offline")
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body(), headers=headers)
        assert response.status_code == 502
        assert len(local.calls) == 1 and not cloud.calls
        # An operator request is not silently constrained by somebody else's key.
        response = client.post("/v1/chat/completions", json=body(), headers=operator)
        assert response.status_code == 200
        assert len(cloud.calls) == 1


@pytest.mark.parametrize("changed", ["external", "unreachable"])
def test_stale_routing_snapshot_is_rechecked_before_wake_or_forward(setup, changed):
    app, authority, driver, cloud, headers, _ = setup
    authority.local_only = True
    describe_as(driver, "local")
    install_snapshot(app.state.routing, driver, cloud)
    if changed == "external":
        describe_as(driver, "external")
    else:
        driver.info_error = httpx.ConnectError("gone")
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body(), headers=headers)
        assert response.status_code == 403
        assert not driver.calls and not cloud.calls


def test_protected_wake_keeps_only_local_candidates_and_rechecks_after_start(setup):
    from eugene_plexus_gateway.lifecycle import WakeResult

    app, authority, local, cloud, headers, _ = setup
    authority.local_only = True
    authority.allowed = None
    local.runtime = "sleeping"
    cloud.runtime = "cloud-sleeping"
    describe_as(local, "local")
    describe_as(cloud, "external")
    install_snapshot(
        app.state.routing,
        local,
        cloud,
        runtimes=[
            runtime_facts("sleeping", status="stopped", start_on_demand=True),
            runtime_facts("cloud-sleeping", status="stopped", start_on_demand=True),
        ],
    )
    woken = []

    class Lifecycle:
        async def wake(self, resolution):
            woken.extend(r.name for r in resolution.startable())
            install_snapshot(
                app.state.routing,
                local,
                cloud,
                runtimes=[
                    runtime_facts("sleeping", status="ready"),
                    runtime_facts("cloud-sleeping", status="stopped", start_on_demand=True),
                ],
            )
            return WakeResult(True, "sleeping", 1, "ready")

    async def no_refresh():
        return False

    app.state.lifecycle = Lifecycle()
    app.state.routing.refresh_if_stale = no_refresh
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body(), headers=headers)
        assert response.status_code == 200, response.text
        assert woken == ["sleeping"]
        assert local.calls[0].localOnly is True and not cloud.calls
