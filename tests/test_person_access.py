"""Person proofs are required to survive admission, renewals and every inference door."""

import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import Usage
from tests import test_admission
from tests.conftest import install_snapshot
from tests.test_admission import body

setup = test_admission.setup


def verified(authority, *, person="ada"):
    previous = authority._handle

    def handle(request):
        response = previous(request)
        if request.url.path.endswith("/admission") and response.status_code == 200:
            assert json.loads(request.content).get("personToken") == "PRIVATE_PROOF"
            return httpx.Response(
                200, json={**response.json(), "personId": person, "appClientId": "workbench"}
            )
        return response

    authority._handle = handle


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages", "/v1/embeddings"])
def test_verified_person_is_recorded_without_credentials_and_limits_still_apply(setup, path):
    app, authority, allowed, excluded, headers, operator = setup
    verified(authority)
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    headers["X-Eugene-Person-Token"] = "PRIVATE_PROOF"
    allowed.usage = Usage(promptTokens=7, completionTokens=3, totalTokens=10)
    if path.endswith("embeddings"):
        allowed.supports_embeddings = True
        install_snapshot(app.state.routing, allowed)
    payload = (
        {"model": "allowed", "input": "PRIVATE_INPUT"}
        if path.endswith("embeddings")
        else body("allowed", max_tokens=10)
    )
    with TestClient(app) as c:
        assert c.post(path, json=payload, headers=headers).status_code == 200
        for _ in range(100):
            history = c.get("/v1/metrics/requests", headers=operator)
            if history.json()["requests"]:
                break
            time.sleep(0.01)
        row = history.json()["requests"][0]
        assert row["personId"] == "ada" and row["appClientId"] == "workbench"
        assert "PRIVATE_PROOF" not in history.text and "PRIVATE_INPUT" not in history.text
        assert not excluded.calls
        authority.refuse_renew = 403
        before = len(allowed.calls)
        refused = c.post(path, json=payload, headers=headers)
        assert refused.status_code == 403
        assert len(allowed.calls) == before


def test_old_authority_cannot_silently_drop_person_proof(setup):
    app, _, allowed, excluded, headers, _ = setup
    headers["X-Eugene-Person-Token"] = "PRIVATE_PROOF"
    with TestClient(app) as c:
        assert c.post("/v1/chat/completions", json=body(), headers=headers).status_code == 503
        assert not allowed.calls and not excluded.calls


def test_person_header_cannot_turn_an_operator_token_into_an_app_request(setup):
    app, _, allowed, _, _, operator = setup
    with TestClient(app) as c:
        response = c.post(
            "/v1/chat/completions",
            json=body(),
            headers={**operator, "X-Eugene-Person-Token": "PRIVATE_PROOF"},
        )
        assert response.status_code == 401 and not allowed.calls


def test_catalog_exception_cannot_authorize_inference(setup):
    app, authority, allowed, _, headers, _ = setup
    headers["X-Eugene-App-Operation"] = "catalog"
    with TestClient(app) as c:
        assert c.get("/v1/models", headers=headers).status_code == 200
        assert authority.calls[-1]["purpose"] == "catalog"
        assert c.post("/v1/chat/completions", json=body(), headers=headers).status_code == 403
        assert not allowed.calls
