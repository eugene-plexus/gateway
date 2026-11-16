"""Row 40: a release carries what the request used; a spent allowance is
OpenAI's `insufficient_quota`; an older root is asked again without the new
fields."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import Usage
from tests.test_admission import Authority, body
from tests.test_admission import setup as setup

SENTENCE = "Your group Sales used $4.00 of its $4.00 for November. It resets on 1 December."


def _by_action(authority: Authority, action: str) -> list[dict]:
    return [c for c in authority.calls if c["action"] == action]


@pytest.mark.parametrize("stream", [False, True])
def test_release_carries_tokens_and_price(setup, stream):
    app, authority, allowed, _, headers, _ = setup
    allowed.usage = Usage(promptTokens=7, completionTokens=3, totalTokens=10)
    allowed.locality = "external"
    allowed.price = (3_000_000, 15_000_000)
    from tests.conftest import install_snapshot

    install_snapshot(app.state.routing, allowed)
    with TestClient(app) as c:
        response = c.post(
            "/v1/chat/completions", json=body("allowed", stream=stream), headers=headers
        )
        assert response.status_code == 200, response.text
    hint = {
        "locality": "external",
        "providerPrice": {"inputPerMillionMicros": 3_000_000, "outputPerMillionMicros": 15_000_000},
    }
    (acquire,) = _by_action(authority, "acquire")
    assert acquire["price"] == hint
    (release,) = _by_action(authority, "release")
    assert release["usage"] == {
        "model": "allowed",
        "promptTokens": 7,
        "completionTokens": 3,
        "price": hint,
    }
    # A check carries neither: it takes nothing and counts nothing.
    assert all("usage" not in c and "price" not in c for c in _by_action(authority, "check"))


def test_a_request_that_failed_releases_with_no_usage(setup):
    app, authority, allowed, _, headers, _ = setup
    allowed.generate_error = httpx.ConnectError("offline")
    with TestClient(app) as c:
        assert c.post("/v1/chat/completions", json=body("allowed"), headers=headers).status_code
    (release,) = _by_action(authority, "release")
    assert "usage" not in release


class Spent(Authority):
    """A root whose allowance for this key is spent."""

    def __init__(self, via_agent: bool = False):
        super().__init__()
        self.via_agent = via_agent

    def _handle(self, request):
        if request.url.path.endswith("/admission"):
            payload = json.loads(request.content)
            if payload["action"] in ("check", "acquire") and payload.get("model"):
                self.calls.append(payload)
                said = {"detail": SENTENCE}
                return httpx.Response(
                    429,
                    json={"detail": said} if self.via_agent else said,
                    headers={"Retry-After": "1209600", "X-Eugene-Refusal-Reason": "allowance"},
                )
        return super()._handle(request)


@pytest.mark.parametrize("via_agent", [False, True])
@pytest.mark.parametrize(
    "path,extra,kind",
    [
        ("/v1/chat/completions", {}, "insufficient_quota"),
        ("/v1/chat/completions", {"stream": True}, "insufficient_quota"),
        ("/v1/messages", {"max_tokens": 10}, "rate_limit_error"),
        ("/v1/embeddings", {"input": "x"}, "insufficient_quota"),
    ],
)
def test_a_spent_allowance_is_openais_insufficient_quota(setup, via_agent, path, extra, kind):
    app, _, allowed, _, headers, _ = setup
    spent = Spent(via_agent=via_agent)
    app.state.client_key_guard = spent.as_guard(ttl_seconds=0)
    allowed.supports_embeddings = path.endswith("embeddings")
    from tests.conftest import install_snapshot

    install_snapshot(app.state.routing, allowed)
    request = {"model": "allowed", **extra}
    if not path.endswith("embeddings"):
        request["messages"] = [{"role": "user", "content": "hi"}]
    with TestClient(app) as c:
        response = c.post(path, json=request, headers=headers)
    assert response.status_code == 429, response.text
    assert response.headers["Retry-After"] == "1209600"
    error = response.json()["error"]
    assert error["type"] == kind and error["message"] == SENTENCE
    if kind == "insufficient_quota":
        assert error["code"] == "insufficient_quota"
        assert response.headers["X-Eugene-Refusal-Reason"] == "insufficient_quota"
    assert not allowed.calls, "nothing reached the backend"


class Older(Authority):
    """A root before row 40: its contract refuses the new fields."""

    def _handle(self, request):
        if request.url.path.endswith("/admission"):
            payload = json.loads(request.content)
            if "price" in payload or "usage" in payload:
                self.calls.append({**payload, "refused": True})
                return httpx.Response(422, json={"detail": "extra fields not permitted"})
        return super()._handle(request)


def test_an_older_root_is_asked_again_without_the_new_fields(setup):
    app, _, allowed, _, headers, _ = setup
    older = Older()
    app.state.client_key_guard = older.as_guard(ttl_seconds=0)
    allowed.usage = Usage(promptTokens=7, completionTokens=3, totalTokens=10)
    with TestClient(app) as c:
        for _ in range(2):
            answer = c.post("/v1/chat/completions", json=body("allowed"), headers=headers)
            assert answer.status_code == 200, answer.text
    refused = [call for call in older.calls if call.get("refused")]
    assert [call["action"] for call in refused] == ["acquire"], "asked once, then left out"
    assert [c["action"] for c in older.calls if not c.get("refused")].count("release") == 2


def test_a_served_name_the_gateway_does_not_route_takes_the_asked_models_hint(setup):
    """S1: a backend answering as its dated snapshot keeps the asked model's price."""
    from eugene_plexus_gateway import admission

    app, _, allowed, _, _, _ = setup
    allowed.locality = "external"
    allowed.price = (3_000_000, 15_000_000)
    from tests.conftest import install_snapshot

    install_snapshot(app.state.routing, allowed)
    request = admission.ClientRequest({"type": "http", "app": app, "path": "/v1/chat/completions"})
    request.model = "allowed"
    request.used("allowed-2024-08-06", 10, 5)
    assert request.usage is not None
    assert request.usage["model"] == "allowed-2024-08-06"
    assert request.usage["price"]["providerPrice"]["inputPerMillionMicros"] == 3_000_000
