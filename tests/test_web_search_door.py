"""Row 34: Open WebUI uses the same search accounts and key policy as chat."""

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway.tool_client import ToolDriverError
from tests import test_admission
from tests.test_server_tools import RESULTS, FakeSearch, account

setup = test_admission.setup


def configured(setup, searches=None):
    app, authority, _, _, headers, _ = setup
    search = FakeSearch()
    app.state.routing._snapshot.tools = searches or [account(search)]
    return app, authority, headers, search


def test_answers_in_open_webui_shape(setup):
    app, authority, headers, search = configured(setup)
    with TestClient(app) as client:
        reply = client.post("/v1/web_search", headers=headers, json={"query": "Eugene", "count": 1})
    assert reply.status_code == 200, reply.text
    assert reply.json() == [
        {"link": RESULTS[0]["url"], "title": RESULTS[0]["title"], "snippet": RESULTS[0]["snippet"]}
    ]
    assert len(search.asked) == 1
    # Admitted under the key's limits as a search: no model, released when the answer is sent.
    calls = [c for c in authority.calls if c["action"] != "check"]
    assert [c["action"] for c in calls] == ["acquire", "release"]
    assert all(c.get("purpose") == "search" and "model" not in c for c in calls)
    assert calls[0]["requestId"] == calls[1]["requestId"]


def test_search_is_refused_when_the_keys_allowance_is_used(setup):
    app, authority, headers, search = configured(setup)
    authority.refuse_acquire = 429
    with TestClient(app) as client:
        reply = client.post("/v1/web_search", headers=headers, json={"query": "Eugene"})
    assert reply.status_code == 429, reply.text
    assert reply.headers["Retry-After"] == "7"
    assert not search.asked


def test_local_only_key_refused(setup):
    app, authority, headers, search = configured(setup)
    authority.local_only = True
    with TestClient(app) as client:
        reply = client.post("/v1/web_search", headers=headers, json={"query": "Eugene"})
    assert reply.status_code == 403, reply.text
    assert "this key is local-only" in reply.text
    assert not search.asked


def test_google_account_is_used(setup):
    # OW-GOOGLE overrides section 4's original skipped-account test.
    google = FakeSearch()
    app, _, headers, _ = configured(setup, [account(google, provider="google")])
    with TestClient(app) as client:
        reply = client.post("/v1/web_search", headers=headers, json={"query": "Eugene"})
    assert reply.status_code == 200, reply.text
    assert google.asked


def test_tool_scope_refused(setup):
    app, authority, headers, search = configured(setup)
    authority.allowed_tools = []
    with TestClient(app) as client:
        reply = client.post("/v1/web_search", headers=headers, json={"query": "Eugene"})
    assert reply.status_code == 403, reply.text
    assert "this key's tool scope does not include web_search" in reply.text
    assert not search.asked


@pytest.mark.parametrize(
    "body, words",
    [
        ({"query": ""}, "query: must be 1 to 2000 characters."),
        ({"query": "x" * 2001}, "query: must be 1 to 2000 characters."),
        ({"query": True}, "query: must be 1 to 2000 characters."),
        ({"query": "ok", "count": True}, "count: must be an integer from 1 to 20."),
        ({"query": "ok", "count": 21}, "count: must be an integer from 1 to 20."),
    ],
)
def test_invalid_request(setup, body, words):
    app, _, headers, search = configured(setup)
    with TestClient(app) as client:
        reply = client.post("/v1/web_search", headers=headers, json=body)
    assert reply.status_code == 400, reply.text
    assert words in reply.text
    assert not search.asked


def test_account_failures_and_retry_after(setup):
    down = FakeSearch(error=ToolDriverError(driver="a", status=503, detail="a is offline"))
    busy = FakeSearch(
        error=ToolDriverError(driver="b", status=429, detail="b is busy", retry_after=17)
    )
    app, _, headers, _ = configured(setup, [account(busy, "b"), account(down, "a")])
    with TestClient(app) as client:
        reply = client.post("/v1/web_search", headers=headers, json={"query": "Eugene"})
    assert reply.status_code == 429, reply.text
    assert reply.headers["Retry-After"] == "17"
    assert "b is busy" in reply.text and "a is offline" in reply.text


def test_operator_search_uses_runner_and_shapes_results(setup):
    app, _, _, search = configured(setup)
    with TestClient(app) as client:
        reply = client.post(
            "/v1/web_search", headers=setup[-1], json={"query": "Eugene", "count": 1}
        )
    assert reply.status_code == 200, reply.text
    assert reply.json() == [
        {"link": RESULTS[0]["url"], "title": RESULTS[0]["title"], "snippet": RESULTS[0]["snippet"]}
    ]
    assert search.asked


def test_operator_search_propagates_account_failures(setup):
    busy = FakeSearch(
        error=ToolDriverError(driver="b", status=429, detail="b is busy", retry_after=17)
    )
    app, _, _, _ = configured(setup, [account(busy, "b")])
    with TestClient(app) as client:
        reply = client.post("/v1/web_search", headers=setup[-1], json={"query": "Eugene"})
    assert reply.status_code == 429, reply.text
    assert reply.headers["Retry-After"] == "17"


def test_successful_empty_search_returns_empty_list(setup):
    empty = FakeSearch(results=[])
    app, _, _, _ = configured(setup, [account(empty)])
    with TestClient(app) as client:
        reply = client.post("/v1/web_search", headers=setup[-1], json={"query": "Eugene"})
    assert reply.status_code == 200, reply.text
    assert reply.json() == []
