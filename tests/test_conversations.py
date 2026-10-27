"""Row 54b: OpenAI's Conversations API over the install's store, and `conversation`
on the Responses door.

The store is scripted (`tests/fake_response_store.py`); the store's own tests
prove its rules, and `specs/scripts/row54-stored-responses-acceptance.py` runs
the real store, agent and gateway together.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tests.fake_response_store import response_store  # noqa: F401
from tests.test_admission import setup  # noqa: F401  (the fixture)
from tests.test_responses_store import _as_person, _ask, _said

MISSING = "conv_" + "0" * 48


def _texts(items: list[dict[str, Any]]) -> list[str]:
    return [p["text"] for item in items for p in item.get("content", [])]


def test_a_conversation_carries_items_between_requests(setup, response_store):  # noqa: F811
    app, _, allowed, _, headers, _ = setup
    allowed.responses = ["uno", "dos", "tres"]
    with TestClient(app) as c:
        made = c.post(
            "/v1/conversations",
            json={
                "metadata": {"topic": "numbers"},
                "items": [{"type": "message", "role": "user", "content": "Count in Spanish."}],
            },
            headers=headers,
        )
        assert made.status_code == 200, made.text
        conversation = made.json()
        assert conversation["object"] == "conversation"
        assert conversation["metadata"] == {"topic": "numbers"}
        ident = conversation["id"]

        first = c.post("/v1/responses", json=_ask("one", conversation=ident), headers=headers)
        assert first.status_code == 200, first.text
        assert first.json()["conversation"] == {"id": ident}
        assert [t for _, t in _said(allowed.calls[-1])] == ["Count in Spanish.\n\none"]
        # The object form names it too, as OpenAI's SDKs may send it.
        second = c.post(
            "/v1/responses", json=_ask("two", conversation={"id": ident}), headers=headers
        )
        assert second.status_code == 200, second.text
        assert [t for _, t in _said(allowed.calls[-1])] == [
            "Count in Spanish.\n\none",
            "uno",
            "two",
        ]
        # store: false keeps no response, and the conversation still gains its items.
        third = c.post(
            "/v1/responses", json=_ask("three", conversation=ident, store=False), headers=headers
        )
        assert third.status_code == 200, third.text
        assert third.json()["store"] is False
        assert third.json()["id"] not in response_store.responses

        listed = c.get(f"/v1/conversations/{ident}/items", params={"order": "asc"}, headers=headers)
        assert listed.status_code == 200, listed.text
        page = listed.json()
        assert page["object"] == "list"
        assert _texts(page["data"]) == [
            "Count in Spanish.",
            "one",
            "uno",
            "two",
            "dos",
            "three",
            "tres",
        ]
        assert all(item.get("id") for item in page["data"])

        item = page["data"][0]["id"]
        assert c.get(f"/v1/conversations/{ident}/items/{item}", headers=headers).json()[
            "content"
        ] == [{"type": "input_text", "text": "Count in Spanish."}]
        assert (
            c.delete(f"/v1/conversations/{ident}/items/{item}", headers=headers).json()["id"]
            == ident
        )
        added = c.post(
            f"/v1/conversations/{ident}/items",
            json={"items": [{"role": "user", "content": "four"}]},
            headers=headers,
        )
        assert added.status_code == 200, added.text
        assert _texts(added.json()["data"]) == ["four"]
        updated = c.post(
            f"/v1/conversations/{ident}", json={"metadata": {"topic": "more"}}, headers=headers
        )
        assert updated.json()["metadata"] == {"topic": "more"}
        gone = c.delete(f"/v1/conversations/{ident}", headers=headers)
        assert gone.json() == {"id": ident, "object": "conversation.deleted", "deleted": True}
        assert c.get(f"/v1/conversations/{ident}", headers=headers).status_code == 404


def test_another_owners_conversation_is_the_missing_404(setup, response_store):  # noqa: F811
    app, authority, allowed, _, headers, _ = setup
    ada = _as_person(app, authority, headers, "ada")
    with TestClient(app) as c:
        ident = c.post(
            "/v1/conversations",
            json={"items": [{"role": "user", "content": "Ada's secret"}]},
            headers=ada,
        ).json()["id"]
        ben = _as_person(app, authority, headers, "ben")
        before = len(allowed.calls)
        theirs = c.post("/v1/responses", json=_ask("go on", conversation=ident), headers=ben)
        none = c.post("/v1/responses", json=_ask("go on", conversation=MISSING), headers=ben)
        assert theirs.status_code == none.status_code == 404
        assert theirs.text.replace(ident, "X") == none.text.replace(MISSING, "X")
        assert "secret" not in theirs.text
        assert len(allowed.calls) == before
        for method, suffix in (("GET", ""), ("GET", "/items"), ("DELETE", "")):
            got = c.request(method, f"/v1/conversations/{ident}{suffix}", headers=ben)
            missing = c.request(method, f"/v1/conversations/{MISSING}{suffix}", headers=ben)
            assert got.status_code == missing.status_code == 404, got.text
            assert got.text.replace(ident, "X") == missing.text.replace(MISSING, "X")
        added = c.post(
            f"/v1/conversations/{ident}/items",
            json={"items": [{"role": "user", "content": "x"}]},
            headers=ben,
        )
        assert added.status_code == 404
        assert ident in response_store.conversations


def test_the_conversation_doors_take_openais_shapes_only(setup, response_store):  # noqa: F811
    app, _, _, _, headers, operator = setup
    with TestClient(app) as c:
        ident = c.post("/v1/conversations", json={}, headers=headers).json()["id"]
        too_many = {"items": [{"role": "user", "content": str(n)} for n in range(21)]}
        assert (
            c.post(f"/v1/conversations/{ident}/items", json=too_many, headers=headers).status_code
            == 400
        )
        assert c.post("/v1/conversations", json={"title": "x"}, headers=headers).status_code == 400
        assert c.post("/v1/conversations", content=b"[]", headers=headers).status_code == 400
        assert (
            c.get(
                f"/v1/conversations/{ident}/items", params={"order": "sideways"}, headers=headers
            ).status_code
            == 400
        )
        # The owner's console session keeps no conversations.
        console = c.post("/v1/conversations", json={}, headers=operator)
        assert console.status_code == 403
        assert "console" in console.json()["error"]["message"]
        # Turned off, a conversation is not started, and a request naming one says why.
        response_store.on = False
        off = c.post("/v1/conversations", json={}, headers=headers)
        assert off.status_code == 403
        assert "Settings -> Store" in off.json()["error"]["message"]
        named = c.post("/v1/responses", json=_ask("x", conversation=ident), headers=headers)
        assert named.status_code == 400
        assert "Settings -> Store" in named.json()["error"]["message"]
