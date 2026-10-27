"""Row 54: the second-model review's findings at the gateway's doors, each a test
that failed before its fix (findings 3, 4, 9a and 10)."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx
from fastapi.testclient import TestClient

from tests.fake_response_store import FakeResponseStore, response_store  # noqa: F401
from tests.test_admission import setup  # noqa: F401  (the fixture)
from tests.test_responses_store import _ask, _said

ITEM_ID = re.compile(r"^[a-z]{1,16}_[0-9a-f]{16,64}$")


def _scope(headers: dict[str, str], body: bytes) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"authorization", headers["Authorization"].encode()),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 5555),
        "server": ("127.0.0.1", 8080),
    }


def test_a_client_that_leaves_after_its_answer_was_kept_finds_it_kept_once(
    setup,  # noqa: F811
    response_store: FakeResponseStore,  # noqa: F811
) -> None:
    """Finding 3: the client leaves after the store has the answer and before the last
    event. Keeping is one operation, done once: the answer stays as it was kept, and
    no second save races it (which the store refused with a 409)."""
    app, _, allowed, _, headers, _ = setup
    allowed.responses = ["the whole answer"]
    body = json.dumps(_ask("one", stream=True)).encode()
    gone = asyncio.Event()
    keeps: list[str] = []

    async def store(request: httpx.Request) -> httpx.Response:
        answer = response_store.handle(request)
        if request.method == "POST" and request.url.path.endswith("/v1/responses"):
            keeps.append(json.loads(request.content)["status"])
            # Kept; the client leaves before the last event reaches it.
            gone.set()
            await asyncio.sleep(0.3)
        return answer

    app.state.store_transport = httpx.MockTransport(store)

    async def drive() -> list[bytes]:
        sent: list[bytes] = []
        asked = False

        async def receive() -> dict[str, Any]:
            nonlocal asked
            if not asked:
                asked = True
                return {"type": "http.request", "body": body, "more_body": False}
            await gone.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.body":
                sent.append(message.get("body", b""))

        await asyncio.wait_for(app(_scope(headers, body), receive, send), timeout=10)
        await asyncio.sleep(0.6)
        return sent

    with TestClient(app) as c:
        sent = c.portal.call(drive)
    assert b"response.completed" not in b"".join(sent)
    assert keeps == ["completed"], keeps
    [kept] = response_store.responses.values()
    assert kept["status"] == "completed"
    assert kept["output"][0]["content"][0]["text"] == "the whole answer"


def test_an_item_id_in_another_shape_is_given_a_new_one(
    setup,  # noqa: F811
    response_store: FakeResponseStore,  # noqa: F811
) -> None:
    """Finding 4: an id that carries words never reaches a path, a log or an error."""
    app, _, _, _, headers, _ = setup
    worded = {"id": "msg_my secret password", "role": "user", "content": "hi"}
    hex_id = "msg_" + "ab" * 16
    with TestClient(app) as c:
        made = c.post(
            "/v1/conversations",
            json={"items": [worded, {"id": hex_id, "role": "user", "content": "ho"}]},
            headers=headers,
        )
        assert made.status_code == 200, made.text
        listed = c.get(
            f"/v1/conversations/{made.json()['id']}/items", params={"order": "asc"}, headers=headers
        ).json()["data"]
        ids = [item["id"] for item in listed]
        assert all(ITEM_ID.match(i) for i in ids), ids
        assert "secret" not in json.dumps(ids)
        # An id already in OpenAI's shape is kept as given.
        assert ids[1] == hex_id
        answer = c.post(
            "/v1/responses",
            json={"model": "allowed", "input": [{**worded, "type": "message"}]},
            headers=headers,
        )
        assert answer.status_code == 200, answer.text
        kept = response_store.responses[answer.json()["id"]]
        assert ITEM_ID.match(kept["input"][0]["id"]), kept["input"][0]["id"]
        # A cursor that is not an item id is refused without being repeated.
        cursor = c.get(
            f"/v1/conversations/{made.json()['id']}/items",
            params={"after": "my secret cursor"},
            headers=headers,
        )
        assert cursor.status_code == 400 and "secret" not in cursor.text


def test_a_conversation_continues_with_no_input(setup, response_store):  # noqa: F811
    """Finding 9a: OpenAI makes `input` optional; with a conversation, none is none."""
    app, _, allowed, _, headers, _ = setup
    with TestClient(app) as c:
        ident = c.post(
            "/v1/conversations",
            json={"items": [{"role": "user", "content": "What is two and two?"}]},
            headers=headers,
        ).json()["id"]
        answer = c.post(
            "/v1/responses", json={"model": "allowed", "conversation": ident}, headers=headers
        )
        assert answer.status_code == 200, answer.text
        assert _said(allowed.calls[-1]) == [("user", "What is two and two?")]
        first = c.post("/v1/responses", json=_ask("hi"), headers=headers).json()["id"]
        again = c.post(
            "/v1/responses",
            json={"model": "allowed", "previous_response_id": first},
            headers=headers,
        )
        assert again.status_code == 200, again.text
        # Without either, input is still required.
        bare = c.post("/v1/responses", json={"model": "allowed"}, headers=headers)
        assert bare.status_code == 400


def test_a_turn_is_kept_as_what_it_read_and_its_items_go_in_at_once(
    setup,  # noqa: F811
    response_store: FakeResponseStore,  # noqa: F811
) -> None:
    """Finding 10: the position a turn read the conversation at goes back with it, and a
    `store: false` turn's items are one append, however many."""
    app, _, _, _, headers, _ = setup
    with TestClient(app) as c:
        ident = c.post(
            "/v1/conversations",
            json={"items": [{"role": "user", "content": "start"}]},
            headers=headers,
        ).json()["id"]
        kept = c.post("/v1/responses", json=_ask("one", conversation=ident), headers=headers)
        assert kept.status_code == 200, kept.text
        assert response_store.bodies[-1]["conversationPosition"] == 1
        many = [{"role": "user", "content": f"line {n}"} for n in range(25)]
        before = len(response_store.appends)
        unkept = c.post(
            "/v1/responses",
            json={"model": "allowed", "input": many, "conversation": ident, "store": False},
            headers=headers,
        )
        assert unkept.status_code == 200, unkept.text
        assert response_store.appends[before:] == [26], response_store.appends
