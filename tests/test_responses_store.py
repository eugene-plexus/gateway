"""Row 54a: stored responses -- `store: true` keeps a response for its owner, and
`previous_response_id` continues it.

The store is scripted (`tests/fake_response_store.py`); the store's own tests
prove its rules, and `specs/scripts/row54-stored-responses-acceptance.py`
runs the real store, agent and gateway together.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi.testclient import TestClient

from tests.fake_response_store import FakeResponseStore, response_store  # noqa: F401
from tests.test_admission import setup  # noqa: F401  (the fixture)
from tests.test_person_access import verified

MISSING = "resp_" + "0" * 32


def _ask(text: str, **extra: Any) -> dict[str, Any]:
    return {"model": "allowed", "input": text, **extra}


def _said(call: Any) -> list[tuple[str, str]]:
    """What the backend was sent, as (role, text) pairs."""
    out = []
    for message in call.messages:
        content = message.content
        content = getattr(content, "root", content)
        if not isinstance(content, str):
            content = " ".join(
                getattr(getattr(p, "root", p), "text", "") or "" for p in content or []
            )
        role = getattr(message.role, "value", message.role)
        out.append((str(role), content or ""))
    return out


def _as_person(app: Any, authority: Any, headers: dict[str, str], person: str) -> dict[str, str]:
    verified(authority, person=person)
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    return {**headers, "X-Eugene-Person-Token": "PRIVATE_PROOF"}


def test_previous_response_id_rebuilds_the_conversation(setup, response_store):  # noqa: F811
    app, _, allowed, _, headers, _ = setup
    allowed.responses = ["uno", "dos", "tres"]
    with TestClient(app) as c:
        first = c.post(
            "/v1/responses", json=_ask("one", instructions="Answer in Spanish."), headers=headers
        )
        assert first.status_code == 200, first.text
        # Absent `store` is OpenAI's default: kept.
        assert first.json()["store"] is True
        assert "store" not in first.headers.get("x-eugene-plexus-ignored-settings", "")
        assert len(response_store.kept) == 1
        second = c.post(
            "/v1/responses",
            json=_ask("two", previous_response_id=first.json()["id"], instructions="Be brief."),
            headers=headers,
        )
        assert second.status_code == 200, second.text
        assert second.json()["previous_response_id"] == first.json()["id"]
        assert second.json()["store"] is True
        # The backend was given the whole conversation, as for a client that sends it all;
        # the first request's instructions are not carried (OpenAI's rule).
        assert _said(allowed.calls[-1]) == [
            ("system", "Be brief."),
            ("user", "one"),
            ("assistant", "uno"),
            ("user", "two"),
        ]
        third = c.post(
            "/v1/responses",
            json=_ask("three", previous_response_id=second.json()["id"]),
            headers=headers,
        )
        assert third.status_code == 200, third.text
        assert [text for _, text in _said(allowed.calls[-1])] == [
            "one",
            "uno",
            "two",
            "dos",
            "three",
        ]
        # The previous response had instructions and this one sends none: said.
        assert (
            "previous_response_id.instructions" in third.headers["x-eugene-plexus-ignored-settings"]
        )
        # The store was told only this request's own input, not the chain again.
        kept = response_store.responses[third.json()["id"]]
        assert [p["text"] for i in kept["input"] for p in i["content"]] == ["three"]
        assert kept["previous"] == second.json()["id"]


def test_another_owners_id_is_the_missing_404(setup, response_store):  # noqa: F811
    app, authority, allowed, _, headers, _ = setup
    ada = _as_person(app, authority, headers, "ada")
    with TestClient(app) as c:
        hers = c.post("/v1/responses", json=_ask("Ada's secret"), headers=ada)
        assert hers.status_code == 200, hers.text
        ident = hers.json()["id"]
        assert response_store.kept[-1][2] == "person:ada"

        ben = _as_person(app, authority, headers, "ben")
        before = len(allowed.calls)
        theirs = c.post(
            "/v1/responses", json=_ask("go on", previous_response_id=ident), headers=ben
        )
        none = c.post(
            "/v1/responses", json=_ask("go on", previous_response_id=MISSING), headers=ben
        )
        # OpenAI's answer for a previous response it does not have: one 400.
        assert theirs.status_code == none.status_code == 400
        assert theirs.json()["error"]["code"] == "previous_response_not_found"
        assert theirs.json()["error"]["param"] == "previous_response_id"
        assert theirs.text.replace(ident, "X") == none.text.replace(MISSING, "X")
        assert "secret" not in theirs.text
        assert len(allowed.calls) == before, "a backend was asked"

        for method, suffix in (("GET", ""), ("GET", "/input_items"), ("DELETE", "")):
            got = c.request(method, f"/v1/responses/{ident}{suffix}", headers=ben)
            missing = c.request(method, f"/v1/responses/{MISSING}{suffix}", headers=ben)
            assert got.status_code == missing.status_code == 404, got.text
            assert got.text.replace(ident, "X") == missing.text.replace(MISSING, "X")
        assert ident in response_store.responses

        mine = c.get(f"/v1/responses/{ident}", headers=_as_person(app, authority, headers, "ada"))
        assert mine.status_code == 200 and mine.json()["id"] == ident


def test_store_false_is_never_kept(setup, response_store):  # noqa: F811
    app, _, _, _, headers, _ = setup
    with TestClient(app) as c:
        answer = c.post("/v1/responses", json=_ask("forget me", store=False), headers=headers)
        assert answer.status_code == 200, answer.text
        assert answer.json()["store"] is False
        streamed = c.post(
            "/v1/responses", json=_ask("forget me too", store=False, stream=True), headers=headers
        )
        assert streamed.status_code == 200
        assert '"store": false' in streamed.text
    assert response_store.kept == []
    assert response_store.responses == {}
    # Codex sends store: false on every request; nothing asks the store for anything.
    assert response_store.calls == []


def test_a_stream_is_kept_before_its_last_event(setup, response_store):  # noqa: F811
    app, _, allowed, _, headers, _ = setup
    allowed.responses = ["first answer", "second answer"]
    with TestClient(app) as c:
        streamed = c.post("/v1/responses", json=_ask("one", stream=True), headers=headers)
        assert streamed.status_code == 200
        events = [
            json.loads(line[len("data: ") :])
            for line in streamed.text.splitlines()
            if line.startswith("data: ")
        ]
        done = events[-1]
        assert done["type"] == "response.completed"
        assert done["response"]["store"] is True
        ident = done["response"]["id"]
        kept = response_store.responses[ident]
        assert kept["output"][0]["content"][0]["text"] == "first answer"
        # Chained at once: the answer was kept before its last event was sent.
        follow = c.post(
            "/v1/responses", json=_ask("two", previous_response_id=ident), headers=headers
        )
        assert follow.status_code == 200, follow.text
        assert [t for _, t in _said(allowed.calls[-1])] == ["one", "first answer", "two"]
        # GET gives back the response as the stream ended it.
        got = c.get(f"/v1/responses/{ident}", headers=headers).json()
        assert got["output"] == done["response"]["output"]
        assert got["status"] == "completed"


def test_the_consoles_own_session_keeps_nothing(setup, response_store):  # noqa: F811
    app, _, _, _, _, operator = setup
    with TestClient(app) as c:
        answer = c.post("/v1/responses", json=_ask("console test", store=True), headers=operator)
        assert answer.status_code == 200, answer.text
        assert answer.json()["store"] is False
        assert "store" in answer.headers["x-eugene-plexus-ignored-settings"]
        chained = c.post(
            "/v1/responses", json=_ask("go on", previous_response_id=MISSING), headers=operator
        )
        assert chained.status_code == 400
        assert "console" in chained.json()["error"]["message"]
    assert response_store.kept == []


def test_responses_turned_off_name_the_setting(setup, response_store):  # noqa: F811
    app, _, allowed, _, headers, _ = setup
    response_store.on = False
    with TestClient(app) as c:
        answer = c.post("/v1/responses", json=_ask("one", store=True), headers=headers)
        assert answer.status_code == 200, answer.text
        assert answer.json()["store"] is False
        assert "store" in answer.headers["x-eugene-plexus-ignored-settings"]
        before = len(allowed.calls)
        chained = c.post(
            "/v1/responses", json=_ask("two", previous_response_id=MISSING), headers=headers
        )
        assert chained.status_code == 400, chained.text
        assert "Settings -> Store" in chained.json()["error"]["message"]
        assert len(allowed.calls) == before
    assert response_store.kept == []


def test_a_store_that_is_down_never_stops_a_plain_response(setup, response_store):  # noqa: F811
    app, _, _, _, headers, _ = setup
    response_store.down = httpx.ConnectError("[Errno 111] Connection refused")
    with TestClient(app) as c:
        answer = c.post("/v1/responses", json=_ask("one"), headers=headers)
        assert answer.status_code == 200, answer.text
        assert answer.json()["store"] is False
        chained = c.post(
            "/v1/responses", json=_ask("two", previous_response_id=MISSING), headers=headers
        )
        assert chained.status_code == 503, chained.text
        message = chained.json()["error"]["message"]
        assert message.startswith("Eugene's store is not answering: ConnectError")


def test_a_deleted_response_cannot_be_continued(setup, response_store):  # noqa: F811
    app, _, _, _, headers, _ = setup
    with TestClient(app) as c:
        first = c.post("/v1/responses", json=_ask("one"), headers=headers).json()["id"]
        second = c.post(
            "/v1/responses", json=_ask("two", previous_response_id=first), headers=headers
        ).json()["id"]
        gone = c.delete(f"/v1/responses/{first}", headers=headers)
        assert gone.status_code == 200
        assert gone.json() == {"id": first, "object": "response.deleted", "deleted": True}
        again = c.post(
            "/v1/responses", json=_ask("one more", previous_response_id=first), headers=headers
        )
        assert again.status_code == 400
        assert again.json()["error"]["code"] == "previous_response_not_found"
        broken = c.post(
            "/v1/responses", json=_ask("three", previous_response_id=second), headers=headers
        )
        assert broken.status_code == 400, broken.text
        assert first in broken.json()["error"]["message"]


def test_input_items_in_openais_list(setup, response_store):  # noqa: F811
    app, _, allowed, _, headers, _ = setup
    allowed.responses = ["uno", "dos"]
    with TestClient(app) as c:
        first = c.post("/v1/responses", json=_ask("one"), headers=headers).json()["id"]
        second = c.post(
            "/v1/responses", json=_ask("two", previous_response_id=first), headers=headers
        ).json()["id"]
        listed = c.get(f"/v1/responses/{second}/input_items", headers=headers)
        assert listed.status_code == 200, listed.text
        page = listed.json()
        assert page["object"] == "list" and page["has_more"] is False
        texts = [p["text"] for item in page["data"] for p in item["content"]]
        # The chain before it, then its own input; newest first, as OpenAI lists.
        assert texts == ["two", "uno", "one"]
        assert page["first_id"] == page["data"][0]["id"]
        assert page["last_id"] == page["data"][-1]["id"]
        # Every item carries an id, and a string input is kept as OpenAI keeps it.
        assert page["data"][0]["type"] == "message"
        assert page["data"][0]["content"] == [{"type": "input_text", "text": "two"}]
        assert (
            c.get(
                f"/v1/responses/{second}/input_items", params={"limit": 0}, headers=headers
            ).status_code
            == 400
        )
        cancel = c.post(f"/v1/responses/{second}/cancel", headers=headers)
        assert cancel.status_code == 400
        assert "background" in cancel.json()["error"]["message"]


def test_prompt_is_refused_naming_why(setup, response_store):  # noqa: F811
    app, _, _, _, headers, _ = setup
    with TestClient(app) as c:
        refused = c.post("/v1/responses", json=_ask("x", prompt={"id": "pmpt_1"}), headers=headers)
        assert refused.status_code == 400
        assert "prompt dashboard" in refused.json()["error"]["message"]
        both = c.post(
            "/v1/responses",
            json=_ask("x", previous_response_id=MISSING, conversation="conv_" + "0" * 48),
            headers=headers,
        )
        assert both.status_code == 400
        assert "not both" in both.json()["error"]["message"]


def test_the_chain_keeps_a_conversation_on_one_replica_and_apart_from_lookalikes(
    setup,  # noqa: F811
    response_store,  # noqa: F811
):
    """Two chats that open with the same words are two conversations: each keeps to
    its own replica (affinity by the chain's first response, before any hash)."""
    from tests.conftest import FakeDriverClient, make_routing_table, runtime_facts

    app, _, _, _, headers, _ = setup
    a = FakeDriverClient(name="a-driver", base_url="http://a", model_id="allowed", runtime="ra")
    b = FakeDriverClient(name="b-driver", base_url="http://b", model_id="allowed", runtime="rb")
    a.responses = ["a"] * 10
    b.responses = ["b"] * 10
    app.state.routing = make_routing_table(
        a, b, runtimes=[runtime_facts("ra"), runtime_facts("rb")], strategy="round_robin"
    )
    with TestClient(app) as c:
        chains = []
        for _ in range(2):
            before = len(a.calls)
            first = c.post("/v1/responses", json=_ask("same opening"), headers=headers)
            assert first.status_code == 200, first.text
            chains.append((first.json()["id"], a if len(a.calls) > before else b))
        # Placed in turn as two conversations, not sent home as one by their
        # shared opening.
        assert {home.name for _, home in chains} == {"a-driver", "b-driver"}
        for head, home in chains:
            previous = head
            for turn in range(3):
                before = len(home.calls)
                follow = c.post(
                    "/v1/responses",
                    json=_ask(f"turn {turn}", previous_response_id=previous),
                    headers=headers,
                )
                assert follow.status_code == 200, follow.text
                assert len(home.calls) == before + 1, f"turn {turn} left its replica"
                previous = follow.json()["id"]


def test_a_stream_the_client_left_is_kept_with_what_was_sent(setup, response_store):  # noqa: F811
    """A cancelled stream is kept with what was sent, `incomplete` (design, row 54)."""
    import asyncio

    from eugene_plexus_gateway.driver_client import StreamEvent
    from tests.conftest import FakeDriverClient, make_routing_table

    class Slow(FakeDriverClient):
        async def stream(self, request):  # type: ignore[no-untyped-def,override]
            self.calls.append(request)
            yield StreamEvent(text="partial ")
            yield StreamEvent(text="answer")
            await asyncio.sleep(3600)
            yield StreamEvent(text="never")

    app, _, _, _, headers, _ = setup
    slow = Slow(name="allowed-driver", model_id="allowed")
    app.state.routing = make_routing_table(slow)
    body = json.dumps(_ask("one", stream=True)).encode()

    async def drive() -> list[bytes]:
        gone = asyncio.Event()
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
                if b'"delta": "answer"' in b"".join(sent):
                    gone.set()

        scope = {
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
        await asyncio.wait_for(app(scope, receive, send), timeout=10)
        for _ in range(200):
            if response_store.responses:
                break
            await asyncio.sleep(0.02)
        return sent

    with TestClient(app) as c:
        sent = c.portal.call(drive)
    assert b"response.completed" not in b"".join(sent)
    [kept] = response_store.responses.values()
    assert kept["status"] == "incomplete"
    assert kept["output"][0]["content"][0]["text"] == "partial answer"
    assert kept["output"][0]["status"] == "incomplete"
    assert [p["text"] for i in kept["input"] for p in i["content"]] == ["one"]


def test_a_client_that_resends_its_history_keeps_its_replica(setup, response_store):  # noqa: F811
    """OpenAI's default `store` is true, and many clients resend their whole history
    rather than chain. Each turn is a new response, so its own id cannot be the key:
    after the opening, the history's own key holds the chat to one replica."""
    from tests.conftest import FakeDriverClient, make_routing_table, runtime_facts

    app, _, _, _, headers, _ = setup
    a = FakeDriverClient(name="a-driver", base_url="http://a", model_id="allowed", runtime="ra")
    b = FakeDriverClient(name="b-driver", base_url="http://b", model_id="allowed", runtime="rb")
    a.responses = ["ok"] * 10
    b.responses = ["ok"] * 10
    app.state.routing = make_routing_table(
        a, b, runtimes=[runtime_facts("ra"), runtime_facts("rb")], strategy="round_robin"
    )
    history: list[dict[str, Any]] = [{"role": "user", "content": "fix the bug"}]
    served = []
    with TestClient(app) as c:
        for turn in range(4):
            before = (len(a.calls), len(b.calls))
            answer = c.post(
                "/v1/responses", json={"model": "allowed", "input": history}, headers=headers
            )
            assert answer.status_code == 200, answer.text
            assert answer.json()["store"] is True
            served.append("a" if len(a.calls) > before[0] else "b")
            history += [
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": f"and turn {turn}"},
            ]
    # The opening is placed as a chat of its own; every turn after it goes home.
    assert len(set(served[1:])) == 1, served
