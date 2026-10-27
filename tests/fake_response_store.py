"""The store's response and conversation kinds (row 54), as the agent's proxy reaches them.

Scripted here, beside `test_files_door.FakeStore`; the store's own tests prove
its rules, and `specs/scripts/row54-stored-responses-acceptance.py` runs the
real store, agent and gateway together. Kept per owner as the store keeps
them: another owner's id answers as a missing one.
"""

from __future__ import annotations

import json
import secrets
from typing import Any

import httpx
import pytest

NOW = 1_800_000_000
DAY = 24 * 3600


def _problem(status: int, title: str, detail: str) -> httpx.Response:
    return httpx.Response(
        status,
        json={"detail": {"title": title, "status": status, "detail": detail, "component": "store"}},
    )


def _page(items: list[dict[str, Any]], params: Any) -> dict[str, Any]:
    order = params.get("order", "desc")
    limit = int(params.get("limit", 20))
    ordered = items if order == "asc" else list(reversed(items))
    start = 0
    if params.get("after"):
        start = [i["id"] for i in ordered].index(params["after"]) + 1
    return {"data": ordered[start : start + limit], "hasMore": start + limit < len(ordered)}


class FakeResponseStore:
    def __init__(self, gateway_token: str) -> None:
        self.gateway_token = gateway_token
        self.responses: dict[str, dict[str, Any]] = {}
        self.conversations: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str, str]] = []
        self.on = True
        self.down: Exception | None = None
        #: Every keep's body, and the size of every append to a conversation.
        self.bodies: list[dict[str, Any]] = []
        self.appends: list[int] = []

    # -- helpers ------------------------------------------------------------

    def _mine(self, kind: dict[str, dict[str, Any]], ident: str, owner: str) -> Any:
        found = kind.get(ident)
        return found if found is not None and found["owner"] == owner else None

    def _chain(self, row: dict[str, Any], owner: str) -> list[dict[str, Any]] | httpx.Response:
        chain = [row]
        while chain[-1]["previous"]:
            link = self._mine(self.responses, chain[-1]["previous"], owner)
            if link is None:
                return _problem(
                    409,
                    "Chain broken",
                    f"Stored response {chain[-1]['previous']}, earlier in this conversation, "
                    "was deleted, so the conversation cannot be rebuilt.",
                )
            chain.append(link)
        return chain

    @staticmethod
    def _meta(row: dict[str, Any]) -> dict[str, Any]:
        meta = {
            k: row[k]
            for k in ("id", "model", "status", "createdAt", "expiresAt", "rootId", "bytes")
        }
        if row["previous"]:
            meta["previousResponseId"] = row["previous"]
        if row["conversation"]:
            meta["conversationId"] = row["conversation"]
        return meta

    @staticmethod
    def _conversation(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "createdAt": NOW,
            "expiresAt": NOW + 30 * DAY,
            "metadata": row["metadata"],
            "items": len(row["items"]),
        }

    def _off(self) -> httpx.Response:
        return _problem(
            403,
            "Responses are off",
            "This install keeps no responses or conversations: the owner turned it off in "
            "Settings -> Store.",
        )

    # -- the routes -----------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.down is not None:
            raise self.down
        assert request.headers["authorization"] == f"Bearer {self.gateway_token}"
        owner = request.headers["x-eugene-plexus-owner"]
        path = request.url.path.removeprefix("/api/proxy/store")
        self.calls.append((request.method, path, owner))
        parts = path.strip("/").split("/")
        body = json.loads(request.content) if request.content else None
        if path == "/v1/store/status":
            return httpx.Response(200, json={"parts": {"files": True, "responses": self.on}})
        if path.endswith("/context") and not self.on:
            # The store says so before it looks anything up.
            return self._off()
        if path == "/v1/responses" and request.method == "POST":
            return self._keep(owner, body)
        if parts[:2] == ["v1", "responses"]:
            row = self._mine(self.responses, parts[2], owner)
            if row is None:
                return _problem(404, "Not found", f"No such response: {parts[2]}")
            if len(parts) == 3 and request.method == "GET":
                return httpx.Response(
                    200,
                    json={**self._meta(row), "response": {**row["body"], "output": row["output"]}},
                )
            if len(parts) == 3 and request.method == "DELETE":
                del self.responses[row["id"]]
                return httpx.Response(200, json={"id": row["id"], "deleted": True})
            if parts[3] == "context":
                if not self.on:
                    return self._off()
                chain = self._chain(row, owner)
                if isinstance(chain, httpx.Response):
                    return chain
                items = [i for link in reversed(chain) for i in [*link["input"], *link["output"]]]
                return httpx.Response(
                    200,
                    json={
                        "items": items,
                        "bytes": 1,
                        "rootId": row["rootId"],
                        "hadInstructions": bool(row["body"].get("instructions")),
                    },
                )
            if parts[3] == "input_items":
                before: list[dict[str, Any]] = []
                if row["previous"]:
                    chain = self._chain(self.responses[row["previous"]], owner)
                    assert not isinstance(chain, httpx.Response)
                    before = [
                        i for link in reversed(chain) for i in [*link["input"], *link["output"]]
                    ]
                return httpx.Response(200, json=_page([*before, *row["input"]], request.url.params))
        if path == "/v1/conversations" and request.method == "POST":
            if not self.on:
                return self._off()
            ident = "conv_" + secrets.token_hex(24)
            self.conversations[ident] = {
                "id": ident,
                "owner": owner,
                "metadata": body.get("metadata") or {},
                "items": list(body.get("items") or []),
            }
            return httpx.Response(201, json=self._conversation(self.conversations[ident]))
        if parts[:2] == ["v1", "conversations"]:
            row = self._mine(self.conversations, parts[2], owner)
            if row is None:
                return _problem(404, "Not found", f"No such conversation: {parts[2]}")
            if len(parts) == 3:
                if request.method == "GET":
                    return httpx.Response(200, json=self._conversation(row))
                if request.method == "POST":
                    row["metadata"] = body["metadata"] or {}
                    return httpx.Response(200, json=self._conversation(row))
                del self.conversations[row["id"]]
                return httpx.Response(200, json={"id": row["id"], "deleted": True})
            if parts[3] == "context":
                if not self.on:
                    return self._off()
                return httpx.Response(
                    200,
                    json={
                        "items": row["items"],
                        "bytes": 1,
                        "conversationId": row["id"],
                        "position": len(row["items"]),
                    },
                )
            if len(parts) == 4 and request.method == "GET":
                return httpx.Response(200, json=_page(row["items"], request.url.params))
            if len(parts) == 4 and request.method == "POST":
                if not self.on:
                    return self._off()
                row["items"] += body["items"]
                self.appends.append(len(body["items"]))
                return httpx.Response(201, json={"data": body["items"], "hasMore": False})
            item = next((i for i in row["items"] if i["id"] == parts[4]), None)
            if item is None:
                return _problem(404, "Not found", f"No such item: {parts[4]}")
            if request.method == "GET":
                return httpx.Response(200, json=item)
            row["items"].remove(item)
            return httpx.Response(200, json=self._conversation(row))
        raise AssertionError(f"unexpected store call {request.method} {path}")

    def _keep(self, owner: str, body: dict[str, Any]) -> httpx.Response:
        self.bodies.append(body)
        if not self.on:
            return self._off()
        ident = body["id"]
        if ident in self.responses:
            return _problem(409, "Already kept", f"A response {ident} is already kept.")
        previous = body.get("previousResponseId")
        conversation = body.get("conversationId")
        root = ident
        if previous is not None:
            link = self._mine(self.responses, previous, owner)
            if link is None:
                return _problem(404, "Not found", f"No such response: {previous}")
            root = link["rootId"]
        if conversation is not None:
            held = self._mine(self.conversations, conversation, owner)
            if held is None:
                return _problem(404, "Not found", f"No such conversation: {conversation}")
            held["items"] += [*body["input"], *body["output"]]
        for item in [*body["input"], *body["output"]]:
            assert isinstance(item.get("id"), str) and item["id"], item
        row = {
            "id": ident,
            "owner": owner,
            "model": body["model"],
            "status": body["status"],
            "createdAt": NOW,
            "expiresAt": NOW + 30 * DAY,
            "rootId": root,
            "bytes": len(json.dumps(body)),
            "previous": previous,
            "conversation": conversation,
            "input": body["input"],
            "output": body["output"],
            "body": body["response"],
        }
        self.responses[ident] = row
        return httpx.Response(201, json=self._meta(row))

    @property
    def kept(self) -> list[tuple[str, str, str]]:
        return [c for c in self.calls if c[:2] == ("POST", "/v1/responses")]


@pytest.fixture
def response_store(setup) -> FakeResponseStore:  # type: ignore[no-untyped-def]
    app = setup[0]
    fake = FakeResponseStore(app.state.auth_state.service_token)
    app.state.store_transport = httpx.MockTransport(fake.handle)
    return fake
