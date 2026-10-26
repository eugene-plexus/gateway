"""Row 53: /v1/files and /v1/uploads over the install's store, and file_id on the doors.

The store is scripted here (`FakeStore`, behind the agent's proxy path);
`specs/scripts/row53-store-acceptance.py` runs the real store, agent and
gateway together.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.test_admission import body, setup  # noqa: F401  (the fixture)
from tests.test_person_access import verified

PDF = b"%PDF-1.4\n% a tiny document\n%%EOF\n"


class FakeStore:
    """The store, as the agent's `/api/proxy/store` reaches it: files kept per owner."""

    def __init__(self, gateway_token: str) -> None:
        self.gateway_token = gateway_token
        self.files: dict[str, dict[str, Any]] = {}
        self.content: dict[str, bytes] = {}
        self.uploads: dict[str, dict[str, Any]] = {}
        self.parts: dict[str, tuple[str, str, bytes]] = {}
        self.owners_seen: list[str] = []
        self.down: Exception | None = None
        self.content_reads = 0
        self.proxy_problem: tuple[int, str] | None = None

    def _missing(self, kind: str, ident: str) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "detail": {
                    "title": "Not found",
                    "status": 404,
                    "detail": f"No such {kind}: {ident}",
                    "component": "store",
                }
            },
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.down is not None:
            raise self.down
        if self.proxy_problem is not None:
            status, detail = self.proxy_problem
            return httpx.Response(
                status,
                json={
                    "detail": {
                        "title": "Upstream unreachable",
                        "status": status,
                        "detail": detail,
                        "component": "agent",
                    }
                },
            )
        assert request.headers["authorization"] == f"Bearer {self.gateway_token}"
        if request.url.path.endswith("/v1/store/status"):
            # Row 54: a Responses request asks whether answers are kept; this store keeps none.
            return httpx.Response(200, json={"parts": {"files": True, "responses": False}})
        owner = request.headers["x-eugene-plexus-owner"]
        self.owners_seen.append(owner)
        path = request.url.path.removeprefix("/api/proxy/store")
        parts = path.strip("/").split("/")
        now = 1_800_000_000
        if path == "/v1/files" and request.method == "POST":
            ident = "file-" + secrets.token_hex(12)
            data = request.content
            self.files[ident] = {
                "id": ident,
                "owner": owner,
                "filename": request.url.params["filename"],
                "purpose": request.url.params["purpose"],
                "bytes": len(data),
                "mimeType": request.url.params.get("mimeType") or "application/octet-stream",
                "sha256": hashlib.sha256(data).hexdigest(),
                "createdAt": now,
                "status": "processed",
            }
            if request.url.params.get("expiresAfterSeconds"):
                self.files[ident]["expiresAt"] = now + int(
                    request.url.params["expiresAfterSeconds"]
                )
            self.content[ident] = data
            return httpx.Response(201, json=self._public(ident))
        if path == "/v1/files" and request.method == "GET":
            mine = [self._public(i) for i, f in self.files.items() if f["owner"] == owner]
            return httpx.Response(200, json={"data": mine, "hasMore": False})
        if parts[:2] == ["v1", "files"] and len(parts) >= 3:
            ident = parts[2]
            found = self.files.get(ident)
            if found is None or found["owner"] != owner:
                return self._missing("file", ident)
            if len(parts) == 4 and parts[3] == "content":
                self.content_reads += 1
                return httpx.Response(
                    200, content=self.content[ident], headers={"content-type": found["mimeType"]}
                )
            if request.method == "DELETE":
                del self.files[ident]
                return httpx.Response(200, json={"id": ident, "deleted": True})
            return httpx.Response(200, json=self._public(ident))
        if path == "/v1/uploads":
            sent = json.loads(request.content)
            ident = "upload_" + secrets.token_hex(12)
            self.uploads[ident] = {**sent, "owner": owner, "status": "pending"}
            return httpx.Response(201, json=self._upload(ident))
        if parts[:2] == ["v1", "uploads"]:
            ident = parts[2]
            upload = self.uploads.get(ident)
            if upload is None or upload["owner"] != owner:
                return self._missing("upload", ident)
            if parts[3] == "parts":
                part = "part_" + secrets.token_hex(12)
                self.parts[part] = (ident, owner, request.content)
                return httpx.Response(
                    201,
                    json={
                        "id": part,
                        "uploadId": ident,
                        "createdAt": now,
                        "bytes": len(request.content),
                    },
                )
            if parts[3] == "complete":
                sent = json.loads(request.content)
                data = b"".join(self.parts[p][2] for p in sent["partIds"])
                file_id = "file-" + secrets.token_hex(12)
                self.files[file_id] = {
                    "id": file_id,
                    "owner": owner,
                    "filename": upload["filename"],
                    "purpose": upload["purpose"],
                    "bytes": len(data),
                    "mimeType": upload["mimeType"],
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "createdAt": now,
                    "expiresAt": now + 2592000,
                    "status": "processed",
                }
                self.content[file_id] = data
                upload.update(status="completed", file=file_id)
                return httpx.Response(200, json=self._upload(ident))
        raise AssertionError(f"unexpected store call {request.method} {path}")

    def _public(self, ident: str) -> dict[str, Any]:
        return {k: v for k, v in self.files[ident].items() if k != "owner"}

    def _upload(self, ident: str) -> dict[str, Any]:
        upload = self.uploads[ident]
        answer = {
            "id": ident,
            "filename": upload["filename"],
            "purpose": upload["purpose"],
            "bytes": upload["bytes"],
            "mimeType": upload["mimeType"],
            "createdAt": 1_800_000_000,
            "expiresAt": 1_800_003_600,
            "status": upload["status"],
        }
        if upload.get("file"):
            answer["file"] = self._public(upload["file"])
        return answer


@pytest.fixture
def store(setup, install):  # noqa: F811
    app = setup[0]
    fake = FakeStore(app.state.auth_state.service_token)
    app.state.store_transport = httpx.MockTransport(fake.handle)
    return fake


@pytest.fixture
def reader(setup):  # noqa: F811
    """A backend whose model reads PDFs and images, under the name the key allows."""
    from tests.conftest import make_routing_table
    from tests.test_attachments import Takes

    app = setup[0]
    reads = Takes("file", "image", name="reader", model_id="allowed")
    app.state.routing = make_routing_table(reads)
    return reads


def _upload(
    c: TestClient, headers: dict[str, str], data: bytes, name: str = "doc.pdf", **form: str
) -> httpx.Response:
    return c.post(
        "/v1/files",
        files={"file": (name, data, "application/pdf")},
        data={"purpose": "user_data", **form},
        headers=headers,
    )


def _chat_with(file_id: str) -> dict[str, Any]:
    return body(
        "allowed",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize"},
                    {"type": "file", "file": {"file_id": file_id}},
                ],
            }
        ],
    )


def test_files_are_503_naming_the_cause_when_the_store_is_down(setup, store):  # noqa: F811
    app, _, allowed, _, headers, _ = setup
    store.down = httpx.ConnectError("[Errno 111] Connection refused")
    with TestClient(app) as c:
        listed = c.get("/v1/files", headers=headers)
        assert listed.status_code == 503
        message = listed.json()["error"]["message"]
        assert message.startswith("Eugene's store is not answering: ConnectError")
        assert "Connection refused" in message
        assert "Chat keeps working" in message
        kept = _upload(c, headers, PDF)
        assert kept.status_code == 503
        # A chat that names no file never asks the store, and answers.
        store.down = AssertionError("the store was asked")
        assert (
            c.post("/v1/chat/completions", json=body("allowed"), headers=headers).status_code == 200
        )
        # The agent's own words when it cannot reach the store are the cause.
        store.down = None
        store.proxy_problem = (
            502,
            "store at http://127.0.0.1:8084/v1/files did not answer: refused",
        )
        refused = c.get("/v1/files", headers=headers)
        assert refused.status_code == 503
        assert "store at http://127.0.0.1:8084" in refused.json()["error"]["message"]
        # And a chat with a file_id says the same, before any backend is asked.
        before = len(allowed.calls)
        chat = c.post("/v1/chat/completions", json=_chat_with("file-" + "0" * 24), headers=headers)
        assert chat.status_code == 503
        assert "Eugene's store is not answering" in chat.json()["error"]["message"]
        assert len(allowed.calls) == before


def test_a_file_id_resolves_for_its_owner_only(setup, store, reader):  # noqa: F811
    app, authority, _, _, headers, _ = setup
    allowed = reader
    verified(authority, person="ada")
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    ada = {**headers, "X-Eugene-Person-Token": "PRIVATE_PROOF"}
    with TestClient(app) as c:
        kept = _upload(c, ada, PDF)
        assert kept.status_code == 200, kept.text
        ident = kept.json()["id"]
        assert kept.json()["object"] == "file"
        assert store.owners_seen[-1] == "person:ada"

        answered = c.post("/v1/chat/completions", json=_chat_with(ident), headers=ada)
        assert answered.status_code == 200, answered.text
        sent = allowed.calls[-1].model_dump_json()
        assert "data:application/pdf;base64," + base64.b64encode(PDF).decode() in sent
        assert ident not in sent

        # Ben is another person; his chat with Ada's id is refused exactly as
        # one with an id that does not exist, and no backend is asked.
        verified(authority, person="ben")
        app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
        ben = {**headers, "X-Eugene-Person-Token": "PRIVATE_PROOF"}
        before = len(allowed.calls)
        hers = c.post("/v1/chat/completions", json=_chat_with(ident), headers=ben)
        missing = "file-" + "0" * 24
        none = c.post("/v1/chat/completions", json=_chat_with(missing), headers=ben)
        assert hers.status_code == none.status_code == 400
        assert hers.json()["error"]["message"].replace(ident, "X") == none.json()["error"][
            "message"
        ].replace(missing, "X")
        assert len(allowed.calls) == before
        assert c.get(f"/v1/files/{ident}", headers=ben).status_code == 404
        assert c.get("/v1/files", headers=ben).json()["data"] == []


def test_the_owner_is_never_taken_from_the_request(setup, store, install):  # noqa: F811
    app, _, _, _, headers, operator = setup
    with TestClient(app) as c:
        forged = {**headers, "X-Eugene-Plexus-Owner": "person:someone-else"}
        assert _upload(c, forged, PDF).status_code == 200
        assert store.owners_seen[-1] == "key:key-1"
        assert _upload(c, operator, PDF).status_code == 200
        assert store.owners_seen[-1] == "install"
        # One of this machine's own components keeps files of its own.
        agent = {"Authorization": f"Bearer {install.service('agent')}"}
        assert _upload(c, agent, PDF).status_code == 200
        assert store.owners_seen[-1] == "component:agent"


def test_a_key_with_a_lineage_keeps_its_files_after_rotation(setup, store):  # noqa: F811
    app, authority, _, _, headers, _ = setup
    previous = authority._handle

    def with_lineage(request: httpx.Request) -> httpx.Response:
        response = previous(request)
        if request.url.path.endswith("/admission") and response.status_code == 200:
            return httpx.Response(200, json={**response.json(), "lineage": "key-0"})
        return response

    authority._handle = with_lineage
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    with TestClient(app) as c:
        assert _upload(c, headers, PDF).status_code == 200
        assert store.owners_seen[-1] == "key:key-0"


def test_the_files_doors_admit_only_who_the_inference_doors_admit(setup, store, install):  # noqa: F811
    """The admission matrix (review of row 53): every credential that may not
    keep a file is refused before the store is asked."""
    app, authority, _, _, headers, operator = setup
    previous = authority._handle

    def judged(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        if request.url.path.endswith("/admission") and body.get("personToken"):
            if body["personToken"] == "REMOVED_PERSON":
                return httpx.Response(401)
            response = previous(request)
            return httpx.Response(
                200, json={**response.json(), "personId": "ada", "appClientId": "workbench"}
            )
        return previous(request)

    authority._handle = judged
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    refused = {
        "no credential": {},
        "an expired client key": {
            "Authorization": "Bearer "
            + install.client_key(jti="key-1", ttl=60, now=int(time.time()) - 7200)
        },
        "a session addressed to another machine": {
            "Authorization": "Bearer " + install.session(aud=["node:elsewhere", "control"])
        },
        "another machine's service token": {
            "Authorization": "Bearer " + install.foreign_service("gateway")
        },
        "a person token beside the owner's session": {
            **operator,
            "X-Eugene-Person-Token": "PRIVATE_PROOF",
        },
        "a person the root no longer knows": {
            **headers,
            "X-Eugene-Person-Token": "REMOVED_PERSON",
        },
    }
    with TestClient(app) as c:
        for label, sent in refused.items():
            answer = c.get("/v1/files", headers=sent)
            assert answer.status_code == 401, (label, answer.status_code, answer.text)
        assert store.owners_seen == []

        # The app key alone keeps the key's files; with a verified person, the person's.
        assert c.get("/v1/files", headers=headers).status_code == 200
        assert store.owners_seen[-1] == "key:key-1"
        person = {**headers, "X-Eugene-Person-Token": "PRIVATE_PROOF"}
        assert c.get("/v1/files", headers=person).status_code == 200
        assert store.owners_seen[-1] == "person:ada"

        # Revoked for real: the policy says so and the authority refuses it.
        authority.revoke("key-1")
        seen = len(store.owners_seen)
        for sent in (headers, person):
            answer = c.get("/v1/files", headers=sent)
            assert answer.status_code == 401, answer.text
        assert len(store.owners_seen) == seen


def test_a_file_named_twice_counts_twice_against_one_budget(setup, store, reader):  # noqa: F811
    """Review of row 53: a 6 MiB text file named twice is 12 MiB of attachments.
    Refused before the second copy is downloaded, and no backend is asked."""
    app, _, _, _, headers, _ = setup
    with TestClient(app) as c:
        big = c.post(
            "/v1/files",
            files={"file": ("big.txt", b"a" * (6 * 1024 * 1024), "text/plain")},
            data={"purpose": "user_data"},
            headers=headers,
        )
        assert big.status_code == 200, big.text
        ident = big.json()["id"]
        before = len(reader.calls)
        downloads = store.content_reads
        twice = body(
            "allowed",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "file", "file": {"file_id": ident}},
                        {"type": "file", "file": {"file_id": ident}},
                    ],
                }
            ],
        )
        refused = c.post("/v1/chat/completions", json=twice, headers=headers)
        assert refused.status_code == 400, refused.text
        assert "11 MiB" in refused.json()["error"]["message"]
        assert len(reader.calls) == before
        assert store.content_reads == downloads + 1, "the second copy was downloaded"


def test_file_id_text_and_inline_attachments_share_one_budget(setup, store, reader):  # noqa: F811
    """Second review of row 53: a 6 MiB text file named by id plus a 6 MiB PDF sent
    inline is 12 MiB against the 11 MiB limit. Each alone passes its own check."""
    import base64

    app, _, _, _, headers, _ = setup
    with TestClient(app) as c:
        big = c.post(
            "/v1/files",
            files={"file": ("big.txt", b"a" * (6 * 1024 * 1024), "text/plain")},
            data={"purpose": "user_data"},
            headers=headers,
        )
        assert big.status_code == 200, big.text
        ident = big.json()["id"]
        pdf = base64.b64encode(b"%PDF-1.4\n" + b"x" * (6 * 1024 * 1024)).decode()
        inline = {"type": "file", "file": {"filename": "x.pdf", "file_data": pdf}}

        def ask(parts):
            return c.post(
                "/v1/chat/completions",
                json=body("allowed", messages=[{"role": "user", "content": parts}]),
                headers=headers,
            )

        # Each alone is within the limit.
        before = len(reader.calls)
        assert ask([{"type": "file", "file": {"file_id": ident}}]).status_code == 200
        assert ask([inline]).status_code == 200
        assert len(reader.calls) == before + 2
        # Together they are not, in either order, and no backend is asked.
        for parts in (
            [{"type": "file", "file": {"file_id": ident}}, inline],
            [inline, {"type": "file", "file": {"file_id": ident}}],
        ):
            refused = ask(parts)
            assert refused.status_code == 400, refused.text
            assert "11 MiB" in refused.json()["error"]["message"]
        assert len(reader.calls) == before + 2


def test_bytes_that_are_not_utf8_are_not_text_whatever_they_were_called(setup, store, reader):  # noqa: F811
    app, _, _, _, headers, _ = setup
    with TestClient(app) as c:
        odd = c.post(
            "/v1/files",
            files={"file": ("odd.txt", b"\xff\x00", "text/plain")},
            data={"purpose": "user_data"},
            headers=headers,
        )
        ident = odd.json()["id"]
        answer = c.post("/v1/chat/completions", json=_chat_with(ident), headers=headers)
        assert answer.status_code == 400, answer.text
        assert "application/octet-stream" in answer.json()["error"]["message"]


def test_an_end_is_the_apps_choice_within_openais_bounds(setup, store):  # noqa: F811
    app, _, _, _, headers, _ = setup
    with TestClient(app) as c:
        kept = _upload(c, headers, PDF)
        assert kept.json()["expires_at"] is None, "a file is kept until deleted, as OpenAI keeps"
        for seconds, status in (
            ("1", 400),
            ("3599", 400),
            ("3600", 200),
            ("2592000", 200),
            ("2592001", 400),
        ):
            answer = _upload(
                c,
                headers,
                PDF,
                **{"expires_after[anchor]": "created_at", "expires_after[seconds]": seconds},
            )
            assert answer.status_code == status, (seconds, answer.text)
        ended = _upload(
            c,
            headers,
            PDF,
            **{"expires_after[anchor]": "created_at", "expires_after[seconds]": "3600"},
        ).json()
        assert ended["expires_at"] == ended["created_at"] + 3600


def test_anthropics_files_api_is_the_same_door(setup, store):  # noqa: F811
    app, _, _, _, headers, _ = setup
    anthropic = {
        "x-api-key": headers["Authorization"].split()[1],
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "files-api-2025-04-14",
    }
    with TestClient(app) as c:
        kept = c.post(
            "/v1/files",
            files={"file": ("notes.txt", b"plain words", "text/plain")},
            headers=anthropic,
        )
        assert kept.status_code == 200, kept.text
        meta = kept.json()
        assert (
            meta["type"] == "file"
            and meta["size_bytes"] == 11
            and meta["mime_type"] == "text/plain"
        )
        assert meta["created_at"].endswith("Z")
        listed = c.get("/v1/files", headers=anthropic).json()
        assert [f["id"] for f in listed["data"]] == [meta["id"]] and "object" not in listed
        # The OpenAI door sees the same file.
        assert c.get(f"/v1/files/{meta['id']}", headers=headers).json()["purpose"] == "user_data"
        missing = c.get("/v1/files/file-" + "0" * 24, headers=anthropic)
        assert missing.status_code == 404
        assert missing.json()["error"]["type"] == "not_found_error"
        # A text file in a Messages document is carried as text.
        message = {
            "model": "allowed",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "document", "source": {"type": "file", "file_id": meta["id"]}},
                        {"type": "text", "text": "Read it"},
                    ],
                }
            ],
        }
        assert c.post("/v1/messages", json=message, headers=anthropic).status_code == 200
        gone = c.delete(f"/v1/files/{meta['id']}", headers=anthropic)
        assert gone.json() == {"id": meta["id"], "type": "file_deleted"}


def test_a_responses_input_file_is_read_from_the_store(setup, store, reader):  # noqa: F811
    app, _, _, _, headers, _ = setup
    allowed = reader
    with TestClient(app) as c:
        ident = _upload(c, headers, PDF).json()["id"]
        response = c.post(
            "/v1/responses",
            json={
                "model": "allowed",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Summarize"},
                            {"type": "input_file", "file_id": ident},
                        ],
                    }
                ],
            },
            headers=headers,
        )
        assert response.status_code == 200, response.text
        assert "data:application/pdf;base64," in allowed.calls[-1].model_dump_json()


def test_uploads_in_openais_shapes(setup, store):  # noqa: F811
    app, _, _, _, headers, _ = setup
    whole = b"a" * 10 + b"b" * 5
    with TestClient(app) as c:
        made = c.post(
            "/v1/uploads",
            json={
                "bytes": 15,
                "filename": "big.jsonl",
                "mime_type": "text/jsonl",
                "purpose": "batch",
            },
            headers=headers,
        )
        assert made.status_code == 200, made.text
        upload = made.json()
        assert (
            upload["object"] == "upload"
            and upload["status"] == "pending"
            and upload["file"] is None
        )
        first = c.post(
            f"/v1/uploads/{upload['id']}/parts",
            files={"data": ("blob", whole[:10])},
            headers=headers,
        )
        second = c.post(
            f"/v1/uploads/{upload['id']}/parts",
            files={"data": ("blob", whole[10:])},
            headers=headers,
        )
        assert first.json()["object"] == "upload.part"
        done = c.post(
            f"/v1/uploads/{upload['id']}/complete",
            json={"part_ids": [first.json()["id"], second.json()["id"]]},
            headers=headers,
        )
        assert done.json()["status"] == "completed"
        assert done.json()["file"]["object"] == "file"
        content = c.get(f"/v1/files/{done.json()['file']['id']}/content", headers=headers)
        assert content.content == whole


def test_a_body_over_16_mib_is_refused_before_the_store(setup, store):  # noqa: F811
    app, _, _, _, headers, _ = setup
    with TestClient(app) as c:
        too_big = _upload(c, headers, b"x" * (16 * 1024 * 1024 + 300 * 1024))
        assert too_big.status_code == 413
        assert "16 MiB" in too_big.json()["error"]["message"]
        assert store.owners_seen == []


def test_every_files_door_that_takes_a_body_is_bounded(settings) -> None:
    """These doors check the caller in their own bodies, so the walks in
    `test_front_door_paths.py` do not find them; this one reads their routes."""
    from fastapi.routing import APIRoute

    from eugene_plexus_gateway import files_door
    from eugene_plexus_gateway.app import create_app
    from tests.test_front_door_paths import _routes

    app = create_app(settings=settings)
    posts = {
        route.path
        for route in _routes(app.routes)
        if isinstance(route, APIRoute)
        and route.endpoint.__module__ == files_door.__name__
        and "POST" in route.methods
    }
    assert {"/v1/files", "/v1/uploads/{upload_id}/parts"} <= posts
    assert posts <= set(files_door.BODY_LIMITS)
    assert any(m.cls is files_door.FilesBodyLimit for m in app.user_middleware)
