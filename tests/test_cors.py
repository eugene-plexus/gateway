"""CORS on the front door -- and the sabotage check that it does not buffer.

Measured before any of this existed (2026-09-13, against the live
gateway): a browser's preflight on `/v1/chat/completions` was answered
`405 Method Not Allowed` with no `access-control-*` header, so no page
on another origin could use the OpenAI-compatible surface at all. These
tests pin the three properties `cors.py` claims: only the front-door
paths answer, the configuration is live, and a stream through the
middleware is still a stream.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.cors import FrontDoorCors
from eugene_plexus_gateway.settings import Settings
from tests.conftest import FakeDriverClient, FakeInstall, make_routing_table

ORIGIN = "http://192.168.16.75:8079"
OTHER = "http://evil.example"

PREFLIGHT = {
    "origin": ORIGIN,
    "access-control-request-method": "POST",
    "access-control-request-headers": "authorization,content-type",
}


@pytest.fixture
def cors_client(
    settings: Settings, fake_driver: FakeDriverClient, tmp_path: Path
) -> Iterator[TestClient]:
    """Auth ON, so a preflight that reached a route would 401.

    The default `client` fixture runs with auth disabled, which would let
    a preflight through the dependency layer and hide the one property
    that matters most: preflights carry no bearer, so they must be
    answered before auth or every browser is refused.
    """
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake_driver)
    app.state.auth_state = FakeInstall(tmp_path / "node").auth_state()
    with TestClient(app) as c:
        yield c


def _preflight(client: TestClient, path: str, **extra: str) -> Any:
    return client.options(path, headers={**PREFLIGHT, **extra})


# --------------------------------------------------------------------------- #
# the three paths, and nothing else
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", ["/v1/models", "/v1/chat/completions", "/v1/embeddings"])
def test_a_preflight_on_the_front_door_is_answered_before_auth(
    cors_client: TestClient, path: str
) -> None:
    r = _preflight(cors_client, path)
    assert r.status_code == 204, r.text
    assert r.headers["access-control-allow-origin"] == "*"
    assert "POST" in r.headers["access-control-allow-methods"]
    # Echoed, so an SDK's extra headers are admitted too.
    assert r.headers["access-control-allow-headers"] == "authorization,content-type"
    assert r.headers["access-control-max-age"]
    # Any origin: no Vary, since the answer does not depend on it.
    assert "vary" not in r.headers


def test_a_preflight_on_an_operator_path_gets_no_cors_at_all(cors_client: TestClient) -> None:
    for path in ["/v1/config", "/v1/admin/drivers", "/v1/metrics"]:
        r = _preflight(cors_client, path)
        assert "access-control-allow-origin" not in r.headers, path
        # Whatever the route says to OPTIONS, it is not a CORS answer.
        assert r.status_code != 204, (path, r.status_code)


def test_a_real_response_carries_the_header(cors_client: TestClient) -> None:
    # 401 -- no bearer -- and the header is still there, because the
    # browser has to be able to read the 401 to show it.
    r = cors_client.get("/v1/models", headers={"origin": ORIGIN})
    assert r.status_code == 401
    assert r.headers["access-control-allow-origin"] == "*"


def test_a_request_with_no_origin_is_untouched(cors_client: TestClient) -> None:
    r = cors_client.get("/v1/models")
    assert "access-control-allow-origin" not in r.headers


# --------------------------------------------------------------------------- #
# live configuration
# --------------------------------------------------------------------------- #


def test_narrowing_the_origins_takes_effect_without_restart(client: TestClient) -> None:
    patched = client.patch("/v1/config", json={"corsAllowedOrigins": [ORIGIN + "/"]})
    assert patched.status_code == 200, patched.text
    assert patched.json()["applied"] == ["corsAllowedOrigins"]
    assert patched.json()["requiresRestart"] is False

    # Listed (trailing slash and all): echoed, with Vary.
    r = _preflight(client, "/v1/chat/completions")
    assert r.status_code == 204
    assert r.headers["access-control-allow-origin"] == ORIGIN
    assert r.headers["vary"] == "origin"

    # Not listed: a 403 that says which key, not a bare refusal.
    r = _preflight(client, "/v1/chat/completions", origin=OTHER)
    assert r.status_code == 403
    body = r.json()
    assert body["title"] == "Origin not allowed"
    assert "corsAllowedOrigins" in body["detail"]
    assert OTHER in body["detail"]
    assert "access-control-allow-origin" not in r.headers

    # And the real request from the unlisted origin still runs -- the
    # server cannot stop the send, only the read -- with no header.
    r = client.get("/v1/models", headers={"origin": OTHER})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_turning_it_off_takes_effect_without_restart(client: TestClient) -> None:
    patched = client.patch("/v1/config", json={"corsEnabled": False})
    assert patched.status_code == 200
    assert patched.json()["requiresRestart"] is False

    r = _preflight(client, "/v1/models")
    assert r.status_code == 403
    assert "corsEnabled" in r.json()["detail"]
    r = client.get("/v1/models", headers={"origin": ORIGIN})
    assert "access-control-allow-origin" not in r.headers

    # Back on, same process.
    assert client.patch("/v1/config", json={"corsEnabled": True}).status_code == 200
    assert _preflight(client, "/v1/models").status_code == 204


def test_the_origin_list_is_validated_like_the_control_roots(client: TestClient) -> None:
    r = client.patch("/v1/config", json={"corsAllowedOrigins": [ORIGIN, ORIGIN]})
    assert r.status_code == 200
    rejected = r.json()["rejected"]
    assert rejected and rejected[0]["key"] == "corsAllowedOrigins"
    assert "duplicates" in rejected[0]["message"]

    r = client.patch("/v1/config", json={"corsAllowedOrigins": "not-a-list"})
    assert r.json()["rejected"][0]["message"].startswith("expected a list")


def test_the_schema_offers_both_fields(client: TestClient) -> None:
    schema = client.get("/v1/config/schema").json()
    by_key = {f["key"]: f for f in schema["fields"]}
    assert by_key["corsEnabled"]["valueType"] == "boolean"
    assert by_key["corsEnabled"]["default"] is True
    assert by_key["corsAllowedOrigins"]["valueType"] == "url_list"
    assert by_key["corsAllowedOrigins"]["default"] == []
    assert schema["categories"]["clients"]


# --------------------------------------------------------------------------- #
# a stream through the middleware is still a stream
# --------------------------------------------------------------------------- #


def test_a_streamed_completion_through_the_middleware_keeps_its_frames(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    fake_driver.responses = ["one two three four"]
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "Qwen3-30B-A3B-Q4_K_M",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers={"origin": ORIGIN},
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "*"
    frames = [
        json.loads(line[6:])
        for line in r.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    content = [f for f in frames if f["choices"][0]["delta"].get("content")]
    assert len(content) > 1


def test_body_chunks_are_forwarded_as_they_are_sent() -> None:
    """The sabotage check.

    A middleware that collects body messages and sends them at the end
    still delivers every byte, still passes the frame-count test above
    (the TestClient reads the whole body anyway), and has silently undone
    M10. So the inner app here refuses to emit its second chunk until the
    test has *observed* the first one leave the middleware. Buffering
    deadlocks; `wait_for` turns the deadlock into a failure.
    """
    first_forwarded = asyncio.Event()
    forwarded: list[Any] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"data: one\n\n", "more_body": True})
        # Only released once the test has seen "one" on the far side.
        await first_forwarded.wait()
        await send({"type": "http.response.body", "body": b"data: two\n\n", "more_body": False})

    async def send(message: Any) -> None:
        forwarded.append(message)
        if message["type"] == "http.response.body" and b"one" in message["body"]:
            first_forwarded.set()

    async def receive() -> Any:
        return {"type": "http.request", "body": b"", "more_body": False}

    class _State:
        config_store = None

    class _App:
        state = _State()

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(b"origin", ORIGIN.encode())],
        "app": _App(),
    }

    async def run() -> None:
        await asyncio.wait_for(FrontDoorCors(inner)(scope, receive, send), timeout=2)

    asyncio.run(run())

    start = forwarded[0]
    assert start["type"] == "http.response.start"
    assert (b"access-control-allow-origin", b"*") in start["headers"]
    bodies = [m["body"] for m in forwarded if m["type"] == "http.response.body"]
    assert bodies == [b"data: one\n\n", b"data: two\n\n"]
