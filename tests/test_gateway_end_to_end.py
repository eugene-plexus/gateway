"""End-to-end: the real app, the real routing table, real HTTP.

M0's gateway acceptance test. Everything the other tests inject is real
here — the lifespan builds an actual `RoutingTable`, which fetches an
actual `/v1/components`, probes an actual `/v1/info`, and a chat
completion goes over the wire to an actual `/v1/generate`.

The watchdog and the inference-driver are stand-ins: one threaded HTTP
server answering all three paths. The point is to prove OUR wiring, and
the contracts it speaks are the ones in specs.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

MODEL = "Qwen3-30B-A3B-Q4_K_M"


class _FakeInstall(BaseHTTPRequestHandler):
    """One server playing both the watchdog and an inference-driver.

    Collapsing them is fine because the gateway reaches both purely over
    HTTP and never assumes they are distinct hosts — which is itself
    worth knowing, since on a real multi-host install they are not on the
    same box either.
    """

    # Overridden per-server by the fixture with a fresh list, so tests
    # don't see each other's calls. ClassVar because it is deliberately
    # class-level state — the handler is instantiated per request.
    generate_calls: ClassVar[list[dict]] = []

    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/v1/components":
            self._send(
                200,
                {
                    "components": [
                        {
                            "name": "gateway",
                            "kind": "gateway",
                            "url": f"http://127.0.0.1:{self.server.server_port}",
                            "status": "running",
                        },
                        {
                            "name": "qwen-box",
                            "kind": "inference-driver",
                            "url": f"http://127.0.0.1:{self.server.server_port}",
                            "status": "running",
                        },
                    ]
                },
            )
        elif self.path == "/v1/info":
            self._send(
                200,
                {
                    "backend": "openai_compat_http",
                    "provider": "local",
                    "modelId": MODEL,
                    "capabilities": {"streaming": True, "maxContextTokens": 8192},
                    "version": "0.1.0-fake",
                },
            )
        else:
            self._send(404, {"error": "nope"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/v1/generate":
            type(self).generate_calls.append(payload)
            self._send(
                200,
                {
                    "content": "hello from the engine",
                    "finishReason": "stop",
                    "backend": "openai_compat_http",
                    "modelId": MODEL,
                    "usage": {
                        "promptTokens": 5,
                        "completionTokens": 4,
                        "totalTokens": 9,
                    },
                    "latencyMs": 12,
                },
            )
        else:
            self._send(404, {"error": "nope"})

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def fake_install() -> Iterator[tuple[str, list[dict]]]:
    """A running stand-in install. Yields (base_url, generate_calls)."""
    calls: list[dict] = []

    class Handler(_FakeInstall):
        generate_calls = calls

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def real_client(tmp_path: Path, fake_install: tuple[str, list[dict]]) -> Iterator[TestClient]:
    """A TestClient over the real app, with nothing injected.

    No `app.state.routing` — the lifespan builds the genuine table and
    resolves against the fake install.
    """
    base_url, _ = fake_install
    settings = Settings(config_file=tmp_path / "config.yaml", watchdog_url=base_url)
    with TestClient(create_app(settings=settings)) as c:
        yield c


def test_a_model_becomes_routable_without_any_gateway_config(
    real_client: TestClient,
) -> None:
    """The whole claim of the derived routing table: an engine appears in
    the topology and becomes routable. No config edit, no restart."""
    response = real_client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert [m["id"] for m in body["data"]] == [MODEL]
    assert body["data"][0]["owned_by"] == "local"
    assert body["data"][0]["x_eugene_plexus"]["drivers"] == ["qwen-box"]
    assert body["data"][0]["x_eugene_plexus"]["context_length"] == 8192


def test_a_completion_travels_the_whole_chain(
    real_client: TestClient, fake_install: tuple[str, list[dict]]
) -> None:
    """gateway -> routing table -> driver over real HTTP, and back as an
    OpenAI-shaped response."""
    _, generate_calls = fake_install

    response = real_client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello from the engine"
    assert body["usage"]["total_tokens"] == 9
    assert body["x_eugene_plexus"]["driver"] == "qwen-box"
    assert body["x_eugene_plexus"]["attempts"] == 1

    # The driver got a camelCase GenerateRequest with every param filled
    # in — the caller's temperature, and the install default for the
    # max-tokens it didn't send.
    assert len(generate_calls) == 1
    sent = generate_calls[0]
    assert sent["temperature"] == 0.2
    assert sent["maxTokens"] == 2048
    assert sent["messages"] == [{"role": "user", "content": "hi"}]


def test_admin_drivers_shows_the_resolved_driver(real_client: TestClient) -> None:
    response = real_client.get("/v1/admin/drivers")
    assert response.status_code == 200
    drivers = response.json()["drivers"]
    assert len(drivers) == 1
    assert drivers[0]["name"] == "qwen-box"
    assert drivers[0]["reachable"] is True
    assert drivers[0]["modelId"] == MODEL


def test_config_test_reports_the_reachable_driver(real_client: TestClient) -> None:
    """The Test button reads the topology fresh, so it should find the
    same driver and name what it serves."""
    response = real_client.post("/v1/config/test")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True, body
    assert "1 driver(s) reachable" in body["summary"]
    assert MODEL in body["summary"]


def test_streaming_travels_the_whole_chain(real_client: TestClient) -> None:
    response = real_client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = [
        line[len("data: ") :] for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert frames[-1] == "[DONE]"
    chunks = [json.loads(f) for f in frames[:-1]]
    assert (
        "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        == "hello from the engine"
    )
