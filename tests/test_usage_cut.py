"""Row 40 review B1, S3, S4: an answer cut short still counts, a release the
authority did not take is asked again, and a count the backend never gave is the
gateway's estimate. In-process only: the fake authority and driver; no sockets."""

import asyncio
import json

import httpx
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import Usage
from tests.conftest import install_snapshot
from tests.test_admission import Authority, body
from tests.test_admission import setup as setup


def _run(app, headers, *, hang_up_after: int | None):
    """One streamed chat through the raw ASGI app. The client hangs up after it has
    read `hang_up_after` content deltas (None: it reads to the end)."""
    payload = json.dumps(body("allowed", stream=True)).encode()
    read: list[str] = []

    async def go():
        gone = asyncio.Event()
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": payload, "more_body": False}
            await gone.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] != "http.response.body":
                return
            for line in message.get("body", b"").decode().splitlines():
                if line.startswith("data: {"):
                    delta = (json.loads(line[6:]).get("choices") or [{}])[0].get("delta") or {}
                    if delta.get("content"):
                        read.append(delta["content"])
            if hang_up_after is not None and len(read) >= hang_up_after:
                gone.set()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
                (b"authorization", headers["Authorization"].encode()),
            ],
            "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80),
            "app": app,
        }
        await app(scope, receive, send)

    return go, read


def _slow(driver, seconds=0.05):
    """A backend that takes a moment for each token, as every real one does."""
    plain = driver.stream

    async def stream(request):
        async for event in plain(request):
            await asyncio.sleep(seconds)
            yield event

    driver.stream = stream


def _releases(authority):
    return [c for c in authority.calls if c["action"] == "release"]


TWELVE = "one two three four five six seven eight nine ten eleven twelve"


def test_a_stream_read_to_the_end_counts_the_backends_own_numbers(setup):
    app, authority, allowed, _, headers, _ = setup
    allowed.usage = Usage(promptTokens=7000, completionTokens=12, totalTokens=7012)
    allowed.responses = [TWELVE]
    _slow(allowed)
    install_snapshot(app.state.routing, allowed)
    with TestClient(app) as c:
        go, _ = _run(app, headers, hang_up_after=None)
        c.portal.call(go)
    (release,) = _releases(authority)
    assert release["usage"]["promptTokens"] == 7000
    assert release["usage"]["completionTokens"] == 12
    assert "estimated" not in release["usage"]


def test_a_stream_the_caller_hangs_up_on_is_counted_as_estimated(setup):
    """B1: closing before the last frame must not make a request free."""
    app, authority, allowed, _, headers, _ = setup
    allowed.usage = Usage(promptTokens=7000, completionTokens=12, totalTokens=7012)
    allowed.responses = [TWELVE]
    _slow(allowed)
    install_snapshot(app.state.routing, allowed)
    with TestClient(app) as c:
        go, read = _run(app, headers, hang_up_after=3)
        c.portal.call(go)
    assert 3 <= len(read) < 12
    (release,) = _releases(authority)
    used = release["usage"]
    assert used["model"] == "allowed" and used["estimated"] is True
    assert used["completionTokens"] >= len(read)
    assert used["promptTokens"] > 0


def test_a_backend_that_fails_after_its_first_token_is_counted(setup):
    app, authority, allowed, _, headers, _ = setup
    allowed.usage = Usage(promptTokens=50, completionTokens=12, totalTokens=62)
    allowed.responses = [TWELVE]
    allowed.stream_error_after = 2
    allowed.stream_error = httpx.ReadError("upstream gone")
    install_snapshot(app.state.routing, allowed)
    with TestClient(app) as c:
        answer = c.post("/v1/chat/completions", json=body("allowed", stream=True), headers=headers)
        assert answer.status_code == 200
    (release,) = _releases(authority)
    assert release["usage"]["estimated"] is True and release["usage"]["completionTokens"] >= 2


def test_a_backend_that_fails_before_any_output_counts_nothing(setup):
    app, authority, allowed, _, headers, _ = setup
    allowed.responses = [TWELVE]
    allowed.stream_error_after = 0
    allowed.stream_error = httpx.ReadError("upstream gone")
    install_snapshot(app.state.routing, allowed)
    with TestClient(app) as c:
        c.post("/v1/chat/completions", json=body("allowed", stream=True), headers=headers)
    (release,) = _releases(authority)
    assert "usage" not in release


def test_a_count_the_backend_did_not_report_is_the_gateways_estimate(setup):
    """S4: no usage from the backend is not zero."""
    app, authority, allowed, _, headers, _ = setup
    allowed.usage = None
    allowed.responses = [TWELVE]
    with TestClient(app) as c:
        answer = c.post("/v1/chat/completions", json=body("allowed"), headers=headers)
        assert answer.status_code == 200
    (release,) = _releases(authority)
    used = release["usage"]
    assert used["estimated"] is True and used["promptTokens"] > 0


class Flaky(Authority):
    """An authority whose first two releases fail."""

    def __init__(self):
        super().__init__()
        self.failures = 2

    def _handle(self, request):
        if request.url.path.endswith("/admission"):
            payload = json.loads(request.content)
            if payload["action"] == "release" and self.failures:
                self.failures -= 1
                self.calls.append({**payload, "failed": True})
                return httpx.Response(503, json={"detail": "busy"})
        return super()._handle(request)


def test_a_release_the_authority_did_not_take_is_asked_again_and_said(setup, caplog, monkeypatch):
    """S3: never silently dropped: logged, and retried while the lease can be held."""
    from eugene_plexus_gateway import admission

    app, _, allowed, _, headers, _ = setup
    flaky = Flaky()
    app.state.client_key_guard = flaky.as_guard(ttl_seconds=0)
    allowed.usage = Usage(promptTokens=7, completionTokens=3, totalTokens=10)
    monkeypatch.setattr(admission, "RELEASE_RETRY_SECONDS", 5.0)
    with TestClient(app) as c:
        assert (
            c.post("/v1/chat/completions", json=body("allowed"), headers=headers).status_code == 200
        )

        async def settle():
            for _ in range(100):
                if any(x["action"] == "release" and not x.get("failed") for x in flaky.calls):
                    return
                await asyncio.sleep(0.1)

        c.portal.call(settle)
    releases = [x for x in flaky.calls if x["action"] == "release"]
    assert [bool(x.get("failed")) for x in releases] == [True, True, False]
    assert releases[-1]["usage"]["promptTokens"] == 7
    assert "did not reach the admission authority" in caplog.text
