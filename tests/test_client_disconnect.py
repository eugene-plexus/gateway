"""R2.5 — nothing cancels the backend when the client gives up.

The reproduction, written before the fix (roadmap §1).

**Why it compounds.** The OpenAI Python SDK retries twice by default, so
a client whose own deadline is shorter than ours does not simply go
away: it goes away and asks again, and again. On a box slow enough for
that to happen at all, three identical generations end up queued for a
caller that left after the first — each holding the GPU for the full
answer, each finishing into a closed socket. `is_disconnected` occurs
zero times in this repo.

The assertion is not *the route returned*; it is **the backend call was
cancelled**, because a check that only watches the route pass against a
fix that abandons the task without cancelling it. That is R1.4's lesson
(`project_r14_stream_bookkeeping`): assert something the defect can
actually prevent.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from eugene_plexus_gateway._generated.models import EmbeddingRequest
from eugene_plexus_gateway.disconnect import _watch
from eugene_plexus_gateway.routes.inference import create_chat_completion, create_embedding

from .conftest import FakeDriverClient, make_routing_table


class _Hang:
    """A backend that never answers, and remembers how it ended."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = False
        self.finished = False

    async def run(self) -> Any:
        self.entered.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        self.finished = True


_SCOPE: dict[str, Any] = {
    "type": "http",
    "asgi": {"version": "3.0", "spec_version": "2.3"},
    "http_version": "1.1",
    "method": "POST",
    "scheme": "http",
    "path": "/v1/chat/completions",
    "raw_path": b"/v1/chat/completions",
    "query_string": b"",
    "root_path": "",
    "headers": [],
    "client": ("127.0.0.1", 5555),
    "server": ("127.0.0.1", 8080),
}
"""The scope uvicorn builds, minus the app. `spec_version` 2.3 is what
uvicorn really advertises, and it decides which branch of Starlette's
`StreamingResponse` runs -- so it is part of the subject, not padding."""


def _request(app: FastAPI, gone: asyncio.Event) -> Request:
    """A real `Request` whose receive channel reports a disconnect once
    `gone` is set — the shape uvicorn delivers."""

    sent_body = False

    async def receive() -> dict[str, Any]:
        # The shape uvicorn delivers: the body, then nothing until the
        # socket closes. FastAPI consumes the first message before the
        # route body parameter exists; the watcher blocks on the second.
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {
                "type": "http.request",
                "body": b'{"model":"m","messages":[{"role":"user","content":"hi"}]}',
                "more_body": False,
            }
        await gone.wait()
        return {"type": "http.disconnect"}

    return Request(_SCOPE | {"app": app}, receive)


async def _drive(app: FastAPI, hang: _Hang, call: Any) -> Any:
    gone = asyncio.Event()
    request = _request(app, gone)
    route = asyncio.ensure_future(call(request))
    await asyncio.wait_for(hang.entered.wait(), timeout=5)
    gone.set()
    # **Bounded, and the bound is an assertion rather than a convenience.**
    # A version that abandons the task instead of cancelling it blocks
    # here on `await task` forever -- and a generous `wait_for` HIDES
    # that: its own cancellation reaches the route, is swallowed by the
    # `suppress`, cancels the backend on the way past, and the route
    # returns a perfectly good 499. Every assertion then passes, five
    # seconds late. So the deadline is short and overrunning it is the
    # failure, not the cleanup.
    started = asyncio.get_running_loop().time()
    try:
        answer = await asyncio.wait_for(route, timeout=2)
    except asyncio.TimeoutError:  # noqa: UP041 - 3.12 compatibility
        route.cancel()
        pytest.fail("the route kept waiting on a backend nobody is listening to")
    except asyncio.CancelledError:
        return None
    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 1.5, f"the route took {elapsed:.1f}s to notice the client had gone"
    return answer


async def test_a_closed_tab_cancels_the_completion(app: FastAPI) -> None:
    hang = _Hang()
    driver = FakeDriverClient(name="qwen-box", model_id="m")
    driver.generate_hook = hang.run
    app.state.routing = make_routing_table(driver)

    answer = await _drive(app, hang, create_chat_completion)

    assert hang.cancelled, "the backend was left computing for a client that had gone"
    assert not hang.finished
    # 499, and deliberately not 500: "the caller left" and "we failed"
    # must not read the same in an access log or in /v1/metrics.
    assert answer is not None and answer.status_code == 499


async def test_a_closed_tab_cancels_the_embedding(app: FastAPI) -> None:
    hang = _Hang()
    driver = FakeDriverClient(name="qwen-box", model_id="m", supports_embeddings=True)
    driver.embed_hook = hang.run
    app.state.routing = make_routing_table(driver)

    body = EmbeddingRequest(model="m", input="hi")
    answer = await _drive(app, hang, lambda r: create_embedding(r, body))

    assert hang.cancelled
    assert not hang.finished
    assert answer is not None and answer.status_code == 499


async def test_a_client_that_stays_gets_its_answer(app: FastAPI) -> None:
    """The control: the watcher must not cancel a live request."""
    driver = FakeDriverClient(name="qwen-box", model_id="m")
    driver.responses = ["the answer"]
    app.state.routing = make_routing_table(driver)

    gone = asyncio.Event()  # never set
    request = _request(app, gone)
    result = await asyncio.wait_for(create_chat_completion(request), timeout=5)

    # A serialised body since 2026-09-23, so that fields which say nothing
    # (a non-reasoning model's `reasoning_content`) are absent, not null.
    assert json.loads(bytes(result.body))["choices"][0]["message"]["content"] == "the answer"


def test_the_watcher_does_not_hang_a_plain_request(client) -> None:  # type: ignore[no-untyped-def]
    """The trap this cost, pinned.

    The first version used `Request.is_disconnected()`, which peeks at
    the receive channel inside an already-cancelled `anyio.CancelScope`.
    That only behaves inside anyio's own task tree, and the watcher is a
    raw `asyncio.Task` -- so under `TestClient`, whose receive blocks
    until the response is complete, the peek never came back and the
    whole suite wedged on the first route test that reached it. Nothing
    asserted "an ordinary request still returns", so the failure looked
    like an infrastructure problem rather than like this change.
    """
    response = client.post(
        "/v1/chat/completions",
        json={"model": "Qwen3-30B-A3B-Q4_K_M", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code < 500, response.text


async def test_the_watcher_returns_only_for_a_DISCONNECT() -> None:
    """`_watch` must read the message type, not merely that a message came.

    The sabotage pass found this missing. Every other check here feeds a
    channel that goes quiet after the body and then delivers a
    disconnect, so a watcher that treated ANY message as "the client is
    gone" was never contradicted — the one message it would have
    misread never arrived. Here one does.
    """
    delivered: list[dict[str, Any]] = [
        {"type": "http.request", "body": b"{}", "more_body": True},
        {"type": "http.request", "body": b"", "more_body": False},
        {"type": "http.request", "body": b"", "more_body": False},
    ]
    seen = asyncio.Event()

    async def receive() -> dict[str, Any]:
        if delivered:
            message = delivered.pop(0)
            if not delivered:
                seen.set()
            return message
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    request = Request(_SCOPE | {"app": None}, receive)
    watcher = asyncio.ensure_future(_watch(request))
    await asyncio.wait_for(seen.wait(), timeout=5)
    await asyncio.sleep(0.05)
    assert not watcher.done(), "a message that is not a disconnect was read as one"
    watcher.cancel()


async def test_the_watcher_does_return_for_a_disconnect() -> None:
    """The twin, so the test above cannot pass by never returning."""

    async def receive() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    request = Request(_SCOPE | {"app": None}, receive)
    await asyncio.wait_for(_watch(request), timeout=5)
