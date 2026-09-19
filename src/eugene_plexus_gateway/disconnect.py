"""Stop computing for a caller that has gone away (R2.5, review §6.2 #16).

**The problem is not one wasted request, it is three.** The OpenAI
Python SDK retries twice by default, so a client whose own deadline is
shorter than ours does not simply leave — it leaves and asks again, and
again. On a box slow enough for that to happen at all, three identical
generations end up queued for a caller that stopped listening after the
first, each holding the GPU for a full answer and each finishing into a
closed socket. Nothing in this repo looked: `is_disconnected` occurred
zero times.

**Streaming was already covered and non-streaming was not**, which is
why this is easy to miss by reading. `StreamingResponse` races the body
iterator against `http.disconnect` itself, so a stream in flight is
cancelled by Starlette. A plain `await` inside a route is not raced
against anything: the socket closes, and the coroutine runs to
completion against a `send` nobody will read.

**What "cancelled" has to mean here.** Abandoning the task is not
enough — it has to be cancelled *and awaited*, so the `CancelledError`
actually propagates into httpx, closes the connection to the driver, and
lets the driver's own watcher (the same module, one layer down) kill the
engine call. Returning before that happens leaves exactly the situation
this exists to remove.

Duplicated rather than shared: components share schemas, not code.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable

from starlette.requests import ClientDisconnect, Request

log = logging.getLogger(__name__)


class ClientGone(Exception):
    """The caller disconnected while its request was still being served."""


async def _watch(request: Request) -> None:
    """Block on the receive channel until `http.disconnect` arrives.

    **Not `Request.is_disconnected()`, and this was measured rather than
    chosen.** That method peeks at the channel inside an
    already-cancelled `anyio.CancelScope`, which only behaves inside
    anyio's own task tree; called from the raw `asyncio.Task` this
    module creates, it wedges -- the whole suite hung on the first route
    test that reached it. What is here instead is what Starlette's own
    `StreamingResponse.listen_for_disconnect` does: one await, no
    polling, and cancelling it is an ordinary task cancellation.

    The body is read first so this cannot swallow it. FastAPI has
    already parsed it by the time a route's body parameter exists, and
    `Request.body()` caches, so on the real path this is a no-op -- but
    an endpoint added later that takes no body would otherwise lose its
    first `http.request` message to this loop.
    """
    try:
        await request.body()
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return
    except ClientDisconnect:
        # The body itself was cut short: the caller left mid-upload.
        # Same answer, and it has to be caught here or it becomes a
        # 500 raised by the watcher instead of a cancelled backend.
        return


async def serve_while_connected[T](
    request: Request,
    work: Awaitable[T],
    *,
    what: str = "request",
) -> T:
    """Await `work`, cancelling it if the client leaves first.

    Raises `ClientGone` when the client left. Anything `work` itself
    raises propagates untouched, so the routes' existing `DriverError` /
    `httpx` handling is unchanged.
    """
    task: asyncio.Task[T] = asyncio.ensure_future(work)
    watcher = asyncio.ensure_future(_watch(request))
    try:
        done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        # This handler is itself being torn down (the server is stopping,
        # or Starlette cancelled us). Take the backend call with us —
        # otherwise it outlives the thing that started it.
        task.cancel()
        watcher.cancel()
        raise
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher

    if task in done:
        return task.result()

    task.cancel()
    # Awaited, not merely cancelled: this is the line that actually
    # closes the socket to the backend. A version that returned here
    # would pass a test asserting the route came back and change nothing
    # about what the GPU is doing.
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task
    log.info("client disconnected during %s; the backend call was cancelled", what)
    raise ClientGone(what)


__all__ = ["ClientGone", "serve_while_connected"]
