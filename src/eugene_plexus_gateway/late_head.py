"""A streamed response whose head is sent with its first frame, not before.

Starlette's `StreamingResponse` sends the head the moment the response starts,
before the body's generator has run once. A header that depends on what the
generator learns from its backend -- which settings the serving backend was
sent and does not honour (`ignored_settings`, 2026-10-12) -- has gone by then,
and a stream that has a header to give cannot give it.

This runs the generator to its first frame, adds what `late()` says, and sends
head and frame together. **It runs in the task that goes on to stream**, which
is why this is not a peek in the route: the request-collection scopes the
generators open have to be entered and left in one context.

**What it costs and what it does not.** The head now comes with the first
frame, so a client that waited for the head before reading waits for the first
frame instead. A client's read timeout covers both, and nothing is sent in the
body before that frame on the doors that use this, so the stream itself is
unchanged. A generator that fails before its first frame fails before the head,
as a failed request, and a client that leaves while it waits cancels it as it
cancels one that is streaming.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from starlette.responses import StreamingResponse
from starlette.types import Send

log = logging.getLogger(__name__)

_NO_FRAME = object()


class LateHeadStreamingResponse(StreamingResponse):
    """`StreamingResponse`, with `late()`'s headers added once the first frame exists.

    `late` is called after that frame and before the head, and may be called
    with a stream that has produced nothing to say (it returns `{}` then). A
    header it fails to build is left off: a diagnostic must not fail a stream.
    """

    def __init__(
        self, content: Any, *, late: Callable[[], Mapping[str, str]], **kwargs: Any
    ) -> None:
        super().__init__(content, **kwargs)
        self._late = late

    def _body(self, chunk: Any) -> bytes | memoryview:
        return chunk if isinstance(chunk, bytes | memoryview) else chunk.encode(self.charset)

    async def stream_response(self, send: Send) -> None:
        frames = aiter(self.body_iterator)
        first = await anext(frames, _NO_FRAME)
        try:
            for name, value in self._late().items():
                self.headers[name] = value
        except Exception:
            log.debug("could not add a late header to a streamed response", exc_info=True)
        await send(
            {"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers}
        )
        if first is not _NO_FRAME:
            await send({"type": "http.response.body", "body": self._body(first), "more_body": True})
        async for chunk in frames:
            await send({"type": "http.response.body", "body": self._body(chunk), "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})


__all__ = ["LateHeadStreamingResponse"]
