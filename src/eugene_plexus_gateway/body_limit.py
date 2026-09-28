"""Bound the JSON body before parsing, including chunked requests."""

from __future__ import annotations

from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_BODY_BYTES = 16 * 1024 * 1024


class InferenceBodyLimit:
    """`paths` are bounded at `MAX_BODY_BYTES`; `limits` names a path with a
    limit of its own (P3b: an audio upload is bigger than any JSON body)."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        paths: set[str],
        driver: bool = False,
        limits: dict[str, int] | None = None,
    ) -> None:
        self.app, self.paths, self.driver = app, paths, driver
        self.limits = limits or {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path")
        if scope["type"] != "http" or (path not in self.paths and path not in self.limits):
            await self.app(scope, receive, send)
            return
        limit = self.limits.get(path, MAX_BODY_BYTES)
        body = bytearray()
        for key, value in scope.get("headers", []):
            if key == b"content-length" and value.isdigit():
                significant = value.lstrip(b"0")
                if len(significant) <= 10 and int(significant or b"0") <= limit:
                    continue
                await self._refuse(scope, receive, send, limit)
                return
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            chunk = event.get("body", b"")
            if len(body) + len(chunk) > limit:
                await self._refuse(scope, receive, send, limit)
                return
            body.extend(chunk)
            if not event.get("more_body", False):
                break
        delivered = False

        async def replay() -> Message:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)

    async def _refuse(self, scope: Scope, receive: Receive, send: Send, limit: int) -> None:
        # The documented number for the shared limit, whatever a test patches
        # the byte count to; a path's own limit is named from its bytes.
        size = f"{limit // (1024 * 1024)} MiB" if scope.get("path") in self.limits else "16 MiB"
        message = f"body: exceeds the {size} request limit; resize or remove attachments."
        payload: dict[str, Any]
        if self.driver:
            payload = {"detail": {"title": "Request too large", "status": 413, "detail": message}}
        elif scope["path"] == "/v1/messages":
            payload = {"type": "error", "error": {"type": "request_too_large", "message": message}}
        else:
            payload = {
                "error": {"type": "invalid_request_error", "param": "body", "message": message}
            }
        await JSONResponse(payload, status_code=413)(scope, receive, send)
