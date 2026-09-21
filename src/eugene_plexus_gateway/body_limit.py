"""Bound the JSON body before parsing, including chunked requests."""

from __future__ import annotations

from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_BODY_BYTES = 16 * 1024 * 1024


class InferenceBodyLimit:
    def __init__(self, app: ASGIApp, *, paths: set[str], driver: bool = False) -> None:
        self.app, self.paths, self.driver = app, paths, driver

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] not in self.paths:
            await self.app(scope, receive, send)
            return
        body = bytearray()
        for key, value in scope.get("headers", []):
            if key == b"content-length" and value.isdigit():
                significant = value.lstrip(b"0")
                if len(significant) <= 8 and int(significant or b"0") <= MAX_BODY_BYTES:
                    continue
                await self._refuse(scope, receive, send)
                return
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            chunk = event.get("body", b"")
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                await self._refuse(scope, receive, send)
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

    async def _refuse(self, scope: Scope, receive: Receive, send: Send) -> None:
        message = "body: exceeds the 16 MiB request limit; resize or remove attachments."
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
