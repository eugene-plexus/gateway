"""Adapt the shared, vendored body limiter to Gemini's parameterized paths."""

from __future__ import annotations

from starlette.types import ASGIApp, Receive, Scope, Send

from . import door_paths, gemini
from .body_limit import InferenceBodyLimit


class _GeminiLimit(InferenceBodyLimit):
    async def _refuse(self, scope: Scope, receive: Receive, send: Send, limit: int) -> None:
        await gemini.error_response(
            413, "body: exceeds the 16 MiB request limit; resize or remove attachments."
        )(scope, receive, send)


class GeminiBodyLimit:
    """Use the shared limiter unchanged, with the actual matched request path.

    Each limiter is request-local: adding a dynamic path to a shared set would
    retain arbitrary model names. Other doors retain their existing limiter.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and door_paths.matches(scope["path"], gemini.BODY_PATHS):
            await _GeminiLimit(self.app, paths={scope["path"]})(scope, receive, send)
        else:
            await self.app(scope, receive, send)
