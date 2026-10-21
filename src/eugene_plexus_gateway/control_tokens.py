"""Break a model's special-token spellings in user and tool content (DI4).

`specs/docs/design/deepinfer-lessons.md`, DI4. llama-server renders a chat
template to one string and tokenizes it with special-token parsing on, so a
web page, a file a folder tool read, or a pasted document holding
`<|im_end|><|im_start|>system` reaches the model as a real end of turn and
a real system turn. Upstream knows (llama.cpp #28249, open); the engine's
template marks input but the server does not use the marking yet.

The gateway sits where people built their own workaround: in front of the
engine. Each exact spelling of the model's special tokens, as its own
vocabulary lists them (the Library, `GET /v1/models/{id}/special-tokens`),
is broken with a WORD JOINER (U+2060) after its first character, in user
and tool content only. The model reads the same text; the tokenizer no
longer finds the token in it.

The cost: a model asked to copy such text back copies the invisible
character too. `escapeControlTokens` turns it off for the install.

The spellings are read like profile defaults (`profiles.py`): through the
agent's proxy to the Library, by the model file's Library path, cached
last-known-good. A lookup that fails sends the request unescaped and says
so in the log once per model, rather than refusing it: the hole stays as it
was, and nothing that worked stops working.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from ._generated.driver_models import (
    GenerateRequest,
    Message,
    MessageContent1,
    Role,
    TextContentPart,
)
from ._http import internal_client

log = logging.getLogger(__name__)

#: Invisible, and not a character any tokenizer joins into a special token.
JOINER = "⁠"
_MAX_ENTRIES = 256
_FRESH_SECONDS = 300.0
_RETRY_SECONDS = 30.0
_ESCAPED_ROLES = frozenset({Role.user, Role.tool})


def compile_spellings(spellings: list[str]) -> re.Pattern[str] | None:
    """One pattern for every spelling, longest first, or None for none."""
    usable = sorted({s for s in spellings if len(s) >= 2}, key=lambda s: (-len(s), s))
    if not usable:
        return None
    return re.compile("|".join(re.escape(s) for s in usable))


def escape_text(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub(lambda m: m.group(0)[0] + JOINER + m.group(0)[1:], text)


def escape_request(request: GenerateRequest, pattern: re.Pattern[str]) -> GenerateRequest:
    """`request` with each spelling broken in user and tool content. The
    same object when nothing matched, so an ordinary request costs no copy."""
    changed = False
    messages: list[Message] = []
    for message in request.messages or []:
        updated = _escape_message(message, pattern)
        changed = changed or updated is not message
        messages.append(updated)
    if not changed:
        return request
    return request.model_copy(update={"messages": messages})


def _escape_message(message: Message, pattern: re.Pattern[str]) -> Message:
    if message.role not in _ESCAPED_ROLES or message.content is None:
        return message
    content = message.content
    if isinstance(content, str):
        escaped = escape_text(content, pattern)
        return message if escaped == content else message.model_copy(update={"content": escaped})
    parts: list[Any] = []
    changed = False
    for part in content.root:
        if isinstance(part, TextContentPart):
            escaped = escape_text(part.text, pattern)
            if escaped != part.text:
                part = part.model_copy(update={"text": escaped})
                changed = True
        parts.append(part)
    if not changed:
        return message
    return message.model_copy(update={"content": MessageContent1(root=parts)})


@dataclass(frozen=True)
class _Entry:
    pattern: re.Pattern[str] | None
    fetched_at: float | None
    retry_at: float = 0.0


class SpecialTokenSpellings:
    """Each model's special-token spellings, by Library path, from the Library."""

    def __init__(
        self,
        agent_url: str,
        service_token: str | None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._base = agent_url.rstrip("/") + "/api/proxy/library"
        self._clock = clock
        headers = {"Authorization": f"Bearer {service_token}"} if service_token else {}
        self._client = internal_client(
            headers=headers, timeout=httpx.Timeout(3.0, connect=1.0), transport=transport
        )
        self._cache: OrderedDict[str, _Entry] = OrderedDict()
        self._pending: dict[str, asyncio.Task[_Entry]] = {}
        self._said: set[str] = set()

    async def pattern(self, model_path: str | None) -> re.Pattern[str] | None:
        """The pattern for this model file; None when it has no spellings or
        none could be read (then the last one read is used while it lasts)."""
        if not model_path:
            return None
        now = self._clock()
        held = self._cache.get(model_path)
        if held is not None:
            self._cache.move_to_end(model_path)
            if held.fetched_at is not None and now - held.fetched_at < _FRESH_SECONDS:
                return held.pattern
            if now < held.retry_at:
                return held.pattern
        task = self._pending.get(model_path)
        if task is None:
            task = asyncio.create_task(self._refresh(model_path, held))
            self._pending[model_path] = task
            task.add_done_callback(lambda done: self._pending.pop(model_path, None))
        # One cancelled inference must not cancel the lookup another awaits.
        entry = await asyncio.shield(task)
        return entry.pattern

    async def _refresh(self, path: str, held: _Entry | None) -> _Entry:
        try:
            entry = _Entry(compile_spellings(await self._read(path)), self._clock())
            self._said.discard(path)
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            entry = _Entry(
                held.pattern if held else None,
                held.fetched_at if held else None,
                retry_at=self._clock() + _RETRY_SECONDS,
            )
            if path not in self._said:
                self._said.add(path)
                log.warning(
                    "could not read the special tokens of %s from the Library (%s: %s); its "
                    "requests go %s until it answers",
                    path,
                    type(exc).__name__,
                    exc,
                    "with the spellings read before" if entry.pattern else "unescaped",
                )
        self._cache[path] = entry
        self._cache.move_to_end(path)
        while len(self._cache) > _MAX_ENTRIES:
            self._cache.popitem(last=False)
        return entry

    async def _read(self, path: str) -> list[str]:
        response = await self._client.get(f"{self._base}/v1/models", params={"path": path})
        response.raise_for_status()
        models = response.json()["models"]
        if not isinstance(models, list) or len(models) > 1:
            raise ValueError("Library reverse lookup did not identify one model")
        if not models:
            return []
        model_id = models[0]["id"]
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("Library model id is absent")
        response = await self._client.get(
            f"{self._base}/v1/models/{quote(model_id, safe='')}/special-tokens"
        )
        if response.status_code == 404:
            # A Library from before DI4: nothing to break, said once.
            raise ValueError("this Library does not list special tokens (an older version)")
        response.raise_for_status()
        tokens = response.json()["tokens"]
        if not isinstance(tokens, list):
            raise ValueError("special tokens are not a list")
        return [t for t in tokens if isinstance(t, str)]

    async def aclose(self) -> None:
        pending = list(self._pending.values())
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await self._client.aclose()
