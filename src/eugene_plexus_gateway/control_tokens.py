"""Break a model's special-token spellings in what people and tools send (DI4).

`specs/docs/design/deepinfer-lessons.md`, DI4. llama-server renders a chat
template to one string and tokenizes it with special-token parsing on, so a
web page, a file a folder tool read, or a pasted document holding
`<|im_end|><|im_start|>system` reaches the model as a real end of turn and
a real system turn. Upstream knows (llama.cpp #28249, open); the engine's
template marks input but the server does not use the marking yet.

The gateway sits where people built their own workaround: in front of the
engine. Each exact spelling of the model's special tokens, as its own
vocabulary lists them (the Library, `GET /v1/models/{id}/special-tokens`),
is broken with a WORD JOINER (U+2060) after its first character. The model
reads the same text; the tokenizer no longer finds the token in it.

**What is broken:** user and tool content (string or text parts, and a
spelling split across two adjacent text parts); the arguments of tool calls
in the history; the descriptions in tool definitions (an MCP server's text
is third-party). **What is not:** system messages and assistant text. Those
are the client's and the model's own, and templates parse thinking and tool
markers inside assistant text, which a joiner would break.

**Which models:** local GGUF models the Library lists by the runtime's
model path. A cloud backend renders its own template; a model the Library
cannot answer for is sent as written, said in the log once per model, so
the setting never reads as protecting a model it does not.

The cost: a model asked to copy such text back copies the invisible
character too. `escapeControlTokens` turns it off for the install.
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
#: Text longer than this is scanned in a worker thread: an HTML page holding
#: a hundred thousand `<` costs a large vocabulary a measurable fraction of
#: a second, which must not stall every other request on the event loop.
_THREAD_CHARS = 64 * 1024


@dataclass(frozen=True)
class Spellings:
    """A model's spellings, arranged for a scan that costs one set lookup per
    candidate start and distinct length, not one comparison per spelling."""

    words: frozenset[str]
    lengths: tuple[int, ...]
    starts: re.Pattern[str]
    longest: int


def compile_spellings(spellings: list[str]) -> Spellings | None:
    """None for a model with nothing to break."""
    words = frozenset(s for s in spellings if len(s) >= 2)
    if not words:
        return None
    firsts = sorted({w[0] for w in words})
    return Spellings(
        words=words,
        lengths=tuple(sorted({len(w) for w in words}, reverse=True)),
        starts=re.compile("[" + "".join(re.escape(c) for c in firsts) + "]"),
        longest=max(len(w) for w in words),
    )


def escape_text(text: str, spellings: Spellings) -> str:
    """Each spelling broken after its first character, leftmost and longest
    first, as a regex alternation ordered by length would."""
    if not spellings.starts.search(text):
        return text
    out: list[str] = []
    done = 0
    for match in spellings.starts.finditer(text):
        at = match.start()
        if at < done:
            continue
        for length in spellings.lengths:
            word = text[at : at + length]
            if len(word) == length and word in spellings.words:
                out.append(text[done:at])
                out.append(word[0] + JOINER + word[1:])
                done = at + length
                break
    if not out:
        return text
    out.append(text[done:])
    return "".join(out)


def _crosses(left: str, right: str, spellings: Spellings) -> bool:
    """Whether a spelling starts in `left` and ends in `right`."""
    reach = spellings.longest - 1
    window = left[-reach:] + right[:reach]
    cut = len(left[-reach:])
    for start in range(cut):
        for length in spellings.lengths:
            if start + length > cut and window[start : start + length] in spellings.words:
                return True
    return False


def escape_request(request: GenerateRequest, spellings: Spellings) -> GenerateRequest:
    """`request` with the spellings broken where this module says. The same
    object when nothing matched, so an ordinary request costs no copy."""
    changed = False
    messages: list[Message] = []
    for message in request.messages or []:
        updated = _escape_message(message, spellings)
        changed = changed or updated is not message
        messages.append(updated)
    update: dict[str, Any] = {}
    if changed:
        update["messages"] = messages
    if request.tools:
        tools = [_escape_tool(tool, spellings) for tool in request.tools]
        if any(new is not old for new, old in zip(tools, request.tools, strict=True)):
            update["tools"] = tools
    return request.model_copy(update=update) if update else request


async def escape_request_async(request: GenerateRequest, spellings: Spellings) -> GenerateRequest:
    """`escape_request`, in a worker thread when the request is large."""
    size = sum(len(_plain(m)) for m in request.messages or [])
    if size > _THREAD_CHARS:
        return await asyncio.to_thread(escape_request, request, spellings)
    return escape_request(request, spellings)


def _plain(message: Message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return "".join(p.text for p in content.root if isinstance(p, TextContentPart))


def _escape_message(message: Message, spellings: Spellings) -> Message:
    update: dict[str, Any] = {}
    if message.role in _ESCAPED_ROLES and message.content is not None:
        content = message.content
        if isinstance(content, str):
            escaped = escape_text(content, spellings)
            if escaped != content:
                update["content"] = escaped
        else:
            parts = _escape_parts(list(content.root), spellings)
            if parts is not None:
                update["content"] = MessageContent1(root=parts)
    if message.toolCalls:
        calls = [_escape_call(call, spellings) for call in message.toolCalls]
        if any(new is not old for new, old in zip(calls, message.toolCalls, strict=True)):
            update["toolCalls"] = calls
    return message.model_copy(update=update) if update else message


def _escape_parts(parts: list[Any], spellings: Spellings) -> list[Any] | None:
    changed = False
    out: list[Any] = []
    for part in parts:
        if isinstance(part, TextContentPart):
            escaped = escape_text(part.text, spellings)
            previous = out[-1] if out else None
            if isinstance(previous, TextContentPart) and _crosses(
                previous.text, escaped, spellings
            ):
                # A template joins a message's text parts, so a spelling
                # split across two of them would form again: break it there.
                out[-1] = previous.model_copy(update={"text": previous.text + JOINER})
                changed = True
            if escaped != part.text:
                part = part.model_copy(update={"text": escaped})
                changed = True
        out.append(part)
    return out if changed else None


def _escape_call(call: dict[str, Any], spellings: Spellings) -> dict[str, Any]:
    function = call.get("function")
    if not isinstance(function, dict):
        return call
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        return call
    escaped = escape_text(arguments, spellings)
    if escaped == arguments:
        return call
    return {**call, "function": {**function, "arguments": escaped}}


def _escape_tool(tool: Any, spellings: Spellings) -> Any:
    function = tool.function
    description = function.description
    new_description = escape_text(description, spellings) if description else description
    parameters = function.parameters
    new_parameters = _escape_descriptions(parameters, spellings) if parameters else parameters
    if new_description == description and new_parameters is parameters:
        return tool
    return tool.model_copy(
        update={
            "function": function.model_copy(
                update={"description": new_description, "parameters": new_parameters}
            )
        }
    )


def _escape_descriptions(value: Any, spellings: Spellings) -> Any:
    """`description` and `title` strings anywhere in a JSON schema; names,
    types and enum values are left alone, because they are matched."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        changed = False
        for key, item in value.items():
            if key in ("description", "title") and isinstance(item, str):
                new = escape_text(item, spellings)
            else:
                new = _escape_descriptions(item, spellings)
            changed = changed or (new is not item and new != item)
            out[key] = new
        return out if changed else value
    if isinstance(value, list):
        items = [_escape_descriptions(item, spellings) for item in value]
        return items if any(a is not b for a, b in zip(items, value, strict=True)) else value
    return value


@dataclass(frozen=True)
class _Entry:
    spellings: Spellings | None
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
            headers=headers, timeout=httpx.Timeout(2.0, connect=1.0), transport=transport
        )
        self._cache: OrderedDict[str, _Entry] = OrderedDict()
        self._pending: dict[str, asyncio.Task[_Entry]] = {}
        self._said: set[str] = set()

    async def get(self, model_path: str | None) -> Spellings | None:
        """This model file's spellings; None when it has none or none could be
        read (then the last ones read are used while they last)."""
        if not model_path:
            return None
        now = self._clock()
        held = self._cache.get(model_path)
        if held is not None:
            self._cache.move_to_end(model_path)
            if held.fetched_at is not None and now - held.fetched_at < _FRESH_SECONDS:
                return held.spellings
            if now < held.retry_at:
                return held.spellings
        task = self._pending.get(model_path)
        if task is None:
            task = asyncio.create_task(self._refresh(model_path, held))
            self._pending[model_path] = task
            task.add_done_callback(lambda done: self._pending.pop(model_path, None))
        # One cancelled inference must not cancel the lookup another awaits.
        entry = await asyncio.shield(task)
        return entry.spellings

    def _say(self, path: str, why: str) -> None:
        if path not in self._said:
            self._said.add(path)
            log.warning("requests for %s go with control tokens unescaped: %s", path, why)

    async def _refresh(self, path: str, held: _Entry | None) -> _Entry:
        try:
            listed = await self._read(path)
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            entry = _Entry(
                held.spellings if held else None,
                held.fetched_at if held else None,
                retry_at=self._clock() + _RETRY_SECONDS,
            )
            if entry.spellings is None:
                self._say(
                    path,
                    f"the Library could not list its special tokens ({type(exc).__name__}: {exc})",
                )
        else:
            if listed is None:
                self._say(path, "the Library lists no model at that path")
                entry = _Entry(None, self._clock())
            else:
                entry = _Entry(compile_spellings(listed), self._clock())
                self._said.discard(path)
        self._cache[path] = entry
        self._cache.move_to_end(path)
        while len(self._cache) > _MAX_ENTRIES:
            self._cache.popitem(last=False)
        return entry

    async def _read(self, path: str) -> list[str] | None:
        response = await self._client.get(f"{self._base}/v1/models", params={"path": path})
        response.raise_for_status()
        models = response.json()["models"]
        if not isinstance(models, list) or len(models) > 1:
            raise ValueError("Library reverse lookup did not identify one model")
        if not models:
            return None
        model_id = models[0]["id"]
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("Library model id is absent")
        response = await self._client.get(
            f"{self._base}/v1/models/{quote(model_id, safe='')}/special-tokens"
        )
        if response.status_code == 404:
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
