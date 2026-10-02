"""A conversation goes back to the replica that holds its prompt (PC4).

**An engine can reuse a prompt only from its own cache**, so a conversation
that alternates between two replicas reads its history again on every
turn. Measured 2026-10-02 on two Llama 3.1 8B replicas: one Claude Code
session alternating reused 52.8% of its prompt tokens and took 6.0 s;
pinned to one replica, 75.3% and 3.3 s
(`specs/docs/acceptance/prompt-cache-measurement.md` §7).

**The key** is the client's own when it sends one -- `prompt_cache_key` on
chat and Responses (Codex sends its session id), Claude Code's `session_id`
inside `metadata.user_id` -- and otherwise a fingerprint of how the
conversation starts: the model, the first system message and the first
user message. That start does not change while a conversation grows by
appending, and two conversations that start alike share that much prompt,
so sending them to one replica is right rather than a collision.

**The table** remembers, per model and key, the replica a request was sent
to: a bounded LRU with an idle expiry, in memory only. Losing it (a
restart) costs one cold turn per conversation, nothing more.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any

#: A conversation idle this long has probably been evicted from the
#: engine's cache anyway; forgetting it lets the next turn be balanced.
TTL_SECONDS = 1800.0
#: Bounded, so a flood of one-off requests cannot grow it without end.
MAX_KEYS = 4096

HIT = "hit"
NEW = "new"
MOVED = "moved"


def _text(content: Any) -> str | None:
    content = getattr(content, "root", content)
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if hasattr(part, "model_dump"):
                part = part.model_dump(mode="json", exclude_none=True)
            if isinstance(part, dict):
                # An image or a file is identified by its type alone: the
                # bytes would make a fingerprint expensive and say nothing
                # a conversation's first words do not.
                out.append(
                    str(part.get("text") if part.get("type") == "text" else part.get("type"))
                )
        return "\n".join(out)
    return str(content)


def key_for(body: Any, explicit: str | None = None) -> str | None:
    """The conversation a chat-shaped request belongs to, or None."""
    if explicit:
        return f"k:{explicit}"
    own = getattr(body, "prompt_cache_key", None)
    if own:
        return f"k:{own}"
    system = user = None
    for message in getattr(body, "messages", None) or []:
        role = getattr(getattr(message, "role", None), "value", None)
        if role in ("system", "developer") and system is None and user is None:
            system = _text(message.content)
        elif role == "user":
            user = _text(message.content)
            break
    if user is None:
        return None
    digest = hashlib.sha256(json.dumps([getattr(body, "model", None), system, user]).encode())
    return "f:" + digest.hexdigest()[:32]


class AffinityTable:
    """Model and key -> the backend a request was last sent to."""

    def __init__(self, *, ttl: float = TTL_SECONDS, size: int = MAX_KEYS) -> None:
        self._ttl = ttl
        self._size = size
        self._entries: OrderedDict[tuple[str, str], tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, target: str, key: str) -> Any | None:
        with self._lock:
            entry = self._entries.get((target, key))
            if entry is None:
                return None
            backend, seen = entry
            if time.perf_counter() - seen > self._ttl:
                del self._entries[(target, key)]
                return None
            return backend

    def put(self, target: str, key: str, backend: Any) -> None:
        with self._lock:
            self._entries[(target, key)] = (backend, time.perf_counter())
            self._entries.move_to_end((target, key))
            while len(self._entries) > self._size:
                self._entries.popitem(last=False)

    def held(self, target: str, *, besides: str | None = None) -> dict[Any, int]:
        """Backend -> how many live conversations of `target` it holds (CB2).

        Every key seen inside its TTL counts, whether or not a request is in
        flight for it: a replica whose agents are all thinking between turns
        holds their prompts all the same. `besides` is left out, so a
        conversation is not counted against the replica it is being placed on.
        """
        now = time.perf_counter()
        counts: dict[Any, int] = {}
        with self._lock:
            for (entry_target, key), (backend, seen) in self._entries.items():
                if entry_target != target or key == besides or now - seen > self._ttl:
                    continue
                counts[backend] = counts.get(backend, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._entries)
