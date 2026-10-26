"""Stored responses and conversations at the Responses door (row 54).

OpenAI keeps a Responses request's answer when it says `store: true` -- its
default when `store` is absent -- and lets the next request continue it by
id (`previous_response_id`) or through a conversation (`/v1/conversations`).
This module is that state at this gateway's door. The objects themselves
live in the install's store (row 53), which keeps each for one owner and
ends it; this module never keeps anything of its own.

**Whose it is.** The owner is row 53's, from `files_door.owner_for`: the
person the root verified, else the client key's lineage, never anything the
request says. The owner's console session keeps nothing (design: "not
stored: any request whose owner is the console's session without a
person"), so its `store: true` is named as ignored and its
`previous_response_id` is refused.

**When it is kept.** After the answer has ended and before its last event
or its body is sent, so a client that sends its next request the moment it
reads `response.completed` finds it (the Agents SDK does). The response says
`store: true` only when it was kept. A store that cannot be reached, or an
owner who turned responses off, never stops an answer: the answer is given,
says `store: false`, and the ignored-settings header names `store` when the
caller sent it. A stream the client left is kept with what was sent, as
`incomplete`, by a task of its own (the request's task is being cancelled).

**Continuing.** `prepare` reads the chain or the conversation from the
store before the request is translated, and hands the translation the
conversation so far followed by this request's input: the model is given
what a client that sends it all would send. Only this request's own input is
kept with its answer, so a chain costs its links once each. A missing,
ended or other owner's id is OpenAI's one 400 `previous_response_not_found`.

**Nothing kept reaches a log or a metric.** A failure to keep is logged by
its status and the store's title, never with an item, an id that carries
text, or the store's words about content.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from . import files_door
from .files_door import DoorRefusal
from .responses import Refusal
from .store_client import StoreClient, StoreRefusal, StoreUnavailable

log = logging.getLogger(__name__)

router = APIRouter(tags=["stored"])

#: How long the store's word on whether responses are kept is believed.
PARTS_SECONDS = 15.0
#: A store slow to say whether it keeps responses is one that cannot keep them now.
PARTS_TIMEOUT = httpx.Timeout(3.0, connect=2.0)
#: Reading a chain and keeping an answer: seconds, not an upload's minutes.
CALL_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
#: OpenAI's: a conversation takes 20 items at a time.
MAX_ADDED_ITEMS = 20
#: A conversation door's JSON body, as every front door's.
MAX_BODY_BYTES = 16 * 1024 * 1024
OFF = "this install keeps no responses: the owner turned it off in Settings -> Store"
CONSOLE = (
    "the owner's console session keeps no responses or conversations (they are kept for a "
    "person or an app key)"
)
#: Answers kept after their client left, held so they finish.
_LATE: set[asyncio.Task[Any]] = set()


def _store(request: Request) -> StoreClient:
    return files_door._store(request)


def _unavailable(exc: StoreUnavailable) -> str:
    return (
        f"Eugene's store is not answering: {exc.cause}. Chat keeps working; stored responses "
        "and conversations wait for it."
    )


# --------------------------------------------------------------------------- #
# Is anything kept now
# --------------------------------------------------------------------------- #


@dataclass
class _Parts:
    at: float = float("-inf")
    on: bool | None = None
    cause: str | None = None


async def responses_on(request: Request) -> tuple[bool | None, str | None]:
    """Whether the store keeps responses now: True, False (the owner turned it
    off), or None when it cannot be asked (with the observed cause)."""
    parts: _Parts | None = getattr(request.app.state, "store_parts", None)
    if parts is None:
        parts = _Parts()
        request.app.state.store_parts = parts
    now = time.perf_counter()
    if now - parts.at < PARTS_SECONDS:
        return parts.on, parts.cause
    on: bool | None
    try:
        answer = await _store(request).call(
            "GET", "/v1/store/status", "install", time_limit=PARTS_TIMEOUT
        )
        said = ((answer or {}).get("parts") or {}).get("responses")
        on = said is True
        cause = None if on else OFF if said is False else "the install's store keeps none yet"
    except StoreUnavailable as exc:
        on, cause = None, exc.cause
    except StoreRefusal as exc:
        on, cause = None, f"it answered {exc.status} {exc.title}".strip()
    parts.at, parts.on, parts.cause = now, on, cause
    return on, cause


# --------------------------------------------------------------------------- #
# Items, as OpenAI keeps them
# --------------------------------------------------------------------------- #

_PREFIX = {
    "message": "msg",
    "function_call": "fc",
    "function_call_output": "fco",
    "reasoning": "rs",
}


def _item_id(kind: Any) -> str:
    return f"{_PREFIX.get(kind if isinstance(kind, str) else '', 'item')}_{uuid.uuid4().hex}"


def normalized(value: Any, *, taken: set[str] | None = None) -> list[dict[str, Any]]:
    """A request's input as OpenAI keeps it: a string as one user message, a
    message's string content as one text part, every item with an `id` of its
    own (a new one when it has none, or one already `taken`). An item that is
    not an object is left out: the translation refuses it before anything is
    kept."""
    if isinstance(value, str):
        value = [{"type": "message", "role": "user", "content": value}]
    if not isinstance(value, list):
        return []
    seen = set(taken or ())
    out: list[dict[str, Any]] = []
    for given in value:
        if not isinstance(given, Mapping):
            continue
        item = dict(given)
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind is not None:
            item["type"] = kind
        if kind == "message":
            content = item.get("content")
            if isinstance(content, str):
                if item.get("role") == "assistant":
                    part: dict[str, Any] = {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                else:
                    part = {"type": "input_text", "text": content}
                item["content"] = [part]
            item.setdefault("status", "completed")
        ident = item.get("id")
        if not isinstance(ident, str) or not ident or len(ident) > 128 or ident in seen:
            item["id"] = _item_id(kind)
        seen.add(item["id"])
        out.append(item)
    return out


# --------------------------------------------------------------------------- #
# A request's plan: what it continues, and whether its answer is kept
# --------------------------------------------------------------------------- #


@dataclass
class Plan:
    """What `prepare` decided for one request."""

    response_id: str
    #: The answer is kept (store: true, an owner, the store says it keeps).
    keep: bool = False
    owner: str | None = None
    store: StoreClient | None = None
    previous: str | None = None
    conversation: str | None = None
    #: The conversation affinity key: the conversation, or the chain's first response.
    chain: str | None = None
    #: This request's own input, as it is kept.
    own_input: list[dict[str, Any]] = field(default_factory=list)
    #: For the ignored-settings header.
    ignored: list[str] = field(default_factory=list)

    @property
    def writes(self) -> bool:
        return self.keep or self.conversation is not None

    def echo(self, echo: dict[str, Any]) -> None:
        """What the response object says of its state, before it is kept."""
        echo["store"] = self.keep
        echo["previous_response_id"] = self.previous
        if self.conversation is not None:
            echo["conversation"] = {"id": self.conversation}


def _conversation_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("id")
    if not isinstance(value, str) or not value:
        raise Refusal(
            "conversation: a conversation id, or an object with its id.", param="conversation"
        )
    return value


async def _owner(request: Request) -> str:
    try:
        return await files_door.owner_for(request, anthropic=False)
    except DoorRefusal as refusal:
        raise Refusal(
            refusal.message,
            status=refusal.status,
            error_type="invalid_request_error" if refusal.status < 500 else "service_unavailable",
            code=refusal.code,
        ) from refusal


def _context_refusal(exc: StoreRefusal, field_name: str, ident: str) -> Refusal:
    """The store's refusal of a chain or a conversation, in this door's words."""
    if exc.status == 404:
        if field_name == "previous_response_id":
            return Refusal(
                f"Previous response with id '{ident}' not found. A stored response is kept 30 "
                "days, for the key or person that made it.",
                param=field_name,
                code="previous_response_not_found",
            )
        return Refusal(f"Conversation with id '{ident}' not found.", status=404, param=field_name)
    if exc.status == 403 and exc.title == "Responses are off":
        return Refusal(
            f"{field_name}: {OFF}. Send the conversation in `input` instead.", param=field_name
        )
    if exc.status == 401 or exc.title == "Only the gateway":
        return Refusal(
            _unavailable(StoreUnavailable(f"it refused the gateway's credential ({exc.detail})")),
            status=503,
            error_type="service_unavailable",
            code="store_unavailable",
        )
    if exc.status in (409, 413):
        return Refusal(f"{field_name}: {exc.detail}", param=field_name)
    return Refusal(exc.detail, status=exc.status, param=field_name)


async def prepare(
    request: Request, raw: dict[str, Any], *, response_id: str, keeps: bool = True
) -> tuple[dict[str, Any], Plan]:
    """`raw` with the conversation so far before its input, and the plan for its
    answer. `keeps` is false on the doors that answer without keeping (the count,
    compaction). Raises `Refusal` in the Responses door's words."""
    store = _store(request)
    plan = Plan(response_id=response_id, store=store)
    flag = raw.get("store")
    if flag is not None and not isinstance(flag, bool):
        raise Refusal("store: must be true, false or null.", param="store")
    previous = raw.get("previous_response_id")
    if previous is not None and (not isinstance(previous, str) or not previous):
        raise Refusal("previous_response_id: a response id.", param="previous_response_id")
    conversation = _conversation_id(raw.get("conversation"))
    if previous is not None and conversation is not None:
        raise Refusal(
            "previous_response_id and conversation: send one or the other, not both.",
            param="conversation",
        )
    plan.previous, plan.conversation = previous, conversation
    context: dict[str, Any] | None = None
    if previous is not None or conversation is not None:
        field_name = "previous_response_id" if previous is not None else "conversation"
        ident = previous if previous is not None else conversation
        assert ident is not None
        plan.owner = await _owner(request)
        if plan.owner == "install":
            raise Refusal(
                f"{field_name}: {CONSOLE}. Send the conversation in `input`.", param=field_name
            )
        path = (
            f"/v1/responses/{files_door._safe(ident)}/context"
            if previous is not None
            else f"/v1/conversations/{files_door._safe(ident)}/context"
        )
        try:
            context = await store.call("GET", path, plan.owner, time_limit=CALL_TIMEOUT)
        except StoreUnavailable as exc:
            raise Refusal(
                _unavailable(exc),
                status=503,
                error_type="service_unavailable",
                code="store_unavailable",
            ) from exc
        except StoreRefusal as exc:
            raise _context_refusal(exc, field_name, ident) from exc
        if not isinstance(context, Mapping) or not isinstance(context.get("items"), list):
            raise Refusal(
                _unavailable(StoreUnavailable("its answer was not a conversation")),
                status=503,
                error_type="service_unavailable",
                code="store_unavailable",
            )
        plan.chain = conversation if conversation is not None else context.get("rootId")
        if previous is not None and context.get("hadInstructions") and not raw.get("instructions"):
            # OpenAI's rule: instructions are not carried along a chain.
            plan.ignored.append("previous_response_id.instructions")

    if keeps and flag is not False:
        on, _cause = await responses_on(request)
        if on:
            plan.owner = plan.owner or await _owner(request)
            plan.keep = plan.owner != "install"
        if not plan.keep and flag is True:
            plan.ignored.append("store")
        if plan.keep and plan.chain is None:
            # The first response of a chain to come: its own id is the key.
            plan.chain = response_id
    elif flag is True:
        plan.ignored.append("store")

    taken = {
        str(item.get("id"))
        for item in (context or {}).get("items", [])
        if isinstance(item, Mapping) and item.get("id")
    }
    if plan.writes:
        plan.own_input = normalized(raw.get("input"), taken=taken)
    expanded = {
        k: v for k, v in raw.items() if k not in ("store", "previous_response_id", "conversation")
    }
    if context is not None:
        given = raw.get("input")
        if isinstance(given, str):
            given = [{"type": "message", "role": "user", "content": given}]
        if isinstance(given, list):
            expanded["input"] = [*context["items"], *given]
    if plan.chain is not None:
        request.state.affinity_chain = plan.chain
    return expanded, plan


# --------------------------------------------------------------------------- #
# Keeping an answer
# --------------------------------------------------------------------------- #


async def keep(plan: Plan, response: dict[str, Any]) -> bool:
    """Keep `response` (the object as the client gets it) for its owner, and add
    its items to its conversation. True when it was kept; `response["store"]`
    is set to say so. Never raises: an answer is never refused for want of a
    store."""
    if not plan.writes or plan.store is None or plan.owner is None:
        response["store"] = False
        return False
    output = [item for item in response.get("output") or [] if isinstance(item, Mapping)]
    status = response.get("status")
    status = (
        status if status in ("completed", "incomplete", "failed", "cancelled") else "incomplete"
    )
    kept = False
    try:
        if plan.keep:
            await plan.store.call(
                "POST",
                "/v1/responses",
                plan.owner,
                json={
                    "id": plan.response_id,
                    "model": str(response.get("model") or ""),
                    "status": status,
                    **({"previousResponseId": plan.previous} if plan.previous else {}),
                    **({"conversationId": plan.conversation} if plan.conversation else {}),
                    "input": plan.own_input,
                    "output": output,
                    "response": {**response, "store": True},
                },
                time_limit=CALL_TIMEOUT,
            )
            kept = True
        else:
            # `store: false` in a conversation: its items are added all the same.
            items = [*plan.own_input, *output]
            for start in range(0, len(items), MAX_ADDED_ITEMS):
                await plan.store.call(
                    "POST",
                    f"/v1/conversations/{files_door._safe(plan.conversation or '')}/items",
                    plan.owner,
                    json={"items": items[start : start + MAX_ADDED_ITEMS]},
                    time_limit=CALL_TIMEOUT,
                )
    except StoreRefusal as exc:
        log.warning("an answer was not kept: the store answered %d %s", exc.status, exc.title)
    except StoreUnavailable:
        log.warning("an answer was not kept: the store is not answering")
    except Exception as exc:  # an answer is never refused for want of a store
        log.warning("an answer was not kept: %s", type(exc).__name__)
    response["store"] = kept
    return kept


def keep_later(plan: Plan, response: dict[str, Any]) -> None:
    """Keep the answer of a stream whose client left, in a task of its own: the
    request's task is being cancelled and can await nothing more."""
    if not plan.writes:
        return
    try:
        task = asyncio.get_running_loop().create_task(keep(plan, response))
    except RuntimeError:
        return
    _LATE.add(task)
    task.add_done_callback(_LATE.discard)


# --------------------------------------------------------------------------- #
# The doors: a stored response, its input items, and conversations
# --------------------------------------------------------------------------- #

Handler = Callable[[Request, str], Awaitable[Any]]


def _error(refusal: DoorRefusal) -> JSONResponse:
    return files_door.error_response(False, refusal)


async def _serve(request: Request, handler: Handler) -> Response:
    try:
        owner = await files_door.owner_for(request, anthropic=False)
        answer = await handler(request, owner)
    except DoorRefusal as refusal:
        return _error(refusal)
    except StoreUnavailable as exc:
        return _error(DoorRefusal(503, _unavailable(exc), code="store_unavailable"))
    if isinstance(answer, Response):
        return answer
    return JSONResponse(answer)


def _refused(exc: StoreRefusal, kind: str, ident: str, item: str | None = None) -> DoorRefusal:
    if exc.status == 404:
        if item is not None and exc.detail.startswith("No such item"):
            return DoorRefusal(404, f"Item with id '{item}' not found.", param="item_id")
        param = {"Response": "response_id", "Conversation": "conversation_id"}[kind]
        return DoorRefusal(404, f"{kind} with id '{ident}' not found.", param=param)
    if exc.status == 409 and exc.title == "Chain broken":
        return DoorRefusal(400, exc.detail)
    return files_door._refused(exc, kind=kind.lower(), ident=ident)


def _limit(request: Request) -> int:
    value = request.query_params.get("limit")
    if value is None:
        return 20
    try:
        limit = int(value)
    except ValueError:
        raise DoorRefusal(400, "limit: a whole number from 1 to 100.", param="limit") from None
    if not 1 <= limit <= 100:
        raise DoorRefusal(400, "limit: a whole number from 1 to 100.", param="limit")
    return limit


def _paging(request: Request) -> dict[str, Any]:
    order = request.query_params.get("order") or "desc"
    if order not in ("asc", "desc"):
        raise DoorRefusal(400, "order: asc or desc.", param="order")
    after = request.query_params.get("after")
    if after is not None and (not after or len(after) > 128):
        raise DoorRefusal(400, "after: an item id.", param="after")
    return {"limit": _limit(request), "order": order, "after": after}


def _list(page: Mapping[str, Any]) -> dict[str, Any]:
    data = list(page.get("data") or [])
    return {
        "object": "list",
        "data": data,
        "first_id": data[0].get("id") if data else None,
        "last_id": data[-1].get("id") if data else None,
        "has_more": bool(page.get("hasMore")),
    }


@router.get("/v1/responses/{response_id}")
async def retrieve_response(request: Request, response_id: str) -> Response:
    async def handler(request: Request, owner: str) -> Any:
        if request.query_params.get("stream", "").lower() == "true":
            raise DoorRefusal(
                400,
                "stream: a stored response is returned whole here; only a background response "
                "is replayed as a stream, and this gateway makes none.",
                param="stream",
            )
        try:
            kept = await _store(request).call(
                "GET",
                f"/v1/responses/{files_door._safe(response_id)}",
                owner,
                time_limit=CALL_TIMEOUT,
            )
        except StoreRefusal as exc:
            raise _refused(exc, "Response", response_id) from exc
        return kept.get("response") or {}

    return await _serve(request, handler)


@router.delete("/v1/responses/{response_id}")
async def delete_response(request: Request, response_id: str) -> Response:
    async def handler(request: Request, owner: str) -> Any:
        try:
            gone = await _store(request).call(
                "DELETE",
                f"/v1/responses/{files_door._safe(response_id)}",
                owner,
                time_limit=CALL_TIMEOUT,
            )
        except StoreRefusal as exc:
            raise _refused(exc, "Response", response_id) from exc
        return {"id": gone["id"], "object": "response.deleted", "deleted": True}

    return await _serve(request, handler)


@router.get("/v1/responses/{response_id}/input_items")
async def list_input_items(request: Request, response_id: str) -> Response:
    async def handler(request: Request, owner: str) -> Any:
        params = _paging(request)
        try:
            page = await _store(request).call(
                "GET",
                f"/v1/responses/{files_door._safe(response_id)}/input_items",
                owner,
                params=params,
                time_limit=CALL_TIMEOUT,
            )
        except StoreRefusal as exc:
            raise _refused(exc, "Response", response_id) from exc
        return _list(page)

    return await _serve(request, handler)


@router.post("/v1/responses/{response_id}/cancel")
async def cancel_response(response_id: str) -> Response:
    return _error(
        DoorRefusal(
            400,
            "Only a response made with `background: true` can be cancelled, and this gateway "
            "makes none: every response is answered while the request is open.",
            param="response_id",
        )
    )


async def _body(request: Request) -> dict[str, Any]:
    """The JSON object a conversation door was sent, at most 16 MiB."""
    declared = request.headers.get("content-length")
    if declared is not None and declared.strip().isdigit() and int(declared) > MAX_BODY_BYTES:
        raise DoorRefusal(413, "body: exceeds the 16 MiB request limit.", param="body")
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_BODY_BYTES:
            raise DoorRefusal(413, "body: exceeds the 16 MiB request limit.", param="body")
    try:
        body = json.loads(bytes(data) or b"null")
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise DoorRefusal(400, "body: must be a JSON object.", param="body")
    return body


def _items(value: Any, *, required: bool) -> list[dict[str, Any]]:
    if value is None and not required:
        return []
    if not isinstance(value, list) or (required and not value):
        raise DoorRefusal(400, "items: a list of items.", param="items")
    if len(value) > MAX_ADDED_ITEMS:
        raise DoorRefusal(
            400, f"items: at most {MAX_ADDED_ITEMS} at a time, as OpenAI takes them.", param="items"
        )
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise DoorRefusal(400, f"items[{index}]: must be an object.", param="items")
    return normalized(value)


def _conversation(kept: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": kept["id"],
        "object": "conversation",
        "created_at": kept["createdAt"],
        "metadata": kept.get("metadata") or {},
    }


def _writable(owner: str) -> None:
    if owner == "install":
        raise DoorRefusal(403, f"Not kept: {CONSOLE}.")


@router.post("/v1/conversations")
async def create_conversation(request: Request) -> Response:
    async def handler(request: Request, owner: str) -> Any:
        _writable(owner)
        body = await _body(request)
        unknown = sorted(set(body) - {"items", "metadata"})
        if unknown:
            raise DoorRefusal(400, f"{unknown[0]}: unsupported setting.", param=unknown[0])
        try:
            kept = await _store(request).call(
                "POST",
                "/v1/conversations",
                owner,
                json={
                    "metadata": body.get("metadata") or {},
                    "items": _items(body.get("items"), required=False),
                },
                time_limit=CALL_TIMEOUT,
            )
        except StoreRefusal as exc:
            raise _refused(exc, "Conversation", "") from exc
        return _conversation(kept)

    return await _serve(request, handler)


@router.get("/v1/conversations/{conversation_id}")
async def retrieve_conversation(request: Request, conversation_id: str) -> Response:
    async def handler(request: Request, owner: str) -> Any:
        try:
            kept = await _store(request).call(
                "GET",
                f"/v1/conversations/{files_door._safe(conversation_id)}",
                owner,
                time_limit=CALL_TIMEOUT,
            )
        except StoreRefusal as exc:
            raise _refused(exc, "Conversation", conversation_id) from exc
        return _conversation(kept)

    return await _serve(request, handler)


@router.post("/v1/conversations/{conversation_id}")
async def update_conversation(request: Request, conversation_id: str) -> Response:
    async def handler(request: Request, owner: str) -> Any:
        body = await _body(request)
        if "metadata" not in body or set(body) != {"metadata"}:
            raise DoorRefusal(400, "metadata: required, and the only field.", param="metadata")
        try:
            kept = await _store(request).call(
                "POST",
                f"/v1/conversations/{files_door._safe(conversation_id)}",
                owner,
                json={"metadata": body["metadata"]},
                time_limit=CALL_TIMEOUT,
            )
        except StoreRefusal as exc:
            raise _refused(exc, "Conversation", conversation_id) from exc
        return _conversation(kept)

    return await _serve(request, handler)


@router.delete("/v1/conversations/{conversation_id}")
async def delete_conversation(request: Request, conversation_id: str) -> Response:
    async def handler(request: Request, owner: str) -> Any:
        try:
            gone = await _store(request).call(
                "DELETE",
                f"/v1/conversations/{files_door._safe(conversation_id)}",
                owner,
                time_limit=CALL_TIMEOUT,
            )
        except StoreRefusal as exc:
            raise _refused(exc, "Conversation", conversation_id) from exc
        return {"id": gone["id"], "object": "conversation.deleted", "deleted": True}

    return await _serve(request, handler)


@router.get("/v1/conversations/{conversation_id}/items")
async def list_conversation_items(request: Request, conversation_id: str) -> Response:
    async def handler(request: Request, owner: str) -> Any:
        params = _paging(request)
        try:
            page = await _store(request).call(
                "GET",
                f"/v1/conversations/{files_door._safe(conversation_id)}/items",
                owner,
                params=params,
                time_limit=CALL_TIMEOUT,
            )
        except StoreRefusal as exc:
            raise _refused(exc, "Conversation", conversation_id) from exc
        return _list(page)

    return await _serve(request, handler)


@router.post("/v1/conversations/{conversation_id}/items")
async def create_conversation_items(request: Request, conversation_id: str) -> Response:
    async def handler(request: Request, owner: str) -> Any:
        _writable(owner)
        body = await _body(request)
        if set(body) != {"items"}:
            raise DoorRefusal(400, "items: required, and the only field.", param="items")
        try:
            page = await _store(request).call(
                "POST",
                f"/v1/conversations/{files_door._safe(conversation_id)}/items",
                owner,
                json={"items": _items(body["items"], required=True)},
                time_limit=CALL_TIMEOUT,
            )
        except StoreRefusal as exc:
            raise _refused(exc, "Conversation", conversation_id) from exc
        return _list(page)

    return await _serve(request, handler)


@router.get("/v1/conversations/{conversation_id}/items/{item_id}")
async def retrieve_conversation_item(
    request: Request, conversation_id: str, item_id: str
) -> Response:
    async def handler(request: Request, owner: str) -> Any:
        try:
            return await _store(request).call(
                "GET",
                f"/v1/conversations/{files_door._safe(conversation_id)}/items/"
                f"{_safe_item(item_id)}",
                owner,
                time_limit=CALL_TIMEOUT,
            )
        except StoreRefusal as exc:
            raise _refused(exc, "Conversation", conversation_id, item_id) from exc

    return await _serve(request, handler)


@router.delete("/v1/conversations/{conversation_id}/items/{item_id}")
async def delete_conversation_item(
    request: Request, conversation_id: str, item_id: str
) -> Response:
    async def handler(request: Request, owner: str) -> Any:
        try:
            kept = await _store(request).call(
                "DELETE",
                f"/v1/conversations/{files_door._safe(conversation_id)}/items/"
                f"{_safe_item(item_id)}",
                owner,
                time_limit=CALL_TIMEOUT,
            )
        except StoreRefusal as exc:
            raise _refused(exc, "Conversation", conversation_id, item_id) from exc
        return _conversation(kept)

    return await _serve(request, handler)


def _safe_item(ident: str) -> str:
    """An item id as one path segment: one of OpenAI's shapes, else a name that
    matches nothing (the store answers it as a missing item)."""
    if ident and len(ident) <= 128 and all(c.isalnum() or c in "-_" for c in ident):
        return ident
    return "not-an-id"


@router.api_route("/v1/responses/{rest:path}", methods=["GET", "POST", "DELETE"])
async def no_such_responses_path(request: Request, rest: str) -> Response:
    """Any other path under `/v1/responses`, in OpenAI's envelope rather than a
    framework's bare 404."""
    return _error(
        DoorRefusal(
            404,
            f"No such endpoint: {request.method} /v1/responses/{rest}.",
            param="response_id",
        )
    )
