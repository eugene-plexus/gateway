"""GET /v1/conversations/{id}."""

from __future__ import annotations

import logging
from uuid import UUID

import httpx
from fastapi import APIRouter, HTTPException, Request, status

from .._generated.models import Conversation, Problem
from ..memory import MemoryClient

router = APIRouter(tags=["chat"])

log = logging.getLogger(__name__)


@router.get("/v1/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(request: Request, conversation_id: UUID) -> Conversation:
    memory: MemoryClient = request.app.state.memory
    try:
        conversation = await memory.get(conversation_id)
    except httpx.HTTPError as e:
        log.warning("memory service unreachable: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=Problem(
                type="https://github.com/eugene-plexus/gateway#memory-error",
                title="Memory service error",
                status=502,
                detail=f"Memory service at {request.app.state.memory_url} is unreachable: {e}",
                component="gateway",
            ).model_dump(exclude_none=True),
        ) from e

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=Problem(
                type="https://github.com/eugene-plexus/gateway#conversation-not-found",
                title="Conversation not found",
                status=404,
                detail=f"No conversation with id {conversation_id}.",
                component="gateway",
            ).model_dump(exclude_none=True),
        )
    return conversation
