"""Config protocol routes."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
from fastapi import APIRouter, Request

from .._generated.models import (
    ConfigDocument,
    ConfigSchema,
    ConfigTestRequest,
    ConfigTestResult,
    ConfigUpdateRequest,
    ConfigUpdateResult,
)
from .._http import internal_client
from ..config import ConfigStore, as_schema
from ..routing import RoutingTable

router = APIRouter(tags=["config"])


@router.get("/v1/config", response_model=ConfigDocument)
async def get_config(request: Request) -> ConfigDocument:
    store: ConfigStore = request.app.state.config_store
    return store.as_document()


@router.get("/v1/config/schema", response_model=ConfigSchema)
async def get_config_schema() -> ConfigSchema:
    return as_schema()


@router.patch("/v1/config", response_model=ConfigUpdateResult)
async def patch_config(request: Request, body: ConfigUpdateRequest) -> ConfigUpdateResult:
    store: ConfigStore = request.app.state.config_store
    return store.apply_patch(body)


@router.post("/v1/config/test", response_model=ConfigTestResult)
async def test_config(
    request: Request,
    body: ConfigTestRequest | None = None,
) -> ConfigTestResult:
    """Probe every driver the gateway can route to, hitting each `/v1/info`.

    Uses the saved config merged with the optional `overrides`; override
    values are NOT persisted. There is no driver list to test any more —
    the drivers come from the agent topology — so what this verifies
    is that the topology resolves and its entries answer, which is
    exactly what a chat turn depends on.
    """
    start = time.perf_counter()
    store: ConfigStore = request.app.state.config_store

    overrides: dict[str, Any] = {}
    if body and body.overrides:
        overrides = body.overrides.model_dump(exclude_none=True)

    def get(key: str) -> Any:
        return overrides[key] if key in overrides else store.get(key)

    timeout = float(get("requestTimeoutSeconds") or 30)
    settings = request.app.state.settings
    outbound = getattr(request.app.state, "outbound", None)

    # Read the topology fresh rather than off the routing table: the
    # point of a Test button is to check the world as it is now, which
    # may differ from the last refresh.
    probe_table = RoutingTable(
        agent_url=settings.agent_url,
        outbound=outbound,
        request_timeout_seconds=timeout,
    )
    try:
        entries = await probe_table.fetch_driver_entries()
        # Each probe carries the token for its own driver's machine.
        recipients = {node: probe_table.recipient_for(node) for node, _, _ in entries}
    finally:
        await probe_table.aclose()

    def elapsed() -> int:
        return int((time.perf_counter() - start) * 1000)

    if not entries:
        return ConfigTestResult(
            ok=False,
            component="gateway",
            latencyMs=elapsed(),
            error=(
                f"No inference-driver entries in the agent topology at "
                f"{settings.agent_url}, so there is nothing to route to. Add one "
                f"via the agent's POST /v1/components."
            ),
        )

    async def probe(
        client: httpx.AsyncClient, node: str | None, name: str, base_url: str
    ) -> tuple[str, str | None, str | None]:
        """Returns (name, error-or-None, modelId-or-None)."""
        headers = await outbound.headers(recipients[node]) if outbound is not None else {}
        try:
            response = await client.get(base_url.rstrip("/") + "/v1/info", headers=headers)
            response.raise_for_status()
            info = response.json()
        except Exception as e:
            return name, f"{name} ({base_url}/v1/info) — {e}", None
        model_id = info.get("modelId") if isinstance(info, dict) else None
        return name, None, model_id if isinstance(model_id, str) else None

    # ONE client for every probe, not one each. These run concurrently
    # under `gather`, so a per-probe client made an N-driver install pay
    # N certifi parses (~104 ms of synchronous CPU apiece) back to back
    # on the event loop -- inside an operator-facing request.
    async with internal_client(timeout=timeout) as client:
        results = await asyncio.gather(
            *(probe(client, node, name, url) for node, name, url in entries)
        )

    failures = [err for _, err, _ in results if err]
    if failures:
        return ConfigTestResult(
            ok=False,
            component="gateway",
            latencyMs=elapsed(),
            error="; ".join(failures),
        )

    models = sorted({model for _, _, model in results if model})
    unnamed = [name for name, _, model in results if not model]
    summary = f"{len(results)} driver(s) reachable in {elapsed()}ms"
    if models:
        summary += f"; serving {', '.join(models)}"
    if unnamed:
        # Reachable but not routable: no model id means no key to route
        # on. Worth saying, because everything else looks fine.
        summary += f"; {', '.join(unnamed)} reported no modelId and will not be routable"
    return ConfigTestResult(
        ok=True,
        component="gateway",
        latencyMs=elapsed(),
        summary=summary,
    )
