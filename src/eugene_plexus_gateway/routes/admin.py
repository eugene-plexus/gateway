"""Admin endpoints: drivers (list, probe) and restart."""

from __future__ import annotations

import asyncio
import logging
import os

import httpx
from fastapi import APIRouter, HTTPException, Request, status

from .._generated.models import (
    BackendKind,
    DriverHealth,
    DriverProbeRequest,
    DriversInfo,
    Problem,
    RestartResult,
)
from ..driver_client import DriverClient, HttpDriverClient
from ..routing import RoutingTable

router = APIRouter(tags=["admin"])

log = logging.getLogger(__name__)

# Probe timeout: deliberately short. The UI's per-row Test button is an
# interactive affordance — operators want a quick yes/no, not a 3-minute
# wait on a hung URL. Real generation calls use the full
# `requestTimeoutSeconds` from config.
_PROBE_TIMEOUT_SECONDS = 10.0


async def _driver_health(client: DriverClient) -> DriverHealth:
    base_url = client.base_url
    try:
        info = await client.info()
        # info.backend is inference-driver.yaml's BackendKind; DriverHealth
        # expects gateway.yaml's BackendKind. Same wire values, distinct
        # generated classes — bridge via .value.
        backend = BackendKind(info.backend.value)
        return DriverHealth(
            name=client.name,
            reachable=True,
            url=base_url,  # type: ignore[arg-type]
            backend=backend,
            modelId=info.modelId,
            version=info.version,
        )
    except httpx.HTTPError as e:
        log.warning("driver %r at %s unreachable: %s", client.name, base_url, e)
        return DriverHealth(
            name=client.name,
            reachable=False,
            url=base_url,  # type: ignore[arg-type]
            error=str(e),
        )


@router.get("/v1/admin/drivers", response_model=DriversInfo)
async def list_drivers(request: Request) -> DriversInfo:
    """What the gateway currently sees behind each driver in the topology.

    Reads the last routing-table refresh rather than re-probing: the
    refresh already asked every driver `/v1/info`, and an admin view that
    fanned out again would report a different world from the one requests
    are actually routed against.
    """
    table: RoutingTable | None = getattr(request.app.state, "routing", None)
    if table is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=Problem(
                type="https://github.com/eugene-plexus/gateway#no-routing-table",
                title="No routing table",
                status=503,
                detail=(
                    "The gateway is starting up or in safe mode, so it has not "
                    "resolved any drivers yet."
                ),
                component="gateway",
            ).model_dump(exclude_none=True),
        )

    healths = table.as_driver_health()
    if not healths:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=Problem(
                type="https://github.com/eugene-plexus/gateway#no-drivers-in-topology",
                title="No drivers in the topology",
                status=503,
                detail=(
                    "The agent topology contains no inference-driver entries, "
                    "so there is nothing to route to. Add one via the agent's "
                    "POST /v1/components."
                ),
                component="gateway",
            ).model_dump(exclude_none=True),
        )

    if not any(h.reachable for h in healths):
        summary = "; ".join(f"{h.name}={h.url} ({h.error})" for h in healths)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=Problem(
                type="https://github.com/eugene-plexus/gateway#drivers-unreachable",
                title="No drivers reachable",
                status=503,
                detail=f"No driver in the topology is reachable. {summary}",
                component="gateway",
            ).model_dump(exclude_none=True),
        )

    return DriversInfo(drivers=healths)


@router.post("/v1/admin/drivers/probe", response_model=DriverHealth)
async def probe_driver(request: Request, body: DriverProbeRequest) -> DriverHealth:
    """Test-connect to a single backend URL without persisting it.

    Backs the UI's per-URL Test button in the drivers list editor —
    operators verify each backend in a slot's priority list is
    reachable before saving the topology. Builds a one-shot HTTP
    client, hits the URL's `/v1/info`, and returns the same
    `DriverHealth` shape the list endpoint uses.
    """
    url = str(body.url).rstrip("/")
    service_token = request.app.state.auth_state.service_token
    client = HttpDriverClient(
        # `name` is optional on a probe (the operator may be testing a
        # URL before naming the slot); fall back to a diagnostic label.
        name=body.name or "(probe)",
        base_url=url,
        timeout_seconds=_PROBE_TIMEOUT_SECONDS,
        service_token=service_token,
    )
    try:
        return await _driver_health(client)
    finally:
        await client.aclose()


# Long enough for the 202 response body to flush back to the client over
# a slow LAN, short enough that the operator doesn't sit waiting.
_RESTART_DELAY_MS = 500


@router.post("/v1/admin/restart", response_model=RestartResult, status_code=202)
async def restart() -> RestartResult:
    """Schedule a process exit so a supervisor can relaunch with new config.

    Mirrors the inference-driver restart endpoint. The gateway only
    re-reads `requiresRestart: true` config keys at startup; this is the
    UI's mechanism for completing a config-change flow.
    """
    log.warning("restart requested via /v1/admin/restart; exiting in %dms", _RESTART_DELAY_MS)

    loop = asyncio.get_event_loop()
    loop.call_later(_RESTART_DELAY_MS / 1000.0, lambda: os._exit(0))

    return RestartResult(
        scheduled=True,
        delayMs=_RESTART_DELAY_MS,
        message=(
            f"Process exiting in {_RESTART_DELAY_MS}ms. A supervisor (systemd, "
            "docker, deploy launcher, …) is expected to relaunch it; in v0.1 "
            "personal-use installs without one, relaunch manually."
        ),
    )
