"""Admin endpoints: drivers (list, probe) and restart."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException, Request, status

from .. import drain
from .._generated.models import (
    BackendKind,
    DriverHealth,
    DriverProbeRequest,
    DriversInfo,
    Problem,
    RestartResult,
    RoutingTableView,
    RuntimeSwitchRequest,
)
from .._generated.models import (
    GatewayDrain as GatewayDrainView,
)
from ..drain import GatewayDrain
from ..driver_client import DriverClient, HttpDriverClient
from ..routing import RoutingTable

router = APIRouter(tags=["admin"])

log = logging.getLogger(__name__)

# Probe timeout: deliberately short. The UI's per-row Test button is an
# interactive affordance — operators want a quick yes/no, not a 3-minute
# wait on a hung URL. Real generation calls use the full
# `requestTimeoutSeconds` from config.
_PROBE_TIMEOUT_SECONDS = 10.0


@router.post("/v1/runtimes/switch", response_model=RestartResult, status_code=202)
async def switch_runtime(request: Request, body: RuntimeSwitchRequest) -> RestartResult:
    lifecycle = getattr(request.app.state, "lifecycle", None)
    if lifecycle is None:
        raise HTTPException(status_code=409, detail="Runtime lifecycle is unavailable.")
    try:
        await lifecycle.switch(body.node, body.source, body.target)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RestartResult(
        scheduled=True,
        delayMs=0,
        message=(
            f"{body.source} stopped; {body.target} is starting. "
            "Watch its status for readiness or errors."
        ),
    )


async def _driver_health(client: DriverClient) -> DriverHealth:
    base_url = client.base_url
    try:
        info = await client.info()
        # info.backend is inference-driver.yaml's BackendKind; DriverHealth
        # expects gateway.yaml's BackendKind. Same wire values, distinct
        # generated classes — bridge via .value.
        backend = BackendKind(info.backend.value)
        served = info.models
        account = info.catalogue is not None
        return DriverHealth(
            name=client.name,
            reachable=True,
            url=base_url,  # type: ignore[arg-type]
            backend=backend,
            # A probe of a URL an operator typed, so no name to prefix
            # with: an account's count, or a single model's own id.
            modelId=served[0].id if served and len(served) == 1 and not account else None,
            modelCount=len(served) if served is not None else None,
            account=account,
            catalogueRefreshedAt=info.catalogue.refreshedAt if info.catalogue else None,
            catalogueError=info.catalogue.error if info.catalogue else None,
            runtime=info.runtime,
            version=info.version,
        )
    except httpx.HTTPStatusError as e:
        # **Answering is reachable.** The probe is anonymous (see
        # `probe_driver`), so a driver that requires a credential says
        # 401 -- and reporting that as `reachable: false` sends an
        # operator to check cables and firewalls when their backend is
        # up and wants a key. It is the more diagnostic answer, which is
        # what made dropping the credential cheap.
        code = e.response.status_code
        log.info("driver %r at %s answered %d to an anonymous probe", client.name, base_url, code)
        if code in (401, 403):
            detail = (
                f"{base_url} answered HTTP {code}: it is reachable and wants a credential. "
                "The gateway probes without one, so a URL typed into a form never collects "
                "this install's service token. Save the backend with its own key."
            )
        else:
            detail = f"{base_url} answered HTTP {code} to GET /v1/info: {e}"
        return DriverHealth(
            name=client.name,
            reachable=True,
            url=base_url,  # type: ignore[arg-type]
            error=detail,
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

    # Every driver, whatever it answered, each unreachable or degraded one
    # with its reason; none at all is an empty list. An all-unreachable
    # topology was a 503 here, and the console lost every reason and every
    # companion's runtime with it (2026-10-10).
    healths = table.as_driver_health()
    return DriversInfo(drivers=healths)


@router.get("/v1/admin/routing", response_model=RoutingTableView)
async def get_routing_table(request: Request) -> RoutingTableView:
    """The resolved table: slots, tiers, backends, and what each is doing.

    A view of the last refresh — the same snapshot requests are routed
    from — so what it shows is what a request would meet. The page an
    operator reads when a request went somewhere surprising, and what
    the acceptance run reads to see the balancer alternate.
    """
    table: RoutingTable | None = getattr(request.app.state, "routing", None)
    if table is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=Problem(
                type="https://github.com/eugene-plexus/gateway#no-routing-table",
                title="No routing table",
                status=503,
                detail="The gateway is starting up or in safe mode, so there is no routing table.",
                component="gateway",
            ).model_dump(exclude_none=True),
        )
    return table.as_routing_view()


@router.post("/v1/admin/drivers/probe", response_model=DriverHealth)
async def probe_driver(request: Request, body: DriverProbeRequest) -> DriverHealth:
    """Test-connect to a single backend URL without persisting it.

    Backs the UI's per-URL Test button in the drivers list editor —
    operators verify each backend in a slot's priority list is
    reachable before saving the topology. Builds a one-shot HTTP
    client, hits the URL's `/v1/info`, and returns the same
    `DriverHealth` shape the list endpoint uses.

    **Anonymously.** The behaviour is unchanged and the credential is
    gone: a URL typed into a form no longer collects this install's
    `service:gateway` token. A backend that wants one answers 401, and
    that is reported as *reachable, and it refused an unauthenticated
    probe* — which is a better answer than the `reachable: false` a
    credentialled probe would have turned it into anyway.
    """
    url = str(body.url).rstrip("/")
    client = HttpDriverClient(
        # `name` is optional on a probe (the operator may be testing a
        # URL before naming the slot); fall back to a diagnostic label.
        name=body.name or "(probe)",
        base_url=url,
        timeout_seconds=_PROBE_TIMEOUT_SECONDS,
        # **No credential, deliberately** (R2.4 / review §6.3 #33). This
        # dials whatever the operator typed, which is the feature; it
        # used to dial it holding this install's `service:gateway`
        # token, which every component here accepts, so a URL in a form
        # was enough to collect one. The route is operator-gated, so
        # this is not an open relay -- but an operator typing a URL is
        # not an operator deciding to hand out a credential, and those
        # are different acts.
        auth=None,
    )
    try:
        return await _driver_health(client)
    finally:
        await client.aclose()


#: Row 13: the gateway's own drain. On its own router, because it takes this
#: machine's agent as well as an operator (`require_operator_or_local_agent`).
drain_router = APIRouter(tags=["admin"])


def _drain(request: Request) -> GatewayDrain:
    found = drain.of(request.app)
    if found is None:
        raise HTTPException(status_code=503, detail="The gateway's drain is not set up.")
    return found


@router.post("/v1/admin/restart", response_model=RestartResult, status_code=202)
async def restart(request: Request) -> RestartResult:
    """Restart once the answers in flight finish, so a supervisor relaunches
    the gateway with new config (`requiresRestart` keys are read at startup).

    Row 13: it drains first (`POST /v1/admin/drain`). Until then it exited
    500 ms after answering and cut every stream.
    """
    view = _drain(request).start()
    log.warning("restart requested via /v1/admin/restart; draining first")
    return RestartResult(
        scheduled=True,
        delayMs=0,
        message=view.message or "Restarting once the answers in flight finish.",
    )


@drain_router.post("/v1/admin/drain", response_model=GatewayDrainView, status_code=202)
async def drain_gateway(request: Request) -> GatewayDrainView:
    """Refuse new inference, finish what runs (up to `shutdownDrainSeconds`),
    cut what is left, then exit 0. A second call answers the drain running."""
    return _drain(request).start()


@drain_router.get("/v1/admin/drain", response_model=GatewayDrainView)
async def get_gateway_drain(request: Request) -> GatewayDrainView:
    return _drain(request).view()
