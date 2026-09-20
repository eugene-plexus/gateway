"""FastAPI app factory.

The gateway's whole job is routing, so the lifespan is short: load
config, resolve auth, and build the routing table. Everything the table
needs comes from the agent topology plus each driver's `/v1/info`, so
there are no peer URLs to resolve and no clients to construct from
config.

Safe mode (`EUGENE_PLEXUS_GATEWAY_SAFE_MODE=1`) skips the on-disk config
and skips building the table. `PATCH /v1/config` still writes to disk, so
the operator's repair survives the next normal boot; inference returns
503 until then.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from . import __version__
from .auth_state import AuthState, load_auth_state
from .client_keys import ClientKeyGuard
from .config import DEFAULT_REQUEST_TIMEOUT_SECONDS, ConfigStore
from .cors import FrontDoorCors
from .dependencies import require_operator
from .lifecycle import AgentLifecycleClient, LifecycleManager
from .metrics import MetricsStore
from .routes import admin as admin_routes
from .routes import config as config_routes
from .routes import health as health_routes
from .routes import inference as inference_routes
from .routes import metrics as metrics_routes
from .routing import RoutingTable
from .settings import Settings, load_settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    store = ConfigStore(settings.config_file)
    if settings.safe_mode:
        log.warning(
            "starting in SAFE MODE (EUGENE_PLEXUS_GATEWAY_SAFE_MODE=1); "
            "ignoring %s and running on defaults. Fix config via "
            "/v1/config, then restart without the env var.",
            settings.config_file,
        )
    else:
        store.load()
    app.state.config_store = store
    app.state.safe_mode = settings.safe_mode

    # Tests can pre-populate `app.state.auth_state` to exercise authed
    # paths; the default build reads env vars via Settings and produces
    # an auth-disabled state when the agent didn't supply a signing
    # key (dev / standalone).
    if not hasattr(app.state, "auth_state"):
        app.state.auth_state = load_auth_state(
            signing_key_b64=settings.auth_signing_key,
            verify_key_b64=settings.auth_verify_key,
            service_token=settings.service_token,
            master_key_b64=settings.master_key,
        )
    auth_state: AuthState = app.state.auth_state

    # Retained request metrics (M8). Built before the routing table, so
    # the very first completion is recorded — the gateway is routable the
    # moment `table.start()` returns, and a store opened after it would
    # miss whatever arrived in between.
    #
    # Off in safe mode: safe mode exists to get a broken install back to
    # a config endpoint, and opening a database is one more thing that
    # can fail on the way there.
    metrics: MetricsStore | None = None
    if not hasattr(app.state, "metrics"):
        app.state.metrics = None
        if not settings.safe_mode and bool(store.get("metricsEnabled")):
            candidate = MetricsStore(
                settings.metrics_file,
                retention_days=int(store.get("metricsRetentionDays") or 7),
                rollup_enabled=bool(store.get("metricsRollupEnabled")),
            )
            try:
                await candidate.start()
            except Exception as e:
                # Never fatal. A gateway that will not serve because it
                # could not open a metrics file has traded the product
                # for its instrumentation.
                log.warning(
                    "could not open the metrics store at %s: %s. Inference is "
                    "unaffected; GET /v1/metrics will report it as disabled.",
                    settings.metrics_file,
                    e,
                )
            else:
                metrics = candidate
                app.state.metrics = candidate

    # Which client keys have been turned off (S4). Asks this gateway's
    # OWN node's agent -- the one that minted them, and the one the
    # contract tells an operator to mint against. Consulted only for a
    # token that carries `aud: client`, so an install with no client
    # keys never makes the call.
    guard: ClientKeyGuard | None = None
    if not hasattr(app.state, "client_key_guard"):
        if auth_state.auth_disabled:
            # Nothing verifies here, so nothing can be revoked here.
            app.state.client_key_guard = None
        else:
            guard = ClientKeyGuard(
                agent_url=settings.agent_url,
                service_token=auth_state.service_token,
                ttl_seconds=float(store.get("routingRefreshSeconds") or 15),
            )
            app.state.client_key_guard = guard

    # Tests inject `app.state.routing` with a pre-populated table; the
    # lifespan otherwise builds the real one and owns its teardown.
    owns_routing = False
    lifecycle: LifecycleManager | None = None
    if not hasattr(app.state, "routing"):
        if settings.safe_mode:
            # No table at all in safe mode, rather than an empty one:
            # "we never looked" and "we looked and found nothing" deserve
            # different answers, and the inference route says so.
            app.state.routing = None
        else:
            table = RoutingTable(
                agent_url=settings.agent_url,
                service_token=auth_state.service_token,
                request_timeout_seconds=float(
                    store.get("requestTimeoutSeconds") or DEFAULT_REQUEST_TIMEOUT_SECONDS
                ),
                refresh_seconds=float(store.get("routingRefreshSeconds") or 15),
                # Read live, so a PATCH takes effect on the next request.
                slots=lambda: store.get("modelSlots"),
                strategy=lambda: store.get("loadBalancing"),
                # Live, like the two above it: `controlUrl` is documented
                # as taking effect on the next routing refresh.
                control_url=lambda: store.get("controlUrl"),
            )
            app.state.routing = table
            owns_routing = True
            # Awaited, so the gateway is routable the moment it serves.
            await table.start()
            # Lifecycle policy rides on the same table: idle unload and
            # start on demand, with the gateway's own service token.
            lifecycle = LifecycleManager(
                table,
                client=AgentLifecycleClient(auth_state.service_token),
                swap_wait_seconds=lambda: float(store.get("swapWaitSeconds") or 120),
                idle_check_seconds=lambda: float(store.get("idleCheckSeconds") or 15),
            )
            await lifecycle.start()
            app.state.lifecycle = lifecycle

    try:
        yield
    finally:
        if lifecycle is not None:
            await lifecycle.aclose()
        if owns_routing and app.state.routing is not None:
            await app.state.routing.aclose()
        if guard is not None:
            await guard.aclose()
        # Last, so rows queued by requests still in flight during
        # shutdown are flushed rather than dropped.
        if metrics is not None:
            await metrics.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app with all routers mounted."""
    settings = settings or load_settings()

    app = FastAPI(
        title="Eugene Plexus — gateway",
        description=(
            "One OpenAI-compatible endpoint over every configured backend. "
            "Routes, load-balances and fails over; holds no backend knowledge."
        ),
        version=__version__,
        lifespan=_lifespan,
    )
    app.state.settings = settings

    # Health stays unauthenticated — supervisors and load balancers need
    # to probe it without holding credentials.
    app.include_router(health_routes.router)

    # Operator-only: config edits and admin actions ride on the UI's
    # session token. Service tokens are rejected here, because a peer
    # component has no business restarting the gateway.
    operator_only = [Depends(require_operator)]
    app.include_router(config_routes.router, dependencies=operator_only)
    app.include_router(admin_routes.router, dependencies=operator_only)
    # Metrics are operator-only too, and unlike other reads they do not
    # accept a service token: no component needs them, and model names
    # plus traffic volumes are not nothing on a shared tailnet.
    app.include_router(metrics_routes.router, dependencies=operator_only)

    # The front door accepts operator OR service tokens: a UI playground
    # and another component are both legitimate callers. Declared on the
    # router's own routes, so this is just the mount.
    app.include_router(inference_routes.router)

    # CORS on the three OpenAI-compatible paths, and on nothing else, so a
    # browser-based client -- the playground's direct mode among them --
    # can use the front door the way a non-browser client always could.
    # Configured live from the store; see `cors.py`.
    app.add_middleware(FrontDoorCors)

    return app
