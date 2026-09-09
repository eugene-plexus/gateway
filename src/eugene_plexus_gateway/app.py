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
from .config import ConfigStore
from .dependencies import require_operator
from .routes import admin as admin_routes
from .routes import config as config_routes
from .routes import health as health_routes
from .routes import inference as inference_routes
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
            service_token=settings.service_token,
            master_key_b64=settings.master_key,
        )
    auth_state: AuthState = app.state.auth_state

    # Tests inject `app.state.routing` with a pre-populated table; the
    # lifespan otherwise builds the real one and owns its teardown.
    owns_routing = False
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
                request_timeout_seconds=float(store.get("requestTimeoutSeconds") or 180),
                refresh_seconds=float(store.get("routingRefreshSeconds") or 15),
            )
            app.state.routing = table
            owns_routing = True
            # Awaited, so the gateway is routable the moment it serves.
            await table.start()

    try:
        yield
    finally:
        if owns_routing and app.state.routing is not None:
            await app.state.routing.aclose()


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

    # The front door accepts operator OR service tokens: a UI playground
    # and another component are both legitimate callers. Declared on the
    # router's own routes, so this is just the mount.
    app.include_router(inference_routes.router)

    return app
