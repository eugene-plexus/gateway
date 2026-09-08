"""FastAPI app factory."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI

from . import __version__
from .auth_state import AuthState, load_auth_state
from .bicameral.callosum import JaccardAgreementScorer, load_default_scorer
from .bicameral.nt import neutral_state
from .config import ConfigStore
from .dependencies import require_authorized, require_operator
from .driver_client import DriverClient, FailoverDriverClient, HttpDriverClient
from .identity import HttpIdentity, IdentityClient
from .memory import HttpMemory, MemoryClient
from .routes import admin as admin_routes
from .routes import config as config_routes
from .routes import conversations as conversations_routes
from .routes import events as events_routes
from .routes import health as health_routes
from .routes import stream as stream_routes
from .runtime.loop import ConsciousnessLoop
from .runtime.stream import ConsciousnessBroker
from .settings import Settings, load_settings
from .tools import build_tool_runner

log = logging.getLogger(__name__)


def _resolve_backend_url(backend: str, topology: dict[str, str]) -> str | None:
    """Resolve one slot backend (a topology entry name) to a URL.

    Resolution order:
      1. Exact match against a watchdog-topology inference-driver name.
      2. URL-shaped fallback — a backend that looks like a URL (`http…`)
         is used directly. This covers configs migrated from the
         pre-v0.2.1-item-2 `urls` shape, where the stored values are
         URLs rather than names; the operator should re-save via the UI
         dropdown to convert them to names (logged below).
      3. Unresolvable — None (caller skips + warns).
    """
    url = topology.get(backend)
    if url:
        return url.rstrip("/")
    if backend.startswith(("http://", "https://")):
        log.warning(
            "driver backend %r is a raw URL, not a topology name — using it directly. "
            "Re-save the drivers config to pick a topology entry instead.",
            backend,
        )
        return backend.rstrip("/")
    return None


def build_clients(
    store: ConfigStore,
    auth_state: AuthState,
    topology: dict[str, str],
) -> list[DriverClient]:
    """Construct one DriverClient per configured driver slot.

    Reads the `drivers` config — `[{name, backends: [topology-name, …]}]`
    — and resolves each backend NAME to a URL via `topology` (a
    name→url map of the watchdog's inference-driver entries, fetched
    once at startup). Order is preserved so the bicameral loop and admin
    endpoints walk slots and their failover backends in declared order.
    Threads `auth_state.service_token` through so each outbound
    /v1/generate carries the gateway's service-audience bearer.

    A backend that resolves to no URL (unknown topology name, not
    URL-shaped) is skipped with a warning. A slot left with zero
    resolvable backends is not built — running with a partial/empty
    driver set degrades to a 503 at chat time rather than crashing.
    """
    raw = store.get("drivers") or []
    timeout = float(store.get("requestTimeoutSeconds") or 180)
    clients: list[DriverClient] = []
    for entry in raw:
        # Validation in ConfigStore.apply_patch already enforces shape, but
        # in-memory config can also be loaded straight from YAML so guard
        # again here. `load()` migrates legacy `urls`/`url` entries to the
        # `backends` shape, so by here every slot carries `backends`.
        name = entry["name"]
        backends = entry["backends"]
        candidates: list[DriverClient] = []
        for backend in backends:
            url = _resolve_backend_url(backend, topology)
            if url is None:
                log.warning(
                    "driver slot %r: backend %r not found in watchdog topology "
                    "(known: %s) — skipping it",
                    name,
                    backend,
                    sorted(topology),
                )
                continue
            candidates.append(
                HttpDriverClient(
                    name=name,
                    base_url=url,
                    timeout_seconds=timeout,
                    service_token=auth_state.service_token,
                )
            )
        if not candidates:
            log.warning(
                "driver slot %r has no resolvable backends %s — not building it",
                name,
                backends,
            )
            continue
        # One slot = an ordered priority list of backends. A single-backend
        # slot still goes through FailoverDriverClient (one candidate),
        # which behaves identically to a bare HttpDriverClient.
        clients.append(FailoverDriverClient(name=name, candidates=candidates))
    return clients


async def fetch_components(
    *,
    watchdog_url: str,
    service_token: str | None,
    timeout_seconds: float = 5.0,
) -> list[dict[str, Any]]:
    """Fetch the watchdog topology (`GET /v1/components`), once.

    The watchdog is the source of truth for body-component topology;
    duplicating URLs in every component's config is the OpenClaw-style
    trap we're avoiding. The gateway resolves its peers (memory,
    identity) and its driver backends from this one snapshot at startup
    — item 3 (v0.2.1) made the endpoint accept the gateway's
    service token.

    Returns the list of component dicts, or `[]` when the watchdog
    can't be reached / returns an error. An empty result degrades
    cleanly: peers stay unresolved and drivers don't build, rather than
    crashing startup.
    """
    headers = {"Authorization": f"Bearer {service_token}"} if service_token else {}
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(
                f"{watchdog_url.rstrip('/')}/v1/components",
                headers=headers,
            )
        if response.status_code >= 400:
            return []
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return []
    components = body.get("components") if isinstance(body, dict) else None
    return components if isinstance(components, list) else []


def peer_url_from(components: list[dict[str, Any]], kind: str) -> str | None:
    """First topology URL of the given `kind` (trailing slash stripped)."""
    for c in components:
        if isinstance(c, dict) and c.get("kind") == kind:
            url = c.get("url")
            if isinstance(url, str) and url:
                return url.rstrip("/")
    return None


def driver_topology(components: list[dict[str, Any]]) -> dict[str, str]:
    """Build a {name: url} map of the inference-driver topology entries.

    This is what `build_clients` resolves slot backends against. URLs
    keep their trailing slash here; `_resolve_backend_url` strips it.
    """
    out: dict[str, str] = {}
    for c in components:
        if isinstance(c, dict) and c.get("kind") == "inference-driver":
            name, url = c.get("name"), c.get("url")
            if isinstance(name, str) and name and isinstance(url, str) and url:
                out[name] = url
    return out


def build_memory(memory_url: str, auth_state: AuthState) -> tuple[MemoryClient, str]:
    """Construct the memory client from a resolved URL. Returns (client, url)."""
    # Memory ops are short — append a message, fetch a conversation. The
    # 30s ceiling here is far more generous than needed but matches the
    # connection-timeout shape used elsewhere; bump only if a future
    # backend (DB-backed retrieval, vector search) needs longer.
    timeout = 30.0
    return (
        HttpMemory(
            base_url=memory_url,
            timeout_seconds=timeout,
            service_token=auth_state.service_token,
        ),
        memory_url,
    )


def build_identity(identity_url: str, auth_state: AuthState) -> tuple[IdentityClient, str]:
    """Construct the identity client from a resolved URL. Returns (client, url)."""
    return (
        HttpIdentity(
            base_url=identity_url,
            timeout_seconds=30.0,
            service_token=auth_state.service_token,
        ),
        identity_url,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    store = ConfigStore(settings.config_file)
    if settings.safe_mode:
        # Safe mode: ignore the on-disk config, leaving the store with
        # built-in defaults. PATCH /v1/config still writes to disk so the
        # operator's repair survives the next non-safe-mode boot. No
        # drivers, no memory client — chat returns 503 until restart.
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

    # v0.2 NT state. In-memory only — a restart resets Eugene to neutral
    # (matches "anatomy cools down after a reboot"). The chat handler
    # reads + ticks this per turn; `/v1/admin/nt-state` surfaces it.
    if not hasattr(app.state, "nt_state"):
        app.state.nt_state = neutral_state()

    # v0.2 auth state. Tests can pre-populate `app.state.auth_state` to
    # exercise authed paths; the default lifespan build reads env vars
    # via Settings and produces an auth-disabled state when the watchdog
    # didn't supply AUTH_SIGNING_KEY.
    if not hasattr(app.state, "auth_state"):
        app.state.auth_state = load_auth_state(
            signing_key_b64=settings.auth_signing_key,
            service_token=settings.service_token,
            master_key_b64=settings.master_key,
        )
    auth_state: AuthState = app.state.auth_state

    # Fetch the watchdog topology ONCE and resolve everything from it:
    # peer URLs (memory, identity) and the inference-driver name→url map
    # that `build_clients` resolves slot backends against. One round-trip
    # instead of one-per-peer; the source of truth is the watchdog, so
    # backend URLs aren't duplicated into the gateway's own config.
    components = (
        []
        if settings.safe_mode
        else await fetch_components(
            watchdog_url=settings.watchdog_url,
            service_token=auth_state.service_token,
        )
    )

    # Resolve peer URLs. Operator-supplied config wins; otherwise we take
    # it from the topology snapshot. The duplicate-URL trap — where an
    # operator who set up identity in the wizard nevertheless has
    # identityUrl unset on the gateway and silently runs without
    # identity — is exactly what this auto-resolve closes.
    def _resolve(kind: str, config_key: str) -> str | None:
        explicit = str(store.get(config_key) or "").strip()
        if explicit:
            return explicit
        if settings.safe_mode:
            return None
        resolved = peer_url_from(components, kind)
        if resolved:
            log.info(
                "auto-resolved %s from watchdog: %s (config field %s was unset)",
                kind,
                resolved,
                config_key,
            )
        return resolved

    # Same injection trick as hemisphere clients: tests pre-populate
    # `app.state.memory` with an `InProcessMemory` so the lifespan
    # doesn't try to reach a real memory service. In production the
    # `HttpMemory` is built here.
    if not hasattr(app.state, "memory"):
        if settings.safe_mode:
            app.state.memory = None
            app.state.memory_url = ""
            owns_memory = False
        else:
            memory_url = _resolve("memory", "memoryUrl") or "http://127.0.0.1:8083"
            memory, memory_url = build_memory(memory_url, auth_state)
            app.state.memory = memory
            app.state.memory_url = memory_url
            owns_memory = True
    else:
        owns_memory = False
        app.state.memory_url = getattr(app.state, "memory_url", "")

    # Identity is optional: when unconfigured AND the watchdog has no
    # identity entry, the chat handler falls back to the v0.1
    # prompt-building path. Mirrors the memory-injection pattern so
    # tests can pre-populate `app.state.identity`.
    if not hasattr(app.state, "identity"):
        if settings.safe_mode:
            app.state.identity = None
            app.state.identity_url = ""
            owns_identity = False
        else:
            identity_url = _resolve("identity", "identityUrl")
            if identity_url:
                identity, identity_url = build_identity(identity_url, auth_state)
                app.state.identity = identity
                app.state.identity_url = identity_url
                owns_identity = True
            else:
                app.state.identity = None
                app.state.identity_url = ""
                owns_identity = False
    else:
        owns_identity = False
        app.state.identity_url = getattr(app.state, "identity_url", "")

    if not hasattr(app.state, "drivers"):
        # Slot backends are resolved against the inference-driver entries
        # in the topology snapshot. In safe mode (or when the watchdog is
        # unreachable) the map is empty → no drivers built → chat 503.
        app.state.drivers = (
            []
            if settings.safe_mode
            else build_clients(store, auth_state, driver_topology(components))
        )
        owns_clients = not settings.safe_mode
    else:
        owns_clients = False

    # Corpus-callosum scorer. The `disable_embedding_scorer` Settings
    # flag short-circuits the heavy load path — tests set it via the
    # fixture in `conftest.py` so the suite doesn't pull torch +
    # transformers, and operators can set
    # EUGENE_PLEXUS_GATEWAY_DISABLE_EMBEDDING_SCORER=1 on resource-
    # constrained boxes. Production loads the embedding model via
    # `to_thread` so the lifespan doesn't block uvicorn's event loop
    # during the (multi-second) first-time model download + warmup.
    if not hasattr(app.state, "scorer"):
        if settings.safe_mode or settings.disable_embedding_scorer:
            app.state.scorer = JaccardAgreementScorer()
        else:
            model_name = str(store.get("agreementModel") or "all-MiniLM-L6-v2")
            app.state.scorer = await asyncio.to_thread(load_default_scorer, model_name)

    # Tool-wire retrofit (Phase 1): in-process tool registry. Re-expresses
    # the gateway's own operations as channel-tagged tools dispatched
    # through a ToolRunner. Built once here; reload-capable via
    # build_tool_runner for a future config surface. Tests that inject
    # app.state.memory get a runner over their InProcessMemory.
    if not hasattr(app.state, "tool_runner"):
        app.state.tool_runner = build_tool_runner(
            memory=getattr(app.state, "memory", None),
            identity=getattr(app.state, "identity", None),
        )

    # The continuous consciousness loop — Eugene's single cognitive task.
    # Built after all of app.state is in place (drivers, memory, identity,
    # scorer, tools, NT) and started here; the HTTP endpoints inject
    # afferent events onto its queue and subscribe to its observability
    # stream. Always built (even in safe mode) so the wiring is uniform —
    # it simply has nothing to think with until drivers resolve, and
    # POST /v1/events returns 503 in the meantime.
    if not hasattr(app.state, "broker"):
        app.state.broker = ConsciousnessBroker()
    if not hasattr(app.state, "loop"):
        app.state.loop = ConsciousnessLoop(app, app.state.broker)
    app.state.event_queue = app.state.loop.queue
    app.state.loop.start()

    try:
        yield
    finally:
        # Stop cognition before tearing down the clients it depends on.
        await app.state.loop.stop()
        if owns_memory:
            await app.state.memory.aclose()
        if owns_identity and app.state.identity is not None:
            await app.state.identity.aclose()
        if owns_clients:
            for client in app.state.drivers:
                await client.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app with all routers mounted."""
    settings = settings or load_settings()

    app = FastAPI(
        title="Eugene Plexus — gateway",
        description="Continuous-loop consciousness gateway.",
        version=__version__,
        lifespan=_lifespan,
    )
    app.state.settings = settings
    # `drivers` is the canonical list of hemisphere clients in declared
    # order. Tests inject this on `app.state` before the lifespan runs;
    # the lifespan otherwise builds it from `ConfigStore`.
    _: Any = app.state  # placate mypy on dynamic state access below

    # Health stays unauthenticated — supervisors and load balancers need
    # to probe it without holding credentials.
    app.include_router(health_routes.router)

    # Operator-only surfaces: config edits + admin actions ride on the
    # UI's session token (or any future operator-scoped client). Service
    # tokens are rejected.
    operator_only = [Depends(require_operator)]
    app.include_router(config_routes.router, dependencies=operator_only)
    app.include_router(admin_routes.router, dependencies=operator_only)

    # Mixed surfaces: chat and conversation reads are reachable from
    # both the UI (operator token) and peer components (service tokens
    # — e.g. connector → gateway when a Discord message comes in).
    authorized = [Depends(require_authorized)]
    app.include_router(conversations_routes.router, dependencies=authorized)
    app.include_router(events_routes.router, dependencies=authorized)
    app.include_router(stream_routes.router, dependencies=authorized)

    return app
