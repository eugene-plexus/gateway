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
from ..config import ConfigStore, as_schema

router = APIRouter(tags=["config"])


@router.get("/v1/config", response_model=ConfigDocument)
async def get_config(request: Request) -> ConfigDocument:
    store: ConfigStore = request.app.state.config_store
    return store.as_document()


@router.get("/v1/config/schema", response_model=ConfigSchema)
async def get_config_schema(request: Request) -> ConfigSchema:
    # Pull configured driver names from the live config store so the
    # `voiceDriver` field carries an up-to-date dropdown. Tolerate
    # malformed `drivers` (defensive — validation rejects this shape on
    # PATCH but the schema endpoint should never error on it).
    store: ConfigStore = request.app.state.config_store
    raw = store.get("drivers") or []
    driver_names: list[str] = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                name = entry.get("name")
                if isinstance(name, str) and name.strip():
                    driver_names.append(name)
    return as_schema(driver_names=driver_names)


@router.patch("/v1/config", response_model=ConfigUpdateResult)
async def patch_config(request: Request, body: ConfigUpdateRequest) -> ConfigUpdateResult:
    store: ConfigStore = request.app.state.config_store
    return store.apply_patch(body)


@router.post("/v1/config/test", response_model=ConfigTestResult)
async def test_config(
    request: Request,
    body: ConfigTestRequest | None = None,
) -> ConfigTestResult:
    """Probe every configured driver's `/v1/info` and the memory
    service's `/healthz` using saved config + optional overrides.
    Override values are NOT persisted."""
    start = time.perf_counter()
    store: ConfigStore = request.app.state.config_store

    overrides: dict[str, Any] = {}
    if body and body.overrides:
        overrides = body.overrides.model_dump(exclude_none=True)

    def get(key: str) -> Any:
        return overrides[key] if key in overrides else store.get(key)

    # Resolve slot backends (topology names) → URLs the same way the
    # lifespan does, so the Test button probes the real endpoints a chat
    # turn would reach. Lazy import breaks the app↔routes import cycle.
    from ..app import _resolve_backend_url, driver_topology, fetch_components

    drivers_raw = get("drivers") or []
    memory_url = str(get("memoryUrl") or "")
    timeout = float(get("requestTimeoutSeconds") or 30)
    settings = request.app.state.settings
    service_token = request.app.state.auth_state.service_token
    probe_headers = {"Authorization": f"Bearer {service_token}"} if service_token else None

    topo = driver_topology(
        await fetch_components(watchdog_url=settings.watchdog_url, service_token=service_token)
    )

    async def probe(name: str, base_url: str, path: str) -> tuple[str, str | None]:
        """Returns (target-name, error-or-None)."""
        if not base_url:
            return name, f"{name} URL is empty"
        try:
            async with httpx.AsyncClient(timeout=timeout, headers=probe_headers) as client:
                response = await client.get(base_url.rstrip("/") + path)
                response.raise_for_status()
        except Exception as e:
            return name, f"{name} ({base_url}{path}) — {e}"
        return name, None

    probes: list[Any] = [probe("memory", memory_url, "/healthz")]
    # Backends that don't resolve to a URL are immediate failures (no
    # endpoint to probe); collected here and merged with the probe results.
    immediate: list[tuple[str, str | None]] = []
    for entry in drivers_raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "<unnamed>")
        # A slot is a priority list of backend NAMES (v0.2.1). Resolve and
        # probe each so a misconfigured fallback is caught before it's
        # needed. Tolerate legacy `urls`/`url` overrides that predate the
        # migration. Label multi-backend slots `name[0]`, `name[1]`, ….
        backends = entry.get("backends")
        if not isinstance(backends, list) or not backends:
            legacy = entry.get("urls")
            if not isinstance(legacy, list) or not legacy:
                single = entry.get("url")
                legacy = [single] if single else []
            backends = legacy
        for i, backend in enumerate(backends):
            label = name if len(backends) == 1 else f"{name}[{i}]"
            url = _resolve_backend_url(str(backend), topo)
            if url is None:
                immediate.append(
                    (label, f"{label}: backend {backend!r} not found in watchdog topology")
                )
            else:
                probes.append(probe(label, url, "/v1/info"))

    results = list(await asyncio.gather(*probes)) + immediate

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    failures = [err for _, err in results if err]
    if failures:
        return ConfigTestResult(
            ok=False,
            component="gateway",
            latencyMs=elapsed_ms,
            error="; ".join(failures),
        )
    driver_count = len(drivers_raw)
    return ConfigTestResult(
        ok=True,
        component="gateway",
        latencyMs=elapsed_ms,
        summary=f"all {driver_count} driver(s) + memory reachable in {elapsed_ms}ms",
    )
