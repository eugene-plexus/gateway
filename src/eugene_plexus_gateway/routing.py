"""The routing table: model id -> the drivers that serve it.

The gateway stores no backend URLs and no model list. It reads the
watchdog topology for `inference-driver` entries, asks each one's
`/v1/info` what it is serving, and groups the answers by model id. So:

  * backend addresses live in exactly one place, the watchdog topology
  * adding a model is not a config edit — start an engine, point a
    driver at it, and it becomes routable on the next refresh
  * two drivers serving the same model are automatically a priority
    list, which means failover falls out of the topology rather than
    needing to be configured

The refresh is periodic rather than on-demand because a request should
never pay for topology discovery, and because engines come and go under
the watchdog without telling anyone.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from ._generated.driver_models import DriverInfo
from ._generated.models import (
    BackendKind,
    DriverHealth,
    Model,
    ModelRoutingInfo,
)
from .driver_client import DriverClient, FailoverDriverClient, HttpDriverClient

log = logging.getLogger(__name__)


@dataclass
class _Backend:
    """One reachable driver and what it told us it serves."""

    name: str
    url: str
    client: HttpDriverClient
    info: DriverInfo


@dataclass
class _Unreachable:
    """A driver in the topology that didn't answer `/v1/info`."""

    name: str
    url: str
    error: str


@dataclass
class _Snapshot:
    """One resolved view of the world. Replaced wholesale on refresh so a
    request never sees a half-rebuilt table."""

    by_model: dict[str, list[_Backend]] = field(default_factory=dict)
    reachable: list[_Backend] = field(default_factory=list)
    unreachable: list[_Unreachable] = field(default_factory=list)


class RoutingTable:
    """Resolves a requested model to the backends that can serve it."""

    def __init__(
        self,
        *,
        watchdog_url: str,
        service_token: str | None = None,
        request_timeout_seconds: float = 180.0,
        refresh_seconds: float = 15.0,
    ) -> None:
        self._watchdog_url = watchdog_url
        self._service_token = service_token
        self._request_timeout = request_timeout_seconds
        self._refresh_seconds = refresh_seconds
        self._snapshot = _Snapshot()
        # Clients are cached by (name, url) and reused across refreshes.
        # Rebuilding an httpx.AsyncClient every 15s would throw away the
        # connection pool and leak sockets.
        self._clients: dict[tuple[str, str], HttpDriverClient] = {}
        self._task: asyncio.Task[None] | None = None

    # --- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        """Do one refresh now, then keep refreshing in the background.

        The first refresh is awaited so the gateway is routable as soon
        as it is serving, rather than 404ing for the first interval.
        """
        await self.refresh()
        if self._task is None:
            self._task = asyncio.create_task(self._refresh_loop(), name="routing-refresh")

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, BaseException):
                await self._task
            self._task = None
        for client in self._clients.values():
            with contextlib.suppress(BaseException):
                await client.aclose()
        self._clients.clear()
        self._snapshot = _Snapshot()

    async def _refresh_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._refresh_seconds)
                try:
                    await self.refresh()
                except Exception as e:
                    # Never let a bad refresh kill the loop — the previous
                    # snapshot stays serving, which is strictly better than
                    # a gateway that stops routing because the watchdog
                    # blipped.
                    log.warning("routing refresh failed; keeping previous table: %s", e)
        except asyncio.CancelledError:
            return

    # --- refresh ----------------------------------------------------------

    async def refresh(self) -> None:
        entries = await self.fetch_driver_entries()

        # Drop clients for drivers that left the topology.
        live_keys = {(name, url) for name, url in entries}
        for key in list(self._clients):
            if key not in live_keys:
                client = self._clients.pop(key)
                log.info("driver %r left the topology; closing its client", key[0])
                with contextlib.suppress(BaseException):
                    await client.aclose()

        results = await asyncio.gather(
            *(self._probe(name, url) for name, url in entries),
            return_exceptions=True,
        )

        snapshot = _Snapshot()
        for result in results:
            if isinstance(result, BaseException):
                log.warning("driver probe raised unexpectedly: %s", result)
                continue
            if isinstance(result, _Unreachable):
                snapshot.unreachable.append(result)
                continue
            snapshot.reachable.append(result)

        for backend in snapshot.reachable:
            model_id = backend.info.modelId
            if not model_id:
                # A driver that won't say what it serves cannot be routed
                # to — there is no key to route on. Its health still shows
                # up on the admin surface.
                log.debug(
                    "driver %r reports no modelId; not routable (it is probably in degraded mode)",
                    backend.name,
                )
                continue
            snapshot.by_model.setdefault(model_id, []).append(backend)

        # Stable order per model so repeated requests hit the same backend
        # first. Round-robin across replicas is load balancing, which is
        # deliberately M5 — doing it accidentally here would make failover
        # untestable.
        for backends in snapshot.by_model.values():
            backends.sort(key=lambda b: b.name)

        self._snapshot = snapshot
        log.debug(
            "routing table refreshed: %d model(s) across %d reachable driver(s), %d unreachable",
            len(snapshot.by_model),
            len(snapshot.reachable),
            len(snapshot.unreachable),
        )

    async def fetch_driver_entries(self) -> list[tuple[str, str]]:
        """`(name, url)` for every inference-driver in the watchdog topology.

        Public because `POST /v1/config/test` reads the topology fresh
        rather than off the last refresh — the point of a Test button is
        the world as it is now.

        An unreachable watchdog yields an empty list, which degrades to
        "nothing is routable" rather than crashing — the config endpoints
        stay up so the operator can fix whatever is wrong.
        """
        headers = {"Authorization": f"Bearer {self._service_token}"} if self._service_token else {}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self._watchdog_url.rstrip('/')}/v1/components",
                    headers=headers,
                )
            if response.status_code >= 400:
                log.warning(
                    "watchdog topology returned %d; nothing is routable this refresh",
                    response.status_code,
                )
                return []
            body: Any = response.json()
        except (httpx.HTTPError, ValueError) as e:
            log.warning("could not reach the watchdog topology (%s); nothing is routable", e)
            return []

        components = body.get("components") if isinstance(body, dict) else None
        if not isinstance(components, list):
            return []

        out: list[tuple[str, str]] = []
        for entry in components:
            if not isinstance(entry, dict) or entry.get("kind") != "inference-driver":
                continue
            name, url = entry.get("name"), entry.get("url")
            if isinstance(name, str) and name and isinstance(url, str) and url:
                out.append((name, url.rstrip("/")))
        return out

    async def _probe(self, name: str, url: str) -> _Backend | _Unreachable:
        client = self._clients.get((name, url))
        if client is None:
            client = HttpDriverClient(
                name=name,
                base_url=url,
                timeout_seconds=self._request_timeout,
                service_token=self._service_token,
            )
            self._clients[(name, url)] = client
        try:
            info = await client.info()
        except (httpx.HTTPError, ValueError) as e:
            return _Unreachable(name=name, url=url, error=str(e))
        return _Backend(name=name, url=url, client=client, info=info)

    # --- resolution -------------------------------------------------------

    def resolve(self, model: str) -> DriverClient | None:
        """The client to send `model` to, or None if nothing serves it.

        Several backends become a `FailoverDriverClient`, so the
        priority-list cascade (transport error / 5xx cascade, 4xx hard
        fail) applies without any configuration. One backend is returned
        directly — wrapping a single candidate would only add a frame.
        """
        backends = self._snapshot.by_model.get(model)
        if not backends:
            return None
        if len(backends) == 1:
            return backends[0].client
        return FailoverDriverClient(
            name=model,
            candidates=[b.client for b in backends],
        )

    def backends_for(self, model: str) -> list[_Backend]:
        return list(self._snapshot.by_model.get(model, []))

    def known_models(self) -> list[str]:
        return sorted(self._snapshot.by_model)

    def is_empty(self) -> bool:
        return not self._snapshot.by_model

    # --- read models ------------------------------------------------------

    def as_model_list(self) -> list[Model]:
        """The OpenAI-compatible model list.

        Two drivers serving one model produce one entry: replicas are a
        routing detail, and a client should not have to know how many
        GPUs are behind a name.
        """
        out: list[Model] = []
        for model_id in sorted(self._snapshot.by_model):
            backends = self._snapshot.by_model[model_id]
            providers = {b.info.provider for b in backends if b.info.provider}
            out.append(
                Model(
                    id=model_id,
                    object="model",
                    # OpenAI sends an organisation here; we send the
                    # provider. `local` when the driver didn't name one,
                    # which is the openai_compat_http local-engine case.
                    owned_by=sorted(providers)[0] if len(providers) == 1 else "local",
                    x_eugene_plexus=ModelRoutingInfo(
                        drivers=[b.name for b in backends],
                        backends=sorted({_backend_kind(b) for b in backends}),
                        context_length=_smallest_context(backends),
                    ),
                )
            )
        return out

    def as_driver_health(self) -> list[DriverHealth]:
        """Per-driver snapshot for the admin surface, reachable first."""
        out = [
            DriverHealth(
                name=b.name,
                reachable=True,
                url=b.url,  # type: ignore[arg-type]
                backend=_backend_kind(b),
                modelId=b.info.modelId,
                version=b.info.version,
            )
            for b in sorted(self._snapshot.reachable, key=lambda b: b.name)
        ]
        out += [
            DriverHealth(
                name=u.name,
                reachable=False,
                url=u.url,  # type: ignore[arg-type]
                error=u.error,
            )
            for u in sorted(self._snapshot.unreachable, key=lambda u: u.name)
        ]
        return out


def _backend_kind(backend: _Backend) -> BackendKind:
    """Bridge inference-driver.yaml's BackendKind to gateway.yaml's.

    Same wire values, two generated classes — the price of "components
    share schemas, not code", and cheaper than the coupling a shared
    module would create. Crossing via `.value` keeps the boundary
    explicit instead of hiding it behind a cast.
    """
    return BackendKind(backend.info.backend.value)


def _smallest_context(backends: list[_Backend]) -> int | None:
    """Smallest context window among the backends serving one model.

    The honest number to report, since a request may land on any of
    them — promising the largest would mean a prompt that fits the
    advertised window can still be rejected.
    """
    lengths = [
        b.info.capabilities.maxContextTokens
        for b in backends
        if b.info.capabilities is not None and b.info.capabilities.maxContextTokens
    ]
    return min(lengths) if lengths else None


__all__ = ["BackendKind", "RoutingTable"]
