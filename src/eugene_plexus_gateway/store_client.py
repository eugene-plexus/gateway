"""The gateway's one client of the install's store (row 53).

The store keeps what the platform doors keep -- files and uploads first.
The gateway reaches it the way it reaches the library: through its own
agent's `/api/proxy/store`, which finds the store on this machine or on
the machine that holds it, with the gateway's own service token. Every
call names the owner in `X-Eugene-Plexus-Owner`, which the store believes
only beside a gateway's token; the gateway sets it from its own admission,
never from anything the client sent.

**A store that is not answering is said as what was observed** and
touches nothing else: chat, completions and Messages without a `file_id`
never call it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from ._http import internal_client

OWNER_HEADER = "X-Eugene-Plexus-Owner"
#: Generous: an upload's `complete` joins up to gigabytes of parts on disk.
TIMEOUT = httpx.Timeout(300.0, connect=5.0)


class StoreUnavailable(Exception):
    """The store could not be reached, or said it cannot work; `cause` says what was seen."""

    def __init__(self, cause: str) -> None:
        super().__init__(cause)
        self.cause = cause

    def message(self) -> str:
        return (
            f"Eugene's store is not answering: {self.cause}. Chat keeps working; files wait for it."
        )


class StoreRefusal(Exception):
    """The store answered with a refusal (4xx): status and its own words."""

    def __init__(self, status: int, title: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.title = title
        self.detail = detail


def _problem(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {"detail": response.text[:300]}
    if isinstance(body, Mapping):
        inner = body.get("detail", body)
        if isinstance(inner, Mapping):
            return dict(inner)
        return {"detail": str(inner)}
    return {"detail": str(body)[:300]}


class StoreClient:
    """One HTTP client for the gateway's life."""

    def __init__(
        self,
        agent_url: str | None,
        service_token: str | None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base = f"{agent_url.rstrip('/')}/api/proxy/store" if agent_url else None
        headers = {"Authorization": f"Bearer {service_token}"} if service_token else {}
        self._client = internal_client(headers=headers, timeout=TIMEOUT, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _check(self, response: httpx.Response) -> httpx.Response:
        if response.status_code < 400:
            return response
        problem = _problem(response)
        detail = str(problem.get("detail") or problem.get("title") or response.status_code)
        # The agent's proxy answers 502/503 in its own name when it cannot
        # reach the store; the store answers 503 itself when it cannot open
        # its data. Either way nothing was kept or read.
        if response.status_code in (502, 503, 504) or problem.get("component") == "agent":
            raise StoreUnavailable(detail.rstrip("."))
        raise StoreRefusal(response.status_code, str(problem.get("title") or ""), detail)

    async def call(
        self,
        method: str,
        path: str,
        owner: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        content: bytes | None = None,
        time_limit: httpx.Timeout | None = None,
    ) -> Any:
        """One JSON answer from the store, or `StoreRefusal` / `StoreUnavailable`.
        `time_limit` replaces the client's own for this call (row 54: a stored
        response's read and keep wait seconds, not an upload's minutes)."""
        response = await self._send(
            method, path, owner, params=params, json=json, content=content, time_limit=time_limit
        )
        try:
            return response.json()
        except ValueError as exc:
            raise StoreUnavailable("its answer was not JSON") from exc

    async def stream(self, path: str, owner: str) -> tuple[httpx.Response, AsyncIterator[bytes]]:
        """A file's bytes, as the store sends them. The caller closes the response."""
        if self.base is None:
            raise StoreUnavailable("this gateway has no agent to reach it through")
        request = self._client.build_request("GET", self.base + path, headers={OWNER_HEADER: owner})
        try:
            response = await self._client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise StoreUnavailable(_observed(exc)) from exc
        if response.status_code >= 400:
            await response.aread()
            await response.aclose()
            self._check(response)
        return response, response.aiter_bytes()

    async def read(self, path: str, owner: str, *, limit: int) -> tuple[bytes, str]:
        """A whole file's bytes and its media type, refused past `limit`."""
        response, chunks = await self.stream(path, owner)
        try:
            data = bytearray()
            async for chunk in chunks:
                data.extend(chunk)
                if len(data) > limit:
                    raise StoreRefusal(413, "Too large", f"larger than {limit} bytes")
            return bytes(data), response.headers.get("content-type", "application/octet-stream")
        finally:
            await response.aclose()

    async def _send(
        self,
        method: str,
        path: str,
        owner: str,
        *,
        params: Mapping[str, Any] | None,
        json: Any,
        content: bytes | None,
        time_limit: httpx.Timeout | None = None,
    ) -> httpx.Response:
        if self.base is None:
            raise StoreUnavailable("this gateway has no agent to reach it through")
        headers = {OWNER_HEADER: owner}
        if content is not None:
            headers["Content-Type"] = "application/octet-stream"
        try:
            response = await self._client.request(
                method,
                self.base + path,
                params={k: v for k, v in (params or {}).items() if v is not None},
                json=json,
                content=content,
                headers=headers,
                timeout=time_limit if time_limit is not None else httpx.USE_CLIENT_DEFAULT,
            )
        except httpx.HTTPError as exc:
            raise StoreUnavailable(_observed(exc)) from exc
        return self._check(response)


def _observed(exc: Exception) -> str:
    said = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {said}" if said else type(exc).__name__
