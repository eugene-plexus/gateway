"""The gateway's side of a tool-driver (P8): one client per search account.

A tool-driver runs a tool the hub runs itself -- `web_search` today --
for one provider account, and holds that account's secret. The gateway
never holds a search key; it holds this client, which presents the same
service token it presents to an inference-driver on the same machine.

Kept apart from `driver_client.py` on purpose: an inference-driver
generates, a tool-driver searches, and the two share nothing but the
transport rules in `_http.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ._generated.tool_driver_models import WebSearchRequest
from ._http import internal_client

#: How long one search may take before it is reported to the model as
#: failed. A search the model waits on is a pause in someone's answer,
#: and a provider that has not answered in this long is not going to.
SEARCH_TIMEOUT_SECONDS = 30.0


class ToolDriverError(Exception):
    """A tool-driver refused or failed a call.

    `status` is the tool-driver's own HTTP status (0 when it never
    answered); `detail` is its problem's explanation, which names the
    provider's cause -- a refused Brave token, a SearXNG with JSON output
    off -- and is what the model, and then the person, is told.
    `retry_after` is the provider's `Retry-After`, when it gave one.
    """

    def __init__(
        self,
        *,
        driver: str,
        status: int,
        detail: str,
        code: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(f"tool-driver {driver!r}: {detail}")
        self.driver = driver
        self.status = status
        self.detail = detail
        self.code = code
        self.retry_after = retry_after

    @property
    def worth_another_account(self) -> bool:
        """Whether a second search account could do better.

        A provider that is down, busy, slow or refused our credential is
        a failure of THIS account, and the next account may well answer.
        A 400 is the query itself, which the next account would refuse
        in the same words.
        """
        return self.status == 0 or self.status == 429 or self.status >= 500


@dataclass
class ToolDriverInfo:
    """A tool-driver's `/v1/info`, read leniently.

    Not the generated model on purpose: its `tools` is an enum, and a
    newer tool-driver naming a tool this gateway does not know yet must
    still be usable for the tools it does -- a strict read would make the
    whole account unreachable over one new word.
    """

    provider: str
    label: str
    tools: list[str]
    configured: bool
    version: str | None = None
    #: `free` or `per_search`. An account that does not say is read as
    #: billing: nothing is assumed to cost nothing.
    billing: str = "per_search"


@dataclass
class SearchAnswer:
    results: list[dict[str, Any]]
    answer: str | None
    provider: str
    ignored: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    #: HTML the provider's terms require shown with these results (Google's
    #: Search Suggestions), carried to the caller verbatim and never stored.
    search_suggestions: str | None = None


class ToolDriverClient:
    """HTTP client for one tool-driver, built once and kept by the table."""

    def __init__(
        self, *, name: str, base_url: str, node: str | None, auth: httpx.Auth | None = None
    ) -> None:
        self.name = name
        self.node = node
        self.base_url = base_url.rstrip("/")
        # `internal_client`: a tool-driver is this machine or another node
        # of this install, never the public internet -- the provider behind
        # it is, and that hop is the tool-driver's, with the user's proxy.
        self._client = internal_client(
            base_url=self.base_url,
            timeout=httpx.Timeout(SEARCH_TIMEOUT_SECONDS, connect=5.0),
            auth=auth,
        )

    async def info(self) -> ToolDriverInfo:
        response = await self._client.get("/v1/info")
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("tool-driver /v1/info answered something that is not an object")
        tools = body.get("tools")
        return ToolDriverInfo(
            provider=str(body.get("provider") or "unknown"),
            label=str(body.get("label") or body.get("provider") or "search"),
            tools=[t for t in tools if isinstance(t, str)] if isinstance(tools, list) else [],
            configured=bool(body.get("configured", True)),
            version=body.get("version") if isinstance(body.get("version"), str) else None,
            billing="free" if body.get("billing") == "free" else "per_search",
        )

    async def web_search(self, request: WebSearchRequest) -> SearchAnswer:
        """One search. The request is the contract's model, so what the
        gateway sends is what `tool-driver.yaml` says; the answer is read
        leniently, as `info` is."""
        started = time.perf_counter()
        payload = request.model_dump(mode="json", exclude_none=True)
        try:
            response = await self._client.post("/v1/tools/web_search", json=payload)
        except httpx.TimeoutException:
            raise ToolDriverError(
                driver=self.name,
                status=0,
                detail=f"the search account did not answer within {SEARCH_TIMEOUT_SECONDS:g}s",
                code="timeout",
            ) from None
        except httpx.HTTPError as exc:
            raise ToolDriverError(
                driver=self.name,
                status=0,
                detail=f"the search account could not be reached ({type(exc).__name__})",
                code="unreachable",
            ) from None
        elapsed = int((time.perf_counter() - started) * 1000)
        if response.status_code >= 400:
            raise _failure(self.name, response)
        try:
            body = response.json()
        except ValueError:
            raise ToolDriverError(
                driver=self.name,
                status=502,
                detail="the search account answered something unreadable",
            ) from None
        results = body.get("results") if isinstance(body, dict) else None
        return SearchAnswer(
            results=[r for r in results if isinstance(r, dict)]
            if isinstance(results, list)
            else [],
            answer=body.get("answer") if isinstance(body.get("answer"), str) else None,
            provider=str(body.get("provider") or "unknown"),
            ignored=[i for i in body.get("ignored") or [] if isinstance(i, str)],
            elapsed_ms=elapsed,
            search_suggestions=body.get("searchSuggestions")
            if isinstance(body.get("searchSuggestions"), str) and body.get("searchSuggestions")
            else None,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _failure(driver: str, response: httpx.Response) -> ToolDriverError:
    detail = f"HTTP {response.status_code}"
    code = None
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        problem = body.get("detail") if isinstance(body.get("detail"), dict) else body
        if isinstance(problem, dict):
            detail = str(problem.get("detail") or problem.get("title") or detail)
            code = problem.get("code") if isinstance(problem.get("code"), str) else None
        elif isinstance(body.get("detail"), str):
            detail = body["detail"]
    retry_after = None
    header = response.headers.get("retry-after")
    if header:
        try:
            retry_after = float(header)
        except ValueError:
            retry_after = None
    return ToolDriverError(
        driver=driver,
        status=response.status_code,
        detail=detail,
        code=code,
        retry_after=retry_after,
    )
