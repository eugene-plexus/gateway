"""Open WebUI's external-search wire shape, using the chat search runner."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from .server_tools import Execution, SearchPlan, SearchRunner
from .tool_client import ToolDriverError


def parse(body: Any) -> tuple[str, int]:
    query = body.get("query") if isinstance(body, dict) else None
    if not isinstance(query, str) or not 1 <= len(query) <= 2000 or not query.strip():
        raise ValueError("query: must be 1 to 2000 characters.")
    count = body.get("count", 20)
    if type(count) is not int or not 1 <= count <= 20:
        raise ValueError("count: must be an integer from 1 to 20.")
    return query, count


class _Client:
    def __init__(self, client: Any, failures: list[ToolDriverError]) -> None:
        self.client, self.failures = client, failures

    async def web_search(self, request: Any) -> Any:
        try:
            return await self.client.web_search(request)
        except ToolDriverError as exc:
            self.failures.append(exc)
            raise


class _Accounts:
    """Observe retry hints without changing SearchRunner's account failover."""

    def __init__(self, table: Any) -> None:
        self.failures: list[ToolDriverError] = []
        self.accounts = [
            replace(a, client=_Client(a.client, self.failures))
            for a in table.search_accounts("web_search")
        ]

    def search_accounts(self, _tool: str) -> list[Any]:
        return self.accounts


async def run(
    table: Any, query: str, count: int
) -> tuple[Execution, list[dict[str, str]], int, str | None]:
    accounts = _Accounts(table)
    result = await SearchRunner(accounts, SearchPlan(version="external")).run(
        Execution(call_id="external", query=query, version="external")
    )
    status = 200
    retry = None
    if result.outcome != "ok":
        status = (
            429
            if result.error_code == "too_many_requests"
            else 400
            if result.error_code == "invalid_input"
            else 503
        )
        if status == 429 and accounts.failures:
            hint = accounts.failures[0].retry_after
            if hint is not None and math.isfinite(hint) and hint >= 0:
                retry = str(int(hint)) if hint == int(hint) else str(hint)
    shaped = [
        {
            "link": str(r.get("url") or ""),
            "title": str(r.get("title") or ""),
            "snippet": str(r.get("snippet") or ""),
        }
        for r in result.results[:count]
    ]
    return result, shaped, status, retry
