"""Server-run tools (P8): the loop that runs a web search for a model.

OpenAI's and Anthropic's APIs define `web_search` as a tool the provider
runs. On a local model nobody did: Codex's search was removed and Claude
Code's WebSearch was refused (`server-run-tools.md` §0). This module is
the gateway's half of running it here:

* **`SearchPlan`** -- one request for a search, however the door spelled
  it: chat's `web_search_options`, a Responses `web_search` tool, or an
  Anthropic `web_search_<date>` server tool.
* **`why_not`** -- the one place that decides whether a search may run
  for this request: a search account exists, the key's `allowedTools`
  permits it, and the key is not local-only.
* **`SearchingClient`** -- wraps the routed `TieredClient` and runs the
  loop: offer the model a `web_search` function, run each call on a
  search account (a tool-driver), hand the results back, call the model
  again, up to the request's limit. A backend that searches itself is
  sent the request natively instead, and the loop never runs.

The rules, each for a reason recorded in the design:

* **Failover ends at the first search.** The first model call may cascade
  as any request does; every later call goes to the backend that
  answered the first (`TieredClient.pin`), because a search is a side
  effect whose results are now in the conversation.
* **A turn that asks for a search and calls one of the caller's own
  functions has those calls dropped** and is asked again with the
  results, so the caller only ever sees function calls from a turn with
  no search pending -- nothing about a search survives between two of the
  caller's requests.
* **A search that fails is not a failed request.** The model is told the
  search failed and why, and answers without it.
* **When search is the only tool the caller offered, the first call must
  call a tool**: the caller asked for a search, not for the model's
  opinion on whether to search.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from ._generated.driver_models import ChatAnnotation as DriverChatAnnotation
from ._generated.driver_models import (
    FinishReason,
    FunctionDefinition,
    GenerateRequest,
    GenerateResponse,
    Message,
    Role,
    Tool,
    ToolCall,
    ToolChoice,
    Usage,
)
from ._generated.driver_models import WebSearchOptions as DriverWebSearchOptions
from ._generated.tool_driver_models import WebSearchRequest
from .driver_client import DriverClient, StreamEvent
from .model_patterns import permits
from .tool_client import SearchAnswer, ToolDriverError

log = logging.getLogger(__name__)

WEB_SEARCH = "web_search"
#: Design call #6: five searches a request unless the caller or the
#: operator says otherwise.
DEFAULT_MAX_TOOL_CALLS = 5
#: Anthropic's tool is dated, and any date is accepted (Troy, 2026-09-29):
#: refusing an unknown one would break WebSearch the day Claude Code
#: sends a newer date, as `output_config` once 400'd every request.
ANTHROPIC_TOOL = re.compile(r"^web_search_\d{8}$")
#: The Anthropic tool settings this gateway honours. Any other is dropped
#: and named, never assumed to mean what the old date's field meant.
ANTHROPIC_KNOWN = frozenset(
    {"type", "name", "max_uses", "allowed_domains", "blocked_domains", "user_location"}
)
#: `cache_control` rides on any Anthropic tool and changes nothing here.
ANTHROPIC_SILENT = frozenset({"cache_control"})


# --------------------------------------------------------------------------- #
# The plan
# --------------------------------------------------------------------------- #


@dataclass
class SearchPlan:
    """One request for a search, in the gateway's words."""

    #: How the caller named it: `web_search_options`, `web_search`, or
    #: Anthropic's dated tool. Kept on every execution's metrics row.
    version: str
    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    #: `{"type": "approximate", "approximate": {...}}`, the shape all three
    #: doors share.
    user_location: dict[str, Any] | None = None
    context_size: str | None = None
    #: The caller offered no function tools of its own.
    only_tool: bool = False
    #: Tool settings dropped and named on the ignored-settings header.
    ignored: list[str] = field(default_factory=list)
    #: What the door does when a search cannot run here. `native` (chat):
    #: keep P2c's rule, only a model that searches itself. `skip`
    #: (Responses): remove the tool and say why, as before P8. `refuse`
    #: (Anthropic): a 400 naming the reason, because Claude Code's WebSearch
    #: request exists only to search.
    when_blocked: str = "native"

    def limit(self, install_default: int | None) -> int:
        chosen = self.max_uses if self.max_uses is not None else install_default
        return max(1, int(chosen or DEFAULT_MAX_TOOL_CALLS))

    def native_options(self) -> DriverWebSearchOptions:
        """The same request, as a backend that searches itself reads it."""
        values: dict[str, Any] = {}
        if self.context_size:
            values["search_context_size"] = self.context_size
        if self.user_location:
            values["user_location"] = self.user_location
        return DriverWebSearchOptions.model_validate(values)


def _location(value: Any) -> dict[str, Any] | None:
    """A user location in the shape the chat door and a tool-driver share.

    Responses and Anthropic send `{"type": "approximate", "city", ...}`
    flat; chat nests the fields under `approximate`. Either is accepted.
    """
    if not isinstance(value, Mapping):
        return None
    nested = value.get("approximate")
    inner: Mapping[str, Any] = nested if isinstance(nested, Mapping) else value
    kept = {
        k: str(inner[k])
        for k in ("city", "country", "region", "timezone")
        if isinstance(inner.get(k), str) and inner[k]
    }
    return {"type": "approximate", "approximate": kept} if kept else None


def _domains(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    kept = [d.strip() for d in value if isinstance(d, str) and d.strip()]
    return kept or None


def plan_from_chat(options: Any, *, has_tools: bool) -> SearchPlan:
    """Chat's `web_search_options` (a pydantic model or a mapping)."""
    raw = options.model_dump(mode="json") if hasattr(options, "model_dump") else dict(options or {})
    return SearchPlan(
        version="web_search_options",
        user_location=_location(raw.get("user_location")),
        context_size=raw.get("search_context_size"),
        only_tool=not has_tools,
    )


def plan_from_responses(
    definition: Mapping[str, Any], *, max_tool_calls: Any, has_functions: bool
) -> SearchPlan:
    given = definition.get("filters")
    filters: Mapping[str, Any] = given if isinstance(given, Mapping) else {}
    return SearchPlan(
        version=str(definition.get("type") or WEB_SEARCH),
        max_uses=max_tool_calls if isinstance(max_tool_calls, int) and max_tool_calls > 0 else None,
        allowed_domains=_domains(filters.get("allowed_domains")),
        user_location=_location(definition.get("user_location")),
        context_size=definition.get("search_context_size")
        if definition.get("search_context_size") in ("low", "medium", "high")
        else None,
        only_tool=not has_functions,
    )


def plan_from_anthropic(definition: Mapping[str, Any], *, has_functions: bool) -> SearchPlan:
    ignored = [
        f"tools.{definition.get('name') or WEB_SEARCH}.{key}"
        for key in definition
        if key not in ANTHROPIC_KNOWN and key not in ANTHROPIC_SILENT
    ]
    max_uses = definition.get("max_uses")
    return SearchPlan(
        version=str(definition.get("type")),
        max_uses=max_uses if isinstance(max_uses, int) and max_uses > 0 else None,
        allowed_domains=_domains(definition.get("allowed_domains")),
        blocked_domains=_domains(definition.get("blocked_domains")),
        user_location=_location(definition.get("user_location")),
        only_tool=not has_functions,
        ignored=ignored,
    )


# --------------------------------------------------------------------------- #
# Whether a search may run
# --------------------------------------------------------------------------- #


def why_not(table: Any, context: Any) -> str | None:
    """None when a search may run for this request, else why not, in words.

    The order is the order an operator would fix things in, and the
    wording is what the caller -- and through them a person -- reads.
    """
    if context is not None and getattr(context, "local_only", False):
        return (
            "this key is local-only, and a web search sends a query made from the prompt "
            "to the internet"
        )
    allowed = getattr(context, "allowed_tools", None) if context is not None else None
    if not permits(allowed, WEB_SEARCH):
        return "this key's tool scope does not include web_search"
    if table is None:
        return "the gateway is starting up and has no search account yet"
    if table.search_accounts(WEB_SEARCH):
        return None
    accounts = table.tool_accounts()
    if accounts:
        names = ", ".join(sorted({a.name for a in accounts}))
        return (
            f"the install's search account ({names}) is not set up yet -- open it under "
            "Backends and give it an address or a key"
        )
    return "no search account is set up; add one under Backends, then Add a search account"


def searchable(resolution: Any) -> tuple[Any, set[tuple[str | None, str, str | None]]]:
    """The candidates a search can reach, and which of them search themselves.

    A backend whose model lists `webSearchOptions` is sent the request
    natively -- forwarded, not run twice. Any other must call tools, since
    this install's search is offered to the model as one. The rest cannot
    take part and are left out of every tier (empty tiers keep their
    numbers, as everywhere else).
    """
    native: set[tuple[str | None, str, str | None]] = set()
    tiers = []
    for tier in resolution.tiers:
        kept = []
        for backend in tier.backends:
            model = getattr(backend, "model", None)
            caps = getattr(model, "capabilities", None)
            settings = set(getattr(caps, "supportedSettings", None) or [])
            if "webSearchOptions" in settings:
                native.add((backend.node, backend.name, backend.public_id or None))
                kept.append(backend)
            elif caps is not None and getattr(caps, "toolCalling", None) is True:
                kept.append(backend)
        tiers.append(replace(tier, backends=kept))
    return replace(resolution, tiers=tiers), native


# --------------------------------------------------------------------------- #
# One search
# --------------------------------------------------------------------------- #


@dataclass
class Execution:
    """One `web_search` call the model made, and what came of it."""

    call_id: str
    query: str
    version: str
    #: `ok`, `error`, or `over_limit` (answered without running).
    outcome: str = "pending"
    results: list[dict[str, Any]] = field(default_factory=list)
    answer: str | None = None
    provider: str | None = None
    driver: str | None = None
    node: str | None = None
    elapsed_ms: int = 0
    error: str | None = None
    #: Anthropic's `web_search_tool_result_error.error_code` for a failure.
    error_code: str | None = None
    #: A stable id for the door's own record of this search.
    item_id: str = field(default_factory=lambda: uuid.uuid4().hex[:24])

    @property
    def sources(self) -> list[dict[str, Any]]:
        return self.results


@dataclass(frozen=True)
class SearchUpdate:
    """What a door is told while the loop runs: a search started, or ended."""

    phase: str  # "started" | "completed"
    execution: Execution


class SearchRunner:
    """Runs one search on the install's search accounts, in order.

    A second account is tried only when the first could not do the job
    (down, busy, slow, refused our key) -- never after it refused the
    query itself, which the next one would refuse in the same words.
    """

    def __init__(self, table: Any, plan: SearchPlan) -> None:
        self.table = table
        self.plan = plan

    async def run(self, execution: Execution) -> Execution:
        started = time.perf_counter()
        query = execution.query.strip()
        if not query:
            execution.outcome, execution.error_code = "error", "invalid_input"
            execution.error = "the search had no query"
            return execution
        if len(query) > 2000:
            execution.outcome, execution.error_code = "error", "query_too_long"
            execution.error = "the query was longer than 2000 characters"
            return execution
        request = WebSearchRequest.model_validate(
            {
                "query": query,
                **(
                    {"allowedDomains": self.plan.allowed_domains}
                    if self.plan.allowed_domains
                    else {}
                ),
                **(
                    {"blockedDomains": self.plan.blocked_domains}
                    if self.plan.blocked_domains
                    else {}
                ),
                **({"userLocation": self.plan.user_location} if self.plan.user_location else {}),
                **({"contextSize": self.plan.context_size} if self.plan.context_size else {}),
            }
        )
        last: ToolDriverError | None = None
        for account in self.table.search_accounts(WEB_SEARCH):
            execution.driver, execution.node = account.name, account.node
            try:
                answer: SearchAnswer = await account.client.web_search(request)
            except ToolDriverError as exc:
                last = exc
                log.info("search account %r failed: %s", account.name, exc.detail)
                if not exc.worth_another_account:
                    break
                continue
            execution.outcome = "ok"
            execution.results = answer.results
            execution.answer = answer.answer
            execution.provider = answer.provider
            execution.elapsed_ms = int((time.perf_counter() - started) * 1000)
            return execution
        execution.outcome = "error"
        execution.elapsed_ms = int((time.perf_counter() - started) * 1000)
        if last is None:
            execution.error = "no search account could be reached"
            execution.error_code = "unavailable"
        else:
            execution.error = last.detail
            execution.error_code = (
                "too_many_requests"
                if last.status == 429
                else "invalid_input"
                if last.status == 400
                else "unavailable"
            )
        return execution


# --------------------------------------------------------------------------- #
# What the model is offered and told
# --------------------------------------------------------------------------- #


def search_tool(name: str) -> Tool:
    return Tool(
        type="function",
        function=FunctionDefinition(
            name=name,
            description=(
                "Search the web. Returns titles, addresses, excerpts and dates of pages "
                "that match the query. Use it for anything recent, anything you are not "
                "sure of, or when you are asked to look something up. Cite the pages you "
                "use as markdown links."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search the web for."}
                },
                "required": ["query"],
            },
        ),
    )


def tool_name_for(caller_tools: Any) -> str:
    """`web_search`, unless the caller already declared a function by that name."""
    taken = {getattr(getattr(t, "function", None), "name", None) for t in caller_tools or []}
    for name in (WEB_SEARCH, "eugene_web_search", "eugene_plexus_web_search"):
        if name not in taken:
            return name
    return f"web_search_{uuid.uuid4().hex[:6]}"


def query_of(arguments: str | None) -> str:
    """The model's query, from arguments that should be `{"query": ...}`.

    A small model sometimes sends the query as a bare string, or with the
    key misspelled; the first string it gave is taken rather than the
    search refused.
    """
    if not arguments:
        return ""
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return arguments.strip()
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, Mapping):
        value = parsed.get("query")
        if isinstance(value, str):
            return value
        for other in parsed.values():
            if isinstance(other, str):
                return other
    return ""


def result_text(execution: Execution, *, limit: int) -> str:
    """The tool message the model reads for one search."""
    if execution.outcome == "over_limit":
        return (
            f"No more searches are available for this request (its limit is {limit}). "
            "Answer with what you have already found."
        )
    if execution.outcome != "ok":
        return (
            f'The web search for "{execution.query}" could not be run: {execution.error}. '
            "Answer without it, and say that the search failed."
        )
    if not execution.results and not execution.answer:
        return f'The web search for "{execution.query}" found nothing.'
    lines = [f'Web search results for "{execution.query}":', ""]
    if execution.answer:
        lines += [f"Answer from the search provider: {execution.answer}", ""]
    for index, result in enumerate(execution.results, start=1):
        lines.append(f"{index}. {result.get('title') or result.get('url')}")
        lines.append(f"   {result.get('url')}")
        if result.get("publishedAt"):
            lines.append(f"   Published: {result['publishedAt']}")
        if result.get("snippet"):
            lines.append(f"   {result['snippet']}")
        lines.append("")
    lines.append("Cite the pages you use as markdown links, like [title](address).")
    return "\n".join(lines).rstrip()


def citations(text: str | None, executions: list[Execution]) -> list[dict[str, Any]]:
    """`url_citation` annotations for each result address the text cites.

    Only an address that appears in the answer is a citation: annotating
    every result the model was given would claim the answer cites pages it
    never mentions. A markdown link `[title](url)` is cited whole.
    """
    if not text:
        return []
    seen: set[str] = set()
    found: list[dict[str, Any]] = []
    for execution in executions:
        for result in execution.results:
            url = result.get("url")
            if not isinstance(url, str) or not url or url in seen:
                continue
            seen.add(url)
            index = -1
            spelled = url
            for candidate in (url, url.rstrip("/")):
                index = text.find(candidate)
                if index >= 0:
                    spelled = candidate
                    break
            if index < 0:
                continue
            start, end = index, index + len(spelled)
            if text[max(0, start - 2) : start] == "](":
                opening = text.rfind("[", 0, start - 2)
                if opening >= 0 and text[end : end + 1] == ")":
                    start, end = opening, end + 1
            found.append(
                {
                    "type": "url_citation",
                    "url_citation": {
                        "url": url,
                        "title": str(result.get("title") or url),
                        "start_index": start,
                        "end_index": end,
                    },
                }
            )
    found.sort(key=lambda a: a["url_citation"]["start_index"])
    return found


def _sum_usage(total: Usage | None, more: Usage | None) -> Usage | None:
    if more is None:
        return total
    if total is None:
        return more.model_copy()
    values: dict[str, Any] = {}
    for name in (
        "promptTokens",
        "completionTokens",
        "totalTokens",
        "cachedPromptTokens",
        "reasoningTokens",
    ):
        a, b = getattr(total, name), getattr(more, name)
        values[name] = None if a is None and b is None else (a or 0) + (b or 0)
    return Usage(**values)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #

Prepare = Callable[[DriverClient, GenerateRequest], Awaitable[GenerateRequest]]


class SearchingClient:
    """The routed client, with the search loop around it.

    Everything a door reads off a client -- `served_by`, `tier`,
    `served_model` -- is the wrapped client's, except `attempts`, which is
    the FIRST call's: later calls go to one pinned backend and are turns
    of one answer, not backends tried.
    """

    def __init__(
        self,
        inner: Any,
        *,
        plan: SearchPlan,
        runner: SearchRunner,
        limit: int,
        native: set[tuple[str | None, str, str | None]],
    ) -> None:
        self._inner = inner
        self.plan = plan
        self.runner = runner
        self.limit = limit
        #: `(node, driver, public model)` of the candidates that search
        #: themselves and are sent the request natively.
        self._native = native
        self.executions: list[Execution] = []
        #: What happened, in order: `("text", str)`, `("reasoning", str)`,
        #: `("search", Execution)`. The Responses and Anthropic doors render
        #: their items and blocks from it.
        self.transcript: list[tuple[str, Any]] = []
        self.turns = 0
        self.first_attempts: int | None = None
        self.searched_natively = False
        self._tool_name = WEB_SEARCH
        self._base: GenerateRequest | None = None
        previous: Prepare | None = getattr(inner, "prepare_request", None)
        if hasattr(inner, "prepare_request"):
            inner.prepare_request = self._prepared(previous)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def attempts(self) -> int:
        return self.first_attempts if self.first_attempts is not None else self._inner.attempts

    @property
    def max_turns(self) -> int:
        # Every search the limit allows, one turn to read the last results,
        # and one more for a model that asks past the limit to be told no.
        return self.limit + 2

    # -- per candidate: native, or the loop ------------------------------------

    def _is_native(self, candidate: Any) -> bool:
        key = (
            getattr(candidate, "node", None),
            getattr(candidate, "name", ""),
            getattr(candidate, "public_model", None),
        )
        return key in self._native

    def _prepared(self, previous: Prepare | None) -> Prepare:
        async def prepare(candidate: DriverClient, request: GenerateRequest) -> GenerateRequest:
            if self._is_native(candidate) and self._base is not None and self.turns == 0:
                request = self._native_request(self._base)
            return await previous(candidate, request) if previous is not None else request

        return prepare

    def _native_request(self, base: GenerateRequest) -> GenerateRequest:
        settings = set(base.callerSettings or []) | {"webSearchOptions"}
        return base.model_copy(
            update={
                "webSearchOptions": self.plan.native_options(),
                "callerSettings": sorted(settings),
            }
        )

    def _turn_request(
        self, base: GenerateRequest, history: list[Message], turn: int
    ) -> GenerateRequest:
        last = turn >= self.max_turns - 1
        caller_tools = list(base.tools or [])
        tools = caller_tools if last else [*caller_tools, search_tool(self._tool_name)]
        settings = set(base.callerSettings or []) - {"webSearchOptions"}
        if tools:
            settings.add("tools")
        else:
            settings.discard("tools")
        choice = base.toolChoice
        if last and not caller_tools:
            choice = None
        elif turn == 0 and self.plan.only_tool and (choice is None or choice == ToolChoice.auto):
            # The caller asked for a search; the first call makes one.
            choice = ToolChoice.required
        return base.model_copy(
            update={
                "messages": history,
                "tools": tools or None,
                "toolChoice": choice,
                "webSearchOptions": None,
                "callerSettings": sorted(settings) or None,
            }
        )

    def _after_turn(self, turn: int) -> None:
        self.turns = turn + 1
        if turn == 0:
            self.first_attempts = getattr(self._inner, "attempts", 1)
            served = (
                getattr(self._inner, "served_by_node", None),
                getattr(self._inner, "served_by", "") or "",
                getattr(self._inner, "served_model", None),
            )
            self.searched_natively = served in self._native
            pin = getattr(self._inner, "pin", None)
            if pin is not None:
                pin()

    def _split(self, calls: list[ToolCall] | None) -> tuple[list[ToolCall], list[ToolCall]]:
        searches = [c for c in calls or [] if c.function.name == self._tool_name]
        others = [c for c in calls or [] if c.function.name != self._tool_name]
        return searches, others

    async def _run(self, call: ToolCall) -> Execution:
        execution = Execution(
            call_id=call.id, query=query_of(call.function.arguments), version=self.plan.version
        )
        if sum(1 for e in self.executions if e.outcome != "over_limit") >= self.limit:
            execution.outcome, execution.error_code = "over_limit", "max_uses_exceeded"
        else:
            await self.runner.run(execution)
        self.executions.append(execution)
        return execution

    def _history_after(
        self,
        history: list[Message],
        text: str,
        reasoning: str | None,
        searches: list[ToolCall],
        done: list[Execution],
    ) -> list[Message]:
        history = list(history)
        history.append(
            Message(
                role=Role.assistant,
                content=text or None,
                toolCalls=[c.model_dump(mode="json") for c in searches],
                reasoning=reasoning or None,
            )
        )
        for call, execution in zip(searches, done, strict=True):
            history.append(
                Message(
                    role=Role.tool,
                    toolCallId=call.id,
                    content=result_text(execution, limit=self.limit),
                )
            )
        return history

    def _final(
        self, response: GenerateResponse, others: list[ToolCall], usage: Usage | None
    ) -> GenerateResponse:
        texts = [value for kind, value in self.transcript if kind == "text" and value]
        reasoning = [value for kind, value in self.transcript if kind == "reasoning" and value]
        content = "\n\n".join(texts) or None
        finish = response.finishReason
        if not others and finish == FinishReason.tool_calls:
            finish = FinishReason.stop
        annotations = list(response.annotations or [])
        if self.executions:
            annotations += [
                DriverChatAnnotation.model_validate(a) for a in citations(content, self.executions)
            ]
        return response.model_copy(
            update={
                "content": content,
                "reasoning": "\n\n".join(reasoning) or None,
                "toolCalls": others or None,
                "finishReason": finish,
                "usage": usage,
                "annotations": annotations or None,
            }
        )

    # -- the two ways a door calls -------------------------------------------

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self._base = request
        self._tool_name = tool_name_for(request.tools)
        history = list(request.messages)
        usage: Usage | None = None
        for turn in range(self.max_turns):
            response = await self._inner.generate(self._turn_request(request, history, turn))
            self._after_turn(turn)
            usage = _sum_usage(usage, response.usage)
            if response.reasoning:
                self.transcript.append(("reasoning", response.reasoning))
            if response.content:
                self.transcript.append(("text", response.content))
            searches, others = self._split(response.toolCalls)
            if not searches or self.searched_natively or turn >= self.max_turns - 1:
                return self._final(response, others, usage)
            done = []
            for call in searches:
                execution = await self._run(call)
                self.transcript.append(("search", execution))
                done.append(execution)
            history = self._history_after(
                history, response.content or "", response.reasoning, searches, done
            )
        raise AssertionError("unreachable: the last turn always returns")

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        self._base = request
        self._tool_name = tool_name_for(request.tools)
        history = list(request.messages)
        usage: Usage | None = None
        for turn in range(self.max_turns):
            fragments: dict[int, list[dict[str, Any]]] = {}
            names: dict[int, str] = {}
            result: GenerateResponse | None = None
            text: list[str] = []
            thought: list[str] = []
            async for event in self._inner.stream(self._turn_request(request, history, turn)):
                if event.done:
                    result = event.result
                    continue
                if event.tool_calls:
                    # Held until the turn ends: a search call must never
                    # reach the caller, and whether a turn's other calls do
                    # depends on whether it also asked for a search.
                    for fragment in event.tool_calls:
                        index = int(fragment.get("index") or 0)
                        fragments.setdefault(index, []).append(fragment)
                        name = (fragment.get("function") or {}).get("name")
                        if isinstance(name, str) and name:
                            names[index] = name
                    continue
                if event.text:
                    text.append(event.text)
                if event.reasoning:
                    thought.append(event.reasoning)
                yield event
            self._after_turn(turn)
            if result is None:
                # The backend stopped without saying it had finished. Not
                # ours to dress up as a completion: the door reports the
                # truncation exactly as it would without a search.
                return
            usage = _sum_usage(usage, result.usage)
            if thought:
                self.transcript.append(("reasoning", "".join(thought)))
            if text:
                self.transcript.append(("text", "".join(text)))
            searches, others = self._split(result.toolCalls)
            if not searches or self.searched_natively or turn >= self.max_turns - 1:
                for index in sorted(fragments):
                    if names.get(index) == self._tool_name:
                        continue
                    for fragment in fragments[index]:
                        yield StreamEvent(tool_calls=[fragment])
                final = self._final(result, others, usage)
                cited = citations(final.content, self.executions) if self.executions else []
                if cited:
                    yield StreamEvent(annotations=cited)
                yield StreamEvent(done=True, result=final)
                return
            done = []
            for call in searches:
                pending = Execution(
                    call_id=call.id,
                    query=query_of(call.function.arguments),
                    version=self.plan.version,
                )
                yield StreamEvent(search=SearchUpdate("started", pending))
                execution = await self._run(call)
                execution.item_id = pending.item_id
                self.transcript.append(("search", execution))
                done.append(execution)
                yield StreamEvent(search=SearchUpdate("completed", execution))
            history = self._history_after(
                history, "".join(text), "".join(thought) or None, searches, done
            )
