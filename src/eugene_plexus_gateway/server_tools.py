"""Server-run tools (P8): the loop that runs a web search for a model,
and makes an image for one (P8e).

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

`image_generation` (P8e, Responses only) rides the same loop: an
`ImagePlan`, `image_model` to choose which image model answers, and an
`ImageRunner` that routes each call exactly as the images door would.
The model is told in words that an image was made; the bytes go to the
caller and never back into the prompt. One budget covers both tools.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import httpx

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
from .chat_contract import _field_name
from .driver_client import DriverClient, DriverError, StreamEvent
from .image_doors import ImageAsk, format_name, to_driver
from .model_patterns import permits
from .tool_client import SearchAnswer, ToolDriverError

log = logging.getLogger(__name__)

WEB_SEARCH = "web_search"
IMAGE_GENERATION = "image_generation"
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


#: What this gateway reads off a Responses `web_search` tool. Every other
#: key -- `search_content_types`, `image_settings`, `return_token_budget`,
#: Codex's `indexed_web_access` -- is named, never assumed honoured.
#: `external_web_access` is read by the door, which decides from it whether
#: the search may run at all.
RESPONSES_KNOWN = frozenset(
    {"type", "filters", "user_location", "search_context_size", "external_web_access"}
)
RESPONSES_FILTERS = frozenset({"allowed_domains", "blocked_domains"})


def plan_from_responses(
    definition: Mapping[str, Any], *, max_tool_calls: Any, has_functions: bool
) -> SearchPlan:
    """A Responses `web_search` tool.

    `filters.blocked_domains` was dropped while `allowed_domains` beside it
    was honoured, and the tool's other settings went unread and unnamed,
    until 2026-10-03 (upstream drift audit). Both filters ride on the
    search now, and any other key is named on the ignored-settings header,
    as `plan_from_anthropic` names Anthropic's.
    """
    given = definition.get("filters")
    filters: Mapping[str, Any] = given if isinstance(given, Mapping) else {}
    # Keys are the caller's words on a response header: never reflected raw.
    ignored = [
        f"tools.{WEB_SEARCH}.{_field_name(str(key))}"
        for key in definition
        if key not in RESPONSES_KNOWN
    ]
    ignored += [
        f"tools.{WEB_SEARCH}.filters.{_field_name(str(key))}"
        for key in filters
        if key not in RESPONSES_FILTERS
    ]
    return SearchPlan(
        version=str(definition.get("type") or WEB_SEARCH),
        max_uses=max_tool_calls if isinstance(max_tool_calls, int) and max_tool_calls > 0 else None,
        allowed_domains=_domains(filters.get("allowed_domains")),
        blocked_domains=_domains(filters.get("blocked_domains")),
        user_location=_location(definition.get("user_location")),
        context_size=definition.get("search_context_size")
        if definition.get("search_context_size") in ("low", "medium", "high")
        else None,
        only_tool=not has_functions,
        ignored=ignored,
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


# The image tool's settings that ride on the images request, with the type
# each must have. `model` is read apart: it chooses, it does not ride.
_IMAGE_SETTINGS: dict[str, type] = {
    "size": str,
    "quality": str,
    "background": str,
    "output_format": str,
    "output_compression": int,
    "moderation": str,
}


@dataclass
class ImagePlan:
    """A Responses `image_generation` tool (P8e), in the gateway's words."""

    #: The image model the caller's tool named, if any (P8e-1's first choice).
    model: str | None = None
    #: The tool's settings that ride on each images request.
    settings: dict[str, Any] = field(default_factory=dict)
    #: The request's `max_tool_calls`, shared with any search.
    max_uses: int | None = None
    #: Tool settings dropped and named on the ignored-settings header.
    ignored: list[str] = field(default_factory=list)
    #: The image model this request's calls go to, settled by `image_model`.
    chosen: str | None = None

    @property
    def size(self) -> str | None:
        value = self.settings.get("size")
        return value if isinstance(value, str) and value != "auto" else None


def plan_from_image_tool(definition: Mapping[str, Any], *, max_tool_calls: Any) -> ImagePlan:
    """OpenAI's `image_generation` tool. A setting this gateway cannot honour
    is named, never quietly dropped: `partial_images` (no partial frames
    reach a model's tool loop), the edit inputs, and an unknown key."""
    ignored: list[str] = []
    settings: dict[str, Any] = {}
    for key, value in definition.items():
        if key in ("type", "model") or value is None:
            continue
        if key == "action" and value in ("generate", "auto"):
            continue
        wanted = _IMAGE_SETTINGS.get(key)
        if wanted is not None and isinstance(value, wanted) and not isinstance(value, bool):
            settings[key] = value
            continue
        ignored.append(f"tools.image_generation.{key}")
    model = definition.get("model")
    return ImagePlan(
        model=model if isinstance(model, str) and model else None,
        settings=settings,
        max_uses=max_tool_calls if isinstance(max_tool_calls, int) and max_tool_calls > 0 else None,
        ignored=ignored,
    )


def request_limit(
    search: SearchPlan | None, image: ImagePlan | None, install_default: int | None
) -> int:
    """One budget for every server-run tool a request may call (P8e): the
    caller's own limit when it gave one, else the install's."""
    given = search.max_uses if search is not None else None
    if given is None and image is not None:
        given = image.max_uses
    chosen = given if given is not None else install_default
    return max(1, int(chosen or DEFAULT_MAX_TOOL_CALLS))


# --------------------------------------------------------------------------- #
# Which image model answers the tool (P8e-1)
# --------------------------------------------------------------------------- #


def _image_models(table: Any, allowed: set[str] | None, local_only: bool) -> dict[str, Any]:
    """Every image model this key may use, by the id a request names."""
    found: dict[str, Any] = {}
    for model_id in table.known_models():
        if not permits(allowed, model_id):
            continue
        resolution = table.resolve(model_id).restricted(allowed, local_only=local_only)
        for backend in resolution.backends():
            if backend.makes_images and backend.public_id:
                found.setdefault(backend.public_id, backend)
    return found


def _serves_images(table: Any, model: str, allowed: set[str] | None, local_only: bool) -> bool:
    if not permits(allowed, model):
        return False
    resolution = table.resolve(model).restricted(allowed, local_only=local_only)
    return any(b.makes_images for b in resolution.backends())


def image_model(table: Any, plan: ImagePlan, context: Any, setting: Any) -> tuple[str | None, str]:
    """`(model, "")` when the tool can run for this request, else `(None, why)`.

    P8e-1: the tool's own `model` when this install serves it for the key;
    else the gateway's `imageToolModel` setting; else the one image model
    the key may use, when there is exactly one. The reasons are worded for
    the person who will fix them.
    """
    allowed_tools = getattr(context, "allowed_tools", None) if context is not None else None
    if not permits(allowed_tools, IMAGE_GENERATION):
        return None, "this key's tool scope does not include image_generation"
    if table is None:
        return None, "the gateway is starting up and has no image model yet"
    allowed = getattr(context, "allowed_models", None) if context is not None else None
    local_only = bool(getattr(context, "local_only", False)) if context is not None else False
    if plan.model and _serves_images(table, plan.model, allowed, local_only):
        return plan.model, ""
    if isinstance(setting, str) and setting:
        if _serves_images(table, setting, allowed, local_only):
            return setting, ""
        return None, (
            f"the gateway's image model for tools ({setting}) is not an image model this key "
            "may use; change it under Gateway -> Settings, or name one as the tool's model"
        )
    models = sorted(_image_models(table, allowed, local_only))
    if len(models) == 1:
        return models[0], ""
    if models:
        shown = ", ".join(models[:5]) + (", ..." if len(models) > 5 else "")
        return None, (
            f"this key may use several image models ({shown}); name one as the tool's model, "
            "or set Gateway -> Settings -> Image model for tools"
        )
    if local_only:
        return None, "this key is local-only, and no image model here runs on this install"
    if plan.model:
        return None, (
            f"no image model is served here, including the tool's {plan.model}; add a provider "
            "account that makes images under Backends"
        )
    return (
        None,
        "no image model is served here; add a provider account that makes images under Backends",
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


def searchable(
    resolution: Any, *, native_ok: bool = True
) -> tuple[Any, set[tuple[str | None, str, str | None]]]:
    """The candidates a search can reach, and which of them search themselves.

    A backend whose model lists `webSearchOptions` is sent the request
    natively -- forwarded, not run twice. Any other must call tools, since
    this install's search is offered to the model as one. The rest cannot
    take part and are left out of every tier (empty tiers keep their
    numbers, as everywhere else).

    `native_ok=False` when the loop must run whatever the backend can do
    alone -- an `image_generation` call (P8e) is answered only here -- so
    every candidate must call tools and none is sent the search natively.
    """
    native: set[tuple[str | None, str, str | None]] = set()
    tiers = []
    for tier in resolution.tiers:
        kept = []
        for backend in tier.backends:
            model = getattr(backend, "model", None)
            caps = getattr(model, "capabilities", None)
            settings = set(getattr(caps, "supportedSettings", None) or [])
            if native_ok and "webSearchOptions" in settings:
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
    """What a door is told while the loop runs: a search or an image started,
    or ended."""

    phase: str  # "started" | "completed"
    execution: Execution | ImageExecution


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
        #: Every account's reason, in the order tried: the first is the one an
        #: operator most likely has to fix (free accounts are tried first),
        #: and a later, unreachable one must not hide it (found by the P8
        #: acceptance on a slow CI runner, 2026-09-29).
        reasons: list[str] = []
        first: ToolDriverError | None = None
        for account in self.table.search_accounts(WEB_SEARCH):
            execution.driver, execution.node = account.name, account.node
            try:
                answer: SearchAnswer = await account.client.web_search(request)
            except ToolDriverError as exc:
                last = exc
                first = first or exc
                reasons.append(f"{account.name}: {exc.detail}")
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
            execution.error = last.detail if len(reasons) < 2 else "; ".join(reasons)
            code_from = first or last
            execution.error_code = (
                "too_many_requests"
                if code_from.status == 429
                else "invalid_input"
                if code_from.status == 400
                else "unavailable"
            )
        return execution


@dataclass
class ImageExecution:
    """One `image_generation` call the model made (P8e), and what came of it."""

    call_id: str
    prompt: str
    size: str | None = None
    version: str = IMAGE_GENERATION
    #: `ok`, `error`, or `over_limit` (answered without running).
    outcome: str = "pending"
    #: The image, base64, as OpenAI's `image_generation_call.result` carries it.
    image: str | None = None
    output_format: str | None = None
    revised_prompt: str | None = None
    quality: str | None = None
    background: str | None = None
    #: The image model that made it.
    model: str | None = None
    driver: str | None = None
    node: str | None = None
    elapsed_ms: int = 0
    error: str | None = None
    item_id: str = field(default_factory=lambda: uuid.uuid4().hex[:24])


Permit = Callable[[Any], Awaitable[Any]]


class ImageRunner:
    """Makes one image, routed exactly as `POST /v1/images/generations` is.

    The key's model scope and locality (`permit`), the model's listed image
    settings (`image_refusal`, `pick_image`) and its tiers are the images
    door's own, called in-process: a second HTTP hop would be a second
    request with its own recording, for an image that is part of this one.
    The attempts it makes are taken off this request's attempt rows, which
    are the model's, and summarised on the execution instead.
    """

    def __init__(
        self,
        table: Any,
        plan: ImagePlan,
        *,
        permit: Permit,
        local_only: bool = False,
        request_id: Any = None,
        before_attempt: Callable[[], Awaitable[None]] | None = None,
        attempts: Callable[[], list[Any] | None] | None = None,
    ) -> None:
        self.table = table
        self.plan = plan
        self.permit = permit
        self.local_only = local_only
        self.request_id = request_id
        self.before_attempt = before_attempt
        self.attempts = attempts

    def _ask(self, execution: ImageExecution) -> ImageAsk:
        settings = self.plan.settings
        return ImageAsk(
            door="generations",
            model=self.plan.chosen or "",
            prompt=execution.prompt,
            n=1,
            size=self.plan.size or execution.size,
            quality=settings.get("quality"),
            background=settings.get("background"),
            output_format=settings.get("output_format"),
            output_compression=settings.get("output_compression"),
            moderation=settings.get("moderation"),
        )

    async def run(self, execution: ImageExecution) -> ImageExecution:
        started = time.perf_counter()
        execution.model = self.plan.chosen
        try:
            await self._run(execution)
        finally:
            execution.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return execution

    async def _run(self, execution: ImageExecution) -> None:
        if not execution.prompt.strip():
            execution.outcome, execution.error = "error", "the call had no prompt"
            return
        ask = self._ask(execution)
        resolution = await self.permit(self.table.resolve(ask.model))
        refused = self.table.image_refusal(resolution, ask)
        if refused is not None:
            execution.outcome, execution.error = "error", refused[1]
            return
        client = self.table.pick_image(resolution, ask)
        if client is None and await self.table.refresh_if_stale():
            resolution = await self.permit(self.table.resolve(ask.model))
            client = self.table.pick_image(resolution, ask)
        if client is None:
            execution.outcome = "error"
            execution.error = f"no backend serving {ask.model} is ready to make an image"
            return
        if self.before_attempt is not None and hasattr(client, "authorize_attempt"):
            client.authorize_attempt = self.before_attempt
        rows = self.attempts() if self.attempts is not None else None
        before = len(rows) if rows is not None else 0
        try:
            result = await client.image(
                to_driver(ask, local_only=self.local_only, request_id=self.request_id)
            )
        except DriverError as exc:
            execution.outcome = "error"
            execution.error = (
                exc.problem.detail or exc.problem.title if exc.problem is not None else str(exc)
            )
            return
        except httpx.TimeoutException as exc:
            execution.outcome = "error"
            execution.error = f"the image backend took too long ({type(exc).__name__})"
            return
        except httpx.HTTPError as exc:
            # A transport failure is told to the model, not raised: the
            # request is a conversation that has already started.
            execution.outcome = "error"
            execution.error = f"the image backend could not be reached ({type(exc).__name__})"
            return
        finally:
            execution.driver = getattr(client, "served_by", None) or getattr(client, "name", None)
            execution.node = getattr(client, "served_by_node", None)
            if rows is not None:
                del rows[before:]
        if not result.images:
            execution.outcome, execution.error = "error", "the backend returned no image"
            return
        first = result.images[0]
        execution.outcome = "ok"
        execution.image = first.data
        execution.output_format = format_name(first.mediaType)
        execution.revised_prompt = first.revisedPrompt or execution.prompt
        execution.size = result.size or ask.size
        execution.quality = result.quality or ask.quality
        execution.background = result.background or ask.background
        execution.model = getattr(client, "served_model", None) or ask.model


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


def image_tool(name: str, *, with_size: bool) -> Tool:
    """The function the model is offered for `image_generation` (P8e).

    `size` is offered only when the caller's tool left it unset: a size the
    caller configured is the caller's, not the model's, to change.
    """
    properties: dict[str, Any] = {
        "prompt": {
            "type": "string",
            "description": "A full description of the image: subject, style, composition, colours.",
        }
    }
    if with_size:
        properties["size"] = {
            "type": "string",
            "description": "WIDTHxHEIGHT, such as 1024x1024, 1536x1024 or 1024x1536. "
            "Leave it out for the default.",
        }
    return Tool(
        type="function",
        function=FunctionDefinition(
            name=name,
            description=(
                "Make an image from a description. The image is shown to the person; you "
                "are told when it is ready but do not see it. Use it when you are asked to "
                "draw, design, illustrate or make a picture."
            ),
            parameters={"type": "object", "properties": properties, "required": ["prompt"]},
        ),
    )


def tool_name_for(caller_tools: Any, base: str = WEB_SEARCH, *, taken: Iterable[str] = ()) -> str:
    """`base`, unless the caller already declared a function by that name."""
    used = {getattr(getattr(t, "function", None), "name", None) for t in caller_tools or []}
    used |= set(taken)
    for name in (base, f"eugene_{base}", f"eugene_plexus_{base}"):
        if name not in used:
            return name
    return f"{base}_{uuid.uuid4().hex[:6]}"


def image_args(arguments: str | None) -> tuple[str, str | None]:
    """The model's prompt and size, however a small model sends them."""
    if not arguments:
        return "", None
    try:
        parsed = json.loads(arguments)
    except ValueError:
        return arguments.strip(), None
    if isinstance(parsed, str):
        return parsed, None
    if isinstance(parsed, Mapping):
        size = parsed.get("size")
        prompt = parsed.get("prompt")
        if not isinstance(prompt, str):
            prompt = next((v for k, v in parsed.items() if k != "size" and isinstance(v, str)), "")
        return prompt, size if isinstance(size, str) and size else None
    return "", None


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
        # The error is a sentence of its own, usually with its full stop.
        reason = (execution.error or "no reason was given").rstrip().rstrip(".")
        return (
            f'The web search for "{execution.query}" could not be run: {reason}. '
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


def image_text(execution: ImageExecution, *, limit: int) -> str:
    """The tool message the model reads for one image (P8e-2): words only."""
    if execution.outcome == "over_limit":
        return (
            f"No more images or searches are available for this request (its limit is {limit}). "
            "Answer with what you have."
        )
    if execution.outcome != "ok":
        return (
            f'The image for "{execution.prompt}" could not be made: {execution.error}. '
            "Tell the person it failed and why."
        )
    size = f" at {execution.size}" if execution.size else ""
    return (
        f'An image was made{size} for the prompt "{execution.revised_prompt or execution.prompt}" '
        "and is shown to the person. You cannot see it. Tell them briefly that it is ready; "
        "do not describe details you cannot see."
    )


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
    """The routed client, with the server-run tool loop around it.

    Everything a door reads off a client -- `served_by`, `tier`,
    `served_model` -- is the wrapped client's, except `attempts`, which is
    the FIRST call's: later calls go to one pinned backend and are turns
    of one answer, not backends tried.

    It runs a search (`plan`), an image (`image`, P8e), or both; either may
    be None, never both. One budget, `limit`, covers every call of either.
    """

    def __init__(
        self,
        inner: Any,
        *,
        plan: SearchPlan | None,
        runner: SearchRunner | None,
        limit: int,
        native: set[tuple[str | None, str, str | None]],
        image: ImagePlan | None = None,
        image_runner: ImageRunner | None = None,
    ) -> None:
        self._inner = inner
        self.plan = plan
        self.runner = runner
        self.image = image
        self.image_runner = image_runner
        self.limit = limit
        #: `(node, driver, public model)` of the candidates that search
        #: themselves and are sent the request natively.
        self._native = native
        self.executions: list[Execution] = []
        #: Every image call the model made (P8e), in order.
        self.images: list[ImageExecution] = []
        #: What happened, in order: `("text", str)`, `("reasoning", str)`,
        #: `("search", Execution)`, `("image", ImageExecution)`. The Responses
        #: and Anthropic doors render their items and blocks from it.
        self.transcript: list[tuple[str, Any]] = []
        self.turns = 0
        self.first_attempts: int | None = None
        self.searched_natively = False
        self._tool_name = WEB_SEARCH
        self._image_name = IMAGE_GENERATION
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
        # Every call the limit allows, one turn to read the last results,
        # and one more for a model that asks past the limit to be told no.
        return self.limit + 2

    @property
    def _names(self) -> set[str]:
        names = set()
        if self.plan is not None:
            names.add(self._tool_name)
        if self.image is not None:
            names.add(self._image_name)
        return names

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
        assert self.plan is not None, "only a search is ever sent natively"
        settings = set(base.callerSettings or []) | {"webSearchOptions"}
        return base.model_copy(
            update={
                "webSearchOptions": self.plan.native_options(),
                "callerSettings": sorted(settings),
            }
        )

    def _name_tools(self, request: GenerateRequest) -> None:
        self._tool_name = tool_name_for(request.tools)
        self._image_name = tool_name_for(request.tools, IMAGE_GENERATION, taken={self._tool_name})

    def _offered(self) -> list[Tool]:
        offered = []
        if self.plan is not None:
            offered.append(search_tool(self._tool_name))
        if self.image is not None:
            offered.append(image_tool(self._image_name, with_size=self.image.size is None))
        return offered

    def _turn_request(
        self, base: GenerateRequest, history: list[Message], turn: int
    ) -> GenerateRequest:
        last = turn >= self.max_turns - 1
        caller_tools = list(base.tools or [])
        tools = caller_tools if last else [*caller_tools, *self._offered()]
        settings = set(base.callerSettings or []) - {"webSearchOptions"}
        if tools:
            settings.add("tools")
        else:
            settings.discard("tools")
        choice = base.toolChoice
        if last and not caller_tools:
            choice = None
        elif (
            turn == 0
            and self.plan is not None
            and self.plan.only_tool
            and self.image is None
            and (choice is None or choice == ToolChoice.auto)
        ):
            # The caller asked for a search; the first call makes one. Not
            # when an image tool rides too: "hi" is not a request for either.
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
        names = self._names
        ours = [c for c in calls or [] if c.function.name in names]
        others = [c for c in calls or [] if c.function.name not in names]
        return ours, others

    def _spent(self) -> int:
        searches = [e for e in self.executions if e.outcome != "over_limit"]
        images = [e for e in self.images if e.outcome != "over_limit"]
        return len(searches) + len(images)

    def _pending(self, call: ToolCall) -> Execution | ImageExecution:
        if call.function.name == self._image_name and self.image is not None:
            prompt, size = image_args(call.function.arguments)
            return ImageExecution(call_id=call.id, prompt=prompt, size=size)
        assert self.plan is not None
        return Execution(
            call_id=call.id, query=query_of(call.function.arguments), version=self.plan.version
        )

    async def _run(self, execution: Execution | ImageExecution) -> Execution | ImageExecution:
        over = self._spent() >= self.limit
        if isinstance(execution, ImageExecution):
            if over:
                execution.outcome = "over_limit"
            elif self.image_runner is not None:
                await self.image_runner.run(execution)
            self.images.append(execution)
            return execution
        if over:
            execution.outcome, execution.error_code = "over_limit", "max_uses_exceeded"
        elif self.runner is not None:
            await self.runner.run(execution)
        self.executions.append(execution)
        return execution

    def _told(self, execution: Execution | ImageExecution) -> str:
        if isinstance(execution, ImageExecution):
            return image_text(execution, limit=self.limit)
        return result_text(execution, limit=self.limit)

    @staticmethod
    def _kind(execution: Execution | ImageExecution) -> str:
        return "image" if isinstance(execution, ImageExecution) else "search"

    def _history_after(
        self,
        history: list[Message],
        text: str,
        reasoning: str | None,
        calls: list[ToolCall],
        done: list[Execution | ImageExecution],
    ) -> list[Message]:
        history = list(history)
        history.append(
            Message(
                role=Role.assistant,
                content=text or None,
                toolCalls=[c.model_dump(mode="json") for c in calls],
                reasoning=reasoning or None,
            )
        )
        for call, execution in zip(calls, done, strict=True):
            history.append(
                Message(role=Role.tool, toolCallId=call.id, content=self._told(execution))
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
        self._name_tools(request)
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
            ours, others = self._split(response.toolCalls)
            if not ours or self.searched_natively or turn >= self.max_turns - 1:
                return self._final(response, others, usage)
            done: list[Execution | ImageExecution] = []
            for call in ours:
                execution = await self._run(self._pending(call))
                self.transcript.append((self._kind(execution), execution))
                done.append(execution)
            history = self._history_after(
                history, response.content or "", response.reasoning, ours, done
            )
        raise AssertionError("unreachable: the last turn always returns")

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        self._base = request
        self._name_tools(request)
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
                    # Held until the turn ends: a call to one of our tools
                    # must never reach the caller, and whether a turn's other
                    # calls do depends on whether it also called one of ours.
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
            ours, others = self._split(result.toolCalls)
            if not ours or self.searched_natively or turn >= self.max_turns - 1:
                for index in sorted(fragments):
                    if names.get(index) in self._names:
                        continue
                    for fragment in fragments[index]:
                        yield StreamEvent(tool_calls=[fragment])
                final = self._final(result, others, usage)
                cited = citations(final.content, self.executions) if self.executions else []
                if cited:
                    yield StreamEvent(annotations=cited)
                yield StreamEvent(done=True, result=final)
                return
            done: list[Execution | ImageExecution] = []
            for call in ours:
                pending = self._pending(call)
                update = SearchUpdate("started", pending)
                yield (
                    StreamEvent(image=update)
                    if isinstance(pending, ImageExecution)
                    else StreamEvent(search=update)
                )
                execution = await self._run(pending)
                self.transcript.append((self._kind(execution), execution))
                done.append(execution)
                finished = SearchUpdate("completed", execution)
                yield (
                    StreamEvent(image=finished)
                    if isinstance(execution, ImageExecution)
                    else StreamEvent(search=finished)
                )
            history = self._history_after(
                history, "".join(text), "".join(thought) or None, ours, done
            )
