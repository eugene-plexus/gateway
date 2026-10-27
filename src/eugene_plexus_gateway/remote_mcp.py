"""Remote MCP on the Responses door (row 59b): the gateway's half.

A Responses request may name an MCP server, as on OpenAI:
`{"type": "mcp", "server_label", "server_url", "headers" | "authorization",
"allowed_tools", "require_approval"}`. The gateway runs the loop; a
tool-driver lists the server's tools and calls them, because egress, its
address rules and the caller's credential live with the other server-run
tools (`server-run-tools.md` §1). This module holds:

* **The plan** (`McpPlan`): each server, its approval rule, the names the
  model calls its tools by, and what the conversation already holds --
  a tool list from an `mcp_list_tools` item, approvals asked and answered.
* **The history** (`McpInput`): `mcp_*` items in `input`, turned into the
  calls and results the model reads, and the approvals checked: an
  `mcp_approval_response` answers an `mcp_approval_request` the
  conversation carries, or the request is refused.
* **The runner** (`McpRunner`): one tool-driver call to list or to call,
  with the reach the owner's `remoteMcp` setting gives.
* **The items** the caller gets back.

The caller's `headers` and `authorization` live on `McpServer.headers` for
the request and nowhere else: never in an item, an echo, a log line or a
metrics row (`echo_tools` takes them off what the response echoes).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model_patterns import permits
from .tool_client import ToolDriverError

log = logging.getLogger(__name__)

MCP = "mcp"
#: The design's limit: MCP calls one response may make. `max_tool_calls` lowers it.
LIMIT_CALLS = 20
#: The owner's setting (call R59-MCP), and its default.
REACHES = ("off", "internet", "network")
DEFAULT_REACH = "internet"
#: What an approved call's tool message says until the call has run.
_PENDING = "[the approved call has not run yet]"

KNOWN = frozenset(
    {
        "type",
        "server_label",
        "server_url",
        "headers",
        "authorization",
        "allowed_tools",
        "require_approval",
        "server_description",
    }
)
_LABEL = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_FUNCTION = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class InvalidMcp(ValueError):
    """A request's `mcp` tool or `mcp_*` item this gateway will not carry,
    with the field named in `param`."""

    def __init__(self, message: str, *, param: str) -> None:
        super().__init__(message)
        self.message = message
        self.param = param


# --------------------------------------------------------------------------- #
# The tool
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ApprovalRule:
    """`require_approval`, read: which tools wait for the person's answer."""

    default_needs: bool = True
    always: frozenset[str] = frozenset()
    never: frozenset[str] = frozenset()
    always_read_only: bool = False
    never_read_only: bool = False

    def needs(self, name: str, *, read_only: bool) -> bool:
        if name in self.never or (self.never_read_only and read_only):
            return False
        if name in self.always or (self.always_read_only and read_only):
            return True
        return self.default_needs


@dataclass
class ListedTool:
    name: str
    description: str | None
    input_schema: dict[str, Any]
    annotations: dict[str, Any] | None = None

    @property
    def read_only(self) -> bool:
        return bool((self.annotations or {}).get("readOnlyHint") is True)

    def as_item(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "annotations": self.annotations,
        }


@dataclass
class McpServer:
    label: str
    url: str
    #: The caller's headers, `authorization` as `Authorization: Bearer`.
    #: For this request only: never in an item, an echo, a log or a metric.
    headers: dict[str, str] = field(repr=False)
    allowed: list[str] | None
    allowed_read_only: bool
    approval: ApprovalRule
    description: str | None
    where: str


def _names(value: Any, where: str) -> frozenset[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise InvalidMcp(f"{where}: must be a list of tool names.", param="tools")
    return frozenset(value)


def _filter(value: Any, where: str) -> tuple[frozenset[str], bool]:
    """`{"tool_names": [...], "read_only": bool}`, OpenAI's tool filter."""
    if not isinstance(value, Mapping):
        raise InvalidMcp(f"{where}: must be an object with tool_names.", param="tools")
    for key in value:
        if key not in ("tool_names", "read_only"):
            raise InvalidMcp(f"{where}.{key}: unsupported setting.", param="tools")
    names = _names(value.get("tool_names", []), f"{where}.tool_names")
    read_only = value.get("read_only", False)
    if not isinstance(read_only, bool):
        raise InvalidMcp(f"{where}.read_only: must be true or false.", param="tools")
    return names, read_only


def approval_rule(value: Any, where: str) -> ApprovalRule:
    """`always` (OpenAI's default), `never`, or `{"always": {...}, "never": {...}}`.

    In the object form a tool neither filter names waits for approval,
    unless only `always` was given: a caller that lists the tools needing
    approval has said the others do not."""
    if value is None or value == "always":
        return ApprovalRule()
    if value == "never":
        return ApprovalRule(default_needs=False)
    if not isinstance(value, Mapping):
        raise InvalidMcp(f"{where}: must be always, never or an object.", param="tools")
    for key in value:
        if key not in ("always", "never"):
            raise InvalidMcp(f"{where}.{key}: unsupported setting.", param="tools")
    always, always_ro = (
        _filter(value["always"], f"{where}.always")
        if "always" in value
        else (
            frozenset(),
            False,
        )
    )
    never, never_ro = (
        _filter(value["never"], f"{where}.never")
        if "never" in value
        else (
            frozenset(),
            False,
        )
    )
    return ApprovalRule(
        default_needs=not ("always" in value and "never" not in value),
        always=always,
        never=never,
        always_read_only=always_ro,
        never_read_only=never_ro,
    )


def parse_server(definition: Mapping[str, Any], where: str, ignored: list[str]) -> McpServer:
    if definition.get("connector_id") is not None:
        raise InvalidMcp(
            f"{where}.connector_id: names one of OpenAI's connectors, which this install does "
            "not have; give the MCP server's server_url instead.",
            param="tools",
        )
    label = definition.get("server_label")
    if not isinstance(label, str) or not _LABEL.match(label):
        raise InvalidMcp(
            f"{where}.server_label: required, 1 to 64 letters, digits, - or _.", param="tools"
        )
    url = definition.get("server_url")
    if not isinstance(url, str) or not url.strip():
        raise InvalidMcp(f"{where}.server_url: required.", param="tools")
    headers: dict[str, str] = {}
    given = definition.get("headers")
    if given is not None:
        if not isinstance(given, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in given.items()
        ):
            raise InvalidMcp(f"{where}.headers: must be an object of strings.", param="tools")
        headers = dict(given)
    token = definition.get("authorization")
    if token is not None:
        if not isinstance(token, str) or not token.strip():
            raise InvalidMcp(f"{where}.authorization: must be a token.", param="tools")
        if any(k.lower() == "authorization" for k in headers):
            raise InvalidMcp(
                f"{where}.authorization: the headers carry an Authorization header too; "
                "give the token once.",
                param="tools",
            )
        headers["Authorization"] = f"Bearer {token.strip()}"
    allowed: list[str] | None = None
    allowed_read_only = False
    raw_allowed = definition.get("allowed_tools")
    if isinstance(raw_allowed, list):
        allowed = sorted(_names(raw_allowed, f"{where}.allowed_tools"))
    elif raw_allowed is not None:
        names, allowed_read_only = _filter(raw_allowed, f"{where}.allowed_tools")
        allowed = sorted(names) if "tool_names" in raw_allowed else None
    description = definition.get("server_description")
    if description is not None and not isinstance(description, str):
        raise InvalidMcp(f"{where}.server_description: must be a string.", param="tools")
    for key in definition:
        if key not in KNOWN and key != "connector_id":
            # A key's name only, never its value: it may be a credential.
            ignored.append(f"tools.mcp.{key}" if re.match(r"^\w{1,64}$", key) else "tools.mcp")
    return McpServer(
        label=label,
        url=url.strip(),
        headers=headers,
        allowed=allowed,
        allowed_read_only=allowed_read_only,
        approval=approval_rule(definition.get("require_approval"), f"{where}.require_approval"),
        description=description or None,
        where=where,
    )


def echo_tools(tools: Any) -> Any:
    """The request's `tools` as the response echoes them: an `mcp` tool
    without `headers` or `authorization`, which never leave the request."""
    if not isinstance(tools, list):
        return tools
    out = []
    for tool in tools:
        if isinstance(tool, Mapping) and tool.get("type") == MCP:
            tool = {k: v for k, v in tool.items() if k not in ("headers", "authorization")}
        out.append(tool)
    return out


def function_name(label: str, name: str, *, multi: bool, taken: Iterable[str]) -> str:
    """The function name the model calls an MCP tool by.

    The tool's own name when one server is offered and it is free; else
    `<label>__<name>`. A name a function may not carry (or one already
    taken) gets a short hash of the server and the tool, so two tools never
    share one and the same tool gets the same name on every request.
    """
    base = f"{label}__{name}" if multi else name
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", base)[:64]
    if clean != base or clean in set(taken):
        digest = hashlib.sha256(f"{label}\0{name}".encode()).hexdigest()[:8]
        clean = f"{clean[:55]}_{digest}"
    return clean


# --------------------------------------------------------------------------- #
# The plan, and what the conversation already holds
# --------------------------------------------------------------------------- #


@dataclass
class Asked:
    """An `mcp_approval_request` the conversation carries."""

    id: str
    label: str
    name: str
    arguments: str


@dataclass
class Answered:
    asked: Asked
    approve: bool
    reason: str | None


@dataclass
class McpListing:
    """One server's tools, listed for this request (or a failure)."""

    label: str
    tools: list[ListedTool]
    item_id: str = field(default_factory=lambda: f"mcpl_{uuid.uuid4().hex[:24]}")


@dataclass
class McpPlan:
    servers: dict[str, McpServer]
    #: Function names the caller's own tools use, and the server-run tools'.
    taken: set[str]
    #: `max_tool_calls`, when the caller gave one.
    max_tool_calls: int | None = None
    #: Each server's tools, by label: listed here, or from an `mcp_list_tools` in `input`.
    tools: dict[str, list[ListedTool]] = field(default_factory=dict)
    #: The listings this request made, for its `mcp_list_tools` output items.
    listings: list[McpListing] = field(default_factory=list)
    #: Approved calls the conversation answered and nothing ran yet, in order.
    pending: list[Answered] = field(default_factory=list)
    #: The reach the owner's setting gives, read when the request is prepared.
    reach: str = DEFAULT_REACH

    @property
    def limit(self) -> int:
        given = self.max_tool_calls
        return min(LIMIT_CALLS, given) if isinstance(given, int) and given > 0 else LIMIT_CALLS

    def name_for(self, label: str, name: str) -> str:
        return function_name(label, name, multi=len(self.servers) > 1, taken=self.taken)

    def functions(self) -> dict[str, tuple[McpServer, ListedTool]]:
        """Every tool offered to the model, by the name it calls it."""
        out: dict[str, tuple[McpServer, ListedTool]] = {}
        for label, server in self.servers.items():
            for tool in self.tools.get(label, []):
                out[self.name_for(label, tool.name)] = (server, tool)
        return out

    def keep(self, server: McpServer, tools: list[ListedTool]) -> list[ListedTool]:
        """`allowed_tools`, applied to a list from the server or from `input`."""
        kept = []
        for tool in tools:
            if server.allowed is not None and tool.name not in server.allowed:
                continue
            if server.allowed_read_only and not tool.read_only:
                continue
            kept.append(tool)
        return kept


def plan_from_tools(definitions: Any, ignored: list[str], *, max_tool_calls: Any) -> McpPlan | None:
    """The request's `mcp` tools, or None when it has none."""
    if not isinstance(definitions, list):
        return None
    servers: dict[str, McpServer] = {}
    taken: set[str] = {"web_search", "image_generation"}
    for index, definition in enumerate(definitions):
        if not isinstance(definition, Mapping):
            continue
        kind = definition.get("type")
        if kind == MCP:
            server = parse_server(definition, f"tools[{index}]", ignored)
            if server.label in servers:
                raise InvalidMcp(
                    f"tools[{index}].server_label: {server.label!r} names two MCP servers; "
                    "each needs a label of its own.",
                    param="tools",
                )
            servers[server.label] = server
            continue
        name = definition.get("name")
        if isinstance(name, str):
            taken.add(name)
        if isinstance(kind, str):
            taken.add(kind)
        for member in definition.get("tools") or [] if kind == "namespace" else []:
            if isinstance(member, Mapping) and isinstance(member.get("name"), str):
                taken.add(member["name"])
    if not servers:
        return None
    return McpPlan(
        servers=servers,
        taken=taken,
        max_tool_calls=max_tool_calls if isinstance(max_tool_calls, int) else None,
    )


def _listed(tools: Any) -> list[ListedTool]:
    found: list[ListedTool] = []
    for tool in tools if isinstance(tools, list) else []:
        if not isinstance(tool, Mapping) or not isinstance(tool.get("name"), str):
            continue
        schema = tool.get("input_schema", tool.get("inputSchema"))
        annotations = tool.get("annotations")
        description = tool.get("description")
        found.append(
            ListedTool(
                name=tool["name"],
                description=description if isinstance(description, str) else None,
                input_schema=dict(schema) if isinstance(schema, Mapping) else {},
                annotations=dict(annotations) if isinstance(annotations, Mapping) else None,
            )
        )
    return found


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


class McpInput:
    """The `mcp_*` items of a request's `input`, checked first, then read in order.

    `check` runs over the whole input before any message is built, because
    an approval may only answer a request the conversation carries, and
    whether a call already ran is said by an item that can come later.
    """

    def __init__(self, plan: McpPlan | None, items: list[Any]) -> None:
        self.plan = plan
        self.asked: dict[str, Asked] = {}
        self.answers: dict[str, Answered] = {}
        #: Approval ids an `mcp_call` in `input` already ran under.
        self.ran: dict[str, Mapping[str, Any]] = {}
        self._check(items)

    def _check(self, items: list[Any]) -> None:
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                continue
            kind = item.get("type")
            where = f"input[{index}]"
            if kind == "mcp_approval_request":
                ident, label, name = item.get("id"), item.get("server_label"), item.get("name")
                if not all(isinstance(v, str) and v for v in (ident, label, name)):
                    raise InvalidMcp(
                        f"{where}: an mcp_approval_request needs id, server_label and name.",
                        param="input",
                    )
                self.asked[str(ident)] = Asked(
                    id=str(ident),
                    label=str(label),
                    name=str(name),
                    arguments=_text(item.get("arguments")) or "{}",
                )
            elif kind == "mcp_call":
                approved = item.get("approval_request_id")
                if isinstance(approved, str) and approved:
                    self.ran[approved] = item
        for index, item in enumerate(items):
            if not isinstance(item, Mapping) or item.get("type") != "mcp_approval_response":
                continue
            where = f"input[{index}]"
            ident = item.get("approval_request_id")
            if not isinstance(ident, str) or not ident:
                raise InvalidMcp(f"{where}.approval_request_id: required.", param="input")
            asked = self.asked.get(ident)
            if asked is None:
                raise InvalidMcp(
                    f"{where}.approval_request_id: no mcp_approval_request in this "
                    f"conversation has the id {ident!r}; an approval answers a request this "
                    "gateway made, sent back in input.",
                    param="input",
                )
            if ident in self.answers:
                raise InvalidMcp(
                    f"{where}.approval_request_id: {ident!r} is answered twice.", param="input"
                )
            approve = item.get("approve")
            if not isinstance(approve, bool):
                raise InvalidMcp(f"{where}.approve: must be true or false.", param="input")
            reason = item.get("reason")
            answered = Answered(
                asked=asked, approve=approve, reason=reason if isinstance(reason, str) else None
            )
            self.answers[ident] = answered
            if approve and ident not in self.ran:
                if self.plan is None or asked.label not in self.plan.servers:
                    raise InvalidMcp(
                        f"{where}: approves a call to the MCP server {asked.label!r}, which "
                        "this request's tools do not name; send its mcp tool with it.",
                        param="input",
                    )
                self.plan.pending.append(answered)

    # -- in order, as the conversation is built ------------------------------

    def _name(self, label: str, name: str) -> str | None:
        if self.plan is None or label not in self.plan.servers:
            return None
        return self.plan.name_for(label, name)

    def list_tools(self, item: Mapping[str, Any]) -> None:
        """An earlier listing: used instead of listing the server again."""
        label = item.get("server_label")
        if self.plan is None or not isinstance(label, str) or label not in self.plan.servers:
            return
        if item.get("error"):
            return
        tools = _listed(item.get("tools"))
        if tools:
            self.plan.tools[label] = self.plan.keep(self.plan.servers[label], tools)

    def turns(self, item: Mapping[str, Any]) -> list[tuple[str, Any]]:
        """What one `mcp_*` item adds to the history, in the conversation's words:
        `("call", (id, name, arguments))`, `("result", (id, text))` or
        `("note", text)`."""
        kind = item.get("type")
        if kind == "mcp_approval_request":
            asked = self.asked[str(item.get("id"))]
            return self._asked(asked)
        if kind == "mcp_call":
            approved = item.get("approval_request_id")
            if isinstance(approved, str) and approved in self.asked:
                return []  # The request's place in the history carries it.
            label, name = str(item.get("server_label") or ""), str(item.get("name") or "")
            ident = str(item.get("id") or f"mcp_{uuid.uuid4().hex[:24]}")
            return self._ran(ident, label, name, _text(item.get("arguments")) or "{}", item)
        return []  # An approval response: checked above, carried by its request.

    def _asked(self, asked: Asked) -> list[tuple[str, Any]]:
        ran = self.ran.get(asked.id)
        if ran is not None:
            return self._ran(asked.id, asked.label, asked.name, asked.arguments, ran)
        answer = self.answers.get(asked.id)
        if answer is None:
            text = "Not run: the person has not answered this approval request."
        elif not answer.approve:
            text = "Not run: the person declined this call." + (
                f" Their reason: {answer.reason}" if answer.reason else ""
            )
        else:
            text = _PENDING
        return self._pair(asked.id, asked.label, asked.name, asked.arguments, text)

    def _ran(
        self, ident: str, label: str, name: str, arguments: str, item: Mapping[str, Any]
    ) -> list[tuple[str, Any]]:
        error = item.get("error")
        text = (
            f"The call failed: {_text(error)}" if error else _text(item.get("output")) or "(empty)"
        )
        return self._pair(ident, label, name, arguments, text)

    def _pair(
        self, ident: str, label: str, name: str, arguments: str, text: str
    ) -> list[tuple[str, Any]]:
        function = self._name(label, name)
        if function is None:
            # A server this request does not offer: a note, not a call to a
            # function the model is not given.
            return [
                (
                    "note",
                    f'[Called the MCP tool "{name}" on the server "{label}" with {arguments}. '
                    f"Result: {text}]",
                )
            ]
        return [("call", (ident, function, arguments)), ("result", (ident, text))]


# --------------------------------------------------------------------------- #
# Whether it may run, and the runner
# --------------------------------------------------------------------------- #


def reach_of(setting: Any) -> str | None:
    """The owner's `remoteMcp`: one of `REACHES`, the default when unset, and
    None for a value Eugene does not know -- which is refused, never taken for
    on (second-model review: a hand-written `off` read as YAML's False and
    meant the internet)."""
    if setting is None:
        return DEFAULT_REACH
    return setting if isinstance(setting, str) and setting in REACHES else None


def why_not(table: Any, context: Any, setting: Any) -> str | None:
    """None when remote MCP may run for this request, else why not, in words."""
    reach = reach_of(setting)
    if reach is None:
        return (
            f"the gateway's Remote MCP setting holds {setting!r}, a value Eugene does not know; "
            "choose Off, Internet only or Internet and this network under Gateway, Settings"
        )
    if reach == "off":
        return (
            "remote MCP is turned off on this install (Gateway, Settings, Remote MCP from "
            "API clients)"
        )
    if context is not None and getattr(context, "local_only", False):
        return (
            "this key is local-only, and a remote MCP call sends what the model wrote to "
            "another server"
        )
    allowed = getattr(context, "allowed_tools", None) if context is not None else None
    if not permits(allowed, MCP):
        return "this key's tool scope does not include mcp"
    if table is None:
        return "the gateway is starting up"
    if not table.mcp_accounts():
        return (
            "remote MCP runs in a search account's process, and this install has none; add "
            "one under Backends, Add a search account"
        )
    return None


@dataclass
class McpRefusal(Exception):
    """A listing that failed, as the door answers it."""

    status: int
    message: str
    error_type: str
    param: str | None = "tools"


@dataclass
class McpExecution:
    """One MCP call: the model's, or one the person approved."""

    call_id: str
    label: str
    name: str
    arguments: str
    approval_request_id: str | None = None
    #: `ok`, `error`, `over_limit` or `pending`.
    outcome: str = "pending"
    output: str | None = None
    error: str | None = None
    driver: str | None = None
    node: str | None = None
    elapsed_ms: int = 0
    item_id: str = field(default_factory=lambda: f"mcp_{uuid.uuid4().hex[:24]}")


@dataclass
class McpApproval:
    """A call the model made that waits for the person's approval."""

    call_id: str
    label: str
    name: str
    arguments: str
    item_id: str = field(default_factory=lambda: f"mcpr_{uuid.uuid4().hex[:24]}")


@dataclass(frozen=True)
class McpUpdate:
    """What the Responses door is told while the loop runs: a listing done,
    a call started or ended, an approval asked for."""

    phase: str  # "listed" | "started" | "completed" | "approval"
    value: McpListing | McpExecution | McpApproval


class ApprovalLedger:
    """The approvals whose calls ran, so each runs once (second-model review).

    The conversation a request carries is not the only record: answering the
    same approval again against the same `previous_response_id` -- a fork, or
    a client retrying -- carries no `mcp_call` for it, and ran the call again.
    So an approval is taken here, atomically (a primary key in SQLite), just
    before its call runs, and one already taken is refused. Kept beside the
    gateway's config, so a restart remembers; an entry is dropped after the 31
    days a stored response can be continued for. Holds approval ids only.
    """

    KEEP_SECONDS = 31 * 24 * 3600

    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("CREATE TABLE IF NOT EXISTS taken (id TEXT PRIMARY KEY, at REAL NOT NULL)")
        return db

    def _used(self, ident: str) -> bool:
        db = self._connect()
        try:
            return db.execute("SELECT 1 FROM taken WHERE id = ?", (ident,)).fetchone() is not None
        finally:
            db.close()

    def _take(self, ident: str) -> bool:
        now = time.time()  # a date kept across restarts, not a duration
        db = self._connect()
        try:
            with db:
                db.execute("DELETE FROM taken WHERE at < ?", (now - self.KEEP_SECONDS,))
                taken = db.execute(
                    "INSERT OR IGNORE INTO taken (id, at) VALUES (?, ?)", (ident, now)
                )
                return taken.rowcount == 1
        finally:
            db.close()

    async def used(self, ident: str) -> bool:
        return await asyncio.to_thread(self._used, ident)

    async def take(self, ident: str) -> bool:
        """True for the one caller that takes `ident`; False once it is taken."""
        return await asyncio.to_thread(self._take, ident)


def ledger_for(state: Any) -> ApprovalLedger:
    """The gateway's ledger, made once, beside its config file."""
    ledger: ApprovalLedger | None = getattr(state, "mcp_approvals", None)
    if ledger is None:
        folder = Path(state.settings.config_file).resolve().parent
        ledger = ApprovalLedger(folder / "mcp-approvals.sqlite3")
        state.mcp_approvals = ledger
    return ledger


class McpRunner:
    """Lists and calls through this install's tool-drivers, the first that answers."""

    def __init__(self, table: Any, plan: McpPlan, *, ledger: ApprovalLedger | None = None) -> None:
        self.table = table
        self.plan = plan
        self.ledger = ledger

    async def take(self, ident: str) -> bool:
        """Takes an approval just before its call runs; False when it ran before."""
        return True if self.ledger is None else await self.ledger.take(ident)

    def allows(self, label: str, name: str) -> bool:
        """Whether this request still offers the tool: its `allowed_tools` now,
        not the request's that asked for the approval (second-model review)."""
        return any(tool.name == name for tool in self.plan.tools.get(label, []))

    async def list_tools(self, server: McpServer) -> list[ListedTool]:
        last: ToolDriverError | None = None
        for account in self.table.mcp_accounts():
            try:
                tools = await account.client.mcp_list_tools(
                    server.url,
                    headers=server.headers,
                    reach=self.plan.reach,
                    allowed=server.allowed,
                )
            except ToolDriverError as exc:
                last = exc
                if exc.status == 0 and exc.code == "unreachable":
                    continue  # This process never reached the server: try another.
                break
            return self.plan.keep(server, _listed(tools))
        raise _list_refusal(server, last)

    async def call(self, execution: McpExecution) -> McpExecution:
        started = time.perf_counter()
        server = self.plan.servers.get(execution.label)
        try:
            if server is None:
                execution.outcome = "error"
                execution.error = f"the MCP server {execution.label!r} is not in this request"
                return execution
            try:
                arguments = json.loads(execution.arguments or "{}")
            except ValueError:
                arguments = None
            if not isinstance(arguments, dict):
                execution.outcome = "error"
                execution.error = "the arguments were not a JSON object"
                return execution
            last: ToolDriverError | None = None
            for account in self.table.mcp_accounts():
                execution.driver, execution.node = account.name, account.node
                try:
                    output, error = await account.client.mcp_call(
                        server.url,
                        headers=server.headers,
                        reach=self.plan.reach,
                        name=execution.name,
                        arguments=arguments,
                    )
                except ToolDriverError as exc:
                    last = exc
                    # Only a process that never reached the server is passed
                    # over: past that, the call may have run.
                    if exc.status == 0 and exc.code == "unreachable":
                        continue
                    break
                execution.output, execution.error = output, error
                execution.outcome = "error" if error else "ok"
                return execution
            execution.outcome = "error"
            execution.error = last.detail if last is not None else "no process here runs remote MCP"
            return execution
        finally:
            execution.elapsed_ms = int((time.perf_counter() - started) * 1000)


def _list_refusal(server: McpServer, error: ToolDriverError | None) -> McpRefusal:
    """A listing's failure, as OpenAI answers one: the request's fault is a
    400, the server's a 424, this install's a 503."""
    if error is None:
        return McpRefusal(503, "no process here runs remote MCP", "service_unavailable")
    where = server.where
    if error.status in (400, 403, 413):
        return McpRefusal(
            400, f"{where}.server_url: {error.detail}", "invalid_request_error", "tools"
        )
    if error.status == 0 or error.status == 503:
        return McpRefusal(
            503,
            f"The process that runs remote MCP could not be reached: {error.detail}",
            "service_unavailable",
        )
    return McpRefusal(
        424,
        f"Error retrieving tool list from MCP server {server.label!r}: {error.detail}",
        "external_connector_error",
    )


async def list_all(runner: McpRunner) -> None:
    """Each server's tools, unless `input` already held them; raises `McpRefusal`."""
    plan = runner.plan
    for label, server in plan.servers.items():
        if label in plan.tools:
            continue
        tools = await runner.list_tools(server)
        plan.tools[label] = tools
        plan.listings.append(McpListing(label=label, tools=tools))


# --------------------------------------------------------------------------- #
# What the model is offered and told
# --------------------------------------------------------------------------- #


def offered(plan: McpPlan) -> list[dict[str, Any]]:
    """The MCP tools as chat functions: `{"name", "description", "parameters"}`."""
    out = []
    for function, (server, tool) in plan.functions().items():
        schema = tool.input_schema
        if schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}
        words = tool.description or f"The MCP tool {tool.name}."
        about = f"From the MCP server {server.label}"
        about += f" ({server.description})." if server.description else "."
        out.append({"name": function, "description": f"{words}\n\n{about}", "parameters": schema})
    return out


def told(execution: McpExecution, limit: int) -> str:
    if execution.outcome == "over_limit":
        return (
            f"No more MCP calls are available for this response (its limit is {limit}). "
            "Answer with what you have."
        )
    if execution.outcome != "ok":
        return f'The MCP tool "{execution.name}" failed: {execution.error}. Answer without it.'
    return execution.output or "(the tool returned nothing)"


# --------------------------------------------------------------------------- #
# The items the caller gets back
# --------------------------------------------------------------------------- #


def list_item(listing: McpListing) -> dict[str, Any]:
    return {
        "id": listing.item_id,
        "type": "mcp_list_tools",
        "server_label": listing.label,
        "tools": [t.as_item() for t in listing.tools],
    }


def call_item(execution: McpExecution) -> dict[str, Any]:
    if execution.outcome == "pending":
        status = "in_progress"
    else:
        status = "completed" if execution.outcome == "ok" else "failed"
    return {
        "id": execution.item_id,
        "type": "mcp_call",
        "status": status,
        "server_label": execution.label,
        "name": execution.name,
        "arguments": execution.arguments,
        "output": execution.output if execution.outcome == "ok" else None,
        "error": (
            execution.error
            if execution.outcome == "error"
            else "not run: the response's limit on MCP calls was reached"
            if execution.outcome == "over_limit"
            else None
        ),
        "approval_request_id": execution.approval_request_id,
    }


def approval_item(approval: McpApproval) -> dict[str, Any]:
    return {
        "id": approval.item_id,
        "type": "mcp_approval_request",
        "server_label": approval.label,
        "name": approval.name,
        "arguments": approval.arguments,
    }


def pending_text() -> str:
    return _PENDING
