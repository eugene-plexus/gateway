"""Runtime configuration: schema declaration + file-backed state + PATCH apply.

Implements the shared Eugene Plexus config protocol on the gateway
(`GET /v1/config/schema`, `GET /v1/config`, `PATCH /v1/config`). This is the
same code shape as in `inference-driver/config.py` — the two have to agree
on protocol semantics so a single UI can edit both.

The gateway's config is deliberately small. It holds no backend URLs and
no model list: the routing table is *derived* at runtime from the
agent topology plus each driver's `/v1/info`, so backend addresses
live in exactly one place and adding a model is not a config edit. What
is left here is the generation defaults the gateway stamps onto requests
and a few operational knobs.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any

import yaml

from . import _private_files, system_placement
from ._generated.models import (
    ComponentKind,
    ConfigDocument,
    ConfigField,
    ConfigFieldError,
    ConfigSchema,
    ConfigUpdateRequest,
    ConfigUpdateResult,
    ConfigValueType,
)

DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
"""The one place this number is written (R2.5).

It was written in three -- the schema default, `app.py`'s
``or 180`` and `RoutingTable`'s signature default -- which is how a
number drifts. 600 s matches the OpenAI Python SDK's own default, so
the commonest caller stops waiting at the same moment we stop serving,
and it sits one minute BELOW the driver's own backstop so that the
gateway's knob is the one that governs.
"""


REDACTED = "<redacted>"

CATEGORY_LABELS: dict[str, str] = {
    "generation": "Generation defaults",
    "routing": "Routing",
    "lifecycle": "Lifecycle policy",
    "metrics": "Request metrics",
    "logging": "Logging",
    "clients": "Browser clients",
}

#: Where a NEW conversation goes (CB1). Every value keeps a conversation on
#: its replica too, unless `conversationAffinity` is off.
LOAD_BALANCING_VALUES = ["spread", "least_busy", "round_robin"]
#: PC4's value, from before affinity was its own setting: read as the
#: default placement with affinity, which is what it did.
LEGACY_CONVERSATION = "conversation"

FIELDS: list[ConfigField] = [
    ConfigField(
        key="repetitionMode",
        label="Repeated response protection",
        description=(
            "Observe records repeated passages without stopping. Stop cancels streaming "
            "plain-text loops. Reasoning, tool arguments, structured output and whole "
            "responses are observation-only. A request can override this setting."
        ),
        category="generation",
        valueType=ConfigValueType.enum,
        enumValues=["off", "observe", "stop"],
        default="observe",
    ),
    ConfigField(
        key="repetitionStopModels",
        label="Models with automatic loop stopping",
        description=(
            "When protection is Stop, limit automatic stopping to these exact model or "
            "slot IDs, separated by commas. Other models observe only. Blank applies "
            "the selected mode to all models."
        ),
        category="generation",
        valueType=ConfigValueType.string,
        default="",
    ),
    ConfigField(
        key="repetitionMinChars",
        label="Minimum repeated passage length",
        description=(
            "Minimum whitespace-normalized characters per repeated passage. "
            "Initial evaluation threshold: 100. Patterns above 2048 characters "
            "are outside this detector's bounded window. This is not a token cap."
        ),
        category="generation",
        valueType=ConfigValueType.integer,
        default=100,
        minimum=64,
        maximum=2048,
    ),
    ConfigField(
        key="repetitionRepeats",
        label="Consecutive passage copies",
        description="Copies required before detection. Initial evaluation threshold: four.",
        category="generation",
        valueType=ConfigValueType.integer,
        default=4,
        minimum=3,
        maximum=8,
    ),
    ConfigField(
        key="defaultTemperature",
        label="Default temperature",
        description=(
            "Sampling randomness used when neither the request nor the model's "
            "default Library profile specifies one. 0 = deterministic / most likely next "
            "token; 1 = the model's own default randomness; higher "
            "gets more varied. The gateway owns this — whatever value "
            "reaches a backend, the gateway put it there, and a driver "
            "never substitutes a default of its own."
        ),
        category="generation",
        valueType=ConfigValueType.number,
        default=0.7,
        minimum=0.0,
        maximum=2.0,
    ),
    ConfigField(
        key="defaultMaxTokens",
        label="Default max output tokens",
        description=(
            "Cap on a single response when neither the request nor the model's "
            "default Library profile specifies one (roughly 0.75 words per token; 2048 is 1,500 "
            "words). Blank, the default, sets no cap: an answer runs until the model "
            "finishes, bounded by its context window and the request deadline. A cap "
            "cuts off a reasoning model that is still thinking, so it never answers. "
            "Not applied to /v1/responses (Codex CLI) either way. Set Max tokens on a "
            "model's profile to cap one model."
        ),
        category="generation",
        valueType=ConfigValueType.integer,
        # Blank since 2026-09-28: 2048 cut a reasoning model off mid-thought, so
        # it thought and never answered (found by a tester on Qwen).
        default=None,
        minimum=1,
    ),
    ConfigField(
        key="maxToolCalls",
        label="Web searches and images per request",
        description=(
            "How many web searches and images this install runs for one request when "
            "the caller does not say (P8). Past it the model is told no more are "
            "available and answers with what it has. A caller's own max_tool_calls "
            "(Codex) or max_uses (Claude Code) wins over this. Each search sends the "
            "model's query to your search account, and a paid one counts every search; "
            "each image is billed by the provider that makes it."
        ),
        category="generation",
        valueType=ConfigValueType.integer,
        default=5,
        minimum=1,
        maximum=50,
    ),
    ConfigField(
        key="imageToolModel",
        label="Image model for tools",
        description=(
            "The image model that answers an app's image_generation tool on "
            "/v1/responses when the app does not name one this install serves (P8e). "
            "Blank, the default, uses the one image model a key may use, and asks "
            "for this setting only when there are several. An id as /v1/models lists "
            "it, such as openrouter/openai/gpt-image-1."
        ),
        category="generation",
        valueType=ConfigValueType.string,
        default=None,
    ),
    ConfigField(
        key="profileCacheSeconds",
        label="Profile refresh interval",
        description=(
            "Seconds to reuse model generation defaults before reading Library again. "
            "Zero reads on every request. Edits apply without restarting a model."
        ),
        category="generation",
        valueType=ConfigValueType.duration,
        default=30.0,
        minimum=0,
        maximum=3600,
    ),
    ConfigField(
        key="profileMaxStaleSeconds",
        label="Profile outage grace period",
        description=(
            "Seconds after cache expiry to keep the last profile defaults when Library "
            "cannot answer. Then gateway defaults apply. Zero disables stale reuse. "
            "Lookup failures and fallback choices are logged."
        ),
        category="generation",
        valueType=ConfigValueType.duration,
        default=300.0,
        minimum=0,
        maximum=86400,
    ),
    ConfigField(
        key="requestTimeoutSeconds",
        label="Total request deadline",
        description=(
            "One elapsed time budget for admission, routing, model loading, "
            "generation and all fallback attempts. Raise it for large or "
            "partially offloaded models. Expiry cancels owned work and stops "
            "failover; a remote provider may still have acted."
        ),
        category="routing",
        valueType=ConfigValueType.duration,
        # **This is the deadline that governs, and that is deliberate
        # (R2.5).** Two deadlines exist on the path -- this one and the
        # driver's own `requestTimeoutSeconds` -- and until R2.5 they
        # were ordered the wrong way round: the driver's 120 s fired
        # first, so the knob an operator reached for here changed
        # nothing. The driver's default is now 660 s, one minute above
        # this, so the front door owns the answer and the driver's is a
        # backstop. Raise this one past 600 and raise the driver's too.
        #
        # 600 s matches the OpenAI Python SDK's own default timeout, so
        # a client that gives up and a gateway that gives up now agree
        # rather than racing: the commonest caller stops waiting at the
        # same moment we stop serving.
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        minimum=5,
        maximum=3600,
        requiresRestart=True,
    ),
    ConfigField(
        key="routingRefreshSeconds",
        label="Routing table refresh",
        description=(
            "How often the gateway re-reads the agent topology and "
            "asks each driver what it serves. This is what makes a "
            "newly-started engine routable without restarting the "
            "gateway, and what drops one that went away. Lower is more "
            "responsive and costs one cheap request per driver."
        ),
        category="routing",
        valueType=ConfigValueType.duration,
        default=15,
        minimum=2,
        maximum=300,
    ),
    ConfigField(
        key="decisionMaxQuestions",
        label="Decision questions per request",
        description=(
            "Ceiling on how many typed questions one POST /v1/systemone "
            "request may carry. A bound enforced before any backend work, "
            "because every question multiplies what a single-slot decision "
            "backend will hold capacity for. The protocol's own per-question "
            "bounds (255 choice options, 2-10 score levels) are fixed and "
            "not configurable."
        ),
        category="routing",
        valueType=ConfigValueType.integer,
        default=32,
        minimum=1,
        maximum=256,
    ),
    ConfigField(
        key="maxImagesPerRequest",
        label="Images per request",
        description=(
            "Most images one request may carry, counted across the whole "
            "conversation, because a chat client resends its history and the "
            "model sees every picture in it each time. A request over the "
            "limit is refused with a message naming this setting, before "
            "anything is sent to a model. Each image is still limited to "
            "5 MiB and the request to 10 MiB of images in total. At most 64."
        ),
        category="routing",
        valueType=ConfigValueType.integer,
        default=12,
        minimum=1,
        maximum=64,
    ),
    ConfigField(
        key="modelSlots",
        label="Model slots (priority lists)",
        description=(
            'Ordered fallbacks per model. Each entry is {"model": the name a '
            'client asks for, "targets": [model ids to try in order]}. A '
            "request for `model` is served by the drivers serving `model` "
            "itself (load-balanced across replicas), then by each target's "
            "drivers in turn when everything before it has nothing that "
            "answers. Targets are model ids, not driver names, because a "
            "model id names every replica serving it. A cloud subscription "
            "is a target like any other — a claude_code_cli driver already "
            "serves a model id. Example: "
            '[{"model": "coder", "targets": ["qwen3-coder-30b", "claude-opus-4-7"]}]. '
            "Takes effect on the next request; no restart."
        ),
        category="lifecycle",
        valueType=ConfigValueType.model_slots,
        default=[],
    ),
    ConfigField(
        key="inConversationSystem",
        label="System messages after the first",
        description=(
            "Where a system message that comes after the conversation has started "
            "goes. Claude Code sends them mid-conversation, and the Qwen 3.5 and "
            "later chat templates refuse any system message but the first, so those "
            "models answer an error. `user_turn` (the default) keeps each one where "
            "the client put it, as a user turn marked `<system-reminder>`, which "
            "every model measured accepts and which keeps the start of the prompt "
            "the same for the engine's cache; leading system messages become one. "
            "`system` sends them exactly as the client did."
        ),
        category="generation",
        valueType=ConfigValueType.enum,
        default="user_turn",
        enumValues=system_placement.PLACEMENTS,
        enumLabels=["As a user turn, in place", "As a system message, as sent"],
    ),
    ConfigField(
        key="loadBalancing",
        label="New conversations go to",
        description=(
            "Where a NEW conversation goes among the replicas serving one model. "
            "A conversation already under way goes back to the replica that served "
            "it whatever this says (see Keep each conversation on its replica). "
            "`spread` (the default) sends it to the replica holding the fewest "
            "conversations per slot, counting every conversation it served in the "
            "last 30 minutes whether or not a request is in flight, ties to the "
            "fewest in flight, then in turn. Measured with 32 agent sessions on "
            "eight replicas: 80.2% of prompt tokens reused against 60.4% by least "
            "busy, and the median first token in 1.05 s against 4.00 s, because a "
            "replica whose agents are all thinking between turns looks empty to a "
            "count of requests in flight. "
            "`least_busy` sends it to the driver with the fewest requests in flight "
            "per slot of capacity, breaking ties round-robin. `round_robin` "
            "alternates strictly, which is worth having when comparing two replicas "
            "or reproducing a report. The load signal is the gateway's own in-flight "
            "count, so both work for every engine. Until 2026-10-02 this also held "
            "`conversation`, which kept conversations on their replicas: a file "
            "that still says it is read as the default."
        ),
        category="lifecycle",
        valueType=ConfigValueType.enum,
        default="spread",
        enumValues=LOAD_BALANCING_VALUES,
        enumLabels=[
            "The replica holding the fewest conversations",
            "The least busy replica (ties round-robin)",
            "Each replica in turn",
        ],
    ),
    ConfigField(
        key="conversationAffinity",
        label="Keep each conversation on its replica",
        description=(
            "Send each turn of a conversation back to the replica that served its "
            "last one, because only that replica's engine holds its prompt in cache; "
            "it moves only when that replica is full and another has a free slot. "
            "Measured with twelve agent sessions on three replicas: on, 80.8% of "
            "prompt tokens were reused and the median first token came in 0.71 s; "
            "off, 67.7% and 1.49 s. Turn it off only to benchmark replicas against "
            "each other, where a tool repeating one prompt would otherwise land on "
            "one replica every time."
        ),
        category="lifecycle",
        valueType=ConfigValueType.boolean,
        default=True,
    ),
    ConfigField(
        key="swapWaitSeconds",
        label="Start-on-demand wait",
        description=(
            "How long a request waits for a stopped runtime marked "
            "`startOnDemand` to reach `ready` before the next fallback is "
            "tried (or a 503 is returned). Engine-dependent: llama.cpp loads "
            "a small model in seconds, a 30B off disk in a minute; vLLM's "
            "silent load is minutes. The response reports what it waited in "
            "`x_eugene_plexus.waited_ms`."
        ),
        category="lifecycle",
        valueType=ConfigValueType.duration,
        default=120,
        minimum=5,
        maximum=900,
    ),
    ConfigField(
        key="idleCheckSeconds",
        label="Idle check interval",
        description=(
            "How often the gateway checks each runtime's `idleUnloadSeconds` "
            "and asks its agent to unload one that has been idle that long "
            "with nothing in flight. A runtime that declared no timeout is "
            "never unloaded by the gateway."
        ),
        category="lifecycle",
        valueType=ConfigValueType.duration,
        default=15,
        minimum=2,
        maximum=300,
    ),
    ConfigField(
        key="controlUrl",
        label="Control root",
        description=(
            "Where the control root is, when this gateway should not work it "
            "out itself. Leave it empty and the gateway asks its own agent on "
            "every refresh: the control root this node is enrolled to, else "
            "the control component the agent runs, else this host alone. Set "
            "it to override that -- for a gateway whose agent is neither "
            "enrolled nor running a control root, or to force an address -- "
            "and it is used as given, even when wrong. Either way the gateway "
            "reads the node list from the root and talks to each node's own "
            "agent for that node's drivers and runtimes. Takes effect on the "
            "next routing refresh; the Inference screen's routing table says "
            "which address is in use and whether it answered."
        ),
        category="routing",
        valueType=ConfigValueType.url,
        componentKindHint=ComponentKind.control,
    ),
    ConfigField(
        key="logLevel",
        label="Log level",
        description=(
            "How chatty the gateway's terminal output is. `DEBUG` "
            "prints every routing decision and backend dispatch "
            "(useful when a request went somewhere surprising); "
            "`INFO` is the normal level; `WARNING` and `ERROR` go "
            "progressively quieter."
        ),
        category="logging",
        valueType=ConfigValueType.enum,
        default="INFO",
        enumValues=["DEBUG", "INFO", "WARNING", "ERROR"],
        requiresRestart=True,
    ),
    ConfigField(
        key="metricsEnabled",
        label="Retain request metrics",
        description=(
            "Keep a record of how each completion was served - which "
            "backend answered, how long it took, how many tokens it "
            "produced, whether failover fired, whether a sleeping model "
            "had to be woken. This is what makes 'is this backend faster "
            "than that one for this model' answerable.\n\n"
            "Stored on this machine only, in `metrics.sqlite3` beside "
            "this config. Never sent anywhere, and never part of the "
            "replicated install log. Turning it off stops recording from "
            "then on; nothing reconstructs the traffic served while it "
            "was off."
        ),
        category="metrics",
        valueType=ConfigValueType.boolean,
        default=True,
        requiresRestart=True,
    ),
    ConfigField(
        key="metricsRetentionDays",
        label="Keep individual requests for",
        description=(
            "How long the per-request rows are kept. Aggregates by hour "
            "are kept indefinitely regardless - they are tiny - so "
            "lowering this loses the ability to ask about a specific "
            "request, not the shape of a day.\n\n"
            "A request costs roughly 200 bytes, so a busy install at one "
            "completion per second is about 17 MB a day. A normal one is "
            "far below that."
        ),
        category="metrics",
        valueType=ConfigValueType.integer,
        default=7,
        minimum=0,
        maximum=365,
        requiresRestart=True,
    ),
    ConfigField(
        key="metricsRollupEnabled",
        label="Keep hourly aggregates",
        description=(
            "Summarise each hour once it has passed, and keep those "
            "summaries after the individual requests age out. This is "
            "what can still answer 'what did last night look like' a "
            "month later."
        ),
        category="metrics",
        valueType=ConfigValueType.boolean,
        default=True,
        requiresRestart=True,
    ),
    # --- Browser clients ---------------------------------------------------
    #
    # CORS on the OpenAI-compatible paths and nothing else. See `cors.py`
    # for why any origin is the safe default here: the front door
    # authenticates by an explicit bearer and never by a cookie, so a
    # page cannot spend a token it was not given. Both fields are read on
    # every request, so `requiresRestart` is false and true in fact.
    ConfigField(
        key="corsEnabled",
        label="Answer browser clients (CORS)",
        description=(
            "Let a web page on another origin call this gateway's OpenAI-compatible "
            "paths (/v1/models, /v1/chat/completions, /v1/embeddings) directly -- the "
            "playground's direct mode, Open WebUI in a tab, a web app built on the "
            "OpenAI SDK. Operator paths (/v1/config, /v1/admin, /v1/metrics) never "
            "answer browsers from another origin regardless. Off, a browser gets "
            "'Failed to fetch' and a curl of the same preflight gets a 403 that says "
            "why. Takes effect on the next request."
        ),
        category="clients",
        valueType=ConfigValueType.boolean,
        default=True,
    ),
    ConfigField(
        key="corsAllowedOrigins",
        label="Allowed browser origins",
        description=(
            "Which origins may call the OpenAI-compatible paths from a browser, as "
            "the browser spells them: scheme, host and port, no path -- "
            "http://192.168.1.20:8079, for example. Empty means any origin, which is "
            "safe because every request still needs a bearer token the page must "
            "hold; list origins to admit only your own UI's. Takes effect on the "
            "next request."
        ),
        category="clients",
        valueType=ConfigValueType.url_list,
        default=[],
    ),
]

_FIELDS_BY_KEY: dict[str, ConfigField] = {f.key: f for f in FIELDS}


#: What an unset value does -- for a list, an empty one -- shown where the
#: control would otherwise be blank or show a default the gateway is not
#: using (settings never lie, 2026-09-30). An empty `corsAllowedOrigins`
#: read "Default: none" when it means any website.
UNSET_MEANS: dict[str, str] = {
    "defaultMaxTokens": (
        "No cap: an answer runs until the model finishes, bounded by its context window "
        "and the request deadline, unless the request or the model's default profile "
        "sets one."
    ),
    "imageToolModel": (
        "Not set: the image tool uses the model the app names when this install serves "
        "it, else the one image model the key may use."
    ),
    "controlUrl": (
        "Not set: found through this machine's agent at every routing refresh -- the "
        "control root this node is enrolled to, or the one this machine runs."
    ),
    "corsAllowedOrigins": (
        "Empty: any website may call the three OpenAI paths from a browser. Every request "
        "still needs a bearer key the page must hold; list origins to admit only yours."
    ),
    "modelSlots": "None: each model is served only by its own backends, with no fallback.",
}


def as_schema(
    *,
    pending: dict[str, Any] | None = None,
    derived_control_url: str | None = None,
) -> ConfigSchema:
    """Emit the gateway config schema, with what this process knows now.

    `pending` is `ConfigStore.pending_restart()`: a field saved but not in
    effect says so, and what the running gateway uses. `derived_control_url`
    is where the last refresh found the control root, which is what an
    unset `controlUrl` stands for right now.
    """
    fields: list[ConfigField] = []
    for field in FIELDS:
        update: dict[str, Any] = {}
        if field.key in UNSET_MEANS:
            update["unsetMeans"] = UNSET_MEANS[field.key]
        if field.key == "controlUrl" and derived_control_url:
            update["unsetMeans"] = (
                f"Not set: found through this machine's agent -- {derived_control_url} at "
                "the last routing refresh."
            )
            update["unsetResolvesTo"] = derived_control_url
        if pending and field.key in pending:
            update["pendingRestart"] = True
            if not field.sensitive:
                update["inEffect"] = pending[field.key]
        fields.append(field.model_copy(update=update) if update else field)
    return ConfigSchema(component="gateway", fields=fields, categories=CATEGORY_LABELS)


def _defaults() -> dict[str, Any]:
    return {f.key: f.default for f in FIELDS if f.default is not None}


log = logging.getLogger(__name__)

#: The cap every install before 2026-09-28 wrote into its file, because the
#: store writes defaults out on first start. Read once as "never chosen".
_OLD_DEFAULT_MAX_TOKENS = 2048
#: Beside the config file: the old default has been cleared from it once, so a
#: 2048 an operator sets afterwards is theirs and stays.
_MAX_TOKENS_MARKER = ".default-max-tokens-cleared"
#: Beside the config file: the `loadBalancing` in it was written as the
#: default, not chosen, so a later default reaches it. The store writes every
#: default out, and without this a default written once looks like a choice
#: for ever. Removed when an operator sets the field.
_LOAD_BALANCING_DEFAULT_MARKER = ".load-balancing-default"


def _validate_value(field: ConfigField, value: Any) -> str | None:
    if value is None:
        return None

    vt = field.valueType

    if vt in (ConfigValueType.string, ConfigValueType.url, ConfigValueType.file_path):
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        if field.pattern is not None:
            import re

            if re.search(field.pattern, value) is None:
                return f"value does not match pattern {field.pattern!r}"
        return None

    if vt == ConfigValueType.secret:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        if value == REDACTED:
            return "refusing to write the literal redacted value back"
        return None

    if vt == ConfigValueType.integer:
        if isinstance(value, bool) or not isinstance(value, int):
            return f"expected integer, got {type(value).__name__}"
        if field.minimum is not None and value < field.minimum:
            return f"must be >= {field.minimum}"
        if field.maximum is not None and value > field.maximum:
            return f"must be <= {field.maximum}"
        return None

    if vt in (ConfigValueType.number, ConfigValueType.duration):
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"expected number, got {type(value).__name__}"
        if not math.isfinite(value):
            # JSON's `NaN` parses, and every comparison with it is false,
            # so it passed the range check and a timeout of NaN ended
            # every request at once.
            return "must be a finite number"
        if field.minimum is not None and value < field.minimum:
            return f"must be >= {field.minimum}"
        if field.maximum is not None and value > field.maximum:
            return f"must be <= {field.maximum}"
        return None

    if vt == ConfigValueType.boolean:
        if not isinstance(value, bool):
            return f"expected boolean, got {type(value).__name__}"
        return None

    if vt == ConfigValueType.enum:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        allowed = field.enumValues or []
        if value not in allowed:
            return f"must be one of {allowed}"
        return None

    if vt == ConfigValueType.url_list:
        # Same rule as the control root's standby list: a list of
        # non-empty strings, each named once. What an entry means is the
        # field's business -- for `corsAllowedOrigins` it is an origin,
        # which is a URL with no path.
        if not isinstance(value, list):
            return f"expected a list of URLs, got {type(value).__name__}"
        seen: dict[str, int] = {}
        for position, item in enumerate(value):
            if not isinstance(item, str):
                return f"entry {position} is {type(item).__name__}, expected a string"
            if not item.strip():
                return f"entry {position} is empty"
            if item in seen:
                return f"entry {position} duplicates entry {seen[item]} ({item!r})"
            seen[item] = position
        return None

    if vt == ConfigValueType.model_slots:
        return _validate_model_slots(value)

    return f"unsupported valueType: {vt}"


def _validate_model_slots(value: Any) -> str | None:
    """An ordered list of `{model, targets}`; every name a non-empty
    string, every model named once. Validated per entry so a typo says
    which line rather than "invalid"."""
    if not isinstance(value, list):
        return f"expected a list of {{model, targets}} entries, got {type(value).__name__}"
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            return f"entry {index}: expected an object with `model` and `targets`"
        model = item.get("model")
        if not isinstance(model, str) or not model.strip():
            return f"entry {index}: `model` must be a non-empty string"
        if model in seen:
            return f"entry {index}: model {model!r} is listed twice"
        seen.add(model)
        targets = item.get("targets")
        if not isinstance(targets, list) or not targets:
            return f"entry {index} ({model!r}): `targets` must be a non-empty list of model ids"
        for t_index, target in enumerate(targets):
            if not isinstance(target, str) or not target.strip():
                return f"entry {index} ({model!r}): target {t_index} must be a non-empty string"
        unknown = sorted(set(item) - {"model", "targets"})
        if unknown:
            return f"entry {index} ({model!r}): unknown key(s) {unknown}"
    return None


class ConfigStore:
    """File-backed config state. Thread-safe for the simple read/write pattern."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._values: dict[str, Any] = _defaults()
        # What this process runs on, for every `requiresRestart` field: the
        # value it had when the store was loaded. A field is pending a
        # restart while its saved value differs from this, and the schema
        # says so (`ConfigField.pendingRestart` / `inEffect`).
        self._started: dict[str, Any] = dict(self._values)

    def load(self) -> None:
        with self._lock:
            if self._path.exists():
                raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    raise ValueError(f"config file {self._path} must be a YAML mapping at the root")
                merged = _defaults()
                for k, v in raw.items():
                    field = _FIELDS_BY_KEY.get(k)
                    if field is None:
                        continue
                    # A `null` in the file for a field with a default is the
                    # default, as a PATCH of null is. `metricsEnabled: null`
                    # used to turn metrics off while GET said null and the
                    # schema said the default was on.
                    merged[k] = field.default if v is None and field.default is not None else v
                self._values = merged
                self._clear_old_max_tokens_locked()
                self._settle_load_balancing_locked(raw.get("loadBalancing"))
            else:
                self._values = _defaults()
                self._write_locked()
                self._mark_max_tokens_locked()
                self._mark_load_balancing_default_locked(True)
            self._started = dict(self._values)

    def _clear_old_max_tokens_locked(self) -> None:
        """Clear the 2048 cap an older install wrote into its file, once.

        The store writes every default out on first start, so an install
        from before 2026-09-28 holds `defaultMaxTokens: 2048` as though
        someone chose it, and changing the default alone would reach no
        existing install. It cut reasoning models off mid-thought. Read
        once, as the old default it almost always is; the marker keeps a
        2048 set after that.
        """
        marker = self._path.parent / _MAX_TOKENS_MARKER
        if marker.exists():
            return
        if self._values.get("defaultMaxTokens") == _OLD_DEFAULT_MAX_TOKENS:
            self._values.pop("defaultMaxTokens", None)
            self._write_locked()
            log.warning(
                "defaultMaxTokens was %d, the old shipped default, and is now blank: no "
                "cap unless a request or a model's profile sets one. Set it again under "
                "Gateway config to keep a cap.",
                _OLD_DEFAULT_MAX_TOKENS,
            )
        self._mark_max_tokens_locked()

    def _settle_load_balancing_locked(self, raw: Any) -> None:
        """A `loadBalancing` nobody chose follows the default (CB1).

        Unset, the old `conversation` (which since CB1 is the default plus
        affinity, always on), or written out as a default before -- the
        marker says which -- each reads as today's default. Anything else
        was chosen and stays.
        """
        default = _FIELDS_BY_KEY["loadBalancing"].default
        marker = self._path.parent / _LOAD_BALANCING_DEFAULT_MARKER
        unchosen = raw is None or raw == LEGACY_CONVERSATION or marker.exists()
        if not unchosen:
            return
        if raw == LEGACY_CONVERSATION:
            log.info(
                "loadBalancing was %r, which kept conversations on their replicas: that is "
                "conversationAffinity now, on by default, and new conversations go to %r.",
                LEGACY_CONVERSATION,
                default,
            )
        if self._values.get("loadBalancing") != default:
            self._values["loadBalancing"] = default
            self._write_locked()
        self._mark_load_balancing_default_locked(True)

    def _mark_load_balancing_default_locked(self, unchosen: bool) -> None:
        marker = self._path.parent / _LOAD_BALANCING_DEFAULT_MARKER
        try:
            if unchosen:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(
                    "loadBalancing in this file is the default, not a choice; see config.py.\n",
                    encoding="utf-8",
                )
            else:
                marker.unlink(missing_ok=True)
        except OSError as e:
            log.warning("could not record whether loadBalancing was chosen: %s", e)

    def _mark_max_tokens_locked(self) -> None:
        try:
            (self._path.parent / _MAX_TOKENS_MARKER).write_text(
                "defaultMaxTokens: the old 2048 default was cleared once; see config.py.\n",
                encoding="utf-8",
            )
        except OSError as e:
            log.warning("could not record that the old max tokens default was cleared: %s", e)

    def as_document(self) -> ConfigDocument:
        with self._lock:
            out: dict[str, Any] = {}
            for key, value in self._values.items():
                field = _FIELDS_BY_KEY.get(key)
                if field is not None and field.sensitive and value is not None:
                    out[key] = REDACTED
                else:
                    out[key] = value
            return ConfigDocument.model_validate(out)

    def apply_patch(self, request: ConfigUpdateRequest) -> ConfigUpdateResult:
        applied: list[str] = []
        rejected: list[ConfigFieldError] = []
        pending_restart: list[str] = []

        patch: dict[str, Any] = request.model_dump()

        with self._lock:
            for key, new_value in patch.items():
                field = _FIELDS_BY_KEY.get(key)
                if field is None:
                    rejected.append(ConfigFieldError(key=key, message="unknown field"))
                    continue

                err = _validate_value(field, new_value)
                if err is not None:
                    rejected.append(ConfigFieldError(key=key, message=err))
                    continue

                if new_value is None and field.default is not None:
                    self._values[key] = field.default
                else:
                    self._values[key] = new_value
                if key == "loadBalancing":
                    # Chosen now -- or, set back to null, the default again.
                    self._mark_load_balancing_default_locked(new_value is None)

                applied.append(key)
                # Only this PATCH's keys, and only those that now differ from
                # what the process runs on (the contract: a subset of
                # `applied`). This used to be every restart key saved since
                # the process started, so a later save of a live field said
                # a restart was needed and the UI restarted the gateway.
                if field.requiresRestart and self._values.get(key) != self._started.get(key):
                    pending_restart.append(key)

            if applied:
                self._write_locked()

            return ConfigUpdateResult(
                applied=applied,
                rejected=rejected,
                requiresRestart=bool(pending_restart),
                pendingRestart=pending_restart,
            )

    def pending_restart(self) -> dict[str, Any]:
        """`requiresRestart` fields whose saved value is not the one this
        process runs on, with the value it runs on."""
        with self._lock:
            return {
                f.key: self._started.get(f.key)
                for f in FIELDS
                if f.requiresRestart and self._values.get(f.key) != self._started.get(f.key)
            }

    def get(self, key: str) -> Any:
        with self._lock:
            return self._values.get(key)

    def _write_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # 0600 and replaced rather than rewritten; see `_private_files`.
        _private_files.write_private_text(
            self._path, yaml.safe_dump(self._values, sort_keys=True, default_flow_style=False)
        )
