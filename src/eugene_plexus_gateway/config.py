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

import threading
from pathlib import Path
from typing import Any

import yaml

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

REDACTED = "<redacted>"

CATEGORY_LABELS: dict[str, str] = {
    "generation": "Generation defaults",
    "routing": "Routing",
    "lifecycle": "Lifecycle policy",
    "metrics": "Request metrics",
    "logging": "Logging",
    "clients": "Browser clients",
}

LOAD_BALANCING_VALUES = ["least_busy", "round_robin"]

FIELDS: list[ConfigField] = [
    ConfigField(
        key="defaultTemperature",
        label="Default temperature",
        description=(
            "Sampling randomness used when a request doesn't specify "
            "one. 0 = deterministic / always the most likely next "
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
            "Cap on a single response when a request doesn't specify "
            "one (roughly 0.75 words per token, so 2048 is about 1,500 "
            "words). Raise it for long-form work; keep it low for "
            "snappier chat."
        ),
        category="generation",
        valueType=ConfigValueType.integer,
        default=2048,
        minimum=1,
    ),
    ConfigField(
        key="requestTimeoutSeconds",
        label="Backend request timeout",
        description=(
            "How long the gateway waits on one backend before giving "
            "up and cascading to the next. Counts the whole HTTP "
            "request to the driver. Raise this if your slowest model "
            "needs longer — a local 70B on partial offload can take "
            "minutes for a long answer."
        ),
        category="routing",
        valueType=ConfigValueType.duration,
        default=180,
        minimum=5,
        maximum=900,
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
        key="loadBalancing",
        label="Load balancing",
        description=(
            "How a request picks among the replicas serving one model. "
            "`least_busy` sends it to the driver with the fewest requests "
            "in flight per slot of capacity, breaking ties round-robin — "
            "the default, and what fills capacity before queueing on a "
            "saturated replica. `round_robin` alternates strictly, which is "
            "worth having when comparing two replicas or reproducing a "
            "report. The signal is the gateway's own in-flight count, so it "
            "works for every engine."
        ),
        category="lifecycle",
        valueType=ConfigValueType.enum,
        default="least_busy",
        enumValues=LOAD_BALANCING_VALUES,
        enumLabels=["Least busy (ties round-robin)", "Round-robin"],
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


def as_schema() -> ConfigSchema:
    """Emit the gateway config schema.

    No dynamic fields any more: the old `voiceDriver` dropdown had to be
    rebuilt from the configured driver slots, and there are no configured
    driver slots — routing is derived. See `routing.py`.
    """
    return ConfigSchema(
        component="gateway",
        fields=list(FIELDS),
        categories=CATEGORY_LABELS,
    )


def _defaults() -> dict[str, Any]:
    return {f.key: f.default for f in FIELDS if f.default is not None}


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
        self._pending_restart: set[str] = set()

    def load(self) -> None:
        with self._lock:
            if self._path.exists():
                raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    raise ValueError(f"config file {self._path} must be a YAML mapping at the root")
                merged = _defaults()
                for k, v in raw.items():
                    if k in _FIELDS_BY_KEY:
                        merged[k] = v
                self._values = merged
            else:
                self._values = _defaults()
                self._write_locked()

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

                applied.append(key)
                if field.requiresRestart:
                    self._pending_restart.add(key)
                    pending_restart.append(key)

            if applied:
                self._write_locked()

            requires_restart = bool(self._pending_restart)
            return ConfigUpdateResult(
                applied=applied,
                rejected=rejected,
                requiresRestart=requires_restart,
                pendingRestart=sorted(self._pending_restart),
            )

    def get(self, key: str) -> Any:
        with self._lock:
            return self._values.get(key)

    def _write_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self._values, f, sort_keys=True, default_flow_style=False)
