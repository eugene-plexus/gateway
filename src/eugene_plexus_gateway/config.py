"""Runtime configuration: schema declaration + file-backed state + PATCH apply.

Implements the shared Eugene Plexus config protocol on the gateway
(`GET /v1/config/schema`, `GET /v1/config`, `PATCH /v1/config`). This is the
same code shape as in `inference-driver/config.py` — the two have to agree
on protocol semantics so a single UI can edit both.

The gateway's config is deliberately small. It holds no backend URLs and
no model list: the routing table is *derived* at runtime from the
watchdog topology plus each driver's `/v1/info`, so backend addresses
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
    "logging": "Logging",
}

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
            "How often the gateway re-reads the watchdog topology and "
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

    return f"unsupported valueType: {vt}"


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
