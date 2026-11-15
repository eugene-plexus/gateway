"""The gateway's numbers, for a scraper and for a collector (row 12).

**Names that last.** Every name here starts with `eugene_` and is kept until a
major release: an alert or a dashboard built on one keeps working (call
MET-NAMES, Choice 2). Upstream's GenAI names are still in development, so none
is borrowed; the span attributes that upstream defines (`gen_ai.request.model`,
`gen_ai.usage.*`) are written as upstream writes them.

**What is counted** is what the retained request row already says
(`metrics.RequestRow`), seen once as it is recorded (`MetricsStore.observer`).
A row carries no prompt, no answer and no tool argument, so none can leave
here: the exported bytes are made of the row's numbers and the names of the
install's own models, drivers, runtimes and machines. **No person is a label**
(MET-PEOPLE, Choice 2); the key is, by its id, and `eugene_client_key_info`
says its name on the Prometheus page only.

**Bounded series.** `model` is a model the install serves or a slot it
configures, else `other`, so a made-up model name makes no series. A family
holds at most `MAX_SERIES` series, and a label set past it is counted in
`eugene_telemetry_series_dropped_total` and not kept. Buckets are fixed.

**Three ways out, none of them on the request path.** `render()` writes the
Prometheus text a scrape reads. `metrics_payload()` and `trace_payload()` write
OTLP/HTTP JSON by hand (call MET-OTLP: no SDK, no gRPC, no new dependency), and
`OtlpExporter` posts them from a task of its own: a full span buffer drops spans
and counts them, and a failing collector sets `lastError` and nothing else.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import random
import re
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from . import _build, fair_queue
from .metrics import RequestRow

log = logging.getLogger(__name__)

#: Fixed bucket bounds, in seconds, for every histogram: 50 ms to two minutes.
BUCKETS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)

#: The most series one family keeps. Labels are bounded by the install's own
#: names; this is the stop for the day one is not.
MAX_SERIES = 2000

#: The most keys whose names are kept for `eugene_client_key_info`.
MAX_KEYS = 1000

#: The most spans waiting to be sent. A full buffer drops the new span.
MAX_SPANS = 2048
#: The most spans in one post.
SPAN_BATCH = 512

#: Where a request came in, by path, for rows that carry no `door` of their own
#: (chat does not: its stored `door` is null).
DOORS = {
    "/v1/chat/completions": "chat",
    "/v1/completions": "completion",
    "/v1/messages": "messages",
    "/v1/responses": "responses",
    "/v1/embeddings": "embeddings",
    "/v1/rerank": "rerank",
    "/v1/systemone": "decisions",
    "/v1/moderations": "moderation",
    "/v1/audio/speech": "speech",
    "/v1/audio/transcriptions": "transcription",
    "/v1/audio/translations": "translation",
    "/v1/images/generations": "images",
    "/v1/images/edits": "images",
    "/v1/videos": "videos",
    "/v1/web_search": "web_search",
}
_DOOR_NAMES = frozenset(DOORS.values())

SERVICE = "eugene-plexus-gateway"

#: `(kind, help, label names)` for every family this module writes. The order
#: is the order of the page.
FAMILIES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "eugene_requests_total": (
        "counter",
        "Requests the gateway finished, by door, model and outcome.",
        ("door", "model", "outcome", "tier", "key_id"),
    ),
    "eugene_request_duration_seconds": (
        "histogram",
        "How long a request took at the front door, queue and failed attempts included.",
        ("door", "model", "outcome"),
    ),
    "eugene_time_to_first_token_seconds": (
        "histogram",
        "Time from the serving attempt's start to its first streamed event.",
        ("model",),
    ),
    "eugene_time_per_output_token_seconds": (
        "histogram",
        "Time per generated token after the first, for a streamed answer.",
        ("model",),
    ),
    "eugene_tokens_total": (
        "counter",
        "Tokens, as the serving backend reported them (input, output, cached).",
        ("type", "model", "runtime", "key_id"),
    ),
    "eugene_attempts_total": (
        "counter",
        "Backends touched, one per attempt, and whether that attempt served.",
        ("model", "driver", "runtime", "node", "served", "retry_disposition"),
    ),
    "eugene_attempt_duration_seconds": (
        "histogram",
        "How long one attempt took.",
        ("driver", "runtime", "node", "served"),
    ),
    "eugene_cascades_total": (
        "counter",
        "Requests that moved on from their first backend, by that backend.",
        ("model", "driver", "runtime", "node", "retry_disposition"),
    ),
    "eugene_routing_seconds": (
        "histogram",
        "How long deciding where to send a request took.",
        ("strategy",),
    ),
    "eugene_affinity_total": (
        "counter",
        "What conversation affinity did with a request.",
        ("model", "outcome"),
    ),
    "eugene_wake_total": (
        "counter",
        "Requests that had to wake a sleeping runtime.",
        ("model", "runtime"),
    ),
    "eugene_wake_wait_seconds": (
        "histogram",
        "How long a request waited for a runtime to wake.",
        ("model", "runtime"),
    ),
    "eugene_queue_wait_seconds": (
        "histogram",
        "How long a request waited in the fair queue.",
        ("band", "outcome"),
    ),
    "eugene_queue_refused_total": (
        "counter",
        "Requests the queue turned away, and why.",
        ("reason", "band"),
    ),
    "eugene_tool_executions_total": (
        "counter",
        "Tools the install ran for a request.",
        ("tool", "outcome"),
    ),
    "eugene_telemetry_series_dropped_total": (
        "counter",
        "Label sets not kept because a family was full.",
        ("family",),
    ),
    "eugene_telemetry_spans_dropped_total": (
        "counter",
        "Spans dropped because the export buffer was full or a post failed.",
        (),
    ),
}

#: Gauges are read when a scrape arrives: `(help, label names)`.
GAUGES: dict[str, tuple[str, tuple[str, ...]]] = {
    "eugene_in_flight": ("Requests running on a runtime now.", ("runtime", "node")),
    "eugene_runtime_slots": ("Parallel slots a runtime has.", ("runtime", "node")),
    "eugene_budget_reserved_tokens": (
        "Prompt tokens held in a runtime's shared context pool.",
        ("runtime", "node"),
    ),
    "eugene_budget_pool_tokens": (
        "Size of a runtime's shared context pool.",
        ("runtime", "node"),
    ),
    "eugene_queue_waiting": (
        "Requests waiting in the fair queue now.",
        ("runtime", "node", "band"),
    ),
    "eugene_circuit_open": ("1 while a driver's circuit is open.", ("driver",)),
    "eugene_metrics_rows_dropped_total": (
        "Request rows dropped because the write queue was full.",
        (),
    ),
    "eugene_metrics_enabled": (
        "1 while request counters are recorded (metricsEnabled).",
        (),
    ),
    "eugene_client_key_info": (
        "A client key this gateway has seen, with its name. Always 1.",
        ("key_id", "key_name"),
    ),
    "eugene_build_info": ("The build that answers. Always 1.", ("component", "version", "commit")),
}

#: Gauges that are counters in fact: read from a store that counts them.
_COUNTER_GAUGES = frozenset({"eugene_metrics_rows_dropped_total"})

#: Families an exported OTLP payload leaves out: a key's name is for the
#: Prometheus page behind the console's own access.
_NOT_EXPORTED = frozenset({"eugene_client_key_info"})

_LABEL_VALUE_MAX = 200


def _clean(value: object) -> str:
    text = "none" if value is None or value == "" else str(value)
    return text if len(text) <= _LABEL_VALUE_MAX else text[:_LABEL_VALUE_MAX]


@dataclass
class _Hist:
    counts: list[int] = field(default_factory=lambda: [0] * (len(BUCKETS) + 1))
    total: float = 0.0
    count: int = 0

    def add(self, seconds: float) -> None:
        for index, bound in enumerate(BUCKETS):
            if seconds <= bound:
                self.counts[index] += 1
                break
        else:
            self.counts[-1] += 1
        self.total += seconds
        self.count += 1


@dataclass
class Span:
    """One finished span, as OTLP writes it."""

    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    kind: int
    start_ns: int
    end_ns: int
    attributes: dict[str, str | int | float | bool] = field(default_factory=dict)
    error: bool = False


_TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")


def parse_traceparent(value: str | None) -> tuple[str, str, bool] | None:
    """`(trace id, parent span id, sampled)` of a W3C `traceparent`, or None."""
    found = _TRACEPARENT.match((value or "").strip().lower())
    if not found:
        return None
    trace, parent, flags = found.groups()
    if set(trace) == {"0"} or set(parent) == {"0"}:
        return None
    return trace, parent, bool(int(flags, 16) & 1)


def _hex(nbytes: int) -> str:
    return os.urandom(nbytes).hex()


@dataclass(frozen=True)
class OtlpSettings:
    """What the exporter reads from the gateway's config, now."""

    endpoint: str = ""
    authorization: str = ""
    metrics: bool = True
    traces: bool = True
    sample_ratio: float = 1.0
    interval_seconds: float = 60.0


class Telemetry:
    """Instruments, and the two writers over them."""

    def __init__(
        self,
        *,
        known_model: Callable[[str], bool] | None = None,
        settings: Callable[[], OtlpSettings] | None = None,
        version: str = "",
    ) -> None:
        self._known_model = known_model or (lambda _: True)
        self._settings = settings or OtlpSettings
        self._version = version
        self._counters: dict[str, dict[tuple[str, ...], float]] = {
            name: {} for name, (kind, _, _) in FAMILIES.items() if kind == "counter"
        }
        self._hists: dict[str, dict[tuple[str, ...], _Hist]] = {
            name: {} for name, (kind, _, _) in FAMILIES.items() if kind == "histogram"
        }
        self._models: set[str] = set()
        self._keys: dict[str, str] = {}
        self._spans: deque[Span] = deque()
        self.started_ns = time.time_ns()
        self.last_scrape_at: datetime | None = None
        self.last_scrape_key: str | None = None

    # --- bounded labels -----------------------------------------------------

    #: How many distinct model names get a series of their own.
    MAX_MODELS = 256

    def model_label(self, name: str | None) -> str:
        """The model as a label: one the install serves or configures, else `other`."""
        if not name:
            return "other"
        if name in self._models:
            return name
        if len(self._models) >= self.MAX_MODELS:
            return "other"
        try:
            known = self._known_model(name)
        except Exception:
            known = False
        if not known:
            return "other"
        self._models.add(name)
        return name

    # --- instruments --------------------------------------------------------

    def _count(self, family: str, labels: Sequence[object], by: float = 1) -> None:
        key = tuple(_clean(v) for v in labels)
        series = self._counters[family]
        if key not in series and len(series) >= MAX_SERIES:
            self._count("eugene_telemetry_series_dropped_total", [family])
            return
        series[key] = series.get(key, 0) + by

    def _observe_seconds(self, family: str, labels: Sequence[object], seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            return
        key = tuple(_clean(v) for v in labels)
        series = self._hists[family]
        hist = series.get(key)
        if hist is None:
            if len(series) >= MAX_SERIES:
                self._count("eugene_telemetry_series_dropped_total", [family])
                return
            hist = series[key] = _Hist()
        hist.add(seconds)

    # --- what a finished request adds --------------------------------------

    def observe(self, row: RequestRow) -> None:
        """Count one finished request. Never raises: a bug here must not cost a
        request its answer."""
        try:
            self._observe(row)
        except Exception:
            log.debug("could not count a request for telemetry", exc_info=True)

    def _observe(self, row: RequestRow) -> None:
        door = row.door if row.door in _DOOR_NAMES else None
        if door is None:
            door = DOORS.get(row.path or "", "chat" if row.path is None else "other")
        model = self.model_label(row.requested_model)
        key_id = row.client_key_id
        served = next((a for a in row.tries if a.served), None)
        last = served or (row.tries[-1] if row.tries else None)
        runtime = served.runtime if served is not None else None

        self._count(
            "eugene_requests_total",
            [door, model, row.outcome, row.tier if row.tier else None, key_id],
        )
        if row.outcome != "refused":
            self._observe_seconds(
                "eugene_request_duration_seconds", [door, model, row.outcome], row.total_ms / 1000
            )
        if served is not None and served.first_ms is not None:
            self._observe_seconds(
                "eugene_time_to_first_token_seconds", [model], served.first_ms / 1000
            )
            tokens = row.completion_tokens
            window = served.elapsed_ms - served.first_ms
            if tokens is not None and tokens >= 2 and window > 0:
                self._observe_seconds(
                    "eugene_time_per_output_token_seconds", [model], window / 1000 / (tokens - 1)
                )
        for kind, amount in (
            ("input", row.prompt_tokens),
            ("output", row.completion_tokens),
            ("cached", row.cached_tokens),
        ):
            if amount:
                self._count("eugene_tokens_total", [kind, model, runtime, key_id], amount)
        for attempt in row.tries:
            self._count(
                "eugene_attempts_total",
                [
                    model,
                    attempt.driver,
                    attempt.runtime,
                    attempt.node,
                    "true" if attempt.served else "false",
                    attempt.retry_disposition,
                ],
            )
            self._observe_seconds(
                "eugene_attempt_duration_seconds",
                [
                    attempt.driver,
                    attempt.runtime,
                    attempt.node,
                    "true" if attempt.served else "false",
                ],
                attempt.elapsed_ms / 1000,
            )
        if row.attempts > 1 and row.tries:
            first = row.tries[0]
            self._count(
                "eugene_cascades_total",
                [model, first.driver, first.runtime, first.node, first.retry_disposition],
            )
        if row.routing_ms is not None:
            self._observe_seconds("eugene_routing_seconds", [row.strategy], row.routing_ms / 1000)
        if row.affinity:
            self._count("eugene_affinity_total", [model, row.affinity])
        if row.swapped_in:
            self._count("eugene_wake_total", [model, runtime])
            self._observe_seconds(
                "eugene_wake_wait_seconds", [model, runtime], row.waited_ms / 1000
            )
        if row.band is not None:
            self._observe_seconds(
                "eugene_queue_wait_seconds", [row.band, row.outcome], (row.queued_ms or 0) / 1000
            )
        if row.outcome == "refused":
            self._count("eugene_queue_refused_total", [row.refusal, row.band])
        for tool in row.tool_executions:
            self._count("eugene_tool_executions_total", [tool.tool, tool.outcome])
        if key_id and (key_id in self._keys or len(self._keys) < MAX_KEYS):
            self._keys[key_id] = row.client_key_name or self._keys.get(key_id) or ""
        self._span(row, door, model, served, last)

    # --- spans ---------------------------------------------------------------

    def _span(self, row: RequestRow, door: str, model: str, served: Any, last: Any) -> None:
        settings = self._settings()
        if not (settings.endpoint and settings.traces):
            return
        parent = parse_traceparent(row.trace_parent)
        # Sampled once, at the door: a parent that was not sampled is not sampled
        # here, and a parent that was is kept at the install's own ratio.
        if parent is not None and not parent[2]:
            return
        if settings.sample_ratio < 1.0 and random.random() >= settings.sample_ratio:
            return
        # The row is recorded as the request ends, whichever way `started_at` was set.
        end_ns = time.time_ns()
        start_ns = end_ns - int(row.total_ms * 1e6)
        trace_id = parent[0] if parent is not None else _hex(16)
        root_id = _hex(8)
        spans: list[Span] = []
        root = Span(
            trace_id,
            root_id,
            parent[1] if parent is not None else None,
            f"{door} {model}",
            2,
            start_ns,
            end_ns,
            {
                "gen_ai.request.model": row.requested_model if model != "other" else "other",
                "eugene.door": door,
                "eugene.outcome": row.outcome,
                "eugene.streamed": bool(row.streamed),
            },
            error=row.outcome == "error",
        )
        if row.prompt_tokens is not None:
            root.attributes["gen_ai.usage.input_tokens"] = row.prompt_tokens
        if row.completion_tokens is not None:
            root.attributes["gen_ai.usage.output_tokens"] = row.completion_tokens
        if row.tier:
            root.attributes["eugene.tier"] = row.tier
        if row.queued_ms:
            root.attributes["eugene.latency.time_in_queue"] = row.queued_ms / 1000
        spans.append(root)
        cursor = start_ns
        if row.routing_ms is not None:
            spans.append(
                Span(
                    trace_id,
                    _hex(8),
                    root_id,
                    "eugene.routing",
                    1,
                    cursor,
                    cursor + int(row.routing_ms * 1e6),
                    {"eugene.strategy": row.strategy or "none"},
                )
            )
        for attempt in row.tries:
            begin = min(cursor, end_ns)
            finish = min(begin + int(attempt.elapsed_ms * 1e6), end_ns)
            cursor = finish
            attrs: dict[str, str | int | float | bool] = {
                "eugene.driver": attempt.driver,
                "eugene.served": bool(attempt.served),
            }
            if attempt.runtime:
                attrs["eugene.runtime"] = attempt.runtime
            if attempt.node:
                attrs["eugene.node"] = attempt.node
            if attempt.retry_disposition:
                attrs["eugene.retry_disposition"] = attempt.retry_disposition
            if attempt.error:
                attrs["error.type"] = attempt.error
            if attempt.first_ms is not None:
                attrs["eugene.latency.time_to_first_token"] = attempt.first_ms / 1000
            spans.append(
                Span(
                    trace_id,
                    _hex(8),
                    root_id,
                    "eugene.attempt",
                    3,
                    begin,
                    finish,
                    attrs,
                    error=not attempt.served,
                )
            )
        for span in spans:
            if len(self._spans) >= MAX_SPANS:
                self._count("eugene_telemetry_spans_dropped_total", [])
            else:
                self._spans.append(span)

    def take_spans(self, limit: int = SPAN_BATCH) -> list[Span]:
        out: list[Span] = []
        while self._spans and len(out) < limit:
            out.append(self._spans.popleft())
        return out

    def give_back_spans(self, spans: Sequence[Span]) -> None:
        """Spans a failed post did not deliver: counted as dropped, not retried
        without end."""
        if spans:
            self._count("eugene_telemetry_spans_dropped_total", [], len(spans))

    @property
    def spans_waiting(self) -> int:
        return len(self._spans)

    # --- the scrape ---------------------------------------------------------

    def note_scrape(self, who: str) -> None:
        self.last_scrape_at = datetime.now(UTC)
        self.last_scrape_key = who

    def gauge_samples(
        self, gauges: Mapping[str, Sequence[tuple[Sequence[object], float]]]
    ) -> dict[str, list[tuple[tuple[str, ...], float]]]:
        """The gauges a scrape read, plus the ones this object holds itself."""
        out: dict[str, list[tuple[tuple[str, ...], float]]] = {
            name: [(tuple(_clean(v) for v in labels), value) for labels, value in rows]
            for name, rows in gauges.items()
        }
        out["eugene_client_key_info"] = [
            ((_clean(key), _clean(name)), 1.0) for key, name in sorted(self._keys.items())
        ]
        out["eugene_build_info"] = [
            (("gateway", _clean(self._version), _clean(_build.commit())), 1.0)
        ]
        return out

    def render(self, gauges: Mapping[str, Sequence[tuple[Sequence[object], float]]]) -> str:
        """Prometheus text exposition (version 0.0.4)."""
        lines: list[str] = []
        for name, (kind, help_text, labels) in FAMILIES.items():
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {kind}")
            if kind == "counter":
                for key, value in sorted(self._counters[name].items()):
                    lines.append(f"{name}{_labels(labels, key)} {_num(value)}")
            else:
                for key, hist in sorted(self._hists[name].items()):
                    running = 0
                    for bound, count in zip(BUCKETS, hist.counts, strict=False):
                        running += count
                        lines.append(
                            f"{name}_bucket{_labels(labels, key, le=_num(bound))} {running}"
                        )
                    lines.append(f"{name}_bucket{_labels(labels, key, le='+Inf')} {hist.count}")
                    lines.append(f"{name}_sum{_labels(labels, key)} {_num(hist.total)}")
                    lines.append(f"{name}_count{_labels(labels, key)} {hist.count}")
        samples = self.gauge_samples(gauges)
        for name, (help_text, labels) in GAUGES.items():
            kind = "counter" if name in _COUNTER_GAUGES else "gauge"
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {kind}")
            for key, value in samples.get(name, []):
                lines.append(f"{name}{_labels(labels, key)} {_num(value)}")
        return "\n".join(lines) + "\n"

    # --- OTLP/HTTP JSON ------------------------------------------------------

    def _resource(self) -> dict[str, Any]:
        return {
            "attributes": [
                _attr("service.name", SERVICE),
                _attr("service.version", self._version),
            ]
        }

    def metrics_payload(
        self,
        gauges: Mapping[str, Sequence[tuple[Sequence[object], float]]],
        *,
        now_ns: int | None = None,
    ) -> dict[str, Any]:
        """`ExportMetricsServiceRequest` as JSON: cumulative counters and
        histograms since this gateway started, and the gauges now."""
        now = now_ns or time.time_ns()
        start = str(self.started_ns)
        stamp = str(now)
        metrics: list[dict[str, Any]] = []
        for name, (kind, help_text, labels) in FAMILIES.items():
            if name in _NOT_EXPORTED:
                continue
            if kind == "counter":
                points = [
                    {
                        "attributes": _attrs(labels, key),
                        "startTimeUnixNano": start,
                        "timeUnixNano": stamp,
                        "asDouble": value,
                    }
                    for key, value in sorted(self._counters[name].items())
                ]
                if points:
                    metrics.append(
                        {
                            "name": name,
                            "description": help_text,
                            "sum": {
                                "dataPoints": points,
                                "aggregationTemporality": 2,
                                "isMonotonic": True,
                            },
                        }
                    )
            else:
                hpoints = [
                    {
                        "attributes": _attrs(labels, key),
                        "startTimeUnixNano": start,
                        "timeUnixNano": stamp,
                        "count": str(hist.count),
                        "sum": hist.total,
                        "bucketCounts": [str(c) for c in hist.counts],
                        "explicitBounds": list(BUCKETS),
                    }
                    for key, hist in sorted(self._hists[name].items())
                ]
                if hpoints:
                    metrics.append(
                        {
                            "name": name,
                            "unit": "s",
                            "description": help_text,
                            "histogram": {"dataPoints": hpoints, "aggregationTemporality": 2},
                        }
                    )
        samples = self.gauge_samples(gauges)
        for name, (help_text, labels) in GAUGES.items():
            if name in _NOT_EXPORTED:
                continue
            rows = samples.get(name, [])
            if not rows:
                continue
            points = [
                {"attributes": _attrs(labels, key), "timeUnixNano": stamp, "asDouble": value}
                for key, value in rows
            ]
            if name in _COUNTER_GAUGES:
                metrics.append(
                    {
                        "name": name,
                        "description": help_text,
                        "sum": {
                            "dataPoints": [{**p, "startTimeUnixNano": start} for p in points],
                            "aggregationTemporality": 2,
                            "isMonotonic": True,
                        },
                    }
                )
            else:
                metrics.append(
                    {"name": name, "description": help_text, "gauge": {"dataPoints": points}}
                )
        return {
            "resourceMetrics": [
                {
                    "resource": self._resource(),
                    "scopeMetrics": [{"scope": {"name": SERVICE}, "metrics": metrics}],
                }
            ]
        }

    def trace_payload(self, spans: Sequence[Span]) -> dict[str, Any]:
        """`ExportTraceServiceRequest` as JSON."""
        out = []
        for span in spans:
            entry: dict[str, Any] = {
                "traceId": span.trace_id,
                "spanId": span.span_id,
                "name": span.name,
                "kind": span.kind,
                "startTimeUnixNano": str(span.start_ns),
                "endTimeUnixNano": str(span.end_ns),
                "attributes": [_attr(k, v) for k, v in span.attributes.items()],
                "status": {"code": 2 if span.error else 1},
            }
            if span.parent_id:
                entry["parentSpanId"] = span.parent_id
            out.append(entry)
        return {
            "resourceSpans": [
                {
                    "resource": self._resource(),
                    "scopeSpans": [{"scope": {"name": SERVICE}, "spans": out}],
                }
            ]
        }


# --- text helpers ------------------------------------------------------------


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(names: Sequence[str], values: Sequence[str], **extra: str) -> str:
    pairs = [*zip(names, values, strict=False), *extra.items()]
    if not pairs:
        return ""
    return "{" + ",".join(f'{k}="{_escape(v)}"' for k, v in pairs) + "}"


def _num(value: float) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return str(int(value)) if value == int(value) and abs(value) < 1e15 else repr(float(value))


def _attr(key: str, value: str | int | float | bool) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _attrs(names: Sequence[str], values: Sequence[str]) -> list[dict[str, Any]]:
    return [_attr(k, v) for k, v in zip(names, values, strict=False)]


# --- what a scrape reads from the gateway now -----------------------------------


def read_gauges(
    table: Any, *, rows_dropped: int | None, enabled: bool
) -> dict[str, list[tuple[tuple[object, ...], float]]]:
    """The gauges, read when a scrape arrives from the routing table, the pool
    ledger and the fair queue. Read only: nothing here takes a lease or a turn,
    and a part that cannot be read leaves its gauges out and nothing else."""
    out: dict[str, list[tuple[tuple[object, ...], float]]] = {
        "eugene_metrics_enabled": [((), 1.0 if enabled else 0.0)],
    }
    if rows_dropped is not None:
        out["eugene_metrics_rows_dropped_total"] = [((), float(rows_dropped))]
    if table is None:
        return out
    with contextlib.suppress(Exception):
        queue = fair_queue.of(table)
        inflight: list[tuple[tuple[object, ...], float]] = []
        slots: list[tuple[tuple[object, ...], float]] = []
        reserved: list[tuple[tuple[object, ...], float]] = []
        pools: list[tuple[tuple[object, ...], float]] = []
        for facts in table.runtimes():
            who = (facts.name, facts.node)
            inflight.append((who, float(table.runtime_inflight(facts.key))))
            if facts.parallel_slots:
                slots.append((who, float(facts.parallel_slots)))
            if facts.context_pool:
                pools.append((who, float(facts.context_pool)))
                reserved.append((who, float(queue.ledger.in_flight(facts.key))))
        out["eugene_in_flight"] = inflight
        out["eugene_runtime_slots"] = slots
        out["eugene_budget_reserved_tokens"] = reserved
        out["eugene_budget_pool_tokens"] = pools
    with contextlib.suppress(Exception):
        waiting: list[tuple[tuple[object, ...], float]] = []
        for group in fair_queue.of(table).snapshot():
            for band, count in group["waiting"].items():
                if count:
                    waiting.append(
                        (
                            (group["runtime"] or group["model"], group["node"], band),
                            float(count),
                        )
                    )
        out["eugene_queue_waiting"] = waiting
    with contextlib.suppress(Exception):
        open_by_driver: dict[str, float] = {}
        for bound_key, bound in getattr(table, "_bound", {}).items():
            name = bound_key[1]
            open_by_driver[name] = max(
                open_by_driver.get(name, 0.0), 0.0 if bound.circuit.available() else 1.0
            )
        out["eugene_circuit_open"] = [((n,), v) for n, v in sorted(open_by_driver.items())]
    return out


# --- the OTLP exporter -------------------------------------------------------------


def describe(exc: BaseException, endpoint: str) -> str:
    """The observed cause of a failed push, in words, with no credential in it."""
    try:
        host = httpx.URL(endpoint).netloc.decode() if endpoint else "the collector"
    except httpx.InvalidURL:
        host = "the collector"
    if isinstance(exc, httpx.ConnectTimeout | httpx.ReadTimeout | httpx.WriteTimeout):
        return f"{host} did not answer in time"
    if isinstance(exc, httpx.ConnectError):
        said = str(exc).strip()
        return f"could not connect to {host}" + (f": {said[:160]}" if said else "")
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{host} answered {exc.response.status_code}"
    if isinstance(exc, httpx.InvalidURL):
        return "the collector address is not a valid URL"
    # Anything else says its type and nothing more: the text of an error from the
    # request can hold what was sent, and a header holds the credential.
    return type(exc).__name__


def display_endpoint(endpoint: str) -> str | None:
    """The endpoint with no user, password or query in it."""
    if not endpoint:
        return None
    try:
        url = httpx.URL(endpoint)
    except httpx.InvalidURL:
        return None
    return str(url.copy_with(userinfo=b"", query=None, fragment=None)).rstrip("/")


class OtlpExporter:
    """Posts the gateway's metrics and spans to an OTLP/HTTP collector.

    One task, one HTTP client. It never touches a request: a collector that is
    down makes `state` `failing` and `last_error` say why, and nothing else.
    Off with no endpoint. The endpoint, the credential, the signals and the
    interval are read from the config at each round, so a PATCH applies at once.
    """

    def __init__(
        self,
        telemetry: Telemetry,
        *,
        settings: Callable[[], OtlpSettings],
        gauges: Callable[[], Mapping[str, Sequence[tuple[Sequence[object], float]]]],
        client_for: Callable[[str], httpx.AsyncClient],
        timeout_seconds: float = 10.0,
    ) -> None:
        self._telemetry = telemetry
        self._settings = settings
        self._gauges = gauges
        self._client_for = client_for
        self._timeout = timeout_seconds
        self._client: httpx.AsyncClient | None = None
        self._client_for_endpoint = ""
        self._task: asyncio.Task[None] | None = None
        self.state = "off"
        self.last_success_at: datetime | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="otlp-exporter")

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None

    async def _run(self) -> None:
        """Look every second; send when the interval is up or the address is new, so
        a collector set in the console is tried at once and not a minute later."""
        last: float | None = None
        seen: OtlpSettings | None = None
        while True:
            try:
                settings = self._settings()
                endpoint = settings.endpoint.strip()
                due = last is None or time.perf_counter() - last >= max(
                    1.0, settings.interval_seconds
                )
                if endpoint and (due or settings != seen):
                    last = time.perf_counter()
                    await self.push_once()
                elif not endpoint:
                    self.state = "off"
                    last = None
                seen = settings
            except asyncio.CancelledError:
                raise
            except Exception:
                log.debug("OTLP push failed unexpectedly", exc_info=True)
            await asyncio.sleep(1.0)

    def snapshot(self) -> dict[str, Any]:
        """`MetricsExport.otlp`."""
        endpoint = None
        with contextlib.suppress(Exception):
            endpoint = display_endpoint(self._settings().endpoint)
        # With an address and no push yet, nothing has failed: `ok`, and a null
        # `lastSuccessAt` says that nothing has been sent.
        return {
            "state": ("failing" if self.state == "failing" else "ok") if endpoint else "off",
            "endpoint": endpoint,
            "lastSuccessAt": self.last_success_at,
            "lastError": self.last_error if self.state == "failing" else None,
        }

    def _http(self, endpoint: str) -> httpx.AsyncClient:
        if self._client is None or self._client_for_endpoint != endpoint:
            if self._client is not None:
                # The proxy decision belongs to the address, so a new address
                # gets a new client; the old one is closed in the background.
                old = self._client
                asyncio.get_running_loop().create_task(_close(old))
            self._client = self._client_for(endpoint)
            self._client_for_endpoint = endpoint
        return self._client

    async def push_once(self) -> None:
        settings = self._settings()
        endpoint = settings.endpoint.strip().rstrip("/")
        if not endpoint:
            self.state = "off"
            return
        headers = {"Content-Type": "application/json"}
        credential = settings.authorization.strip()
        if credential:
            if any(ord(ch) < 32 or ord(ch) == 127 for ch in credential):
                # A header cannot hold it, and what the HTTP library would say
                # about it would repeat it.
                self._fail("the collector credential has a character a header cannot hold")
                return
            headers["Authorization"] = credential
        try:
            client = self._http(endpoint)
        except (ValueError, httpx.InvalidURL):
            self._fail("the collector address is not a valid URL")
            return
        spans: list[Span] = []
        try:
            if settings.metrics:
                response = await client.post(
                    f"{endpoint}/v1/metrics",
                    json=self._telemetry.metrics_payload(self._gauges()),
                    headers=headers,
                    timeout=self._timeout,
                )
                response.raise_for_status()
            if settings.traces:
                while batch := self._telemetry.take_spans():
                    spans = batch
                    response = await client.post(
                        f"{endpoint}/v1/traces",
                        json=self._telemetry.trace_payload(batch),
                        headers=headers,
                        timeout=self._timeout,
                    )
                    response.raise_for_status()
                    spans = []
        except (httpx.HTTPError, OSError, ValueError) as exc:
            self._telemetry.give_back_spans(spans)
            self._fail(describe(exc, endpoint), endpoint)
            return
        self.state = "ok"
        self.last_error = None
        self.last_success_at = datetime.now(UTC)

    def _fail(self, cause: str, endpoint: str = "") -> None:
        """Say the export is failing, and why, once per change: the log and the
        console read the same sentence, which has no credential in it."""
        changed = self.state != "failing" or self.last_error != cause
        self.state = "failing"
        self.last_error = cause
        if changed:
            log.warning("OTLP export to %s failed: %s", display_endpoint(endpoint), cause)


async def _close(client: httpx.AsyncClient) -> None:
    with contextlib.suppress(Exception):
        await client.aclose()
