"""Bounded, whitespace-normalized repeated-passage observation.

This is a serving safeguard, not a sampler or a judgment of intent. It never
stores generated text in logs. Each channel has its own detector; a tool call
cannot complete a passage begun by another call or by the model's reasoning.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .driver_client import StreamEvent

log = logging.getLogger(__name__)
MODES = ("off", "observe", "stop")
HEADER = "X-Eugene-Repetition-Mode"
STOP_MESSAGE = (
    "Stopped because the response appears to be repeating. "
    "The partial answer has been kept. No automatic retry was made. "
    "For intentional repetition, turn off repetition protection for this request."
)
MAX_PERIOD = 2048
SCAN_EVERY = 64
MAX_TOOL_CHANNELS = 8


@dataclass(frozen=True)
class Policy:
    mode: str = "observe"
    min_chars: int = 100
    repeats: int = 4

    @classmethod
    def resolve(cls, get: Callable[[str], Any], model: str, override: str | None = None) -> Policy:
        mode = get("repetitionMode") or "observe"
        if mode not in MODES:
            mode = "observe"
        models = get("repetitionStopModels")
        if (
            mode == "stop"
            and isinstance(models, str)
            and models.strip()
            and model not in {v.strip() for v in models.replace("\n", ",").split(",")}
        ):
            mode = "observe"
        if override is not None:
            mode = override

        def integer(key: str, default: int, low: int, high: int) -> int:
            value = get(key)
            return value if type(value) is int and low <= value <= high else default

        return cls(
            mode=mode,
            min_chars=integer("repetitionMinChars", 100, 64, MAX_PERIOD),
            repeats=integer("repetitionRepeats", 4, 3, 8),
        )


@dataclass(frozen=True)
class Detection:
    period_chars: int
    repeats: int
    normalized_chars: int


class Detector:
    def __init__(self, policy: Policy | None = None) -> None:
        self.policy = policy or Policy()
        self.window_chars = MAX_PERIOD * self.policy.repeats
        self.buffer = ""
        self.total = 0
        self._pending = 0
        self._space = True
        self.detection: Detection | None = None

    def feed(self, text: str) -> Detection | None:
        if self.detection is not None:
            return None
        # Normalize incrementally, including whitespace split across chunks.
        # Never copy an entire (potentially large) driver chunk into our window.
        batch: list[str] = []
        for char in text:
            space = char.isspace()
            if space and self._space:
                continue
            self._space = space
            batch.append(" " if space else char)
            self.total += 1
            self._pending += 1
            if self._pending == SCAN_EVERY:
                self.buffer = (self.buffer + "".join(batch))[-self.window_chars :]
                batch.clear()
                self._pending = 0
                found = self._check()
                if found is not None:
                    return found
        self.buffer = (self.buffer + "".join(batch))[-self.window_chars :]
        return None

    def finish(self) -> Detection | None:
        if self.detection is not None or not self._pending:
            return None
        self._pending = 0
        return self._check()

    def _check(self) -> Detection | None:
        minimum = self.policy.min_chars
        count = self.policy.repeats
        if len(self.buffer) < minimum * count:
            return None
        # Search in CPython's string implementation instead of walking every
        # character of the window in Python at every checkpoint. Each candidate
        # must share a substantial suffix and then pass a full exact comparison.
        # At most MAX_PERIOD candidates, each at most window_chars characters;
        # no regex backtracking, unbounded index, or probabilistic hash matches.
        size = len(self.buffer)
        maximum = min(MAX_PERIOD, size // count)
        suffix = self.buffer[-minimum:]
        start = max(0, size - minimum - maximum)
        end = size - minimum
        while (position := self.buffer.rfind(suffix, start, end)) >= 0:
            period = size - minimum - position
            end = position + minimum - 1
            block = self.buffer[-period:]
            if self.buffer[-period * count :] != block * count:
                continue
            # A hundred spaces, dashes or copies of a tiny refrain do not
            # become a substantial passage just by grouping them in hundreds.
            primitive = (block + block).find(block, 1)
            if primitive < minimum:
                return None
            self.detection = Detection(period, count, self.total)
            return self.detection
        return None


class Guard:
    def __init__(
        self, policy: Policy, *, structured: bool = False, request_id: str | None = None
    ) -> None:
        self.policy = policy
        self.structured = structured
        self.request_id = request_id
        self.channels: dict[str, Detector] = {}
        self.tool_seen = False

    def take(self, event: StreamEvent) -> bool:
        """Observe independently; only visible, unstructured text may stop."""
        if self.policy.mode == "off":
            return False
        self.tool_seen = self.tool_seen or bool(event.tool_calls)
        stop = False
        if event.text:
            stop = self._feed("text", event.text)
        if event.reasoning:
            self._feed("reasoning", event.reasoning)
        for call in event.tool_calls or []:
            index = call.get("index")
            arguments = (call.get("function") or {}).get("arguments")
            if type(index) is int and 0 <= index < MAX_TOOL_CHANNELS and isinstance(arguments, str):
                self._feed(f"tool:{index}", arguments)
        if event.done:
            for channel, detector in self.channels.items():
                if found := detector.finish():
                    self._report(channel, found, stop=False)
        return stop

    def _feed(self, channel: str, text: str) -> bool:
        if channel not in self.channels:
            self.channels[channel] = Detector(self.policy)
        detector = self.channels[channel]
        found = detector.feed(text)
        if found is None:
            return False
        stop = (
            self.policy.mode == "stop"
            and channel == "text"
            and not self.structured
            and not self.tool_seen
        )
        self._report(channel, found, stop=stop)
        return stop

    def _report(self, channel: str, found: Detection, *, stop: bool) -> None:
        # Numeric metadata only: no prompts, output passages, or tool arguments.
        log.info(
            "repetition_detected channel=%s action=%s period_chars=%d repeats=%d "
            "normalized_chars=%d request_id=%s",
            channel,
            "stop" if stop else "observe",
            found.period_chars,
            found.repeats,
            found.normalized_chars,
            self.request_id,
        )
