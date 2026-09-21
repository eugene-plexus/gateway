"""A bounded per-connection circuit; no waiting queue and one recovery probe."""

import math
import time
from dataclasses import dataclass


@dataclass
class Circuit:
    until: float = 0
    failures: int = 0
    successes: int = 0
    probing: bool = False
    epoch: int = 0
    _owner: int | None = None

    def acquire(self) -> bool:
        if self.failures == 0:
            return True
        if self.probing or time.perf_counter() < self.until:
            return False
        self.probing = True
        self._owner = self.epoch
        return True

    def finish(
        self, *, failed: bool, retry_after: float | None = None, probe_epoch: int | None = None
    ) -> None:
        owns_probe = probe_epoch is not None and probe_epoch == self._owner
        if owns_probe:
            self.probing = False
            self._owner = None
        if failed:
            self.epoch += 1
            self.failures = min(7, self.failures + 1)
            self.successes = 0
            delay = min(60, 2 ** (self.failures - 1))
            if retry_after is not None and math.isfinite(retry_after):
                delay = max(delay, min(300, max(0, retry_after)))
            self.until = max(self.until, time.perf_counter() + delay)
        elif self.failures and owns_probe and probe_epoch == self.epoch:
            # Completions from before the failure (or an invalidated probe)
            # cannot clear a newer cooldown or release another probe's claim.
            self.successes += 1
            if self.successes >= 2:
                self.failures = self.successes = 0
                self.until = 0
            else:
                self.until = time.perf_counter() + 1

    def abandon(self, *, probe_epoch: int | None = None) -> None:
        """Release a cancelled probe without inventing backend health evidence."""
        if probe_epoch is not None and probe_epoch == self._owner:
            self.probing = False
            self._owner = None
