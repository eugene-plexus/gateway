"""The global workspace — the loop's single locus of attention and state.

One Eugene = one consciousness = one workspace = one attention. The
continuous loop owns exactly one `Workspace`; because it is the *only*
thing that mutates cognitive state, there is no locking anywhere in the
cognition path (the single biggest simplification the single-attention
commitment buys).

NT state stays canonical on `app.state.nt_state` — the admin endpoint
reads it there and the loop ticks it in place — so the `Workspace` holds
the rest of the live loop state: what Eugene is attending to, and whether
it is awake or asleep.

Deferred to later M2 increments (hooks noted, not built here):
  * `adenosine` — fatigue / sleep pressure that drives the wake→sleep
    cycle (accumulates with thinking, cleared by sleep).
  * a richer `focus` model + a lossy pending-event buffer for
    salience-driven attention switching across multiple conversations.
See `docs/design/m1-continuous-runtime.md` in the `specs` repo.
"""

from __future__ import annotations

from dataclasses import dataclass

# Wake/sleep phase. `sleep` is the only true rest — offline consolidation
# (M5). `awake` never truly idles in the final design (it seeks
# stimulation); the first increment treats idle as a cheap no-op tick.
AWAKE = "awake"
ASLEEP = "asleep"


@dataclass
class Workspace:
    """Eugene's current locus of attention.

    `focus` is a human-readable label of what's being attended to — a
    conversation id, an internal topic, or None when idle. `phase` is the
    wake/sleep state. Everything else (NT, drivers, scorer, tools) lives
    on `app.state`; the loop reaches it through its dependency handle.
    """

    focus: str | None = None
    phase: str = AWAKE
