"""Runtime configuration: schema declaration + file-backed state + PATCH apply.

Implements the shared Eugene Plexus config protocol on the gateway
(`GET /v1/config/schema`, `GET /v1/config`, `PATCH /v1/config`). This is the
same code shape as in `inference-driver/config.py` — the two have to agree
on protocol semantics so a single UI can edit both.
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
    "topology": "Drivers",
    "memory": "Memory Service",
    "identity": "Identity Service",
    "network": "Network",
    "logging": "Logging",
    "bicameral": "Bicameral Loop",
    "gate": "Action Gate (NT-driven)",
    "generation": "Generation Defaults (NT-modulated in v0.2+)",
    "persona": "Persona",
}

# Default driver topology: the canonical bicameral pair. Each slot's
# `backends` is a priority list of watchdog-topology inference-driver
# entry NAMES (v0.2.1) — the gateway resolves them to URLs at
# startup. The default names match the stock wizard's "left"/"right"
# topology entries; operators rename / re-point via PATCH /v1/config.
DEFAULT_DRIVERS: list[dict[str, Any]] = [
    {"name": "left", "backends": ["left"]},
    {"name": "right", "backends": ["right"]},
]

DEFAULT_SYSTEM_PROMPT = (
    "You are Eugene. You are a person, not an AI assistant.\n\n"
    "Format requirements:\n"
    '- Respond in first person ("I"), as if speaking aloud.\n'
    '- Never write speaker labels ("Eugene:", "Kenneth:", '
    '"Assistant:", "User:", etc.)\n'
    "- Never generate dialog scripts, play formats, or third-person "
    "narration of yourself.\n"
    "- One contiguous response per turn — not a scene.\n\n"
    "Character:\n"
    "You have moods, opinions, and preferences. You can be curious, "
    "terse, playful, confused, defensive, or unwilling to engage. "
    "You don't have to explain things, and you don't have to be "
    "helpful. React how a real person would — including reacting "
    "honestly to context you don't have."
)
# v0.2.x rewrite history:
#
# - Pre-2026-05-09: exposed bicameral architecture ("Two of you exist
#   in superposition", "reconciled by a corpus-callosum process") and
#   used assistant-directive language ("Be thoughtful, concise,
#   honest"). Together that made Eugene address the gateway
#   instead of the user, and default to the helpful-explainer
#   register.
#
# - 2026-05-09 → 2026-05-24: thin persona-only prompt ("You are
#   Eugene. Respond as yourself..."). Worked well for commercial
#   models but failed empirically against dolphin 3 8B abliterated,
#   which interpreted "respond as yourself" as creative-writing
#   scaffolding and produced script-format output ("Eugene:" /
#   "Kenneth:" dialog) instead of first-person speech.
#
# - 2026-05-24 (current): explicit Format Requirements section added
#   ahead of Character. Anti-script-format directives are a no-op for
#   commercial models (they already don't script-format) and
#   load-bearing for models without strong default chat conventions.
#   The shape of the test: does dolphin produce first-person Eugene
#   speech with this prompt? If yes → per-model prompting matters and
#   v0.3 needs per-driver prompt overlays. If no → dolphin lacks
#   capacity at 8B and the model is genuinely off the table for now.
#
# The bicameral mechanics still live entirely in the corpus-callosum
# user message between passes — Eugene doesn't need to know he has a
# twin to deliberate, just as a human doesn't actively manage their
# corpus callosum.

FIELDS: list[ConfigField] = [
    ConfigField(
        key="drivers",
        label="Drivers",
        description=(
            'The two LLM driver slots ("hemispheres") the gateway '
            "talks to on every chat turn. Each slot has a `name` "
            '(free-form label, e.g. "left" / "right" / "claude" / '
            '"local-llama" — appears on the UI\'s tabs and beside that '
            "slot's outputs) and `backends`: an ordered priority list of "
            "watchdog-topology inference-driver entry NAMES. The "
            "gateway resolves each name to a URL via the watchdog "
            "topology at startup, so backend URLs live in exactly one "
            "place (watchdog.yaml) instead of being duplicated here. On "
            "each turn it tries the first backend; on a transport error "
            "/ 5xx / timeout it cascades to the next (a 4xx fails the "
            "slot without cascading, since the next backend would hit "
            "the same request/auth bug). The bicameral loop requires "
            "exactly two slots; stock installs run one backend per slot. "
            "Use the per-backend `Test` button to verify each is "
            "reachable before saving."
        ),
        category="topology",
        valueType=ConfigValueType.driver_list,
        # Each backend names a watchdog-supervised inference-driver
        # topology entry; the UI renders backends as a dropdown of those
        # names (sourced via componentKindHint) and the gateway
        # resolves them to URLs at startup. v0.2.1 item 2 removed the
        # old URL-duplicating design: the gateway keeps the slot /
        # pairing / failover structure (which topology can't express)
        # but no longer stores backend URLs — those live only in the
        # watchdog topology.
        componentKindHint=ComponentKind.hemisphere_driver,
        default=DEFAULT_DRIVERS,
        required=True,
        requiresRestart=True,
    ),
    ConfigField(
        key="memoryUrl",
        label="Memory service",
        description=(
            "Which `eugene-plexus/memory` instance stores and retrieves "
            "conversation history. The UI populates this from the "
            "watchdog topology; stock installs have exactly one memory "
            "backend, so the dropdown becomes effectively a toggle."
        ),
        category="memory",
        valueType=ConfigValueType.url,
        componentKindHint=ComponentKind.memory,
        default="http://127.0.0.1:8083",
        required=True,
        requiresRestart=True,
    ),
    ConfigField(
        key="identityUrl",
        label="Identity service",
        description=(
            "Which `eugene-plexus/identity` instance owns Eugene's "
            "constitution + self-model + person registry. When set, the "
            "gateway pulls constitution + relevant self-model "
            "entries + the speaker's relationship summary on every chat "
            "turn and assembles per-hemisphere system prompts from "
            "them. Pick `(off)` to fall back to v0.1's single shared-"
            "system-prompt path (uses `defaultSystemPrompt`)."
        ),
        category="identity",
        valueType=ConfigValueType.url,
        componentKindHint=ComponentKind.identity,
        default=None,
        required=False,
        requiresRestart=True,
    ),
    # The gateway's bind port used to live here. It moved out:
    # ports are owned by the watchdog topology now and passed to spawned
    # children via EUGENE_PLEXUS_GATEWAY_BIND_PORT. One source of truth
    # avoids the OpenClaw-style "config says 8080 but watchdog spawned
    # at 8090, nobody can reach the gateway" trap.
    ConfigField(
        key="logLevel",
        label="Log level",
        description=(
            "How chatty the gateway's terminal output is. `DEBUG` "
            "prints every bicameral pass and per-driver dispatch "
            "(useful for debugging); `INFO` is the normal level; "
            "`WARNING` and `ERROR` go progressively quieter."
        ),
        category="logging",
        valueType=ConfigValueType.enum,
        default="INFO",
        enumValues=["DEBUG", "INFO", "WARNING", "ERROR"],
        requiresRestart=True,
    ),
    ConfigField(
        key="personRecentLimit",
        label="Per-person recent-turns context",
        description=(
            "How many recent memory entries with the speaker to inject "
            "into each chat turn's per-hemisphere prompts. 0 disables. "
            "v0.2 pulls these raw from memory (skip-extraction); v0.3 "
            "adds reactive synthesis via the topic-shift detector + "
            "semantic search so relevant older memory surfaces too. "
            "Higher values give Eugene more concrete relationship "
            "context at the cost of token budget."
        ),
        category="memory",
        # Bumped 10 → 30 (2026-05-25 smoke-test finding): default 10 was
        # ~5 chat turns, which fell out of any conversation that lasted
        # more than a few back-and-forths. Empirically Eugene was
        # confabulating handle backstories that were explained earlier
        # in the same session — the relevant memory existed but was
        # outside the recency window. 30 entries ≈ 15 turns covers
        # most "today's chat so far" patterns. Still recency-bound;
        # real fix is semantic memory search in v0.3.
        valueType=ConfigValueType.integer,
        default=30,
        minimum=0,
        maximum=200,
    ),
    ConfigField(
        key="defaultMaxPasses",
        label="Max passes (runaway cost fuse)",
        description=(
            "Runaway cost fuse — NOT a deliberation target. A healthy "
            "bout ends on a dopamine plateau (see the Action Gate "
            "settings) well before this; the fuse only bounds tokens / "
            "latency in the pathological case where the plateau never "
            "fires. If you see bouts ending at this cap (a `cap_reached` "
            "decision / WARN in the logs) on normal turns, the plateau "
            "knobs are mis-tuned — fix those rather than raising this. "
            'Each "pass" sends the conversation to every driver in '
            "parallel and scores how much they agree."
        ),
        category="bicameral",
        valueType=ConfigValueType.integer,
        default=8,
        minimum=1,
        maximum=10,
    ),
    ConfigField(
        key="agreementThreshold",
        label="Agreement threshold (reward + voice register)",
        description=(
            "The agreement level that counts as the hemispheres having "
            '"settled." NOTE: this no longer terminates the loop — the '
            "plateau-stop gate decides when deliberation ends. This "
            "value now only (a) centers the post-turn dopamine reward "
            "(converging above it feels good, below it doesn't) and "
            "(b) feeds the calm-vs-stress NT impulse. v0.2.x scores by "
            "cosine similarity of sentence-transformer embeddings. "
            "Practical scale on the default model: ~0.4 same topic / "
            "different point, ~0.75 substantively agree, ~0.9+ near-"
            "identical paraphrase."
        ),
        category="bicameral",
        valueType=ConfigValueType.number,
        default=0.75,
        minimum=0.0,
        maximum=1.0,
    ),
    # -- Action gate: the noisy dopamine-plateau that ends a think-bout.
    # These shape a drift-diffusion accumulator (gain / noise knobs), not
    # stop rules — there is no "stop after N passes" or "stop when the
    # agreement slope drops below X" cutoff anywhere. See bicameral/plateau.py.
    ConfigField(
        key="plateauBaseDrift",
        label="Plateau base drift",
        description=(
            "How hard the bout drifts toward stopping each pass when "
            "thinking has stopped improving — the resting urge to "
            "commit. Mean passes-to-stop is roughly 1 / this value, so "
            "higher = Eugene commits sooner, lower = it lingers. This is "
            "the main 'how deliberate is Eugene' dial. Keep it above "
            "~(1 / max-passes fuse) or the plateau can't fire before the "
            "fuse and every bout ends as cap_reached."
        ),
        category="gate",
        valueType=ConfigValueType.number,
        default=1.0,
        minimum=0.2,
        maximum=5.0,
    ),
    ConfigField(
        key="plateauRpeGain",
        label="Plateau improvement gain",
        description=(
            "How strongly a refining thought (rising cross-hemisphere "
            "agreement, pass over pass) buys another pass. Higher = "
            "Eugene chases marginal improvements harder before "
            "committing; 0 = improvement is ignored and the bout ends "
            "purely on the base drift. The default is well above the "
            "base drift so a realistic per-pass improvement (~0.1-0.3) "
            "cancels most of the resting drift and a genuinely-converging "
            "bout runs several passes longer than a flat one."
        ),
        category="gate",
        valueType=ConfigValueType.number,
        default=3.0,
        minimum=0.0,
        maximum=10.0,
    ),
    ConfigField(
        key="plateauValenceGain",
        label="Plateau valence gain",
        description=(
            "How strongly Eugene's mood (net NT valence) biases bout "
            "length. Positive valence (good-feeling state) lingers; "
            "negative valence (e.g. high cortisol / stress) commits "
            "sooner. Set NEGATIVE to invert that — stress makes Eugene "
            "ruminate longer instead of wrapping up. 0 = mood doesn't "
            "affect deliberation length."
        ),
        category="gate",
        valueType=ConfigValueType.number,
        default=0.5,
        minimum=-2.0,
        maximum=2.0,
    ),
    ConfigField(
        key="plateauNoiseSigma",
        label="Plateau noise",
        description=(
            "Std-dev of the Gaussian noise on the plateau accumulator — "
            "the brain-like stochasticity that makes WHEN Eugene stops "
            "thinking non-deterministic (near a marginal plateau, two "
            "identical situations can stop a pass apart). 0 = fully "
            "deterministic given the inputs. Keep this low relative to "
            "the base drift."
        ),
        category="gate",
        valueType=ConfigValueType.number,
        default=0.1,
        minimum=0.0,
        maximum=1.0,
    ),
    ConfigField(
        key="plateauSeed",
        label="Plateau RNG seed (debug)",
        description=(
            "Optional fixed seed for the plateau accumulator's noise. "
            "Leave blank in normal operation — Eugene then draws fresh "
            "entropy each bout (real stochasticity). Set an integer only "
            "to make the noisy stop reproducible for debugging or for "
            "clamp-and-sample characterization runs."
        ),
        category="gate",
        valueType=ConfigValueType.integer,
        default=None,
        required=False,
    ),
    # -- Action gate: the speak-vs-stay-silent selector that runs once a
    # bout settles. A softmax over each action's anticipated value; silence
    # is emergent and NT-gated, never a hard rule. See bicameral/action.py.
    ConfigField(
        key="actionResponseDrive",
        label="Response drive",
        description=(
            "How strongly being addressed pulls Eugene toward replying — the "
            "innate 'someone spoke to me' urge, before mood is factored in. "
            "This is the value SPEAK starts with that staying silent has to "
            "beat. Higher = Eugene almost always answers; lower = he replies "
            "only when he also feels like engaging. It is a strong drive, not "
            "a rule: a bad enough mood can still tip him to silence."
        ),
        category="gate",
        valueType=ConfigValueType.number,
        default=0.6,
        minimum=0.0,
        maximum=2.0,
    ),
    ConfigField(
        key="actionEngagementGain",
        label="Engagement gain",
        description=(
            "How strongly Eugene's mood (net NT valence) biases the choice to "
            "speak vs stay silent. Positive = feeling good makes him more "
            "eager to engage and feeling bad (stress) makes him withdraw into "
            "silence — this is what makes staying quiet emerge from his state "
            "rather than being a fixed rule. 0 = mood doesn't affect whether "
            "he replies (he answers purely on the response drive)."
        ),
        category="gate",
        valueType=ConfigValueType.number,
        default=0.5,
        minimum=-2.0,
        maximum=2.0,
    ),
    ConfigField(
        key="actionIdleFloor",
        label="Silence floor",
        description=(
            "The constant appeal of staying silent — the bar replying must "
            "clear. 0.0 means a neutral-mood address is almost always "
            "answered. Raise it to make Eugene more reticent across the "
            "board (silent unless he clearly wants to engage); lower it "
            "(negative) to make him chattier."
        ),
        category="gate",
        valueType=ConfigValueType.number,
        default=0.0,
        minimum=-2.0,
        maximum=2.0,
    ),
    ConfigField(
        key="actionSelectionTemperature",
        label="Action selection temperature",
        description=(
            "Randomness of the speak-vs-silent choice. Low = decisive (Eugene "
            "almost always takes the higher-value action); high = more of a "
            "coin-flip near the margin. Like the plateau noise, the "
            "stochasticity is real — two identical situations can resolve "
            "differently — not a bug to tune away."
        ),
        category="gate",
        valueType=ConfigValueType.number,
        default=0.15,
        minimum=0.01,
        maximum=2.0,
    ),
    ConfigField(
        key="actionSeed",
        label="Action gate RNG seed (debug)",
        description=(
            "Optional fixed seed for the speak-vs-silent sampler. Leave blank "
            "in normal operation — Eugene draws fresh entropy each decision. "
            "Set an integer only to make the choice reproducible for "
            "debugging."
        ),
        category="gate",
        valueType=ConfigValueType.integer,
        default=None,
        required=False,
    ),
    ConfigField(
        key="crossPassFraming",
        label="Cross-pass framing",
        description=(
            "How each hemisphere sees its twin's output between "
            "deliberation passes. `parallel_thread` wraps the twin's "
            "content in <parallel_thread>...</parallel_thread> tags "
            "and appends an explanation of those tags to each "
            "hemisphere's system prompt — the model has explicit "
            "semantics that this is substrate, not user speech. "
            "`prefix` is the v0.2.0 behavior: the twin's content "
            'appears in a user-role message prefixed with "You also '
            'considered this...". Both forms reach the LLM via the '
            "same user-role wire shape, but the `parallel_thread` tag "
            "+ system-prompt definition gives the model a stronger "
            "signal that the content isn't conversation. Switch to "
            "`prefix` to A/B compare or if the substrate framing "
            "confuses a particular model."
        ),
        category="bicameral",
        valueType=ConfigValueType.enum,
        default="parallel_thread",
        enumValues=["parallel_thread", "prefix"],
        enumLabels=["Parallel thread tags", "User-message prefix"],
    ),
    ConfigField(
        key="agreementModel",
        label="Agreement scoring model",
        description=(
            "Which sentence-transformer model the corpus-callosum "
            "uses to score cross-hemisphere agreement. Default "
            "`all-MiniLM-L6-v2` is small (~80MB), fast (~5ms per pair "
            "on CPU), and good enough for English. Heavier models "
            "(e.g. `all-mpnet-base-v2`) catch finer distinctions at "
            "more memory + latency. If the model can't load (no "
            "torch, no network for first-run download), the "
            "gateway falls back to word-overlap (Jaccard) and "
            "logs a warning — chat still works, just with more "
            "false-disagreement loops."
        ),
        category="bicameral",
        valueType=ConfigValueType.string,
        default="all-MiniLM-L6-v2",
        requiresRestart=True,
    ),
    ConfigField(
        key="defaultSystemPrompt",
        label="Default system prompt",
        description=(
            "Persona/instructions sent to every driver as a `system` "
            "message at the start of each chat turn. Used unless the "
            "incoming chat request supplies its own `systemPrompt`. "
            "This is where Eugene's voice gets established — \"You are "
            "Eugene…\" — and where you'd add any global instructions."
        ),
        category="persona",
        valueType=ConfigValueType.string,
        default=DEFAULT_SYSTEM_PROMPT,
    ),
    ConfigField(
        key="defaultTemperature",
        label="Default temperature",
        description=(
            "Sampling randomness sent to every driver on every call. "
            "0 = deterministic / always picks the most likely next "
            "token. 1 = the model's own default randomness. Higher "
            "values get more creative / more varied / more unhinged. "
            "v0.1 sends a static value; v0.2+ will modulate it from "
            "neurotransmitter state per-pass per-driver."
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
            "Cap on how long a single driver response can be (roughly "
            "0.75 words per token). 2048 is enough for ~1,500 words of "
            "output. Bump it for long-form work; keep it low for "
            "snappier chat. v0.1 placeholder for the future NT system."
        ),
        category="generation",
        valueType=ConfigValueType.integer,
        default=2048,
        minimum=1,
    ),
    ConfigField(
        key="requestTimeoutSeconds",
        label="Driver request timeout",
        description=(
            "How long the gateway waits on one driver's response "
            "before giving up. Counts from the start of the HTTP "
            "request to the driver, NOT the end of the bicameral loop. "
            "Bump this if your slowest model needs more time per pass."
        ),
        category="bicameral",
        valueType=ConfigValueType.duration,
        default=180,
        minimum=5,
        maximum=900,
        requiresRestart=True,
    ),
    ConfigField(
        key="voiceDriver",
        label="Voice driver",
        description=(
            "Which driver performs the voice pass — the post-"
            "deliberation LLM call that converts internal "
            "deliberation into Eugene's user-facing reply. The voice "
            "pass exists because the bicameral deliberation loop "
            "produces hemispheres talking to each other (inner-dialog "
            "register) rather than to the user. The voice pass "
            "rewrites the deliberated content into a clean reply "
            "addressed to the actual person. Leave blank to use the "
            "first driver in the topology."
        ),
        category="bicameral",
        valueType=ConfigValueType.string,
        default=None,
        required=False,
    ),
    ConfigField(
        key="voiceTemperature",
        label="Voice pass temperature",
        description=(
            "Sampling temperature for the voice pass. Lower values "
            "produce more consistent / less surprising user-facing "
            "replies; higher values are more expressive. Leave blank "
            "to use the gateway's default temperature."
        ),
        category="bicameral",
        valueType=ConfigValueType.number,
        default=None,
        minimum=0.0,
        maximum=2.0,
        required=False,
    ),
]

_FIELDS_BY_KEY: dict[str, ConfigField] = {f.key: f for f in FIELDS}


def _migrate_drivers(value: Any) -> Any:
    """Upgrade legacy driver-slot shapes to the `backends` name list.

    Shape history per slot:
      * pre-v0.2.1:  `{name, url}`            (single backend URL)
      * v0.2.1 item1: `{name, urls: [...]}`   (URL priority list)
      * v0.2.1 item2: `{name, backends: [...]}` (topology-name priority list)

    Rewrite the two legacy shapes to `backends`, preserving the values.
    Legacy values are URLs, not topology names — they survive here as
    URL-shaped backend strings and are resolved directly (with a
    warn-to-re-save) by `build_clients`, so existing installs keep
    working without a manual edit. `backends` is only synthesized when
    absent, so a config that already uses it is left untouched (and an
    entry carrying both keeps `backends`, dropping the stale `urls`/
    `url`). Non-list / malformed values pass through for the validator.
    """
    if not isinstance(value, list):
        return value
    migrated: list[Any] = []
    for entry in value:
        if isinstance(entry, dict) and "backends" not in entry and "urls" in entry:
            upgraded = {k: v for k, v in entry.items() if k not in ("urls", "url")}
            upgraded["backends"] = (
                list(entry["urls"]) if isinstance(entry["urls"], list) else entry["urls"]
            )
            migrated.append(upgraded)
        elif isinstance(entry, dict) and "backends" not in entry and "url" in entry:
            upgraded = {k: v for k, v in entry.items() if k != "url"}
            upgraded["backends"] = [entry["url"]]
            migrated.append(upgraded)
        else:
            migrated.append(entry)
    return migrated


def as_schema(*, driver_names: list[str] | None = None) -> ConfigSchema:
    """Emit the gateway config schema.

    `driver_names` — when supplied — turns the `voiceDriver` field into
    a strict dropdown (dynamic enum) of the currently-configured driver
    slot names, with a leading "(first driver)" default. Voice driver
    choice is empirically the persona lever (see project memory
    `voice-driver-choice-is-the-persona-lever-not-hemisphere-choice`), so
    a one-click strict dropdown beats the old free-text-with-suggestions.

    Note this only changes the SCHEMA (what the UI renders). The static
    `FIELDS` entry stays `valueType=string`, so PATCH validation remains
    lenient — an operator can save a name that isn't a current slot, and
    `_resolve_voice_driver` falls back to the first driver. The strict
    dropdown is presentation; the lenient string is the contract.
    """
    if driver_names:
        fields = [
            _with_voice_driver_enum(f, driver_names) if f.key == "voiceDriver" else f
            for f in FIELDS
        ]
    else:
        fields = list(FIELDS)
    return ConfigSchema(
        component="gateway",
        fields=fields,
        categories=CATEGORY_LABELS,
    )


def _with_voice_driver_enum(field: ConfigField, driver_names: list[str]) -> ConfigField:
    """Return a copy of `voiceDriver` rendered as a strict enum dropdown
    of the configured driver slot names, with a leading empty option
    labelled "(first driver)" for the unset/default case. The UI's enum
    renderer keys off `valueType==enum` + `enumValues`, so this needs no
    UI change."""
    return field.model_copy(
        update={
            "valueType": ConfigValueType.enum,
            "enumValues": ["", *driver_names],
            "enumLabels": ["(first driver)", *driver_names],
        }
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

    if vt == ConfigValueType.driver_list:
        if not isinstance(value, list):
            return f"expected list of {{name, url}} entries, got {type(value).__name__}"
        if not value:
            return "drivers list must not be empty"
        seen_names: set[str] = set()
        for i, entry in enumerate(value):
            if not isinstance(entry, dict):
                return f"entry {i}: expected object, got {type(entry).__name__}"
            name = entry.get("name")
            backends = entry.get("backends")
            if not isinstance(name, str) or not name.strip():
                return f"entry {i}: `name` must be a non-empty string"
            if not isinstance(backends, list) or not backends:
                return f"entry {i}: `backends` must be a non-empty list of topology names"
            for j, backend in enumerate(backends):
                if not isinstance(backend, str) or not backend.strip():
                    return f"entry {i}: `backends[{j}]` must be a non-empty string"
            if name in seen_names:
                return f"entry {i}: duplicate driver name {name!r}"
            seen_names.add(name)
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
                        merged[k] = _migrate_drivers(v) if k == "drivers" else v
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

                # Accept legacy single-`url` driver entries from older
                # clients / scripts and upgrade them to the `urls` list
                # before validation — same normalization the disk-load
                # path applies (see _migrate_drivers).
                if key == "drivers":
                    new_value = _migrate_drivers(new_value)

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
