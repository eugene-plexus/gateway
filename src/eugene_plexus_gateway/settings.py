"""Startup-time settings, sourced from environment variables.

Distinct from the runtime *config* (see `config.py`), which is editable via
`PATCH /v1/config`. These settings only control bootstrap.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EUGENE_PLEXUS_GATEWAY_",
        env_file=None,
        case_sensitive=False,
    )

    config_file: Path = Path("config.yaml")
    """Where the runtime config is persisted. PATCH /v1/config writes here."""

    metrics_file: Path = Path("metrics.sqlite3")
    """Where retained request metrics live (M8).

    Relative, so it lands in the gateway's cwd — which the agent sets to
    the install directory, beside `gateway.yaml` and outside every source
    checkout. That is the property the install rework established and
    that a developer's own install depends on."""

    bind_host: str = "127.0.0.1"
    """Network interface to bind. Override to 0.0.0.0 for tailnet exposure."""

    safe_mode: bool = False
    """If true, skip loading the persisted config file at startup and run on
    built-in defaults (no drivers configured, default memory URL). Set by
    the agent via EUGENE_PLEXUS_GATEWAY_SAFE_MODE=1 when a previous boot
    failed. PATCH /v1/config still writes to `config_file` normally so
    the operator's repair survives the next non-safe-mode boot. Per the
    safe-mode contract in specs/openapi/gateway.yaml."""

    auth_verify_key: str | None = None
    """Base64 public Ed25519 PEM from the agent. Exclusive with auth_signing_key."""

    auth_signing_key: str | None = None
    """Legacy base64 32-byte HS256 key; used only until install rotation."""

    service_token: str | None = None
    """Long-lived service JWT for outbound calls to peer components
    (EUGENE_PLEXUS_GATEWAY_SERVICE_TOKEN). Required when `auth_signing_key`
    is set; the gateway presents this on every outbound httpx call."""

    master_key: str | None = None
    """Base64-encoded 32-byte secretbox key for at-rest decryption
    (EUGENE_PLEXUS_GATEWAY_MASTER_KEY). Populated only after the operator
    has logged in at the agent; absent during the configured-but-
    locked window. Reserved for Phase 6; Phase 3 does not consume it."""

    agent_url: str = "http://127.0.0.1:8079"
    """Agent endpoint the routing table is built from.

    The agent is the source of truth for topology, so the gateway
    reads its inference-driver entries rather than keeping backend URLs
    in its own config — a URL duplicated into two components is the trap
    this avoids. Override with EUGENE_PLEXUS_GATEWAY_AGENT_URL on
    networked deployments where the agent isn't on loopback."""


def load_settings() -> Settings:
    return Settings()
