"""Startup-time settings, sourced from environment variables.

Distinct from the runtime *config* (see `config.py`), which is editable via
`PATCH /v1/config`. These settings only control bootstrap.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EUGENE_PLEXUS_GATEWAY_",
        env_file=None,
        case_sensitive=False,
    )

    config_file: Path = Path("config.yaml")
    """Where the runtime config is persisted. PATCH /v1/config writes here."""

    client_key_refresh_seconds: float = Field(default=15.0, gt=0, le=60)
    client_key_max_age_seconds: float = Field(default=60.0, gt=0, le=86400)
    client_key_timeout_seconds: float = Field(default=4.0, gt=0, le=30)
    client_key_retry_seconds: float = Field(default=1.0, gt=0, le=15)

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

    trust_bundle_file: str | None = None
    """The trust bundle the agent keeps beside `node.yaml`
    (EUGENE_PLEXUS_GATEWAY_TRUST_BUNDLE_FILE), reloaded when it changes."""

    trust_authority: str | None = None
    """The public key that bundle must be signed by (base64url Ed25519)."""

    auth_recipient: str | None = None
    """This machine as a token's audience names it: `node:<name>`."""

    service_token: str | None = None
    """This gateway's own token, addressed to this machine alone
    (EUGENE_PLEXUS_GATEWAY_SERVICE_TOKEN). It reaches this machine's agent,
    drivers and library; for any other machine the gateway asks its agent
    for a fifteen-minute token addressed there (`outbound.py`)."""

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
