"""Entrypoint: `python -m eugene_plexus_gateway`."""

from __future__ import annotations

import logging
import os

import uvicorn

from .app import create_app
from .config import ConfigStore
from .settings import load_settings

# Default bind port for standalone launch (no agent). The agent
# always overrides via EUGENE_PLEXUS_GATEWAY_BIND_PORT.
_DEFAULT_PORT = 8080


class _DropHealthzFilter(logging.Filter):
    """Suppress uvicorn.access lines for /healthz.

    The agent probes /healthz on every supervised child constantly,
    and at INFO each probe produces a line. That floods the log file
    and buries the routing DEBUG trace we actually want to read. Drop
    them — the agent already tracks component health on its side.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return "/healthz" not in msg


def main() -> None:
    settings = load_settings()

    bootstrap_store = ConfigStore(settings.config_file)
    if not settings.safe_mode:
        bootstrap_store.load()

    env_port = os.environ.get("EUGENE_PLEXUS_GATEWAY_BIND_PORT")
    port = int(env_port) if env_port else _DEFAULT_PORT
    log_level = str(bootstrap_store.get("logLevel") or "INFO").upper()

    # Wire the application logger to the same level as uvicorn so
    # `log.debug(...)` calls in our code actually emit when logLevel
    # is DEBUG. uvicorn only configures its own loggers — the root
    # logger has no handler by default, so without this our DEBUG
    # output would be silently dropped. `force=True` is needed
    # because uvicorn has already touched logging by the time we
    # invoke its CLI; without it basicConfig is a no-op.
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("uvicorn.access").addFilter(_DropHealthzFilter())

    app = create_app(settings)
    uvicorn.run(app, host=settings.bind_host, port=port, log_level=log_level.lower())


if __name__ == "__main__":
    main()
