"""Entrypoint: `python -m eugene_plexus_gateway`."""

from __future__ import annotations

import contextlib
import logging
import os
from types import FrameType
from typing import Any

import uvicorn

from .app import close_queue, create_app
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
    config = uvicorn.Config(
        app,
        host=settings.bind_host,
        port=port,
        log_level=log_level.lower(),
        # **Trust no forwarding header from anyone.** uvicorn's default
        # is `"127.0.0.1"`, and every request this component receives
        # arrives over loopback -- from the gateway, from an agent, or
        # from the browser's proxy. So the default let any caller set
        # `scope["client"]`, in the access log and in whatever reads it
        # next. Review §6.1 #1, roadmap R1.2; the agent's `peer.py`
        # carries the argument.
        forwarded_allow_ips=[],
    )
    _Server(config, app).run()


class _Server(uvicorn.Server):
    """uvicorn, refusing the fair queue's waiters as a stop begins (row 11), and
    the exit a finished drain asks for (row 13).

    uvicorn waits for every request in flight before the app's lifespan ends,
    and a request waiting in line is one: closed only there, a stop with
    waiters would hang until each waited its `queueWaitSeconds`. Closed when
    the signal arrives, each is refused `shutting_down` at once.
    """

    def __init__(self, config: uvicorn.Config, app: Any) -> None:
        super().__init__(config)
        self._app = app
        # Row 13: a drain that has finished asks the server to exit, so the
        # lifespan closes normally and the process exits 0.
        app.state.request_exit = self.request_exit

    def request_exit(self) -> None:
        self.should_exit = True

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        loop = getattr(self._app.state, "loop", None)
        if loop is not None:
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(close_queue, self._app)
        super().handle_exit(sig, frame)


if __name__ == "__main__":
    main()
