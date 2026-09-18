"""R1.2: this component trusts no forwarding header.

Roadmap `docs/design/release-roadmap.md` §2.2, finding review §6.1 #1.

The finding's sharp end is the login limiter, which lives on the agent
and the control root. This component has no login — and the rule is
still all five entrypoints, for two reasons that are not tidiness.

uvicorn's `ProxyHeadersMiddleware` is on by default and trusts
`X-Forwarded-For` from `127.0.0.1`; every request this component ever
receives arrives over loopback, from the gateway, from an agent, or
from the browser's proxy. So `scope["client"]` here is
caller-controlled today, in the access log and in anything that reads
it tomorrow — and *tomorrow* is the case worth pinning, because the next
per-source decision written in this repo would inherit a peer that any
caller can set, with nothing to tell its author.

Driving `main()` rather than reading the source: the subject is what
the entrypoint passes, and a text check goes green the moment somebody
writes the kwarg into a comment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from eugene_plexus_gateway import __main__ as entry


def entrypoint_kwargs(tmp_path: Path) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    original = entry.uvicorn.run
    env = dict(os.environ)
    try:
        entry.uvicorn.run = lambda app, **kwargs: captured.update(kwargs)  # type: ignore[assignment]
        os.environ["EUGENE_PLEXUS_GATEWAY_CONFIG_FILE"] = str(tmp_path / "config.yaml")
        os.environ["EUGENE_PLEXUS_GATEWAY_BIND_PORT"] = "18999"
        entry.main()
    finally:
        entry.uvicorn.run = original  # type: ignore[assignment]
        os.environ.clear()
        os.environ.update(env)
    return captured


def test_the_entrypoint_trusts_no_forwarding_header(tmp_path: Path) -> None:
    assert entrypoint_kwargs(tmp_path)["forwarded_allow_ips"] == []
