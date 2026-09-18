"""One SSL context per process, and the rule about `trust_env`.

Constructing `httpx.AsyncClient()` without `verify=` builds a fresh
`ssl.SSLContext` and parses certifi's PEM bundle. Measured in this
repo's own venv on Windows/CPython 3.12 -- the Python both installers
provision -- that is **104-136 ms of synchronous CPU**, and because it
is synchronous it runs *on the event loop*, stalling every concurrent
request in the process. Passing a context built once takes **0.03 ms**.

So: never construct a client without `verify=ssl_context()`. There is
one context per process, it is safe to share across clients and
threads, and it is the whole of the fix for the ~116 ms of control-plane
overhead this project carried as unexplained from M8 until 2026-09-17.

`trust_env` is the second half, and it is per site rather than blanket.
With `trust_env=True` (httpx's default) a user's `HTTP_PROXY` /
`HTTPS_PROXY` is applied to *every* request, including the ones this
install makes to itself. On Windows the logon task inherits the user
environment, so a corporate proxy silently swallows every health probe
and every gateway->driver hop, and the install sits at `starting` while
actually serving. Clients that dial loopback, a private LAN address or
another node of this install therefore pass `trust_env=False`; clients
that dial the public internet -- the model hub, the GitHub release
feed, a cloud inference provider -- must keep it, because a proxy is
how those users reach the internet at all.

`internal_client()` and `egress_client()` say which is which at the
call site. `client_for()` decides from the URL, for the one case where
the same code path can dial either (a driver fronting `127.0.0.1:8081`
or `api.openai.com`).

Deliberately duplicated in each component rather than shared: components
share schemas, not code (see CLAUDE.md, "no shared `core` library").
"""

from __future__ import annotations

import contextlib
import ipaddress
import ssl
import threading
from typing import Any
from urllib.parse import urlsplit

import httpx

# **Reentrant, and that is not decoration.** `shared_internal_client`
# holds this lock while it builds a client, and building one calls
# `ssl_context()`, which takes the same lock. With a plain `Lock` the
# very first engine readiness probe of a fresh process deadlocks the
# event loop for good -- and only on the first one, because every later
# call finds `_CONTEXT` already set and never reaches the acquire. That
# is a hang that a full test run hides and a cold start reproduces.
_LOCK = threading.RLock()
_CONTEXT: ssl.SSLContext | None = None


def ssl_context() -> ssl.SSLContext:
    """The process-wide SSL context, built at most once.

    Double-checked under a lock because the first two requests of a
    freshly-started process routinely arrive together, and building it
    twice would pay the cost this module exists to remove.
    """
    global _CONTEXT
    if _CONTEXT is None:
        with _LOCK:
            if _CONTEXT is None:
                _CONTEXT = httpx.create_ssl_context()
    return _CONTEXT


def is_internal(url: str) -> bool:
    """Whether `url` addresses this machine or this install's own LAN.

    Loopback, link-local, private ranges and a bare hostname with no
    dots (a LAN/tailnet short name) are internal. Anything else --
    including a public IP and any dotted hostname -- is egress, because
    being wrong in that direction only costs a proxy hop, while being
    wrong the other way makes a reachable backend unreachable.
    """
    host = urlsplit(url).hostname
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        # A name. Treat a dotless one as a LAN/tailnet short name;
        # anything with a dot could be public, so do not assume.
        return "." not in host
    return bool(addr.is_loopback or addr.is_private or addr.is_link_local)


def internal_client(**kwargs: Any) -> httpx.AsyncClient:
    """A client for loopback, another node of this install, or a
    supervised engine. Never routed through the user's proxy."""
    kwargs.setdefault("verify", ssl_context())
    kwargs.setdefault("trust_env", False)
    return httpx.AsyncClient(**kwargs)


def egress_client(**kwargs: Any) -> httpx.AsyncClient:
    """A client for the public internet. Honours the user's proxy
    environment, because that is how those users reach it at all."""
    kwargs.setdefault("verify", ssl_context())
    kwargs.setdefault("trust_env", True)
    return httpx.AsyncClient(**kwargs)


def client_for(url: str, **kwargs: Any) -> httpx.AsyncClient:
    """Whichever of the two `url` calls for."""
    kwargs.setdefault("trust_env", not is_internal(url))
    kwargs.setdefault("verify", ssl_context())
    return httpx.AsyncClient(**kwargs)


def sync_client_for(url: str, **kwargs: Any) -> httpx.Client:
    """`client_for`, for the few synchronous call sites."""
    kwargs.setdefault("trust_env", not is_internal(url))
    kwargs.setdefault("verify", ssl_context())
    return httpx.Client(**kwargs)


# --- Process-wide clients, for call sites with no object to hang one on ---
#
# A module-level helper (an engine adapter's readiness probe, say) has no
# instance to own a client, and building one per call is the defect this
# module exists to remove. `shared_internal_client` keeps one per key for
# the life of the process; the app's lifespan calls `aclose_shared()` on
# the way out so the pools are released rather than left to the garbage
# collector.

_SHARED: dict[str, httpx.AsyncClient] = {}


def shared_internal_client(key: str, **kwargs: Any) -> httpx.AsyncClient:
    """The process-wide internal client filed under `key`, built once.

    Rebuilt if a previous one was closed, so a lifespan that tears down
    and a test that starts a second app in the same process both work.
    """
    client = _SHARED.get(key)
    if client is not None and not client.is_closed:
        return client
    with _LOCK:
        client = _SHARED.get(key)
        if client is None or client.is_closed:
            client = internal_client(**kwargs)
            _SHARED[key] = client
    return client


async def aclose_shared() -> None:
    """Close every process-wide client. Idempotent."""
    clients = list(_SHARED.values())
    _SHARED.clear()
    for client in clients:
        # A shutdown never fails over cleanup.
        with contextlib.suppress(Exception):
            await client.aclose()


def set_shared_client(key: str, client: httpx.AsyncClient | None) -> None:
    """Install or drop the process-wide client filed under `key`.

    This is the injection seam. A test that needs an
    `httpx.MockTransport` installs a client here instead of patching
    `httpx.AsyncClient.__init__`, which is what the pre-2026-09-18
    harnesses did and which stops working the moment a client is built
    once rather than per call -- worse, a patched constructor then leaks
    one test's transport into every later test in the process. Dropping
    does NOT close: the caller owns anything it installed.
    """
    if client is None:
        _SHARED.pop(key, None)
    else:
        _SHARED[key] = client


def reset_shared() -> None:
    """Forget every process-wide client without closing it. Tests only."""
    _SHARED.clear()


__all__ = [
    "aclose_shared",
    "client_for",
    "egress_client",
    "internal_client",
    "is_internal",
    "reset_shared",
    "set_shared_client",
    "shared_internal_client",
    "ssl_context",
    "sync_client_for",
]
