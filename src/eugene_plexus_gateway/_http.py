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

**Between machines, every internal client speaks TLS to a pinned key**
(LT1, `specs/docs/design/tls-between-machines.md`). The agent writes the
install's machine table, from a list the root signs, to the file named by
`EUGENE_PLEXUS_MACHINE_PINS_FILE`. An internal client then:

* carries a request for another machine's agent or control root, named by
  its plain `http://` address, to that machine's TLS port instead
  (`https://<machine>:8083/agent/...`), so no call site changes;
* checks the key every `https://` machine address presents against the
  pins the table lists for it, after the handshake and before the first
  request byte, with no certificate chain and no name check;
* when the install requires encryption, refuses plain HTTP to a machine
  the table names but cannot carry.

With no table, behaviour is exactly what it was.

Deliberately duplicated in each component rather than shared: components
share schemas, not code (see CLAUDE.md, "no shared `core` library").
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import ipaddress
import json
import logging
import os
import ssl
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpcore
import httpx

log = logging.getLogger(__name__)

# **Reentrant, and that is not decoration.** `shared_internal_client`
# holds this lock while it builds a client, and building one calls
# `ssl_context()`, which takes the same lock. With a plain `Lock` the
# very first engine readiness probe of a fresh process deadlocks the
# event loop for good -- and only on the first one, because every later
# call finds `_CONTEXT` already set and never reaches the acquire. That
# is a hang that a full test run hides and a cold start reproduces.
_LOCK = threading.RLock()
_CONTEXT: ssl.SSLContext | None = None

# Retire idle connections before the common two- and five-second server
# deadlines. This reduces close/reuse races without replaying a POST whose
# outcome is unknown. Explicit caller limits still take precedence.
_KEEPALIVE_SECONDS = 1.5


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
    kwargs.setdefault("limits", httpx.Limits(keepalive_expiry=_KEEPALIVE_SECONDS))
    _machine_transport(kwargs)
    return httpx.AsyncClient(**kwargs)


def egress_client(**kwargs: Any) -> httpx.AsyncClient:
    """A client for the public internet. Honours the user's proxy
    environment, because that is how those users reach it at all."""
    kwargs.setdefault("verify", ssl_context())
    kwargs.setdefault("trust_env", True)
    kwargs.setdefault("limits", httpx.Limits(keepalive_expiry=_KEEPALIVE_SECONDS))
    return httpx.AsyncClient(**kwargs)


def client_for(url: str, **kwargs: Any) -> httpx.AsyncClient:
    """Whichever of the two `url` calls for."""
    kwargs.setdefault("trust_env", not is_internal(url))
    kwargs.setdefault("verify", ssl_context())
    kwargs.setdefault("limits", httpx.Limits(keepalive_expiry=_KEEPALIVE_SECONDS))
    if not kwargs["trust_env"]:
        _machine_transport(kwargs)
    return httpx.AsyncClient(**kwargs)


def sync_client_for(url: str, **kwargs: Any) -> httpx.Client:
    """`client_for`, for the few synchronous call sites."""
    kwargs.setdefault("trust_env", not is_internal(url))
    kwargs.setdefault("verify", ssl_context())
    kwargs.setdefault("limits", httpx.Limits(keepalive_expiry=_KEEPALIVE_SECONDS))
    if not kwargs["trust_env"]:
        _machine_transport(kwargs, sync=True)
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


# --------------------------------------------------------------------------- #
# Between machines: the machine table and the pinned transport (LT1)
# --------------------------------------------------------------------------- #
#
# **The table** is written by this machine's agent from the list the root
# signs with its identity key (`GET /v1/trust/machines`), which every node
# pinned at enrolment. So the chain is: the root's identity key, the signed
# list, a node's pin, the TLS session. It holds three things:
#
# * `machines`: `host:port` of each machine port -> the pins (base64url
#   SHA-256 of the SubjectPublicKeyInfo) its key may present;
# * `routes`: the plain origin another machine's agent or control root is
#   known by (`http://10.0.0.7:8079`) -> the TLS address that carries it
#   (`https://10.0.0.7:8083/agent`), or null for a machine that has no key
#   on record yet (it runs an older version);
# * `required`: the install refuses plain HTTP between machines.
#
# **Python's `ssl` has no verification callback**, so the pin is checked
# right after the handshake and before httpcore writes the first request
# byte, the way site-host's root link has checked a Job Site's root since
# J7a. A connection whose key is not listed is closed with nothing sent.

MACHINE_PINS_ENV = "EUGENE_PLEXUS_MACHINE_PINS_FILE"
_TABLE_RECHECK_SECONDS = 1.0
_REFUSAL_LOG_SECONDS = 300.0


class MachineRefused(httpx.ConnectError):
    """Plain HTTP to another machine, refused because the install requires
    encryption and that machine has no key on record. Nothing was sent."""


class PinRefused(httpcore.ConnectError):
    """A machine presented a TLS key its pins do not list. Nothing was sent."""

    def __init__(self, machine: str, pin: str, name: str | None) -> None:
        who = name or machine
        super().__init__(
            f"{who}'s machine key doesn't match the one it registered "
            f"(it presented {pin} at {machine}). Nothing was sent to it."
        )
        self.machine = machine
        self.pin = pin


def spki_pin(der: bytes) -> str:
    """Base64url SHA-256 of a DER certificate's SubjectPublicKeyInfo."""
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    spki = (
        x509.load_der_x509_certificate(der)
        .public_key()
        .public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    return base64.urlsafe_b64encode(hashlib.sha256(spki).digest()).rstrip(b"=").decode("ascii")


def _hostport(host: str, port: int) -> str:
    host = host.strip().strip("[]").lower()
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def _origin(url: httpx.URL) -> str:
    port = url.port or (443 if url.scheme == "https" else 80)
    return f"{url.scheme}://{_hostport(url.host, port)}"


@dataclass(frozen=True)
class MachineTable:
    """What this process may say to which machine, and how."""

    machines: dict[str, frozenset[str]] = field(default_factory=dict)
    routes: dict[str, str | None] = field(default_factory=dict)
    required: bool = False
    names: dict[str, str] = field(default_factory=dict)
    local: frozenset[str] = frozenset()
    """Origins that are this machine under a LAN address (its own agent),
    which a request may reach in plain HTTP whatever `required` says."""

    @property
    def empty(self) -> bool:
        return not self.machines and not self.routes

    def pins(self, url: httpx.URL) -> frozenset[str] | None:
        if url.scheme != "https":
            return None
        return self.machines.get(_hostport(url.host, url.port or 443))

    def carry(self, url: httpx.URL) -> httpx.URL | None:
        """The TLS address that carries a plain request for another machine,
        None to send it as it is. Raises `MachineRefused` when it must not go
        plain and cannot go encrypted."""
        if url.scheme != "http":
            return None
        origin = _origin(url)
        if origin not in self.routes:
            return None
        base = self.routes[origin]
        if base is None:
            if self.required:
                who = self.names.get(origin) or origin
                raise MachineRefused(
                    f"Encryption between machines is required and {who} has no machine key "
                    f"on record (it runs an older version; update it). Nothing was sent to "
                    f"{origin}."
                )
            return None
        return httpx.URL(base.rstrip("/") + url.raw_path.decode("ascii"))


EMPTY_TABLE = MachineTable()

#: Where a transport reads its table: `machine_table`, unless a caller with a
#: table of its own (an enrolment that pins the root from its join token)
#: hands one in.
TableSource = Callable[[], MachineTable]


def parse_machine_table(document: Any) -> MachineTable:
    """A table from its JSON form. Entries that do not hold together are
    dropped: a route must name an `https` machine the table pins."""
    if not isinstance(document, dict):
        raise ValueError("a machine table is a JSON object")
    machines: dict[str, frozenset[str]] = {}
    for key, pins in (document.get("machines") or {}).items():
        if isinstance(key, str) and isinstance(pins, list):
            listed = frozenset(p for p in pins if isinstance(p, str) and p)
            if listed:
                parts = urlsplit("//" + key)
                if parts.hostname and parts.port:
                    machines[_hostport(parts.hostname, parts.port)] = listed
    routes: dict[str, str | None] = {}
    for key, target in (document.get("routes") or {}).items():
        if not isinstance(key, str):
            continue
        try:
            origin = _origin(httpx.URL(key))
        except (httpx.InvalidURL, TypeError):
            continue
        if not origin.startswith("http://"):
            continue
        carried: str | None = None
        if isinstance(target, str):
            try:
                parsed = httpx.URL(target)
            except httpx.InvalidURL:
                parsed = None
            if (
                parsed is not None
                and parsed.scheme == "https"
                and _hostport(parsed.host, parsed.port or 443) in machines
            ):
                carried = target.rstrip("/")
        routes[origin] = carried
    names = {
        str(k): str(v)
        for k, v in (document.get("names") or {}).items()
        if isinstance(k, str) and isinstance(v, str)
    }
    local: set[str] = set()
    for key in document.get("local") or []:
        try:
            local.add(_origin(httpx.URL(str(key))))
        except (httpx.InvalidURL, TypeError):
            continue
    return MachineTable(machines, routes, bool(document.get("required")), names, frozenset(local))


_TABLE_OVERRIDE: MachineTable | None = None
_TABLE_FILE: tuple[str, int, int] | None = None
_TABLE_FROM_FILE: MachineTable = EMPTY_TABLE
_TABLE_CHECKED = 0.0


def set_machine_table(table: MachineTable | None) -> None:
    """Install this process's table directly (the agent, which writes the
    file for its children; tests). None returns to the file."""
    global _TABLE_OVERRIDE
    _TABLE_OVERRIDE = table


def machine_table() -> MachineTable:
    """The table in effect: one installed in-process, else the file, re-read
    when it changes (checked at most once a second). A file that becomes
    unreadable keeps the last table read from it; a file removed is none."""
    global _TABLE_FILE, _TABLE_FROM_FILE, _TABLE_CHECKED
    if _TABLE_OVERRIDE is not None:
        return _TABLE_OVERRIDE
    path = os.environ.get(MACHINE_PINS_ENV)
    if not path:
        return EMPTY_TABLE
    now = time.perf_counter()
    if (
        _TABLE_FILE is not None
        and _TABLE_FILE[0] == path
        and now - _TABLE_CHECKED < (_TABLE_RECHECK_SECONDS)
    ):
        return _TABLE_FROM_FILE
    with _LOCK:
        _TABLE_CHECKED = now
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            _TABLE_FILE, _TABLE_FROM_FILE = (path, 0, 0), EMPTY_TABLE
            return EMPTY_TABLE
        except OSError as exc:
            log.warning("could not read this machine's pins at %s: %s", path, exc)
            return _TABLE_FROM_FILE
        seen = (path, stat.st_mtime_ns, stat.st_size)
        if seen == _TABLE_FILE:
            return _TABLE_FROM_FILE
        try:
            with open(path, encoding="utf-8") as handle:
                table = parse_machine_table(json.load(handle))
        except (OSError, ValueError) as exc:
            log.warning("this machine's pins at %s are unreadable, keeping the last: %s", path, exc)
            return _TABLE_FROM_FILE
        _TABLE_FILE, _TABLE_FROM_FILE = seen, table
        return table


def machine_route(url: str) -> str | None:
    """The TLS address a plain URL to another machine is carried on, or None."""
    try:
        carried = machine_table().carry(httpx.URL(url))
    except (httpx.InvalidURL, MachineRefused):
        return None
    return str(carried) if carried is not None else None


def machine_pinned(url: str) -> bool:
    """Whether `url` is a machine port this process holds pins for."""
    try:
        return machine_table().pins(httpx.URL(url)) is not None
    except httpx.InvalidURL:
        return False


def machine_required() -> bool:
    return machine_table().required


def refused_pin(exc: BaseException) -> str | None:
    """The key a connection was refused for, when that is why it failed."""
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, PinRefused):
            return cause.pin
        cause = cause.__cause__ or cause.__context__
    return None


_REFUSALS_LOGGED: dict[tuple[str, str], float] = {}


def _check_pin(machine: str, ssl_object: Any, table: MachineTable) -> None:
    pins = table.machines.get(machine)
    # Positional: the sync backend hands over `_ssl._SSLSocket`, whose
    # argument is named `der`, where `ssl.SSLObject`'s is `binary_form`.
    der = ssl_object.getpeercert(True) if ssl_object is not None else None
    pin = spki_pin(der) if der else "none"
    if pins and pin in pins:
        return
    name = table.names.get(machine)
    now = time.perf_counter()
    if now - _REFUSALS_LOGGED.get((machine, pin), -_REFUSAL_LOG_SECONDS) >= _REFUSAL_LOG_SECONDS:
        _REFUSALS_LOGGED[(machine, pin)] = now
        log.warning(
            "SECURITY: %s presented a TLS key (%s) that is not the one it registered; "
            "refused, and nothing was sent to it",
            name or machine,
            pin,
        )
    raise PinRefused(machine, pin, name)


_MACHINE_CONTEXT: ssl.SSLContext | None = None


def _machine_context() -> ssl.SSLContext:
    """TLS to a machine port: no chain and no name, because the pin is the
    identity and addresses move with DHCP. No CA store is loaded."""
    global _MACHINE_CONTEXT
    if _MACHINE_CONTEXT is None:
        with _LOCK:
            if _MACHINE_CONTEXT is None:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                context.minimum_version = ssl.TLSVersion.TLSv1_2
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                _MACHINE_CONTEXT = context
    return _MACHINE_CONTEXT


class _PinnedStream(httpcore.AsyncNetworkStream):
    def __init__(
        self, inner: httpcore.AsyncNetworkStream, machine: str, table: TableSource
    ) -> None:
        self._inner, self._machine, self._table = inner, machine, table

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:  # noqa: ASYNC109
        return await self._inner.read(max_bytes, timeout)

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:  # noqa: ASYNC109
        await self._inner.write(buffer, timeout)

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 - httpcore's interface
    ) -> httpcore.AsyncNetworkStream:
        stream = await self._inner.start_tls(ssl_context, server_hostname, timeout)
        try:
            _check_pin(self._machine, stream.get_extra_info("ssl_object"), self._table())
        except BaseException:
            await stream.aclose()
            raise
        return _PinnedStream(stream, self._machine, self._table)

    def get_extra_info(self, info: str) -> Any:
        return self._inner.get_extra_info(info)


class _PinnedBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, table: TableSource) -> None:
        self._inner = httpcore.AnyIOBackend()
        self._table = table

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,  # noqa: ASYNC109 - httpcore's interface
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        stream = await self._inner.connect_tcp(host, port, timeout, local_address, socket_options)
        return _PinnedStream(stream, _hostport(host, port), self._table)

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,  # noqa: ASYNC109 - httpcore's interface
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:  # pragma: no cover - never a machine port
        raise httpcore.ConnectError("a machine port is not reached over a Unix socket")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class _SyncPinnedStream(httpcore.NetworkStream):
    def __init__(self, inner: httpcore.NetworkStream, machine: str, table: TableSource) -> None:
        self._inner, self._machine, self._table = inner, machine, table

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self._inner.read(max_bytes, timeout)

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._inner.write(buffer, timeout)

    def close(self) -> None:
        self._inner.close()

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        stream = self._inner.start_tls(ssl_context, server_hostname, timeout)
        try:
            _check_pin(self._machine, stream.get_extra_info("ssl_object"), self._table())
        except BaseException:
            stream.close()
            raise
        return _SyncPinnedStream(stream, self._machine, self._table)

    def get_extra_info(self, info: str) -> Any:
        return self._inner.get_extra_info(info)


class _SyncPinnedBackend(httpcore.NetworkBackend):
    def __init__(self, table: TableSource) -> None:
        self._inner = httpcore.SyncBackend()
        self._table = table

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        stream = self._inner.connect_tcp(host, port, timeout, local_address, socket_options)
        return _SyncPinnedStream(stream, _hostport(host, port), self._table)

    def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: Any = None
    ) -> httpcore.NetworkStream:  # pragma: no cover - never a machine port
        raise httpcore.ConnectError("a machine port is not reached over a Unix socket")

    def sleep(self, seconds: float) -> None:
        self._inner.sleep(seconds)


def _pool_options(limits: httpx.Limits) -> dict[str, Any]:
    return {
        "ssl_context": _machine_context(),
        "max_connections": limits.max_connections,
        "max_keepalive_connections": limits.max_keepalive_connections,
        "keepalive_expiry": limits.keepalive_expiry,
    }


def _route(request: httpx.Request, table: MachineTable) -> MachineTable:
    """Carry a plain request for another machine to its TLS address, in place."""
    if table.empty and not table.required:
        return table
    try:
        carried = table.carry(request.url)
    except MachineRefused as exc:
        exc.request = request
        raise
    if carried is not None:
        request.url = carried
        request.headers["Host"] = carried.netloc.decode("ascii")
    elif table.required and request.url.scheme == "http" and _eugene_token(request.headers):
        # An address the table does not name (a node just enrolled, another
        # spelling of a machine's name): under required, a token of this
        # install never crosses to another machine in plain HTTP. A request
        # without one (a model server on the LAN) is not this rule's business.
        if not _is_loopback(request.url.host) and _origin(request.url) not in table.local:
            raise MachineRefused(
                "Encryption between machines is required, and this request would carry "
                f"an Eugene Plexus token to {request.url.host} in plain HTTP; it was not sent.",
                request=request,
            )
    return table


def _is_loopback(host: str) -> bool:
    host = host.strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _eugene_token(headers: httpx.Headers) -> bool:
    """Whether a request carries one of this install's own tokens: a JWS whose
    header says `typ: ep-...` (sessions, service, client and administration
    tokens all do), as a bearer or an `x-api-key`."""
    values = [headers.get("authorization", ""), headers.get("x-api-key", "")]
    for value in values:
        token = value.split(" ", 1)[-1].strip()
        head = token.split(".", 1)[0]
        if not head.startswith("eyJ"):
            continue
        try:
            padded = head + "=" * (-len(head) % 4)
            typ = json.loads(base64.urlsafe_b64decode(padded)).get("typ", "")
        except (ValueError, TypeError, AttributeError):
            continue
        if isinstance(typ, str) and typ.lower().startswith("ep-"):
            return True
    return False


class MachineTransport(httpx.AsyncBaseTransport):
    """An internal client's transport: plain as before, or pinned TLS to a
    machine port. The pinned pool is httpx's own transport with httpcore's
    pool built on a pin-checking network backend (`_pool` is the one private
    attribute this touches; a test fails when httpx moves it)."""

    def __init__(
        self,
        *,
        verify: Any,
        limits: httpx.Limits,
        table: TableSource | None = None,
        **options: Any,
    ) -> None:
        self._plain = httpx.AsyncHTTPTransport(
            verify=verify, limits=limits, trust_env=False, **options
        )
        self._limits = limits
        self._table: TableSource = table or machine_table
        self._pinned: httpx.AsyncHTTPTransport | None = None

    @property
    def _pool(self) -> httpcore.AsyncConnectionPool:
        """The plain pool, where `httpx.AsyncHTTPTransport` keeps its own."""
        return self._plain._pool

    def _pinned_transport(self) -> httpx.AsyncHTTPTransport:
        if self._pinned is None:
            transport = httpx.AsyncHTTPTransport(verify=_machine_context(), trust_env=False)
            transport._pool = httpcore.AsyncConnectionPool(
                **_pool_options(self._limits), network_backend=_PinnedBackend(self._table)
            )
            self._pinned = transport
        return self._pinned

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        table = _route(request, self._table())
        if table.pins(request.url) is not None:
            return await self._pinned_transport().handle_async_request(request)
        return await self._plain.handle_async_request(request)

    async def aclose(self) -> None:
        await self._plain.aclose()
        if self._pinned is not None:
            await self._pinned.aclose()


class SyncMachineTransport(httpx.BaseTransport):
    """`MachineTransport`, for the synchronous clients."""

    def __init__(
        self,
        *,
        verify: Any,
        limits: httpx.Limits,
        table: TableSource | None = None,
        **options: Any,
    ) -> None:
        self._plain = httpx.HTTPTransport(verify=verify, limits=limits, trust_env=False, **options)
        self._limits = limits
        self._table: TableSource = table or machine_table
        self._pinned: httpx.HTTPTransport | None = None

    @property
    def _pool(self) -> httpcore.ConnectionPool:
        """The plain pool, where `httpx.HTTPTransport` keeps its own."""
        return self._plain._pool

    def _pinned_transport(self) -> httpx.HTTPTransport:
        if self._pinned is None:
            transport = httpx.HTTPTransport(verify=_machine_context(), trust_env=False)
            transport._pool = httpcore.ConnectionPool(
                **_pool_options(self._limits), network_backend=_SyncPinnedBackend(self._table)
            )
            self._pinned = transport
        return self._pinned

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        table = _route(request, self._table())
        if table.pins(request.url) is not None:
            return self._pinned_transport().handle_request(request)
        return self._plain.handle_request(request)

    def close(self) -> None:
        self._plain.close()
        if self._pinned is not None:
            self._pinned.close()


def _machine_transport(kwargs: dict[str, Any], *, sync: bool = False) -> None:
    """Give an internal client the machine transport, unless the caller
    brought a transport of its own (a test's mock, a Unix socket)."""
    # `transport=None` is a test seam left unset, not a transport: callers
    # pass it through from `app.state`, and taking it as one sent their
    # requests to other machines in plain HTTP (found by the LT1 capture).
    if kwargs.get("transport") is not None or kwargs.get("mounts") or kwargs.get("proxy"):
        return
    kwargs.pop("transport", None)
    options = {k: kwargs[k] for k in ("cert", "http1", "http2") if k in kwargs}
    cls = SyncMachineTransport if sync else MachineTransport
    limits = kwargs.get("limits") or httpx.Limits(max_connections=100, max_keepalive_connections=20)
    kwargs["transport"] = cls(verify=kwargs["verify"], limits=limits, **options)


__all__ = [
    "EMPTY_TABLE",
    "MACHINE_PINS_ENV",
    "MachineRefused",
    "MachineTable",
    "MachineTransport",
    "PinRefused",
    "SyncMachineTransport",
    "aclose_shared",
    "client_for",
    "egress_client",
    "internal_client",
    "is_internal",
    "machine_pinned",
    "machine_required",
    "machine_route",
    "machine_table",
    "parse_machine_table",
    "refused_pin",
    "reset_shared",
    "set_machine_table",
    "set_shared_client",
    "shared_internal_client",
    "spki_pin",
    "ssl_context",
    "sync_client_for",
]
