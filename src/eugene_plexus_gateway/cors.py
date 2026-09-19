"""CORS for the front door, and only the front door.

The gateway's OpenAI-compatible paths exist so that an unmodified
OpenAI client can point its base URL here and work. Until the playground
diagnostic (install-paths §9 step 8) that sentence was true of every
client except one kind: a **browser**. A page on any origin other than
the gateway's own -- the UI the agent serves, Open WebUI in a tab, a web
app built on the OpenAI SDK -- sends a preflight before a request that
carries `Authorization`, and this gateway answered it `405 Method Not
Allowed` (measured live, 2026-09-13). `llama-server` answers CORS by
default and Ollama has `OLLAMA_ORIGINS`; we had nothing, and the UI never
noticed because it reaches the gateway through the agent's same-origin
proxy.

## Three properties, each load-bearing

**Only the three OpenAI paths.** `/v1/models`, `/v1/chat/completions`,
`/v1/embeddings`. `/v1/config`, `/v1/admin/*` and `/v1/metrics` stay
same-origin: a browser client needs the front door and nothing else, and
widening the operator surface would make every operator route callable
from any page that holds a token, for no client that asked.

**Pure ASGI, not `BaseHTTPMiddleware`.** The front door streams, and M10
recorded what a buffering layer does to a stream: every frame still
arrives, the framing is correct, every "is it streaming" check passes,
and the tokens land all at once at the end. This class touches the
`http.response.start` message to add headers and forwards every body
message the moment it is sent. `tests/test_cors.py` has the sabotage
check -- a fake app that will not emit its second body chunk until the
test has seen the first one forwarded.

**Live configuration.** `corsEnabled` and `corsAllowedOrigins` are read
from the config store on every request, so a `PATCH /v1/config` takes
effect on the next one with no restart -- which is what
`requiresRestart: false` promises, and what `controlUrl` once promised
and did not deliver until it was read the same way.

## The default

Any origin, no credentials. CORS exists to stop a page from spending
credentials the browser attaches on its own -- cookies, HTTP auth, client
certificates. The front door has none: it authenticates by a bearer the
page must hold and send explicitly, so a page without the token gets the
same 401 a `curl` without one gets. `Access-Control-Allow-Origin: *` and
no `Allow-Credentials` is the shape `llama-server` and OpenAI's own API
ship. The operator who wants only their UI's origin admitted lists it;
the one who wants no browser at all turns it off. A refused preflight is
a `403` with a `Problem` naming the origin and the config key, because a
browser shows `TypeError: Failed to fetch` for every CORS failure alike
and the `curl` that reproduces it deserves a sentence.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from starlette.datastructures import Headers
from starlette.responses import JSONResponse, Response

from ._generated.models import Problem

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

# The front door, and nothing else. Exact paths: the front door has no
# sub-resources, and a prefix match would quietly widen to whatever is
# added under /v1 next.
#
# `/v1/messages` joined at R4 for a reason that is on the wire rather
# than by analogy: Anthropic's own browser path announces itself with
# `anthropic-dangerous-direct-browser-access`, which Claude Code sends
# on every request, so a browser client of that door is a shape its
# authors expect.
FRONT_DOOR_PATHS: frozenset[str] = frozenset(
    {"/v1/models", "/v1/chat/completions", "/v1/embeddings", "/v1/messages"}
)

# What the three paths accept between them. A preflight for PATCH on
# /v1/models is answered with this list and the browser refuses it
# itself, which is the correct outcome and needs no special case.
_ALLOW_METHODS = "GET, POST, OPTIONS"

# Ten minutes. Long enough that a chat session does not preflight every
# turn, short enough that narrowing `corsAllowedOrigins` is felt the
# same afternoon rather than tomorrow.
_MAX_AGE = "600"

# What a preflight is asked for when it does not say: the two headers
# every OpenAI client sends, plus the three an Anthropic one does.
#
# **This list is a fallback and not the mechanism**, which is worth
# stating because R4's first draft claimed the opposite. The reply
# ECHOES `access-control-request-headers` whenever a preflight sends
# one, and a real browser always does -- so `x-api-key` is already
# allowed by the echo and a default list without it would change
# nothing for any browser client. It is listed anyway because the
# fallback should describe the same door the echo does; a fallback that
# contradicts the surface it stands in for is a trap for whoever reads
# it next.
_DEFAULT_ALLOW_HEADERS = "authorization, content-type, x-api-key, anthropic-version, anthropic-beta"

CONFIG_ENABLED_KEY = "corsEnabled"
CONFIG_ORIGINS_KEY = "corsAllowedOrigins"


@dataclass(frozen=True)
class CorsPolicy:
    """What the config says right now: on or off, and which origins."""

    enabled: bool = True
    origins: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_store(cls, store: Any) -> CorsPolicy:
        """Read the two fields off a `ConfigStore`, tolerating its absence.

        No store (a test app that skipped the lifespan) reads as the
        default policy rather than as "off": the default is the
        behaviour an operator gets with no config file at all, and the
        middleware should not answer differently because the store was
        not there to ask.
        """
        if store is None:
            return cls()
        enabled = store.get(CONFIG_ENABLED_KEY)
        raw = store.get(CONFIG_ORIGINS_KEY)
        origins = tuple(_normalise(o) for o in raw if isinstance(o, str)) if raw else ()
        return cls(enabled=True if enabled is None else bool(enabled), origins=origins)

    def allow_origin(self, origin: str) -> str | None:
        """The `Access-Control-Allow-Origin` value for this origin, or None.

        `*` when the list is empty -- any origin -- and the origin itself
        when it is listed, which is when `Vary: Origin` must accompany
        it so a shared cache does not hand one origin's answer to
        another. Matching is on the normalised form (lower-cased, no
        trailing slash): browsers send origins lower-cased already, and
        an operator typing `http://Host:8079/` should not be refused
        for the capital or the slash.
        """
        if not self.enabled:
            return None
        if not self.origins:
            return "*"
        return origin if _normalise(origin) in self.origins else None

    def refusal(self, origin: str) -> str:
        """The sentence a refused preflight carries."""
        if not self.enabled:
            return (
                f"This gateway does not answer browsers: `{CONFIG_ENABLED_KEY}` is false "
                "in its config. Set it true (Config -> Gateway -> Browser clients) to let "
                "a page on another origin call the OpenAI-compatible paths."
            )
        listed = ", ".join(self.origins)
        return (
            f"Origin {origin!r} is not in this gateway's `{CONFIG_ORIGINS_KEY}` "
            f"({listed}). Add it there, or clear the list to admit any origin."
        )


def _normalise(origin: str) -> str:
    return origin.strip().rstrip("/").lower()


def _is_preflight(method: str, headers: Headers) -> bool:
    return method == "OPTIONS" and "access-control-request-method" in headers


class FrontDoorCors:
    """ASGI middleware: CORS on `FRONT_DOOR_PATHS`, configured live."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") not in FRONT_DOOR_PATHS:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        origin = headers.get("origin")
        if origin is None:
            # Not a cross-origin browser request. Every non-browser client
            # and every same-origin page lands here, untouched.
            await self.app(scope, receive, send)
            return

        store = getattr(scope["app"].state, "config_store", None) if "app" in scope else None
        policy = CorsPolicy.from_store(store)
        allow = policy.allow_origin(origin)

        if _is_preflight(scope["method"], headers):
            response = _preflight_response(policy, origin, allow, headers)
            await response(scope, receive, send)
            return

        if allow is None:
            # The request still runs -- the server cannot stop a browser
            # from sending it, only from reading the answer -- and the
            # absence of the header is what makes the browser refuse.
            await self.app(scope, receive, send)
            return

        async def send_with_cors(message: Message) -> None:
            if message["type"] == "http.response.start":
                raw = list(message.get("headers", []))
                raw.append((b"access-control-allow-origin", allow.encode("latin-1")))
                if allow != "*":
                    raw.append((b"vary", b"origin"))
                message["headers"] = raw
            # Body messages pass straight through, one call per chunk.
            # Collecting them here is how a streaming path stops
            # streaming while every frame still arrives.
            await send(message)

        await self.app(scope, receive, send_with_cors)


def _preflight_response(
    policy: CorsPolicy, origin: str, allow: str | None, request_headers: Headers
) -> Response:
    if allow is None:
        problem = Problem(
            type="https://github.com/eugene-plexus/gateway#cors-refused",
            title="Origin not allowed",
            status=403,
            detail=policy.refusal(origin),
            component="gateway",
        )
        return JSONResponse(problem.model_dump(exclude_none=True), status_code=403)

    requested = request_headers.get("access-control-request-headers")
    response_headers = {
        "access-control-allow-origin": allow,
        "access-control-allow-methods": _ALLOW_METHODS,
        # Echo what was asked for rather than a fixed list: an SDK that
        # adds `x-stainless-*` or `openai-organization` would otherwise
        # be refused by the browser for a header this gateway ignores.
        "access-control-allow-headers": requested or _DEFAULT_ALLOW_HEADERS,
        "access-control-max-age": _MAX_AGE,
    }
    if allow != "*":
        response_headers["vary"] = "origin"
    return Response(status_code=204, headers=response_headers)
