"""Which token goes to which machine (per-node token keys, D8).

The gateway holds no key. It presents the token it was spawned with to
its own machine and nothing else, and asks its own agent for a
fifteen-minute token addressed to any other machine. These tests stand
two agents and a control root behind one handler (`test_multi_agent`'s
fakes) and read the `Authorization` header every request carried.

The property the whole design rests on is asserted directly: **the
spawn-time token never leaves this machine**, not even when the agent
will not hand out a remote one.
"""

from __future__ import annotations

import json
import socket
import threading
from dataclasses import replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import pytest

from eugene_plexus_gateway import admission
from eugene_plexus_gateway._generated.driver_models import (
    FunctionDefinition,
    GenerateRequest,
    Message,
    Role,
    Tool,
)
from eugene_plexus_gateway.driver_client import BoundClient, HttpDriverClient
from eugene_plexus_gateway.lifecycle import AgentLifecycleClient
from eugene_plexus_gateway.outbound import Outbound
from eugene_plexus_gateway.routing import RoutingTable

# `route_http` is a fixture, found by name.
from .test_multi_agent import AGENT_A, CONTROL, MODEL, TwoAgents, route_http  # noqa: F401

LOCAL = "local-token-of-node-a"
THIS_MACHINE = {"agent-a", "127.0.0.1"}


class Recorded(TwoAgents):
    """`TwoAgents`, plus node A's agent minting tokens, and a log of headers."""

    def __init__(self) -> None:
        super().__init__()
        self.enrolled_to = CONTROL
        self.seen: list[tuple[str, str, str | None]] = []
        self.minted: list[str] = []
        self.refuse_minting = False
        self.control_requires_token = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization")
        self.seen.append((str(request.url.host), request.url.path, auth))
        if request.url.path == "/v1/auth/service-token":
            assert request.url.host == "agent-a", "only this node's own agent mints"
            assert auth == f"Bearer {LOCAL}"
            if self.refuse_minting:
                return httpx.Response(
                    403,
                    json={
                        "detail": {
                            "title": "Not granted",
                            "detail": "this node may not send a 'gateway' token to control: "
                            "it holds no gateway grant",
                        }
                    },
                )
            audience = json.loads(request.content)["audience"]
            self.minted.append(audience)
            return httpx.Response(
                200,
                json={
                    "token": f"minted-for-{audience}",
                    "expiresAt": datetime.fromtimestamp(4102444800, UTC).isoformat(),
                },
            )
        if request.url.host == "control" and self.control_requires_token and not auth:
            return httpx.Response(401, json={"detail": {"title": "Missing token"}})
        return super().handler(request)

    def tokens_sent_to(self, host: str) -> set[str | None]:
        return {
            auth
            for seen_host, path, auth in self.seen
            if seen_host == host and path != "/v1/auth/service-token"
        }


@pytest.fixture
def recorded(route_http: Any) -> Recorded:  # noqa: F811
    fake = Recorded()
    route_http(fake.handler)
    return fake


def _outbound() -> Outbound:
    return Outbound(recipient="node:node-a", local_token=LOCAL, agent_url=AGENT_A)


async def _table(outbound: Outbound) -> RoutingTable:
    table = RoutingTable(
        agent_url=AGENT_A, control_url=CONTROL, outbound=outbound, refresh_seconds=3600
    )
    await table.refresh()
    return table


async def test_each_machine_gets_a_token_addressed_to_it(recorded: Recorded) -> None:
    outbound = _outbound()
    table = await _table(outbound)
    try:
        assert recorded.tokens_sent_to("agent-a") == {f"Bearer {LOCAL}"}
        assert recorded.tokens_sent_to("127.0.0.1") == {f"Bearer {LOCAL}"}
        assert recorded.tokens_sent_to("control") == {"Bearer minted-for-control"}
        assert recorded.tokens_sent_to("agent-b") == {"Bearer minted-for-node:node-b"}
    finally:
        await table.aclose()
        await outbound.aclose()


async def test_the_spawn_time_token_never_leaves_this_machine(recorded: Recorded) -> None:
    """Not when the agent mints, and not when it refuses: a refusal sends
    nothing, and the far side answers 401 rather than holding a token it
    could replay against this machine for a year."""
    for refuse in (False, True):
        recorded.refuse_minting = refuse
        recorded.seen.clear()
        outbound = _outbound()
        table = await _table(outbound)
        try:
            for host, _, auth in recorded.seen:
                if host not in THIS_MACHINE:
                    assert auth != f"Bearer {LOCAL}", (host, refuse)
            if refuse:
                assert recorded.tokens_sent_to("agent-b") == {None}
                assert recorded.tokens_sent_to("control") == {None}
        finally:
            await table.aclose()
            await outbound.aclose()


async def test_a_refused_mint_is_named_where_the_operator_looks(recorded: Recorded) -> None:
    """The far side sees no token and says only "Missing token"; the
    reason -- this node has no gateway grant -- is on this machine, so the
    routing view says so rather than sending someone to check the root."""
    recorded.refuse_minting = True
    recorded.control_requires_token = True
    outbound = _outbound()
    table = await _table(outbound)
    try:
        error = table.control_root().error or ""
        assert "401 Missing token" in error
        assert "would not give the gateway a token" in error
        assert "holds no gateway grant" in error
    finally:
        await table.aclose()
        await outbound.aclose()


async def test_a_token_is_minted_once_per_machine_not_per_request(recorded: Recorded) -> None:
    outbound = _outbound()
    table = await _table(outbound)
    try:
        await table.refresh()
        await table.refresh()
        assert sorted(recorded.minted) == ["control", "node:node-b"]
    finally:
        await table.aclose()
        await outbound.aclose()


async def test_a_lifecycle_action_carries_the_far_machines_token(recorded: Recorded) -> None:
    outbound = _outbound()
    table = await _table(outbound)
    client = AgentLifecycleClient(outbound)
    try:
        facts = table._snapshot.runtimes[("node-b", "qwen")]
        recorded.seen.clear()
        assert await client.stop(
            table.agent_url_for(facts), facts.name, reason="idle", node=facts.node
        )
        sent = [(h, a) for h, p, a in recorded.seen if p.endswith("/stop")]
        assert sent == [("agent-b", "Bearer minted-for-node:node-b")]
    finally:
        await client.aclose()
        await table.aclose()
        await outbound.aclose()


async def test_a_driver_on_another_machine_is_called_with_that_machines_token(
    recorded: Recorded,
) -> None:
    outbound = _outbound()
    table = await _table(outbound)
    try:
        recorded.seen.clear()
        for backend in table.backends_for(MODEL):
            await backend.client.info()
        by_host = {h: a for h, p, a in recorded.seen if p == "/v1/info"}
        assert by_host == {
            "127.0.0.1": f"Bearer {LOCAL}",
            "agent-b": "Bearer minted-for-node:node-b",
        }
    finally:
        await table.aclose()
        await outbound.aclose()


# --------------------------------------------------------------------------- #
# A 401 from a driver the gateway sent no token
# --------------------------------------------------------------------------- #


class StrictFarDriver(Recorded):
    """`Recorded`, and node B's driver answering its listing as a remote driver does once
    it is strict: a request with no `Authorization` is its 401 (and, with
    `rejects_every_token`, so is one that has a token it does not accept). Open until
    `strict` is set, so the routing table has read it before it turns."""

    strict = False
    rejects_every_token = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        if (
            self.strict
            and request.url.host == "agent-b"
            and request.url.port == 8091
            and request.url.path == "/v1/info"
            and (self.rejects_every_token or not request.headers.get("authorization"))
        ):
            return httpx.Response(401, json={"detail": {"title": "Missing token", "status": 401}})
        return super().handler(request)


@pytest.fixture
def strict(route_http: Any) -> StrictFarDriver:  # noqa: F811
    fake = StrictFarDriver()
    route_http(fake.handler)
    return fake


def _asked_with_tools() -> GenerateRequest:
    tool = Tool(type="function", function=FunctionDefinition(name="get_weather"))
    return GenerateRequest(
        messages=[Message(role=Role.user, content="hi")], tools=[tool], callerSettings=["tools"]
    )


async def _only_node_b(table: RoutingTable) -> Any:
    """`MODEL` resolved to node B's replica alone, so that no other replica can serve and
    what the refusal says is what is under test."""
    resolution = table.resolve(MODEL)
    tiers = [
        replace(tier, backends=[b for b in tier.backends if b.node == "node-b"])
        for tier in resolution.tiers
    ]
    assert sum(len(t.backends) for t in tiers) == 1
    return replace(resolution, tiers=tiers)


async def test_a_401_to_a_driver_the_gateway_sent_no_token_is_the_agents_refusal_and_waiting_can_clear_it(
    strict: StrictFarDriver,
) -> None:
    """Through the real `Outbound`, `RoutingTable`, `HttpDriverClient` and `permitted`: this
    node's agent would not mint a token for node B (it holds no gateway grant), so the
    re-read of node B's driver went out with none and the driver answered 401. That is
    not the driver refusing the gateway's credential (there was none), and the agent is
    asked again within seconds: a 503 to try again, saying why no token was sent, as the
    routing table's own read says it."""
    strict.refuse_minting = True
    outbound = _outbound()
    table = await _table(outbound)
    try:
        strict.strict = True
        with pytest.raises(admission.AdmissionFailure) as raised:
            await admission.permitted(await _only_node_b(table), requirements=_asked_with_tools())
        failure = raised.value
        assert (failure.status, failure.retry, failure.kind) == (503, "5", None)
        message = failure.message
        assert (
            "driver 'qwen-driver' on node 'node-b' (http://agent-b:8091): answered 401" in message
        )
        assert "Missing token" in message
        assert (
            "this node's agent would not give the gateway a token for it: 403: this node may "
            "not send a 'gateway' token to control: it holds no gateway grant"
        ) in message
        assert "refused the gateway's credential" not in message
        assert "try again in a few seconds" in message
        assert "service:gateway" not in message, "nothing to re-check on the driver"
    finally:
        await table.aclose()
        await outbound.aclose()


class _RemoteDriver(BaseHTTPRequestHandler):
    """A remote driver on a real socket: its listing is a 401 to a request that carries no
    `Authorization`, and what each request carried is kept."""

    seen: ClassVar[list[str | None]] = []

    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:
        type(self).seen.append(self.headers.get("Authorization"))
        raw = json.dumps({"detail": {"title": "Not authenticated", "status": 401}}).encode()
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


async def test_a_401_over_a_real_socket_to_a_request_sent_no_token_is_the_agents_refusal() -> None:
    """The review's own evidence as a test: a real `Outbound` whose agent is a closed port (so
    it cannot mint a token for the far machine), a real client, and a remote driver that
    answers 401 to a request with no `Authorization`. The driver received none, and the
    cause says why, in the agent's own words, as one that passes."""

    class Handler(_RemoteDriver):
        seen: ClassVar[list[str | None]] = []

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed = probe.getsockname()[1]
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    outbound = Outbound(
        recipient="node:here", local_token=LOCAL, agent_url=f"http://127.0.0.1:{closed}"
    )
    client = HttpDriverClient(
        name="far-driver", base_url=url, auth=outbound.auth("node:far"), node="far"
    )
    backend = SimpleNamespace(name="far-driver", node="far", url=url, client=client)
    try:
        with pytest.raises(httpx.HTTPStatusError) as raised:
            await client.info(model="m")
        cause = admission.unread_because(backend, raised.value)
        assert Handler.seen == [None], "the gateway's token never left it"
        reason = outbound.refusal("node:far")
        assert reason, "the agent was asked, and said why it would not"
        assert cause.transient and cause.remedy is None and cause.kind == "upstream_error"
        assert cause.text.startswith(f"driver 'far-driver' on node 'far' ({url}): answered 401")
        assert "this node's agent would not give the gateway a token for it: " in cause.text
        assert " ".join(reason.split())[:60] in cause.text
        assert "refused the gateway's credential" not in cause.text
    finally:
        await client.aclose()
        await outbound.aclose()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


async def test_a_401_to_a_token_that_was_sent_is_still_the_drivers_refusal(
    strict: StrictFarDriver,
) -> None:
    """The control: the agent minted the token, the gateway sent it and the driver turned
    it away. That is the driver refusing the gateway's credential, and waiting does not
    mend it."""
    strict.rejects_every_token = True
    outbound = _outbound()
    table = await _table(outbound)
    try:
        strict.strict = True
        with pytest.raises(admission.AdmissionFailure) as raised:
            await admission.permitted(await _only_node_b(table), requirements=_asked_with_tools())
        failure = raised.value
        assert (failure.status, failure.retry, failure.kind) == (502, None, "upstream_auth_error")
        assert "refused the gateway's credential (answered 401: Missing token)" in failure.message
        assert "would not give the gateway a token" not in failure.message
        assert "try again" not in failure.message
    finally:
        await table.aclose()
        await outbound.aclose()


# --------------------------------------------------------------------------- #
# The cache, on its own clock
# --------------------------------------------------------------------------- #


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def _agent(expires_in: float, clock: Clock, calls: list[str], *, fail: list[bool]) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["audience"])
        if fail[0]:
            raise httpx.ConnectError("refused", request=request)
        expires = datetime.fromtimestamp(clock.now + expires_in, UTC).isoformat()
        return httpx.Response(200, json={"token": f"t{len(calls)}", "expiresAt": expires})

    return httpx.MockTransport(handler)


async def test_a_token_is_replaced_at_half_its_life() -> None:
    clock, calls, fail = Clock(), [], [False]
    outbound = Outbound(
        recipient="node:a",
        local_token=LOCAL,
        agent_url="http://agent",
        transport=_agent(900, clock, calls, fail=fail),
        clock=clock,
    )
    assert await outbound.token("node:b") == "t1"
    clock.now += 449
    assert await outbound.token("node:b") == "t1"
    clock.now += 2
    assert await outbound.token("node:b") == "t2"
    await outbound.aclose()


async def test_an_agent_that_does_not_answer_keeps_an_unexpired_token_and_is_not_hammered() -> None:
    clock, calls, fail = Clock(), [], [False]
    outbound = Outbound(
        recipient="node:a",
        local_token=LOCAL,
        agent_url="http://agent",
        transport=_agent(900, clock, calls, fail=fail),
        clock=clock,
    )
    assert await outbound.token("node:b") == "t1"
    fail[0] = True
    clock.now += 500
    # Past half-life: asks, fails, keeps the token that is still good.
    assert await outbound.token("node:b") == "t1"
    asked = len(calls)
    # Inside the retry window: does not ask again.
    assert await outbound.token("node:b") == "t1"
    assert len(calls) == asked
    # Past expiry, still failing: nothing, and never the local token.
    clock.now += 1000
    assert await outbound.token("node:b") is None
    await outbound.aclose()


async def test_a_refusal_is_forgotten_once_the_agent_agrees() -> None:
    """A stale "no gateway grant" beside a working token would send an
    operator to fix something already fixed."""
    clock, answers = Clock(), [403, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        status = answers.pop(0)
        if status == 403:
            return httpx.Response(403, json={"detail": {"title": "Not granted"}})
        expires = datetime.fromtimestamp(clock.now + 900, UTC).isoformat()
        return httpx.Response(200, json={"token": "t", "expiresAt": expires})

    outbound = Outbound(
        recipient="node:a",
        local_token=LOCAL,
        agent_url="http://agent",
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    assert await outbound.token("control") is None
    assert outbound.refusal("control") == "403: Not granted"
    clock.now += 10
    assert await outbound.token("control") == "t"
    assert outbound.refusal("control") is None
    await outbound.aclose()


async def test_the_auth_a_client_carries_says_why_it_carried_nothing() -> None:
    """The far side's bare 401 says nothing of a grant it never saw, and admission's
    re-read of a driver's listing needs to tell that from a credential it refused: the
    auth a driver's client carries knows whether this node's agent would give a token."""
    clock, answers = Clock(), [403, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        if answers.pop(0) == 403:
            return httpx.Response(403, json={"detail": {"title": "Not granted"}})
        expires = datetime.fromtimestamp(clock.now + 900, UTC).isoformat()
        return httpx.Response(200, json={"token": "t", "expiresAt": expires})

    outbound = Outbound(
        recipient="node:a",
        local_token=LOCAL,
        agent_url="http://agent",
        transport=httpx.MockTransport(handler),
        clock=clock,
    )
    client = HttpDriverClient(
        name="far", base_url="http://far:8091", auth=outbound.auth("node:b"), node="b"
    )
    bound = BoundClient(client, model="m", public_model="m")
    try:
        assert client.credential_refusal() is None, "nothing was asked of the agent yet"
        assert await outbound.token("node:b") is None
        assert client.credential_refusal() == "403: Not granted"
        assert bound.credential_refusal() == "403: Not granted", "through the bound client too"
        clock.now += 10
        assert await outbound.token("node:b") == "t"
        assert client.credential_refusal() is None, "forgotten once the agent agrees"
        assert bound.credential_refusal() is None
    finally:
        await client.aclose()
        await outbound.aclose()


async def test_a_client_with_no_auth_has_no_refusal_to_give() -> None:
    client = HttpDriverClient(name="here", base_url="http://127.0.0.1:8091")
    try:
        assert client.credential_refusal() is None
        assert BoundClient(client, model="m", public_model="m").credential_refusal() is None
    finally:
        await client.aclose()


async def test_this_machine_and_no_machine_need_no_fetch() -> None:
    calls: list[str] = []
    outbound = Outbound(
        recipient="node:a",
        local_token=LOCAL,
        agent_url="http://agent",
        transport=_agent(900, Clock(), calls, fail=[True]),
    )
    assert await outbound.token("node:a") == LOCAL
    assert await outbound.token(None) is None
    assert calls == []
    await outbound.aclose()


async def test_unauthenticated_sends_nothing_anywhere() -> None:
    outbound = Outbound(recipient=None, local_token=None, agent_url="http://agent")
    assert not outbound.enabled
    assert await outbound.headers(outbound.for_node("b")) == {}
    await outbound.aclose()
