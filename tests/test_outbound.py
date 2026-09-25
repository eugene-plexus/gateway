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
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

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
