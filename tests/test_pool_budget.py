"""A turn waits for room in its replica's shared context (CB3).

llama-server's automatic slots share one KV pool, and prompts in flight past
it make the engine refuse the next request or cut every stream it is
decoding. Measured on eight 1B replicas, a budget like this took the failed
turns from 44 to 0 (`specs/docs/acceptance/cache-aware-balancing-measurement.md`
§5); on an 8B at 64k it was the difference between 46 and 166 answered of
204 (§8).
"""

from __future__ import annotations

import asyncio

import pytest

from eugene_plexus_gateway._generated.driver_models import GenerateRequest, Usage
from eugene_plexus_gateway.budget import (
    CHARS_PER_TOKEN,
    ConversationSizes,
    PoolLedger,
    estimate,
    request_chars,
)

from .conftest import FakeDriverClient, make_routing_table, runtime_facts

LOCAL = "llama-8b"
POOL = 10_000
A = (None, "llama-a")
B = (None, "llama-b")


def _table(*, pools=(POOL, POOL), wait: float = 120.0, strategy: str = "conversation"):
    a = FakeDriverClient(
        name="llama-a-driver", base_url="http://a", model_id=LOCAL, runtime="llama-a"
    )
    b = FakeDriverClient(
        name="llama-b-driver", base_url="http://b", model_id=LOCAL, runtime="llama-b"
    )
    for fake in (a, b):
        fake.usage = Usage(promptTokens=40, completionTokens=4, totalTokens=44)
    runtimes = [
        runtime_facts("llama-a", parallel_slots=4, context_pool=pools[0]),
        runtime_facts("llama-b", parallel_slots=4, context_pool=pools[1]),
    ]
    table = make_routing_table(a, b, runtimes=runtimes, strategy=strategy, room_wait_seconds=wait)
    return table, a, b


def _turn(text: str = "fix the bug", max_tokens: int = 1000) -> GenerateRequest:
    return GenerateRequest(messages=[{"role": "user", "content": text}], maxTokens=max_tokens)


def _send(table, key: str, request: GenerateRequest):  # type: ignore[no-untyped-def]
    client = table.pick(table.resolve(LOCAL), affinity=key)
    return client, asyncio.ensure_future(client.generate(request))


# --- the estimate --------------------------------------------------------------


def test_a_turn_is_its_last_prompt_plus_its_new_text_plus_its_answer():
    sizes = ConversationSizes()
    # The engine counted 30,000 where characters alone say 20,000: what it
    # counted is the truth, and only the new text is estimated.
    sizes.put("m", "k", prompt_tokens=30_000, chars=70_000)
    seen = sizes.get("m", "k")
    prompt, total = estimate(70_000 + 3_500, 32_000, seen)
    assert prompt == 30_000 + 1_000
    assert total == 31_000 + 32_000


def test_a_new_conversation_is_estimated_from_its_size():
    prompt, total = estimate(7_000, None, None)
    assert prompt == int(7_000 / CHARS_PER_TOKEN) and total == prompt


def test_the_size_counts_tools_too():
    bare = request_chars(_turn())
    tooled = request_chars(
        GenerateRequest(
            messages=[{"role": "user", "content": "fix the bug"}],
            tools=[{"type": "function", "function": {"name": "read", "parameters": {}}}],
        )
    )
    assert tooled > bare


def test_the_ledger_keeps_a_tenth_spare():
    ledger = PoolLedger()
    ledger.take(A, 8_000)
    assert ledger.fits(A, POOL, tokens=1_000, prompt=500)
    assert not ledger.fits(A, POOL, tokens=1_001, prompt=500)


def test_alone_a_turn_always_fits():
    assert PoolLedger().fits(A, POOL, tokens=POOL * 3, prompt=POOL // 2)


def test_a_prompt_bigger_than_the_pool_is_sent_at_once():
    """The engine refuses it outright, so it takes nothing from anyone."""
    ledger = PoolLedger()
    ledger.take(A, 9_000)
    assert ledger.fits(A, POOL, tokens=POOL + 1, prompt=POOL + 1)


def test_an_answer_budget_alone_does_not_make_a_prompt_too_big():
    """Claude Code sends `max_tokens: 32000`: a 6,000-token prompt with that
    answer budget is more than a 10,000 pool and still must wait its turn,
    or it overflows the turns already there."""
    ledger = PoolLedger()
    ledger.take(A, 5_000)
    assert not ledger.fits(A, POOL, tokens=6_000 + 32_000, prompt=6_000)


# --- through the routing table --------------------------------------------------


async def test_a_turn_waits_on_its_own_replica_for_room():
    table, a, b = _table()
    first, task = _send(table, "k:s", _turn())
    await task
    home = first.served_by
    held = A if home == "llama-a-driver" else B
    table._ledger.take(held, 8_500)  # its replica's pool is nearly full

    client, task = _send(table, "k:s", _turn("fix the bug and the test"))
    await asyncio.sleep(0.2)
    assert not task.done(), "it went without room"
    calls = (len(a.calls), len(b.calls))
    table._ledger.give(held, 8_500)
    await asyncio.wait_for(task, 2)
    assert client.served_by == home, "it moved off the replica holding its history"
    assert (len(a.calls), len(b.calls)) != calls


async def test_a_new_conversation_takes_a_replica_with_room():
    table, _a, _b = _table()
    table._ledger.take(A, 9_000)
    table._ledger.take(B, 0)
    client = table.pick(table.resolve(LOCAL), affinity="k:new")
    # Whatever the balancer's order, the replica with room answers.
    await asyncio.wait_for(client.generate(_turn()), 1)
    assert client.served_by == "llama-b-driver"


async def test_the_wait_is_bounded_then_the_usual_order_runs():
    table, _a, _b = _table(wait=0.2)
    table._ledger.take(A, 9_500)
    table._ledger.take(B, 9_500)
    client = table.pick(table.resolve(LOCAL), affinity="k:new")
    started = asyncio.get_running_loop().time()
    await asyncio.wait_for(client.generate(_turn()), 2)
    assert asyncio.get_running_loop().time() - started >= 0.18
    assert client.served_by is not None


async def test_the_room_is_given_back_on_every_path():
    table, a, b = _table()
    await table.pick(table.resolve(LOCAL), affinity="k:1").generate(_turn())
    assert table._ledger.in_flight(A) == table._ledger.in_flight(B) == 0

    # A failed attempt that cascades holds only the backend it moved to.
    from .test_failover_safety import pool_full

    a.generate_error = b.generate_error = None
    a.generate_error = pool_full()
    client = table.pick(table.resolve(LOCAL), affinity="k:2")
    await client.generate(_turn())
    assert table._ledger.in_flight(A) == table._ledger.in_flight(B) == 0

    # An abandoned stream.
    a.generate_error = None
    client = table.pick(table.resolve(LOCAL), affinity="k:3")
    stream = client.stream(_turn())
    await anext(stream)
    assert table._ledger.in_flight(A) + table._ledger.in_flight(B) > 0
    await stream.aclose()
    assert table._ledger.in_flight(A) == table._ledger.in_flight(B) == 0


async def test_while_a_turn_runs_its_room_is_held():
    table, a, b = _table()
    gate = asyncio.Event()

    async def hold() -> None:
        await gate.wait()

    a.generate_hook = b.generate_hook = hold
    _client, task = _send(table, "k:s", _turn(max_tokens=2_000))
    await asyncio.sleep(0.05)
    assert table._ledger.in_flight(A) + table._ledger.in_flight(B) >= 2_000
    gate.set()
    await task


async def test_its_next_turn_is_counted_from_what_the_engine_said():
    table, _a, _b = _table()
    await table.pick(table.resolve(LOCAL), affinity="k:s").generate(_turn())
    seen = table._sizes.get(LOCAL, "k:s")
    assert seen is not None and seen.prompt_tokens == 40


async def test_a_runtime_whose_slots_do_not_share_is_not_budgeted():
    table, _a, _b = _table(pools=(None, None))
    client = table.pick(table.resolve(LOCAL), affinity="k:s")
    assert client.budget is None


@pytest.mark.parametrize("strategy", ["least_busy", "round_robin", "conversation"])
async def test_every_strategy_is_budgeted(strategy):
    table, _a, _b = _table(strategy=strategy)
    assert table.pick(table.resolve(LOCAL), affinity="k:s").budget is not None


async def test_a_cascaded_attempt_holds_room_only_where_it_went():
    """The first replica refused for want of room; while the second
    answers, the first must not still be charged for this turn."""
    from .test_failover_safety import pool_full

    table, a, b = _table()
    gate = asyncio.Event()
    seen: dict[str, int] = {}

    async def hold() -> None:
        seen["a"], seen["b"] = table._ledger.in_flight(A), table._ledger.in_flight(B)
        await gate.wait()

    a.generate_error = pool_full()
    b.generate_hook = hold
    client = table.pick(table.resolve(LOCAL), affinity="k:new")
    assert client.candidates[0].name == a.name, "the fixture needs a first"
    task = asyncio.ensure_future(client.generate(_turn()))
    await asyncio.sleep(0.05)
    gate.set()
    await task
    assert seen == {"a": 0, "b": client.budget.tokens}
