"""A turn that went home and found its history gone was evicted (CB5).

An affinity hit says where a turn went; only the answer says whether the
replica still held the conversation. The measured definition: the turn went
back to the replica that served its last one, and the engine reused less
than that turn's whole prompt. On the Metrics page it reads "this model
needs more context or another replica", where an operator saw only a slow
model.
"""

from __future__ import annotations

import pytest

from eugene_plexus_gateway._generated.driver_models import GenerateRequest, Usage
from eugene_plexus_gateway.affinity import EVICTED, HIT, MOVED

from .conftest import FakeDriverClient, make_routing_table, runtime_facts

LOCAL = "qwen"


def _table():  # type: ignore[no-untyped-def]
    a = FakeDriverClient(name="a-driver", base_url="http://a", model_id=LOCAL, runtime="a")
    b = FakeDriverClient(name="b-driver", base_url="http://b", model_id=LOCAL, runtime="b")
    return make_routing_table(a, b, runtimes=[runtime_facts("a"), runtime_facts("b")]), a, b


def _usage(prompt: int, cached: int | None) -> Usage:
    return Usage(
        promptTokens=prompt, completionTokens=4, totalTokens=prompt + 4, cachedPromptTokens=cached
    )


def _turn() -> GenerateRequest:
    return GenerateRequest(messages=[{"role": "user", "content": "fix the bug"}])


async def _send(table, usage: Usage, *, stream: bool = False):  # type: ignore[no-untyped-def]
    client = table.pick(table.resolve(LOCAL), affinity="k:s")
    for fake in table._snapshot.reachable:
        getattr(fake.client, "_inner", fake.client).usage = usage
    if stream:
        async for _ in client.stream(_turn()):
            pass
    else:
        await client.generate(_turn())
    return client


@pytest.mark.parametrize("stream", [False, True])
async def test_a_hit_that_reused_its_whole_history_is_a_hit(stream):
    table, _a, _b = _table()
    await _send(table, _usage(1_000, 0), stream=stream)
    second = await _send(table, _usage(1_200, 1_000), stream=stream)
    assert second.affinity == HIT


@pytest.mark.parametrize("stream", [False, True])
async def test_a_hit_that_reused_less_than_its_last_prompt_was_evicted(stream):
    table, _a, _b = _table()
    await _send(table, _usage(1_000, 0), stream=stream)
    second = await _send(table, _usage(1_200, 999), stream=stream)
    assert second.affinity == EVICTED


async def test_a_backend_that_does_not_say_cannot_be_judged():
    table, _a, _b = _table()
    await _send(table, _usage(1_000, None))
    second = await _send(table, _usage(1_200, None))
    assert second.affinity == HIT


async def test_a_first_turn_is_new_whatever_it_reused():
    table, _a, _b = _table()
    first = await _send(table, _usage(1_000, 0))
    assert first.affinity == "new"


async def test_a_moved_turn_is_moved_not_evicted():
    table, _a, _b = _table()
    home = (await _send(table, _usage(1_000, 0))).served_by
    table.on_attempt_start(home)  # its one slot busy, the other free: it moves
    moved = await _send(table, _usage(1_200, 0))
    assert moved.affinity == MOVED


async def test_a_hit_that_failed_over_is_not_judged_against_another_replica():
    """The turn left its home replica on a cascade; what the next replica
    reused says nothing about the home's cache."""
    from .test_failover_safety import pool_full

    table, a, b = _table()
    first = await _send(table, _usage(1_000, 0))
    home = a if first.served_by == a.name else b
    other = b if home is a else a
    home.generate_error = pool_full()
    second = await _send(table, _usage(1_200, 0))
    assert second.served_by == other.name
    assert second.affinity == HIT


async def test_the_next_turn_is_judged_against_this_one():
    table, _a, _b = _table()
    await _send(table, _usage(1_000, 0))
    await _send(table, _usage(2_000, 1_000))
    third = await _send(table, _usage(2_500, 1_500))
    assert third.affinity == EVICTED, "judged against the 2,000-token second turn"
