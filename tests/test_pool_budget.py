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
import time

import pytest

from eugene_plexus_gateway._generated.driver_models import GenerateRequest, Usage
from eugene_plexus_gateway.budget import (
    ANSWERS_KEPT,
    CHARS_PER_TOKEN,
    DEFAULT_ANSWER_FLOOR,
    NEW_CONVERSATION_ANSWER,
    ConversationSizes,
    PoolLedger,
    answer_room,
    estimate,
    long_answer,
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
    # CB-C2: not the whole `max_tokens` Claude Code sends (32,000): a
    # conversation that has not answered yet holds 4,096 for its answer.
    assert total == 31_000 + NEW_CONVERSATION_ANSWER


def test_a_new_conversation_is_estimated_from_its_size():
    prompt, total = estimate(7_000, None, None)
    assert prompt == int(7_000 / CHARS_PER_TOKEN)
    assert total == prompt + NEW_CONVERSATION_ANSWER


# --- CB-C2: the room an answer holds ---------------------------------------------


def test_a_conversation_holds_room_for_its_long_answers():
    """The length 95 of 100 of its last 16 answers were shorter than: one long
    answer among short ones is what it holds room for."""
    sizes = ConversationSizes()
    for answer in [300] * 15 + [2_500]:
        sizes.put("m", "k", prompt_tokens=1_000, chars=3_500, answer=answer)
    seen = sizes.get("m", "k")
    assert seen is not None and len(seen.answers) == ANSWERS_KEPT
    assert long_answer(seen.answers) == 2_500
    assert answer_room(32_000, seen) == 2_500


def test_only_the_last_sixteen_answers_count():
    sizes = ConversationSizes()
    for answer in [9_000] + [300] * 16:
        sizes.put("m", "k", prompt_tokens=1_000, chars=3_500, answer=answer)
    seen = sizes.get("m", "k")
    assert seen is not None and 9_000 not in seen.answers
    # Short answers hold the floor, never less.
    assert answer_room(32_000, seen) == DEFAULT_ANSWER_FLOOR
    assert answer_room(32_000, seen, floor=2_048) == 2_048


def test_max_tokens_caps_the_room():
    """An answer cannot be longer than `max_tokens`, so no more is held, whatever
    the floor says."""
    assert answer_room(100, None) == 100
    assert answer_room(100, None, floor=4_000) == 100
    assert answer_room(None, None) == NEW_CONVERSATION_ANSWER


async def test_an_answer_longer_than_its_room_is_counted():
    table, a, b = _table()
    for fake in (a, b):
        fake.usage = Usage(promptTokens=40, completionTokens=5_000, totalTokens=5_040)
    client = table.pick(table.resolve(LOCAL), affinity="k:long")
    await client.generate(_turn(max_tokens=32_000))
    assert client.budget.room == NEW_CONVERSATION_ANSWER
    assert client.outran_reserve is True
    for fake in (a, b):
        fake.usage = Usage(promptTokens=40, completionTokens=5, totalTokens=45)
    short = table.pick(table.resolve(LOCAL), affinity="k:short")
    await short.generate(_turn(max_tokens=32_000))
    assert short.outran_reserve is False


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
    """Rewritten for row 11: the wait for room is the fair queue's, bounded by
    `queueWaitSeconds`. After it a turn is no longer sent to an engine with no
    room (CB3 did, after `swapWaitSeconds`): it moves to the slot's next tier
    when there is one (TIER-SPILL), and is refused when there is none."""
    from eugene_plexus_gateway import admission

    table, a, b = _table()
    table._ledger.take(A, 9_500)
    table._ledger.take(B, 9_500)
    client = table.pick(table.resolve(LOCAL), affinity="k:new")
    client.queue_wait_seconds = 0.2
    started = asyncio.get_running_loop().time()
    with pytest.raises(admission.AdmissionFailure) as refused:
        await asyncio.wait_for(client.generate(_turn()), 2)
    assert asyncio.get_running_loop().time() - started >= 0.18
    assert refused.value.code == "queue_timeout"
    assert not a.calls and not b.calls, "nothing was sent to an engine with no room"
    assert table._ledger.in_flight(A) == table._ledger.in_flight(B) == 9_500

    spill, fakes = _slot_table("cooling")
    for runtime in (A, B):
        spill._ledger.take(runtime, 9_500)
    behind = spill.pick(spill.resolve(SLOT), affinity="k:new")
    (ahead,) = behind._tiers[0]
    behind._tiers = [[], behind._tiers[1], [ahead]]  # the local replicas first, then a fallback
    behind.queue_wait_seconds = 0.2
    await asyncio.wait_for(behind.generate(_turn()), 2)
    assert behind.served_by == "ahead-driver" and behind.tier_moves == 1
    assert not fakes["llama-a-driver"].calls and not fakes["llama-b-driver"].calls


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


# --- read ahead: who would be first ---------------------------------------------
#
# A response's head names what the backend its first attempt goes to does not
# honour, before anything is sent (`TieredClient.model_asked_first`), so it asks
# the budget who `admit` would put first without waiting or reserving.


def _front_tier(client):  # type: ignore[no-untyped-def]
    return client._tiers[client._front()]


async def test_first_names_who_admit_would_put_first_and_changes_nothing():
    table, _a, _b = _table(wait=0.05)
    table._ledger.take(A, 9_000)
    client = table.pick(table.resolve(LOCAL), affinity="k:new")
    tier = _front_tier(client)
    chosen = client.budget.first(_turn(), tier)
    assert chosen is not None and chosen.name == "llama-b-driver", "the replica with room"
    assert table._ledger.in_flight(A) == 9_000 and table._ledger.in_flight(B) == 0
    ordered = await client.budget.admit(_turn(), tier)
    assert ordered[0] is chosen, "and admit puts that one in front"
    client.budget.close()


async def test_first_names_nobody_when_nothing_has_room_and_admit_keeps_the_order():
    """`admit` places a turn no queue holds, so it waits for nothing (row 11: the
    wait is the fair queue's): nothing has room, the usual order stands, and no
    room is reserved."""
    table, _a, _b = _table(wait=0.05)
    table._ledger.take(A, 9_500)
    table._ledger.take(B, 9_500)
    client = table.pick(table.resolve(LOCAL), affinity="k:new")
    tier = _front_tier(client)
    assert client.budget.first(_turn(), tier) is None
    assert await client.budget.admit(_turn(), tier) == tier
    assert not client.budget.reserved


async def test_first_counts_only_the_replica_a_conversation_holds_as_admit_does():
    """Going elsewhere would read the conversation's whole history again, so while
    it holds its replica only that one counts: a full home is a wait, not another
    replica. `first` has to keep the rule `admit` keeps, or a head would name a
    backend the turn never goes to."""
    table, _a, _b = _table(wait=0.05)
    turn, task = _send(table, "k:s", _turn())
    await task
    home = A if turn.served_by == "llama-a-driver" else B
    table._ledger.take(home, 9_000)  # home is full; the other replica is empty

    held = table.pick(table.resolve(LOCAL), affinity="k:s")
    tier = _front_tier(held)
    assert tier[0].name == turn.served_by, "the conversation is placed on its replica"
    assert held.budget.first(_turn("and more"), tier) is None
    assert await held.budget.admit(_turn("and more"), tier) == tier

    new = table.pick(table.resolve(LOCAL), affinity="k:another")
    chosen = new.budget.first(_turn(), _front_tier(new))
    assert chosen is not None and chosen.name != turn.served_by, "a new one may go to the other"


# --- the read looks where the walk does ------------------------------------------
#
# `model_asked_first` is a copy of what `_admit` and `Executor.attempts` do, so each
# thing it copies has a test that fails when the copy is wrong: which tier it reads,
# that it reads only the front one, that it changes nothing, and how it sizes a turn.

SLOT = "slotty"


def _slot_table(front: str, *, wait: float = 0.05):  # type: ignore[no-untyped-def]
    """A slot of two targets: the first is `front`, and the second is the two pooled
    replicas. `asleep`: its runtime is stopped, so its tier is empty (an empty tier
    stays in the list). `cooling`: it is eligible, and a test opens its circuit."""
    ahead = FakeDriverClient(
        name="ahead-driver",
        base_url="http://z",
        model_id="ahead-model",
        runtime="rt-ahead" if front == "asleep" else None,
    )
    a = FakeDriverClient(
        name="llama-a-driver", base_url="http://a", model_id=LOCAL, runtime="llama-a"
    )
    b = FakeDriverClient(
        name="llama-b-driver", base_url="http://b", model_id=LOCAL, runtime="llama-b"
    )
    for fake in (a, b):
        fake.usage = Usage(promptTokens=40, completionTokens=4, totalTokens=44)
    runtimes = [
        *([runtime_facts("rt-ahead", status="stopped")] if front == "asleep" else []),
        runtime_facts("llama-a", parallel_slots=4, context_pool=POOL),
        runtime_facts("llama-b", parallel_slots=4, context_pool=POOL),
    ]
    table = make_routing_table(
        ahead,
        a,
        b,
        runtimes=runtimes,
        strategy="conversation",
        slots=[{"model": SLOT, "targets": ["ahead-model", LOCAL]}],
        room_wait_seconds=wait,
    )
    return table, {fake.name: fake for fake in (ahead, a, b)}


def _pool(candidate):  # type: ignore[no-untyped-def]
    """The ledger key of the runtime a replica of `_table` sits on."""
    return A if candidate.name == "llama-a-driver" else B


async def test_an_empty_first_tier_is_passed_over_by_the_read_and_by_admit_alike():
    """A slot whose first target is asleep has an empty first tier, and the first tier
    that holds anything is the second: the read must look there, for the replica with
    room, and `_admit` must put that one in front and reserve its room. A `_front`
    that always said tier 0 would read an empty tier (and name the tier's usual
    first) and have `_admit` order and reserve nothing."""
    table, _fakes = _slot_table("asleep")
    client = table.pick(table.resolve(SLOT), affinity="k:new")
    assert client._tiers[0] == [] and len(client._tiers[1]) == 2, "the slot's first tier is empty"
    assert client._front() == 1
    usual_first, roomy = client._tiers[1]
    table._ledger.take(_pool(usual_first), 9_000)

    named = client.model_asked_first(_turn())
    assert named is client.entries[roomy], "the replica with room, in the second tier"

    await client._admit(_turn())
    assert client._tiers[1][0] is roomy and client._tiers[0] == []
    assert table._ledger.in_flight(_pool(roomy)) == client.budget.tokens > 0
    client.budget.close()


async def test_a_fallback_tier_is_read_and_walked_to_its_replica_with_room():
    """Rewritten for row 11's review (blocker 2). When the front tier's only backend
    is cooling, the walk reaches the next tier as a fallback, and a fallback is let
    go on a replica of its tier with room, as the first turn is: never into the
    usual first, whose pool is full. The read that names who is asked first says
    the same replica. Checked against the real attempt."""
    table, fakes = _slot_table("cooling")
    client = table.pick(table.resolve(SLOT), affinity="k:new")
    (ahead,) = client._tiers[0]
    ahead.circuit.failures, ahead.circuit.until = 1, time.perf_counter() + 60
    usual_first, roomy = client._tiers[1]
    table._ledger.take(_pool(usual_first), 9_000)

    named = client.model_asked_first(_turn())
    assert named is client.entries[roomy], "the fallback tier's replica with room"

    await client.generate(_turn())
    assert fakes[roomy.name].calls and not fakes[usual_first.name].calls, "and the attempt agrees"


def test_reading_who_is_asked_first_leaves_the_tiers_as_they_were():
    """A read reorders a copy: the client's own tiers are what `Executor.attempts`
    walks next, so a read that moved the replica with room up in them would have
    changed the order the attempt goes in."""
    table, _a, _b = _table(wait=0.05)
    client = table.pick(table.resolve(LOCAL), affinity="k:new")
    usual_first, roomy = client._tiers[0]
    table._ledger.take(_pool(usual_first), 9_000)
    before = [list(tier) for tier in client._tiers]

    named = client.model_asked_first(_turn())
    assert named is client.entries[roomy], "the read moved the replica with room up"
    assert client._tiers == before and client._tiers[0] == [usual_first, roomy]


async def test_first_sizes_a_turn_with_its_max_tokens_as_admit_does():
    """A small prompt with a big `max_tokens` is a turn that does not fit beside what
    is already in the pool, and the same prompt with a small one does: the size that
    decides who has room includes the answer the turn may write (Claude Code sends
    32,000). A `first` that sized the prompt alone would name the full replica."""
    table, _a, _b = _table(wait=0.05)
    client = table.pick(table.resolve(LOCAL), affinity="k:new")
    tier = _front_tier(client)
    usual_first, other = tier
    table._ledger.take(_pool(usual_first), 5_000)
    small, large = _turn(max_tokens=1_000), _turn(max_tokens=6_000)

    assert client.budget.first(small, tier) is usual_first, "its answer fits beside what is there"
    assert client.budget.first(large, tier) is other, "its answer does not: max_tokens decides"
    ordered = await client.budget.admit(large, tier)
    assert ordered[0] is other, "and admit says the same"
    client.budget.close()


async def test_admit_keeps_how_big_the_turn_was_for_the_conversations_next_estimate():
    """The size a turn is estimated from next time is the characters the engine was
    given and the tokens it counted, so a turn's characters are kept when it is
    admitted and again when it is served."""
    table, _a, _b = _table(wait=0.05)
    client = table.pick(table.resolve(LOCAL), affinity="k:size")
    turn = _turn("fix the bug and then the test")
    await client.budget.admit(turn, _front_tier(client))
    assert client.budget.chars == request_chars(turn) > 0
    client.budget.close()

    served = _turn("and what about the docs")
    await table.pick(table.resolve(LOCAL), affinity="k:served").generate(served)
    seen = table._sizes.get(LOCAL, "k:served")
    assert seen is not None and seen.prompt_tokens == 40
    assert seen.chars == request_chars(served) > 0
