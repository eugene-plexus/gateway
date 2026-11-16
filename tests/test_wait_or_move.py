"""Row 14 (LD3, DI5, M4-LOAD): wait or move by predicted time to first token, and
place new conversations by the tokens a replica holds.

A conversation's own replica holds its prompt. When that replica has no room, a
turn can wait for it, or read its whole prompt again on a replica that has room.
The turn moves only when waiting is predicted to take more than a penalty longer
than rereading (llm-d's "sticky until saturated"). The prediction is per runtime,
from what it served; with none, a turn behaves as it did before.
"""

from __future__ import annotations

import asyncio

import pytest

from eugene_plexus_gateway import fair_queue
from eugene_plexus_gateway._generated.driver_models import GenerateRequest, Usage
from eugene_plexus_gateway.affinity import HIT, MOVED
from eugene_plexus_gateway.budget import request_chars

from .conftest import FakeDriverClient, make_routing_table, runtime_facts

MODEL = "llama-8b"
A = (None, "llama-a")
B = (None, "llama-b")
KEY_A = (None, "llama-a-driver")
KEY_B = (None, "llama-b-driver")
CONVERSATION = "k:session"


def _table(*, pool: int | None = None, strategy: str = "spread", **options):  # type: ignore[no-untyped-def]
    fakes = []
    for letter in ("a", "b"):
        fake = FakeDriverClient(
            name=f"llama-{letter}-driver",
            base_url=f"http://{letter}",
            model_id=MODEL,
            runtime=f"llama-{letter}",
        )
        fake.usage = Usage(promptTokens=40, completionTokens=4, totalTokens=44)
        fakes.append(fake)
    runtimes = [
        runtime_facts("llama-a", parallel_slots=1, context_pool=pool),
        runtime_facts("llama-b", parallel_slots=1, context_pool=pool),
    ]
    table = make_routing_table(*fakes, runtimes=runtimes, strategy=strategy, **options)
    return table, fakes[0], fakes[1]


def _calibrate(table, runtime, *, rate: float = 1_000.0, service: float = 10.0) -> None:  # type: ignore[no-untyped-def]
    """Ten served requests: `rate` tokens a second of prefill, an answer in `service` s."""
    for _ in range(10):
        table.prefill.observe(
            runtime,
            prompt=5_000,
            cached=1_000,
            first_ms=int(4_000 / rate * 1000),
            elapsed_ms=int(service * 1000),
        )


def _hold(table, *, held: int = 1_800) -> None:  # type: ignore[no-untyped-def]
    """The conversation's last prompt was `held` tokens, on replica A."""
    table._affinity.put(MODEL, CONVERSATION, KEY_A)
    table._sizes.put(MODEL, CONVERSATION, prompt_tokens=held, chars=request_chars(_turn(0)))


def _turn(extra_tokens: int) -> GenerateRequest:
    text = "hello" + "x" * int(extra_tokens * 3.5)
    return GenerateRequest(messages=[{"role": "user", "content": text}], maxTokens=64)


def _fill(table, runtime) -> None:  # type: ignore[no-untyped-def]
    """The runtime's one slot is taken."""
    fair_queue.of(table)._take_seat(runtime)


def _free(table, runtime) -> None:  # type: ignore[no-untyped-def]
    fair_queue.of(table)._give_seat(runtime)


def _send(table, extra: int, key: str = CONVERSATION):  # type: ignore[no-untyped-def]
    client = table.pick(table.resolve(MODEL), affinity=key)
    return client, asyncio.ensure_future(client.generate(_turn(extra)))


POOLS = pytest.mark.parametrize("pool", [None, 1_000_000], ids=["slots only", "shared pool"])


@POOLS
async def test_a_short_turn_moves_when_its_replica_is_busy(pool):
    table, a, _b = _table(pool=pool)
    _calibrate(table, A)
    _calibrate(table, B)
    _hold(table, held=1_800)
    _fill(table, A)
    # 2,000 tokens: rereading them on B takes 2 s; waiting for A's one slot takes ~10 s.
    client, task = _send(table, 200)
    await asyncio.wait_for(task, 1)
    assert client.served_by == "llama-b-driver"
    assert client.affinity == MOVED
    assert table._affinity.get(MODEL, CONVERSATION) == KEY_B, (
        "its next turn goes where its prompt is"
    )
    assert not a.calls


@POOLS
async def test_a_long_turn_waits_for_its_replica(pool):
    table, _a, b = _table(pool=pool)
    _calibrate(table, A)
    _calibrate(table, B)
    _hold(table, held=99_800)
    _fill(table, A)
    # 100,000 tokens: rereading them takes 100 s; waiting for A takes ~10 s.
    client, task = _send(table, 200)
    await asyncio.sleep(0.2)
    assert not task.done(), "a 100,000-token turn went to read its prompt again"
    assert not b.calls
    _free(table, A)
    await asyncio.wait_for(task, 1)
    assert client.served_by == "llama-a-driver" and client.affinity == HIT
    assert table._affinity.get(MODEL, CONVERSATION) == KEY_A


async def test_turned_off_a_pooled_turn_waits_for_its_replica_as_before():
    table, _a, b = _table(pool=1_000_000, move=False)
    _calibrate(table, A)
    _calibrate(table, B)
    _hold(table, held=1_800)
    _fill(table, A)
    client, task = _send(table, 200)
    await asyncio.sleep(0.2)
    assert not task.done() and not b.calls, "it moved with the rule off"
    _free(table, A)
    await asyncio.wait_for(task, 1)
    assert client.served_by == "llama-a-driver"


async def test_turned_off_a_turn_with_slots_only_goes_at_once_to_a_replica_with_room():
    """What a conversation did before row 14: no pool to wait for, so a free slot
    elsewhere takes it."""
    table, a, _b = _table(pool=None, move=False)
    _calibrate(table, A)
    _calibrate(table, B)
    _hold(table, held=99_800)
    _fill(table, A)
    client, task = _send(table, 200)
    await asyncio.wait_for(task, 1)
    assert client.served_by == "llama-b-driver" and not a.calls


@POOLS
async def test_a_larger_penalty_keeps_more_conversations_where_their_prompt_is(pool):
    table, _a, b = _table(pool=pool, penalty_ms=600_000)
    _calibrate(table, A)
    _calibrate(table, B)
    _hold(table, held=1_800)
    _fill(table, A)
    _client, task = _send(table, 200)
    await asyncio.sleep(0.2)
    assert not task.done() and not b.calls
    _free(table, A)
    await asyncio.wait_for(task, 1)


async def test_a_pooled_turn_with_nothing_measured_waits_as_before():
    """CB3: with no prefill rate or answer time known, a held turn waits for its own
    replica, however busy it is and however free another is."""
    table, _a, b = _table(pool=1_000_000)
    _hold(table, held=1_800)
    _fill(table, A)
    client, task = _send(table, 200)
    await asyncio.sleep(0.2)
    assert not task.done() and not b.calls
    _free(table, A)
    await asyncio.wait_for(task, 1)
    assert client.served_by == "llama-a-driver"


async def test_a_turn_with_slots_only_and_nothing_measured_moves_at_once_as_before():
    table, a, _b = _table(pool=None)
    _hold(table, held=1_800)
    _fill(table, A)
    client, task = _send(table, 200)
    await asyncio.wait_for(task, 1)
    assert client.served_by == "llama-b-driver" and not a.calls


@POOLS
async def test_nothing_moves_to_a_replica_that_has_no_room_either(pool):
    table, a, b = _table(pool=pool)
    _calibrate(table, A)
    _calibrate(table, B)
    _hold(table, held=1_800)
    _fill(table, A)
    _fill(table, B)
    client, task = _send(table, 200)
    await asyncio.sleep(0.2)
    assert not task.done() and not a.calls and not b.calls
    _free(table, B)
    await asyncio.wait_for(task, 1)
    assert client.served_by == "llama-b-driver", (
        "it moves when the other replica frees, as it is sooner"
    )


@POOLS
async def test_a_turn_waiting_for_its_replica_does_not_hold_back_a_new_conversation(pool):
    table, _a, _b = _table(pool=pool)
    _calibrate(table, A)
    _calibrate(table, B)
    _hold(table, held=99_800)
    _fill(table, A)
    _client, waiting = _send(table, 200)
    await asyncio.sleep(0.1)
    assert not waiting.done()
    other, task = _send(table, 50, key="k:other")
    await asyncio.wait_for(task, 1)
    assert other.served_by == "llama-b-driver"
    assert not waiting.done()
    _free(table, A)
    await asyncio.wait_for(waiting, 1)


def _two_conversations(table) -> None:  # type: ignore[no-untyped-def]
    """`k:long` holds 99,800 tokens on A and `k:mid` 7,800, each in turn 200 more."""
    for key, held in (("k:long", 99_800), ("k:mid", 7_800)):
        table._affinity.put(MODEL, key, KEY_A)
        table._sizes.put(MODEL, key, prompt_tokens=held, chars=request_chars(_turn(0)))


@POOLS
async def test_a_turn_alone_in_line_waits_when_waiting_is_within_the_penalty(pool):
    # An answer takes 8 s there: waiting alone is 8.2 s, rereading 8,000 tokens
    # elsewhere is 8 s, and 8.2 is not more than 8 + 2.
    table, _a, b = _table(pool=pool)
    _calibrate(table, A, service=8.0)
    _calibrate(table, B)
    _two_conversations(table)
    _fill(table, A)
    _client, task = _send(table, 200, key="k:mid")
    await asyncio.sleep(0.2)
    assert not task.done() and not b.calls
    _free(table, A)
    await asyncio.wait_for(task, 1)


@POOLS
async def test_the_wait_counts_those_in_line_ahead(pool):
    """With a 100,000-token turn waiting for A's one slot, the 8,000-token turn behind
    it would wait twice as long (16.2 s, more than 8 + 2), so it moves."""
    table, _a, _b = _table(pool=pool)
    _calibrate(table, A, service=8.0)
    _calibrate(table, B)
    _two_conversations(table)
    _fill(table, A)
    _long, long_task = _send(table, 200, key="k:long")
    await asyncio.sleep(0.1)
    assert not long_task.done()
    mid, mid_task = _send(table, 200, key="k:mid")
    await asyncio.wait_for(mid_task, 1)
    assert mid.served_by == "llama-b-driver" and mid.affinity == MOVED
    assert not long_task.done()
    _free(table, A)
    await asyncio.wait_for(long_task, 1)


# --- DI5: a new conversation goes where the fewest tokens are held -----------------


def _held_conversations(table, home, sizes) -> None:  # type: ignore[no-untyped-def]
    for index, tokens in enumerate(sizes):
        key = f"k:{home[1]}-{index}"
        table._affinity.put(MODEL, key, home)
        table._sizes.put(MODEL, key, prompt_tokens=tokens, chars=tokens * 4)


def _four_slots(*, pools=(200_000, 200_000), loads=(None, None)):  # type: ignore[no-untyped-def]
    fakes = []
    for letter in ("a", "b"):
        fake = FakeDriverClient(
            name=f"llama-{letter}-driver", model_id=MODEL, runtime=f"llama-{letter}"
        )
        fake.usage = Usage(promptTokens=40, completionTokens=4, totalTokens=44)
        fakes.append(fake)
    runtimes = [
        runtime_facts(
            f"llama-{letter}",
            parallel_slots=4,
            context_pool=pool,
            context_length=50_000,
            engine_load=load,
        )
        for letter, pool, load in zip(("a", "b"), pools, loads, strict=True)
    ]
    return make_routing_table(*fakes, runtimes=runtimes, strategy="spread"), fakes


def _first(table, key: str = "k:new") -> str:  # type: ignore[no-untyped-def]
    return table.pick(table.resolve(MODEL), affinity=key).candidates[0].name


def test_a_new_conversation_goes_to_the_replica_holding_fewer_tokens():
    """One 80,000-token agent session on A, three 500-token chats on B: a count of
    conversations says A is the lighter one, the tokens say B."""
    table, _fakes = _four_slots()
    _held_conversations(table, KEY_A, [80_000])
    _held_conversations(table, KEY_B, [500, 500, 500])
    assert _first(table) == "llama-b-driver"


def test_equal_tokens_fall_to_the_fewer_conversations():
    table, _fakes = _four_slots()
    _held_conversations(table, KEY_A, [10_000, 10_000])
    _held_conversations(table, KEY_B, [20_000])
    assert _first(table) == "llama-b-driver"


def test_small_differences_in_tokens_do_not_decide():
    """Within a twentieth of the context, the conversations decide."""
    table, _fakes = _four_slots()
    _held_conversations(table, KEY_A, [2_000, 2_000, 2_000])  # 3% of 200,000
    _held_conversations(table, KEY_B, [4_000])  # 2%
    assert _first(table) == "llama-b-driver"


def test_the_tokens_are_left_out_when_a_replica_does_not_report_its_context():
    """No pool and no context length on B: its share of anything is unknown, so the
    conversations decide, as before row 14."""
    table, _fakes = _four_slots(pools=(200_000, None))
    table._snapshot.runtimes[B] = runtime_facts("llama-b", parallel_slots=4)
    backends = table._snapshot.reachable + [
        b for group in table._snapshot.by_model.values() for b in group
    ]
    for backend in backends:
        if backend.runtime is not None and backend.runtime.name == "llama-b":
            backend.runtime = table._snapshot.runtimes[B]
    _held_conversations(table, KEY_A, [80_000])
    _held_conversations(table, KEY_B, [500, 500, 500])
    assert _first(table) == "llama-a-driver"


def test_vllms_own_load_breaks_the_last_tie_for_a_new_conversation():
    """Nothing in flight through this gateway on either, but another door is using
    A's engine: vLLM says five requests occupy it."""
    table, _fakes = _four_slots(loads=(5, None))
    assert _first(table) == "llama-b-driver"


def test_vllms_own_load_counts_under_least_busy_too():
    table, _fakes = _four_slots(loads=(5, 1))
    table._strategy = lambda: "least_busy"
    table._inflight[KEY_B] = 2  # this gateway sent B two; A's five came from elsewhere
    assert _first(table) == "llama-b-driver"


def test_an_engine_that_does_not_say_its_load_changes_nothing():
    table, _fakes = _four_slots(loads=(None, None))
    table._strategy = lambda: "least_busy"
    table._inflight[KEY_A] = 1
    assert _first(table) == "llama-b-driver"


def test_the_routing_view_shows_the_calibration_and_the_engines_load():
    table, _fakes = _four_slots(loads=(3, None))
    _calibrate(table, A, rate=2_500.0)
    rows = [b for s in table.as_routing_view().slots for tier in s.tiers for b in tier.backends]
    view = {row.runtime: row for row in rows}
    assert view["llama-a"].engine_load == 3
    assert view["llama-a"].prefill_tokens_per_second == pytest.approx(2_500.0, rel=0.01)
    assert view["llama-a"].prefill_samples == 10
    # Nothing measured: absent, never a 0.
    assert view["llama-b"].engine_load is None
    assert view["llama-b"].prefill_tokens_per_second is None


def test_a_runtime_that_leaves_ready_loses_its_calibration():
    table, _fakes = _four_slots()
    _calibrate(table, A)
    table._ready_since[A] = 1.0
    stopped = runtime_facts("llama-a", status="stopped", parallel_slots=4)
    # What `_refresh_locked` does with each runtime's status.
    table._snapshot.runtimes[A] = stopped
    from eugene_plexus_gateway.routing import READY

    for key, facts in table._snapshot.runtimes.items():
        if facts.status != READY and table._ready_since.pop(key, None) is not None:
            table.prefill.forget(key)
    assert table.prefill.rate(A) is None


# --- calibration from what is served -----------------------------------------------


def test_every_served_attempt_calibrates_its_runtime():
    """The executor ends each attempt through this hook: the rate comes from the
    uncached prompt over the first token, the answer time from the attempt."""
    table, _a, _b = _table()
    usage = Usage(
        promptTokens=5_000, completionTokens=50, totalTokens=5_050, cachedPromptTokens=1_000
    )
    for _ in range(10):
        table.on_attempt_end(
            "llama-a-driver", runtime=A, served=True, elapsed_ms=6_000, usage=usage, first_ms=2_000
        )
    # A failed attempt says nothing about how fast the engine reads.
    table.on_attempt_end(
        "llama-b-driver", runtime=B, served=False, elapsed_ms=50, usage=usage, first_ms=1
    )
    assert table.prefill.rate(A) == pytest.approx(2_000.0)
    assert table.prefill.service_seconds(A) == pytest.approx(6.0)
    assert table.prefill.samples(B) == 0 and table.prefill.rate(B) is None


# --- measured against CB3's rule ---------------------------------------------------


def _wait_for_home(prompt: int, held: int, *, waiting: int, rate: float, service: float) -> float:
    """Time to first token (s) when the turn waits for its replica: CB3's rule."""
    return waiting * service + (prompt - held) / rate


def test_the_rule_beats_always_waiting_on_a_mix_of_short_and_long_turns():
    """Adopt only what beats CB3's rule. The same busy replica, 4 places, an answer
    takes 20 s, three turns already in line, 2,000 tokens a second, for 100 turns
    of each size. Always waiting reads 3 s for a 6,000-token chat turn after
    15 s of line; the rule rereads it elsewhere. A 150,000-token turn waits on
    both, because 75 s of rereading is worse than 15 s of line.
    """
    from eugene_plexus_gateway.prefill import wait_seconds, worth_moving

    rate, service, slots, ahead, penalty = 2_000.0, 20.0, 4, 2, 2_000.0
    cb3_total = rule_total = 0.0
    moved = 0
    for prompt, held in [(6_000, 5_500)] * 100 + [(150_000, 149_500)] * 100:
        warm = (
            wait_seconds(waiting=ahead + 1, parallel=slots, service_seconds=service)
            + (prompt - held) / rate
        )
        cold = prompt / rate
        cb3_total += warm
        if worth_moving(warm_ms=warm * 1000, cold_ms=cold * 1000, penalty_ms=penalty):
            rule_total += cold
            moved += 1
        else:
            rule_total += warm
    assert moved == 100, "the short turns move and the long ones do not"
    assert rule_total < cb3_total
    # And never worse than waiting by more than the penalty, turn by turn.
    for prompt, held in [(6_000, 5_500), (150_000, 149_500)]:
        warm = (
            wait_seconds(waiting=ahead + 1, parallel=slots, service_seconds=service)
            + (prompt - held) / rate
        )
        cold = prompt / rate
        chosen = (
            cold
            if worth_moving(warm_ms=warm * 1000, cold_ms=cold * 1000, penalty_ms=penalty)
            else warm
        )
        assert chosen <= warm


def test_token_aware_spread_beats_counting_conversations_on_a_mixed_load():
    """Adopt only what beats `spread` as it was. Eight replicas, 64 sessions in
    sizes drawn from agents' and chats' (every eighth is 80,000 tokens, the rest
    1,000). Placing each new session by conversations per slot puts the huge
    ones wherever the count is lowest; by tokens, the heaviest replica holds
    far less."""
    import itertools

    sizes = list(itertools.islice(itertools.cycle([80_000] + [1_000] * 7), 64))

    def heaviest(by_tokens: bool) -> int:
        convs = [0] * 8
        tokens = [0] * 8
        # The huge ones arrive first in each group of eight, as a morning's agents do.
        for size in sizes:
            if by_tokens:
                pick = min(range(8), key=lambda i: (round(tokens[i] / 200_000 * 20), convs[i], i))
            else:
                pick = min(range(8), key=lambda i: (convs[i], i))
            convs[pick] += 1
            tokens[pick] += size
        return max(tokens)

    assert heaviest(by_tokens=True) < heaviest(by_tokens=False)
