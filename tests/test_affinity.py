"""A conversation goes back to the replica that holds its prompt (PC4)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from eugene_plexus_gateway import affinity, anthropic
from eugene_plexus_gateway._generated.models import ChatCompletionRequest
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table, runtime_facts

LOCAL = "qwen3-1.7b"


def _replicas() -> tuple[FakeDriverClient, FakeDriverClient]:
    a = FakeDriverClient(
        name="qwen3-a-driver", base_url="http://a", model_id=LOCAL, runtime="qwen3-a"
    )
    b = FakeDriverClient(
        name="qwen3-b-driver", base_url="http://b", model_id=LOCAL, runtime="qwen3-b"
    )
    return a, b


def _table(strategy: str = "conversation"):  # type: ignore[no-untyped-def]
    a, b = _replicas()
    runtimes = [runtime_facts("qwen3-a"), runtime_facts("qwen3-b")]
    return make_routing_table(a, b, runtimes=runtimes, strategy=strategy)


def _first(table, key: str | None) -> str:  # type: ignore[no-untyped-def]
    client = table.pick(table.resolve(LOCAL), affinity=key)
    return client.candidates[0].name


def _chat(*messages: dict[str, object], **extra: object) -> ChatCompletionRequest:
    return ChatCompletionRequest.model_validate(
        {"model": LOCAL, "messages": list(messages), **extra}
    )


# --- the key ---------------------------------------------------------------------


def test_a_conversation_keeps_its_key_as_it_grows() -> None:
    turn1 = _chat(
        {"role": "system", "content": "rules"}, {"role": "user", "content": "fix the bug"}
    )
    turn2 = _chat(
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": "looking"},
        {"role": "user", "content": "and the test"},
    )
    assert affinity.key_for(turn1) == affinity.key_for(turn2) is not None


def test_two_conversations_that_start_differently_have_different_keys() -> None:
    a = _chat({"role": "user", "content": "fix the bug"})
    b = _chat({"role": "user", "content": "write the docs"})
    assert affinity.key_for(a) != affinity.key_for(b)


def test_the_clients_own_key_wins() -> None:
    body = _chat({"role": "user", "content": "x"}, prompt_cache_key="codex-session-1")
    assert affinity.key_for(body) == "k:codex-session-1"
    assert affinity.key_for(body, "claude-session") == "k:claude-session"


def test_no_user_message_no_key() -> None:
    assert affinity.key_for(_chat({"role": "system", "content": "only rules"})) is None


def test_claude_codes_session_is_read_from_its_metadata() -> None:
    raw = {"metadata": {"user_id": json.dumps({"device_id": "d", "session_id": "s-123"})}}
    assert anthropic.session_of(raw) == "s-123"
    # A bare user id names a person, not a conversation.
    assert anthropic.session_of({"metadata": {"user_id": "alice"}}) is None
    assert anthropic.session_of({}) is None


# --- the balancer ----------------------------------------------------------------


def test_one_conversation_stays_on_one_replica() -> None:
    """**The measured defect**: on an idle install, least-busy's rotation sent
    consecutive turns of one conversation to alternate replicas."""
    table = _table()
    firsts = [_first(table, "k:session") for _ in range(4)]
    assert len(set(firsts)) == 1


def test_every_strategy_keeps_a_conversation_on_its_replica() -> None:
    """CB1: affinity is not a strategy any more. `loadBalancing` says only
    where a NEW conversation goes."""
    for strategy in ("least_busy", "round_robin", "conversation", None):
        table = _table(strategy)  # type: ignore[arg-type]
        firsts = [_first(table, "k:session") for _ in range(4)]
        assert len(set(firsts)) == 1, strategy


def test_with_affinity_off_least_busy_alternates() -> None:
    """The pair that says the setting, not something else, does it: off is
    for benchmarking replicas, where one prompt repeats."""
    a, b = _replicas()
    table = make_routing_table(
        a,
        b,
        runtimes=[runtime_facts("qwen3-a"), runtime_facts("qwen3-b")],
        strategy="least_busy",
        affinity=False,
    )
    firsts = [_first(table, "k:session") for _ in range(4)]
    assert firsts == ["qwen3-a-driver", "qwen3-b-driver", "qwen3-a-driver", "qwen3-b-driver"]
    assert table.pick(table.resolve(LOCAL), affinity="k:session").affinity is None


def test_round_robin_places_new_conversations_in_turn() -> None:
    table = _table("round_robin")
    firsts = [_first(table, f"k:s{i}") for i in range(4)]
    assert firsts == ["qwen3-a-driver", "qwen3-b-driver", "qwen3-a-driver", "qwen3-b-driver"]


def test_the_old_conversation_value_is_the_default_placement() -> None:
    """What every request records: the placement that ran, never a value
    that no longer means anything."""
    for strategy in ("conversation", None, "something-else"):
        assert _table(strategy).placement() == "spread"  # type: ignore[arg-type]
    assert _table("round_robin").placement() == "round_robin"
    assert _table("least_busy").placement() == "least_busy"


# --- spread (CB2) ------------------------------------------------------------------


def _three(strategy: str = "spread", slots: tuple[int, int, int] = (4, 4, 4)):  # type: ignore[no-untyped-def]
    fakes = [
        FakeDriverClient(
            name=f"qwen3-{x}-driver", base_url=f"http://{x}", model_id=LOCAL, runtime=f"qwen3-{x}"
        )
        for x in "abc"
    ]
    runtimes = [
        runtime_facts(f"qwen3-{x}", parallel_slots=n) for x, n in zip("abc", slots, strict=True)
    ]
    return make_routing_table(*fakes, runtimes=runtimes, strategy=strategy)


def test_spread_places_a_new_conversation_where_fewest_are_held():
    """**The measured case**: least busy counts requests in flight, and a
    replica whose agents are all thinking between turns looks empty, so new
    conversations pile onto it. Spread counts what each replica holds."""
    table = _three()
    for key, home in (("k:1", "a"), ("k:2", "a"), ("k:3", "b")):
        table._affinity.put(LOCAL, key, (None, f"qwen3-{home}-driver"))
    # Nothing in flight anywhere: least busy would see three empty replicas.
    assert _first(table, "k:new") == "qwen3-c-driver"


def test_least_busy_does_not_see_what_is_held():
    """The pair that says spread, not something else, does it."""
    table = _three("least_busy")
    for key in ("k:1", "k:2"):
        table._affinity.put(LOCAL, key, (None, "qwen3-a-driver"))
    assert _first(table, "k:new") == "qwen3-a-driver"


def test_spread_counts_per_slot():
    table = _three(slots=(8, 2, 2))
    for key, home in (("k:1", "a"), ("k:2", "a"), ("k:3", "b"), ("k:4", "c")):
        table._affinity.put(LOCAL, key, (None, f"qwen3-{home}-driver"))
    # a holds 2 of 8 slots; b and c 1 of 2 each.
    assert _first(table, "k:new") == "qwen3-a-driver"


def test_spread_breaks_a_tie_by_requests_in_flight():
    table = _three()
    table.on_attempt_start("qwen3-a-driver")
    table.on_attempt_start("qwen3-b-driver")
    assert _first(table, "k:new") == "qwen3-c-driver"


def test_spread_breaks_the_last_tie_in_turn():
    table = _three()
    firsts = [_first(table, f"k:s{i}") for i in range(3)]
    assert sorted(firsts) == ["qwen3-a-driver", "qwen3-b-driver", "qwen3-c-driver"]


def test_a_conversation_is_not_counted_against_its_own_placement():
    table = _three()
    table._affinity.put(LOCAL, "k:mine", (None, "qwen3-a-driver"))
    table._affinity.put(LOCAL, "k:other", (None, "qwen3-b-driver"))
    held = table._affinity.held(LOCAL, besides="k:mine")
    assert held == {(None, "qwen3-b-driver"): 1}


def test_an_expired_conversation_is_not_held(monkeypatch):
    from eugene_plexus_gateway import affinity as module

    table = _three()
    now = [1000.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: now[0])
    table._affinity.put(LOCAL, "k:old", (None, "qwen3-a-driver"))
    now[0] += module.TTL_SECONDS + 1
    assert table._affinity.held(LOCAL) == {}


def test_a_held_conversation_still_goes_home_under_spread():
    table = _three()
    home = _first(table, "k:session")
    for key in ("k:1", "k:2", "k:3"):
        table._affinity.put(LOCAL, key, (None, home))
    assert _first(table, "k:session") == home


def test_new_conversations_are_still_spread() -> None:
    table = _table()
    firsts = {_first(table, f"k:s{i}") for i in range(4)}
    assert firsts == {"qwen3-a-driver", "qwen3-b-driver"}


def test_a_full_replica_lets_its_conversation_move() -> None:
    """A warm cache is worth a short queue, not a long one."""
    table = _table()
    home = _first(table, "k:session")
    runtime = table.on_attempt_start(home)  # its one slot is now busy, the other idle
    client = table.pick(table.resolve(LOCAL), affinity="k:session")
    assert client.candidates[0].name != home
    assert client.affinity == affinity.MOVED
    # And the conversation follows: its next turn goes to the new replica.
    table.on_attempt_end(home, runtime=runtime, served=True, elapsed_ms=1)
    assert _first(table, "k:session") == client.candidates[0].name


def test_when_every_replica_is_full_the_conversation_stays_home() -> None:
    table = _table()
    home = _first(table, "k:session")
    for name in ("qwen3-a-driver", "qwen3-b-driver"):
        table.on_attempt_start(name)
    client = table.pick(table.resolve(LOCAL), affinity="k:session")
    assert client.candidates[0].name == home
    assert client.affinity == affinity.HIT


def test_the_outcome_is_on_the_client() -> None:
    table = _table()
    assert table.pick(table.resolve(LOCAL), affinity="k:s").affinity == affinity.NEW
    assert table.pick(table.resolve(LOCAL), affinity="k:s").affinity == affinity.HIT
    assert table.pick(table.resolve(LOCAL)).affinity is None


def test_the_table_is_bounded_and_forgets_the_idle() -> None:
    t = affinity.AffinityTable(ttl=0.0, size=2)
    t.put("m", "a", 1)
    assert t.get("m", "a") is None  # idle past its expiry
    t = affinity.AffinityTable(size=2)
    for key in ("a", "b", "c"):
        t.put("m", key, key)
    assert len(t) == 2 and t.get("m", "a") is None


# --- through a door --------------------------------------------------------------


def _app(settings: Settings, table) -> TestClient:  # type: ignore[no-untyped-def]
    app = create_app(settings=settings)
    app.state.routing = table
    return TestClient(app)


def test_a_chat_conversation_stays_on_its_replica_through_the_door(settings: Settings) -> None:
    a, b = _replicas()
    a.responses = ["one", "two", "three"]
    b.responses = ["one", "two", "three"]
    table = make_routing_table(
        a, b, runtimes=[runtime_facts("qwen3-a"), runtime_facts("qwen3-b")], strategy="conversation"
    )
    history = [{"role": "user", "content": "fix the bug"}]
    with _app(settings, table) as client:
        for turn in range(3):
            r = client.post("/v1/chat/completions", json={"model": LOCAL, "messages": history})
            assert r.status_code == 200, r.text
            history = [
                *history,
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": f"t{turn}"},
            ]
    assert sorted([len(a.calls), len(b.calls)]) == [0, 3]


def test_claude_codes_session_stays_on_its_replica_through_the_door(settings: Settings) -> None:
    a, b = _replicas()
    a.responses = ["ok"] * 3
    b.responses = ["ok"] * 3
    table = make_routing_table(
        a, b, runtimes=[runtime_facts("qwen3-a"), runtime_facts("qwen3-b")], strategy="conversation"
    )
    meta = {"user_id": json.dumps({"session_id": "cc-1"})}
    with _app(settings, table) as client:
        for text in ("one", "two", "three"):
            # Each turn's first message differs, so only the session id can
            # hold the conversation together.
            r = client.post(
                "/v1/messages",
                json={
                    "model": LOCAL,
                    "max_tokens": 8,
                    "metadata": meta,
                    "messages": [{"role": "user", "content": text}],
                },
            )
            assert r.status_code == 200, r.text
    assert sorted([len(a.calls), len(b.calls)]) == [0, 3]
