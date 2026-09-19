"""M6 routing: slots, tiers, the ready gate, the balancer, the views.

Fakes only. Every test here asks the table a question and checks the
answer; the HTTP refresh that builds the table is in test_routing.py.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table, runtime_facts

LOCAL = "qwen3-1.7b"
CLOUD = "claude-opus-4-7"


def _replicas(**kwargs: object) -> tuple[FakeDriverClient, FakeDriverClient]:
    a = FakeDriverClient(
        name="qwen3-a-driver", base_url="http://a", model_id=LOCAL, runtime="qwen3-a"
    )
    b = FakeDriverClient(
        name="qwen3-b-driver", base_url="http://b", model_id=LOCAL, runtime="qwen3-b"
    )
    return a, b


def _chat(model: str = LOCAL) -> dict[str, object]:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


def _app(settings: Settings, table) -> TestClient:  # type: ignore[no-untyped-def]
    app = create_app(settings=settings)
    app.state.routing = table
    return TestClient(app)


# --- the ready gate -----------------------------------------------------------


def test_a_driver_whose_runtime_is_not_ready_is_not_routed_to() -> None:
    a, b = _replicas()
    table = make_routing_table(
        a,
        b,
        runtimes=[
            runtime_facts("qwen3-a", status="ready"),
            runtime_facts("qwen3-b", status="stopped"),
        ],
    )
    resolution = table.resolve(LOCAL)
    assert [x.name for x in resolution.backends()] == ["qwen3-a-driver", "qwen3-b-driver"]
    assert [x.name for x in resolution.eligible_backends()] == ["qwen3-a-driver"]
    client = table.pick(resolution)
    assert client is not None
    assert [c.name for c in client.candidates] == ["qwen3-a-driver"]


def test_a_driver_following_no_runtime_is_eligible_when_reachable() -> None:
    cloud = FakeDriverClient(
        name="claude", base_url="http://c", model_id=CLOUD, provider="claude_subscription"
    )
    table = make_routing_table(cloud)
    assert [x.name for x in table.resolve(CLOUD).eligible_backends()] == ["claude"]


def test_a_driver_following_an_unknown_runtime_is_routed_to_on_faith() -> None:
    orphan = FakeDriverClient(name="orphan", base_url="http://o", model_id=LOCAL, runtime="gone")
    table = make_routing_table(orphan)
    assert table.resolve(LOCAL).eligible_backends()[0].name == "orphan"


def test_all_replicas_asleep_is_a_503_not_a_404(settings: Settings) -> None:
    a, b = _replicas()
    table = make_routing_table(
        a,
        b,
        runtimes=[
            runtime_facts("qwen3-a", status="stopped"),
            runtime_facts("qwen3-b", status="loading"),
        ],
    )
    with _app(settings, table) as c:
        response = c.post("/v1/chat/completions", json=_chat())
    assert response.status_code == 503, response.text
    message = response.json()["error"]["message"]
    assert "qwen3-a=stopped" in message and "qwen3-b=loading" in message
    assert a.calls == [] and b.calls == []


# --- balancing ----------------------------------------------------------------


def test_idle_replicas_alternate() -> None:
    a, b = _replicas()
    table = make_routing_table(a, b, runtimes=[runtime_facts("qwen3-a"), runtime_facts("qwen3-b")])
    firsts = [table.pick(table.resolve(LOCAL)).candidates[0].name for _ in range(4)]  # type: ignore[union-attr]
    assert firsts == ["qwen3-a-driver", "qwen3-b-driver", "qwen3-a-driver", "qwen3-b-driver"]


def test_least_busy_prefers_the_replica_with_fewer_requests_in_flight() -> None:
    a, b = _replicas()
    table = make_routing_table(a, b, runtimes=[runtime_facts("qwen3-a"), runtime_facts("qwen3-b")])
    # Two requests parked on a.
    table.on_attempt_start("qwen3-a-driver")
    table.on_attempt_start("qwen3-a-driver")
    firsts = {table.pick(table.resolve(LOCAL)).candidates[0].name for _ in range(4)}  # type: ignore[union-attr]
    assert firsts == {"qwen3-b-driver"}
    # And the rest of the tier is still there to cascade to.
    assert [c.name for c in table.pick(table.resolve(LOCAL)).candidates] == [  # type: ignore[union-attr]
        "qwen3-b-driver",
        "qwen3-a-driver",
    ]


def test_capacity_weights_the_load() -> None:
    a, b = _replicas()
    table = make_routing_table(
        a,
        b,
        runtimes=[
            runtime_facts("qwen3-a", parallel_slots=4),
            runtime_facts("qwen3-b", parallel_slots=1),
        ],
    )
    # One request each: a is at 1/4, b at 1/1 — a is the less loaded.
    table.on_attempt_start("qwen3-a-driver")
    table.on_attempt_start("qwen3-b-driver")
    firsts = {table.pick(table.resolve(LOCAL)).candidates[0].name for _ in range(4)}  # type: ignore[union-attr]
    assert firsts == {"qwen3-a-driver"}


def test_round_robin_ignores_load() -> None:
    a, b = _replicas()
    table = make_routing_table(
        a, b, runtimes=[runtime_facts("qwen3-a"), runtime_facts("qwen3-b")], strategy="round_robin"
    )
    table.on_attempt_start("qwen3-a-driver")
    table.on_attempt_start("qwen3-a-driver")
    firsts = [table.pick(table.resolve(LOCAL)).candidates[0].name for _ in range(4)]  # type: ignore[union-attr]
    assert firsts == ["qwen3-a-driver", "qwen3-b-driver", "qwen3-a-driver", "qwen3-b-driver"]


@pytest.mark.anyio
async def test_in_flight_counts_rise_and_fall_around_a_request() -> None:
    a, b = _replicas()
    table = make_routing_table(a, b, runtimes=[runtime_facts("qwen3-a"), runtime_facts("qwen3-b")])
    client = table.pick(table.resolve(LOCAL))
    assert client is not None
    served = client.candidates[0].name
    await client.generate(_generate())
    # Keyed by `(node, name)` since R1.6; single-host here, so `None`.
    assert table.inflight((None, served)) == 0
    runtime = (None, "qwen3-a" if served == "qwen3-a-driver" else "qwen3-b")
    assert table.runtime_inflight(runtime) == 0
    assert table.idle_seconds(runtime) is not None
    assert table.idle_seconds(runtime) < 5  # type: ignore[operator]


def _generate():  # type: ignore[no-untyped-def]
    from eugene_plexus_gateway._generated.driver_models import GenerateRequest

    return GenerateRequest(messages=[{"role": "user", "content": "hi"}])


# --- slots and tiers -----------------------------------------------------------


def test_an_unconfigured_model_is_a_one_tier_slot() -> None:
    a, b = _replicas()
    table = make_routing_table(a, b, runtimes=[runtime_facts("qwen3-a"), runtime_facts("qwen3-b")])
    resolution = table.resolve(LOCAL)
    assert resolution.configured is False
    assert [t.target for t in resolution.tiers] == [LOCAL]


def test_a_configured_slot_adds_tiers_in_order() -> None:
    a, b = _replicas()
    cloud = FakeDriverClient(name="claude", base_url="http://c", model_id=CLOUD)
    table = make_routing_table(
        a,
        b,
        cloud,
        runtimes=[runtime_facts("qwen3-a"), runtime_facts("qwen3-b")],
        slots=[{"model": "coder", "targets": [LOCAL, CLOUD]}],
    )
    resolution = table.resolve("coder")
    assert resolution.configured is True
    assert [t.target for t in resolution.tiers] == [LOCAL, CLOUD]
    assert [[x.name for x in t.backends] for t in resolution.tiers] == [
        ["qwen3-a-driver", "qwen3-b-driver"],
        ["claude"],
    ]
    # A virtual alias is discoverable, or nobody could ask for it.
    assert "coder" in table.known_models()


def test_a_real_alias_can_be_given_a_fallback() -> None:
    a, b = _replicas()
    cloud = FakeDriverClient(name="claude", base_url="http://c", model_id=CLOUD)
    table = make_routing_table(
        a,
        b,
        cloud,
        runtimes=[runtime_facts("qwen3-a"), runtime_facts("qwen3-b")],
        slots=[{"model": LOCAL, "targets": [CLOUD]}],
    )
    assert [t.target for t in table.resolve(LOCAL).tiers] == [LOCAL, CLOUD]


def test_an_empty_tier_is_kept_so_the_ones_after_it_keep_their_number(
    settings: Settings,
) -> None:
    """Reverses an earlier decision, deliberately.

    Dropping a tier whose target nothing serves renumbered every tier
    after it, so a slot configured `[nope, cloud]` reported the cloud
    fallback as `tier: 1` - telling an operator whose primary was never
    launched that the primary had served the request. `TieredClient` was
    already written the other way, and the contract's `tier` description
    says positional. Routing order is identical either way; only the
    label moved.
    """
    cloud = FakeDriverClient(name="claude", base_url="http://c", model_id=CLOUD)
    table = make_routing_table(cloud, slots=[{"model": "coder", "targets": ["nope", CLOUD]}])
    resolution = table.resolve("coder")
    # `coder` is a virtual alias nothing serves directly, so its implicit
    # self-tier is absent - but `nope`, which the operator DID list, is
    # kept as an empty tier so the cloud target stays tier 2.
    assert [t.target for t in resolution.tiers] == ["nope", CLOUD]
    assert [[x.name for x in t.backends] for t in resolution.tiers] == [[], ["claude"]]

    # Still a 404 when nothing in the slot resolves: `has_backends` asks
    # whether any tier HAS backends, not whether any tier exists.
    with _app(
        settings, make_routing_table(cloud, slots=[{"model": "ghost", "targets": ["nope"]}])
    ) as c:
        assert c.post("/v1/chat/completions", json=_chat("ghost")).status_code == 404


def test_a_fallback_after_a_never_launched_primary_reports_its_real_tier(
    settings: Settings,
) -> None:
    """The defect this reversal exists for, end to end.

    Same operator config and same backend answering as when the primary
    is merely asleep - which already reported tier 2 correctly, because
    M6 keeps a stopped runtime's companion driver alive. The two cases
    disagreed only because one target had a driver and the other had
    none.
    """
    cloud = FakeDriverClient(name="claude", base_url="http://c", model_id=CLOUD)
    table = make_routing_table(
        cloud, slots=[{"model": "chat", "targets": ["never-launched", CLOUD]}]
    )
    with _app(settings, table) as c:
        body = c.post("/v1/chat/completions", json=_chat("chat")).json()
    info = body["x_eugene_plexus"]
    assert info["driver"] == "claude"
    # Second of the operator's two targets, and it says so.
    assert info["tier"] == 2, "a fallback still reported itself as the primary"


def test_the_self_tier_is_kept_when_the_install_declares_that_model(
    settings: Settings,
) -> None:
    """R3 item 3 (review 6.2 #20): the surviving second case.

    The 2026-09-10 fix carved the slot's own name out of "keep every
    tier", and that carve-out is right for a **virtual alias** — a name
    nothing is ever launched under, where an empty self tier would push
    every configured target up a number. It is wrong for the shape the UI
    actually produces, `{model: <a real local model>, targets: [cloud]}`:
    there the self tier is the primary, and dropping it when the
    companion driver happens to be down at refresh makes the cloud answer
    `tier: 1` — *the primary served this*, which is the one question
    tiered failover exists to answer, answered backwards.

    **The two are distinguishable and the snapshot already holds the
    discriminator.** `_Snapshot.runtimes` is every runtime the install
    DECLARES, read per node from each agent, independent of whether
    anything is advertising it right now. A name some runtime carries is
    a real primary; a name no runtime carries is an alias.

    Note what is NOT the discriminator: `by_model`. That is what is being
    advertised at this instant, and being empty is precisely the
    condition under test.
    """
    cloud = FakeDriverClient(name="claude", base_url="http://c", model_id=CLOUD)
    table = make_routing_table(
        cloud,
        # Declared, ready, and its companion driver is not answering — so
        # nothing advertises LOCAL at this refresh.
        runtimes=[runtime_facts("qwen3-a", alias=LOCAL)],
        unreachable={"qwen3-a-driver": "connection refused"},
        slots=[{"model": LOCAL, "targets": [CLOUD]}],
    )
    resolution = table.resolve(LOCAL)
    assert [t.target for t in resolution.tiers] == [LOCAL, CLOUD]
    assert [[x.name for x in t.backends] for t in resolution.tiers] == [[], ["claude"]]

    with _app(settings, table) as c:
        body = c.post("/v1/chat/completions", json=_chat(LOCAL)).json()
    info = body["x_eugene_plexus"]
    assert info["driver"] == "claude"
    assert info["tier"] == 2, "the fallback reported itself as the primary"


def test_the_self_tier_is_kept_for_a_runtime_known_only_by_its_name() -> None:
    """`modelAlias` is optional, so the name is the other half of the rule.

    A runtime declared without one still produces a companion driver that
    advertises something, and on this install that something is the
    runtime's own name. Matching only on `alias` would drop the self tier
    for every runtime whose operator never typed an alias, which is the
    same defect for a different reason.
    """
    cloud = FakeDriverClient(name="claude", base_url="http://c", model_id=CLOUD)
    table = make_routing_table(
        cloud,
        runtimes=[runtime_facts("coder-30b", alias=None)],
        slots=[{"model": "coder-30b", "targets": [CLOUD]}],
    )
    assert [t.target for t in table.resolve("coder-30b").tiers] == ["coder-30b", CLOUD]


def test_the_self_tier_is_kept_for_a_primary_on_another_machine() -> None:
    """The question is install-wide, and a sabotage pass is what said so.

    Narrowing `_declares_runtime` to this node escaped every other check
    here, because none of them put the primary anywhere else. That is
    precisely the install R1.6 exists for: one model, two machines. The
    gateway routes across all of them, so a slot's primary being declared
    on `box-b` makes it a primary — and if the console happens to be
    talking to a gateway whose own node declares nothing, the answer must
    not change.
    """
    cloud = FakeDriverClient(name="claude", base_url="http://c", model_id=CLOUD)
    table = make_routing_table(
        cloud,
        runtimes=[runtime_facts("qwen3-a", alias=LOCAL, node="box-b")],
        slots=[{"model": LOCAL, "targets": [CLOUD]}],
    )
    assert [t.target for t in table.resolve(LOCAL).tiers] == [LOCAL, CLOUD]


def test_a_virtual_alias_still_has_no_self_tier() -> None:
    """The half the 2026-09-10 carve-out got right, kept.

    `chat` is not a runtime anywhere in the install, so its implicit self
    tier is not a primary that failed — it is nothing at all, and keeping
    it would renumber the operator's own two targets.
    """
    cloud = FakeDriverClient(name="claude", base_url="http://c", model_id=CLOUD)
    table = make_routing_table(
        cloud,
        runtimes=[runtime_facts("qwen3-a", alias=LOCAL)],
        slots=[{"model": "chat", "targets": ["never-launched", CLOUD]}],
    )
    assert [t.target for t in table.resolve("chat").tiers] == ["never-launched", CLOUD]


def test_the_cascade_walks_the_tier_then_the_next_tier(settings: Settings) -> None:
    a, b = _replicas()
    a.generate_error = httpx.ConnectError("refused")
    b.generate_error = httpx.ConnectError("refused")
    cloud = FakeDriverClient(name="claude", base_url="http://c", model_id=CLOUD)
    cloud.responses = ["from the cloud"]
    table = make_routing_table(
        a,
        b,
        cloud,
        runtimes=[runtime_facts("qwen3-a"), runtime_facts("qwen3-b")],
        slots=[{"model": "coder", "targets": [LOCAL, CLOUD]}],
    )
    with _app(settings, table) as c:
        response = c.post("/v1/chat/completions", json=_chat("coder"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "from the cloud"
    info = body["x_eugene_plexus"]
    assert info["attempts"] == 3
    assert info["tier"] == 2
    assert info["driver"] == "claude"
    assert info["swapped_in"] is False
    assert info["waited_ms"] == 0
    # The response names what answered, which is not what was asked for.
    assert body["model"] == CLOUD


def test_a_healthy_first_tier_never_reaches_the_second(settings: Settings) -> None:
    a, b = _replicas()
    a.responses = ["local"]
    b.responses = ["local"]
    cloud = FakeDriverClient(name="claude", base_url="http://c", model_id=CLOUD)
    table = make_routing_table(
        a,
        b,
        cloud,
        runtimes=[runtime_facts("qwen3-a"), runtime_facts("qwen3-b")],
        slots=[{"model": "coder", "targets": [LOCAL, CLOUD]}],
    )
    with _app(settings, table) as c:
        info = c.post("/v1/chat/completions", json=_chat("coder")).json()["x_eugene_plexus"]
    assert info["tier"] == 1
    assert cloud.calls == []


def test_a_stopped_first_tier_with_no_wake_falls_to_the_second(settings: Settings) -> None:
    """Without a lifecycle manager (or with nothing startable) an
    ineligible tier is simply skipped."""
    a, b = _replicas()
    cloud = FakeDriverClient(name="claude", base_url="http://c", model_id=CLOUD)
    cloud.responses = ["cloud"]
    table = make_routing_table(
        a,
        b,
        cloud,
        runtimes=[
            runtime_facts("qwen3-a", status="stopped"),
            runtime_facts("qwen3-b", status="stopped"),
        ],
        slots=[{"model": "coder", "targets": [LOCAL, CLOUD]}],
    )
    with _app(settings, table) as c:
        info = c.post("/v1/chat/completions", json=_chat("coder")).json()["x_eugene_plexus"]
    assert info["tier"] == 2 and info["attempts"] == 1


# --- attribution by name -------------------------------------------------------


def test_replicas_are_attributed_by_name(settings: Settings) -> None:
    a, b = _replicas()
    table = make_routing_table(a, b, runtimes=[runtime_facts("qwen3-a"), runtime_facts("qwen3-b")])
    with _app(settings, table) as c:
        seen = {
            c.post("/v1/chat/completions", json=_chat()).json()["x_eugene_plexus"]["runtime"]
            for _ in range(4)
        }
    assert seen == {"qwen3-a", "qwen3-b"}


# --- the model list and the routing view -------------------------------------


def test_the_model_list_reports_tiers_readiness_and_on_demand(settings: Settings) -> None:
    a, b = _replicas()
    cloud = FakeDriverClient(name="claude", base_url="http://c", model_id=CLOUD)
    table = make_routing_table(
        a,
        b,
        cloud,
        runtimes=[
            runtime_facts("qwen3-a", status="stopped", start_on_demand=True, stop_reason="idle"),
            runtime_facts("qwen3-b", status="stopped", start_on_demand=True, stop_reason="idle"),
        ],
        slots=[{"model": "coder", "targets": [LOCAL, CLOUD]}],
    )
    with _app(settings, table) as c:
        data = {m["id"]: m for m in c.get("/v1/models").json()["data"]}
    assert set(data) == {LOCAL, CLOUD, "coder"}
    local = data[LOCAL]["x_eugene_plexus"]
    assert local["tiers"] == [["qwen3-a-driver", "qwen3-b-driver"]]
    assert local["ready_backends"] == 0
    assert local["on_demand"] is True
    coder = data["coder"]["x_eugene_plexus"]
    assert coder["tiers"] == [["qwen3-a-driver", "qwen3-b-driver"], ["claude"]]
    assert coder["ready_backends"] == 1
    assert coder["on_demand"] is False


def test_the_routing_view_opens_the_table_up(settings: Settings) -> None:
    a, b = _replicas()
    # **On the same machine as its runtime, which is the only shape that
    # exists.** Before the fixture was keyed `(node, name)` this test had
    # the runtime on `box` and its driver nowhere, and the bare-name join
    # married them anyway — the exact cross-node join R1.6 removed from
    # production, still being asserted here.
    a.node = "box"
    table = make_routing_table(
        a,
        b,
        runtimes=[
            runtime_facts("qwen3-a", node="box", idle_unload_seconds=600, parallel_slots=2),
            runtime_facts("qwen3-b", status="stopped", stop_reason="idle", start_on_demand=True),
        ],
        unreachable={"dead": "connection refused"},
    )
    # The node belongs here too: R1.6 keys every in-flight counter by
    # `(node, driver)`, and a bare name counts against a different bucket
    # than the one this backend reads.
    table.on_attempt_start("qwen3-a-driver", node="box")
    with _app(settings, table) as c:
        view = c.get("/v1/admin/routing").json()
    assert view["load_balancing"] == "least_busy"
    assert view["unreachable_drivers"] == ["dead"]
    [slot] = view["slots"]
    assert slot["model"] == LOCAL and slot["configured"] is False
    [tier] = slot["tiers"]
    by_name = {x["driver"]: x for x in tier["backends"]}
    assert by_name["qwen3-a-driver"]["eligible"] is True
    assert by_name["qwen3-a-driver"]["in_flight"] == 1
    assert by_name["qwen3-a-driver"]["parallel_slots"] == 2
    assert by_name["qwen3-a-driver"]["node"] == "box"
    assert by_name["qwen3-a-driver"]["idle_unload_seconds"] == 600
    assert by_name["qwen3-b-driver"]["eligible"] is False
    assert "stopped" in by_name["qwen3-b-driver"]["ineligible_reason"]
    assert by_name["qwen3-b-driver"]["stop_reason"] == "idle"
    assert by_name["qwen3-b-driver"]["start_on_demand"] is True


def test_the_routing_view_is_503_in_safe_mode(settings: Settings) -> None:
    app = create_app(settings=Settings(config_file=settings.config_file, safe_mode=True))
    with TestClient(app) as c:
        assert c.get("/v1/admin/routing").status_code == 503


# --- config validation ---------------------------------------------------------


def test_model_slots_are_validated_per_entry(client: TestClient) -> None:
    ok = client.patch(
        "/v1/config", json={"modelSlots": [{"model": "coder", "targets": [LOCAL, CLOUD]}]}
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["applied"] == ["modelSlots"]
    assert client.get("/v1/config").json()["modelSlots"] == [
        {"model": "coder", "targets": [LOCAL, CLOUD]}
    ]

    bad = client.patch("/v1/config", json={"modelSlots": [{"model": "coder", "targets": []}]})
    assert bad.json()["rejected"][0]["message"].startswith("entry 0 ('coder'): `targets`")
    twice = client.patch(
        "/v1/config",
        json={"modelSlots": [{"model": "x", "targets": ["a"]}, {"model": "x", "targets": ["b"]}]},
    )
    assert "listed twice" in twice.json()["rejected"][0]["message"]
    shape = client.patch("/v1/config", json={"modelSlots": {"model": "x"}})
    assert "expected a list" in shape.json()["rejected"][0]["message"]
    unknown = client.patch(
        "/v1/config", json={"modelSlots": [{"model": "x", "targets": ["a"], "drivers": []}]}
    )
    assert "unknown key" in unknown.json()["rejected"][0]["message"]


def test_load_balancing_is_an_enum(client: TestClient) -> None:
    assert client.patch("/v1/config", json={"loadBalancing": "round_robin"}).json()["applied"] == [
        "loadBalancing"
    ]
    assert client.patch("/v1/config", json={"loadBalancing": "random"}).json()["rejected"]


# --- refresh on demand -----------------------------------------------------------


def test_a_request_that_finds_nothing_eligible_refreshes_before_answering(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first live run's seam: the snapshot said `loading` for up to a
    refresh interval after the agent said `ready`, and every request in
    that window was a 503. Now a request that finds nothing eligible
    pays for one refresh before concluding anything."""
    from datetime import UTC, datetime, timedelta

    from .conftest import install_snapshot

    a, b = _replicas()
    a.responses = ["fresh"]
    table = make_routing_table(
        a,
        b,
        runtimes=[
            runtime_facts("qwen3-a", status="loading"),
            runtime_facts("qwen3-b", status="loading"),
        ],
    )
    table._snapshot.refreshed_at = datetime.now(UTC) - timedelta(seconds=10)
    refreshed: list[bool] = []

    async def refresh() -> None:
        refreshed.append(True)
        install_snapshot(table, a, b, runtimes=[runtime_facts("qwen3-a"), runtime_facts("qwen3-b")])

    monkeypatch.setattr(table, "refresh", refresh)
    with _app(settings, table) as c:
        response = c.post("/v1/chat/completions", json=_chat())
    assert response.status_code == 200, response.text
    assert refreshed == [True]
    assert response.json()["x_eugene_plexus"]["swapped_in"] is False


def test_a_young_snapshot_is_not_refreshed_again(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = _replicas()
    table = make_routing_table(
        a,
        b,
        runtimes=[
            runtime_facts("qwen3-a", status="loading"),
            runtime_facts("qwen3-b", status="loading"),
        ],
    )
    calls: list[bool] = []

    async def refresh() -> None:
        calls.append(True)

    monkeypatch.setattr(table, "refresh", refresh)
    with _app(settings, table) as c:
        response = c.post("/v1/chat/completions", json=_chat())
    # Freshly installed snapshot: younger than the floor, so no refresh,
    # and the honest 503.
    assert response.status_code == 503
    assert calls == []
