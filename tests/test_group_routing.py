"""Row 37: a verified person's routing by role and model list, from their group.

The root's admission answer carries `roleRouting` and the `routingGroup` it
comes from (the person's active group), and `modelLimits`, the model list of
every group of theirs that sets one (call R37-SELF, Choice 3). The gateway checks
every model a request can reach against the key's list and the person's,
fallbacks and server-run tools included, and lists only those.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.model_patterns import ModelScope, permits
from tests.conftest import FakeDriverClient, make_routing_table
from tests.test_admission import Authority

PROOF = "PRIVATE_PROOF"
SALES = {"id": "g1", "name": "Sales"}


class GroupAuthority(Authority):
    """The root as the gateway meets it, for a person in group Sales."""

    def __init__(self) -> None:
        super().__init__()
        self.allowed = None  # the key itself permits every model
        self.person: dict[str, Any] = {
            "roleRouting": {"chat": ["m2", "m1"]},
            "routingGroup": SALES,
            "modelLimits": [{"group": SALES, "models": ["m1", "m2"]}],
        }

    def _handle(self, request: httpx.Request) -> httpx.Response:
        response = super()._handle(request)
        if not request.url.path.endswith("/admission") or response.status_code != 200:
            return response
        body = json.loads(request.content)
        answer = response.json()
        if answer.get("limits", {}).get("allowedModels") is None:
            answer["limits"].pop("allowedModels", None)
        if body.get("personToken") == PROOF:
            answer |= {"personId": "ada", "appClientId": "workbench", **self.person}
        return httpx.Response(200, json=answer)


@pytest.fixture
def group_setup(settings, install):
    authority = GroupAuthority()
    token = install.client_key(name="app:workbench@node", jti="key-1")
    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    drivers = {m: FakeDriverClient(name=f"{m}-driver", model_id=m) for m in ("m1", "m2", "m3")}
    app.state.routing = make_routing_table(
        *drivers.values(),
        slots=[
            {"model": "role:chat", "targets": ["m3", "m1"]},
            {"model": "role:agentic", "targets": ["m3", "m1"]},
            {"model": "role:vision", "targets": ["m3"]},
        ],
    )
    person = {"Authorization": "Bearer " + token, "X-Eugene-Person-Token": PROOF}
    keyed = {"Authorization": "Bearer " + token}
    return app, authority, drivers, person, keyed


def chat(c: TestClient, headers: dict[str, str], model: str) -> httpx.Response:
    return c.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "PRIVATE_PROMPT"}]},
        headers=headers,
    )


def served(drivers: dict[str, FakeDriverClient]) -> dict[str, int]:
    return {m: len(d.calls) for m, d in drivers.items()}


def test_a_role_resolves_through_the_persons_group_routing(group_setup) -> None:
    app, _, drivers, person, keyed = group_setup
    with TestClient(app) as c:
        answer = chat(c, person, "role:chat")
        assert answer.status_code == 200, answer.text
        assert served(drivers) == {"m1": 0, "m2": 1, "m3": 0}
        # The group's list fails over in its own order.
        drivers["m2"].generate_error = httpx.ConnectError("m2 is down")
        assert chat(c, person, "role:chat").status_code == 200
        assert served(drivers) == {"m1": 1, "m2": 2, "m3": 0}
        # A key with no person: the install's slot, as before row 37.
        assert chat(c, keyed, "role:chat").status_code == 200
        assert drivers["m3"].calls, "the install's routing reaches m3"


def test_a_role_the_group_does_not_route_uses_the_install_list_filtered(group_setup) -> None:
    """G1 routes no `agentic`; the install's list is m3, m1: m3 is dropped,
    not tried, and m1 answers."""
    app, _, drivers, person, _ = group_setup
    with TestClient(app) as c:
        answer = chat(c, person, "role:agentic")
        assert answer.status_code == 200, answer.text
        assert served(drivers) == {"m1": 1, "m2": 0, "m3": 0}


def test_a_group_list_with_no_allowed_model_falls_back_to_the_install_list(group_setup) -> None:
    app, authority, drivers, person, _ = group_setup
    authority.person["roleRouting"] = {"chat": ["m3"]}
    with TestClient(app) as c:
        assert chat(c, person, "role:chat").status_code == 200
        assert served(drivers) == {"m1": 1, "m2": 0, "m3": 0}


def test_a_fallback_outside_the_groups_models_is_never_reached(group_setup) -> None:
    app, _, drivers, person, _ = group_setup
    drivers["m1"].generate_error = httpx.ConnectError("m1 is down")
    with TestClient(app) as c:
        answer = chat(c, person, "role:agentic")
        assert answer.status_code == 502, answer.text
        assert not drivers["m3"].calls, "m3 is outside Sales's models"


def test_a_model_outside_the_group_is_refused_naming_the_group(group_setup) -> None:
    app, _, drivers, person, keyed = group_setup
    with TestClient(app) as c:
        refused = chat(c, person, "m3")
        assert refused.status_code == 403, refused.text
        error = refused.json()["error"]
        assert error["code"] == "model_not_allowed"
        assert error["message"] == (
            "Your group Sales does not include m3. Ask the owner, or choose another model."
        )
        assert not drivers["m3"].calls
        # The same key without a person reaches it.
        assert chat(c, keyed, "m3").status_code == 200


def test_a_role_with_nothing_allowed_names_the_role(group_setup) -> None:
    app, _, drivers, person, _ = group_setup
    with TestClient(app) as c:
        refused = chat(c, person, "role:vision")
        assert refused.status_code == 404, refused.text
        assert refused.json()["error"]["message"] == (
            "No model your groups allow has the role Sees images."
        )
        assert not drivers["m3"].calls


def test_models_listing_is_filtered_for_a_person(group_setup) -> None:
    app, _, _, person, keyed = group_setup
    with TestClient(app) as c:
        mine = {m["id"] for m in c.get("/v1/models", headers=person).json()["data"]}
        assert mine == {"m1", "m2", "role:chat", "role:agentic"}
        everyone = {m["id"] for m in c.get("/v1/models", headers=keyed).json()["data"]}
        assert everyone == {"m1", "m2", "m3", "role:chat", "role:agentic", "role:vision"}


def test_a_role_only_the_group_routes_is_listed_and_served(group_setup) -> None:
    app, authority, drivers, person, _ = group_setup
    authority.person["roleRouting"] = {"rerank": ["m2"], "chat": ["m1"]}
    with TestClient(app) as c:
        mine = {m["id"] for m in c.get("/v1/models", headers=person).json()["data"]}
        assert "role:chat" in mine
        assert chat(c, person, "role:chat").status_code == 200
        assert served(drivers) == {"m1": 1, "m2": 0, "m3": 0}


def test_the_key_and_the_person_must_both_allow_a_model(group_setup) -> None:
    app, authority, drivers, person, _ = group_setup
    authority.allowed = ["m2", "role:*"]
    with TestClient(app) as c:
        assert chat(c, person, "role:chat").status_code == 200
        drivers["m2"].generate_error = httpx.ConnectError("m2 is down")
        # m1 is the person's but not the key's: never tried.
        assert chat(c, person, "role:chat").status_code == 502
        assert not drivers["m1"].calls and not drivers["m3"].calls
        mine = {m["id"] for m in c.get("/v1/models", headers=person).json()["data"]}
        assert mine == {"m2", "role:chat"}


def test_a_person_with_no_group_limit_keeps_the_install_routing(group_setup) -> None:
    app, authority, drivers, person, _ = group_setup
    authority.person = {}
    with TestClient(app) as c:
        assert chat(c, person, "role:chat").status_code == 200
        assert served(drivers) == {"m1": 0, "m2": 0, "m3": 1}


def test_a_scope_permits_only_what_both_lists_allow() -> None:
    scope = ModelScope(["openrouter/*", "m1", "role:chat"], [("Sales", ["m1", "qwen3-*"])])
    assert permits(scope, "m1")
    assert not permits(scope, "openrouter/qwen3-8b"), "the person's list leaves it out"
    assert not permits(scope, "qwen3-8b"), "the key's list leaves it out"
    # A role name is checked against the key only: its models are checked one by one.
    assert permits(scope, "role:chat") and not permits(scope, "role:vision")
    assert permits(ModelScope(None, [("Sales", ["m1"])]), "role:vision")
    assert not permits(ModelScope(None, [("Nothing", [])]), "m1")


def test_a_routing_change_while_waiting_is_a_policy_change() -> None:
    """A request routed under the person's group list that changed before it
    left the line is refused at its acquire (row 11 blocker 1), as for the
    key's limits; the band is not part of it."""
    from eugene_plexus_gateway.admission import bound_policy

    checked = {
        "limits": {"maxConcurrentRequests": 1},
        "modelLimits": [{"group": SALES, "models": ["m1"]}],
    }
    wider = [{"group": SALES, "models": ["m1", "m3"]}]
    assert bound_policy(checked) != bound_policy({**checked, "modelLimits": wider})
    assert bound_policy(checked) != bound_policy({**checked, "roleRouting": {"chat": ["m1"]}})
    assert bound_policy(checked) == bound_policy({**checked, "queueBand": "background"})
    plain = {"limits": {"maxConcurrentRequests": 1}}
    assert bound_policy(plain) == {"maxConcurrentRequests": 1}, "unchanged without a person"


def test_every_groups_list_applies_whichever_group_routes(group_setup) -> None:
    """R37-SELF, Choice 3 (default built): a person in Sales (m1, m2) and
    Night (m2, m3) reaches only m2, and the refusal names the group that
    leaves the model out."""
    app, authority, drivers, person, _ = group_setup
    night = {"id": "g2", "name": "Night"}
    authority.person = {
        "roleRouting": {"chat": ["m3", "m2"]},
        "routingGroup": night,
        "modelLimits": [
            {"group": night, "models": ["m2", "m3"]},
            {"group": SALES, "models": ["m1", "m2"]},
        ],
    }
    with TestClient(app) as c:
        assert chat(c, person, "role:chat").status_code == 200
        assert served(drivers) == {"m1": 0, "m2": 1, "m3": 0}
        refused = chat(c, person, "m3")
        assert refused.status_code == 403
        assert "Your group Sales does not include m3" in refused.json()["error"]["message"]
        refused = chat(c, person, "m1")
        assert "Your group Night does not include m1" in refused.json()["error"]["message"]
        mine = {m["id"] for m in c.get("/v1/models", headers=person).json()["data"]}
        assert mine == {"m2", "role:chat"}


def test_the_responses_door_says_model_not_allowed(group_setup) -> None:
    app, _, drivers, person, _ = group_setup
    with TestClient(app) as c:
        refused = c.post(
            "/v1/responses", json={"model": "m3", "input": "PRIVATE_PROMPT"}, headers=person
        )
        assert refused.status_code == 403, refused.text
        assert refused.json()["error"]["code"] == "model_not_allowed"
        assert "Your group Sales does not include m3" in refused.json()["error"]["message"]
        assert not drivers["m3"].calls


def test_a_pattern_is_matched_in_linear_time() -> None:
    """Row 37 review S1: a regular expression of `.*` took a minute on a
    26-character pattern of many stars against a 40-character name, and
    4.8 s for a 256 KB name against `*qwen*coder*`, on the event loop."""
    import time

    from eugene_plexus_gateway.model_patterns import matches

    began = time.perf_counter()
    assert not matches("*a" * 12 + "*b", "a" * 40)
    assert not matches("*qwen*coder*", "x" * 262144)
    assert time.perf_counter() - began < 1.0
    for pattern, value, expected in (
        ("openrouter/*", "openrouter/anthropic/claude", True),
        ("openrouter/*", "openrouter", False),
        ("a*a", "a", False),
        ("a*a", "aa", True),
        ("*qwen*coder*", "x-qwen3-coder-30b", True),
        ("*qwen*coder*", "coder-qwen", False),
        ("**", "", True),
    ):
        assert matches(pattern, value) is expected, (pattern, value)


def test_a_name_over_256_characters_is_never_matched(group_setup) -> None:
    app, _, drivers, person, _ = group_setup
    scope = ModelScope(None, [("Everyone", ["*"])])
    assert permits(scope, "m" * 256) and not permits(scope, "m" * 257)
    assert not permits(["*"], "m" * 257)
    with TestClient(app) as c:
        answer = chat(c, person, "m" * 300)
        assert answer.status_code in (400, 403, 404), answer.text
        assert not any(d.calls for d in drivers.values())


def test_an_older_agent_is_named_when_it_fails_a_persons_check(group_setup) -> None:
    """Row 37 review: an agent older than its root fails (500) on an answer
    carrying a person's group lists; the gateway says to update it."""
    app, authority, drivers, person, keyed = group_setup
    authority.refuse_acquire = None
    original = authority._handle

    def broken(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/admission") and b"personToken" in request.content:
            return httpx.Response(500)
        return original(request)

    authority._handle = broken  # type: ignore[method-assign]
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    with TestClient(app) as c:
        refused = chat(c, person, "role:chat")
        assert refused.status_code == 503
        message = refused.json()["error"]["message"]
        assert "answered 500" in message and "update it" in message
        plain = chat(c, keyed, "role:chat")
        assert plain.status_code == 200
        assert not any(d.calls for d in (drivers["m1"], drivers["m2"]))
