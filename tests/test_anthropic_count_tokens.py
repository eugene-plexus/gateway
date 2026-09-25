"""`POST /v1/messages/count_tokens`, 2026-09-23.

**Measured, not assumed.** Claude Code's `/context` calls this endpoint
13-14 times, one body per category -- `{model, messages, system?,
tools?}`, no `max_tokens` -- and against this gateway, which had no such
route, each 404 was followed at once by a real `/v1/messages` with
`max_tokens: 1`: the client counts by generating. One diagnostic command
cost fourteen prefills (up to 9,050 tokens, 17.8 s apiece on a CPU),
would wake an idle model, and left fourteen rows in the metrics.

So the count is answered by the backend that would serve -- its own
template and tokenizer, through the driver's `/v1/generate/count` --
and when that is not possible it is a **400**, because (measured)
`/context` tries a 4xx once and falls back, and retries a 5xx first.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from eugene_plexus_gateway._generated.driver_models import Problem
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import DriverError
from eugene_plexus_gateway.settings import Settings
from tests.test_admission import Authority

from .conftest import FakeDriverClient, FakeInstall, make_routing_table, runtime_facts

MODEL = "qwen3-0.6b"
PATH = "/v1/messages/count_tokens?beta=true"


def captured(**overrides: Any) -> dict[str, Any]:
    """One of `/context`'s bodies: a system category, trimmed but not tidied."""
    body: dict[str, Any] = {
        "model": MODEL,
        "system": [{"type": "text", "text": "You are a Claude agent."}],
        "messages": [{"role": "user", "content": "x"}],
    }
    body.update(overrides)
    return body


def counting(name: str = "d", *, count: int | None = 1234, **kw: Any) -> FakeDriverClient:
    driver = FakeDriverClient(name=name, model_id=kw.pop("model_id", MODEL), **kw)
    driver.prompt_tokens = count
    return driver


def serve(settings: Settings, *drivers: FakeDriverClient, **table: Any) -> TestClient:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*drivers, **table)
    return TestClient(app)


class WakeSpy:
    def __init__(self) -> None:
        self.woken = 0

    async def wake(self, resolution: Any) -> Any:  # pragma: no cover - must not run
        self.woken += 1
        raise AssertionError("a count must never wake a model")


# --------------------------------------------------------------------------- #
# The reproduction
# --------------------------------------------------------------------------- #


def test_a_count_is_answered_by_the_serving_backend_and_generates_nothing(
    settings: Settings,
) -> None:
    driver = counting(count=4242)
    with serve(settings, driver) as client:
        r = client.post(PATH, json=captured())
    assert r.status_code == 200, r.text
    assert r.json() == {"input_tokens": 4242}
    assert not driver.calls, "a count must never generate"
    # The driver was handed the request `/v1/messages` would have sent.
    [counted] = driver.count_calls
    roles = [m.role.value for m in counted.messages]
    assert roles == ["system", "user"]
    assert counted.messages[0].content == "You are a Claude agent."


def test_a_tools_category_is_counted_with_its_tools(settings: Settings) -> None:
    driver = counting(supports_tools=True)
    tools = [{"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}}]
    with serve(settings, driver) as client:
        r = client.post(PATH, json=captured(tools=tools))
    assert r.status_code == 200, r.text
    [counted] = driver.count_calls
    assert [t.function.name for t in counted.tools] == ["Read"]


def test_a_count_is_not_recorded_as_inference(settings: Settings) -> None:
    driver = counting()
    with serve(settings, driver) as client:
        client.post(PATH, json=captured())
        rows = client.get("/v1/metrics/requests").json()
    assert rows.get("requests", rows.get("data", [])) == []


# --------------------------------------------------------------------------- #
# When it cannot count: a 400, and nothing woken or generated
# --------------------------------------------------------------------------- #


def test_a_sleeping_model_is_not_woken_to_be_counted(settings: Settings) -> None:
    driver = counting(runtime="qwen3")
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(
        driver,
        runtimes=[runtime_facts("qwen3", alias=MODEL, status="stopped", start_on_demand=True)],
    )
    spy = WakeSpy()
    app.state.lifecycle = spy
    with TestClient(app) as client:
        r = client.post(PATH, json=captured())
    assert r.status_code == 400, r.text
    assert r.json()["type"] == "error"
    assert "running" in r.json()["error"]["message"]
    assert spy.woken == 0 and not driver.count_calls


def test_a_backend_that_cannot_count_is_a_400_with_its_reason(settings: Settings) -> None:
    driver = counting(count=None)  # the driver answers 501
    with serve(settings, driver) as client:
        r = client.post(PATH, json=captured())
    assert r.status_code == 400, r.text
    assert "apply-template" in r.json()["error"]["message"]
    assert not driver.calls


def test_a_replica_that_can_count_answers_for_one_that_cannot(settings: Settings) -> None:
    first = counting("a", count=None, base_url="http://a")
    second = counting("b", count=77, base_url="http://b")
    with serve(settings, first, second) as client:
        r = client.post(PATH, json=captured())
    assert r.status_code == 200, r.text
    assert r.json() == {"input_tokens": 77}


def test_a_count_never_crosses_to_another_model(settings: Settings) -> None:
    """A later tier is another model, whose tokenizer answers a different
    question. The embeddings door's no-cross-model rule, for counts."""
    primary = counting("p", count=None, model_id=MODEL)
    fallback = counting("f", count=999, model_id="backup")
    with serve(
        settings, primary, fallback, slots=[{"model": MODEL, "targets": ["backup"]}]
    ) as client:
        r = client.post(PATH, json=captured())
    assert r.status_code == 400, r.text
    assert not fallback.count_calls


def test_an_image_is_not_counted(settings: Settings) -> None:
    """A real picture, so the refusal is the count's and not the image
    check's -- an invalid one is refused either way, for another reason."""
    driver = counting()
    data = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(data, format="PNG")
    image = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(data.getvalue()).decode(),
        },
    }
    with serve(settings, driver) as client:
        r = client.post(PATH, json=captured(messages=[{"role": "user", "content": [image]}]))
    assert r.status_code == 400, r.text
    assert "projector" in r.json()["error"]["message"]
    assert not driver.count_calls


def test_a_backend_that_fails_while_counting_is_still_a_4xx(settings: Settings) -> None:
    """A 5xx here is retried by Claude Code before it falls back, so the
    door answers every *cannot count* as a 400 -- this one included."""
    driver = counting()
    driver.count_error = httpx.ConnectError("refused")
    with serve(settings, driver) as client:
        r = client.post(PATH, json=captured())
    assert r.status_code == 400, r.text


def test_a_request_the_backend_rejects_is_a_400_naming_it(settings: Settings) -> None:
    driver = counting()
    driver.count_error = DriverError(
        driver_name="d",
        driver_url="http://fake-driver",
        status_code=400,
        problem=Problem(type="about:blank", title="bad", status=400, detail="bad tool schema"),
        raw_body="",
    )
    with serve(settings, driver) as client:
        r = client.post(PATH, json=captured())
    assert r.status_code == 400, r.text
    assert "bad tool schema" in r.json()["error"]["message"]


# --------------------------------------------------------------------------- #
# The Messages door's rules, shared
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"model": "nothing-serves-this"}, "nothing-serves-this"),
        (
            {"messages": [{"role": "user", "content": [{"type": "document", "source": {}}]}]},
            "document",
        ),
        ({"frobnicate": True}, "frobnicate"),
    ],
)
def test_the_messages_doors_refusals_apply(
    settings: Settings, overrides: dict[str, Any], needle: str
) -> None:
    driver = counting()
    with serve(settings, driver) as client:
        r = client.post(PATH, json=captured(**overrides))
    assert r.status_code == 400, r.text
    assert needle in r.json()["error"]["message"]
    assert not driver.count_calls


@pytest.fixture
def keyed(settings: Settings, install: FakeInstall):
    authority = Authority()
    token = install.client_key(name="app", jti="key-1")

    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    allowed = counting("allowed-driver", model_id="allowed", count=11)
    excluded = counting("excluded-driver", model_id="excluded", count=22)
    app.state.routing = make_routing_table(allowed, excluded)
    return app, authority, allowed, excluded, {"x-api-key": token}


def test_a_client_key_is_checked_not_admitted(keyed) -> None:
    """`/context` sends fourteen counts at once; a reservation each would
    exhaust a key's concurrency, and a rate charge each its minute."""
    app, authority, allowed, _, headers = keyed
    with TestClient(app) as client:
        r = client.post(PATH, json=captured(model="allowed"), headers=headers)
    assert r.status_code == 200, r.text
    assert len(allowed.count_calls) == 1
    assert [c["action"] for c in authority.calls] == ["check"]


def test_a_client_key_cannot_count_a_model_it_may_not_use(keyed) -> None:
    app, _, _, excluded, headers = keyed
    with TestClient(app) as client:
        r = client.post(PATH, json=captured(model="excluded"), headers=headers)
    assert r.status_code == 400, r.text
    assert not excluded.count_calls


def test_a_missing_credential_is_403_as_on_messages(keyed) -> None:
    app, *_ = keyed
    with TestClient(app) as client:
        r = client.post(PATH, json=captured(model="allowed"))
    assert r.status_code == 403, r.text
    assert r.json()["error"]["type"] == "authentication_error"


def test_a_backend_rejection_is_not_asked_of_every_replica(settings: Settings) -> None:
    """A 400 is the request's fault and every replica would say the same."""
    first = counting("a", base_url="http://a")
    first.count_error = DriverError(
        driver_name="a",
        driver_url="http://a",
        status_code=400,
        problem=Problem(type="about:blank", title="bad", status=400, detail="bad tool schema"),
        raw_body="",
    )
    second = counting("b", base_url="http://b")
    second.count_error = first.count_error
    with serve(settings, first, second) as client:
        r = client.post(PATH, json=captured())
    assert r.status_code == 400, r.text
    assert len(first.count_calls) + len(second.count_calls) == 1


def test_a_local_only_key_counts_only_on_a_local_backend(keyed) -> None:
    """The policy reaches the driver, and a refusal is in Anthropic's shape."""
    from tests.test_local_only import describe_as

    app, authority, allowed, _, headers = keyed
    authority.local_only = True
    describe_as(allowed, "local")
    with TestClient(app) as client:
        r = client.post(PATH, json=captured(model="allowed"), headers=headers)
        assert r.status_code == 200, r.text
        assert allowed.count_calls[-1].localOnly is True

        describe_as(allowed, "external")
        r = client.post(PATH, json=captured(model="allowed"), headers=headers)
    assert r.status_code == 403, r.text
    assert r.json()["type"] == "error", "a refusal on this path is Anthropic's envelope"
    assert len(allowed.count_calls) == 1
