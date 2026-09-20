"""Four sentences the contract has made since M0, three of them now true.

R3 item 4. Each of these is a place where `gateway.yaml` described a
behaviour and the code did something else, and every one of them is
silent: nothing errors, nothing is logged, and the response looks
exactly like a correct one.

* `top_p` and `seed` were accepted, range-validated and then dropped on
  the floor, because `GenerateRequest` had no field to carry them. A
  caller asking for a seed is asking for a reproducible answer and got a
  different one each time.
* A backend's `content_filter` became the driver's `error` and then the
  gateway's `stop`, so a refusal arrived as a natural end.
* A driver that refused the gateway's OWN credential was reported to the
  caller as `invalid_request_error` -- their fault, and unfixable by
  them.

The fourth, `max_tokens`/`temperature` from gateway-wide defaults where
the contract says *the model's settings profile*, is deliberately NOT
fixed here and is deliberately not softened in the contract either. It
needs a gateway->library edge that does not exist; it is scheduled as
its own slice and the sentence stands as an unmet promise until then.
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import FinishReason as DriverFinishReason
from eugene_plexus_gateway._generated.models import Problem
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import DriverError
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

MODEL = "Qwen3-30B-A3B-Q4_K_M"


def _app_with(settings: Settings, *fakes: FakeDriverClient) -> FastAPI:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*fakes)
    return app


def _chat(model: str = MODEL, **extra: object) -> dict[str, object]:
    body: dict[str, object] = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    return body


def _driver_error(status: int, detail: str = "nope") -> DriverError:
    return DriverError(
        driver_name="a",
        driver_url="http://a",
        status_code=status,
        problem=None,
        raw_body=detail,
    )


# --------------------------------------------------------------------------- #
# top_p and seed reach the driver
# --------------------------------------------------------------------------- #


def test_top_p_and_seed_reach_the_driver(settings: Settings) -> None:
    """The reproduction. Before `GenerateRequest` grew the two fields,
    `_to_generate_request` had nowhere to put them and the values ended
    at the gateway's own validator."""
    fake = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL)
    with TestClient(_app_with(settings, fake)) as c:
        r = c.post("/v1/chat/completions", json=_chat(top_p=0.42, seed=1234))

    assert r.status_code == 200, r.text
    assert len(fake.calls) == 1
    sent = fake.calls[0]
    assert sent.topP == 0.42
    assert sent.seed == 1234


def test_top_p_and_seed_reach_the_driver_when_streaming(settings: Settings) -> None:
    """The other path, because the two build their request separately in
    every other respect and a fix that lands on one is this project's
    most-repeated defect."""
    fake = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL)
    fake.responses = ["one two"]
    with (
        TestClient(_app_with(settings, fake)) as c,
        c.stream("POST", "/v1/chat/completions", json=_chat(top_p=0.9, seed=7, stream=True)) as r,
    ):
        assert r.status_code == 200
        list(r.iter_lines())

    sent = fake.calls[0]
    assert sent.topP == 0.9
    assert sent.seed == 7


def test_omitting_them_sends_nothing_rather_than_a_default(settings: Settings) -> None:
    """No install default to fall back to, unlike `max_tokens` and
    `temperature`: nucleus sampling and a seed are the caller's
    business, and inventing either would change the answer on a request
    that never asked for it."""
    fake = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL)
    with TestClient(_app_with(settings, fake)) as c:
        r = c.post("/v1/chat/completions", json=_chat())

    assert r.status_code == 200, r.text
    assert fake.calls[0].topP is None
    assert fake.calls[0].seed is None


def test_the_anthropic_door_carries_top_p_too(settings: Settings) -> None:
    """`/v1/messages` already read `top_p` off its own body and put it on
    the OpenAI request; it stopped at the same wall one layer down."""
    fake = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL)
    with TestClient(_app_with(settings, fake)) as c:
        r = c.post(
            "/v1/messages",
            json={
                "model": MODEL,
                "max_tokens": 16,
                "top_p": 0.33,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert r.status_code == 200, r.text
    assert fake.calls[0].topP == 0.33


# --------------------------------------------------------------------------- #
# content_filter survives to the caller, on both doors and both paths
# --------------------------------------------------------------------------- #


def test_a_filtered_answer_is_not_a_natural_end(settings: Settings) -> None:
    """The reproduction: `content_filter` used to arrive as `stop`, so a
    caller could not tell a refusal from a reply."""
    fake = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL)
    fake.finish_reason = DriverFinishReason.content_filter
    with TestClient(_app_with(settings, fake)) as c:
        r = c.post("/v1/chat/completions", json=_chat())

    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["finish_reason"] == "content_filter"


def test_a_filtered_answer_is_not_a_natural_end_when_streaming(settings: Settings) -> None:
    fake = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL)
    fake.finish_reason = DriverFinishReason.content_filter
    fake.responses = ["I cannot"]
    frames = []
    with (
        TestClient(_app_with(settings, fake)) as c,
        c.stream("POST", "/v1/chat/completions", json=_chat(stream=True)) as r,
    ):
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data: ") and not line.endswith("[DONE]"):
                frames.append(json.loads(line[6:]))

    # `.get`, not `[...]`: non-terminal frames omit the key entirely
    # rather than carrying a null, which the contract allows and the
    # serializer does.
    terminal = [f for f in frames if f["choices"][0].get("finish_reason") is not None]
    assert terminal, "no frame carried a finish_reason"
    assert terminal[-1]["choices"][0]["finish_reason"] == "content_filter"


def test_a_backend_error_is_still_a_stop(settings: Settings) -> None:
    """The pair that tells the fix from the over-correction. `error`
    means the generation was truncated by something breaking, there is no
    OpenAI value for it, and it keeps reporting `stop` with the text that
    did arrive. Only `content_filter` moved."""
    fake = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL)
    fake.finish_reason = DriverFinishReason.error
    with TestClient(_app_with(settings, fake)) as c:
        r = c.post("/v1/chat/completions", json=_chat())

    assert r.json()["choices"][0]["finish_reason"] == "stop"


def test_the_anthropic_door_calls_it_a_refusal(settings: Settings) -> None:
    """Anthropic's own name for the same state. A client switching on
    `stop_reason` gets a value from the vocabulary it already parses --
    the rule `tool_use` was fixed under, applied to the next value."""
    fake = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL)
    fake.finish_reason = DriverFinishReason.content_filter
    with TestClient(_app_with(settings, fake)) as c:
        r = c.post(
            "/v1/messages",
            json={
                "model": MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert r.status_code == 200, r.text
    assert r.json()["stop_reason"] == "refusal"


def test_the_anthropic_door_still_ends_a_broken_backend_with_end_turn(
    settings: Settings,
) -> None:
    fake = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL)
    fake.finish_reason = DriverFinishReason.error
    with TestClient(_app_with(settings, fake)) as c:
        r = c.post(
            "/v1/messages",
            json={
                "model": MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert r.json()["stop_reason"] == "end_turn"


# --------------------------------------------------------------------------- #
# a driver that refuses OUR credential is not the caller's bad request
# --------------------------------------------------------------------------- #


def test_a_driver_401_is_not_blamed_on_the_caller(settings: Settings) -> None:
    """The reproduction. A rotated `service:gateway` token on one node
    makes that node's driver answer 401. The request was correct; the
    install is not. Reported as `invalid_request_error` it sent a
    harness to re-read a prompt that never had a problem.

    **The driver name here is deliberately not `a`.** The first version
    of this test used the one-character name the rest of the file uses
    and asserted `"a" in message`, which is true of almost any English
    sentence -- a sabotage that removed the name from the message
    escaped, and the check could never have failed. A distinctive name
    and the URL beside it are what make the assertion mean anything.
    """
    fake = FakeDriverClient(
        name="qwen-driver@amish", base_url="http://192.0.2.9:8081", model_id=MODEL
    )
    fake.generate_error = DriverError(
        driver_name="qwen-driver@amish",
        driver_url="http://192.0.2.9:8081",
        status_code=401,
        problem=None,
        raw_body="token expired",
    )
    with TestClient(_app_with(settings, fake)) as c:
        r = c.post("/v1/chat/completions", json=_chat())

    assert r.status_code == 502
    error = r.json()["error"]
    assert error["type"] == "upstream_auth_error"
    # Names the driver AND where it is, because the operator reading
    # this has to know which node's credential to re-mint.
    assert "qwen-driver@amish" in error["message"]
    assert "192.0.2.9:8081" in error["message"]
    assert "token expired" in error["message"]


def test_a_real_driver_problem_body_still_names_the_driver(settings: Settings) -> None:
    """The case where naming the driver is load-bearing, found by a
    sabotage that escaped without it.

    A driver that answers 401 answers with a `problem+json` body, and
    `_driver_failure` prefers `problem.detail` over `str(e)`. The
    exception's own summary happens to carry the driver name and URL,
    so with `problem=None` a message that named neither still passed --
    the belt-and-braces path was doing the work and the assertion could
    not tell. With a real Problem body the summary never appears, and
    the only thing that puts the operator in front of the right node is
    the sentence we write here.
    """
    fake = FakeDriverClient(
        name="qwen-driver@amish", base_url="http://192.0.2.9:8081", model_id=MODEL
    )
    fake.generate_error = DriverError(
        driver_name="qwen-driver@amish",
        driver_url="http://192.0.2.9:8081",
        status_code=401,
        problem=Problem(
            type="about:blank",
            title="Unauthorized",
            status=401,
            detail="The token is not yet valid (iat).",
        ),
        raw_body="{}",
    )
    with TestClient(_app_with(settings, fake)) as c:
        r = c.post("/v1/chat/completions", json=_chat())

    message = r.json()["error"]["message"]
    assert "qwen-driver@amish" in message
    assert "192.0.2.9:8081" in message
    assert "The token is not yet valid (iat)." in message


def test_a_driver_403_is_the_same_answer(settings: Settings) -> None:
    fake = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL)
    fake.generate_error = _driver_error(403, "wrong audience")
    with TestClient(_app_with(settings, fake)) as c:
        r = c.post("/v1/chat/completions", json=_chat())

    assert r.status_code == 502
    assert r.json()["error"]["type"] == "upstream_auth_error"


def test_a_backend_rejection_is_still_the_callers_400(settings: Settings) -> None:
    """The pair, again. A 400 from the driver means the BACKEND refused
    the request -- a prompt longer than the context window is the usual
    one -- and that is the caller's to fix. Moving every driver 4xx would
    be the over-correction, and the check set has to be able to fail it."""
    fake = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL)
    fake.generate_error = _driver_error(400, "prompt is 15010 tokens, n_ctx is 512")
    with TestClient(_app_with(settings, fake)) as c:
        r = c.post("/v1/chat/completions", json=_chat())

    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_a_driver_401_does_not_cascade(settings: Settings) -> None:
    """Changing what we SAY about a 401 must not change what we DO with
    it. The next backend is a different process with a different token
    and might well answer -- but a 4xx has never cascaded here, the rule
    is older than this slice, and quietly widening the cascade under
    cover of an error-message fix is how a correctness pass becomes a
    behaviour change nobody reviewed."""
    first = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL)
    first.generate_error = _driver_error(401)
    second = FakeDriverClient(name="b", base_url="http://b", model_id=MODEL)
    second.responses = ["should never be reached"]

    with TestClient(_app_with(settings, first, second)) as c:
        r = c.post("/v1/chat/completions", json=_chat())

    assert r.status_code == 502
    assert second.calls == []


def test_the_anthropic_door_reports_a_driver_401_as_api_error(settings: Settings) -> None:
    """Not `permission_error`: on this door that reads as the CALLER's
    credential being wrong, and theirs was fine. Anthropic's vocabulary
    has no member for *our upstream refused us*, so the honest render is
    the generic server-side one."""
    fake = FakeDriverClient(name="a", base_url="http://a", model_id=MODEL)
    fake.generate_error = _driver_error(401)
    with TestClient(_app_with(settings, fake)) as c:
        r = c.post(
            "/v1/messages",
            json={
                "model": MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert r.status_code == 502
    assert r.json()["error"]["type"] == "api_error"
