"""Client policy through real gateway middleware, routing and metric storage.

The authority protocol is scripted here; the specs acceptance runner exercises
the durable authority and two independent gateway processes together.
"""

import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway import admission
from eugene_plexus_gateway._generated.driver_models import Usage
from eugene_plexus_gateway.app import create_app
from tests.conftest import FakeDriverClient, install_snapshot, make_routing_table
from tests.test_client_disconnect import _SCOPE, _Hang
from tests.test_client_keys import FakeAgent


class Authority(FakeAgent):
    def __init__(self):
        super().__init__()
        self.allowed = ["alias", "allowed"]
        self.calls = []
        self.refuse_acquire = None
        self.refuse_renew = None
        self.lease = 30
        self.local_only = False
        self.write_logs = False
        self.allowed_tools = None

    def _handle(self, request):
        if request.url.path.endswith("/admission"):
            body = json.loads(request.content)
            self.calls.append(body)
            action = body["action"]
            code = (
                self.refuse_acquire
                if action == "acquire"
                else self.refuse_renew
                if action == "renew"
                else None
            )
            if code:
                return httpx.Response(code, headers={"Retry-After": "7"})
            return httpx.Response(
                200,
                json={
                    "keyId": body["keyId"],
                    "keyName": "Verified app",
                    "limits": {
                        **({"localOnly": True} if self.local_only else {}),
                        **({"writeLogs": True} if self.write_logs else {}),
                        **(
                            {"allowedTools": self.allowed_tools}
                            if self.allowed_tools is not None
                            else {}
                        ),
                        "allowedModels": self.allowed,
                        "maxConcurrentRequests": 1,
                        "requestsPerMinute": 2,
                    },
                    "leaseSeconds": self.lease,
                },
            )
        return super()._handle(request)


@pytest.fixture
def setup(settings, install):
    authority = Authority()
    token = install.client_key(name="Original name", jti="key-1")
    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    allowed = FakeDriverClient(name="allowed-driver", model_id="allowed")
    excluded = FakeDriverClient(name="excluded-driver", model_id="excluded")
    app.state.routing = make_routing_table(
        allowed, excluded, slots=[{"model": "alias", "targets": ["allowed", "excluded"]}]
    )
    headers = {"Authorization": "Bearer " + token}
    operator = {"Authorization": "Bearer " + install.session(sub="operator")}
    return app, authority, allowed, excluded, headers, operator


def body(model="alias", **extra):
    return {"model": model, "messages": [{"role": "user", "content": "PRIVATE_PROMPT"}], **extra}


def test_discovery_and_fallback_never_reveal_or_call_excluded_targets(setup):
    app, authority, allowed, excluded, headers, _ = setup
    with TestClient(app) as c:
        models = c.get("/v1/models", headers=headers)
        assert {m["id"] for m in models.json()["data"]} == {"alias", "allowed"}
        assert "excluded" not in models.text
        allowed.generate_error = httpx.ConnectError("offline")
        response = c.post("/v1/chat/completions", json=body(), headers=headers)
        assert response.status_code == 502
        assert len(allowed.calls) == 1 and not excluded.calls
        # Row 11: the chat checks when it arrives and acquires when it leaves its line.
        assert [x["action"] for x in authority.calls] == [
            "check",
            "check",
            "acquire",
            "renew",
            "release",
        ]


def test_alias_permission_alone_cannot_authorize_its_target(setup):
    app, authority, allowed, excluded, headers, _ = setup
    authority.allowed = ["alias"]
    with TestClient(app) as c:
        assert c.get("/v1/models", headers=headers).json()["data"] == []
        response = c.post("/v1/chat/completions", json=body(), headers=headers)
        assert response.status_code == 404
        assert "excluded" not in response.text and "allowed-driver" not in response.text
        assert not allowed.calls and not excluded.calls


@pytest.mark.parametrize(
    "path,extra",
    [
        ("/v1/chat/completions", {}),
        ("/v1/chat/completions", {"stream": True}),
        ("/v1/messages", {"max_tokens": 10}),
        ("/v1/messages", {"max_tokens": 10, "stream": True}),
        ("/v1/embeddings", {"input": "PRIVATE_INPUT"}),
    ],
)
def test_all_surfaces_record_verified_identity_and_release(setup, path, extra):
    app, authority, allowed, _, headers, operator = setup
    allowed.usage = Usage(promptTokens=7, completionTokens=3, totalTokens=10)
    if path.endswith("embeddings"):
        allowed.supports_embeddings = True
        install_snapshot(app.state.routing, allowed)
    with TestClient(app) as c:
        payload = (
            {"model": "allowed", **extra}
            if path.endswith("embeddings")
            else body("allowed", **extra)
        )
        if path.endswith("completions"):
            payload["user"] = "FORGED_ID"
        response = c.post(path, json=payload, headers=headers)
        assert response.status_code == 200, response.text
        assert authority.calls[-1]["action"] == "release"
        for _ in range(100):
            usage = c.get("/v1/metrics/clients", headers=operator).json()
            if usage["clients"]:
                break
            time.sleep(0.01)
        row = usage["clients"][0]
        assert row["clientKeyId"] == "key-1" and row["clientKeyName"] == "Verified app"
        assert row["requests"] == row["served"] == row["attempts"] == 1
        assert row["failed"] == row["incompleteUsageRequests"] == 0
        history = c.get("/v1/metrics/requests", headers=operator)
        assert history.json()["requests"][0]["clientKeyId"] == "key-1"
        assert not any(
            secret in history.text + json.dumps(usage)
            for secret in ["PRIVATE_PROMPT", "PRIVATE_INPUT", "FORGED_ID", headers["Authorization"]]
        )


@pytest.mark.parametrize("code", [429, 503])
def test_refusal_precedes_backend_and_keeps_operator_repair_available(setup, code):
    app, authority, allowed, excluded, headers, operator = setup
    authority.refuse_acquire = code
    headers["Origin"] = "http://browser.test"
    with TestClient(app) as c:
        response = c.post("/v1/chat/completions", json=body(), headers=headers)
        assert response.status_code == code and response.headers["Retry-After"] == "7"
        assert response.headers["Access-Control-Allow-Origin"] == "*"
        assert "retry-after" in response.headers["Access-Control-Expose-Headers"].lower()
        assert not allowed.calls and not excluded.calls
        assert c.get("/v1/config", headers=operator).status_code == 200


@pytest.mark.parametrize("code", [404, 500, 502, 504])
def test_a_door_that_generates_says_what_the_authority_answered_and_keeps_its_503(setup, code):
    """Every status the authority answers outside the four it refuses a key with is a 503 to
    try again, with the authority's own `Retry-After`, as it always was. What changed is that
    the sentence says which status it was: a 500, a 404 (an agent that has no admission
    route) and a 504 read the same, and a person was left to guess."""
    app, authority, allowed, excluded, headers, _ = setup
    authority.refuse_acquire = code
    with TestClient(app) as c:
        response = c.post("/v1/chat/completions", json=body(), headers=headers)
    assert response.status_code == 503 and response.headers["Retry-After"] == "7"
    message = response.json()["error"]["message"]
    assert f"needs the admission authority, which answered {code}. Check the" in message
    assert not allowed.calls and not excluded.calls


class _Guard:
    """The authority as `ClientRequest.contact` meets it: it answers, or the call ends in an
    exception."""

    def __init__(self, outcome):
        self.outcome = outcome

    async def admission(self, body):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


async def _contact(outcome, action="check", *, deadline=None):
    context = admission.ClientRequest({"type": "http"})
    context.key_id, context.guard = "key-1", _Guard(outcome)
    context.deadline = time.perf_counter() + 30 if deadline is None else deadline
    return await context.contact(action)


@pytest.mark.parametrize("answered", [400, 404, 500, 502, 503, 504])
@pytest.mark.parametrize(("header", "retry"), [({"Retry-After": "7"}, "7"), ({}, "5")])
async def test_an_authority_that_cannot_answer_is_a_503_that_says_what_it_answered(
    answered, header, retry
):
    with pytest.raises(admission.AdmissionFailure) as raised:
        await _contact(httpx.Response(answered, headers=header))
    assert (raised.value.status, raised.value.retry) == (503, retry)
    assert raised.value.message == (
        f"Client access needs the admission authority, which answered {answered}. Check the "
        "local agent and active control root; operator management remains available."
    )


@pytest.mark.parametrize(
    ("answered", "said", "retry"),
    [
        (401, "This client key is unregistered, expired or revoked.", None),
        (403, "This client key does not permit the requested model.", None),
        (409, "Request permissions or reservation changed; start a new request.", None),
        (
            429,
            "This key's shared concurrency or request-rate limit is reached. Wait or cancel an "
            "active request; switching gateways does not add allowance.",
            "7",
        ),
    ],
)
async def test_what_the_authority_refuses_a_key_with_keeps_its_own_words(answered, said, retry):
    """The control: these four are the key's refusal, said as it always was, not a failure
    to reach the authority."""
    with pytest.raises(admission.AdmissionFailure) as raised:
        await _contact(httpx.Response(answered, headers={"Retry-After": "7"}))
    assert (raised.value.status, raised.value.message, raised.value.retry) == (
        answered,
        said,
        retry,
    )


@pytest.mark.parametrize(
    ("outcome", "action", "observed"),
    [
        pytest.param(
            httpx.ConnectError("connection refused"),
            "check",
            "ConnectError: connection refused",
            id="a refused connection",
        ),
        pytest.param(httpx.ReadTimeout(""), "check", "ReadTimeout", id="a timeout has no words"),
        pytest.param(
            httpx.RemoteProtocolError("Server disconnected without sending a response."),
            "acquire",
            "RemoteProtocolError: Server disconnected without sending a response",
            id="a dropped connection, whose own full stop is not said twice",
        ),
        pytest.param(
            httpx.Response(200, content=b"this is not json"),
            "check",
            "JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
            id="an answer that is not JSON",
        ),
        pytest.param(
            httpx.Response(200, json={"keyId": "another-key", "keyName": "k", "leaseSeconds": 30}),
            "check",
            "ValueError: authority returned another key",
            id="the answer for another key",
        ),
        pytest.param(
            httpx.Response(200, json={"keyId": "key-1", "keyName": "k"}),
            "acquire",
            "ValueError: no reservation duration",
            id="a reservation with no duration",
        ),
    ],
)
async def test_an_authority_that_did_not_answer_is_a_503_naming_what_the_call_ended_in(
    outcome, action, observed
):
    with pytest.raises(admission.AdmissionFailure) as raised:
        await _contact(outcome, action)
    assert (raised.value.status, raised.value.retry) == (503, "5")
    assert raised.value.message == (
        f"Client admission is unavailable ({observed}). Restore the agent/control "
        "connection; this gateway will not create a separate allowance."
    )


async def test_a_renewal_past_its_deadline_names_the_timeout():
    """A bare `TimeoutError` has no words, and is said as its class."""
    with pytest.raises(admission.AdmissionFailure) as raised:
        await _contact(httpx.Response(200), "renew", deadline=0.0)
    assert raised.value.message.startswith(
        "Client admission is unavailable (TimeoutError). Restore"
    )


async def test_an_answer_that_is_not_an_admission_result_is_named_on_one_line_and_capped():
    """A malformed answer's validation error is many lines, and as long as the answer."""
    with pytest.raises(admission.AdmissionFailure) as raised:
        await _contact(httpx.Response(200, json={"keyId": "key-1", "leaseSeconds": "x" * 500}))
    message = raised.value.message
    head = "Client admission is unavailable ("
    assert message.startswith(head + "ValidationError: ")
    observed = message[len(head) :].split("). Restore", 1)[0]
    assert len(observed) <= admission.MAX_CAUSE_CHARS and observed.endswith("...")
    assert "\n" not in message and "  " not in message


@pytest.mark.parametrize(
    ("exc", "said"),
    [
        (ValueError("two\n   lines"), "ValueError: two lines"),
        (ValueError("   "), "ValueError"),
        (httpx.ReadTimeout(""), "ReadTimeout"),
        (TimeoutError(), "TimeoutError"),
        (ValueError("ends in a stop."), "ValueError: ends in a stop"),
        (ValueError("was cut..."), "ValueError: was cut..."),
    ],
)
def test_what_a_failure_was_is_its_class_and_its_words_on_one_line(exc, said):
    assert admission._observed(exc) == said


def test_what_a_failure_was_is_capped_at_a_word():
    said = admission._observed(ValueError("word " * 100))
    assert len(said) <= admission.MAX_CAUSE_CHARS
    assert said.startswith("ValueError: word word") and said.endswith("word...")


def test_anthropic_current_authority_revocation_keeps_non_retrying_403(setup):
    app, authority, allowed, _, headers, _ = setup
    authority.refuse_acquire = 401
    with TestClient(app) as c:
        response = c.post("/v1/messages", headers=headers, json=body(max_tokens=10))
        assert response.status_code == 403
        assert not allowed.calls


@pytest.mark.parametrize("surface", ["chat", "stream", "embeddings", "wake"])
async def test_disconnect_cancels_work_and_releases_reservation(setup, surface):
    app, authority, allowed, _, headers, _ = setup
    hang = _Hang()
    allowed.generate_hook = allowed.embed_hook = hang.run
    path = "/v1/embeddings" if surface == "embeddings" else "/v1/chat/completions"
    if surface == "embeddings":
        allowed.supports_embeddings = True
        install_snapshot(app.state.routing, allowed)
    if surface == "stream":

        async def hanging_stream(request):
            await hang.run()
            if False:
                yield

        allowed.stream = hanging_stream
    if surface == "wake":
        from tests.conftest import runtime_facts

        allowed.runtime = "sleeping"
        install_snapshot(
            app.state.routing,
            allowed,
            runtimes=[runtime_facts("sleeping", status="stopped", start_on_demand=True)],
        )

        class Lifecycle:
            async def wake(self, resolution):
                return await hang.run()

        app.state.lifecycle = Lifecycle()

        async def no_refresh():
            return False

        app.state.routing.refresh_if_stale = no_refresh
    payload = (
        {"model": "allowed", "input": "x"}
        if surface == "embeddings"
        else body("allowed", stream=surface == "stream")
    )
    gone = asyncio.Event()
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {
                "type": "http.request",
                "body": json.dumps(payload).encode(),
                "more_body": False,
            }
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    scope = _SCOPE | {
        "path": path,
        "raw_path": path.encode(),
        "headers": [
            (b"authorization", headers["Authorization"].encode()),
            (b"content-type", b"application/json"),
        ],
    }
    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(hang.entered.wait(), 3)
        # Row 11: a check on arrival, the lease when it leaves the line (before a
        # wake too, so a key at its limit cannot start a runtime).
        assert [c["action"] for c in authority.calls][:2] == ["check", "acquire"]
        gone.set()
        await asyncio.wait_for(task, 2)
        assert hang.cancelled
        assert authority.calls[-1]["action"] == "release"
        assert not any(app.state.routing._inflight.values())
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await app.state.client_key_guard.aclose()
        await app.state.routing.aclose()


@pytest.mark.parametrize(
    "path,extra,failing",
    [
        ("/v1/chat/completions", {}, False),
        ("/v1/chat/completions", {}, True),
        ("/v1/chat/completions", {"stream": True}, False),
        ("/v1/messages", {"max_tokens": 10}, False),
        ("/v1/messages", {"max_tokens": 10, "stream": True}, False),
    ],
)
async def test_the_slot_is_free_before_the_client_can_read_the_end(setup, path, extra, failing):
    """gateway #9: released after the last byte, a key limited to one request
    was refused 429 by its next request, sent at once through another gateway."""
    app, authority, allowed, _, headers, _ = setup
    if failing:
        allowed.generate_error = httpx.ConnectError("offline")
    payload = json.dumps(body("allowed", **extra)).encode()
    delivered = False
    at_end: list[str] = []
    status = None

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        elif not message.get("more_body", False):
            at_end.extend(x["action"] for x in authority.calls)

    scope = _SCOPE | {
        "path": path,
        "raw_path": path.encode(),
        "headers": [
            (b"authorization", headers["Authorization"].encode()),
            (b"content-type", b"application/json"),
        ],
    }
    try:
        await asyncio.wait_for(app(scope, receive, send), 3)
        assert status == (502 if failing else 200)
        assert at_end and at_end[-1] == "release", at_end
        assert [x["action"] for x in authority.calls].count("release") == 1
    finally:
        await app.state.client_key_guard.aclose()
        await app.state.routing.aclose()


async def test_no_renewal_after_the_release(setup):
    """A renewal that reached the authority after the release would be refused
    409 and cut the reply short, so a released request renews nothing."""
    app, authority, *_ = setup
    from eugene_plexus_gateway.admission import ClientRequest

    context = ClientRequest({"type": "http"})
    context.key_id, context.guard, context.attempted = "key-1", app.state.client_key_guard, True
    context.ready.set()
    try:
        await context.release()
        await context.renew()
        monitor = asyncio.create_task(context.monitor())
        await asyncio.sleep(0.05)
        assert not monitor.done()  # Parked until cancelled, not finished.
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)
        assert [x["action"] for x in authority.calls] == ["release"]
    finally:
        await app.state.client_key_guard.aclose()
        await app.state.routing.aclose()


async def test_a_release_waits_for_a_renewal_in_flight(setup):
    """Otherwise the renewal can reach the authority second and be refused."""
    app, authority, _, _, _, _ = setup
    from eugene_plexus_gateway.admission import ClientRequest

    gate = asyncio.Event()
    answered: list[str] = []

    async def handle(request):
        action = json.loads(request.content)["action"]
        if action == "renew":
            await gate.wait()
        response = authority._handle(request)
        answered.append(action)
        return response

    guard = app.state.client_key_guard
    guard._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    context = ClientRequest({"type": "http"})
    context.key_id, context.guard, context.attempted = "key-1", guard, True
    context.deadline = time.perf_counter() + 10
    context.ready.set()
    try:
        renewing = asyncio.create_task(context.renew())
        await asyncio.sleep(0.05)
        releasing = asyncio.create_task(context.release())
        await asyncio.sleep(0.05)
        assert answered == []  # The release waits its turn.
        gate.set()
        await asyncio.wait_for(asyncio.gather(renewing, releasing), 2)
        assert answered == ["renew", "release"]
    finally:
        await guard.aclose()
        await app.state.routing.aclose()


async def test_a_cancelled_release_is_tried_again(setup):
    """A client leaving during the last-byte release cancels it; the
    middleware's `finally` must still free the slot."""
    app, authority, *_ = setup
    from eugene_plexus_gateway.admission import ClientRequest

    gate = asyncio.Event()
    answered: list[str] = []

    async def handle(request):
        await gate.wait()
        response = authority._handle(request)
        answered.append(json.loads(request.content)["action"])
        return response

    guard = app.state.client_key_guard
    guard._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    context = ClientRequest({"type": "http"})
    context.key_id, context.guard, context.attempted = "key-1", guard, True
    try:
        first = asyncio.create_task(context.release())
        await asyncio.sleep(0.05)
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        gate.set()
        await asyncio.wait_for(context.release(), 2)
        assert answered == ["release"]
    finally:
        await guard.aclose()
        await app.state.routing.aclose()


async def test_renewal_outage_cancels_generation_before_lease_expiry(setup):
    app, authority, allowed, _, headers, _ = setup
    authority.lease = 0.6
    hang = _Hang()
    allowed.generate_hook = hang.run
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway"
    ) as c:
        task = asyncio.create_task(c.post("/v1/chat/completions", json=body(), headers=headers))
        try:
            await asyncio.wait_for(hang.entered.wait(), 2)
            authority.refuse_renew = 503
            response = await asyncio.wait_for(task, 2)
            assert response.status_code == 503
            assert hang.cancelled and authority.calls[-1]["action"] == "release"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await app.state.client_key_guard.aclose()
            await app.state.routing.aclose()


def test_an_apps_key_that_may_send_logs_still_reaches_the_models(setup):
    """C1: every app's key carries `writeLogs` for the agent's log ingress.
    The gateway's model of the limits refuses fields it does not know, so
    before it knew this one, an app's every request failed admission."""
    app, authority, allowed, _excluded, headers, _ = setup
    authority.write_logs = True
    with TestClient(app) as c:
        response = c.post("/v1/chat/completions", json=body(model="allowed"), headers=headers)
        assert response.status_code == 200, response.text
        assert len(allowed.calls) == 1


def test_a_key_denied_search_is_told_so_by_the_model_list(setup):
    """C3: the list's search answer is this key's, not the install's: a key
    whose tool scope leaves out web_search reads `available: false` with
    the refusal's own words, while the operator reads the install."""
    app, authority, _allowed, _excluded, headers, operator = setup
    authority.allowed_tools = []
    with TestClient(app) as c:
        mine = c.get("/v1/models", headers=headers).json()["x_eugene_plexus"]["web_search"]
        assert mine["available"] is False
        assert "tool scope does not include web_search" in mine["reason"]
        authority.allowed_tools = ["web_search"]
        allowed = c.get("/v1/models", headers=headers).json()["x_eugene_plexus"]["web_search"]
        assert "tool scope" not in (allowed["reason"] or ""), (
            "permitted: only the account is missing"
        )
        theirs = c.get("/v1/models", headers=operator).json()["x_eugene_plexus"]["web_search"]
        assert "tool scope" not in (theirs["reason"] or "")


async def test_a_root_that_names_a_queue_band_is_understood():
    """Row 70: a root answers a verified person's `queueBand`. A gateway whose
    models predate it refuses the whole answer (`extra='forbid'`) as a 503;
    this one keeps it, for row 11's queue to read."""
    answer = await _contact(
        httpx.Response(200, json={"keyId": "key-1", "keyName": "k", "queueBand": "background"})
    )
    assert answer["queueBand"] == "background"
