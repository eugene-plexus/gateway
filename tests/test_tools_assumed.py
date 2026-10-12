"""Tools are assumed, and turned off only on evidence (Troy, 2026-10-12).

"We should always assume in 2026 that a model & engine can use tools." The
design is `specs/docs/design/tools-assumed.md`; this is the gateway's half:

- **What the serving backend does not honour is named.** A driver's
  `x-eugene-plexus-ignored-settings` on `/v1/generate` and `/v1/generate/stream`
  (Strata ignores `tool_choice` `required` and a named function, measured) is
  read off the attempt that served, said in each door's own spelling, merged
  into the door's own header, and sent on both response modes. A streamed
  response's head waits for its first frame so that it can carry it.
- **A backend whose driver does not say counts as one that calls tools**, in
  every place the gateway decides: `GET /v1/models`, the front-door refusal,
  the admission check, a search's candidates. Only `toolCalling: false` is
  evidence, and it refuses as it always did.

The driver is a fake of the driver's contract (`FakeDriverClient`), and the
real `HttpDriverClient` is shown the same contract on a `MockTransport`: a
component test is not a wiring test.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.types import Message as AsgiMessage

from eugene_plexus_gateway import ignored_settings, responses
from eugene_plexus_gateway._generated.driver_models import (
    GenerateRequest,
    Message,
    Role,
    ToolChoice,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.capabilities import calls_tools
from eugene_plexus_gateway.driver_client import (
    DriverAnswer,
    DriverError,
    FailoverDriverClient,
    HttpDriverClient,
    StreamEvent,
)
from eugene_plexus_gateway.late_head import LateHeadStreamingResponse
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table
from .test_anthropic_messages import body as claude_code_body
from .test_responses import codex_request
from .test_server_tools import (
    ANSWER,
    FakeSearch,
    ScriptedDriver,
    Turn,
    account,
    app_with,
    chat,
    search_call,
)

HEADER = ignored_settings.HEADER
MODEL = "qwen3.8-flash-next"
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _asked(**extra: Any) -> GenerateRequest:
    """A generate request carrying `toolChoice` and `topK`, the two fields these
    tests have a backend ignore."""
    fields: dict[str, Any] = {
        "messages": [Message(role=Role.user, content="hi")],
        "toolChoice": ToolChoice.required,
        "topK": 5,
    }
    return GenerateRequest(**{**fields, **extra})


def _slot(*fakes: FakeDriverClient) -> FailoverDriverClient:
    return FailoverDriverClient(name="slot", candidates=list(fakes))  # type: ignore[arg-type]


def _wired(handler: Any) -> HttpDriverClient:
    client = HttpDriverClient(name="d", base_url="http://d.invalid")
    client._client = httpx.AsyncClient(
        base_url=client.base_url, transport=httpx.MockTransport(handler)
    )
    return client


# --------------------------------------------------------------------------- #
# The driver's header, read off its wire
# --------------------------------------------------------------------------- #

WHOLE = {"content": "ok", "finishReason": "stop", "backend": "openai_compat_http"}
STREAM = (
    'event: token\ndata: {"text":"ok"}\n\n'
    'event: done\ndata: {"content":"ok","finishReason":"stop","backend":"openai_compat_http"}\n\n'
)


async def test_the_driver_client_reads_the_header_off_a_whole_answer() -> None:
    def wire(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=WHOLE, headers={HEADER: "toolChoice, topK"})

    answer = await _wired(wire).generate(_asked())
    assert answer.ignored_settings == ("toolChoice", "topK")
    # Beside the body and never in it: what the answer dumps to is the contract's.
    assert "ignored_settings" not in answer.model_dump()


async def test_a_header_with_junk_in_it_names_only_what_is_a_field_once() -> None:
    """A driver's words go out on the caller's response header, so a token that
    is not a field name is left off rather than passed on."""

    def wire(request: httpx.Request) -> httpx.Response:
        junk = "toolChoice, toolChoice ,, not a name, <script>, topK"
        return httpx.Response(200, json=WHOLE, headers={HEADER: junk})

    answer = await _wired(wire).generate(_asked())
    assert answer.ignored_settings == ("toolChoice", "topK")


async def test_a_driver_that_names_nothing_has_nothing_to_say() -> None:
    def wire(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=WHOLE)

    answer = await _wired(wire).generate(_asked())
    assert answer.ignored_settings == ()


async def test_a_stream_says_it_before_its_first_frame() -> None:
    def wire(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=STREAM,
            headers={"content-type": "text/event-stream", HEADER: "toolChoice"},
        )

    events = [e async for e in _wired(wire).stream(_asked())]
    assert events[0].ignored == ("toolChoice",)
    assert events[0].text == "" and not events[0].done
    assert [e.text for e in events[1:] if e.text] == ["ok"]
    assert events[-1].done and all(e.ignored is None for e in events[1:])


async def test_a_stream_that_names_nothing_sends_no_event_for_it() -> None:
    def wire(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=STREAM, headers={"content-type": "text/event-stream"})

    events = [e async for e in _wired(wire).stream(_asked())]
    assert all(e.ignored is None for e in events)


async def test_a_refused_request_names_nothing_even_if_the_refusal_carries_the_header() -> None:
    def wire(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"detail": {"title": "no", "status": 400, "type": "x"}},
            headers={HEADER: "topK"},
        )

    seen: list[StreamEvent] = []
    with pytest.raises(DriverError):
        async for event in _wired(wire).stream(_asked()):
            seen.append(event)
    assert seen == []
    with pytest.raises(DriverError):
        await _wired(wire).generate(_asked())


# --------------------------------------------------------------------------- #
# Only the attempt that served counts
# --------------------------------------------------------------------------- #


async def test_a_backend_that_failed_before_it_served_has_nothing_to_say_whole() -> None:
    dead = FakeDriverClient(name="dead", ignored=["toolChoice"])
    dead.generate_error = httpx.ConnectError("refused")
    alive = FakeDriverClient(name="alive", ignored=["topK"])
    slot = _slot(dead, alive)
    await slot.generate(_asked())
    assert slot.served_by == "alive"
    assert slot.ignored_settings == ["topK"]


async def test_a_backend_that_opened_and_failed_before_any_output_is_not_counted() -> None:
    """The case a header read at open gets wrong: attempt one opens its stream, so
    its driver has already said `toolChoice`, and dies before a token. Attempt two
    serves and says `topK`. The answer was sent and not honoured by the second."""
    dead = FakeDriverClient(name="dead", ignored=["toolChoice"])
    dead.stream_error_after = 0
    dead.stream_error = httpx.ConnectError("refused")
    alive = FakeDriverClient(name="alive", ignored=["topK"])
    slot = _slot(dead, alive)
    events = [e async for e in slot.stream(_asked())]
    assert slot.served_by == "alive" and slot.attempts == 2
    assert all(e.ignored is None for e in events), "the header is the slot's, never a door's"
    assert slot.ignored_settings == ["topK"]


async def test_the_attempt_in_flight_is_named_from_the_moment_its_stream_opens() -> None:
    """A stream's head can go out on a progress frame, before anything has
    served. The attempt then in flight is the best there is to say."""
    fake = FakeDriverClient(name="a", ignored=["toolChoice"])
    fake.progress = [{"stage": "prompt"}]
    slot = _slot(fake)
    events = slot.stream(_asked(reportProgress=True))
    first = await anext(events)
    assert first.progress is not None
    assert slot.ignored_settings == ["toolChoice"]
    await events.aclose()


async def test_a_backend_that_died_after_its_say_but_before_it_served_takes_it_back() -> None:
    fake = FakeDriverClient(name="a", ignored=["toolChoice"])
    fake.stream_error_after = 0
    slot = _slot(fake)
    with pytest.raises(RuntimeError):
        _ = [e async for e in slot.stream(_asked())]
    assert slot.ignored_settings == []


async def test_the_turns_of_one_answer_add_up_each_name_once_in_the_order_named() -> None:
    """The search loop calls one pinned slot turn after turn; its first turn
    forces a tool choice and later ones do not, so the answer's backend was sent
    `toolChoice` once and `topK` after. Both were not honoured."""
    fake = FakeDriverClient(name="a", ignored=["toolChoice", "topK"])
    slot = _slot(fake)
    await slot.generate(_asked(topK=None))
    assert slot.ignored_settings == ["toolChoice"]
    await slot.generate(_asked(toolChoice=None))
    assert slot.ignored_settings == ["toolChoice", "topK"]
    await slot.generate(_asked())
    assert slot.ignored_settings == ["toolChoice", "topK"]


# --------------------------------------------------------------------------- #
# The mapping, the order, the dedupe
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("door", ["chat", "messages", "responses"])
def test_tool_choice_is_tool_choice_on_every_door(door: str) -> None:
    assert ignored_settings.spelled("toolChoice", door) == "tool_choice"


@pytest.mark.parametrize(
    ("name", "door", "spelled"),
    [
        ("maxTokens", "chat", "max_tokens"),
        ("maxTokens", "messages", "max_tokens"),
        ("maxTokens", "responses", "max_output_tokens"),
        ("topK", "chat", "top_k"),
        ("topK", "messages", "top_k"),
        ("stop", "chat", "stop"),
        ("stop", "messages", "stop_sequences"),
        ("responseFormat", "chat", "response_format"),
        ("responseFormat", "messages", "output_config.format"),
        ("responseFormat", "responses", "text.format"),
        ("parallelToolCalls", "chat", "parallel_tool_calls"),
        ("parallelToolCalls", "messages", "tool_choice.disable_parallel_tool_use"),
        ("parallelToolCalls", "responses", "parallel_tool_calls"),
        ("temperature", "responses", "temperature"),
        # A field nobody has listed is the chat door's snake_case guess.
        ("mirostatTau", "chat", "mirostat_tau"),
        ("mirostatTau", "responses", "mirostat_tau"),
    ],
)
def test_a_field_is_spelled_the_way_the_door_s_caller_spells_it(
    name: str, door: str, spelled: str
) -> None:
    assert ignored_settings.spelled(name, door) == spelled


def test_the_chat_table_is_the_one_admission_words_its_refusals_from() -> None:
    """One table, not two: admission names the setting a backend lacks in the
    caller's words from the same place."""
    from eugene_plexus_gateway import admission

    assert not hasattr(admission, "_WIRE_NAMES")
    assert admission.WIRE_NAMES is ignored_settings.WIRE_NAMES


def test_the_doors_own_come_first_then_the_backends_each_once() -> None:
    merged = ignored_settings.merge(
        "service_tier, tool_choice", ["topK", "toolChoice", "topK", "stop"], "messages"
    )
    assert merged == "service_tier, tool_choice, top_k, stop_sequences"


def test_nothing_to_name_is_no_header() -> None:
    assert ignored_settings.merge(None, [], "chat") is None
    assert ignored_settings.merge("", [], "chat") is None
    assert ignored_settings.added_to({"x-a": "1"}, [], "chat") == {"x-a": "1"}


def test_a_doors_own_header_is_kept_when_the_backend_has_nothing_to_add() -> None:
    own = {HEADER: "output_config, cache_control", "x-a": "1"}
    assert ignored_settings.added_to(own, [], "messages") == own


def test_the_backend_alone_makes_the_header() -> None:
    assert ignored_settings.added_to({}, ["toolChoice"], "chat") == {HEADER: "tool_choice"}


def test_parse_keeps_the_order_the_driver_named_them_in() -> None:
    assert ignored_settings.parse("topK,toolChoice, topK") == ("topK", "toolChoice")
    assert ignored_settings.parse(None) == () and ignored_settings.parse(" , ") == ()


# --------------------------------------------------------------------------- #
# The three doors, whole and streamed
# --------------------------------------------------------------------------- #


def _backend(**knobs: Any) -> FakeDriverClient:
    """What Strata is to the gateway: it calls tools, and ignores `toolChoice`."""
    fake = FakeDriverClient(name="strata", model_id=MODEL, supports_tools=True, **knobs)
    fake.responses = ["ok"]
    return fake


def _app(settings: Settings, *fakes: FakeDriverClient, **table: Any) -> FastAPI:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*fakes, **table)
    return app


def _chat(**extra: Any) -> dict[str, Any]:
    return {"model": MODEL, "messages": [{"role": "user", "content": "weather?"}], **extra}


def _messages(**extra: Any) -> dict[str, Any]:
    tool = {
        "name": "get_weather",
        "description": "Look up the weather.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
    return claude_code_body(model=MODEL, tools=[tool], **extra)


def _post(client: TestClient, door: str, **extra: Any) -> httpx.Response:
    if door == "chat":
        return client.post("/v1/chat/completions", json=_chat(tools=[WEATHER], **extra))
    if door == "messages":
        return client.post("/v1/messages", json=_messages(**extra))
    return client.post("/v1/responses", json=codex_request(model=MODEL, **extra))


#: The door's body for a request that forces a tool, whole and streamed.
FORCED = {
    "chat": {"tool_choice": "required"},
    "messages": {"tool_choice": {"type": "any"}},
    "responses": {"tool_choice": "required"},
}


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("door", ["chat", "messages", "responses"])
def test_every_door_names_what_the_backend_was_sent_and_does_not_honour(
    settings: Settings, door: str, streamed: bool
) -> None:
    with TestClient(_app(settings, _backend())) as client:
        plain = _post(client, door, stream=streamed)
    with TestClient(_app(settings, _backend(ignored=["toolChoice"]))) as client:
        forced = _post(client, door, stream=streamed, **FORCED[door])
        # Not forced: the field the backend ignores is not on the request.
        unforced = _post(client, door, stream=streamed)
    assert plain.status_code == forced.status_code == unforced.status_code == 200
    own = plain.headers.get(HEADER)
    # The door's own first, then the backend's, in the door's own words.
    assert forced.headers[HEADER] == (f"{own}, tool_choice" if own else "tool_choice")
    assert unforced.headers.get(HEADER) == own


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize(
    ("door", "extra", "ignored", "named"),
    [
        ("chat", {"max_tokens": 50}, "maxTokens", "max_tokens"),
        ("messages", {"max_tokens": 50}, "maxTokens", "max_tokens"),
        ("messages", {"stop_sequences": ["END"]}, "stop", "stop_sequences"),
        ("responses", {"max_output_tokens": 50}, "maxTokens", "max_output_tokens"),
    ],
)
def test_a_door_names_a_field_the_way_its_own_caller_spelled_it(
    settings: Settings,
    door: str,
    extra: dict[str, Any],
    ignored: str,
    named: str,
    streamed: bool,
) -> None:
    with TestClient(_app(settings, _backend(ignored=[ignored]))) as client:
        r = _post(client, door, stream=streamed, **extra)
    assert r.status_code == 200, r.text
    assert r.headers[HEADER].split(", ")[-1] == named


def test_a_chat_stream_that_reports_progress_is_named_from_its_first_frame(
    settings: Settings,
) -> None:
    """With `include_progress` the first frame is the backend's running count,
    before any token and before any attempt has served. The head goes out with
    it, and what the attempt then in flight said is what it can carry."""
    fake = _backend(ignored=["toolChoice"])
    fake.progress = [{"stage": "prompt", "promptTokens": 9, "processedTokens": 4}]
    options = {"include_progress": True}
    with TestClient(_app(settings, fake)) as client:
        r = _post(client, "chat", stream=True, stream_options=options, **FORCED["chat"])
    assert r.headers[HEADER] == "tool_choice"
    first = json.loads(r.text.split("\n\n")[0][len("data: ") :])
    assert first["choices"] == [] and first["x_eugene_plexus"]["progress"]["stage"] == "prompt"


def test_the_chat_door_which_named_nothing_before_names_it_alone(settings: Settings) -> None:
    with TestClient(_app(settings, _backend(ignored=["toolChoice"]))) as client:
        whole = _post(client, "chat", **FORCED["chat"])
        streamed = _post(client, "chat", stream=True, **FORCED["chat"])
        silent = _post(client, "chat")
    assert whole.headers[HEADER] == streamed.headers[HEADER] == "tool_choice"
    assert HEADER not in silent.headers


def test_the_messages_door_keeps_the_hints_it_dropped_ahead_of_the_backends(
    settings: Settings,
) -> None:
    with TestClient(_app(settings, _backend(ignored=["toolChoice"]))) as client:
        r = _post(client, "messages", **FORCED["messages"])
    own, backend = r.headers[HEADER].rsplit(", ", 1)
    assert backend == "tool_choice"
    # `context_management` and an `output_config` effort are on every Claude Code request.
    assert own.split(", ")[:2] == ["context_management", "output_config"]


def test_a_name_both_the_door_and_the_backend_give_is_one_entry(settings: Settings) -> None:
    """The backend names `toolChoice` twice, beside the door's own `service_tier`."""
    fake = _backend(ignored=["toolChoice", "toolChoice"])
    with TestClient(_app(settings, fake)) as client:
        r = _post(client, "responses", service_tier="priority", **FORCED["responses"])
    names = r.headers[HEADER].split(", ")
    assert names.count("tool_choice") == 1
    assert "service_tier" in names and names.index("service_tier") < names.index("tool_choice")


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("door", ["chat", "messages", "responses"])
def test_only_the_backend_that_served_is_named_through_a_failover(
    settings: Settings, door: str, streamed: bool
) -> None:
    """The first tier is a backend that is down and would have named `toolChoice`
    (a streamed one even opens first). The second serves and names `topK`."""
    dead = FakeDriverClient(
        name="dead", model_id="dead-model", supports_tools=True, ignored=["toolChoice"]
    )
    dead.generate_error = httpx.ConnectError("refused")
    if streamed:
        dead.generate_error = None
        dead.stream_error_after = 0
        dead.stream_error = httpx.ConnectError("refused")
    alive = FakeDriverClient(
        name="alive", model_id="alive-model", supports_tools=True, ignored=["topK"]
    )
    alive.responses = ["ok"]
    slots = [{"model": MODEL, "targets": ["dead-model", "alive-model"]}]
    with TestClient(_app(settings, dead, alive, slots=slots)) as client:
        sent = {"top_k": 5} if door != "responses" else {}
        r = _post(client, door, stream=streamed, **FORCED[door], **sent)
    assert r.status_code == 200, r.text
    assert dead.calls and alive.calls, "the first was tried and the second served"
    if door == "responses":
        # The Responses door has no `top_k`: its request cannot carry the field
        # that backend ignores, so there is nothing to name.
        assert "tool_choice" not in r.headers.get(HEADER, "")
        return
    assert "tool_choice" not in r.headers[HEADER].split(", ")
    assert r.headers[HEADER].split(", ")[-1] == "top_k"


class OverTheWire(FakeDriverClient):
    """The same backend, answering through the driver's HTTP contract: the
    routing table, the slot, the bound client and `HttpDriverClient` are the
    real ones, and only the socket is a `MockTransport`.

    A component test is not a wiring test. The tests above prove each door
    against a fake that hands the slot its header already read; this one shows
    the header on the driver's response is what reaches the door's."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.wire = _wired(self._driver)

    def _driver(self, request: httpx.Request) -> httpx.Response:
        asked = GenerateRequest.model_validate(json.loads(request.content))
        self.calls.append(asked)
        named = {HEADER: ", ".join(self.named(asked))} if self.named(asked) else {}
        if request.url.path == "/v1/generate":
            return httpx.Response(200, json={**WHOLE, "modelId": asked.model}, headers=named)
        return httpx.Response(
            200, text=STREAM, headers={"content-type": "text/event-stream", **named}
        )

    async def generate(self, request: GenerateRequest) -> DriverAnswer:
        return await self.wire.generate(request)

    def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:  # type: ignore[override]
        return self.wire.stream(request)


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("door", ["chat", "messages", "responses"])
def test_the_header_on_the_drivers_wire_is_the_header_on_the_doors(
    settings: Settings, door: str, streamed: bool
) -> None:
    fake = OverTheWire(name="strata", model_id=MODEL, supports_tools=True, ignored=["toolChoice"])
    with TestClient(_app(settings, fake)) as client:
        r = _post(client, door, stream=streamed, **FORCED[door])
        silent = _post(client, door, stream=streamed)
    assert r.status_code == 200, r.text
    assert r.headers[HEADER].split(", ")[-1] == "tool_choice"
    assert "tool_choice" not in silent.headers.get(HEADER, "").split(", ")
    assert len(fake.calls) == 2


# --------------------------------------------------------------------------- #
# The Responses door opens early, and what that costs
# --------------------------------------------------------------------------- #


class OpensThenPrefills(FakeDriverClient):
    """A backend whose driver answers at once and then says nothing while it
    prefills: the response head is early, the first frame is not."""

    delay = 0.4

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        if named := self.named(request):
            yield StreamEvent(ignored=named)
        await asyncio.sleep(self.delay)
        async for event in super().stream(request):
            if event.ignored is None:
                yield event


class PrefillsThenOpens(FakeDriverClient):
    """The real driver today: it awaits the engine's first frame before it
    answers, so its response head, and the header on it, arrive with the first
    token."""

    delay = 1.0

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        await asyncio.sleep(self.delay)
        async for event in super().stream(request):
            yield event


def _slow(cls: type[FakeDriverClient]) -> FakeDriverClient:
    fake = cls(name="strata", model_id=MODEL, supports_tools=True, ignored=["toolChoice"])
    fake.responses = ["ok"]
    return fake


def test_the_responses_stream_names_a_backend_that_opened_while_it_prefills(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(responses, "KEEPALIVE_SECONDS", 0.05)
    with TestClient(_app(settings, _slow(OpensThenPrefills))) as client:
        r = _post(client, "responses", stream=True, **FORCED["responses"])
    assert r.headers[HEADER].split(", ")[-1] == "tool_choice"
    assert "response.completed" in r.text


def test_the_responses_head_does_not_wait_past_the_first_keepalive(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex drops a stream that is silent for five minutes, before its first
    event too (record §5), so a backend still prefilling is not waited for. Its
    stream opens, says it is working, and cannot yet say what the backend
    was sent and does not honour: the one place the head does not name it."""
    monkeypatch.setattr(responses, "KEEPALIVE_SECONDS", 0.05)
    with TestClient(_app(settings, _slow(PrefillsThenOpens))) as client:
        r = _post(client, "responses", stream=True, **FORCED["responses"])
    assert "tool_choice" not in r.headers.get(HEADER, "").split(", ")
    names = [line[7:] for line in r.text.splitlines() if line.startswith("event: ")]
    assert names[:2] == ["response.created", "response.in_progress"]
    assert names.count("response.in_progress") >= 3 and names[-1] == "response.completed"


class Silent(FakeDriverClient):
    """A backend whose stream opens and ends without a frame: no output, no `done`."""

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        self.calls.append(request)
        return
        yield StreamEvent()  # pragma: no cover


def test_a_responses_stream_that_fails_before_the_backend_is_heard_from_still_opens(
    settings: Settings,
) -> None:
    """Codex has not seen a stream start until `response.created`, and a bare
    `response.failed` is one it reports without our message (record §4). The
    stream opens with the backend's first frame, so a backend that fails before
    it owes the opening all the same."""
    fake = _backend()
    fake.stream_error_after = 0
    fake.stream_error = httpx.ConnectError("gone")
    with TestClient(_app(settings, fake)) as client:
        r = _post(client, "responses", stream=True)
    names = [line[7:] for line in r.text.splitlines() if line.startswith("event: ")]
    assert names == ["response.created", "response.in_progress", "response.failed"]


def test_a_responses_stream_that_ends_without_a_frame_still_opens_before_it_fails(
    settings: Settings,
) -> None:
    silent = Silent(name="strata", model_id=MODEL, supports_tools=True)
    with TestClient(_app(settings, silent)) as client:
        r = _post(client, "responses", stream=True)
    names = [line[7:] for line in r.text.splitlines() if line.startswith("event: ")]
    assert names == ["response.created", "response.in_progress", "response.failed"]
    assert '"code": "server_error"' in r.text


# --------------------------------------------------------------------------- #
# A response whose head waits for its first frame
# --------------------------------------------------------------------------- #


async def _serve(response: Any, *, disconnect_after: float | None = None) -> list[AsgiMessage]:
    sent: list[AsgiMessage] = []

    async def receive() -> AsgiMessage:
        if disconnect_after is None:
            await asyncio.sleep(3600)
        else:
            await asyncio.sleep(disconnect_after)
        return {"type": "http.disconnect"}

    async def send(message: AsgiMessage) -> None:
        sent.append(message)

    await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
    return sent


def _headers(start: AsgiMessage) -> dict[str, str]:
    return {k.decode(): v.decode() for k, v in start["headers"]}


async def test_the_head_is_sent_with_the_first_frame_and_carries_what_was_learned() -> None:
    learned: list[str] = []

    async def frames() -> AsyncGenerator[str, None]:
        learned.append("the backend answered")
        yield "one"
        yield "two"

    def late() -> dict[str, str]:
        # Called after the first frame exists, which is when it can know.
        return {"x-learned": ",".join(learned)}

    sent = await _serve(
        LateHeadStreamingResponse(frames(), media_type="text/event-stream", late=late)
    )
    assert [m["type"] for m in sent] == ["http.response.start"] + ["http.response.body"] * 3
    assert _headers(sent[0])["x-learned"] == "the backend answered"
    assert _headers(sent[0])["content-type"].startswith("text/event-stream")
    assert [m["body"] for m in sent[1:]] == [b"one", b"two", b""]
    assert [m["more_body"] for m in sent[1:]] == [True, True, False]


async def test_a_header_set_at_the_start_survives_a_late_one_of_the_same_name() -> None:
    async def frames() -> AsyncGenerator[str, None]:
        yield "one"

    response = LateHeadStreamingResponse(
        frames(), headers={"x-a": "early", "x-b": "kept"}, late=lambda: {"x-a": "merged"}
    )
    sent = await _serve(response)
    headers = _headers(sent[0])
    assert headers["x-a"] == "merged" and headers["x-b"] == "kept"
    assert [k for k, _ in sent[0]["headers"]].count(b"x-a") == 1


async def test_a_stream_that_says_nothing_still_gets_a_head() -> None:
    async def frames() -> AsyncGenerator[str, None]:
        return
        yield ""  # pragma: no cover

    sent = await _serve(LateHeadStreamingResponse(frames(), late=lambda: {"x-late": "1"}))
    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]
    assert _headers(sent[0])["x-late"] == "1" and sent[1]["body"] == b""


async def test_a_stream_that_fails_before_its_first_frame_fails_before_its_head() -> None:
    async def frames() -> AsyncGenerator[str, None]:
        raise RuntimeError("the backend is gone")
        yield ""  # pragma: no cover

    served: list[AsgiMessage] = []

    async def send(message: AsgiMessage) -> None:
        served.append(message)

    async def receive() -> AsgiMessage:
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    response = LateHeadStreamingResponse(frames(), late=lambda: {})
    with pytest.raises(RuntimeError, match="gone"):
        await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
    assert served == [], "no 200 was sent for a stream that never started"


async def test_a_header_that_cannot_be_built_does_not_fail_the_stream() -> None:
    async def frames() -> AsyncGenerator[str, None]:
        yield "one"

    def late() -> dict[str, str]:
        raise RuntimeError("a diagnostic")

    sent = await _serve(LateHeadStreamingResponse(frames(), late=late))
    assert sent[0]["type"] == "http.response.start"
    assert sent[1]["body"] == b"one"


async def test_a_client_that_leaves_while_the_head_waits_cancels_the_backend() -> None:
    started = asyncio.Event()
    closed = asyncio.Event()

    async def frames() -> AsyncGenerator[str, None]:
        try:
            started.set()
            await asyncio.sleep(3600)
            yield "never"
        finally:
            closed.set()

    sent = await _serve(LateHeadStreamingResponse(frames(), late=lambda: {}), disconnect_after=0.05)
    assert started.is_set() and closed.is_set()
    assert sent == []


# --------------------------------------------------------------------------- #
# Unknown counts as yes, in each place; only `false` is evidence
# --------------------------------------------------------------------------- #


def _models(client: TestClient) -> dict[str, dict[str, Any]]:
    return {m["id"]: m["x_eugene_plexus"] for m in client.get("/v1/models").json()["data"]}


UNSAID = [
    pytest.param({"supports_tools": None}, id="toolCalling not reported"),
    pytest.param({"capabilities_reported": False}, id="no capabilities at all"),
]


@pytest.mark.parametrize("knobs", UNSAID)
def test_the_listing_says_a_backend_that_does_not_say_calls_tools(
    settings: Settings, knobs: dict[str, Any]
) -> None:
    fake = FakeDriverClient(name="custom", model_id=MODEL, **knobs)
    with TestClient(_app(settings, fake)) as client:
        listed = _models(client)[MODEL]
    assert listed["tool_calling"] is True
    assert listed["web_search"] is True


def test_the_listing_still_says_no_to_a_driver_that_says_no(settings: Settings) -> None:
    fake = FakeDriverClient(name="cli", model_id=MODEL, supports_tools=False)
    with TestClient(_app(settings, fake)) as client:
        listed = _models(client)[MODEL]
    assert listed["tool_calling"] is False
    assert listed["web_search"] is False


@pytest.mark.parametrize(
    ("reports", "tools"),
    [
        ([True, None], True),
        ([None, None], True),
        ([None, False], False),
        ([True, False], False),
    ],
)
def test_one_backend_that_says_no_takes_the_weakest_answer_and_unsaid_does_not(
    settings: Settings, reports: list[bool | None], tools: bool
) -> None:
    fakes = [
        FakeDriverClient(name=f"gpu-{i}", model_id=MODEL, supports_tools=r)
        for i, r in enumerate(reports)
    ]
    with TestClient(_app(settings, *fakes)) as client:
        assert _models(client)[MODEL]["tool_calling"] is tools


def test_an_embedding_model_that_does_not_say_is_not_one_that_calls_tools(
    settings: Settings,
) -> None:
    """The rule is about a chat model. An embedding model's silence is not a yes."""
    fake = FakeDriverClient(
        name="embed", model_id="nomic", supports_embeddings=True, capabilities_reported=False
    )
    with TestClient(_app(settings, fake)) as client:
        assert _models(client)["nomic"]["tool_calling"] is False


@pytest.mark.parametrize("knobs", UNSAID)
def test_a_request_with_tools_is_asked_of_a_backend_that_does_not_say(
    settings: Settings, knobs: dict[str, Any]
) -> None:
    """Troy's case: a custom engine endpoint under test. It was refused its tools
    because nobody had said it takes them."""
    fake = FakeDriverClient(name="custom", model_id=MODEL, **knobs)
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/chat/completions", json=_chat(tools=[WEATHER]))
    assert r.status_code == 200, r.text
    assert fake.calls[0].tools is not None
    assert fake.calls[0].tools[0].function.name == "get_weather"


def test_a_driver_that_says_no_is_still_refused_before_anything_is_sent(
    settings: Settings,
) -> None:
    fake = FakeDriverClient(name="cli", model_id=MODEL, supports_tools=False)
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/chat/completions", json=_chat(tools=[WEATHER]))
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "tools"
    assert fake.calls == []


def test_a_listing_that_names_no_settings_does_not_decide_tools(settings: Settings) -> None:
    """Admission re-reads the model's listing for a request that carries
    settings, and `tools` is one: a driver that says nothing of tool calling takes
    them whatever its list of settings holds. Only `false` decides."""
    fake = FakeDriverClient(name="custom", model_id=MODEL, supports_tools=None)
    fake.supported_settings = ["maxTokens"]
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/chat/completions", json=_chat(tools=[WEATHER]))
    assert r.status_code == 200, r.text

    other = FakeDriverClient(name="custom", model_id=MODEL, supports_tools=None)
    other.supported_settings = ["maxTokens"]
    with TestClient(_app(settings, other)) as client:
        # Another setting the backend does not list is still refused, by name.
        r = client.post("/v1/chat/completions", json=_chat(tools=[WEATHER], top_k=5))
    assert r.status_code == 400
    assert "top_k" in r.json()["error"]["message"]
    assert other.calls == []


@pytest.mark.parametrize(("said", "counted"), [(None, True), (False, False)])
def test_a_token_count_with_tools_follows_the_same_rule(
    settings: Settings, said: bool | None, counted: bool
) -> None:
    fake = FakeDriverClient(name="custom", model_id=MODEL, supports_tools=said)
    fake.prompt_tokens = 42
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/messages/count_tokens", json=_messages())
    if counted:
        assert r.status_code == 200 and r.json() == {"input_tokens": 42}
    else:
        assert r.status_code == 400
        assert "does not call tools" in r.json()["error"]["message"]


def test_a_search_is_offered_to_a_backend_that_does_not_say_it_calls_tools(
    settings: Settings,
) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    model.supports_tools = None
    with TestClient(app_with(settings, model, searches=[account(FakeSearch())])) as client:
        r = chat(client)
    assert r.status_code == 200, r.text
    assert r.json()["x_eugene_plexus"]["web_searches"] == 1


def test_a_search_is_still_refused_for_a_backend_that_says_it_does_not(
    settings: Settings,
) -> None:
    model = ScriptedDriver(turns=[Turn("no search")])
    model.supports_tools = False
    with TestClient(app_with(settings, model, searches=[account(FakeSearch())])) as client:
        r = chat(client)
    assert r.status_code == 400
    assert "can search the web" in r.json()["error"]["message"]
    assert model.calls == []


class IgnoringScripted(ScriptedDriver):
    """The scripted model of the search tests, behind a driver that names what
    its backend ignores, as `FakeDriverClient` does on its own answers."""

    async def generate(self, request: GenerateRequest) -> DriverAnswer:
        answer = DriverAnswer.model_validate((await super().generate(request)).model_dump())
        answer.ignored_settings = self.named(request)
        return answer

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        if named := self.named(request):
            yield StreamEvent(ignored=named)
        async for event in super().stream(request):
            yield event


def test_the_search_loops_forced_first_turn_is_named_when_the_backend_ignores_it(
    settings: Settings,
) -> None:
    """The loop asks the model's first turn for a tool (`required`) when search
    is the caller's only tool. A backend that ignores that is named, so the
    answer's caller learns the search was not forced; the loop's later turns send
    no choice, and the name is the answer's all the same."""
    model = IgnoringScripted(
        turns=[Turn(calls=[search_call()]), Turn(ANSWER)], ignored=["toolChoice"]
    )
    with TestClient(app_with(settings, model, searches=[account(FakeSearch())])) as client:
        r = chat(client)
    assert r.status_code == 200, r.text
    assert str(model.calls[0].toolChoice) == "required" and model.calls[1].toolChoice is None
    assert r.headers[HEADER] == "tool_choice"


def test_the_rule_in_one_place() -> None:
    from eugene_plexus_gateway._generated.driver_models import Capabilities, DriverModel

    def model(caps: Capabilities | None, *surfaces: str) -> DriverModel:
        return DriverModel(id="m", surfaces=list(surfaces or ["chat"]), capabilities=caps)

    assert calls_tools(None)
    assert calls_tools(model(None))
    assert calls_tools(model(Capabilities()))
    assert calls_tools(model(Capabilities(toolCalling=True)))
    assert not calls_tools(model(Capabilities(toolCalling=False)))
    assert not calls_tools(model(None, "embeddings"))
    assert calls_tools(model(Capabilities(toolCalling=True), "embeddings"))
