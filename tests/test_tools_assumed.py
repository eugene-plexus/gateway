"""Tools are assumed, and turned off only on evidence (Troy, 2026-10-12).

"We should always assume in 2026 that a model & engine can use tools." The
design is `specs/docs/design/tools-assumed.md` (round two, contract specs
6178255); this is the gateway's half:

- **What the chosen backend does not honour is named**, from its listing and
  before anything is sent: the settings the caller sent, with a value the
  backend's `capabilities.settingsNotHonoured` lists, said in each door's own
  spelling on the door's `x-eugene-plexus-ignored-settings`, on both response
  modes. The list is static per backend, so **a stream opens as it always has**:
  its head never waits for the backend's first frame. (An earlier round read a
  header off the driver's response instead, which held every stream's head,
  broke the Responses door's opening and turned a failure before output into a
  bare status.)
- **A backend whose driver does not say counts as one that calls tools**, in
  every place the gateway decides: `GET /v1/models`, the front-door refusal, the
  admission check, a search's candidates. Only `toolCalling: false` is evidence.
  Of a chat model: one that does not chat is sent to the door that serves it.
- **The refusal is the driver's report**, never the model failing, and is worded
  the same wherever it is made.

The driver is a fake of the driver's contract (`FakeDriverClient`), and the real
routing table and `HttpDriverClient` are shown the same contract over a socket:
a component test is not a wiring test.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncGenerator, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.types import Message as AsgiMessage

from eugene_plexus_gateway import admission, ignored_settings, responses, server_tools
from eugene_plexus_gateway._generated.driver_models import (
    Capabilities,
    DriverModel,
    Function,
    FunctionDefinition,
    GenerateRequest,
    Message,
    NamedToolChoice,
    Role,
    SettingNotHonoured,
    Tool,
    ToolChoice,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.capabilities import calls_tools, chats, said_tools, tools_refused
from eugene_plexus_gateway.driver_client import StreamEvent
from eugene_plexus_gateway.routes import inference
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

#: What the Strata driver lists as its engine's measured gaps (2026-10-12): it
#: ignores `tool_choice` `required` and a named function, and
#: `parallel_tool_calls: false`. Not `auto`, not `none` (the driver honours that
#: by sending no tools), not `parallel_tool_calls: true`.
STRATA_GAPS: dict[str, list[str | bool]] = {
    "toolChoice": ["required", "function"],
    "parallelToolCalls": [False],
}


def _req(**fields: Any) -> GenerateRequest:
    """A generate request as the gateway builds one for a caller."""
    return GenerateRequest(messages=[Message(role=Role.user, content="hi")], **fields)


def _listing(**gaps: list[str | bool]) -> list[SettingNotHonoured]:
    return [SettingNotHonoured(setting=name, values=values) for name, values in gaps.items()]


STRATA = _listing(**STRATA_GAPS)
NAMED = NamedToolChoice(type="function", function=Function(name="get_weather"))
TOOL = Tool(type="function", function=FunctionDefinition(name="get_weather"))
#: A request carrying tools, a choice and the parallel switch, all the caller's.
WITH_TOOLS: dict[str, Any] = {
    "tools": [TOOL],
    "toolChoice": ToolChoice.auto,
    "parallelToolCalls": False,
    "callerSettings": ["tools", "toolChoice", "parallelToolCalls"],
}


# --------------------------------------------------------------------------- #
# The rule in one place
# --------------------------------------------------------------------------- #


def _model(caps: Capabilities | None, *surfaces: str) -> DriverModel:
    return DriverModel(id="m", surfaces=list(surfaces or ["chat"]), capabilities=caps)


def test_the_rule_in_one_place() -> None:
    assert calls_tools(None)
    assert calls_tools(_model(None))
    assert calls_tools(_model(Capabilities()))
    assert calls_tools(_model(Capabilities(toolCalling=True)))
    assert not calls_tools(_model(Capabilities(toolCalling=False)))
    # Of a chat model: one that does not chat is not one that calls tools
    # because nothing said it does not. A driver's own word still stands.
    assert not calls_tools(_model(None, "embeddings"))
    assert not calls_tools(_model(Capabilities(), "rerank"))
    assert calls_tools(_model(Capabilities(toolCalling=True), "embeddings"))


def test_what_a_driver_said_is_told_apart_from_what_is_assumed() -> None:
    assert said_tools(None) is None
    assert said_tools(_model(None)) is None
    assert said_tools(_model(Capabilities())) is None
    assert said_tools(_model(Capabilities(toolCalling=True))) is True
    assert said_tools(_model(Capabilities(toolCalling=False))) is False
    # Assumed is not said: an embedding model's silence is neither.
    assert said_tools(_model(None, "embeddings")) is None


def test_a_model_is_a_chat_model_unless_its_listed_surfaces_leave_chat_out() -> None:
    assert chats(None)
    assert chats(_model(None))
    assert chats(_model(None, "chat", "embeddings"))
    assert not chats(_model(None, "embeddings"))
    assert not chats(_model(None, "rerank"))
    assert not chats(_model(None, "speech", "transcription"))
    # Nothing listed is not a model shown to be anything else.
    assert chats(DriverModel(id="m", surfaces=[]))


@pytest.mark.parametrize("backends", [1, 2])
def test_the_refusal_is_the_drivers_report_and_never_the_models_failing(backends: int) -> None:
    said = tools_refused("qwen", backends)
    assert "'qwen'" in said
    assert "reports that tools cannot be carried to it" in said
    assert "capabilities.toolCalling: false" in said
    assert "Nothing was forwarded or woken." in said
    # Nobody tested a model's tools and found it failing: the driver reported.
    for model_failing in (
        "does not call tools",
        "cannot call tools",
        "No backend",
        "Every backend",
    ):
        assert model_failing not in said
    assert ("its driver reports" in said) == (backends == 1)
    assert ("every driver serving it reports" in said) == (backends > 1)
    # The front door's refusal is this sentence, so the two cannot drift.
    assert inference._tools_unsupported("qwen", backends).message == said


# --------------------------------------------------------------------------- #
# Reading a listing against a request
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("choice", "mode"),
    [
        (ToolChoice.auto, "auto"),
        (ToolChoice.none, "none"),
        (ToolChoice.required, "required"),
        (NAMED, "function"),
    ],
    ids=["auto", "none", "required", "a named function"],
)
def test_a_tool_choice_is_read_as_its_mode(choice: Any, mode: str) -> None:
    request = _req(toolChoice=choice, callerSettings=["toolChoice"])
    assert ignored_settings.sent(request, "toolChoice") == mode


@pytest.mark.parametrize("value", [True, False])
def test_a_boolean_setting_is_read_as_itself_and_false_is_a_value(value: bool) -> None:
    request = _req(parallelToolCalls=value, callerSettings=["parallelToolCalls"])
    assert ignored_settings.sent(request, "parallelToolCalls") is value


def test_what_the_caller_did_not_send_is_not_read() -> None:
    """`callerSettings` is the caller's own list. The search loop forces a first
    turn's `required` on a request of its own making, and a default is never the
    caller's; neither is a setting the caller sent."""
    forced = _req(toolChoice=ToolChoice.required)
    assert ignored_settings.sent(forced, "toolChoice") is None
    other = _req(toolChoice=ToolChoice.required, callerSettings=["tools"])
    assert ignored_settings.sent(other, "toolChoice") is None
    assert ignored_settings.not_honoured(STRATA, forced) == ()
    # Nor what nobody could have sent: a name the request has no field for.
    assert ignored_settings.sent(_req(callerSettings=["nonesuch"]), "nonesuch") is None
    assert ignored_settings.sent(_req(callerSettings=["messages"]), "messages") is None


def test_a_setting_is_named_only_for_a_value_the_listing_names() -> None:
    def named(**fields: Any) -> tuple[str, ...]:
        return ignored_settings.not_honoured(STRATA, _req(**fields))

    def sent(**fields: Any) -> dict[str, Any]:
        return {"callerSettings": [name for name in fields if name != "callerSettings"], **fields}

    assert named(**sent(toolChoice=ToolChoice.required)) == ("toolChoice",)
    assert named(**sent(toolChoice=NAMED)) == ("toolChoice",)
    assert named(**sent(toolChoice=ToolChoice.auto)) == ()
    assert named(**sent(toolChoice=ToolChoice.none)) == ()
    assert named(**sent(parallelToolCalls=False)) == ("parallelToolCalls",)
    assert named(**sent(parallelToolCalls=True)) == ()
    assert named() == ()


def test_the_listing_decides_the_order_and_each_setting_is_named_once() -> None:
    request = _req(
        toolChoice=ToolChoice.required,
        parallelToolCalls=False,
        callerSettings=["parallelToolCalls", "toolChoice"],
    )
    assert ignored_settings.not_honoured(STRATA, request) == ("toolChoice", "parallelToolCalls")
    twice = _listing(toolChoice=["required"]) + _listing(toolChoice=["function", "required"])
    assert ignored_settings.not_honoured(twice, request) == ("toolChoice",)


def test_a_boolean_in_a_listing_is_not_a_string() -> None:
    request = _req(parallelToolCalls=False, callerSettings=["parallelToolCalls"])
    assert ignored_settings.not_honoured(_listing(parallelToolCalls=["false"]), request) == ()
    assert ignored_settings.not_honoured(_listing(parallelToolCalls=[True]), request) == ()
    assert ignored_settings.not_honoured(_listing(parallelToolCalls=[False]), request) == (
        "parallelToolCalls",
    )


def test_a_backend_with_no_listing_names_nothing() -> None:
    request = _req(toolChoice=ToolChoice.required, callerSettings=["toolChoice"])
    assert ignored_settings.not_honoured(None, request) == ()
    assert ignored_settings.not_honoured([], request) == ()


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


def test_the_driver_per_request_header_is_not_read_by_this_module() -> None:
    """The gateway names from the listing. Reading a header off each response is
    what held every stream's head, and nothing here parses one."""
    assert not hasattr(ignored_settings, "parse")


# --------------------------------------------------------------------------- #
# The three doors, whole and streamed
# --------------------------------------------------------------------------- #


def _backend(name: str = "strata", **knobs: Any) -> FakeDriverClient:
    """What Strata is to the gateway: it calls tools, and its listing names its gaps."""
    knobs.setdefault("supports_tools", True)
    knobs.setdefault("not_honoured", STRATA_GAPS)
    fake = FakeDriverClient(name=name, model_id=knobs.pop("model_id", MODEL), **knobs)
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


def _responses(**extra: Any) -> dict[str, Any]:
    """Codex's request with the two settings it always sends taken out, so a case
    says exactly what the caller sent. `codex_request()` itself is the Codex case."""
    return codex_request(model=MODEL, **{"tool_choice": None, "parallel_tool_calls": None, **extra})


def _post(client: TestClient, door: str, *, stream: bool = False, **extra: Any) -> httpx.Response:
    if door == "chat":
        return client.post(
            "/v1/chat/completions", json=_chat(tools=[WEATHER], stream=stream, **extra)
        )
    if door == "messages":
        return client.post("/v1/messages", json=_messages(stream=stream, **extra))
    return client.post("/v1/responses", json=_responses(stream=stream, **extra))


def _header(response: httpx.Response) -> list[str]:
    value = response.headers.get(HEADER, "")
    return value.split(", ") if value else []


#: The door's body for a request that forces a tool.
FORCED = {
    "chat": {"tool_choice": "required"},
    "messages": {"tool_choice": {"type": "any"}},
    "responses": {"tool_choice": "required"},
}

#: What each door's caller sends and what it is named as, on a backend that
#: lists `toolChoice` `required` and `function` and `parallelToolCalls` `false`.
#: A choice is its mode (`auto`, `none`, `required`, `function`); Anthropic's
#: `any` is `required`, its `tool` is a named function, and its
#: `disable_parallel_tool_use: true` is `parallel_tool_calls: false`.
NAMED_ON = {
    "chat": [
        ("required", {"tool_choice": "required"}, ["tool_choice"]),
        (
            "a named function",
            {"tool_choice": {"type": "function", "function": {"name": "get_weather"}}},
            ["tool_choice"],
        ),
        ("auto", {"tool_choice": "auto"}, []),
        ("none", {"tool_choice": "none"}, []),
        ("parallel false", {"parallel_tool_calls": False}, ["parallel_tool_calls"]),
        ("parallel true", {"parallel_tool_calls": True}, []),
        (
            "both",
            {"tool_choice": "required", "parallel_tool_calls": False},
            ["tool_choice", "parallel_tool_calls"],
        ),
        ("neither", {}, []),
    ],
    "messages": [
        ("any", {"tool_choice": {"type": "any"}}, ["tool_choice"]),
        (
            "a named tool",
            {"tool_choice": {"type": "tool", "name": "get_weather"}},
            ["tool_choice"],
        ),
        ("auto", {"tool_choice": {"type": "auto"}}, []),
        ("none", {"tool_choice": {"type": "none"}}, []),
        (
            "parallel disabled",
            {"tool_choice": {"type": "auto", "disable_parallel_tool_use": True}},
            ["tool_choice.disable_parallel_tool_use"],
        ),
        (
            "parallel not disabled",
            {"tool_choice": {"type": "auto", "disable_parallel_tool_use": False}},
            [],
        ),
        (
            "both",
            {"tool_choice": {"type": "any", "disable_parallel_tool_use": True}},
            ["tool_choice", "tool_choice.disable_parallel_tool_use"],
        ),
        ("neither", {}, []),
    ],
    "responses": [
        ("required", {"tool_choice": "required"}, ["tool_choice"]),
        (
            "a named function",
            {"tool_choice": {"type": "function", "name": "shell"}},
            ["tool_choice"],
        ),
        ("auto", {"tool_choice": "auto"}, []),
        ("none", {"tool_choice": "none"}, []),
        ("parallel false", {"parallel_tool_calls": False}, ["parallel_tool_calls"]),
        ("parallel true", {"parallel_tool_calls": True}, []),
        (
            "both",
            {"tool_choice": "required", "parallel_tool_calls": False},
            ["tool_choice", "parallel_tool_calls"],
        ),
        ("neither", {}, []),
    ],
}
CASES = [
    pytest.param(door, extra, named, id=f"{door}: {label}")
    for door, cases in NAMED_ON.items()
    for label, extra, named in cases
]


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize(("door", "extra", "named"), CASES)
def test_a_door_names_what_the_caller_sent_and_the_backend_does_not_honour(
    settings: Settings, door: str, extra: dict[str, Any], named: list[str], streamed: bool
) -> None:
    """The door's own names first, then the backend's in its listing's order, in
    the door's own spelling: `own` is what the same request says without any
    tool setting, so what is added is exactly `named`, once each."""
    fake = _backend()
    with TestClient(_app(settings, fake)) as client:
        own = _header(_post(client, door, stream=streamed))
        r = _post(client, door, stream=streamed, **extra)
    assert r.status_code == 200, r.text
    assert _header(r) == own + named
    # Answered, not refused: both requests reached the backend, as sent.
    assert len(fake.calls) == 2


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
def test_a_request_that_sends_nothing_the_backend_ignores_has_no_header_on_the_chat_door(
    settings: Settings, streamed: bool
) -> None:
    with TestClient(_app(settings, _backend())) as client:
        r = _post(client, "chat", stream=streamed)
    assert r.status_code == 200
    assert HEADER not in r.headers


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
def test_a_backend_that_lists_nothing_is_never_named(settings: Settings, streamed: bool) -> None:
    plain = _backend(not_honoured={})
    with TestClient(_app(settings, plain)) as client:
        r = _post(
            client, "chat", stream=streamed, tool_choice="required", parallel_tool_calls=False
        )
    assert r.status_code == 200
    assert HEADER not in r.headers


def test_the_messages_door_keeps_the_hints_it_dropped_ahead_of_the_backends(
    settings: Settings,
) -> None:
    with TestClient(_app(settings, _backend())) as client:
        r = _post(client, "messages", **FORCED["messages"])
    own, backend = r.headers[HEADER].rsplit(", ", 1)
    assert backend == "tool_choice"
    # `context_management` and an `output_config` effort are on every Claude Code request.
    assert own.split(", ")[:2] == ["context_management", "output_config"]


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
def test_a_browser_can_read_the_header_on_a_stream_and_a_plain_answer(
    settings: Settings, streamed: bool
) -> None:
    """Workbench is a page on another origin: the header must be on the head the
    browser reads and in the list of headers its script may read."""
    body = _chat(tools=[WEATHER], stream=streamed, **FORCED["chat"])
    with TestClient(_app(settings, _backend())) as client:
        r = client.post(
            "/v1/chat/completions", json=body, headers={"origin": "http://192.168.16.75:8079"}
        )
    assert r.headers[HEADER] == "tool_choice"
    assert r.headers["access-control-allow-origin"] == "*"
    assert HEADER in r.headers["access-control-expose-headers"].lower().split(", ")


# --------------------------------------------------------------------------- #
# Codex on Strata
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
def test_codex_is_answered_on_strata_and_the_header_names_parallel_tool_calls(
    settings: Settings, streamed: bool
) -> None:
    """The regression (round one, reviewed): Codex sends `parallel_tool_calls:
    false` and `tool_choice: auto` on every request, and a Strata that did not
    list `parallelToolCalls` refused every one, so Codex could not use it at all.
    Strata ignores `false` (measured; `true` is honoured), so it is sent and
    named, as `required` is."""
    fake = _backend()
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/responses", json=codex_request(model=MODEL, stream=streamed))
    assert r.status_code == 200, r.text
    names = _header(r)
    assert "parallel_tool_calls" in names
    assert "tool_choice" not in names, "`auto` is what Strata does anyway"
    # Sent as Codex sent it, not dropped to make it fit.
    sent = fake.calls[0]
    assert sent.parallelToolCalls is False
    assert sent.toolChoice == ToolChoice.auto
    assert "parallelToolCalls" in (sent.callerSettings or [])


def test_a_backend_that_does_not_carry_the_parallel_switch_is_still_refused_for_it(
    settings: Settings,
) -> None:
    """What was wrong with Strata then was its list, not the gateway's rule: a
    backend whose `supportedSettings` omits a setting that says what it carries
    is refused it, by name, so that a setting is never silently dropped."""
    fake = _backend(not_honoured={})
    fake.supported_settings = [s for s in fake.supported_settings if s != "parallelToolCalls"]
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/responses", json=codex_request(model=MODEL))
    assert r.status_code == 400
    assert "parallel_tool_calls" in r.json()["error"]["message"]
    assert fake.calls == []


# --------------------------------------------------------------------------- #
# Read from the backend chosen for the first attempt
# --------------------------------------------------------------------------- #


def test_the_backend_chosen_first_is_the_one_named(settings: Settings) -> None:
    """Two replicas of one model with different listings, the balancer
    alternating: the header says what the backend this request goes to first does
    not honour, on every request, and not what the other one does."""
    picky = _backend("a-picky", model_id=MODEL)
    keen = _backend("b-keen", model_id=MODEL, not_honoured={})
    with TestClient(_app(settings, picky, keen, affinity=False)) as client:
        seen = []
        for _ in range(4):
            before = (len(picky.calls), len(keen.calls))
            r = _post(client, "chat", **FORCED["chat"])
            assert r.status_code == 200
            went_to_picky = len(picky.calls) > before[0]
            assert went_to_picky != (len(keen.calls) > before[1])
            seen.append(went_to_picky)
            assert _header(r) == (["tool_choice"] if went_to_picky else [])
    assert True in seen and False in seen, "the balancer alternated, so both were checked"


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("door", ["chat", "messages", "responses"])
def test_a_failover_is_not_re_said_the_first_attempts_backend_is_the_one_named(
    settings: Settings, door: str, streamed: bool
) -> None:
    """The contract: read from the backend chosen for the first attempt, before
    anything is sent. The first tier is down; the second serves, and the header,
    which was decided before either was asked, is the first's."""
    dead = _backend("dead", model_id="dead-model")
    dead.generate_error = httpx.ConnectError("refused")
    alive = _backend("alive", model_id="alive-model", not_honoured={})
    slots = [{"model": MODEL, "targets": ["dead-model", "alive-model"]}]
    with TestClient(_app(settings, dead, alive, slots=slots)) as client:
        own = _header(_post(client, door, stream=streamed))
        r = _post(client, door, stream=streamed, **FORCED[door])
    assert r.status_code == 200, r.text
    assert dead.calls and alive.calls, "the first was tried and the second served"
    assert _header(r) == [*own, "tool_choice"]


# --------------------------------------------------------------------------- #
# A stream opens as it always has
# --------------------------------------------------------------------------- #


class LateFirstFrame(FakeDriverClient):
    """A backend whose first frame comes late: the prefill a processor spends on
    a long prompt, which is exactly when a client needs the stream open."""

    delay = 0.5

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        await asyncio.sleep(self.delay)
        async for event in super().stream(request):
            yield event


class Timeline:
    """An ASGI app wrapped to stamp each message it sends, as a client would see
    them arrive. The test client buffers a whole response, so "the head came
    before the first token" cannot be read off it; this can."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.sent: list[tuple[float, AsgiMessage]] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def stamped(message: AsgiMessage) -> None:
            self.sent.append((time.perf_counter(), message))
            await send(message)

        await self.app(scope, receive, stamped)

    def head(self) -> tuple[float, dict[str, str]]:
        """When the response head went out, and its headers."""
        at, message = next((t, m) for t, m in self.sent if m["type"] == "http.response.start")
        return at, {k.decode(): v.decode() for k, v in message["headers"]}

    def first(self, text: bytes) -> float:
        """When the first body message holding `text` went out."""
        return next(
            t for t, m in self.sent if m["type"] == "http.response.body" and text in m["body"]
        )


@pytest.mark.parametrize("door", ["chat", "messages", "responses"])
def test_a_stream_opens_before_its_backends_first_frame_and_the_head_carries_the_header(
    settings: Settings, door: str
) -> None:
    """The head goes out when a backend has been chosen, not when it first speaks:
    the first token is half a second behind it, with what the backend does not
    honour already on it. On the Responses door the stream's own opening frames
    (`response.created`) go out ahead of the token too, which is what a client
    that drops a silent stream needs."""
    fake = LateFirstFrame(
        name="strata", model_id=MODEL, supports_tools=True, not_honoured=STRATA_GAPS
    )
    fake.responses = ["pelican"]
    timeline = Timeline(_app(settings, fake))
    with TestClient(timeline) as client:
        r = _post(client, door, stream=True, **FORCED[door])
    assert r.status_code == 200 and "pelican" in r.text
    opened, headers = timeline.head()
    token = timeline.first(b"pelican")
    assert token - opened >= 0.4, f"the head came {token - opened:.2f}s before the first token"
    assert headers[HEADER].split(", ")[-1] == "tool_choice"
    if door == "responses":
        assert token - timeline.first(b"response.created") >= 0.4
        assert token - timeline.first(b"response.in_progress") >= 0.4


def test_the_responses_stream_says_it_is_working_while_the_backend_prefills(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`response.in_progress` every keepalive interval until output arrives,
    with the header on the head all the while."""
    monkeypatch.setattr(responses, "KEEPALIVE_SECONDS", 0.05)
    fake = LateFirstFrame(
        name="strata", model_id=MODEL, supports_tools=True, not_honoured=STRATA_GAPS
    )
    fake.responses = ["ok"]
    with TestClient(_app(settings, fake)) as client:
        r = _post(client, "responses", stream=True, **FORCED["responses"])
    assert _header(r)[-1] == "tool_choice"
    names = [line[7:] for line in r.text.splitlines() if line.startswith("event: ")]
    assert names[:2] == ["response.created", "response.in_progress"]
    assert names.count("response.in_progress") >= 3 and names[-1] == "response.completed"


class Silent(FakeDriverClient):
    """A backend whose stream opens and ends without a frame: no output, no `done`."""

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        self.calls.append(request)
        return
        yield StreamEvent()  # pragma: no cover


def _failed_stream(response: httpx.Response) -> tuple[list[str], dict[str, Any]]:
    names = [line[7:] for line in response.text.splitlines() if line.startswith("event: ")]
    last = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")][-1]
    return names, json.loads(last)["response"]["error"]


def test_a_responses_stream_that_fails_before_any_output_is_response_failed(
    settings: Settings,
) -> None:
    """Codex has not seen a stream start until `response.created`, and a bare
    status or `response.failed` without it is one it reports without our message
    (record §4). A backend that dies before its first frame still owes the
    opening, and its failure is a `response.failed` coded for what Codex does
    next: `server_error`, which it retries, for a backend that died."""
    fake = _backend()
    fake.stream_error_after = 0
    fake.stream_error = httpx.ConnectError("gone")
    with TestClient(_app(settings, fake)) as client:
        r = _post(client, "responses", stream=True, **FORCED["responses"])
    assert r.status_code == 200
    assert _header(r)[-1] == "tool_choice"
    names, error = _failed_stream(r)
    assert names == ["response.created", "response.in_progress", "response.failed"]
    assert error["code"] == "server_error"


def test_a_responses_stream_that_ends_without_a_frame_is_response_failed(
    settings: Settings,
) -> None:
    silent = Silent(name="strata", model_id=MODEL, supports_tools=True)
    with TestClient(_app(settings, silent)) as client:
        r = _post(client, "responses", stream=True)
    names, error = _failed_stream(r)
    assert names == ["response.created", "response.in_progress", "response.failed"]
    assert error["code"] == "server_error"


# --------------------------------------------------------------------------- #
# The listing travels the wire; the driver's per-request header is not read
# --------------------------------------------------------------------------- #

WIRE_TOKEN = "pelican"
CHUNK_SEP = chr(10)
FRAME_SEP = chr(10) * 2


class _Install(BaseHTTPRequestHandler):
    """An agent and an inference driver in one process, speaking their contracts
    over a real socket. The driver's listing carries `settingsNotHonoured` as the
    real driver writes it, and each generate answer also carries the driver's
    per-request header, naming a field the listing does not: the gateway names
    from the listing, so that header must not reach a caller."""

    generate_calls: ClassVar[list[dict[str, Any]]] = []
    per_request = "topK"

    def log_message(self, *_args: object) -> None:
        pass

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        port = self.server.server_port
        if path == "/v1/components":
            driver = {
                "name": "strata-box",
                "kind": "inference-driver",
                "url": f"http://127.0.0.1:{port}",
                "status": "running",
            }
            self._send(200, {"components": [driver]})
        elif path == "/v1/info":
            self._send(
                200,
                {
                    "backend": "openai_compat_http",
                    "provider": "local",
                    "models": [
                        {
                            "id": MODEL,
                            "surfaces": ["chat"],
                            "capabilities": {
                                "streaming": True,
                                "toolCalling": True,
                                "supportedSettings": [
                                    "maxTokens",
                                    "temperature",
                                    "tools",
                                    "toolChoice",
                                    "parallelToolCalls",
                                    "topK",
                                ],
                                "settingsNotHonoured": [
                                    {"setting": "toolChoice", "values": ["required", "function"]},
                                    {"setting": "parallelToolCalls", "values": [False]},
                                ],
                            },
                        }
                    ],
                    "version": "0.1.0-fake",
                },
            )
        else:
            self._send(404, {"error": "nope"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        type(self).generate_calls.append(payload)
        final = {
            "content": WIRE_TOKEN,
            "finishReason": "stop",
            "backend": "openai_compat_http",
            "modelId": MODEL,
            "latencyMs": 3,
        }
        if self.path == "/v1/generate":
            self._send(200, final)
        elif self.path == "/v1/generate/stream":
            frames = [
                "event: token" + CHUNK_SEP + "data: " + json.dumps({"text": WIRE_TOKEN}),
                "event: done" + CHUNK_SEP + "data: " + json.dumps(final),
            ]
            raw = (FRAME_SEP.join(frames) + FRAME_SEP).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header(HEADER, self.per_request)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        else:
            self._send(404, {"error": "nope"})

    def _send(self, code: int, body: dict[str, Any]) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        if self.path == "/v1/generate":
            self.send_header(HEADER, self.per_request)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def stand_in() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    calls: list[dict[str, Any]] = []

    class Handler(_Install):
        generate_calls = calls

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def wired(tmp_path: Path, stand_in: tuple[str, list[dict[str, Any]]]) -> Iterator[TestClient]:
    """The real app with nothing injected: its lifespan builds the genuine routing
    table, which reads the stand-in's `/v1/info` off the socket."""
    settings = Settings(
        config_file=tmp_path / "config.yaml",
        metrics_file=tmp_path / "metrics.sqlite3",
        agent_url=stand_in[0],
    )
    with TestClient(create_app(settings=settings)) as client:
        yield client


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("door", ["chat", "responses"])
def test_the_listing_on_the_wire_is_the_header_on_the_door_and_the_drivers_own_is_not(
    wired: TestClient, stand_in: tuple[str, list[dict[str, Any]]], door: str, streamed: bool
) -> None:
    own = _header(_post(wired, door, stream=streamed))
    forced = _post(wired, door, stream=streamed, tool_choice="required", parallel_tool_calls=False)
    assert forced.status_code == 200, forced.text
    assert WIRE_TOKEN in forced.text
    assert _header(forced) == [*own, "tool_choice", "parallel_tool_calls"]
    # The driver said `topK` on every answer it gave, and the caller never sees it.
    assert "top_k" not in forced.headers.get(HEADER, "")
    sent = stand_in[1][-1]
    assert sent["toolChoice"] == "required" and sent["parallelToolCalls"] is False


def test_codex_through_the_wire_is_answered_and_names_the_parallel_switch(
    wired: TestClient,
) -> None:
    r = wired.post("/v1/responses", json=codex_request(model=MODEL))
    assert r.status_code == 200, r.text
    assert "parallel_tool_calls" in _header(r) and "tool_choice" not in _header(r)


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


@pytest.mark.parametrize("knobs", [{"supports_embeddings": True}, {"surfaces": ["rerank"]}])
def test_a_model_that_does_not_chat_is_not_one_that_calls_tools_because_nothing_said_so(
    settings: Settings, knobs: dict[str, Any]
) -> None:
    fake = FakeDriverClient(name="sorter", model_id="nomic", capabilities_reported=False, **knobs)
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


# --------------------------------------------------------------------------- #
# What a listing owes a request that carries tools
# --------------------------------------------------------------------------- #


def _short(fake: FakeDriverClient, *listed: str) -> FakeDriverClient:
    fake.supported_settings = list(listed)
    return fake


async def _left(*fakes: FakeDriverClient, **fields: Any) -> list[str]:
    """The names of the backends `admission.permitted` leaves for this request."""
    table = make_routing_table(*fakes, affinity=False)
    resolution = await admission.permitted(table.resolve(MODEL), requirements=_req(**fields))
    return [b.name for b in resolution.backends()]


def test_a_short_list_of_settings_does_not_decide_tools_for_a_backend_that_does_not_say(
    settings: Settings,
) -> None:
    """Admission re-reads the model's listing for a request that carries
    settings, and `tools` is one: a driver that says nothing of tool calling takes
    them whatever its list of settings holds. Only `false` decides."""
    fake = _short(FakeDriverClient(name="custom", model_id=MODEL, supports_tools=None), "maxTokens")
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/chat/completions", json=_chat(tools=[WEATHER]))
    assert r.status_code == 200, r.text
    assert fake.calls[0].tools is not None


@pytest.mark.parametrize("knobs", UNSAID)
def test_the_settings_that_ride_with_tools_are_admitted_for_a_backend_that_does_not_say(
    settings: Settings, knobs: dict[str, Any]
) -> None:
    """`tool_choice` and `parallel_tool_calls` come with tools, and a driver that
    reports nothing is not a no: a backend with no capabilities at all, or no
    `toolCalling`, is not refused them for a list it may not have written."""
    fake = _short(FakeDriverClient(name="custom", model_id=MODEL, **knobs), "maxTokens")
    with TestClient(_app(settings, fake)) as client:
        r = client.post(
            "/v1/chat/completions",
            json=_chat(tools=[WEATHER], tool_choice="required", parallel_tool_calls=False),
        )
    assert r.status_code == 200, r.text
    sent = fake.calls[0]
    assert sent.toolChoice == ToolChoice.required and sent.parallelToolCalls is False
    assert HEADER not in r.headers, "nothing is listed, so nothing is named"


@pytest.mark.parametrize("knobs", UNSAID)
def test_every_other_setting_keeps_the_rule_for_a_backend_that_does_not_say(
    settings: Settings, knobs: dict[str, Any]
) -> None:
    """Unknown is yes for tools and the settings that ride with them, and nowhere
    else: a setting that is not one of those is refused by name unless the
    listing names it, as it always was."""
    fake = _short(FakeDriverClient(name="custom", model_id=MODEL, **knobs), "maxTokens")
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/chat/completions", json=_chat(tools=[WEATHER], top_k=5))
    assert r.status_code == 400
    assert "top_k" in r.json()["error"]["message"]
    assert r.json()["error"]["message"].startswith(
        "No permitted backend serving this model takes top_k."
    )
    assert fake.calls == []


@pytest.mark.parametrize("setting", ["tool_choice", "parallel_tool_calls"])
def test_a_backend_that_says_keeps_the_rule_that_its_list_names_what_it_takes(
    settings: Settings, setting: str
) -> None:
    """`toolCalling: true` says the driver is reporting, so its list is the list:
    it is not asked to carry a setting it does not name."""
    fake = _short(FakeDriverClient(name="said", model_id=MODEL, supports_tools=True), "maxTokens")
    extra = (
        {"tool_choice": "required"} if setting == "tool_choice" else {"parallel_tool_calls": False}
    )
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/chat/completions", json=_chat(tools=[WEATHER], **extra))
    assert r.status_code == 400
    assert setting in r.json()["error"]["message"]
    assert fake.calls == []


async def test_permitted_leaves_the_backends_that_carry_the_whole_request() -> None:
    unsaid = _short(FakeDriverClient(name="a-unsaid", model_id=MODEL, supports_tools=None), "stop")
    silent = FakeDriverClient(name="b-silent", model_id=MODEL, capabilities_reported=False)
    said = _short(FakeDriverClient(name="c-said", model_id=MODEL, supports_tools=True), "stop")
    full = FakeDriverClient(name="d-full", model_id=MODEL, supports_tools=True)
    refused = FakeDriverClient(name="e-no", model_id=MODEL, supports_tools=False)
    left = await _left(unsaid, silent, said, full, refused, **WITH_TOOLS)
    assert left == ["a-unsaid", "b-silent", "d-full"]


# --------------------------------------------------------------------------- #
# The refusal, and what it counts
# --------------------------------------------------------------------------- #


def _tools_post(client: TestClient, **extra: Any) -> httpx.Response:
    return client.post("/v1/chat/completions", json=_chat(tools=[WEATHER], **extra))


def test_a_backend_whose_driver_says_no_is_refused_in_the_drivers_words(
    settings: Settings,
) -> None:
    fake = FakeDriverClient(name="cli", model_id=MODEL, supports_tools=False)
    with TestClient(_app(settings, fake)) as client:
        r = _tools_post(client)
    error = r.json()["error"]
    assert r.status_code == 400 and error["param"] == "tools"
    assert error["message"] == tools_refused(MODEL, 1)
    assert "its driver reports that tools cannot be carried to it" in error["message"]
    assert fake.calls == []


def test_when_every_backend_says_no_the_refusal_says_every_driver_does(
    settings: Settings,
) -> None:
    fakes = [FakeDriverClient(name=f"cli-{i}", model_id=MODEL, supports_tools=False) for i in "ab"]
    with TestClient(_app(settings, *fakes)) as client:
        r = _tools_post(client)
    assert r.status_code == 400
    assert r.json()["error"]["message"] == tools_refused(MODEL, 2)
    assert "every driver serving it reports that tools cannot be carried to it" in r.text


def test_a_refusal_for_another_setting_is_not_worded_as_the_tools_one(
    settings: Settings,
) -> None:
    """One backend says no to tools; the other carries them and not `top_k`. The
    request cannot be served, and what to change is the setting the backend that
    could serve it lacks: the first does not count toward what "some backend
    takes", since it could not serve the request anyway."""
    no_tools = FakeDriverClient(name="a-cli", model_id=MODEL, supports_tools=False)
    no_top_k = FakeDriverClient(name="b-gpu", model_id=MODEL, supports_tools=True)
    no_top_k.supported_settings = [s for s in no_top_k.supported_settings if s != "topK"]
    with TestClient(_app(settings, no_tools, no_top_k)) as client:
        r = _tools_post(client, top_k=5)
    error = r.json()["error"]
    assert r.status_code == 400
    assert error["param"] == "top_k"
    assert "takes top_k" in error["message"]
    assert "driver" not in error["message"] and "cannot be carried" not in error["message"]
    assert no_tools.calls == [] and no_top_k.calls == []


def test_a_mixed_refusal_with_nothing_to_name_says_what_could_not_be_confirmed_together(
    settings: Settings,
) -> None:
    """When every setting is taken somewhere and still no one backend takes
    them all, the refusal is the old one, which has no one thing to name."""
    no_tools = FakeDriverClient(name="a-cli", model_id=MODEL, supports_tools=False)
    no_top_k = FakeDriverClient(name="b-gpu", model_id=MODEL, supports_tools=True)
    no_top_k.supported_settings = [s for s in no_top_k.supported_settings if s != "topK"]
    other = FakeDriverClient(name="c-gpu", model_id=MODEL, supports_tools=True)
    other.supported_settings = [s for s in other.supported_settings if s != "minP"]
    with TestClient(_app(settings, no_tools, no_top_k, other)) as client:
        r = _tools_post(client, top_k=5, min_p=0.1)
    error = r.json()["error"]
    assert r.status_code == 400
    assert "together" in error["message"] and "cannot be carried" not in error["message"]


def test_a_backend_that_does_not_chat_is_not_counted_among_those_a_tools_refusal_is_about(
    settings: Settings,
) -> None:
    """A slot of a model that cannot take tools and one that is not a chat model:
    the refusal is the tools one, because it is the only chat backend that was
    turned away, and the embedding model was never asked about tools."""
    cli = FakeDriverClient(name="cli", model_id="chatty", supports_tools=False)
    embed = FakeDriverClient(name="embed", model_id="nomic", supports_embeddings=True)
    slots = [{"model": MODEL, "targets": ["chatty", "nomic"]}]
    with TestClient(_app(settings, cli, embed, slots=slots)) as client:
        r = _tools_post(client)
    assert r.status_code == 400
    assert r.json()["error"]["message"] == tools_refused(MODEL, 1)
    assert cli.calls == [] and embed.embed_calls == 0


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
        assert tools_refused(MODEL, 1) in r.json()["error"]["message"]


@pytest.mark.parametrize("replicas", [1, 2])
@pytest.mark.parametrize(
    "path",
    ["/v1/chat/completions", "/v1/messages", "/v1/responses", "/v1/messages/count_tokens"],
)
def test_the_front_doors_own_refusal_is_the_same_sentence_for_the_same_reason(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, path: str, replicas: int
) -> None:
    """Admission turns a backend away for its tools first, and the front door
    checks again on what admission let through: a second line, and the one a
    refresh that raced a driver's change would meet. Reached here by letting
    every backend through admission. It says what admission says, in the same
    words, naming one driver or all of them by how many serve the model."""

    async def through(resolution: Any, requirements: Any = None) -> Any:
        return resolution

    monkeypatch.setattr(admission, "permitted", through)
    fakes = [
        FakeDriverClient(name=f"cli-{i}", model_id=MODEL, supports_tools=False)
        for i in range(replicas)
    ]
    bodies = {
        "/v1/chat/completions": _chat(tools=[WEATHER]),
        "/v1/messages": _messages(),
        "/v1/responses": _responses(),
        "/v1/messages/count_tokens": _messages(),
    }
    with TestClient(_app(settings, *fakes)) as client:
        r = client.post(path, json=bodies[path])
    assert r.status_code == 400, r.text
    assert tools_refused(MODEL, replicas) in r.json()["error"]["message"]
    assert all(fake.calls == [] for fake in fakes)


# --------------------------------------------------------------------------- #
# A model that does not chat is sent to its own door
# --------------------------------------------------------------------------- #

NOT_CHAT = [
    pytest.param({"supports_embeddings": True}, "embeddings", "/v1/embeddings", id="embeddings"),
    pytest.param({"surfaces": ["rerank"]}, "rerank", "/v1/rerank", id="rerank"),
    pytest.param({"surfaces": ["speech"]}, "speech", "/v1/audio/speech", id="speech"),
]
REPORTS = [
    pytest.param({"capabilities_reported": False}, id="nothing reported"),
    pytest.param({"supports_tools": None}, id="toolCalling not reported"),
    pytest.param({"supports_tools": False}, id="toolCalling false"),
    pytest.param({"supports_tools": True}, id="toolCalling true"),
]


@pytest.mark.parametrize("reports", REPORTS)
@pytest.mark.parametrize(("knobs", "surface", "door"), NOT_CHAT)
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages", "/v1/responses"])
def test_tools_sent_to_a_model_that_does_not_chat_are_refused_for_what_it_is(
    settings: Settings,
    path: str,
    knobs: dict[str, Any],
    surface: str,
    door: str,
    reports: dict[str, Any],
) -> None:
    """Whatever the driver says of tools, and whatever else rides with them: the
    model is on the wrong door, and says so in the door's words, not as a model
    that cannot call tools."""
    fake = FakeDriverClient(name="other", model_id="sorter", **knobs, **reports)
    wrong = {
        "/v1/chat/completions": _chat(model="sorter", tools=[WEATHER], tool_choice="auto"),
        "/v1/messages": {**_messages(), "model": "sorter"},
        "/v1/responses": {**_responses(tool_choice="auto"), "model": "sorter"},
    }[path]
    with TestClient(_app(settings, fake)) as client:
        r = client.post(path, json=wrong)
    assert r.status_code == 400, r.text
    message = r.json()["error"]["message"]
    assert f"serves {surface} and not chat" in message
    assert door in message
    assert "cannot be carried" not in message and "tool_calling" not in message
    assert fake.calls == []


@pytest.mark.parametrize(("knobs", "surface", "door"), NOT_CHAT)
def test_a_token_count_with_tools_for_a_model_that_does_not_chat_is_the_wrong_door_too(
    settings: Settings, knobs: dict[str, Any], surface: str, door: str
) -> None:
    fake = FakeDriverClient(name="other", model_id="sorter", capabilities_reported=False, **knobs)
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/messages/count_tokens", json={**_messages(), "model": "sorter"})
    assert r.status_code == 400
    assert f"serves {surface} and not chat" in r.json()["error"]["message"]


def test_a_slot_of_a_chat_model_and_an_embedding_model_serves_tools_from_the_chat_one(
    settings: Settings,
) -> None:
    chatty = FakeDriverClient(name="chatty", model_id="chatty", supports_tools=None)
    chatty.responses = ["from the chat model"]
    embed = FakeDriverClient(name="embed", model_id="nomic", supports_embeddings=True)
    slots = [{"model": MODEL, "targets": ["chatty", "nomic"]}]
    with TestClient(_app(settings, chatty, embed, slots=slots)) as client:
        r = _tools_post(client, tool_choice="auto")
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "from the chat model"
    assert chatty.calls[0].tools is not None and embed.calls == []


async def test_permitted_keeps_a_model_that_does_not_chat_for_the_door_to_refuse_by_name() -> None:
    """When nothing in the slot chats the request is not refused here: the door
    names the surface, which is the answer. When something chats, the others
    are left out."""
    embed = FakeDriverClient(name="embed", model_id=MODEL, supports_embeddings=True)
    assert await _left(embed, **WITH_TOOLS) == ["embed"]
    chatty = FakeDriverClient(name="chatty", model_id=MODEL, supports_tools=True)
    assert await _left(embed, chatty, **WITH_TOOLS) == ["chatty"]


# --------------------------------------------------------------------------- #
# A search is offered to a backend that does not say it calls tools
# --------------------------------------------------------------------------- #


def _reaches(*fakes: FakeDriverClient, native_ok: bool = True) -> tuple[list[str], set[Any]]:
    table = make_routing_table(*fakes, affinity=False)
    resolution, native = server_tools.searchable(table.resolve(MODEL), native_ok=native_ok)
    return [b.name for b in resolution.backends()], native


@pytest.mark.parametrize("knobs", UNSAID)
def test_searchable_keeps_a_backend_that_does_not_say_it_calls_tools(knobs: dict[str, Any]) -> None:
    fake = FakeDriverClient(name="custom", model_id=MODEL, **knobs)
    assert _reaches(fake) == (["custom"], set())
    # Nor does the image tool, which is answered only by a backend that calls tools.
    assert _reaches(fake, native_ok=False) == (["custom"], set())


def test_searchable_leaves_out_a_backend_that_says_it_does_not_call_tools() -> None:
    assert _reaches(FakeDriverClient(name="cli", model_id=MODEL, supports_tools=False)) == (
        [],
        set(),
    )


def test_searchable_leaves_out_a_model_that_does_not_chat_when_nothing_said_it_does() -> None:
    embed = FakeDriverClient(name="embed", model_id=MODEL, supports_embeddings=True)
    assert _reaches(embed) == ([], set())
    silent = FakeDriverClient(
        name="silent", model_id=MODEL, supports_embeddings=True, capabilities_reported=False
    )
    assert _reaches(silent) == ([], set())


def test_searchable_sends_the_search_natively_to_a_backend_that_searches_itself() -> None:
    fake = FakeDriverClient(name="native", model_id=MODEL, supports_tools=False)
    fake.supported_settings = [*fake.supported_settings, "webSearchOptions"]
    names, native = _reaches(fake)
    assert names == ["native"] and native == {(None, "native", MODEL)}
    # Not when the loop must run whatever the backend can do alone.
    assert _reaches(fake, native_ok=False) == ([], set())


@pytest.mark.parametrize("knobs", UNSAID)
def test_a_search_is_offered_to_a_backend_that_does_not_say_it_calls_tools(
    settings: Settings, knobs: dict[str, Any]
) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    for knob, value in knobs.items():
        setattr(model, knob, value)  # the scripted model says it calls tools until told not to
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


def test_the_search_loops_own_forced_first_turn_is_not_named(settings: Settings) -> None:
    """The loop asks the model's first turn for a tool (`required`) when search is
    the caller's only tool, on a request it made itself. The caller sent no
    `tool_choice`, so a backend that ignores `required` is not named for it: the
    header says what the caller was not given, not what the gateway chose to ask."""
    model = ScriptedDriver(
        turns=[Turn(calls=[search_call()]), Turn(ANSWER)], not_honoured=STRATA_GAPS
    )
    with TestClient(app_with(settings, model, searches=[account(FakeSearch())])) as client:
        r = chat(client)
    assert r.status_code == 200, r.text
    assert str(model.calls[0].toolChoice) == "required" and model.calls[1].toolChoice is None
    assert HEADER not in r.headers


def test_the_search_loops_forcing_is_not_named_even_when_the_caller_said_auto(
    settings: Settings,
) -> None:
    model = ScriptedDriver(
        turns=[Turn(calls=[search_call()]), Turn(ANSWER)], not_honoured=STRATA_GAPS
    )
    with TestClient(app_with(settings, model, searches=[account(FakeSearch())])) as client:
        r = chat(client, tool_choice="auto")
    assert r.status_code == 200, r.text
    assert str(model.calls[0].toolChoice) == "required"
    assert HEADER not in r.headers


def test_a_callers_own_forced_choice_on_a_search_is_named(settings: Settings) -> None:
    """The positive control: what the caller sent is named, whatever else the loop
    does with the request."""
    model = ScriptedDriver(
        turns=[Turn(calls=[search_call()]), Turn(ANSWER)], not_honoured=STRATA_GAPS
    )
    with TestClient(app_with(settings, model, searches=[account(FakeSearch())])) as client:
        r = chat(client, tool_choice="required")
    assert r.status_code == 200, r.text
    assert _header(r) == ["tool_choice"]
