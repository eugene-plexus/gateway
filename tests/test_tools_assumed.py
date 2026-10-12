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
import functools
import json
import re
import sys
import threading
import time
from collections.abc import AsyncGenerator, Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace
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
    DriverInfo,
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
from eugene_plexus_gateway.driver_client import DriverError, StreamEvent
from eugene_plexus_gateway.routes import inference
from eugene_plexus_gateway.settings import Settings
from tests.test_admission import Authority

from .conftest import FakeDriverClient, FakeInstall, make_routing_table, runtime_facts
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


def test_none_sends_no_tools_so_no_parallel_switch_is_named_for_an_unreading_backend() -> None:
    """`tool_choice: none` to Strata: its driver sends no tools and no
    `parallel_tool_calls` with them, so nothing is ignored, though `false` is in
    its listing. Named for any other tool choice, and when sent alone."""
    both = {"parallelToolCalls": False, "callerSettings": ["toolChoice", "parallelToolCalls"]}
    assert ignored_settings.not_honoured(STRATA, _req(toolChoice=ToolChoice.none, **both)) == ()
    assert ignored_settings.not_honoured(STRATA, _req(toolChoice=ToolChoice.auto, **both)) == (
        "parallelToolCalls",
    )
    assert ignored_settings.not_honoured(STRATA, _req(toolChoice=ToolChoice.required, **both)) == (
        "toolChoice",
        "parallelToolCalls",
    )
    # The parallel switch alone is still named: the choice is not `none`.
    alone = _req(parallelToolCalls=False, callerSettings=["parallelToolCalls"])
    assert ignored_settings.not_honoured(STRATA, alone) == ("parallelToolCalls",)


def test_none_names_the_parallel_switch_for_a_backend_that_does_read_tool_choice() -> None:
    """The rule is the driver's: a backend that lists no `toolChoice` is sent `none`
    as `none` (and the switch with it), so its listing of the switch stands."""
    reads = _listing(parallelToolCalls=[False])
    request = _req(
        toolChoice=ToolChoice.none,
        parallelToolCalls=False,
        callerSettings=["toolChoice", "parallelToolCalls"],
    )
    assert ignored_settings.not_honoured(reads, request) == ("parallelToolCalls",)


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
        # `none` is honoured by sending no tools, and no parallel switch with them:
        # nothing is left on the request for the backend to ignore.
        ("none with parallel false", {"tool_choice": "none", "parallel_tool_calls": False}, []),
        (
            "auto with parallel false",
            {"tool_choice": "auto", "parallel_tool_calls": False},
            ["parallel_tool_calls"],
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
        (
            "none with parallel disabled",
            {"tool_choice": {"type": "none", "disable_parallel_tool_use": True}},
            [],
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
        ("none with parallel false", {"tool_choice": "none", "parallel_tool_calls": False}, []),
        (
            "auto with parallel false",
            {"tool_choice": "auto", "parallel_tool_calls": False},
            ["parallel_tool_calls"],
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
# Read from the backend the first attempt goes to
# --------------------------------------------------------------------------- #

#: Two runtimes sharing nothing, each with a pool of its own to budget (CB3).
POOLED = [runtime_facts("rt-a", context_pool=10_000), runtime_facts("rt-b", context_pool=10_000)]

#: Which backend is the balancer's first pick, and which the one asked: round robin
#: from a fresh table starts at the first name, and `a-` sorts before `b-`.
PICKED_ASKED = [
    pytest.param(STRATA_GAPS, {}, False, id="the balancers pick lists gaps"),
    pytest.param({}, STRATA_GAPS, True, id="the backend asked lists gaps"),
]


def _cool(table: Any, name: str) -> None:
    """Open one driver's circuit, as a failure a moment ago leaves it."""
    for backend in table.resolve(MODEL).backends():
        if backend.name == name:
            backend.client.circuit.failures = 1
            backend.client.circuit.until = time.perf_counter() + 60


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("door", ["chat", "messages", "responses"])
@pytest.mark.parametrize(("picked_lists", "asked_lists", "named"), PICKED_ASKED)
def test_the_header_names_the_backend_asked_when_the_balancers_pick_has_an_open_circuit(
    settings: Settings,
    door: str,
    streamed: bool,
    picked_lists: dict[str, list[str | bool]],
    asked_lists: dict[str, list[str | bool]],
    named: bool,
) -> None:
    """`Executor.attempts` skips a candidate whose circuit is open, so the balancer's
    first pick is not where the request goes first. The head names what the backend
    the request actually goes to first does not honour, whichever of the two lists
    a gap: reading the pick would have said the cooling one's."""
    picked = _backend("a-picked", model_id=MODEL, not_honoured=picked_lists)
    asked = _backend("b-asked", model_id=MODEL, not_honoured=asked_lists)
    app = _app(settings, picked, asked, strategy="round_robin", affinity=False)
    _cool(app.state.routing, "a-picked")
    with TestClient(app) as client:
        r = _post(client, door, stream=streamed, **FORCED[door])
    assert r.status_code == 200, r.text
    assert asked.calls and not picked.calls, "the cooling backend was not asked"
    assert ("tool_choice" in _header(r)) is named


def test_a_backend_whose_cooldown_is_over_is_asked_as_the_probe_and_is_the_one_named(
    settings: Settings,
) -> None:
    """The read ahead takes no probe. A circuit past its cooldown hands out one
    recovery probe, to whichever attempt asks first: if the head's read took it,
    the attempt would find it taken, skip this backend and go to the other, and the
    header would name a backend that was never asked."""
    probing = _backend("a-probe", model_id=MODEL)
    other = _backend("b-other", model_id=MODEL, not_honoured={})
    app = _app(settings, probing, other, strategy="round_robin", affinity=False)
    for backend in app.state.routing.resolve(MODEL).backends():
        if backend.name == "a-probe":
            backend.client.circuit.failures = 1  # it failed, and the cooldown has passed
            backend.client.circuit.until = 0.0
    with TestClient(app) as client:
        r = _post(client, "chat", **FORCED["chat"])
    assert r.status_code == 200, r.text
    assert probing.calls and not other.calls, "the probe went to the backend that was read"
    assert _header(r) == ["tool_choice"]


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("door", ["chat", "messages", "responses"])
@pytest.mark.parametrize(("full_lists", "roomy_lists", "named"), PICKED_ASKED)
def test_the_header_names_the_replica_with_room_when_the_pool_moves_it_to_the_front(
    settings: Settings,
    door: str,
    streamed: bool,
    full_lists: dict[str, list[str | bool]],
    roomy_lists: dict[str, list[str | bool]],
    named: bool,
) -> None:
    """`TieredClient._admit` tries first where there is room in a shared pool: a
    replica whose pool is full of other turns' prompts is not asked first, though
    the balancer picked it. The head names the replica that has room."""
    full = _backend("a-full", model_id=MODEL, runtime="rt-a", not_honoured=full_lists)
    roomy = _backend("b-roomy", model_id=MODEL, runtime="rt-b", not_honoured=roomy_lists)
    app = _app(settings, full, roomy, runtimes=POOLED, strategy="round_robin", affinity=False)
    app.state.routing._ledger.take((None, "rt-a"), 9_500)
    with TestClient(app) as client:
        r = _post(client, door, stream=streamed, **FORCED[door])
    assert r.status_code == 200, r.text
    assert roomy.calls and not full.calls, "the replica with room was moved to the front"
    assert ("tool_choice" in _header(r)) is named


def test_reading_the_backend_asked_first_reserves_nothing_and_takes_no_probe() -> None:
    """A read: asked any number of times, before anything is sent, it leaves the
    pools' ledger as it was and the circuit's probe where it was."""
    a = _backend("a", model_id=MODEL, runtime="rt-a")
    b = _backend("b", model_id=MODEL, runtime="rt-b", not_honoured={})
    table = make_routing_table(a, b, runtimes=POOLED, strategy="round_robin", affinity=False)
    for backend in table.resolve(MODEL).backends():
        if backend.name == "a":
            backend.client.circuit.failures = 1
            backend.client.circuit.until = 0.0
    client = table.pick(table.resolve(MODEL))
    assert client is not None and client.budget is not None
    for _ in range(3):
        entry = client.model_asked_first(_req(maxTokens=100))
        assert entry is not None and entry.capabilities is not None
        assert entry.capabilities.settingsNotHonoured, "a is first, and is the probe"
    assert table._ledger.in_flight((None, "rt-a")) == 0
    assert table._ledger.in_flight((None, "rt-b")) == 0
    assert client.budget._lease is None
    circuits = [backend.client.circuit for backend in table.resolve(MODEL).backends()]
    assert not any(circuit.probing for circuit in circuits)


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
def test_a_failover_is_not_re_said_on_a_stream_and_a_whole_answer_names_who_served_it(
    settings: Settings, door: str, streamed: bool
) -> None:
    """Strata's listing first (it ignores `required`), and it fails; the backend that
    honours the choice serves.

    **A stream's head cannot be changed by a failover.** It goes out before any
    backend has been asked, so it names the first attempt's backend, whoever then
    serves (`TieredClient.model_asked_first`). **A whole answer has no head to send
    early**: its header is built after the call, from the backend that served it, so
    here it names nothing, because the backend that answered honoured the choice.

    The first tier is healthy for the request that sets the door's own names (a
    failure of its would open its circuit, and the next request would go past it
    to the second: that is `test_the_header_names_the_backend_asked...`)."""
    dead = _backend("dead", model_id="dead-model")
    alive = _backend("alive", model_id="alive-model", not_honoured={})
    slots = [{"model": MODEL, "targets": ["dead-model", "alive-model"]}]
    with TestClient(_app(settings, dead, alive, slots=slots)) as client:
        own = _header(_post(client, door, stream=streamed))
        tried = len(dead.calls)
        dead.generate_error = httpx.ConnectError("refused")
        r = _post(client, door, stream=streamed, **FORCED[door])
    assert r.status_code == 200, r.text
    assert len(dead.calls) == tried + 1 and len(alive.calls) == 1, "tried, failed, then served"
    assert _header(r) == ([*own, "tool_choice"] if streamed else own)


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("door", ["chat", "messages", "responses"])
def test_a_whole_answer_served_by_strata_after_a_failover_is_named_and_a_stream_cannot_be(
    settings: Settings, door: str, streamed: bool
) -> None:
    """The other way round: the backend that honours `required` is asked first and
    fails, and Strata, which ignores it, serves. The whole answer says so, because
    it is built from who served it (the round-three header, read before the call,
    named the failed backend's listing and said nothing: slot [first (no gaps,
    ConnectError), strata (gaps)], `tool_choice: required`, served by Strata,
    header empty). A stream's head was on the wire before either was asked and is
    the first backend's: empty, which is the limit, said once in
    `TieredClient.model_asked_first`."""
    first = _backend("first", model_id="first-model", not_honoured={})
    strata = _backend("strata", model_id="strata-model")
    slots = [{"model": MODEL, "targets": ["first-model", "strata-model"]}]
    with TestClient(_app(settings, first, strata, slots=slots)) as client:
        own = _header(_post(client, door, stream=streamed))
        tried = len(first.calls)
        first.generate_error = httpx.ConnectError("refused")
        r = _post(client, door, stream=streamed, **FORCED[door])
    assert r.status_code == 200, r.text
    assert len(first.calls) == tried + 1 and len(strata.calls) == 1, "tried, failed, then served"
    assert _header(r) == (own if streamed else [*own, "tool_choice"])


def _serving(client: Any) -> inference._Serving:
    """A `_Serving` with only what the header's names are read from."""
    return inference._Serving(
        client=client,
        table=None,  # type: ignore[arg-type]
        generate=_req(toolChoice=ToolChoice.required, callerSettings=["toolChoice"]),
        rec=None,  # type: ignore[arg-type]
        started=0.0,
        waited_ms=0,
        swapped_in=False,
        not_honoured=("parallelToolCalls",),
    )


def test_a_whole_answers_names_are_read_from_the_backend_that_answered() -> None:
    """Whoever answered, by its own listing: its gaps, its honouring, or no listing
    at all. The first attempt's names are only the answer when nobody is known to
    have answered."""
    strata = _backend("strata")
    honours = _backend("honours", not_honoured={})
    entries = {
        strata: strata.describe().models[0],  # type: ignore[index]
        honours: honours.describe().models[0],  # type: ignore[index]
        "no-entry": None,
    }
    named = lambda who: _serving(  # noqa: E731
        SimpleNamespace(served_candidate=who, entries=entries)
    ).not_honoured_by_answer()
    assert named(strata) == ("toolChoice",)
    assert named(honours) == (), "an honouring backend answered: nothing was ignored"
    assert named("no-entry") == (), "a backend with no listing names nothing"
    assert named(None) == ("parallelToolCalls",), "nobody known: the first attempt's"


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


@pytest.mark.parametrize("door", ["chat", "messages", "responses"])
def test_a_stream_whose_turn_waits_for_room_in_a_pool_still_opens_at_once(
    settings: Settings, door: str
) -> None:
    """The wait for room in a shared pool (CB3) is after the head, as it always
    was. Reading which backend is asked first asks the budget who has room now and
    never waits for room, so a turn that must wait for it (its only replica's pool
    is full) has its head out, with the header on it, long before it is let in."""
    fake = _backend(runtime="rt-a")
    fake.responses = ["pelican"]
    app = _app(settings, fake, runtimes=POOLED[:1], room_wait_seconds=0.6)
    app.state.routing._ledger.take((None, "rt-a"), 9_500)
    timeline = Timeline(app)
    with TestClient(timeline) as client:
        r = _post(client, door, stream=True, **FORCED[door])
    assert r.status_code == 200 and "pelican" in r.text
    opened, headers = timeline.head()
    token = timeline.first(b"pelican")
    assert token - opened >= 0.5, f"the head came {token - opened:.2f}s before the first token"
    assert headers[HEADER].split(", ")[-1] == "tool_choice"


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
    #: A driver restarting: its engine is loading, so its listing answers 503.
    info_down: ClassVar[bool] = False
    #: A driver answering its listing some other way: its status and raw body.
    info_answer: ClassVar[tuple[int, bytes] | None] = None

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
        elif path == "/v1/info" and self.info_down:
            problem = {"type": "about:blank", "title": "The engine is still loading", "status": 503}
            self._send(503, {"detail": problem})
        elif path == "/v1/info" and self.info_answer is not None:
            code, raw = self.info_answer
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
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


def test_a_real_driver_that_answers_its_listing_503_is_named_by_what_it_said(
    wired: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real routing table and HTTP client over a real socket: the driver
    restarts, and while its engine loads it answers `/v1/info` 503. The re-read of
    its listing that a request with tools makes meets the real error, and the
    caller is told what the driver said, as a 503 to try again, not that its tools
    are the trouble."""
    monkeypatch.setattr(_Install, "info_down", True)
    r = _post(wired, "chat")
    assert r.status_code == 503, r.text
    assert r.headers["Retry-After"] == "5"
    message = r.json()["error"]["message"]
    assert "driver 'strata-box' (http://127.0.0.1:" in message
    assert "answered 503: The engine is still loading" in message
    assert "Remove" not in message


#: What a real driver may answer its listing with, through the real client over a real
#: socket: the status, the body, what the caller is told and the status it is told with.
REAL_ANSWERS = [
    pytest.param(
        (500, b'{"detail": "boom"}'), 503, "answered 500: boom", True, id="a 500: waiting helps"
    ),
    pytest.param(
        (403, b'{"detail": {"title": "Invalid token", "status": 403}}'),
        502,
        "refused the gateway's credential (answered 403: Invalid token)",
        False,
        id="a 403: the credential is refused",
    ),
    pytest.param(
        (404, b'{"detail": "Not Found"}'),
        502,
        "does not serve GET /v1/info (answered 404: Not Found)",
        False,
        id="a 404: not an inference driver",
    ),
    pytest.param(
        (200, b"this is not json"),
        502,
        "something that is not a driver listing (Expecting value: line 1 column 1 (char 0))",
        False,
        id="a 200 that is not JSON",
    ),
    pytest.param(
        (200, b'{"nope": 1}'),
        502,
        "something that is not a driver listing (",
        False,
        id="a 200 that is not a listing",
    ),
]


@pytest.mark.parametrize(("answer", "status", "shown", "waits"), REAL_ANSWERS)
def test_a_real_driver_that_answers_its_listing_wrongly_is_named_by_how_and_waited_for_only_if_it_can_pass(
    wired: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    answer: tuple[int, bytes],
    status: int,
    shown: str,
    waits: bool,
) -> None:
    """The real exceptions of the real client, not a fake's: `HTTPStatusError` from
    `raise_for_status`, `JSONDecodeError` from `.json()` and pydantic's
    `ValidationError` from `DriverInfo.model_validate` are what classes a cause as
    one a retry can mend or not."""
    monkeypatch.setattr(_Install, "info_answer", answer)
    r = _post(wired, "chat")
    assert r.status_code == status, r.text
    assert ("Retry-After" in r.headers) is waits
    message = r.json()["error"]["message"]
    assert "driver 'strata-box' (http://127.0.0.1:" in message
    assert shown in message
    assert ("try again in a few seconds" in message) is waits
    assert "Remove" not in message


@pytest.fixture
def header_lookups(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every read of the driver's per-request header, by anything outside httpx, on
    headers that carry the stand-in driver's value for it. Spies on every way
    `httpx.Headers` gives a header up:

    - **by key**: `get`, `get_list`, `[]` and `in`, recorded as the key was looked
      up (lower-cased); reading by key is how a module would read it, and
      `dict(...)` does it through `[]` too;
    - **the whole of them**: `items`, `multi_items`, `raw`, `keys`, `values`,
      iteration and `copy`, each of which includes it, so each is recorded as the
      header. A module that logged or forwarded a driver's answer headers wholesale
      would read it this way, and a spy that only watched the keys would pass it.

    **A whole read from inside httpx is not recorded**: its own cookie handling calls
    `multi_items()` on every response it receives, which is not a module here
    reading anything (`test_the_spy_does_not_count_httpxs_own_reads...`). Who is
    reading is the frame that called the spied method.
    """
    looked: list[str] = []
    driver_says = _Install.per_request.encode()

    def carries(headers: httpx.Headers) -> bool:
        # `_list`, not `raw`, `items` or `get`: they are spied below.
        return any(
            key == HEADER.encode() and value == driver_says for _raw, key, value in headers._list
        )

    def by_key(method: Any) -> Any:
        @functools.wraps(method)
        def spy(self: httpx.Headers, key: str, *args: Any, **kwargs: Any) -> Any:
            if str(key).lower() == HEADER and carries(self):
                looked.append(str(key).lower())
            return method(self, key, *args, **kwargs)

        return spy

    def whole(method: Any) -> Any:
        @functools.wraps(method)
        def spy(self: httpx.Headers, *args: Any, **kwargs: Any) -> Any:
            # Frame 1 is whoever called the spied method.
            if carries(self) and not sys._getframe(1).f_globals.get("__name__", "").startswith(
                "httpx."
            ):
                looked.append(HEADER)
            return method(self, *args, **kwargs)

        return spy

    for name in ("get", "get_list", "__getitem__", "__contains__"):
        monkeypatch.setattr(httpx.Headers, name, by_key(getattr(httpx.Headers, name)))
    for name in ("items", "multi_items", "keys", "values", "__iter__", "copy"):
        monkeypatch.setattr(httpx.Headers, name, whole(getattr(httpx.Headers, name)))
    monkeypatch.setattr(httpx.Headers, "raw", property(whole(httpx.Headers.raw.fget)))
    return looked


#: Every way a module could read a header off an answer, and what the spy must say of it.
READS_BY_KEY = {
    "get": lambda h: h.get(HEADER.title()),
    "[]": lambda h: h[HEADER],
    "in": lambda h: HEADER in h,
    "get_list": lambda h: h.get_list(HEADER),
}
READS_OF_THE_WHOLE = {
    "items": lambda h: h.items(),
    "multi_items": lambda h: h.multi_items(),
    "raw": lambda h: h.raw,
    "keys": lambda h: h.keys(),
    "values": lambda h: h.values(),
    "iteration": lambda h: list(h),
    "copy": lambda h: h.copy(),
    "dict()": lambda h: dict(h),
}


@pytest.mark.parametrize("read", {**READS_BY_KEY, **READS_OF_THE_WHOLE})
def test_the_spy_sees_each_way_the_drivers_header_can_be_read(
    header_lookups: list[str], read: str
) -> None:
    """A check that cannot fail checks nothing. Every way of reading is recorded on
    headers carrying the driver's value, however it is spelled or asked."""
    driver = httpx.Response(200, headers={HEADER: _Install.per_request, "retry-after": "5"})
    ask = {**READS_BY_KEY, **READS_OF_THE_WHOLE}[read]
    ask(driver.headers)
    assert set(header_lookups) == {HEADER}, "a read the spy cannot see would pass any test"


@pytest.mark.parametrize("read", {**READS_BY_KEY, **READS_OF_THE_WHOLE})
def test_the_spy_does_not_see_the_gateways_own_header_of_that_name(
    header_lookups: list[str], read: str
) -> None:
    """The same header with another value is the gateway's own, which its callers
    read (and the tests do), by every one of these ways."""
    gateway = httpx.Response(200, headers={HEADER: "tool_choice", "retry-after": "5"})
    {**READS_BY_KEY, **READS_OF_THE_WHOLE}[read](gateway.headers)
    assert header_lookups == []


def test_the_spy_does_not_see_the_other_headers_of_the_drivers_answer(
    header_lookups: list[str],
) -> None:
    """What else the driver said (a retry hint) is read by key, and is not its
    per-request header."""
    driver = httpx.Response(200, headers={HEADER: _Install.per_request, "retry-after": "5"})
    assert driver.headers.get("retry-after") == "5" and driver.headers["Retry-After"] == "5"
    assert "retry-after" in driver.headers and driver.headers.get_list("retry-after") == ["5"]
    assert header_lookups == []


def test_the_spy_does_not_count_httpxs_own_reads_of_a_driver_answer(
    header_lookups: list[str],
) -> None:
    """httpx reads every response's headers whole for its cookies (`multi_items()`),
    which is the library and not a module here. Counted, every request through the
    real client would be a read of the driver's header, and the check below would
    fail for nothing; a spy that did not tell the two apart could not be used."""

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={}, headers={HEADER: _Install.per_request})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        response = client.get("http://driver/v1/generate")
    assert response.status_code == 200
    assert header_lookups == [], "httpx's own cookie handling is not a module reading it"
    # The same read, made by a caller outside httpx, is the one that is recorded.
    response.headers.multi_items()
    assert header_lookups == [HEADER]


@pytest.mark.parametrize("streamed", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("door", ["chat", "messages", "responses"])
def test_no_module_reads_the_header_the_driver_answers_with(
    header_lookups: list[str],
    wired: TestClient,
    stand_in: tuple[str, list[dict[str, Any]]],
    door: str,
    streamed: bool,
) -> None:
    """The driver also says `x-eugene-plexus-ignored-settings` on every answer, for
    direct callers, and the gateway names from the listing instead. A module that
    read it, to name from it, to log it or to hold a head for it (as round one did,
    and held every stream's head), would look it up on the driver's answer. Nothing
    does, on any door, whole or streamed, through the real client over a real
    socket; and what the driver said never reaches the caller."""
    r = _post(wired, door, stream=streamed, **FORCED[door])
    assert r.status_code == 200, r.text
    assert WIRE_TOKEN in r.text
    assert stand_in[1], "the driver was asked, and answered with its header"
    assert header_lookups == []
    assert _Install.per_request not in r.headers.get(HEADER, "")


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


@pytest.mark.parametrize("knobs", UNSAID)
def test_a_refusal_for_another_setting_does_not_also_name_the_tool_settings_it_admitted(
    settings: Settings, knobs: dict[str, Any]
) -> None:
    """`tool_choice` and `parallel_tool_calls` are admitted for a backend that says
    nothing of tool calling, whatever its list holds: they are settings it takes.
    When the request is refused for `top_k`, which its list lacks, the refusal
    names `top_k` and only that: the others were never the problem, and "remove
    them" would send the caller off to take out what was fine."""
    fake = _short(FakeDriverClient(name="custom", model_id=MODEL, **knobs), "maxTokens")
    with TestClient(_app(settings, fake)) as client:
        r = _tools_post(client, tool_choice="required", parallel_tool_calls=False, top_k=5)
    error = r.json()["error"]
    assert r.status_code == 400
    assert error["param"] == "top_k"
    assert error["message"].startswith(
        "No permitted backend serving this model takes top_k. Remove it or choose"
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
# A listing that cannot be confirmed is not a refusal of the request
# --------------------------------------------------------------------------- #


def _answered(status: int, detail: str = "") -> httpx.HTTPStatusError:
    """What the real client raises when a driver answers `status` to GET /v1/info."""
    request = httpx.Request("GET", "http://fake-driver/v1/info")
    problem: dict[str, Any] = {"type": "about:blank", "title": f"Status {status}", "status": status}
    if detail:
        problem["detail"] = detail
    response = httpx.Response(status, request=request, json={"detail": problem})
    return httpx.HTTPStatusError(f"status {status}", request=request, response=response)


def _not_json() -> ValueError:
    """What the real client raises for a 200 whose body is not JSON."""
    try:
        httpx.Response(200, content=b"this is not json").json()
    except ValueError as exc:
        return exc
    raise AssertionError("not JSON parsed")


def _not_a_listing() -> ValueError:
    """What the real client raises for a 200 whose body is JSON and not a `DriverInfo`
    (pydantic's `ValidationError`, which is a `ValueError`)."""
    try:
        DriverInfo.model_validate({"nope": 1})
    except ValueError as exc:
        return exc
    raise AssertionError("a listing validated")


@pytest.mark.parametrize("door", ["chat", "messages", "responses"])
def test_a_driver_whose_listing_cannot_be_read_is_a_retryable_503_that_names_it(
    settings: Settings, door: str
) -> None:
    """Admission re-reads the listing of a request that carries tools. A driver
    that does not answer (down, restarting) was left out without a word, and the
    request was then refused as if it were at fault: a 400, "takes tools, remove
    it", which a client such as Codex does not retry. It is a 503 with a
    `Retry-After`, naming the driver and what was observed."""
    fake = _backend("strata-box")
    fake.info_error = httpx.ConnectError("All connection attempts failed")
    with TestClient(_app(settings, fake)) as client:
        r = _post(client, door)
    assert r.status_code == 503, r.text
    assert r.headers["Retry-After"] == "5"
    message = r.json()["error"]["message"]
    assert "driver 'strata-box' (http://fake-driver)" in message
    assert "could not connect to fake-driver" in message
    assert "All connection attempts failed" in message
    assert "tools" in message, "what could not be confirmed is named"
    assert "not a refusal of the request" in message and "try again" in message
    assert "Remove" not in message and "No permitted backend" not in message
    assert fake.calls == []


def test_a_driver_that_does_not_answer_in_time_is_named_with_the_deadline(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(admission, "CONFIRM_TIMEOUT_SECONDS", 0.05)
    fake = _backend("slow-box")

    async def never(*, models: bool = True, model: str | None = None) -> Any:
        await asyncio.sleep(30)

    fake.info = never  # type: ignore[method-assign]
    with TestClient(_app(settings, fake)) as client:
        r = _tools_post(client)
    assert r.status_code == 503, r.text
    assert "driver 'slow-box'" in r.json()["error"]["message"]
    assert "no answer to GET /v1/info within 0.05 s" in r.json()["error"]["message"]
    assert fake.calls == []


def test_a_driver_that_answers_an_error_is_named_with_its_answer(settings: Settings) -> None:
    fake = _backend("restarting")
    request = httpx.Request("GET", "http://fake-driver/v1/info")
    fake.info_error = httpx.HTTPStatusError(
        "boom", request=request, response=httpx.Response(503, request=request)
    )
    with TestClient(_app(settings, fake)) as client:
        r = _tools_post(client)
    assert r.status_code == 503
    assert "driver 'restarting'" in r.json()["error"]["message"]
    assert "answered 503" in r.json()["error"]["message"]


def test_a_driver_that_could_not_be_read_is_not_counted_as_one_that_said_no(
    settings: Settings,
) -> None:
    """One driver cannot be read and the other says it carries no tools. Nothing
    is known of the first, so "every driver serving it reports that tools cannot
    be carried to it" would be false, and "takes tools, remove it" worse."""
    unread = FakeDriverClient(name="a-unread", model_id=MODEL, supports_tools=True)
    unread.info_error = httpx.ConnectError("refused")
    no = FakeDriverClient(name="b-no", model_id=MODEL, supports_tools=False)
    with TestClient(_app(settings, unread, no)) as client:
        r = _tools_post(client)
    message = r.json()["error"]["message"]
    assert r.status_code == 503, r.text
    assert "'a-unread'" in message and "b-no" not in message
    assert "cannot be carried" not in message and "Remove" not in message


def test_an_account_whose_listing_cannot_be_read_is_named_once(settings: Settings) -> None:
    """Candidates are per model: an account serving two of a slot's targets is one
    driver, and is said once."""
    account = FakeDriverClient(name="openrouter", models=["m1", "m2"], account=True)
    account.info_error = httpx.ConnectError("refused")
    slots = [{"model": MODEL, "targets": ["openrouter/m1", "openrouter/m2"]}]
    with TestClient(_app(settings, account, slots=slots)) as client:
        r = _tools_post(client)
    assert r.status_code == 503, r.text
    assert r.json()["error"]["message"].count("driver 'openrouter'") == 1


def test_a_driver_that_could_not_be_read_is_left_out_when_another_can_serve(
    settings: Settings,
) -> None:
    unread = FakeDriverClient(name="a-unread", model_id=MODEL, supports_tools=True)
    unread.info_error = httpx.ConnectError("refused")
    fine = FakeDriverClient(name="b-fine", model_id=MODEL, supports_tools=True)
    fine.responses = ["from b"]
    with TestClient(_app(settings, unread, fine)) as client:
        r = _tools_post(client)
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == "from b"
    assert unread.calls == []


def test_a_driver_that_answers_without_the_model_is_named_and_not_blamed_on_the_request(
    settings: Settings,
) -> None:
    """A restarted driver's list is empty until its engine loads: the model it
    listed at the last refresh is not there now. Said as that, a retryable 503."""
    fake = _backend("restarted")
    app = _app(settings, fake)
    fake.model_id = None  # its list is empty from here
    with TestClient(app) as client:
        r = _tools_post(client)
    assert r.status_code == 503, r.text
    message = r.json()["error"]["message"]
    assert "driver 'restarted'" in message
    assert (
        f"answered GET /v1/info without {MODEL!r}, which it listed at the last refresh" in message
    )
    assert fake.calls == []


def test_a_driver_that_now_follows_another_runtime_is_named_and_not_blamed_on_the_request(
    settings: Settings,
) -> None:
    fake = _backend("moved")
    app = _app(settings, fake)
    fake.runtime = "rt-new"  # reconfigured since the last refresh
    with TestClient(app) as client:
        r = _tools_post(client)
    assert r.status_code == 503, r.text
    message = r.json()["error"]["message"]
    assert "driver 'moved'" in message
    assert "now follows runtime 'rt-new', not no runtime as at the last refresh" in message
    assert fake.calls == []


def test_a_driver_that_answers_and_does_not_take_the_setting_is_still_the_400(
    settings: Settings,
) -> None:
    """The positive control for the rule above: a listing that WAS read, and says
    the backend does not take what was sent, is the request's refusal, as ever."""
    fake = _short(FakeDriverClient(name="said", model_id=MODEL, supports_tools=True), "maxTokens")
    with TestClient(_app(settings, fake)) as client:
        r = _tools_post(client, top_k=5)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "top_k"


#: Causes waiting can clear, and how each is said: a 503 with a `Retry-After`.
WAITING_CLEARS = [
    pytest.param(
        httpx.ConnectError("refused"),
        "could not connect to fake-driver (refused): nothing is listening there",
        id="connection refused",
    ),
    pytest.param(
        httpx.ConnectTimeout(""),
        "no answer within 10 s connecting to fake-driver",
        id="connect timeout",
    ),
    pytest.param(
        httpx.ReadTimeout(""),
        "connected to fake-driver, then no answer within 4 s",
        id="read timeout",
    ),
    pytest.param(httpx.ReadError("connection reset"), "ReadError: connection reset", id="reset"),
    pytest.param(
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
        "RemoteProtocolError: Server disconnected without sending a response.",
        id="hung up",
    ),
    *(
        pytest.param(_answered(status, said), f"answered {status}: {said}", id=f"a {status}")
        for status, said in [
            (500, "boom"),
            (502, "bad gateway"),
            (503, "The engine is still loading"),
            (504, "slow"),
            (408, "too slow"),
            (429, "slow down"),
        ]
    ),
]

#: Causes it cannot, how each is said, what to look at, and the error type: a 502.
WAITING_DOES_NOT_CLEAR = (
    [
        pytest.param(
            _answered(status, "bad token"),
            f"refused the gateway's credential (answered {status}: bad token)",
            admission.CREDENTIAL_REFUSED_NEXT_STEP,
            "upstream_auth_error",
            id=f"a {status}",
        )
        for status in (401, 403)
    ]
    + [
        pytest.param(
            _answered(status, "nope"),
            f"does not serve GET /v1/info (answered {status}: nope)",
            admission.NOT_A_DRIVER_NEXT_STEP,
            "upstream_error",
            id=f"a {status}",
        )
        for status in (400, 404, 405, 410, 422, 301)
    ]
    + [
        pytest.param(
            _not_json(),
            "answered GET /v1/info with something that is not a driver listing "
            "(Expecting value: line 1 column 1 (char 0))",
            admission.NOT_A_DRIVER_NEXT_STEP,
            "upstream_error",
            id="not JSON",
        ),
        pytest.param(
            _not_a_listing(),
            "answered GET /v1/info with something that is not a driver listing (",
            admission.NOT_A_DRIVER_NEXT_STEP,
            "upstream_error",
            id="not a listing",
        ),
        # Not claimed to be passing: said as what it is, with nothing to look at.
        pytest.param(
            RuntimeError("boom"), "RuntimeError: boom", None, "upstream_error", id="unknown"
        ),
    ]
)


@pytest.mark.parametrize(("error", "shown"), WAITING_CLEARS)
def test_a_cause_that_passes_is_a_503_with_a_retry_after(
    settings: Settings, error: Exception, shown: str
) -> None:
    """Timeouts, connect errors and a 5xx (and a 408 or 429): the driver may be back
    in seconds, and the caller is told so."""
    fake = _backend("strata-box")
    fake.info_error = error
    with TestClient(_app(settings, fake)) as client:
        r = _tools_post(client)
    assert r.status_code == 503, r.text
    assert r.headers["Retry-After"] == "5"
    message = r.json()["error"]["message"]
    assert f"driver 'strata-box' (http://fake-driver): {shown}" in message
    assert "try again in a few seconds" in message
    assert "Remove" not in message and "No permitted backend" not in message
    assert fake.calls == []


@pytest.mark.parametrize(("error", "shown", "step", "kind"), WAITING_DOES_NOT_CLEAR)
def test_a_cause_waiting_cannot_clear_is_a_502_naming_it_with_no_retry_advice(
    settings: Settings, error: Exception, shown: str, step: str | None, kind: str
) -> None:
    """A driver that refuses the gateway's credential, is not an inference driver or
    answers with what is not a listing is the same on the next try. It gets the
    treatment a credential refusal gets on the front door (`_driver_failure_status`):
    a 502, because the request was fine and the install is not, naming the observed
    cause, with no retry advice and no `Retry-After`."""
    fake = _backend("strata-box")
    fake.info_error = error
    with TestClient(_app(settings, fake)) as client:
        r = _tools_post(client)
    assert r.status_code == 502, r.text
    assert "Retry-After" not in r.headers
    body = r.json()["error"]
    assert body["type"] == kind
    message = body["message"]
    assert f"driver 'strata-box' (http://fake-driver): {shown}" in message
    assert "try again" not in message.lower() and "few seconds" not in message
    assert "This is not a refusal of the request" in message
    if step is not None:
        assert step in message
    assert "Remove" not in message and "No permitted backend" not in message
    assert fake.calls == []


@pytest.mark.parametrize("door", ["chat", "messages", "responses"])
def test_every_door_renders_the_502_in_its_own_envelope(settings: Settings, door: str) -> None:
    """The Anthropic door's error types are a closed set, so it keeps the one its
    status gets; the others say `upstream_auth_error`, as the front door's own 502
    for a driver refusing its credential does."""
    fake = _backend("strata-box")
    fake.info_error = _answered(403, "bad token")
    with TestClient(_app(settings, fake)) as client:
        r = _post(client, door)
    assert r.status_code == 502, r.text
    assert "Retry-After" not in r.headers
    body = r.json()
    assert "refused the gateway's credential (answered 403: bad token)" in body["error"]["message"]
    if door == "messages":
        assert body["type"] == "error" and body["error"]["type"] == "api_error"
    else:
        assert body["error"]["type"] == "upstream_auth_error"


def test_a_driver_that_will_answer_among_drivers_that_will_not_makes_it_a_retry(
    settings: Settings,
) -> None:
    """The request is served by the one that comes back, so waiting helps, whatever
    the other says; each is named by what it said, and what to look at is said for
    the one that needs it."""
    loading = _backend("a-loading")
    loading.info_error = httpx.ConnectError("refused")
    refused = _backend("b-refused")
    refused.info_error = _answered(403, "bad token")
    with TestClient(_app(settings, loading, refused)) as client:
        r = _tools_post(client)
    assert r.status_code == 503 and r.headers["Retry-After"] == "5"
    message = r.json()["error"]["message"]
    assert "driver 'a-loading'" in message and "could not connect" in message
    assert "driver 'b-refused'" in message and "refused the gateway's credential" in message
    assert "try again in a few seconds" in message
    assert message.count(admission.CREDENTIAL_REFUSED_NEXT_STEP) == 1


def test_two_drivers_that_refuse_say_what_to_look_at_once(settings: Settings) -> None:
    fakes = [_backend(f"driver-{i}", base_url=f"http://box-{i}") for i in "ab"]
    for fake in fakes:
        fake.info_error = _answered(403, "bad token")
    with TestClient(_app(settings, *fakes)) as client:
        r = _tools_post(client)
    assert r.status_code == 502
    message = r.json()["error"]["message"]
    assert "driver 'driver-a'" in message and "driver 'driver-b'" in message
    assert message.count(admission.CREDENTIAL_REFUSED_NEXT_STEP) == 1


# --------------------------------------------------------------------------- #
# What the refusal says: what was carried, which driver, how much of its answer
# --------------------------------------------------------------------------- #


def _carried(message: str) -> list[str]:
    """The settings an unconfirmed refusal says the request carries."""
    found = re.search(r"takes what this request carries \((.*?)\): driver", message)
    assert found, message
    return found.group(1).split(", ")


def test_what_a_request_carries_is_said_in_the_callers_words_in_a_readable_order() -> None:
    """The tools first and the settings that ride with them next, the others
    alphabetically: the same request is worded the same way every time, whatever
    order a set gives its members in, and in the spelling the chat door's caller
    uses (`tool_choice`, never `toolChoice`)."""
    names = ["topK", "parallelToolCalls", "maxTokens", "toolChoice", "tools", "temperature"]
    expected = [
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "max_tokens",
        "temperature",
        "top_k",
    ]
    assert ignored_settings.carried(names) == expected
    assert ignored_settings.carried(reversed(names)) == expected
    # `tools` only when the caller sent the key; a field nobody has listed is as it came.
    assert ignored_settings.carried(["topK"]) == ["top_k"]
    assert ignored_settings.carried(["mirostatTau"]) == ["mirostatTau"]
    assert ignored_settings.carried([]) == []


def test_a_refusal_names_every_setting_the_caller_sent_in_its_words(settings: Settings) -> None:
    """tool_choice, parallel_tool_calls and top_k, with the tools: said as the caller
    wrote them and not as the driver does, in the order above."""
    fake = _backend("strata-box")
    fake.info_error = httpx.ConnectError("refused")
    with TestClient(_app(settings, fake)) as client:
        r = _tools_post(client, tool_choice="required", parallel_tool_calls=False, top_k=5)
    assert r.status_code == 503, r.text
    message = r.json()["error"]["message"]
    assert "tool_choice, parallel_tool_calls, top_k" in message
    assert "(tools, tool_choice, parallel_tool_calls, top_k)" in message
    assert (
        "toolChoice" not in message and "parallelToolCalls" not in message and "topK" not in message
    )
    assert _carried(message) == ["tools", "tool_choice", "parallel_tool_calls", "top_k"]


@pytest.mark.parametrize("door", ["messages", "responses"])
def test_the_other_doors_say_what_they_carry_in_the_same_words_and_order(
    settings: Settings, door: str
) -> None:
    fake = _backend("strata-box")
    fake.info_error = httpx.ConnectError("refused")
    extra = {
        "messages": {"tool_choice": {"type": "any", "disable_parallel_tool_use": True}},
        "responses": {"tool_choice": "required", "parallel_tool_calls": False},
    }[door]
    with TestClient(_app(settings, fake)) as client:
        r = _post(client, door, **extra)
    assert r.status_code == 503, r.text
    carried = _carried(r.json()["error"]["message"])
    assert carried[:3] == ["tools", "tool_choice", "parallel_tool_calls"]
    assert "toolChoice" not in carried and "parallelToolCalls" not in carried


def test_tools_sent_empty_are_still_said_to_be_carried(settings: Settings) -> None:
    """`tools: []` is the key with nothing in it, and a caller that sends it said
    something: the refusal that names what was carried said "()" for it, which names
    nothing. (The chat door forwards the key; the Anthropic and Responses doors read
    an empty list as no tools at all.)"""
    fake = _backend("strata-box")
    fake.info_error = httpx.ConnectError("refused")
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/chat/completions", json=_chat(tools=[]))
    assert r.status_code == 503, r.text
    message = r.json()["error"]["message"]
    assert "()" not in message
    assert _carried(message) == ["tools"]


def test_a_driver_that_answers_without_the_model_says_why_when_it_says_so(
    settings: Settings,
) -> None:
    """A driver that could not build its engine reports why in its own words
    (`DriverInfo.degraded`), and "the model is gone" is only half of that."""
    fake = _backend("restarted", degraded="could not build its engine: the model file is missing")
    app = _app(settings, fake)
    fake.model_id = None
    with TestClient(app) as client:
        r = _tools_post(client)
    assert r.status_code == 503, r.text
    assert (
        f"answered GET /v1/info without {MODEL!r}, which it listed at the last refresh; "
        "the driver says it serves nothing: could not build its engine: the model file is missing"
    ) in r.json()["error"]["message"]


def test_what_a_degraded_driver_says_is_capped_like_any_other_cause(settings: Settings) -> None:
    """Its own words may be as long as an error's: one line, whole words, an ellipsis."""
    fake = _backend("restarted", degraded="  not\nbuilt  " + "because " * 100)
    app = _app(settings, fake)
    fake.model_id = None
    with TestClient(app) as client:
        r = _tools_post(client)
    message = r.json()["error"]["message"]
    cause = message.split("driver 'restarted' (http://fake-driver): ", 1)[1].split(
        ". This is not a refusal", 1
    )[0]
    assert len(cause) <= admission.MAX_CAUSE_CHARS and cause.endswith("because...")
    assert "\n" not in message and "  " not in message


def test_a_driver_that_answers_without_the_model_and_says_nothing_more_adds_nothing(
    settings: Settings,
) -> None:
    fake = _backend("restarted")
    app = _app(settings, fake)
    fake.model_id = None
    with TestClient(app) as client:
        r = _tools_post(client)
    message = r.json()["error"]["message"]
    assert "listed at the last refresh." in message and "serves nothing" not in message


def test_two_drivers_of_one_name_on_two_nodes_are_each_named_with_their_node(
    settings: Settings,
) -> None:
    """Two machines run one driver name: neither the key nor the words may merge them.
    Both cannot be read, and the refusal names both, each on its own node."""
    fakes = [
        _backend("strata", node=node, base_url=f"http://box-{node}") for node in ("amish", "nas")
    ]
    for fake in fakes:
        fake.info_error = httpx.ConnectError("refused")
    with TestClient(_app(settings, *fakes)) as client:
        r = _tools_post(client)
    assert r.status_code == 503, r.text
    message = r.json()["error"]["message"]
    assert message.count("driver 'strata'") == 2
    assert "driver 'strata' on node 'amish' (http://box-amish): " in message
    assert "driver 'strata' on node 'nas' (http://box-nas): " in message


def test_a_driver_with_no_node_is_named_without_one() -> None:
    backend = SimpleNamespace(name="strata", node=None, url="http://box")
    said = admission.unread_because(backend, httpx.ConnectError("refused")).text
    assert said.startswith("driver 'strata' (http://box): ")
    assert " on node " not in said


def test_a_long_answer_is_one_line_cut_at_a_word_with_an_ellipsis() -> None:
    """A malformed listing's validation error is many lines. What a person sees is
    one line of whole words, ended with an ellipsis, in the length the cap says."""
    backend = SimpleNamespace(name="a", node=None, url="http://a")
    cause = ValueError("one  two\n\tthree " + "word " * 100 + "\n   last words here")
    said = admission.unread_because(backend, cause).text
    head = "driver 'a' (http://a): "
    assert said.startswith(head)
    body = said[len(head) :]
    assert "\n" not in said and "\t" not in said and "  " not in said
    assert len(body) <= admission.MAX_CAUSE_CHARS
    assert body.endswith("...")
    kept = body[: -len("...")]
    flat = " ".join(
        f"answered GET /v1/info with something that is not a driver listing ({cause})".split()
    )
    assert flat.startswith(kept) and flat[len(kept)] == " ", "the cut falls between two words"


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        # Collapsed to one line, and left alone when it fits (exactly at the limit too).
        ("  one   two\nthree  ", 40, "one two three"),
        ("a" * 40, 40, "a" * 40),
        ("aaaa bbbb cccc dddd", 19, "aaaa bbbb cccc dddd"),
        # Cut inside a word: back to the word before it.
        ("aaaa bbbb cccc dddd", 18, "aaaa bbbb cccc..."),
        ("aaaa bbbb cccc dddd", 15, "aaaa bbbb..."),
        ("aaaa bbbb cccc dddd", 11, "aaaa..."),
        # Cut exactly between two words: both are kept.
        ("aaaa bbbb cccc dddd", 12, "aaaa bbbb..."),
        ("aaaa bbbb cccc dddd", 13, "aaaa bbbb..."),
        # One word is all there is: cut where the room ends.
        ("x" * 100, 20, "x" * 17 + "..."),
    ],
)
def test_the_cap_cuts_between_words_and_ends_with_an_ellipsis(
    text: str, limit: int, expected: str
) -> None:
    said = admission._plain(text, limit)
    assert said == expected
    assert len(said) <= limit


# --------------------------------------------------------------------------- #
# A key that requires local-only inference keeps its 403
# --------------------------------------------------------------------------- #


def test_a_local_only_keys_403_keeps_its_status_and_says_what_could_not_be_confirmed(
    keyed_count: Any,
) -> None:
    """The refusal is the policy's, whatever the cause, so it stays a 403 with no
    `Retry-After`; but a driver that could not be read is why nothing could confirm
    local execution, and the caller is told what was observed."""
    from tests.test_local_only import describe_as

    app, authority, allowed, _, headers = keyed_count
    authority.local_only = True
    describe_as(allowed, "local")
    allowed.info_error = httpx.ConnectError("refused")
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions", json=_chat(model="allowed", tools=[WEATHER]), headers=headers
        )
    assert r.status_code == 403, r.text
    assert "Retry-After" not in r.headers
    message = r.json()["error"]["message"]
    assert message.startswith("This key requires local-only inference, but no permitted backend")
    assert (
        "Could not confirm: driver 'allowed-driver' (http://fake-driver): could not connect to "
        "fake-driver (refused): nothing is listening there."
    ) in message
    assert message.endswith("Nothing was forwarded or woken.")
    assert allowed.calls == []


def test_a_local_only_keys_403_with_nothing_unreadable_says_no_more_than_it_did(
    keyed_count: Any,
) -> None:
    """The positive control: a driver that was read and is not local is the policy's
    refusal alone."""
    from tests.test_local_only import describe_as

    app, authority, allowed, _, headers = keyed_count
    authority.local_only = True
    describe_as(allowed, "external")
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions", json=_chat(model="allowed", tools=[WEATHER]), headers=headers
        )
    assert r.status_code == 403, r.text
    assert r.json()["error"]["message"] == (
        "This key requires local-only inference, but no permitted backend confirms local "
        "execution and policy enforcement. Nothing was forwarded or woken."
    )


# --------------------------------------------------------------------------- #
# Without tools, the tool settings say nothing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("off", ["listed", "learned"])
def test_a_request_with_no_tools_is_not_refused_the_tool_settings_by_a_model_with_tools_off(
    settings: Settings, off: str
) -> None:
    """Codex sends `tool_choice` and `parallel_tool_calls` on every request. A model
    whose tool calling is off (its listing says so from the start, or its driver
    learned it from the backend since the table was built) and a list of settings
    that leaves them out: without tools they say nothing, and the driver drops them
    with a warning. Refused, a caller could not chat with it at all."""
    fake = _short(
        FakeDriverClient(name="cli", model_id=MODEL, supports_tools=off != "listed"), "maxTokens"
    )
    app = _app(settings, fake)
    if off == "learned":
        fake.supports_tools = False  # said by the driver after the table was built
    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions", json=_chat(tool_choice="auto", parallel_tool_calls=False)
        )
    assert r.status_code == 200, r.text
    sent = fake.calls[0]
    assert sent.toolChoice == ToolChoice.auto and sent.parallelToolCalls is False
    assert not sent.tools


def test_the_same_model_is_still_refused_tools_in_the_drivers_words(settings: Settings) -> None:
    """The exemption is for a request that carries no tools. One that does is turned
    away, as it always was, whatever settings come with it."""
    fake = _short(FakeDriverClient(name="cli", model_id=MODEL, supports_tools=False), "maxTokens")
    with TestClient(_app(settings, fake)) as client:
        r = _tools_post(client, tool_choice="auto", parallel_tool_calls=False)
    assert r.status_code == 400
    assert r.json()["error"]["message"] == tools_refused(MODEL, 1)
    assert fake.calls == []


def test_the_exemption_does_not_name_the_tool_settings_in_a_refusal_for_another(
    settings: Settings,
) -> None:
    """`top_k` is the trouble, and it is the only thing named: the tool settings were
    admitted, so "remove them" would send the caller to take out what was fine."""
    fake = _short(FakeDriverClient(name="cli", model_id=MODEL, supports_tools=False), "maxTokens")
    with TestClient(_app(settings, fake)) as client:
        r = client.post(
            "/v1/chat/completions",
            json=_chat(tool_choice="auto", parallel_tool_calls=False, top_k=5),
        )
    error = r.json()["error"]
    assert r.status_code == 400 and error["param"] == "top_k"
    assert error["message"].startswith(
        "No permitted backend serving this model takes top_k. Remove it"
    )


@pytest.mark.parametrize("setting", ["tool_choice", "parallel_tool_calls"])
def test_a_model_that_calls_tools_and_does_not_list_the_setting_is_still_refused_it(
    settings: Settings, setting: str
) -> None:
    """The exemption is for a model whose tool calling is OFF. One whose driver says
    it calls tools and lists what it takes keeps the rule that its list is the list."""
    fake = _short(FakeDriverClient(name="said", model_id=MODEL, supports_tools=True), "maxTokens")
    extra = {"tool_choice": "auto"} if setting == "tool_choice" else {"parallel_tool_calls": False}
    with TestClient(_app(settings, fake)) as client:
        r = client.post("/v1/chat/completions", json=_chat(**extra))
    assert r.status_code == 400
    assert setting in r.json()["error"]["message"]
    assert fake.calls == []


def test_the_credential_refusal_is_the_front_doors_sentence() -> None:
    """One sentence for what to do about a driver that refuses the gateway's
    credential, said by admission and by the front door's 502 for a generate: the
    two cannot drift."""
    failure = inference._driver_failure_status(
        DriverError(
            driver_name="d", driver_url="http://d", status_code=403, problem=None, raw_body="no"
        )
    )
    assert failure.code == 502 and failure.error_type == "upstream_auth_error"
    assert failure.message.endswith(admission.CREDENTIAL_REFUSED_NEXT_STEP)


def test_the_message_names_what_the_request_carries_in_the_callers_words() -> None:
    said = admission.unconfirmed_refusal(
        "qwen",
        ["tools", "tool_choice"],
        [admission.Unconfirmed("driver 'a' (http://a): no answer")],
    )
    assert said == (
        "Could not confirm that a backend serving 'qwen' takes what this request carries "
        "(tools, tool_choice): driver 'a' (http://a): no answer. This is not a refusal of the "
        "request: try again in a few seconds. Nothing was forwarded or woken."
    )


def test_a_cause_waiting_cannot_clear_says_what_to_look_at_and_not_to_try_again() -> None:
    """The retry advice is for causes that pass. A driver that refuses the gateway's
    credential or is not an inference driver is the same on the next try, so the
    sentence says what to check instead, once however many drivers say the same."""
    refused = admission.Unconfirmed(
        "driver 'a' (http://a): refused the gateway's credential (answered 403: no)",
        transient=False,
        remedy=admission.CREDENTIAL_REFUSED_NEXT_STEP,
        kind="upstream_auth_error",
    )
    other = replace(refused, text="driver 'b' (http://b): refused the gateway's credential")
    said = admission.unconfirmed_refusal("qwen", ["tools"], [refused, other])
    assert said == (
        "Could not confirm that a backend serving 'qwen' takes what this request carries "
        "(tools): driver 'a' (http://a): refused the gateway's credential (answered 403: no); "
        "driver 'b' (http://b): refused the gateway's credential. This is not a refusal of the "
        f"request. {admission.CREDENTIAL_REFUSED_NEXT_STEP} Nothing was forwarded or woken."
    )
    assert "try again" not in said


def test_a_cause_that_passes_among_causes_that_do_not_keeps_the_retry_advice() -> None:
    """One driver that is about to answer is worth a retry, whatever the other says:
    the request is served by the one that does, once it does."""
    loading = admission.Unconfirmed("driver 'a' (http://a): answered 503: loading")
    wrong = admission.Unconfirmed(
        "driver 'b' (http://b): does not serve GET /v1/info (answered 404: nope)",
        transient=False,
        remedy=admission.NOT_A_DRIVER_NEXT_STEP,
    )
    said = admission.unconfirmed_refusal("qwen", ["tools"], [loading, wrong])
    assert "try again in a few seconds" in said
    assert said.count(admission.NOT_A_DRIVER_NEXT_STEP) == 1
    assert admission.unconfirmed_failure("qwen", ["tools"], [loading, wrong]).status == 503


def test_what_waiting_cannot_clear_is_a_502_with_no_retry_after_and_the_error_type_of_the_cause() -> (
    None
):
    """The treatment `_driver_failure_status` gives a driver that refuses the
    gateway's credential (a 502: the request was fine and the install is not)."""
    refused = admission.Unconfirmed("x", transient=False, kind="upstream_auth_error")
    wrong = admission.Unconfirmed("y", transient=False)
    failure = admission.unconfirmed_failure("qwen", ["tools"], [refused, wrong])
    assert (failure.status, failure.retry, failure.kind) == (502, None, "upstream_auth_error")
    failure = admission.unconfirmed_failure("qwen", ["tools"], [wrong])
    assert (failure.status, failure.retry, failure.kind) == (502, None, "upstream_error")
    failure = admission.unconfirmed_failure("qwen", ["tools"], [admission.Unconfirmed("z")])
    assert (failure.status, failure.retry, failure.kind) == (503, "5", None)


# --------------------------------------------------------------------------- #
# A count is a 400 and never a 5xx
# --------------------------------------------------------------------------- #


#: Both counting doors. `/context` retries a 5xx before it falls back to counting by
#: sending the request, so every "cannot count" is a 400 (`inference._cannot_count`).
COUNT_DOORS = [
    pytest.param("/v1/messages/count_tokens", _messages, id="messages"),
    pytest.param("/v1/responses/input_tokens", _responses, id="responses"),
]

#: A driver whose listing cannot be read, and how the count says so: the transient one
#: admission answers as a 503, and the one it answers as a 502 (a driver refusing the
#: gateway's own credential). Neither may reach a counting client as a 5xx.
UNREAD = [
    pytest.param(
        httpx.ConnectError("refused"),
        "could not connect to fake-driver (refused): nothing is listening there",
        id="nothing listening",
    ),
    pytest.param(
        _answered(403, "bad token"),
        "refused the gateway's credential (answered 403: bad token)",
        id="credential refused",
    ),
]


@pytest.mark.parametrize(("cause", "said"), UNREAD)
@pytest.mark.parametrize(("path", "body"), COUNT_DOORS)
def test_a_count_whose_driver_cannot_be_read_is_a_400_with_the_cause_and_never_a_5xx(
    settings: Settings, path: str, body: Any, cause: Exception, said: str
) -> None:
    """Admission re-reads the listing of a request that carries tools, and a driver
    it cannot read is a 503 (or a 502) on the doors that generate. A count is not
    one of them: Claude Code's `/context` retries a 5xx, so the count answers the
    400 it falls back on, with the same cause."""
    fake = _backend("strata-box")
    fake.prompt_tokens = 42  # it would count, if it were asked
    fake.info_error = cause
    with TestClient(_app(settings, fake)) as client:
        r = client.post(path, json=body())
    assert r.status_code == 400, r.text
    assert "Retry-After" not in r.headers
    message = r.json()["error"]["message"]
    assert message.startswith(
        f"Cannot count tokens for {MODEL!r} without generating: Could not confirm that a backend"
    )
    assert "driver 'strata-box' (http://fake-driver): " + said in message
    assert ".." not in message, "the failure's own full stop is not said twice"
    assert fake.count_calls == [] and fake.calls == []


@pytest.mark.parametrize(("cause", "said"), UNREAD)
@pytest.mark.parametrize(("path", "body"), COUNT_DOORS)
def test_a_count_whose_driver_goes_unreadable_across_the_refresh_is_a_400_too(
    settings: Settings, path: str, body: Any, cause: Exception, said: str
) -> None:
    """The count's second look: nothing is eligible at first (the engine is stopped,
    and a count wakes nothing), so the table is refreshed and the listing asked
    again, which is where this driver turns out to be unreadable."""
    fake = _backend("strata-box", runtime="qwen3")
    fake.prompt_tokens = 42
    runtimes = [runtime_facts("qwen3", alias=MODEL, status="stopped", start_on_demand=True)]
    app = _app(settings, fake, runtimes=runtimes)

    async def refreshed(max_age_seconds: float = 1.0) -> bool:
        fake.info_error = cause
        return True

    app.state.routing.refresh_if_stale = refreshed
    with TestClient(app) as client:
        r = client.post(path, json=body())
    assert r.status_code == 400, r.text
    assert said in r.json()["error"]["message"]
    assert fake.info_calls and fake.count_calls == []


@pytest.mark.parametrize(("path", "body"), COUNT_DOORS)
def test_a_count_still_refuses_for_the_request_with_the_400_it_always_did(
    settings: Settings, path: str, body: Any
) -> None:
    """The positive control: a listing that WAS read, and says no tool settings are
    taken, is the request's refusal, in the 400 it has always been."""
    fake = _short(FakeDriverClient(name="said", model_id=MODEL, supports_tools=False), "maxTokens")
    with TestClient(_app(settings, fake)) as client:
        r = client.post(path, json=body())
    assert r.status_code == 400, r.text
    assert tools_refused(MODEL, 1) in r.json()["error"]["message"]


# --------------------------------------------------------------------------- #
# `/v1/responses/input_tokens` is under client admission, as `/v1/responses` is
# --------------------------------------------------------------------------- #

INPUT_TOKENS = "/v1/responses/input_tokens"


@pytest.fixture
def keyed_count(settings: Settings, install: FakeInstall) -> Any:
    """An app key that may use `allowed` and not `excluded`, on the real middleware."""
    authority = Authority()
    token = install.client_key(name="app", jti="key-1")
    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    allowed = FakeDriverClient(name="allowed-driver", model_id="allowed", supports_tools=True)
    excluded = FakeDriverClient(name="excluded-driver", model_id="excluded", supports_tools=True)
    allowed.prompt_tokens, excluded.prompt_tokens = 11, 22
    app.state.routing = make_routing_table(allowed, excluded)
    return app, authority, allowed, excluded, {"authorization": f"Bearer {token}"}


def test_input_tokens_applies_the_keys_model_scope_and_checks_without_admitting(
    keyed_count: Any,
) -> None:
    """Under no admission, the path had no client context: a key that may not use a
    model was counted on it, and the key's limits and attribution applied to nothing.
    A count is a check and not a reservation (`/v1/messages/count_tokens`'s rule)."""
    app, authority, _, excluded, headers = keyed_count
    with TestClient(app) as client:
        counted = client.post(
            INPUT_TOKENS, json={**_responses(), "model": "allowed"}, headers=headers
        )
        refused = client.post(
            INPUT_TOKENS, json={**_responses(), "model": "excluded"}, headers=headers
        )
    assert counted.status_code == 200, counted.text
    assert counted.json() == {"object": "response.input_tokens", "input_tokens": 11}
    assert refused.status_code == 400, refused.text
    assert "unavailable or not permitted" in refused.json()["error"]["message"]
    assert "excluded-driver" not in refused.text
    assert excluded.count_calls == [], "a model the key may not use was not counted on"
    assert [call["action"] for call in authority.calls] == ["check", "check"]


def test_input_tokens_without_a_credential_is_refused_in_openais_shape(keyed_count: Any) -> None:
    app, authority, allowed, *_ = keyed_count
    with TestClient(app) as client:
        r = client.post(INPUT_TOKENS, json={**_responses(), "model": "allowed"})
    assert r.status_code == 401, r.text
    assert r.json()["error"]["message"]
    assert allowed.count_calls == [] and authority.calls == []


@pytest.mark.parametrize("locality", ["local", "external"])
def test_input_tokens_applies_a_local_only_key(keyed_count: Any, locality: str) -> None:
    """The key's `localOnly` reaches the driver on a local backend, and an external
    one is refused for it, with the status the policy always had (403)."""
    from tests.test_local_only import describe_as

    app, authority, allowed, _, headers = keyed_count
    authority.local_only = True
    describe_as(allowed, locality)
    with TestClient(app) as client:
        r = client.post(INPUT_TOKENS, json={**_responses(), "model": "allowed"}, headers=headers)
    if locality == "local":
        assert r.status_code == 200, r.text
        assert allowed.count_calls[-1].localOnly is True
    else:
        assert r.status_code == 403, r.text
        assert "local-only" in r.json()["error"]["message"]
        assert allowed.count_calls == []


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


@pytest.mark.parametrize(
    ("chat_says", "listed"),
    [(False, False), (None, True)],
    ids=["the chat model says no tools", "the chat model says nothing"],
)
def test_a_silent_model_that_does_not_chat_does_not_make_a_search_reach_the_slot(
    settings: Settings, chat_says: bool | None, listed: bool
) -> None:
    """`web_search` on `GET /v1/models` is about the chat models of a slot. A model
    that does not chat and says nothing of tools (an embedding model here) is not
    a chat model that is taken to call tools, so it does not make a search reach a
    slot whose chat model says it takes none; and the chat model's own silence
    still does."""
    chatty = FakeDriverClient(name="chatty", model_id="chatty", supports_tools=chat_says)
    embed = FakeDriverClient(
        name="embed", model_id="nomic", supports_embeddings=True, capabilities_reported=False
    )
    slots = [{"model": MODEL, "targets": ["chatty", "nomic"]}]
    with TestClient(_app(settings, chatty, embed, slots=slots)) as client:
        assert _models(client)[MODEL]["web_search"] is listed


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
