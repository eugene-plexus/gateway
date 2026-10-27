"""Remote MCP on the Responses door (row 59b).

A scripted model and a fake tool-driver stand in for an engine and a
remote MCP server; the acceptance run in
`specs/scripts/row59-remote-mcp-acceptance.py` drives the real processes,
including the tool-driver's address rules. What is pinned here:

- the server's tools are listed (or taken from an `mcp_list_tools` in
  `input`), offered to the model, and a call round-trips as `mcp_call`;
- a call that needs approval is not run: the response ends with an
  `mcp_approval_request`, and an `mcp_approval_response` for it runs it --
  an approval for an id the conversation does not carry is refused, and a
  call that ran is never run again;
- when it cannot run (the owner's setting, the key, no process that runs
  it), the 400 says why; a listing that fails is a 424, an address the
  rule refuses a 400 naming the rule;
- the caller's headers reach the tool-driver for the server, and nothing
  else: not the echo, not a log line, not a metrics row;
- the limit of 20 calls names itself.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway import admission, remote_mcp
from eugene_plexus_gateway._generated.driver_models import FunctionCall, ToolCall
from eugene_plexus_gateway.routing import ToolAccount
from eugene_plexus_gateway.tool_client import ToolDriverError, ToolDriverInfo

from .test_server_tools import MODEL, ScriptedDriver, Turn, app_with, events

TOKEN = "mcp-caller-token-do-not-leak-0123456789"
URL = "https://mcp.example.com/mcp"
TOOLS = [
    {
        "name": "echo",
        "description": "Says the text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "delete_everything",
        "description": "Deletes.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class FakeMcp:
    """A tool-driver's remote MCP endpoints, recording what reached them."""

    def __init__(self, *, tools: list[dict[str, Any]] | None = None) -> None:
        self.tools = TOOLS if tools is None else tools
        self.listed: list[dict[str, Any]] = []
        self.called: list[dict[str, Any]] = []
        self.list_error: ToolDriverError | None = None
        self.call_error: ToolDriverError | None = None

    async def mcp_list_tools(
        self, server_url: str, *, headers: dict[str, str], reach: str, allowed: Any
    ) -> list[dict[str, Any]]:
        self.listed.append(
            {"url": server_url, "headers": dict(headers), "reach": reach, "allowed": allowed}
        )
        if self.list_error is not None:
            raise self.list_error
        return [t for t in self.tools if allowed is None or t["name"] in allowed]

    async def mcp_call(
        self,
        server_url: str,
        *,
        headers: dict[str, str],
        reach: str,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        self.called.append(
            {
                "url": server_url,
                "headers": dict(headers),
                "reach": reach,
                "name": name,
                "arguments": arguments,
            }
        )
        if self.call_error is not None:
            raise self.call_error
        return f"echo: {arguments.get('text', '')}", None

    async def web_search(self, request: Any) -> Any:
        raise AssertionError("no search here")

    async def info(self) -> ToolDriverInfo:
        return ToolDriverInfo(provider="brave", label="Brave", tools=["mcp"], configured=False)

    async def aclose(self) -> None:
        return None


def mcp_account(fake: FakeMcp, name: str = "brave") -> ToolAccount:
    return ToolAccount(
        node=None,
        name=name,
        url=f"http://{name}.invalid:8190",
        client=fake,  # type: ignore[arg-type]
        info=ToolDriverInfo(provider="brave", label="Brave", tools=["mcp"], configured=False),
    )


def mcp_tool(**extra: Any) -> dict[str, Any]:
    return {
        "type": "mcp",
        "server_label": "fixture",
        "server_url": URL,
        "authorization": TOKEN,
        "headers": {"X-Team": "blue"},
        **extra,
    }


def mcp_call_of(name: str, arguments: dict[str, Any], call_id: str = "call_m1") -> ToolCall:
    return ToolCall(
        id=call_id,
        type="function",
        function=FunctionCall(name=name, arguments=json.dumps(arguments)),
    )


def ask(client: TestClient, *, tools: list[Any], **extra: Any) -> Any:
    return client.post(
        "/v1/responses",
        json={
            "model": MODEL,
            "input": extra.pop("input", "Say hi through the tool."),
            "tools": tools,
            **extra,
        },
    )


@pytest.fixture
def fake() -> FakeMcp:
    return FakeMcp()


# --------------------------------------------------------------------------- #
# The round trip
# --------------------------------------------------------------------------- #


def test_list_tools_and_a_call_round_trip(settings, fake) -> None:
    model = ScriptedDriver(
        turns=[Turn(calls=[mcp_call_of("echo", {"text": "hi"})]), Turn("The tool said hi.")]
    )
    with TestClient(app_with(settings, model, searches=[mcp_account(fake)])) as client:
        response = ask(client, tools=[mcp_tool(require_approval="never")])
    assert response.status_code == 200, response.text
    output = response.json()["output"]
    assert [i["type"] for i in output] == ["mcp_list_tools", "mcp_call", "message"]
    listed, called, message = output
    assert listed["server_label"] == "fixture"
    assert [t["name"] for t in listed["tools"]] == ["echo", "delete_everything"]
    assert called["status"] == "completed" and called["output"] == "echo: hi"
    assert called["name"] == "echo" and json.loads(called["arguments"]) == {"text": "hi"}
    assert called["approval_request_id"] is None
    assert message["content"][0]["text"] == "The tool said hi."
    # The model was offered the tools as functions and read the result.
    offered = {t.function.name for t in model.calls[0].tools or []}
    assert {"echo", "delete_everything"} <= offered
    assert model.calls[1].messages[-1].role.value == "tool"
    assert model.calls[1].messages[-1].content == "echo: hi"
    # The server's headers went to the tool-driver, for the server.
    assert fake.called[0]["headers"] == {"X-Team": "blue", "Authorization": f"Bearer {TOKEN}"}
    assert fake.called[0]["reach"] == "internet" and fake.called[0]["url"] == URL


def test_the_stream_says_each_step_as_openai_does(settings, fake) -> None:
    model = ScriptedDriver(
        turns=[Turn(calls=[mcp_call_of("echo", {"text": "hi"})]), Turn("Done now.")]
    )
    with TestClient(app_with(settings, model, searches=[mcp_account(fake)])) as client:
        response = ask(client, tools=[mcp_tool(require_approval="never")], stream=True)
    assert response.status_code == 200, response.text
    names = [name for name, _ in events(response)]
    for wanted in (
        "response.mcp_list_tools.in_progress",
        "response.mcp_list_tools.completed",
        "response.mcp_call.in_progress",
        "response.mcp_call_arguments.done",
        "response.mcp_call.completed",
    ):
        assert wanted in names, wanted
    assert names.index("response.mcp_call.completed") < names.index("response.output_text.delta")
    final = next(data for name, data in events(response) if name == "response.completed")
    assert [i["type"] for i in final["response"]["output"]] == [
        "mcp_list_tools",
        "mcp_call",
        "message",
    ]
    assert TOKEN not in response.text


def test_a_listing_in_input_is_used_instead_of_listing_again(settings, fake) -> None:
    model = ScriptedDriver(turns=[Turn("No call needed.")])
    earlier = {
        "type": "mcp_list_tools",
        "id": "mcpl_1",
        "server_label": "fixture",
        "tools": [{"name": "echo", "input_schema": {"type": "object"}}],
    }
    with TestClient(app_with(settings, model, searches=[mcp_account(fake)])) as client:
        response = ask(
            client,
            tools=[mcp_tool(require_approval="never")],
            input=[earlier, {"role": "user", "content": "hi"}],
        )
    assert response.status_code == 200, response.text
    assert fake.listed == []
    assert [i["type"] for i in response.json()["output"]] == ["message"]
    assert [t.function.name for t in model.calls[0].tools or []] == ["echo"]


def test_a_call_that_fails_is_told_to_the_model(settings, fake) -> None:
    fake.call_error = ToolDriverError(
        driver="brave",
        status=502,
        detail="The MCP server at https://mcp.example.com answered HTTP 500",
    )
    model = ScriptedDriver(
        turns=[Turn(calls=[mcp_call_of("echo", {"text": "hi"})]), Turn("It failed.")]
    )
    with TestClient(app_with(settings, model, searches=[mcp_account(fake)])) as client:
        response = ask(client, tools=[mcp_tool(require_approval="never")])
    called = response.json()["output"][1]
    assert called["status"] == "failed" and "HTTP 500" in called["error"]
    assert called["output"] is None
    assert "failed" in model.calls[1].messages[-1].content


# --------------------------------------------------------------------------- #
# Approvals
# --------------------------------------------------------------------------- #


def test_approval_requests_wait_for_a_response(settings, fake) -> None:
    """Default `require_approval` is `always`: the call is not run, the
    response asks; the answer in the next request runs it, once."""
    first_model = ScriptedDriver(
        turns=[Turn("Let me check.", calls=[mcp_call_of("echo", {"text": "hi"})])]
    )
    with TestClient(app_with(settings, first_model, searches=[mcp_account(fake)])) as client:
        first = ask(client, tools=[mcp_tool()])
    assert first.status_code == 200, first.text
    output = first.json()["output"]
    assert [i["type"] for i in output] == ["mcp_list_tools", "message", "mcp_approval_request"]
    asked = output[-1]
    assert asked["name"] == "echo" and asked["server_label"] == "fixture"
    assert json.loads(asked["arguments"]) == {"text": "hi"}
    assert fake.called == [], "nothing runs before the person answers"
    assert len(first_model.calls) == 1

    history = [{"role": "user", "content": "Say hi through the tool."}, *output]
    answer = {"type": "mcp_approval_response", "approval_request_id": asked["id"], "approve": True}
    second_model = ScriptedDriver(turns=[Turn("The tool said hi.")])
    with TestClient(app_with(settings, second_model, searches=[mcp_account(fake)])) as client:
        second = ask(client, tools=[mcp_tool()], input=[*history, answer])
    assert second.status_code == 200, second.text
    items = second.json()["output"]
    assert [i["type"] for i in items] == ["mcp_call", "message"]
    assert items[0]["approval_request_id"] == asked["id"]
    assert items[0]["output"] == "echo: hi"
    assert len(fake.called) == 1
    sent = second_model.calls[0].messages
    assert sent[-2].role.value == "assistant" and sent[-2].toolCalls[0]["id"] == asked["id"]
    assert sent[-1].role.value == "tool" and sent[-1].content == "echo: hi"

    # Handed back once more with the call that ran: never run again.
    third_model = ScriptedDriver(turns=[Turn("Still hi.")])
    with TestClient(app_with(settings, third_model, searches=[mcp_account(fake)])) as client:
        third = ask(
            client,
            tools=[mcp_tool()],
            input=[*history, answer, *items, {"role": "user", "content": "again?"}],
        )
    assert third.status_code == 200, third.text
    assert len(fake.called) == 1
    tool_messages = [m for m in third_model.calls[0].messages if m.role.value == "tool"]
    assert [m.content for m in tool_messages] == ["echo: hi"]


def test_an_approval_for_an_id_no_request_carries_is_refused(settings, fake) -> None:
    model = ScriptedDriver(turns=[Turn("never")])
    with TestClient(app_with(settings, model, searches=[mcp_account(fake)])) as client:
        response = ask(
            client,
            tools=[mcp_tool()],
            input=[
                {"role": "user", "content": "hi"},
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": "mcpr_forged",
                    "approve": True,
                },
            ],
        )
    assert response.status_code == 400, response.text
    assert "mcpr_forged" in response.json()["error"]["message"]
    assert fake.called == [] and fake.listed == [] and model.calls == []


def test_an_approval_answered_twice_is_refused(settings, fake) -> None:
    asked = {
        "type": "mcp_approval_request",
        "id": "mcpr_1",
        "server_label": "fixture",
        "name": "echo",
        "arguments": "{}",
    }
    answer = {"type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": True}
    with TestClient(
        app_with(settings, ScriptedDriver(turns=[]), searches=[mcp_account(fake)])
    ) as c:
        response = ask(c, tools=[mcp_tool()], input=[asked, answer, answer])
    assert response.status_code == 400 and "twice" in response.json()["error"]["message"]
    assert fake.called == []


def test_an_approval_for_a_server_this_request_does_not_name_is_refused(settings, fake) -> None:
    asked = {
        "type": "mcp_approval_request",
        "id": "mcpr_1",
        "server_label": "elsewhere",
        "name": "echo",
        "arguments": "{}",
    }
    answer = {"type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": True}
    with TestClient(
        app_with(settings, ScriptedDriver(turns=[]), searches=[mcp_account(fake)])
    ) as c:
        response = ask(c, tools=[mcp_tool()], input=[asked, answer])
    assert response.status_code == 400 and "elsewhere" in response.json()["error"]["message"]
    assert fake.called == []


def test_a_declined_call_is_told_to_the_model_and_not_run(settings, fake) -> None:
    asked = {
        "type": "mcp_approval_request",
        "id": "mcpr_1",
        "server_label": "fixture",
        "name": "delete_everything",
        "arguments": "{}",
    }
    answer = {
        "type": "mcp_approval_response",
        "approval_request_id": "mcpr_1",
        "approve": False,
        "reason": "too risky",
    }
    model = ScriptedDriver(turns=[Turn("Understood.")])
    with TestClient(app_with(settings, model, searches=[mcp_account(fake)])) as client:
        response = ask(client, tools=[mcp_tool()], input=[asked, answer])
    assert response.status_code == 200, response.text
    assert fake.called == []
    told = model.calls[0].messages[-1]
    assert told.role.value == "tool" and "declined" in told.content and "too risky" in told.content


def test_the_approval_filter_lets_named_tools_run(settings, fake) -> None:
    rule = {"never": {"tool_names": ["echo"]}}
    model = ScriptedDriver(
        turns=[
            Turn(calls=[mcp_call_of("echo", {"text": "a"}, "c1")]),
            Turn(calls=[mcp_call_of("delete_everything", {}, "c2")]),
        ]
    )
    with TestClient(app_with(settings, model, searches=[mcp_account(fake)])) as client:
        response = ask(client, tools=[mcp_tool(require_approval=rule)])
    kinds = [i["type"] for i in response.json()["output"]]
    assert kinds == ["mcp_list_tools", "mcp_call", "mcp_approval_request"]
    assert [c["name"] for c in fake.called] == ["echo"]


def test_approval_rules_read_as_documented() -> None:
    always_only = remote_mcp.approval_rule({"always": {"tool_names": ["x"]}}, "r")
    assert always_only.needs("x", read_only=False) and not always_only.needs("y", read_only=False)
    never_only = remote_mcp.approval_rule({"never": {"tool_names": ["x"]}}, "r")
    assert not never_only.needs("x", read_only=False) and never_only.needs("y", read_only=False)
    assert remote_mcp.approval_rule(None, "r").needs("x", read_only=True)
    assert not remote_mcp.approval_rule("never", "r").needs("x", read_only=False)
    read_only = remote_mcp.approval_rule({"never": {"read_only": True}}, "r")
    assert not read_only.needs("x", read_only=True) and read_only.needs("x", read_only=False)


# --------------------------------------------------------------------------- #
# When it cannot run
# --------------------------------------------------------------------------- #


def test_refused_when_off_and_named(settings, fake) -> None:
    with TestClient(
        app_with(settings, ScriptedDriver(turns=[]), searches=[mcp_account(fake)])
    ) as c:
        assert c.patch("/v1/config", json={"remoteMcp": "off"}).status_code == 200
        response = ask(c, tools=[mcp_tool()])
    assert response.status_code == 400
    assert "turned off" in response.json()["error"]["message"]
    assert fake.listed == []


def test_the_expert_reach_is_passed_to_the_tool_driver(settings, fake) -> None:
    model = ScriptedDriver(turns=[Turn("ok")])
    with TestClient(app_with(settings, model, searches=[mcp_account(fake)])) as c:
        assert c.patch("/v1/config", json={"remoteMcp": "network"}).status_code == 200
        schema = c.get("/v1/config/schema").json()
        assert ask(c, tools=[mcp_tool()]).status_code == 200
    assert fake.listed[0]["reach"] == "network"
    field = next(f for f in schema["fields"] if f["key"] == "remoteMcp")
    assert field["status"]["level"] == "warning"
    assert field["enumLabels"] == ["Off", "Internet only", "Internet and this network (expert)"]


def test_the_setting_defaults_to_the_internet_only() -> None:
    from eugene_plexus_gateway.config import as_schema

    field = next(f for f in as_schema(values={}).fields if f.key == "remoteMcp")
    assert field.default == "internet" and field.status is None


def test_without_a_process_that_runs_it_the_request_is_refused(settings) -> None:
    with TestClient(app_with(settings, ScriptedDriver(turns=[]))) as client:
        response = ask(client, tools=[mcp_tool()])
    assert response.status_code == 400
    assert "Add a search account" in response.json()["error"]["message"]


def test_a_local_only_key_and_a_key_denied_mcp_are_refused(fake) -> None:
    from .conftest import make_routing_table

    table = make_routing_table(ScriptedDriver(turns=[]))
    table._snapshot.tools = [mcp_account(fake)]
    for limits, words in (
        ({"localOnly": True}, "local-only"),
        ({"allowedTools": ["web_search"]}, "tool scope"),
    ):
        context = admission.ClientRequest({})
        context.access = {"limits": limits}
        reason = remote_mcp.why_not(table, context, "internet")
        assert reason is not None and words in reason
    allowed = admission.ClientRequest({})
    allowed.access = {"limits": {"allowedTools": ["mcp"]}}
    assert remote_mcp.why_not(table, allowed, "internet") is None


def test_an_address_the_rule_refuses_is_a_400_naming_it(settings, fake) -> None:
    fake.list_error = ToolDriverError(
        driver="brave",
        status=403,
        detail="127.0.0.1 is this machine's loopback, and a remote MCP server is never reached there",
        code="address_refused",
    )
    with TestClient(
        app_with(settings, ScriptedDriver(turns=[]), searches=[mcp_account(fake)])
    ) as c:
        response = ask(c, tools=[mcp_tool(server_url="https://127.0.0.1/mcp")])
    assert response.status_code == 400
    error = response.json()["error"]
    assert "loopback" in error["message"] and error["param"] == "tools"


def test_a_listing_that_fails_is_a_424(settings, fake) -> None:
    fake.list_error = ToolDriverError(
        driver="brave",
        status=424,
        detail="The MCP server at https://mcp.example.com refused the request's credentials (HTTP 401)",
        code="server_refused_credentials",
    )
    with TestClient(
        app_with(settings, ScriptedDriver(turns=[]), searches=[mcp_account(fake)])
    ) as c:
        response = ask(c, tools=[mcp_tool()])
    assert response.status_code == 424
    assert "fixture" in response.json()["error"]["message"]
    assert TOKEN not in response.text


def test_a_connector_is_refused(settings, fake) -> None:
    with TestClient(
        app_with(settings, ScriptedDriver(turns=[]), searches=[mcp_account(fake)])
    ) as c:
        response = ask(
            c, tools=[{"type": "mcp", "server_label": "gmail", "connector_id": "connector_gmail"}]
        )
    assert response.status_code == 400 and "connector" in response.json()["error"]["message"]


# --------------------------------------------------------------------------- #
# The caller's credential
# --------------------------------------------------------------------------- #


def test_headers_never_reach_metrics_or_logs(settings, fake, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    model = ScriptedDriver(
        turns=[Turn(calls=[mcp_call_of("echo", {"text": "hi"})]), Turn("The tool said hi.")]
    )
    app = app_with(settings, model, searches=[mcp_account(fake)])
    with TestClient(app) as client:
        response = ask(client, tools=[mcp_tool(require_approval="never")])
        assert response.status_code == 200, response.text
        rows: list[Any] = []
        for _ in range(100):
            rows = client.get("/v1/metrics/requests").json()["requests"]
            if rows:
                break
            time.sleep(0.05)
    assert fake.called[0]["headers"]["Authorization"] == f"Bearer {TOKEN}", (
        "it did reach the server"
    )
    body = response.json()
    echoed = body["tools"][0]
    assert "headers" not in echoed and "authorization" not in echoed
    assert TOKEN not in response.text and "blue" not in response.text
    assert TOKEN not in caplog.text
    row = rows[0]
    assert row["mcpCalls"] == [
        {
            "tool": "mcp",
            "driver": "brave",
            "node": None,
            "provider": None,
            "version": None,
            "outcome": "ok",
            "results": None,
            "elapsedMs": row["mcpCalls"][0]["elapsedMs"],
        }
    ]
    assert not row.get("webSearches")
    raw = Path(settings.metrics_file).read_bytes()
    assert TOKEN.encode() not in raw and b"echo" not in raw


def test_the_stream_echoes_no_credential(settings, fake) -> None:
    model = ScriptedDriver(turns=[Turn("hi")])
    with TestClient(app_with(settings, model, searches=[mcp_account(fake)])) as client:
        response = ask(client, tools=[mcp_tool()], stream=True)
    assert TOKEN not in response.text and '"X-Team"' not in response.text


# --------------------------------------------------------------------------- #
# Names and limits
# --------------------------------------------------------------------------- #


def test_the_limit_of_calls_names_itself(settings, fake) -> None:
    calls = [Turn(calls=[mcp_call_of("echo", {"text": str(i)}, f"c{i}")]) for i in range(3)]
    model = ScriptedDriver(turns=[*calls, Turn("done")])
    with TestClient(app_with(settings, model, searches=[mcp_account(fake)])) as client:
        response = ask(client, tools=[mcp_tool(require_approval="never")], max_tool_calls=2)
    items = response.json()["output"]
    called = [i for i in items if i["type"] == "mcp_call"]
    assert [c["status"] for c in called] == ["completed", "completed", "failed"]
    assert "limit" in called[2]["error"]
    assert len(fake.called) == 2
    assert "limit is 2" in model.calls[3].messages[-1].content


def test_tool_names_never_collide_and_stay_valid() -> None:
    taken = {"web_search", "image_generation", "echo"}
    assert remote_mcp.function_name("fixture", "echo", multi=False, taken=taken) != "echo"
    assert remote_mcp.function_name("fixture", "read", multi=False, taken=taken) == "read"
    assert remote_mcp.function_name("fixture", "read", multi=True, taken=taken) == "fixture__read"
    odd = remote_mcp.function_name("fixture", "a.b c", multi=False, taken=taken)
    assert odd.startswith("a_b_c_") and len(odd) <= 64
    assert remote_mcp.function_name("fixture", "a.b c", multi=False, taken=taken) == odd
    assert remote_mcp.function_name("f", "x" * 200, multi=True, taken=taken).__len__() <= 64


def test_two_servers_are_offered_by_label(settings) -> None:
    one, two = FakeMcp(), FakeMcp(tools=[TOOLS[0]])
    model = ScriptedDriver(turns=[Turn("ok")])
    table_accounts = [mcp_account(one)]
    with TestClient(app_with(settings, model, searches=table_accounts)) as client:
        response = ask(
            client,
            tools=[
                mcp_tool(),
                mcp_tool(server_label="second", server_url="https://other.example/mcp"),
            ],
        )
    assert response.status_code == 200, response.text
    names = {t.function.name for t in model.calls[0].tools or []}
    assert {"fixture__echo", "second__echo"} <= names
    assert two.listed == []  # One process runs both servers' calls.
    assert [entry["url"] for entry in one.listed] == [URL, "https://other.example/mcp"]
