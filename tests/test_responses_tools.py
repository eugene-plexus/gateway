"""Row 57: client-run Responses tools, over both public response paths."""

import json

import pytest

from eugene_plexus_gateway._generated.driver_models import FunctionCall, ToolCall
from eugene_plexus_gateway.settings import Settings

from .test_responses import MODEL, driver, sent, serve, stream

PATCH = "*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch"
CASES = [
    ({"type": "custom", "name": "write"}, {"input": "hello\n世界"}, "custom_tool_call"),
    (
        {"type": "local_shell"},
        {"command": ["echo", "hello"], "workdir": "/tmp", "timeout_ms": 500, "env": {"A": "b"}},
        "local_shell_call",
    ),
    (
        {"type": "shell", "environment": {"type": "local"}},
        {"commands": ["echo hello", "pwd"], "timeout_ms": 500, "max_output_length": 100},
        "shell_call",
    ),
    ({"type": "apply_patch"}, {"input": PATCH}, "apply_patch_call"),
]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "definition,arguments,kind", CASES, ids=["custom", "local_shell", "shell", "apply_patch"]
)
def test_client_tool_round_trip(settings: Settings, definition, arguments, kind, streaming):
    fake = driver()
    name = definition.get("name", definition["type"])
    fake.tool_calls = [
        ToolCall(
            id="call_57",
            type="function",
            function=FunctionCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)),
        )
    ]
    body = {"model": MODEL, "input": "do it", "tools": [definition], "stream": streaming}
    with serve(settings, fake) as client:
        if streaming:
            response, frames = stream(client, body)
            assert response.status_code == 200, response.text
            output = frames[-1][1]["response"]["output"]
            names = [n for n, _ in frames]
            assert "response.output_item.added" in names
            assert "response.output_item.done" in names
            assert not any("function_call_arguments" in n for n in names)
            if kind == "custom_tool_call":
                assert "response.custom_tool_call_input.done" in names
                assert (
                    "".join(
                        d["delta"]
                        for n, d in frames
                        if n == "response.custom_tool_call_input.delta"
                    )
                    == arguments["input"]
                )
            if kind == "shell_call":
                assert [
                    d["command"] for n, d in frames if n == "response.shell_call_command.done"
                ] == arguments["commands"]
            assert [d["sequence_number"] for _, d in frames] == list(range(len(frames)))
        else:
            response = client.post("/v1/responses", json=body)
            assert response.status_code == 200, response.text
            output = response.json()["output"]
        [call] = output
        assert call["type"] == kind and call["call_id"] == "call_57"
        if kind == "custom_tool_call":
            assert call["input"] == arguments["input"] and call["name"] == name
        elif kind == "local_shell_call":
            assert call["action"] == {
                "type": "exec",
                "command": ["echo", "hello"],
                "working_directory": "/tmp",
                "timeout_ms": 500,
                "env": {"A": "b"},
            }
        elif kind == "shell_call":
            assert call["action"] == arguments
        else:
            assert call["operation"] == {
                "type": "create_file",
                "path": "hello.txt",
                "diff": "+hello",
            }
        offered = fake.calls[-1].tools[0].function
        assert offered.name == name
        assert set(offered.parameters["properties"]) == set(arguments)
        fake.tool_calls = None
        result = {"type": kind + "_output", "call_id": "call_57", "output": "done"}
        if kind == "custom_tool_call":
            result["type"] = "custom_tool_call_output"
        if kind == "shell_call":
            result["output"] = [
                {"stdout": "done", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}
            ]
        if kind == "apply_patch_call":
            result["status"] = "failed"
            result["output"] = "file is read-only"
        answer = client.post(
            "/v1/responses",
            json={
                **body,
                "stream": False,
                "input": [{"role": "user", "content": "do it"}, call, result],
            },
        )
        assert answer.status_code == 200, answer.text
        history = sent(fake)
        assert json.loads(history[-2]["toolCalls"][0]["function"]["arguments"]) == arguments
        assert history[-1]["role"] == "tool" and history[-1]["toolCallId"] == "call_57"
        assert ("read-only" if kind == "apply_patch_call" else "done") in history[-1]["content"]


def test_a_grammar_is_named_as_ignored(settings: Settings):
    fake = driver()
    grammar = {"type": "grammar", "syntax": "lark", "definition": 'start: "hello"'}
    with serve(settings, fake) as client:
        result = client.post(
            "/v1/responses",
            json={
                "model": MODEL,
                "input": "go",
                "tools": [{"type": "custom", "name": "write", "format": grammar}],
            },
        )
    assert result.status_code == 200, result.text
    assert "tools[0].format" in result.headers["x-eugene-plexus-ignored-settings"]
    assert grammar["definition"] in fake.calls[-1].tools[0].function.description


@pytest.mark.parametrize("definition,_,kind", CASES)
def test_tool_choice_forces_a_declared_client_tool(settings: Settings, definition, _, kind):
    fake = driver()
    with serve(settings, fake) as client:
        result = client.post(
            "/v1/responses",
            json={
                "model": MODEL,
                "input": "go",
                "tools": [definition],
                "tool_choice": {
                    "type": definition["type"],
                    **({"name": definition["name"]} if "name" in definition else {}),
                },
            },
        )
    assert result.status_code == 200, result.text
    assert fake.calls[-1].toolChoice.function.name == definition.get("name", definition["type"])


@pytest.mark.parametrize(
    "kind,cause",
    [
        ("computer_use_preview", "no local model"),
        ("code_interpreter", "no sandbox"),
        ("mcp", "server_label: required"),
        ("file_search", "row 56"),
    ],
)
def test_refusals_name_the_cause(settings: Settings, kind, cause):
    with serve(settings, driver()) as client:
        result = client.post(
            "/v1/responses", json={"model": MODEL, "input": "go", "tools": [{"type": kind}]}
        )
    assert result.status_code == 400
    assert cause in result.json()["error"]["message"]


def test_hosted_shell_is_refused(settings: Settings):
    with serve(settings, driver()) as client:
        result = client.post(
            "/v1/responses",
            json={
                "model": MODEL,
                "input": "go",
                "tools": [{"type": "shell", "environment": {"type": "container_auto"}}],
            },
        )
    assert result.status_code == 400
    assert "shell execution service" in result.json()["error"]["message"]


def test_legacy_local_shell_output_id(settings: Settings):
    fake = driver()
    with serve(settings, fake) as client:
        result = client.post(
            "/v1/responses",
            json={
                "model": MODEL,
                "input": [
                    {"type": "local_shell_call_output", "id": "call_old", "output": "legacy"}
                ],
            },
        )
    assert result.status_code == 200, result.text
    assert sent(fake)[-1]["toolCallId"] == "call_old"


@pytest.mark.parametrize("streaming", [False, True])
def test_malformed_backend_arguments_name_the_cause(settings: Settings, streaming):
    fake = driver()
    fake.tool_calls = [
        ToolCall(
            id="call_bad",
            type="function",
            function=FunctionCall(name="write", arguments='{"input":'),
        )
    ]
    body = {
        "model": MODEL,
        "input": "go",
        "tools": [{"type": "custom", "name": "write"}],
        "stream": streaming,
    }
    with serve(settings, fake) as client:
        if streaming:
            _, frames = stream(client, body)
            assert frames[-1][0] == "response.failed"
            assert "invalid JSON" in frames[-1][1]["response"]["error"]["message"]
        else:
            result = client.post("/v1/responses", json=body)
            assert result.status_code == 502
            assert "invalid JSON" in result.json()["error"]["message"]


def test_an_interrupted_custom_call_still_has_a_failed_event(settings: Settings):
    from eugene_plexus_gateway import responses

    translator = responses.StreamTranslator(
        "resp_57", MODEL, echo={}, client_tools={"write": "custom"}
    )
    translator.tool_fragments(
        [
            {
                "index": 0,
                "id": "call_bad",
                "function": {"name": "write", "arguments": '{"input": "half'},
            }
        ]
    )
    frames = translator.failed("The backend disconnected.", code="server_error")
    assert json.loads(frames[-1].split("data: ")[1])["type"] == "response.failed"


@pytest.mark.parametrize(
    "definition",
    [
        {"type": "custom", "name": "write", "unexpected_setting": 1},
        {"type": "shell", "environment": {"type": "local", "skills": ["something"]}},
    ],
)
def test_unsupported_tool_settings_are_named(settings: Settings, definition):
    with serve(settings, driver()) as client:
        result = client.post(
            "/v1/responses", json={"model": MODEL, "input": "go", "tools": [definition]}
        )
    assert result.status_code == 400
    assert (
        "unexpected_setting" if "unexpected_setting" in definition else "skills"
    ) in result.json()["error"]["message"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("tools", [{"type": []}]),
        ("tool_choice", {"type": []}),
        ("input", [{"type": [], "content": "go"}]),
    ],
)
def test_malformed_tool_types_are_refused(settings: Settings, field, value):
    with serve(settings, driver()) as client:
        result = client.post("/v1/responses", json={"model": MODEL, "input": "go", field: value})
    assert result.status_code == 400
    assert field in result.json()["error"]["message"]
