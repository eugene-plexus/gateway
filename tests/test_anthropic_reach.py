"""MR9a: `search_result` blocks and the `memory_20250818` tool on the Anthropic door.

Shapes are Anthropic's documented ones (platform.claude.com, read 2026-10-18):
a search result is `source`, `title` and text `content`, numbered across the
whole request; the memory tool is client-run, its call returned as `tool_use`.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway import anthropic
from eugene_plexus_gateway._generated.driver_models import FunctionCall, ToolCall
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

MODEL = "qwen3-coder"


def _result(n: int, *, citations: bool | None = None, blocks: int = 1) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "search_result",
        "source": f"kb://doc-{n}",
        "title": f"Doc {n}",
        "content": [{"type": "text", "text": f"fact {n}.{i}"} for i in range(blocks)],
    }
    if citations is not None:
        block["citations"] = {"enabled": citations}
    return block


def _request(messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"model": MODEL, "max_tokens": 64, "messages": messages, **extra}


def _texts(request: dict[str, Any]) -> list[str]:
    translated = anthropic.translate_request(request)
    return [m.content for m in translated.messages if isinstance(m.content, str)]


def test_a_top_level_search_result_reaches_the_model_numbered() -> None:
    texts = _texts(
        _request(
            [
                {
                    "role": "user",
                    "content": [_result(1, blocks=2), {"type": "text", "text": "Which?"}],
                }
            ]
        )
    )
    assert texts == [
        "[Search result 1: Doc 1 (kb://doc-1)]\nfact 1.0\n\nfact 1.1\n"
        "[End of search result 1]\n\nWhich?"
    ]


def test_numbers_run_across_messages_and_tool_results_in_request_order() -> None:
    """Anthropic's `search_result_index` counts every block in the request in
    order, top level and tool results alike; the model's numbers match."""
    request = _request(
        [
            {"role": "user", "content": [_result(1), {"type": "text", "text": "look it up"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "kb", "input": {"q": "x"}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [_result(2), {"type": "text", "text": "two hits"}, _result(3)],
                    }
                ],
            },
        ]
    )
    translated = anthropic.translate_request(request)
    tool = next(m for m in translated.messages if m.role.value == "tool")
    assert tool.tool_call_id == "t1"
    assert "[Search result 2: Doc 2 (kb://doc-2)]\nfact 2.0\n[End of search result 2]" in (
        tool.content or ""
    )
    assert "two hits" in (tool.content or "")
    assert "[Search result 3: Doc 3" in (tool.content or "")
    first = translated.messages[0].content
    assert isinstance(first, str) and first.startswith("[Search result 1: Doc 1")


@pytest.mark.parametrize(
    ("block", "words"),
    [
        (
            {"type": "search_result", "title": "t", "content": [{"type": "text", "text": "x"}]},
            "source",
        ),
        (
            {"type": "search_result", "source": "s", "content": [{"type": "text", "text": "x"}]},
            "title",
        ),
        ({"type": "search_result", "source": "s", "title": "t", "content": []}, "content"),
        (
            {
                "type": "search_result",
                "source": "s",
                "title": "t",
                "content": [{"type": "image", "source": {}}],
            },
            "text blocks only",
        ),
    ],
)
def test_a_malformed_search_result_is_refused(block: dict[str, Any], words: str) -> None:
    with pytest.raises(anthropic.Refusal) as e:
        anthropic.translate_request(_request([{"role": "user", "content": [block]}]))
    assert words in str(e.value)
    assert "messages.0.content.0" in str(e.value)


def test_a_search_result_on_an_assistant_turn_is_refused() -> None:
    with pytest.raises(anthropic.Refusal) as e:
        anthropic.translate_request(
            _request(
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [_result(1)]},
                ]
            )
        )
    assert "messages.1.content.0" in str(e.value)
    assert "assistant turn" in str(e.value)


def test_mixed_citation_settings_are_refused_as_upstream_refuses_them() -> None:
    with pytest.raises(anthropic.Refusal) as e:
        anthropic.translate_request(
            _request([{"role": "user", "content": [_result(1, citations=True), _result(2)]}])
        )
    assert "citations" in str(e.value)


def test_enabled_citations_are_named_ignored_not_faked() -> None:
    raw = _request(
        [{"role": "user", "content": [_result(1, citations=True), _result(2, citations=True)]}]
    )
    anthropic.translate_request(raw)
    assert "citations" in anthropic.compatibility_headers(raw)[
        "x-eugene-plexus-ignored-settings"
    ].split(", ")
    plain = _request([{"role": "user", "content": [_result(1, citations=False)]}])
    assert "citations" not in anthropic.compatibility_headers(plain).get(
        "x-eugene-plexus-ignored-settings", ""
    )


MEMORY = {"type": "memory_20250818", "name": "memory"}


def test_the_memory_tool_becomes_a_function_and_the_protocol_follows_the_system() -> None:
    translated = anthropic.translate_request(
        _request([{"role": "user", "content": "hi"}], system="Be brief.", tools=[MEMORY])
    )
    assert translated.messages[0].content == f"Be brief.\n\n{anthropic.MEMORY_PROTOCOL}"
    assert translated.tools is not None and len(translated.tools) == 1
    function = translated.tools[0].function
    assert function.name == "memory"
    params = function.parameters or {}
    assert params["required"] == ["command"]
    assert params["properties"]["command"]["enum"] == [
        "view",
        "create",
        "str_replace",
        "insert",
        "delete",
        "rename",
    ]


def test_without_a_system_prompt_the_protocol_is_the_system() -> None:
    translated = anthropic.translate_request(
        _request([{"role": "user", "content": "hi"}], tools=[MEMORY])
    )
    assert translated.messages[0].content == anthropic.MEMORY_PROTOCOL


def test_without_the_memory_tool_there_is_no_protocol() -> None:
    translated = anthropic.translate_request(
        _request([{"role": "user", "content": "hi"}], system="Be brief.")
    )
    assert translated.messages[0].content == "Be brief."


def test_the_memory_tool_must_be_named_memory_and_may_not_share_the_name() -> None:
    with pytest.raises(anthropic.Refusal, match="must be named 'memory'"):
        anthropic.translate_request(
            _request(
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "memory_20250818", "name": "notes"}],
            )
        )
    with pytest.raises(anthropic.Refusal, match="two tools are named 'memory'"):
        anthropic.translate_request(
            _request(
                [{"role": "user", "content": "hi"}],
                tools=[MEMORY, {"name": "memory", "input_schema": {"type": "object"}}],
            )
        )


def test_naming_the_memory_tool_in_tool_choice_works() -> None:
    translated = anthropic.translate_request(
        _request(
            [{"role": "user", "content": "hi"}],
            tools=[MEMORY],
            tool_choice={"type": "tool", "name": "memory"},
        )
    )
    assert translated.tool_choice.function.name == "memory"  # type: ignore[union-attr]


def test_count_tokens_counts_the_expansion() -> None:
    translated = anthropic.translate_count_request(
        {"model": MODEL, "messages": [{"role": "user", "content": [_result(1)]}], "tools": [MEMORY]}
    )
    assert anthropic.MEMORY_PROTOCOL in str(translated.messages[0].content)
    assert "[Search result 1" in str(translated.messages[1].content)


def test_the_models_memory_call_goes_back_as_tool_use(settings: Settings) -> None:
    """Through the real route: the function and protocol reach the driver,
    and the backend's call comes back as the `tool_use` Anthropic returns."""
    call = ToolCall(
        id="call_1",
        type="function",
        function=FunctionCall(name="memory", arguments='{"command": "view", "path": "/memories"}'),
    )
    fake = FakeDriverClient(name="d1", model_id=MODEL, supports_tools=True)
    fake.tool_calls = [call]
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake)
    with TestClient(app) as client:
        r = client.post(
            "/v1/messages",
            json=_request([{"role": "user", "content": "What do you remember?"}], tools=[MEMORY]),
        )
    assert r.status_code == 200, r.text
    sent = fake.calls[0]
    assert [t.function.name for t in sent.tools or []] == ["memory"]
    assert anthropic.MEMORY_PROTOCOL in str(sent.messages[0].content)
    out = r.json()
    assert out["stop_reason"] == "tool_use"
    use = next(b for b in out["content"] if b["type"] == "tool_use")
    assert use["name"] == "memory"
    assert use["input"] == {"command": "view", "path": "/memories"}
