"""P8: web search run by this install, through all three doors.

A scripted model (it calls `web_search`, then answers citing a result)
and a fake search account stand in for a real engine and a real SearXNG;
the acceptance run in `specs/scripts/p8-search-acceptance.py` drives the
real processes. What is pinned here:

- the loop offers the model a `web_search` function, runs each call on a
  search account, hands the results back and asks again;
- **failover ends at the first search** (and the first call still
  cascades as any request does -- the positive twin);
- a turn asking for a search and a caller's function has the function
  dropped and is asked again;
- a search that fails is told to the model, not turned into a failed
  request; a second account is tried only when the first could not do
  the job;
- a backend that searches itself is sent the request natively;
- each door's own record: annotations and `web_searches` on chat,
  `web_search_call` items on Responses, `server_tool_use` and
  `web_search_tool_result` blocks on messages;
- when a search cannot run, each door says why, in its own way;
- metrics keep one row per search and no query text, and the loop's
  turns are not counted as a cascade.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway import admission
from eugene_plexus_gateway._generated.driver_models import (
    FinishReason,
    FunctionCall,
    GenerateRequest,
    GenerateResponse,
    Problem,
    ToolCall,
    Usage,
)
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import DriverError, StreamEvent
from eugene_plexus_gateway.routing import ToolAccount
from eugene_plexus_gateway.server_tools import citations, query_of, tool_name_for
from eugene_plexus_gateway.settings import Settings
from eugene_plexus_gateway.tool_client import SearchAnswer, ToolDriverError, ToolDriverInfo

from .conftest import FakeDriverClient, make_routing_table

MODEL = "local-model"
RESULTS = [
    {
        "url": "https://github.com/eugene-plexus",
        "title": "Eugene Plexus on GitHub",
        "snippet": "A self-hosted control plane for local LLM inference.",
        "publishedAt": "2026-09-11",
    },
    {"url": "https://eugeneplexus.com/", "title": "Eugene Plexus", "snippet": "Home page."},
]
ANSWER = "It is a control plane. See [Eugene Plexus on GitHub](https://github.com/eugene-plexus)."


def search_call(query: str = "eugene plexus", call_id: str = "call_s1") -> ToolCall:
    return ToolCall(
        id=call_id,
        type="function",
        function=FunctionCall(name="web_search", arguments=json.dumps({"query": query})),
    )


def function_call(name: str = "read_file", call_id: str = "call_f1") -> ToolCall:
    return ToolCall(
        id=call_id,
        type="function",
        function=FunctionCall(name=name, arguments=json.dumps({"path": "a.txt"})),
    )


class Turn:
    """One scripted model turn: optional text, optional calls, or a failure."""

    def __init__(
        self,
        text: str | None = None,
        calls: list[ToolCall] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.text, self.calls, self.error = text, calls, error


class ScriptedDriver(FakeDriverClient):
    """A model that answers turn by turn from a script.

    The FakeDriverClient answers every call the same way; a loop needs a
    model whose second answer depends on being asked a second time.
    """

    def __init__(self, *, name: str = "local", turns: list[Turn], **kwargs: Any) -> None:
        super().__init__(name=name, model_id=MODEL, supports_tools=True, **kwargs)
        self.turns = list(turns)
        self.usage = Usage(promptTokens=10, completionTokens=5, totalTokens=15)

    def _next(self) -> Turn:
        return self.turns.pop(0) if self.turns else Turn("<no more script>")

    def _result(self, turn: Turn, request: GenerateRequest) -> GenerateResponse:
        return GenerateResponse(
            content=turn.text,
            toolCalls=turn.calls,
            finishReason=FinishReason.tool_calls if turn.calls else FinishReason.stop,
            backend=self.backend,
            modelId=request.model or self.model_id,
            usage=self.usage,
            latencyMs=1,
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        self.calls.append(request)
        turn = self._next()
        if turn.error is not None:
            raise turn.error
        return self._result(turn, request)

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamEvent, None]:
        self.calls.append(request)
        turn = self._next()
        if turn.error is not None:
            raise turn.error
        for word in (turn.text or "").split(" "):
            if word:
                yield StreamEvent(text=word + " ")
        for index, call in enumerate(turn.calls or []):
            args = call.function.arguments
            yield StreamEvent(
                tool_calls=[
                    {
                        "index": index,
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": args[:5]},
                    }
                ]
            )
            yield StreamEvent(tool_calls=[{"index": index, "function": {"arguments": args[5:]}}])
        yield StreamEvent(done=True, result=self._result(turn, request))


class FakeSearch:
    """A search account's client: answers, or fails the way a tool-driver does."""

    def __init__(
        self,
        results: list[dict[str, Any]] | None = None,
        *,
        error: ToolDriverError | None = None,
        suggestions: list[str | None] | None = None,
        answer: str | None = None,
    ) -> None:
        self.results = RESULTS if results is None else results
        self.error = error
        self.asked: list[Any] = []
        #: One per search, in order (None: that search's provider gave none).
        self.suggestions = list(suggestions or [])
        self.answer = answer

    async def web_search(self, request: Any) -> SearchAnswer:
        self.asked.append(request)
        if self.error is not None:
            raise self.error
        return SearchAnswer(
            results=self.results,
            answer=self.answer,
            provider="searxng",
            elapsed_ms=3,
            search_suggestions=self.suggestions.pop(0) if self.suggestions else None,
        )

    async def info(self) -> ToolDriverInfo:
        return ToolDriverInfo(
            provider="searxng", label="SearXNG", tools=["web_search"], configured=True
        )

    async def aclose(self) -> None:
        return None


def account(
    client: FakeSearch,
    name: str = "searx",
    *,
    configured: bool = True,
    node: str | None = None,
    provider: str = "searxng",
    label: str = "SearXNG",
    billing: str | None = "per_search",
) -> ToolAccount:
    return ToolAccount(
        node=node,
        name=name,
        url=f"http://{name}.invalid:8190",
        client=client,  # type: ignore[arg-type]
        info=ToolDriverInfo(
            provider=provider,
            label=label,
            tools=["web_search"] if configured else [],
            configured=configured,
            billing=billing,
        ),
    )


def app_with(
    settings: Settings,
    *drivers: FakeDriverClient,
    searches: list[ToolAccount] | None = None,
    slots: list[dict[str, Any]] | None = None,
) -> FastAPI:
    app = create_app(settings=settings)
    table = make_routing_table(*drivers, slots=slots)
    table._snapshot.tools = list(searches or [])
    app.state.routing = table
    return app


@pytest.fixture
def search() -> FakeSearch:
    return FakeSearch()


def chat(client: TestClient, **extra: Any) -> Any:
    return client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is Eugene Plexus?"}],
            "web_search_options": {},
            **extra,
        },
    )


def sse(response: Any) -> list[Any]:
    frames = []
    for line in response.text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            frames.append(json.loads(line[6:]))
    return frames


def events(response: Any) -> list[tuple[str, dict[str, Any]]]:
    out = []
    name = None
    for line in response.text.splitlines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: ") and name is not None:
            out.append((name, json.loads(line[6:])))
            name = None
    return out


# --------------------------------------------------------------------------- #
# The loop, on chat
# --------------------------------------------------------------------------- #


def test_a_local_model_searches_and_answers_with_citations(settings, search) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = chat(client)
    assert response.status_code == 200, response.text
    body = response.json()
    message = body["choices"][0]["message"]
    assert message["content"] == ANSWER
    assert body["choices"][0]["finish_reason"] == "stop"
    cited = message["annotations"]
    assert [a["url_citation"]["url"] for a in cited] == ["https://github.com/eugene-plexus"]
    span = cited[0]["url_citation"]
    assert ANSWER[span["start_index"] : span["end_index"]] == (
        "[Eugene Plexus on GitHub](https://github.com/eugene-plexus)"
    )
    assert body["x_eugene_plexus"]["web_searches"] == 1
    assert body["x_eugene_plexus"]["attempts"] == 1, "a second turn is not a second backend"
    # What the model was offered and told.
    first, second = model.calls
    offered = [t.function.name for t in first.tools or []]
    assert offered == ["web_search"]
    assert str(first.toolChoice) == "required", "search was the only tool: the first turn makes one"
    assert "webSearchOptions" not in (first.callerSettings or [])
    assert first.webSearchOptions is None
    told = second.messages[-1]
    assert told.role.value == "tool" and told.toolCallId == "call_s1"
    assert "https://github.com/eugene-plexus" in told.content
    assert "Cite the pages you use" in told.content
    assert second.messages[-2].toolCalls[0]["function"]["name"] == "web_search"
    assert [r.query for r in search.asked] == ["eugene plexus"]


def test_the_chat_stream_says_a_search_ran_only_to_a_caller_that_asked(settings, search) -> None:
    model = ScriptedDriver(
        turns=[Turn("Let me look.", calls=[search_call()]), Turn(ANSWER)],
    )
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = chat(
            client, stream=True, stream_options={"include_progress": True, "include_usage": True}
        )
    frames = sse(response)

    def progress_of(f: dict) -> dict | None:
        if f.get("choices"):
            return None
        return (f.get("x_eugene_plexus") or {}).get("progress")

    def text_of(f: dict) -> str:
        return "".join((c["delta"].get("content") or "") for c in f.get("choices") or [])

    progress = [p for f in frames if (p := progress_of(f))]
    assert progress == [
        {"stage": "tool", "tool": "web_search", "phase": "started"},
        {"stage": "tool", "tool": "web_search", "phase": "finished"},
    ]
    text = "".join(text_of(f) for f in frames)
    assert text.startswith("Let me look. \n\nIt is a control plane.")
    # The marks fall between the turns (gateway#4): what came before the
    # start was written before the search, and the answer after the finish.
    started = next(
        i for i, f in enumerate(frames) if (progress_of(f) or {}).get("phase") == "started"
    )
    finished = next(
        i for i, f in enumerate(frames) if (progress_of(f) or {}).get("phase") == "finished"
    )
    before = "".join(text_of(f) for f in frames[:started])
    after = "".join(text_of(f) for f in frames[finished:])
    assert before == "Let me look. "
    assert after.startswith("\n\nIt is a control plane.")
    assert not "".join(text_of(f) for f in frames[started:finished]), "no text while it searched"
    assert not any(c["delta"].get("tool_calls") for f in frames for c in f.get("choices") or []), (
        "the search call never reaches the caller"
    )
    annotations = [
        a
        for f in frames
        for c in f.get("choices") or []
        for a in c["delta"].get("annotations") or []
    ]
    assert annotations and annotations[0]["url_citation"]["url"] == RESULTS[0]["url"]
    final = [f for f in frames if (f.get("choices") or [{}])[0].get("finish_reason")]
    assert final[-1]["choices"][0]["finish_reason"] == "stop"
    assert final[-1]["x_eugene_plexus"]["web_searches"] == 1
    usage = [f for f in frames if f.get("usage") and not f.get("choices")]
    assert usage[-1]["usage"]["prompt_tokens"] == 20, "both turns' tokens are counted"


def test_a_chat_stream_without_progress_is_not_told_a_search_ran(settings, search) -> None:
    model = ScriptedDriver(
        turns=[Turn("Let me look.", calls=[search_call()]), Turn(ANSWER)],
    )
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = chat(client, stream=True, stream_options={"include_usage": True})
    frames = sse(response)
    assert not [f for f in frames if not f.get("choices") and "progress" in str(f)]
    text = "".join(
        (c["delta"].get("content") or "") for f in frames for c in f.get("choices") or []
    )
    assert text.startswith("Let me look. \n\nIt is a control plane.")


def test_a_turn_with_a_search_and_a_function_is_asked_again(settings, search) -> None:
    """The caller only ever sees function calls from a turn with no search pending."""
    model = ScriptedDriver(
        turns=[
            Turn(calls=[search_call(), function_call()]),
            Turn(calls=[function_call(call_id="call_f2")]),
        ]
    )
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = chat(
            client,
            tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        )
    body = response.json()
    calls = body["choices"][0]["message"]["tool_calls"]
    assert [c["id"] for c in calls] == ["call_f2"], "the dropped call is never handed back"
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    first = model.calls[0]
    assert str(first.toolChoice) != "required", "the caller offered a function: auto, not forced"
    assert [t.function.name for t in first.tools] == ["read_file", "web_search"]
    history = model.calls[1].messages
    assert [c["id"] for c in history[-2].toolCalls] == ["call_s1"]


def test_the_limit_ends_the_loop_and_the_model_is_told(settings, search) -> None:
    looping = [Turn(calls=[search_call(f"q{i}", f"call_{i}")]) for i in range(10)]
    model = ScriptedDriver(turns=looping)
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": MODEL,
                "input": "search forever",
                "tools": [{"type": "web_search"}],
                "max_tool_calls": 2,
            },
        )
    assert response.status_code == 200, response.text
    assert len(search.asked) == 2, "no more searches run than the limit"
    told = [m for m in model.calls[-1].messages if m.role.value == "tool"]
    assert "No more searches are available" in told[-1].content
    # Two searches, a third call answered "no", and a last turn with no tool.
    assert len(model.calls) == 4
    assert model.calls[-1].tools is None
    kinds = [i["type"] for i in response.json()["output"]]
    assert kinds.count("web_search_call") == 3
    statuses = [i["status"] for i in response.json()["output"] if i["type"] == "web_search_call"]
    assert statuses == ["completed", "completed", "failed"]


def test_a_search_asked_for_on_the_last_turn_never_reaches_the_caller(settings, search) -> None:
    """The last turn offers no search tool, and a model may call it anyway.
    That turn's calls are re-framed for the caller -- all but the search,
    which the caller never offered and cannot run."""
    looping = [Turn(calls=[search_call(f"q{i}", f"call_{i}")]) for i in range(10)]
    model = ScriptedDriver(turns=looping)
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": MODEL,
                "input": "search forever",
                "tools": [{"type": "web_search"}],
                "max_tool_calls": 1,
                "stream": True,
            },
        )
    assert response.status_code == 200, response.text
    assert len(model.calls) == 3, "one search, one refused, and the last turn"
    assert model.calls[-1].tools is None
    added = [d["item"] for name, d in events(response) if name == "response.output_item.added"]
    assert not [i for i in added if i["type"] == "function_call"], added
    assert not [n for n, _ in events(response) if n.startswith("response.function_call")]


# --------------------------------------------------------------------------- #
# Failover ends at the first search
# --------------------------------------------------------------------------- #


def test_after_a_search_a_failure_is_reported_and_never_cascaded(settings, search) -> None:
    """The second turn fails the way a dead backend does -- a failure the
    first call WOULD cascade on. A `RuntimeError` here cascades nowhere
    whether or not the loop pins, so the check could not fail (the P8
    sabotage pass, 2026-09-29)."""
    import httpx

    primary = ScriptedDriver(
        name="a",
        turns=[Turn(calls=[search_call()]), Turn(error=httpx.ConnectError("engine died"))],
    )
    backup = ScriptedDriver(name="b", turns=[Turn("<backup would have answered>")])
    slots = [{"model": "slot", "targets": [MODEL, "b-model"]}]
    backup.model_id = "b-model"
    app = app_with(settings, primary, backup, searches=[account(search)], slots=slots)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "slot",
                "messages": [{"role": "user", "content": "q"}],
                "web_search_options": {},
            },
        )
    assert response.status_code >= 500
    assert backup.calls == [], "the backup never saw the conversation"
    assert len(primary.calls) == 2


def test_before_any_search_the_first_call_still_cascades(settings, search) -> None:
    """The positive twin: without it the check above passes with failover broken."""
    import httpx

    primary = ScriptedDriver(name="a", turns=[Turn(error=httpx.ConnectError("down"))])
    backup = ScriptedDriver(name="b", turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    backup.model_id = "b-model"
    slots = [{"model": "slot", "targets": [MODEL, "b-model"]}]
    app = app_with(settings, primary, backup, searches=[account(search)], slots=slots)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "slot",
                "messages": [{"role": "user", "content": "q"}],
                "web_search_options": {},
            },
        )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == ANSWER
    assert len(backup.calls) == 2 and len(primary.calls) == 1
    assert response.json()["x_eugene_plexus"]["tier"] == 2
    assert response.json()["x_eugene_plexus"]["attempts"] == 2


# --------------------------------------------------------------------------- #
# Searches that fail, and accounts
# --------------------------------------------------------------------------- #


def test_a_failed_search_is_told_to_the_model_not_turned_into_a_failed_request(settings) -> None:
    broken = FakeSearch(
        error=ToolDriverError(
            driver="searx",
            status=502,
            detail="SearXNG at http://x refused JSON output (HTTP 403). Add `json` to `search.formats`",
            code="json_disabled",
        )
    )
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn("I could not search.")])
    with TestClient(app_with(settings, model, searches=[account(broken)])) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": MODEL,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "q"}],
                "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
            },
        )
    assert response.status_code == 200, response.text
    told = model.calls[1].messages[-1].content
    assert "could not be run" in told and "search.formats" in told
    assert "search.formats`. Answer without it" in told
    blocks = response.json()["content"]
    result = next(b for b in blocks if b["type"] == "web_search_tool_result")
    assert result["content"] == {
        "type": "web_search_tool_result_error",
        "error_code": "unavailable",
    }


def test_a_second_account_is_tried_only_when_the_first_could_not_do_the_job(settings) -> None:
    down = FakeSearch(error=ToolDriverError(driver="a", status=503, detail="not set up"))
    good = FakeSearch()
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(
        app_with(settings, model, searches=[account(down, "a"), account(good, "b")])
    ) as c:
        assert chat(c).status_code == 200
    assert len(down.asked) == 1 and len(good.asked) == 1

    refused = FakeSearch(error=ToolDriverError(driver="a", status=400, detail="bad query"))
    other = FakeSearch()
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(
        app_with(settings, model, searches=[account(refused, "a"), account(other, "b")])
    ) as c:
        assert chat(c).status_code == 200
    assert other.asked == [], "a refused query would be refused again in the same words"


def test_when_every_account_fails_the_model_is_told_each_reason_first_first(settings) -> None:
    """An unreachable second account must not hide the first one's reason,
    which is the one an operator has to fix (found on a slow CI runner,
    2026-09-29: the model was told only "could not be reached")."""
    json_off = FakeSearch(
        error=ToolDriverError(
            driver="searx",
            status=502,
            detail="SearXNG refused JSON output; add json to search.formats",
        )
    )
    gone = FakeSearch(
        error=ToolDriverError(driver="brave", status=0, detail="could not be reached")
    )
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn("answered without it")])
    accounts = [account(json_off, "searx"), account(gone, "brave")]
    with TestClient(app_with(settings, model, searches=accounts)) as c:
        assert chat(c).status_code == 200
    told = model.calls[1].messages[-1].content
    assert "searx: SearXNG refused JSON output" in told and "brave: could not be reached" in told
    assert told.index("search.formats") < told.index("could not be reached")


# --------------------------------------------------------------------------- #
# A backend that searches itself
# --------------------------------------------------------------------------- #


def test_a_backend_that_searches_itself_is_sent_the_request_natively(settings, search) -> None:
    hosted = ScriptedDriver(name="openai", turns=[Turn("Searched by the provider.")])
    hosted.supported_settings = [*hosted.supported_settings, "webSearchOptions"]
    with TestClient(app_with(settings, hosted, searches=[account(search)])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "q"}],
                "web_search_options": {"search_context_size": "high"},
            },
        )
    assert response.status_code == 200, response.text
    (sent,) = hosted.calls
    assert sent.webSearchOptions is not None
    assert sent.webSearchOptions.search_context_size.value == "high"
    assert not any(t.function.name == "web_search" for t in sent.tools or [])
    assert search.asked == [], "forwarded, not run twice"
    assert response.json()["x_eugene_plexus"].get("web_searches") is None


def test_a_model_that_neither_searches_nor_calls_tools_is_refused_saying_so(
    settings, search
) -> None:
    plain = FakeDriverClient(name="plain", model_id=MODEL, supports_tools=False)
    with TestClient(app_with(settings, plain, searches=[account(search)])) as client:
        response = chat(client)
    assert response.status_code == 400
    assert "neither runs web search itself nor calls tools" in response.json()["error"]["message"]
    assert plain.calls == []


# --------------------------------------------------------------------------- #
# When a search cannot run, each door says why
# --------------------------------------------------------------------------- #


def test_chat_without_a_search_account_keeps_p2c_and_names_the_reason(settings) -> None:
    model = ScriptedDriver(turns=[Turn("x")])
    with TestClient(app_with(settings, model)) as client:
        response = chat(client)
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "web_search_options" in message and "no search account is set up" in message


def test_responses_without_a_search_account_removes_the_tool_and_says_why(settings) -> None:
    model = ScriptedDriver(turns=[Turn("answered without searching")])
    with TestClient(app_with(settings, model)) as client:
        response = client.post(
            "/v1/responses",
            json={"model": MODEL, "input": "q", "tools": [{"type": "web_search"}]},
        )
    assert response.status_code == 200, response.text
    assert "tools.web_search" in response.headers["x-eugene-plexus-ignored-settings"]
    assert "no search account" in response.headers["x-eugene-plexus-web-search"]
    assert model.calls[0].tools is None


def test_codex_default_cached_search_is_removed_even_with_an_account(settings, search) -> None:
    """Design call #2: `external_web_access: false` did not ask for the prompt
    to leave the machine."""
    model = ScriptedDriver(turns=[Turn("fine")])
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": MODEL,
                "input": "q",
                "tools": [{"type": "web_search", "external_web_access": False}],
            },
        )
    assert "external_web_access is false" in response.headers["x-eugene-plexus-web-search"]
    assert search.asked == []


def test_a_local_only_key_and_a_key_denied_search_never_search(
    settings, search, monkeypatch
) -> None:
    for limits, words in (
        ({"localOnly": True}, "local-only"),
        ({"allowedTools": []}, "tool scope"),
    ):
        context = admission.ClientRequest({})
        context.access = {"limits": limits}
        from eugene_plexus_gateway import server_tools

        table = make_routing_table(ScriptedDriver(turns=[]))
        table._snapshot.tools = [account(search)]
        reason = server_tools.why_not(table, context)
        assert reason is not None and words in reason
    allowed = admission.ClientRequest({})
    allowed.access = {"limits": {"allowedTools": ["web_*"]}}
    assert server_tools.why_not(table, allowed) is None


def test_the_model_list_says_whether_a_search_can_run_and_where(settings, search) -> None:
    """C3 (workbench-v1.md section 3): a client can say why a search switch is
    off before it asks. The install-and-key answer is on the list, the
    reach on each model: one that calls tools or searches itself is
    reachable, one that does neither is not."""
    caller = ScriptedDriver(turns=[])
    hosted = FakeDriverClient(name="hosted", model_id="hosted-model", supports_tools=False)
    hosted.supported_settings = [*hosted.supported_settings, "webSearchOptions"]
    plain = FakeDriverClient(name="plain", model_id="plain-model", supports_tools=False)
    with TestClient(
        app_with(settings, caller, hosted, plain, searches=[account(search)])
    ) as client:
        listing = client.get("/v1/models").json()
    assert listing["x_eugene_plexus"]["web_search"] == {"available": True, "reason": None}
    reach = {m["id"]: m["x_eugene_plexus"]["web_search"] for m in listing["data"]}
    assert reach == {MODEL: True, "hosted-model": True, "plain-model": False}


def test_with_no_search_account_the_list_says_what_a_refusal_would(settings) -> None:
    model = ScriptedDriver(turns=[Turn("x")])
    with TestClient(app_with(settings, model)) as client:
        listing = client.get("/v1/models").json()
        refused = chat(client)
    web = listing["x_eugene_plexus"]["web_search"]
    assert web["available"] is False
    assert "no search account is set up" in web["reason"]
    assert web["reason"] in refused.json()["error"]["message"], "the same words, before and after"
    assert listing["data"][0]["x_eugene_plexus"]["web_search"] is True, "the model could be reached"


def test_an_account_that_is_not_set_up_is_named(settings) -> None:
    from eugene_plexus_gateway import server_tools

    table = make_routing_table(ScriptedDriver(turns=[]))
    table._snapshot.tools = [account(FakeSearch(), "my-searx", configured=False)]
    assert "my-searx" in (server_tools.why_not(table, None) or "")


# --------------------------------------------------------------------------- #
# Responses: the web_search_call item
# --------------------------------------------------------------------------- #


def test_responses_records_each_search_as_a_web_search_call_item(settings, search) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": MODEL,
                "input": "What is Eugene Plexus?",
                "tools": [
                    {"type": "web_search", "external_web_access": True},
                    {"type": "function", "name": "shell", "parameters": {}},
                ],
            },
        )
    assert response.status_code == 200, response.text
    assert "x-eugene-plexus-web-search" not in response.headers
    assert "tools.web_search" not in response.headers.get("x-eugene-plexus-ignored-settings", "")
    output = response.json()["output"]
    assert [i["type"] for i in output] == ["web_search_call", "message"]
    call = output[0]
    assert call["action"]["query"] == "eugene plexus"
    assert [s["url"] for s in call["action"]["sources"]] == [r["url"] for r in RESULTS]
    annotations = output[1]["content"][0]["annotations"]
    assert annotations[0]["type"] == "url_citation" and annotations[0]["url"] == RESULTS[0]["url"]


def test_responses_streams_a_search_as_openai_does(settings, search) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = client.post(
            "/v1/responses",
            json={"model": MODEL, "input": "q", "tools": [{"type": "web_search"}], "stream": True},
        )
    names = [n for n, _ in events(response)]
    at = names.index("response.web_search_call.in_progress")
    assert names[at - 1] == "response.output_item.added"
    assert names[at + 1] == "response.web_search_call.searching"
    assert "response.web_search_call.completed" in names
    assert "response.output_text.annotation.added" in names
    completed = next(d for n, d in events(response) if n == "response.completed")["response"]
    assert [i["type"] for i in completed["output"]] == ["web_search_call", "message"]
    sequence = [d["sequence_number"] for _, d in events(response)]
    assert sequence == sorted(sequence)


def test_responses_web_search_blocks_domains_and_names_what_it_does_not_read(
    settings, search
) -> None:
    """`filters.blocked_domains` was dropped on the floor while
    `allowed_domains` beside it was honoured, and every other setting on the
    tool went unread and unnamed (upstream drift audit, 2026-10-03). The
    shape is Codex 0.160's `ToolSpec::WebSearch` with its search config set."""
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    tool = {
        "type": "web_search",
        "external_web_access": True,
        "indexed_web_access": True,
        "filters": {
            "allowed_domains": ["github.com"],
            "blocked_domains": ["example.com"],
            "a_future_filter": True,
        },
        "user_location": {"type": "approximate", "country": "US"},
        "search_context_size": "low",
        "search_content_types": ["text", "image"],
        "image_settings": {"max_results": 4, "caption": True},
        "return_token_budget": 2000,
    }
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = client.post(
            "/v1/responses", json={"model": MODEL, "input": "q", "tools": [tool]}
        )
    assert response.status_code == 200, response.text
    (asked,) = search.asked
    assert [d.root for d in asked.allowedDomains] == ["github.com"]
    assert [d.root for d in asked.blockedDomains] == ["example.com"]
    ignored = response.headers["x-eugene-plexus-ignored-settings"].split(", ")
    for name in (
        "tools.web_search.search_content_types",
        "tools.web_search.image_settings",
        "tools.web_search.return_token_budget",
        "tools.web_search.indexed_web_access",
        "tools.web_search.filters.a_future_filter",
    ):
        assert name in ignored, ignored
    # What is honoured is not named, and the tool itself ran.
    assert not [i for i in ignored if i.startswith("tools.web_search.filters.") and "domains" in i]
    for honoured in (
        "tools.web_search",
        "tools.web_search.user_location",
        "tools.web_search.search_context_size",
        "tools.web_search.external_web_access",
        "tools.web_search.type",
    ):
        assert honoured not in ignored, ignored


def test_a_web_search_call_handed_back_becomes_history(settings) -> None:
    model = ScriptedDriver(turns=[Turn("ok")])
    with TestClient(app_with(settings, model)) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": MODEL,
                "input": [
                    {"type": "message", "role": "user", "content": "first"},
                    {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "status": "completed",
                        "action": {
                            "type": "search",
                            "query": "eugene",
                            "sources": [{"type": "url", "url": "https://a.example"}],
                        },
                    },
                    {"type": "message", "role": "assistant", "content": "It is a control plane."},
                    {"type": "message", "role": "user", "content": "more"},
                ],
            },
        )
    assert response.status_code == 200, response.text
    assistant = next(m for m in model.calls[0].messages if m.role.value == "assistant")
    text = (
        assistant.content
        if isinstance(assistant.content, str)
        else json.dumps([p.model_dump() for p in assistant.content.root])
    )
    assert 'Searched the web for \\"eugene\\"' in text or 'Searched the web for "eugene"' in text
    assert "https://a.example" in text


# --------------------------------------------------------------------------- #
# Messages: Claude Code's WebSearch request
# --------------------------------------------------------------------------- #

CLAUDE_CODE_SEARCH = {
    "model": MODEL,
    "max_tokens": 32000,
    "system": [
        {"type": "text", "text": "You are an assistant for performing a web search tool use"}
    ],
    "messages": [{"role": "user", "content": "Perform a web search for the query: eugene plexus"}],
    "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
    "tool_choice": {"type": "auto"},
    "output_config": {"effort": "high"},
}


def test_claude_codes_websearch_request_is_answered_with_anthropics_blocks(
    settings, search
) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = client.post("/v1/messages", json=CLAUDE_CODE_SEARCH)
    assert response.status_code == 200, response.text
    body = response.json()
    kinds = [b["type"] for b in body["content"]]
    assert kinds == ["server_tool_use", "web_search_tool_result", "text"]
    use, result, text = body["content"]
    assert use["name"] == "web_search" and use["input"] == {"query": "eugene plexus"}
    assert result["tool_use_id"] == use["id"] and use["id"].startswith("srvtoolu_")
    assert [r["url"] for r in result["content"]] == [r["url"] for r in RESULTS]
    assert all(r["encrypted_content"] for r in result["content"])
    assert text["text"] == ANSWER
    assert body["usage"]["server_tool_use"] == {"web_search_requests": 1}
    assert body["stop_reason"] == "end_turn"
    # It offered the model exactly one tool, and forced the first call.
    first = model.calls[0]
    assert [t.function.name for t in first.tools] == ["web_search"]
    assert str(first.toolChoice) == "required"


def test_the_messages_stream_carries_the_search_blocks_in_order(settings, search) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = client.post("/v1/messages", json={**CLAUDE_CODE_SEARCH, "stream": True})
    started = [
        d["content_block"]["type"] for n, d in events(response) if n == "content_block_start"
    ]
    assert started == ["server_tool_use", "web_search_tool_result", "text"]
    indexes = [d["index"] for n, d in events(response) if n == "content_block_start"]
    assert indexes == [0, 1, 2]
    stops = [d["index"] for n, d in events(response) if n == "content_block_stop"]
    assert stops == [0, 1, 2], "every block that opened also closed"
    delta = next(d for n, d in events(response) if n == "message_delta")
    assert delta["usage"]["server_tool_use"] == {"web_search_requests": 1}


def _backend_400(detail: str) -> DriverError:
    return DriverError(
        driver_name="local",
        driver_url="http://local.invalid",
        status_code=400,
        problem=Problem(type="about:blank", title="Backend rejected", status=400, detail=detail),
        raw_body="",
    )


#: A hosted Claude 5.5 refusing a forced tool choice, as OpenRouter relays
#: an Anthropic 400 through the driver. Anthropic's docs say Opus 5.5,
#: Sonnet 5.5 and Fable 5.1 reject `tool_choice` `any` and `tool`; the exact
#: words are not captured here, so the rule keys on the parameter's name.
FORCED_REFUSED = (
    'openai_compat_http returned 400: {"error":{"message":"Provider returned error","code":400,'
    '"metadata":{"raw":"{\\"type\\":\\"error\\",\\"error\\":{\\"type\\":'
    '\\"invalid_request_error\\",\\"message\\":\\"tool_choice: forcing tool use is not '
    'supported with this model.\\"}}","provider_name":"Anthropic"}}}'
)


@pytest.mark.parametrize("stream", [False, True])
def test_a_model_that_refuses_a_forced_first_search_is_asked_again_with_auto(
    settings, search, stream: bool
) -> None:
    """The loop forces the first call when search is the only tool; a model
    that rejects forced tool choice failed Claude Code's WebSearch outright
    (upstream drift audit, 2026-10-03). Asked once more with `auto` and told
    to search, it searches."""
    model = ScriptedDriver(
        turns=[
            Turn(error=_backend_400(FORCED_REFUSED)),
            Turn(calls=[search_call()]),
            Turn(ANSWER),
        ]
    )
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = client.post("/v1/messages", json={**CLAUDE_CODE_SEARCH, "stream": stream})
    assert response.status_code == 200, response.text
    if stream:
        kinds = [
            d["content_block"]["type"] for n, d in events(response) if n == "content_block_start"
        ]
        assert not [n for n, _ in events(response) if n == "error"]
    else:
        kinds = [b["type"] for b in response.json()["content"]]
    assert kinds == ["server_tool_use", "web_search_tool_result", "text"]
    forced, unforced, answered = model.calls
    assert str(forced.toolChoice) == "required"
    assert str(unforced.toolChoice) == "auto"
    assert [t.function.name for t in unforced.tools] == ["web_search"]
    # Told in words what the forced choice said in a parameter.
    instruction = unforced.messages[0]
    assert instruction.role.value == "system" and "web_search" in instruction.content
    assert instruction.content.startswith(forced.messages[0].content)
    assert [r.query for r in search.asked] == ["eugene plexus"]
    assert str(answered.toolChoice) != "required"


def test_a_400_that_is_not_about_tool_choice_is_not_asked_again(settings, search) -> None:
    """The twin: a refusal of anything else is the caller's answer, once."""
    model = ScriptedDriver(
        turns=[Turn(error=_backend_400("Jinja Exception: bad role")), Turn(ANSWER)]
    )
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = client.post("/v1/messages", json=CLAUDE_CODE_SEARCH)
    assert response.status_code == 400
    assert len(model.calls) == 1
    assert not search.asked


def test_a_callers_own_forced_choice_is_never_loosened(settings, search) -> None:
    """Only the loop's own forcing is softened: a caller that asked for a
    forced call and got a refusal is told so, not answered some other way."""
    model = ScriptedDriver(turns=[Turn(error=_backend_400(FORCED_REFUSED)), Turn(ANSWER)])
    asked_for_any = {**CLAUDE_CODE_SEARCH, "tool_choice": {"type": "any"}}
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = client.post("/v1/messages", json=asked_for_any)
    assert response.status_code == 400
    assert len(model.calls) == 1


def test_search_blocks_handed_back_become_the_turns_history(settings) -> None:
    from eugene_plexus_gateway.anthropic import encode_signature

    model = ScriptedDriver(turns=[Turn("ok")])
    with TestClient(app_with(settings, model)) as client:
        response = client.post(
            "/v1/messages",
            json={
                "model": MODEL,
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "q"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "server_tool_use",
                                "id": "srvtoolu_1",
                                "name": "web_search",
                                "input": {"query": "eugene"},
                            },
                            {
                                "type": "web_search_tool_result",
                                "tool_use_id": "srvtoolu_1",
                                "content": [
                                    {
                                        "type": "web_search_result",
                                        "url": "https://a.example",
                                        "title": "A",
                                        "encrypted_content": encode_signature(
                                            "the excerpt it read"
                                        ),
                                    }
                                ],
                            },
                            {"type": "text", "text": "Answer."},
                        ],
                    },
                    {"role": "user", "content": "and?"},
                ],
            },
        )
    assert response.status_code == 200, response.text
    assistant = next(m for m in model.calls[0].messages if m.role.value == "assistant")
    text = json.dumps(assistant.model_dump(mode="json"))
    assert "eugene" in text and "https://a.example" in text and "the excerpt it read" in text


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_metrics_keep_each_search_and_no_query_text(settings, search) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call("secret words")]), Turn(ANSWER)])
    app = app_with(settings, model, searches=[account(search)])
    with TestClient(app) as client:
        assert chat(client).status_code == 200
        rows: list[Any] = []
        for _ in range(100):
            rows = client.get("/v1/metrics/requests").json()["requests"]
            if rows:
                break
            time.sleep(0.05)
    row = rows[0]
    assert row["attempts"] == 1, "a turn of one answer is not a cascade"
    assert len(row["tries"]) == 2, "both turns are still visible as attempts"
    assert row["webSearches"] == [
        {
            "tool": "web_search",
            "driver": "searx",
            "node": None,
            "provider": "searxng",
            "version": "web_search_options",
            "outcome": "ok",
            "results": 2,
            "elapsedMs": row["webSearches"][0]["elapsedMs"],
        }
    ]
    raw = Path(settings.metrics_file).read_bytes()
    assert b"secret words" not in raw


# --------------------------------------------------------------------------- #
# The helpers
# --------------------------------------------------------------------------- #


def test_the_search_tool_never_takes_a_callers_function_name() -> None:
    from eugene_plexus_gateway._generated.driver_models import FunctionDefinition, Tool

    mine = [Tool(type="function", function=FunctionDefinition(name="web_search"))]
    assert tool_name_for(mine) == "eugene_web_search"
    assert tool_name_for(None) == "web_search"


def test_a_query_is_read_however_a_small_model_sends_it() -> None:
    assert query_of('{"query": "a"}') == "a"
    assert query_of('"b"') == "b"
    assert query_of('{"q": "c"}') == "c"
    assert query_of("plain words") == "plain words"
    assert query_of(None) == ""


def test_only_an_address_the_answer_cites_is_a_citation() -> None:
    from eugene_plexus_gateway.server_tools import Execution

    done = Execution(call_id="c", query="q", version="v", outcome="ok", results=RESULTS)
    text = "See https://eugeneplexus.com for more."
    cited = citations(text, [done])
    assert [c["url_citation"]["url"] for c in cited] == ["https://eugeneplexus.com/"]
    assert text[
        cited[0]["url_citation"]["start_index"] : cited[0]["url_citation"]["end_index"]
    ] == ("https://eugeneplexus.com")
    assert citations("nothing cited", [done]) == []


def test_a_profile_default_does_not_throw_away_the_loops_turns(settings, search) -> None:
    """`prepare_candidate` used to rebuild each candidate's request from the
    caller's body, so the search results appended by the loop were lost
    whenever a profile default applied -- nearly every request."""

    class Profiles:
        async def get(self, path: Any) -> dict[str, Any]:
            return {"maxTokens": 77, "temperature": 0.3}

    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    app = app_with(settings, model, searches=[account(search)])
    app.state.profile_defaults = Profiles()
    with TestClient(app) as client:
        assert chat(client).status_code == 200
    second = model.calls[1]
    assert second.maxTokens == 77
    assert second.messages[-1].role.value == "tool", "the results survived the defaults"


@pytest.fixture(autouse=True)
def _no_ambient(monkeypatch) -> Iterator[None]:
    yield


def test_a_newer_anthropic_tool_date_is_accepted_and_its_unknown_settings_named(
    settings, search
) -> None:
    """Refusing an unknown date would break WebSearch the day Claude Code
    sends one (Troy, 2026-09-29): any `web_search_<8 digits>` runs, the
    settings this gateway knows are honoured, and any other is named."""
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    newer = {
        **CLAUDE_CODE_SEARCH,
        "tools": [
            {
                "type": "web_search_20991231",
                "name": "web_search",
                "max_uses": 3,
                "allowed_domains": ["github.com"],
                "a_future_knob": True,
            }
        ],
    }
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = client.post("/v1/messages", json=newer)
    assert response.status_code == 200, response.text
    assert "tools.web_search.a_future_knob" in response.headers["x-eugene-plexus-ignored-settings"]
    (asked,) = search.asked
    assert [d.root for d in asked.allowedDomains] == ["github.com"]
    malformed = {**CLAUDE_CODE_SEARCH, "tools": [{"type": "web_search_2099", "name": "web_search"}]}
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        refused = client.post("/v1/messages", json=malformed)
    assert refused.status_code == 400 and "web_search_2099" in refused.json()["error"]["message"]


def test_a_failed_search_ends_its_reason_with_one_full_stop(settings) -> None:
    # Brave's refusal is a sentence with its own full stop; the model was told
    # "...(HTTP 429).. Answer without it" until 2026-10-01 (gateway#4).
    limited = FakeSearch(
        error=ToolDriverError(
            driver="brave",
            status=429,
            detail="Brave Search says this key is over its rate or monthly limit (HTTP 429).",
            code="rate_limited",
        )
    )
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn("I could not search.")])
    with TestClient(app_with(settings, model, searches=[account(limited)])) as client:
        assert chat(client).status_code == 200
    told = model.calls[1].messages[-1].content
    assert "(HTTP 429). Answer without it, and say that the search failed." in told
    assert ".." not in told


# --------------------------------------------------------------------------- #
# Search Suggestions (GS4): carried verbatim to the caller, never to the model
# --------------------------------------------------------------------------- #

#: Google's own markup is passed through whole: quotes, a newline, a style
#: block and an entity must all survive, because its terms forbid editing it.
SUGGEST = (
    '<div class="container">\n<style>.chip{color:#1a73e8}</style>&amp; '
    '<a href="https://www.google.com/search?q=a">a "b"</a> é</div>'
)
SUGGEST_2 = '<div class="container">second</div>'


def _suggesting(*suggestions: str | None, answer: str | None = None) -> FakeSearch:
    return FakeSearch(suggestions=list(suggestions), answer=answer)


def test_chat_carries_each_searchs_suggestions_in_run_order_and_not_to_the_model(settings) -> None:
    found = _suggesting(SUGGEST, None, SUGGEST_2, answer="Google says so.")
    model = ScriptedDriver(
        turns=[
            Turn(calls=[search_call("one", "call_a")]),
            Turn(calls=[search_call("two", "call_b")]),
            Turn(calls=[search_call("three", "call_c")]),
            Turn(ANSWER),
        ]
    )
    with TestClient(app_with(settings, model, searches=[account(found)])) as client:
        response = chat(client)
    body = response.json()
    assert body["x_eugene_plexus"]["search_suggestions"] == [SUGGEST, SUGGEST_2]
    assert body["x_eugene_plexus"]["web_searches"] == 3
    for call in model.calls:
        for message in call.messages:
            assert "container" not in (message.content or ""), "the model never reads the HTML"
            assert "chip" not in (message.content or "")
    told = model.calls[1].messages[-1].content
    assert "Answer from the search provider: Google says so." in told


def test_chat_has_no_suggestions_when_the_provider_gave_none(settings, search) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = chat(client)
    # The door's other extension fields read null when unset; none is null here.
    assert response.json()["x_eugene_plexus"].get("search_suggestions") is None


def test_the_chat_stream_carries_suggestions_in_its_final_extension_block(settings) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(app_with(settings, model, searches=[account(_suggesting(SUGGEST))])) as client:
        response = chat(client, stream=True)
    final = [f for f in sse(response) if (f.get("choices") or [{}])[0].get("finish_reason")]
    assert final[-1]["x_eugene_plexus"]["search_suggestions"] == [SUGGEST]


def test_the_chat_stream_has_none_when_the_provider_gave_none(settings, search) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = chat(client, stream=True)
    final = [f for f in sse(response) if (f.get("choices") or [{}])[0].get("finish_reason")]
    assert "search_suggestions" not in final[-1]["x_eugene_plexus"]


def test_responses_item_carries_its_searchs_suggestions(settings) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(app_with(settings, model, searches=[account(_suggesting(SUGGEST))])) as client:
        response = client.post(
            "/v1/responses",
            json={"model": MODEL, "input": "q", "tools": [{"type": "web_search"}]},
        )
    call = response.json()["output"][0]
    assert call["type"] == "web_search_call"
    assert call["x_eugene_plexus"] == {"search_suggestions": SUGGEST}
    assert "search-suggestions" not in str(dict(response.headers)).lower()
    assert "container" not in str(model.calls[1].messages[-1].content)


def test_responses_stream_item_done_carries_suggestions_and_the_added_one_does_not(
    settings,
) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(app_with(settings, model, searches=[account(_suggesting(SUGGEST))])) as client:
        response = client.post(
            "/v1/responses",
            json={"model": MODEL, "input": "q", "tools": [{"type": "web_search"}], "stream": True},
        )
    seen = events(response)
    added = next(d for n, d in seen if n == "response.output_item.added")["item"]
    done = next(
        d
        for n, d in seen
        if n == "response.output_item.done" and d["item"]["type"] == "web_search_call"
    )["item"]
    assert "x_eugene_plexus" not in added
    assert done["x_eugene_plexus"] == {"search_suggestions": SUGGEST}
    completed = next(d for n, d in seen if n == "response.completed")["response"]
    assert completed["output"][0]["x_eugene_plexus"] == {"search_suggestions": SUGGEST}


def test_responses_item_has_no_extension_when_the_provider_gave_none(settings, search) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = client.post(
            "/v1/responses",
            json={"model": MODEL, "input": "q", "tools": [{"type": "web_search"}]},
        )
    assert "x_eugene_plexus" not in response.json()["output"][0]


def test_messages_result_block_carries_its_searchs_suggestions(settings) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(app_with(settings, model, searches=[account(_suggesting(SUGGEST))])) as client:
        response = client.post("/v1/messages", json=CLAUDE_CODE_SEARCH)
    use, result, _ = response.json()["content"]
    assert "x_eugene_plexus" not in use
    assert result["x_eugene_plexus"] == {"search_suggestions": SUGGEST}
    assert "search-suggestions" not in str(dict(response.headers)).lower()
    assert "container" not in str(model.calls[1].messages[-1].content)


def test_messages_stream_result_block_carries_suggestions(settings) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(app_with(settings, model, searches=[account(_suggesting(SUGGEST))])) as client:
        response = client.post("/v1/messages", json={**CLAUDE_CODE_SEARCH, "stream": True})
    blocks = [d["content_block"] for n, d in events(response) if n == "content_block_start"]
    result = next(b for b in blocks if b["type"] == "web_search_tool_result")
    assert result["x_eugene_plexus"] == {"search_suggestions": SUGGEST}
    assert all("x_eugene_plexus" not in b for b in blocks if b is not result)


def test_messages_result_block_has_no_extension_when_the_provider_gave_none(
    settings, search
) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    with TestClient(app_with(settings, model, searches=[account(search)])) as client:
        response = client.post("/v1/messages", json=CLAUDE_CODE_SEARCH)
    assert "x_eugene_plexus" not in response.json()["content"][1]


def test_suggestions_are_not_in_the_metrics_row_or_its_file(settings) -> None:
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)])
    app = app_with(settings, model, searches=[account(_suggesting(SUGGEST))])
    with TestClient(app) as client:
        assert chat(client).status_code == 200
        text = ""
        for _ in range(100):
            text = client.get("/v1/metrics/requests").text
            if "webSearches" in text:
                break
            time.sleep(0.05)
    assert "webSearches" in text
    assert "container" not in text and "chip" not in text
    raw = Path(settings.metrics_file).read_bytes()
    assert b"container" not in raw and b"1a73e8" not in raw


def test_the_models_tool_message_keeps_the_providers_answer_and_never_the_html() -> None:
    from eugene_plexus_gateway.server_tools import Execution, result_text

    execution = Execution(
        call_id="c",
        query="q",
        version="web_search",
        outcome="ok",
        results=RESULTS,
        answer="Words from Google.",
        search_suggestions=SUGGEST,
    )
    text = result_text(execution, limit=5)
    assert "Answer from the search provider: Words from Google." in text
    assert "container" not in text and "1a73e8" not in text


# --------------------------------------------------------------------------- #
# The order a search tries its accounts (GS7)
# --------------------------------------------------------------------------- #


def _two_accounts() -> tuple[FakeSearch, FakeSearch, list[ToolAccount]]:
    free, paid = FakeSearch(), FakeSearch()
    # The snapshot's order is the default rule's: free first.
    return (
        free,
        paid,
        [
            account(free, "searx", billing="free"),
            account(paid, "google", provider="google", label="Google Search"),
        ],
    )


def test_web_search_order_takes_effect_on_the_next_search_after_a_patch(settings) -> None:
    free, paid, accounts = _two_accounts()
    model = ScriptedDriver(turns=[Turn(calls=[search_call()]), Turn(ANSWER)] * 3)
    app = app_with(settings, model, searches=accounts)
    with TestClient(app) as client:
        # Wired as app.py wires it: read from the live store on every call.
        app.state.routing._search_order = lambda: app.state.config_store.get("webSearchOrder")
        assert chat(client).status_code == 200
        assert (len(free.asked), len(paid.asked)) == (1, 0), "default rule: free first"
        patched = client.patch("/v1/config", json={"webSearchOrder": ["google"]})
        assert patched.status_code == 200, patched.text
        assert chat(client).status_code == 200
        assert (len(free.asked), len(paid.asked)) == (1, 1), "named account first, no restart"
        assert client.patch("/v1/config", json={"webSearchOrder": []}).status_code == 200
        assert chat(client).status_code == 200
        assert (len(free.asked), len(paid.asked)) == (2, 1), "emptied: the default rule again"


def test_web_search_order_is_a_string_list_setting_that_says_what_unset_means(settings) -> None:
    app = create_app(settings=settings)
    with TestClient(app) as client:
        schema = client.get("/v1/config/schema").json()
        field = next(f for f in schema["fields"] if f["key"] == "webSearchOrder")
        assert field["label"] == "Web search order"
        assert field["valueType"] == "string_list"
        assert field["requiresRestart"] is False
        assert "free search accounts before billed ones" in field["unsetMeans"]
        assert client.get("/v1/config").json()["webSearchOrder"] == []
        for bad in ("google", [3], ["a", "a"], [""]):
            body = client.patch("/v1/config", json={"webSearchOrder": bad}).json()
            assert body["applied"] == [], bad
            assert [r["key"] for r in body["rejected"]] == ["webSearchOrder"], bad
        assert client.get("/v1/config").json()["webSearchOrder"] == []


def _table_with(order: list[str] | None, *accounts: ToolAccount) -> Any:
    from eugene_plexus_gateway.routing import RoutingTable

    table = RoutingTable(agent_url="http://fake-agent", search_order=lambda: order)
    table._snapshot.tools = list(accounts)
    return table


def _names(table: Any) -> list[str]:
    return [f"{a.node}:{a.name}" for a in table.search_accounts()]


def test_order_entries_are_bare_names_or_node_colon_name_and_the_rest_keep_the_default() -> None:
    a = account(FakeSearch(), "g", node="box", billing="free")
    b = account(FakeSearch(), "g", node="other")
    c = account(FakeSearch(), "brave", node="box")
    d = account(FakeSearch(), "searx", node="box", billing="free")
    default = ["box:g", "other:g", "box:brave", "box:searx"]
    assert _names(_table_with(None, a, b, c, d)) == default
    assert _names(_table_with([], a, b, c, d)) == default
    # node:name picks one account.
    assert _names(_table_with(["other:g"], a, b, c, d)) == [
        "other:g",
        "box:g",
        "box:brave",
        "box:searx",
    ]
    # A bare name matches every node's account of it, in default order.
    assert _names(_table_with(["g", "searx"], a, b, c, d)) == [
        "box:g",
        "other:g",
        "box:searx",
        "box:brave",
    ]
    # The order is the list's; unknown names and a repeat change nothing else.
    assert _names(_table_with(["nope", "box:searx", "nobody:g", "box:searx", " "], a, b, c, d)) == [
        "box:searx",
        "box:g",
        "other:g",
        "box:brave",
    ]
    # A node's name alone is not an account's.
    assert _names(_table_with(["box"], a, b, c, d)) == default


def test_an_account_that_does_not_run_is_still_skipped_when_named() -> None:
    off = account(FakeSearch(), "google", configured=False)
    on = account(FakeSearch(), "searx", billing="free")
    table = _table_with(["google"], on, off)
    assert [x.name for x in table.search_accounts()] == ["searx"]


def test_the_routing_view_lists_search_accounts_in_the_order_tried_with_where_from(
    settings,
) -> None:
    _, _, accounts = _two_accounts()
    idle = account(FakeSearch(), "brave", configured=False, provider="brave", label="Brave")
    app = app_with(settings, ScriptedDriver(turns=[]), searches=[*accounts, idle])
    with TestClient(app) as client:
        app.state.routing._search_order = lambda: app.state.config_store.get("webSearchOrder")
        before = client.get("/v1/admin/routing").json()["search_accounts"]
        assert [(x["name"], x["placed_by"]) for x in before] == [
            ("searx", "default"),
            ("google", "default"),
            ("brave", "default"),
        ]
        client.patch("/v1/config", json={"webSearchOrder": ["google", "brave", "ghost"]})
        after = client.get("/v1/admin/routing").json()["search_accounts"]
    assert [(x["name"], x["placed_by"]) for x in after] == [
        ("google", "order"),
        ("brave", "order"),
        ("searx", "default"),
    ]
    assert after[0] == {
        "name": "google",
        "node": None,
        "provider": "google",
        "label": "Google Search",
        "billing": "per_search",
        "runs": True,
        "placed_by": "order",
    }
    assert after[1]["runs"] is False, "set up or not, it is listed in its place"
    assert after[2]["billing"] == "free"


def test_an_account_that_reports_no_billing_is_listed_as_billed(settings) -> None:
    """An older tool driver says nothing of billing; absent is read as
    billed (tool-driver.yaml `ToolBilling`), and the view still answers."""
    quiet = account(FakeSearch(), "old", billing=None)
    app = app_with(settings, ScriptedDriver(turns=[]), searches=[quiet])
    with TestClient(app) as client:
        response = client.get("/v1/admin/routing")
    assert response.status_code == 200, response.text
    assert response.json()["search_accounts"][0]["billing"] == "per_search"
