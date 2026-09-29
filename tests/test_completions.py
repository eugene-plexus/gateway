"""POST /v1/completions (P6): OpenAI's legacy completions, on the chat path.

The request shape is OpenAI's (the SDK's `client.completions.create`); the
prompt reaches the driver as `completion`, continued as written, and a suffix
only a model that fills in the middle. Every test here fails against the
gateway before P6, which had no completions door.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import FinishReason, Usage
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table

PROMPT = "<|fim_prefix|>def add(a, b):\n    return<|fim_suffix|>\n<|fim_middle|>"


class Completer(FakeDriverClient):
    """A fake whose model continues raw text, and with `fills` fills the
    middle, as a local llama-server's driver reports since P6."""

    def __init__(self, *, fills: bool = False, **kw: Any) -> None:
        super().__init__(**kw)
        self.fills = fills

    def describe(self):  # type: ignore[no-untyped-def]
        info = super().describe()
        for model in info.models or []:
            model.surfaces = ["chat", "completion"]
            if model.capabilities is not None:
                model.capabilities.fillInMiddle = self.fills
        return info


def serve(settings: Settings, *drivers: FakeDriverClient, **table: Any) -> TestClient:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*drivers, **table)
    return TestClient(app)


def complete(client: TestClient, **body: Any) -> httpx.Response:
    return client.post("/v1/completions", json={"model": "coder", "prompt": PROMPT, **body})


def _rows(client: TestClient) -> list[dict[str, Any]]:
    for _ in range(50):
        rows = client.get("/v1/metrics/requests").json()["requests"]
        if rows:
            return rows
        time.sleep(0.02)
    return []


def test_a_model_that_continues_raw_text_is_listed_so(settings: Settings) -> None:
    fills = Completer(name="a", model_id="coder", fills=True)
    chat = FakeDriverClient(name="c", model_id="chatty")
    with serve(settings, fills, chat) as client:
        listed = {m["id"]: m["x_eugene_plexus"] for m in client.get("/v1/models").json()["data"]}
    assert listed["coder"]["surfaces"] == ["chat", "completion"]
    assert listed["coder"]["fill_in_middle"] is True
    assert "completion" not in listed["chatty"]["surfaces"]


def test_the_prompt_goes_to_the_driver_as_written_and_comes_back_as_text(
    settings: Settings,
) -> None:
    coder = Completer(name="a", model_id="coder")
    coder.responses = [" a + b"]
    coder.usage = Usage(promptTokens=20, completionTokens=4, totalTokens=24)
    with serve(settings, coder) as client:
        response = complete(client, max_tokens=8, temperature=0.01, stop=["\n\n"])
        rows = _rows(client)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "text_completion" and body["model"] == "coder"
    assert body["choices"] == [
        {"text": " a + b", "index": 0, "logprobs": None, "finish_reason": "stop"}
    ]
    assert body["usage"] == {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
    [sent] = coder.calls
    assert sent.completion is not None and sent.completion.prompt == PROMPT
    assert sent.completion.suffix is None and sent.messages == []
    assert (sent.maxTokens, sent.temperature) == (8, 0.01)
    assert (rows[0]["door"], rows[0]["servedModel"]) == ("completion", "coder")
    # What served it, as chat's answer says (found missing by the U7 browser run).
    assert body["x_eugene_plexus"]["driver"] == "a"
    assert body["x_eugene_plexus"]["attempts"] == 1 and body["x_eugene_plexus"]["tier"] == 1


def test_a_suffix_routes_only_to_a_model_that_fills_in_the_middle(settings: Settings) -> None:
    plain = Completer(name="a", model_id="plain")
    filler = Completer(name="b", model_id="filler", fills=True)
    slots = [{"model": "coder", "targets": ["plain", "filler"]}]
    with serve(settings, plain, filler, slots=slots) as client:
        response = complete(client, prompt="def add(a, b):\n    return", suffix="\nprint(1)\n")
        rows = _rows(client)
    assert response.status_code == 200, response.text
    assert not plain.calls and filler.calls
    assert (
        filler.calls[0].completion is not None
        and filler.calls[0].completion.suffix == "\nprint(1)\n"
    )
    assert rows[0]["tier"] == 2


def test_a_suffix_nothing_can_fill_is_refused_naming_it(settings: Settings) -> None:
    plain = Completer(name="a", model_id="coder")
    with serve(settings, plain) as client:
        response = complete(client, suffix="\n")
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == "suffix" and "fill" in error["message"]
    assert not plain.calls


def test_a_completion_never_reaches_a_model_that_only_chats(settings: Settings) -> None:
    chat = FakeDriverClient(name="c", model_id="chatty")
    coder = Completer(name="a", model_id="coder")
    slots = [{"model": "mixed", "targets": ["chatty", "coder"]}]
    with serve(settings, chat, coder, slots=slots) as client:
        via_slot = complete(client, model="mixed")
        direct = complete(client, model="chatty")
    assert via_slot.status_code == 200, via_slot.text
    assert not chat.calls and coder.calls
    assert direct.status_code == 400 and "/v1/chat/completions" in direct.json()["error"]["message"]


def test_what_this_door_does_not_take_is_refused_naming_the_field(settings: Settings) -> None:
    coder = Completer(name="a", model_id="coder")
    with serve(settings, coder) as client:
        answers = {
            "n": complete(client, n=2),
            "best_of": complete(client, best_of=3),
            "echo": complete(client, echo=True),
            "logprobs": complete(client, logprobs=2),
            "prompt": complete(client, prompt=["a", "b"]),
            "tokens": complete(client, prompt=[1, 2, 3]),
            "bogus": complete(client, bogus=1),
            "stream_options": complete(client, stream=True, stream_options={"x": 1}),
        }
    for field, response in answers.items():
        assert response.status_code == 400, (field, response.text)
    assert answers["n"].json()["error"]["param"] == "n"
    assert answers["prompt"].json()["error"]["param"] == "prompt"
    assert answers["bogus"].json()["error"]["param"] == "bogus"
    assert not coder.calls


def test_a_one_item_prompt_array_is_a_prompt(settings: Settings) -> None:
    coder = Completer(name="a", model_id="coder")
    with serve(settings, coder) as client:
        response = complete(client, prompt=["x = "])
    assert response.status_code == 200, response.text
    assert coder.calls[0].completion is not None and coder.calls[0].completion.prompt == "x = "


def test_a_stream_is_text_completion_chunks_then_usage_then_done(settings: Settings) -> None:
    coder = Completer(name="a", model_id="coder")
    coder.responses = ["one two three"]
    coder.usage = Usage(promptTokens=5, completionTokens=3, totalTokens=8)
    coder.finish_reason = FinishReason.length
    with serve(settings, coder) as client:
        response = complete(client, stream=True, stream_options={"include_usage": True})
    lines = [ln[6:] for ln in response.text.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    frames = [json.loads(ln) for ln in lines[:-1]]
    text = "".join(c["text"] for f in frames for c in f["choices"])
    assert text == "one two three"
    assert all(f["object"] == "text_completion" for f in frames)
    finishes = [c["finish_reason"] for f in frames for c in f["choices"] if c["finish_reason"]]
    assert finishes == ["length"]
    assert frames[-1]["choices"] == [] and frames[-1]["usage"]["completion_tokens"] == 3
    assert coder.calls[0].completion is not None
    # The finishing frame says what served it, as chat's final frame does.
    [finishing] = [f for f in frames if any(c["finish_reason"] for c in f["choices"])]
    assert finishing["x_eugene_plexus"]["driver"] == "a"
    assert all("x_eugene_plexus" not in f for f in frames if f is not finishing)


def test_a_stream_without_include_usage_carries_no_usage_frame(settings: Settings) -> None:
    coder = Completer(name="a", model_id="coder")
    coder.usage = Usage(promptTokens=5, completionTokens=3, totalTokens=8)
    with serve(settings, coder) as client:
        response = complete(client, stream=True)
    frames = [
        json.loads(ln[6:])
        for ln in response.text.splitlines()
        if ln.startswith("data: ") and ln != "data: [DONE]"
    ]
    assert all(f["choices"] for f in frames)
