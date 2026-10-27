"""POST /v1/rerank (MR6): a result sorter behind Jina's, Cohere's and TEI's bodies.

A sorter scores each passage a search found against the question. What is not
plumbing, and what most of these tests are about:

* **Failover does not cross models.** Two sorters' scores are not on one scale
  (measured: Qwen3-Reranker answers probabilities, bge-reranker-v2-m3 logits
  from 8.19 to -11.04), so a fallback to a different target would be a 200 that
  mixes them. Replicas of one model balance and fail over, as for embeddings.
* **The answer follows the request's shape**: `documents` is answered as
  `results`, `texts` as TEI's bare array.
* **What `llama-server` does not do is done here**: it refuses objects and
  answers `return_documents` with no documents, so the driver is sent text and
  the passage is put back from the request.
* **A bad request is a 400 that says where**, never FastAPI's 422.

The driver is a fake of its contract (`inference-driver.yaml` `POST
/v1/rerank`), which another component builds against the same text.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from eugene_plexus_gateway._generated.driver_models import (
    Document,
    Locality,
    Problem,
    RerankedPassage,
    RerankPassagesRequest,
    RerankPassagesResponse,
    RerankScale,
    Usage,
)
from eugene_plexus_gateway._generated.models import RerankResponse, TeiRerankResponse
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import DriverError, HttpDriverClient, TieredClient
from eugene_plexus_gateway.routes.inference import create_rerank
from eugene_plexus_gateway.settings import Settings
from tests.conftest import FakeDriverClient, install_snapshot, make_routing_table, runtime_facts
from tests.test_admission import Authority
from tests.test_client_disconnect import _SCOPE, _Hang
from tests.test_local_only import describe_as

IGNORED = "x-eugene-plexus-ignored-settings"
MODEL = "bge-reranker-v2-m3"
QUERY = "What is the capital of France?"

#: The four passages the measurements used, and the logits bge-reranker-v2-m3
#: gave them (brief, 2026-10-12): the best is not the first, and one is negative.
BERLIN = "Berlin is the capital of Germany."
PARIS = "Paris is the capital of France."
BANANAS = "Bananas are rich in potassium."
SEINE = "The capital of France is Paris, on the Seine."
PASSAGES = [BERLIN, PARIS, BANANAS, SEINE]
LOGITS = {BERLIN: -5.45, PARIS: 8.19, BANANAS: -11.04, SEINE: 7.80}
#: Best first, by score and then by index: Paris, the Seine, Berlin, bananas.
BEST_FIRST = [1, 3, 0, 2]
TOKENS = 40


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class Reranker(FakeDriverClient):
    """A fake whose model sorts, as a driver in front of `llama-server
    --reranking` does.

    Each passage's logit comes from `LOGITS` (0 for one it does not know), so a
    test can tell that the answer derives from the request, and `score` is its
    logistic, as the driver's contract says. `topN` is honoured, as it says.
    """

    def __init__(
        self, *, fail: Exception | None = None, sorters: list[str] | None = None, **kw: Any
    ) -> None:
        super().__init__(**kw)
        self.fail = fail
        #: The models that sort, when the driver serves others too (an account).
        self.sorters = sorters
        self.heard: list[RerankPassagesRequest] = []
        self.hook: Any = None
        self.usage = Usage(promptTokens=TOKENS, completionTokens=None, totalTokens=TOKENS)

    def describe(self):  # type: ignore[no-untyped-def]
        info = super().describe()
        for model in info.models or []:
            if self.sorters is None or model.id in self.sorters:
                model.surfaces = ["rerank"]
        return info

    def answer(self, request: RerankPassagesRequest) -> RerankPassagesResponse:
        """What the driver's contract says to answer: scored, best first, by
        index where two are equal, `topN` long."""
        raws = [LOGITS.get(d.root, 0.0) for d in request.documents]
        scored = [
            RerankedPassage(index=i, score=sigmoid(raw), rawScore=raw) for i, raw in enumerate(raws)
        ]
        scored.sort(key=lambda p: (-p.score, p.index))
        if request.topN is not None:
            scored = scored[: request.topN]
        return RerankPassagesResponse(
            results=scored,
            scale=RerankScale.logit,
            modelId=request.model or self.model_id,
            backend=self.backend,
            usage=self.usage,
            latencyMs=1,
        )

    async def rerank(self, request: RerankPassagesRequest) -> RerankPassagesResponse:
        self.heard.append(request)
        if self.hook is not None:
            await self.hook()
        if self.fail is not None:
            raise self.fail
        return self.answer(request)


def sorter(name: str = "sorter", model_id: str = MODEL, **kw: Any) -> Reranker:
    return Reranker(name=name, model_id=model_id, **kw)


def serve(settings: Settings, *drivers: FakeDriverClient, **table: Any) -> TestClient:
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(*drivers, **table)
    return TestClient(app)


#: Leave a field of `_body` out, rather than send it as a JSON null.
OMIT = object()


def _body(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"model": MODEL, "query": QUERY, "documents": PASSAGES, **over}
    return {key: value for key, value in body.items() if value is not OMIT}


def _in(shape: str, passages: list[str] = PASSAGES) -> dict[str, Any]:
    """The passages under `documents` (Jina, Cohere) or `texts` (TEI), and not both."""
    return {"documents": OMIT, "texts": OMIT, shape: passages}


def rerank(client: TestClient, **over: Any) -> httpx.Response:
    return client.post("/v1/rerank", json=_body(**over))


def _rows(client: TestClient) -> list[dict[str, Any]]:
    for _ in range(50):
        rows = client.get("/v1/metrics/requests").json()["requests"]
        if rows:
            return rows
        time.sleep(0.02)
    return []


def _failure(
    status: int, disposition: str | None, detail: str = "the backend said no"
) -> DriverError:
    return DriverError(
        driver_name="a-replica",
        driver_url="http://a-replica",
        status_code=status,
        problem=Problem(
            type="about:blank",
            title="Backend error",
            status=status,
            detail=detail,
            retryDisposition=disposition,
        ),
        raw_body="{}",
    )


# --------------------------------------------------------------------------- #
# the answer, in the shape asked for
# --------------------------------------------------------------------------- #


def test_documents_are_answered_as_a_rerank_response_best_first(settings: Settings) -> None:
    fake = sorter()
    with serve(settings, fake) as client:
        response = rerank(client)
    assert response.status_code == 200, response.text
    body = response.json()
    # The contract's own model reads it, and nothing else is in it.
    RerankResponse.model_validate(body)
    assert set(body) == {"id", "object", "model", "results", "usage", "x_eugene_plexus"}
    assert re.fullmatch(r"rerank-[0-9a-f]+", body["id"])
    assert body["object"] == "list" and body["model"] == MODEL
    assert [r["index"] for r in body["results"]] == BEST_FIRST
    assert [r["relevance_score"] for r in body["results"]] == pytest.approx(
        [sigmoid(8.19), sigmoid(7.80), sigmoid(-5.45), sigmoid(-11.04)]
    )
    # Scores are 0 to 1 whatever the engine's own scale, and the passage is not
    # sent back unless it is asked for (Cohere's default).
    assert all(0 <= r["relevance_score"] <= 1 for r in body["results"])
    assert all(set(r) == {"index", "relevance_score"} for r in body["results"])
    assert body["usage"] == {"prompt_tokens": TOKENS, "total_tokens": TOKENS}
    routing = body["x_eugene_plexus"]
    assert (routing["driver"], routing["attempts"], routing["tier"]) == ("sorter", 1, 1)


def test_texts_are_answered_as_tei_answers_a_bare_array(settings: Settings) -> None:
    with serve(settings, sorter()) as client:
        response = rerank(client, documents=OMIT, texts=PASSAGES)
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body, list)
    TeiRerankResponse.model_validate(body)
    assert [r["index"] for r in body] == BEST_FIRST
    assert all(set(r) == {"index", "score"} for r in body)
    assert [r["score"] for r in body] == pytest.approx(
        [sigmoid(8.19), sigmoid(7.80), sigmoid(-5.45), sigmoid(-11.04)]
    )
    # The array has nowhere to say what served it, so that rides the headers.
    assert response.headers["x-eugene-plexus-driver"] == "sorter"
    assert response.headers["x-eugene-plexus-attempts"] == "1"


def test_the_driver_is_sent_the_question_and_the_passages_as_text(settings: Settings) -> None:
    fake = sorter()
    with serve(settings, fake) as client:
        response = rerank(
            client,
            documents=[BERLIN, {"text": PARIS, "title": "kept by the caller, not read"}, BANANAS],
        )
    assert response.status_code == 200, response.text
    (heard,) = fake.heard
    # Objects reach the backend as their text: llama-server refuses an object.
    assert [d.root for d in heard.documents] == [BERLIN, PARIS, BANANAS]
    assert heard.query == QUERY
    assert heard.model == MODEL
    assert heard.topN is None
    assert heard.localOnly is False
    # The caller's request id is the driver's: one request, one id, end to end.
    assert heard.requestId == response.headers["x-request-id"]
    assert [r["index"] for r in response.json()["results"]] == [1, 0, 2]


@pytest.mark.parametrize("shape", ["documents", "texts"])
def test_a_thousand_passages_are_taken(settings: Settings, shape: str) -> None:
    many = [f"passage {i}" for i in range(1000)]
    fake = sorter()
    with serve(settings, fake) as client:
        response = rerank(client, **_in(shape, many))
    assert response.status_code == 200, response.text
    assert len(fake.heard[0].documents) == 1000


# --------------------------------------------------------------------------- #
# what the engine does not do, done here
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("flag", ["return_documents", "return_text"])
def test_either_flag_puts_the_passage_back_on_a_documents_result(
    settings: Settings, flag: str
) -> None:
    """llama-server answers `return_documents` with no documents (measured on
    b10948 and b11552). Either name asks, as the contract says."""
    with serve(settings, sorter()) as client:
        response = rerank(client, documents=[BERLIN, {"text": PARIS}], **{flag: True})
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    RerankResponse.model_validate(response.json())
    assert [(r["index"], r["document"]) for r in results] == [
        (1, {"text": PARIS}),
        (0, {"text": BERLIN}),
    ]


@pytest.mark.parametrize("flag", ["return_text", "return_documents"])
def test_either_flag_puts_the_passage_back_on_a_tei_result(settings: Settings, flag: str) -> None:
    with serve(settings, sorter()) as client:
        response = rerank(client, documents=OMIT, texts=[BERLIN, PARIS], **{flag: True})
    assert response.status_code == 200, response.text
    assert [(r["index"], r["text"]) for r in response.json()] == [(1, PARIS), (0, BERLIN)]


def test_a_passage_is_not_sent_back_when_neither_flag_is_true(settings: Settings) -> None:
    with serve(settings, sorter()) as client:
        off = rerank(client, return_documents=False, return_text=False)
        null = rerank(client, return_documents=None, return_text=None)
    assert all("document" not in r for r in off.json()["results"])
    assert all("document" not in r for r in null.json()["results"])


@pytest.mark.parametrize("shape", ["documents", "texts"])
def test_raw_scores_are_the_engines_own_numbers_in_either_shape(
    settings: Settings, shape: str
) -> None:
    with serve(settings, sorter()) as client:
        raw = rerank(client, raw_scores=True, **_in(shape))
        scaled = rerank(client, **_in(shape))
    name = "relevance_score" if shape == "documents" else "score"
    rows = raw.json()["results"] if shape == "documents" else raw.json()
    assert [r[name] for r in rows] == pytest.approx([8.19, 7.80, -5.45, -11.04])
    rows = scaled.json()["results"] if shape == "documents" else scaled.json()
    assert [r[name] for r in rows] == pytest.approx(
        [sigmoid(8.19), sigmoid(7.80), sigmoid(-5.45), sigmoid(-11.04)]
    )


def test_top_n_is_how_many_of_the_best_come_back(settings: Settings) -> None:
    fake = sorter()
    with serve(settings, fake) as client:
        two = rerank(client, top_n=2)
        every = rerank(client, top_n=99)
        unasked = rerank(client)
    assert [r["index"] for r in two.json()["results"]] == [1, 3]
    assert len(every.json()["results"]) == len(unasked.json()["results"]) == 4
    # Never more than there are: "all of them" is what a larger number means.
    assert [h.topN for h in fake.heard] == [2, 4, None]


# --------------------------------------------------------------------------- #
# a request that cannot be answered says where
# --------------------------------------------------------------------------- #


def _refused(settings: Settings, **over: Any) -> tuple[httpx.Response, Reranker]:
    fake = sorter()
    with serve(settings, fake) as client:
        return rerank(client, **over), fake


@pytest.mark.parametrize(
    ("label", "over", "param", "says"),
    [
        ("neither", {"documents": OMIT}, "documents", "`documents`"),
        ("both", {"texts": PASSAGES}, "documents", "not both"),
        ("too many", {"documents": ["x"] * 1001}, "documents", "1,001 passages; at most 1,000"),
        ("too many texts", {"documents": OMIT, "texts": ["x"] * 1001}, "texts", "at most 1,000"),
        ("not an array", {"documents": "abc"}, "documents", "must be an array of passages"),
        ("no query", {"query": OMIT}, "query", "is required"),
        ("empty query", {"query": ""}, "query", "must not be empty"),
        ("query a number", {"query": 7}, "query", "this one is a number"),
        ("top_n zero", {"top_n": 0}, "top_n", "at least 1"),
        ("top_n text", {"top_n": "3"}, "top_n", "this one is a string"),
        ("top_n true", {"top_n": True}, "top_n", "this one is a boolean"),
        ("top_n fraction", {"top_n": 1.5}, "top_n", "this one is a number"),
        ("flag text", {"return_documents": "yes"}, "return_documents", "true or false"),
        ("raw_scores number", {"raw_scores": 1}, "raw_scores", "true or false"),
        ("empty model", {"model": ""}, "model", "non-empty string"),
        ("model number", {"model": 7}, "model", "this one is a number"),
    ],
)
def test_a_malformed_request_is_a_400_naming_the_field(
    settings: Settings, label: str, over: dict[str, Any], param: str, says: str
) -> None:
    response, fake = _refused(settings, **over)
    assert response.status_code == 400, (label, response.text)
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error" and error["param"] == param, label
    assert says in error["message"], (label, error["message"])
    assert not fake.heard


@pytest.mark.parametrize(
    ("label", "over", "position", "says"),
    [
        ("a number", {"documents": [BERLIN, PARIS, 7]}, "documents[2]", "this one is a number"),
        ("a list", {"documents": [[BERLIN]]}, "documents[0]", "this one is an array"),
        ("a null", {"documents": [BERLIN, None]}, "documents[1]", "this one is null"),
        (
            "no text",
            {"documents": [BERLIN, {"body": PARIS}]},
            "documents[1]",
            'this one has no "text"',
        ),
        (
            "text a number",
            {"documents": [{"text": 5}, BERLIN]},
            "documents[0]",
            'has a "text" that is a number',
        ),
        (
            "tei object",
            {"documents": OMIT, "texts": [BERLIN, {"text": PARIS}]},
            "texts[1]",
            "must be a string; this one is an object",
        ),
    ],
)
def test_a_bad_passage_is_named_by_its_position(
    settings: Settings, label: str, over: dict[str, Any], position: str, says: str
) -> None:
    """The contract: "a passage that is neither text nor an object with a string
    `text`, which the message names by position". Without the position a caller
    with a thousand passages is left to find the one."""
    response, fake = _refused(settings, **over)
    assert response.status_code == 400, (label, response.text)
    error = response.json()["error"]
    assert error["param"] == position, label
    assert error["message"].startswith(f"{position}: "), (label, error["message"])
    assert says in error["message"], (label, error["message"])
    assert not fake.heard


def test_the_position_is_where_the_passage_is_not_where_it_was_counted_from(
    settings: Settings,
) -> None:
    """Positions are the array's own, from zero, as `index` is on an answer."""
    response, _ = _refused(settings, documents=[BERLIN] * 40 + [{"text": []}] + [BERLIN])
    assert response.json()["error"]["param"] == "documents[40]"


LIMIT = 1_048_576
LIMITED = ["query", "documents string", "documents object", "texts item"]


def _limited(characters: int) -> list[Any]:
    """The four places the contract gives `maxLength: 1048576`: a body that puts
    `characters` in each, and the position a refusal must name."""
    long = "x" * characters
    return [
        pytest.param({"query": long}, "query", id=LIMITED[0]),
        pytest.param({"documents": [BERLIN, long]}, "documents[1]", id=LIMITED[1]),
        pytest.param({"documents": [{"text": long}]}, "documents[0]", id=LIMITED[2]),
        pytest.param(_in("texts", [BERLIN, long]), "texts[1]", id=LIMITED[3]),
    ]


@pytest.mark.parametrize(("over", "position"), _limited(LIMIT + 1))
def test_a_passage_or_query_past_the_limit_is_a_400_naming_where_never_a_422(
    settings: Settings, over: dict[str, Any], position: str
) -> None:
    """The contract takes at most 1,048,576 characters of the query and of one
    passage in any of its four shapes. The body is read by hand, so a longer one
    is the 400 the contract says, in OpenAI's envelope and naming where: never
    FastAPI's 422 `{"detail": [...]}`, and never a validation error raised while
    the request to the driver is built."""
    response, fake = _refused(settings, **over)
    assert response.status_code == 400, (response.status_code, response.text[:300])
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error" and error["param"] == position
    assert f"{LIMIT + 1:,} characters" in error["message"], error["message"]
    assert f"{LIMIT:,}" in error["message"], error["message"]
    assert not fake.heard


@pytest.mark.parametrize(("over", "position"), _limited(LIMIT))
def test_a_passage_or_query_of_exactly_the_limit_is_taken(
    settings: Settings, over: dict[str, Any], position: str
) -> None:
    """The limit is inclusive, as `maxLength` is."""
    fake = sorter()
    with serve(settings, fake) as client:
        response = rerank(client, **over)
    assert response.status_code == 200, (response.status_code, response.text[:300])
    assert len(fake.heard) == 1


def test_a_body_that_is_not_json_or_not_an_object_is_a_400(settings: Settings) -> None:
    with serve(settings, sorter()) as client:
        broken = client.post("/v1/rerank", content=b"{not json")
        listed = client.post("/v1/rerank", json=[1, 2])
    assert broken.status_code == 400 and broken.json()["error"]["param"] == "body"
    assert "must be JSON" in broken.json()["error"]["message"]
    assert listed.status_code == 400 and listed.json()["error"]["param"] == "body"


def test_a_field_this_door_does_not_know_is_ignored(settings: Settings) -> None:
    """As the other OpenAI-shaped bodies' are: `user`, a Jina `task`, anything."""
    with serve(settings, sorter()) as client:
        response = rerank(client, user="person-1", task="retrieval.query", dimensions=3)
    assert response.status_code == 200, response.text
    assert IGNORED not in response.headers


# --------------------------------------------------------------------------- #
# which model: named, or the one this key may use
# --------------------------------------------------------------------------- #


def test_a_left_out_model_is_the_only_sorter(settings: Settings) -> None:
    fake = sorter()
    chat = FakeDriverClient(name="chat", model_id="chatty")
    with serve(settings, chat, fake) as client:
        response = rerank(client, model=OMIT)
    assert response.status_code == 200, response.text
    assert response.json()["model"] == MODEL and fake.heard
    assert not chat.calls


def test_a_null_model_is_a_left_out_model(settings: Settings) -> None:
    """An SDK that sends every field sends `model: null` for TEI's shape."""
    fake = sorter()
    with serve(settings, fake) as client:
        response = rerank(client, model=None)
    assert response.status_code == 200, response.text
    assert response.json()["model"] == MODEL


def test_a_slot_alias_is_not_a_second_sorter(settings: Settings) -> None:
    """`best -> [bge]` would otherwise make the one sorter look like two, and a
    left-out model would be refused (found by moderation's run, P6-1)."""
    fake = sorter()
    with serve(settings, fake, slots=[{"model": "best", "targets": [MODEL]}]) as client:
        response = rerank(client, model=OMIT)
    assert response.status_code == 200, response.text
    assert response.json()["model"] == MODEL


def test_a_left_out_model_with_none_served_is_a_404_in_plain_words(settings: Settings) -> None:
    chat = FakeDriverClient(name="chat", model_id="chatty")
    with serve(settings, chat) as client:
        response = rerank(client, model=OMIT)
    assert response.status_code == 404, response.text
    error = response.json()["error"]
    assert error["param"] == "model"
    assert "no result sorter" in error["message"] and "this key" in error["message"]
    assert not chat.calls


def test_a_left_out_model_with_two_served_is_a_400_naming_them(settings: Settings) -> None:
    one, two = sorter("a", "sorter-a"), sorter("b", "sorter-b")
    with serve(settings, one, two) as client:
        response = rerank(client, model=OMIT)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == "model"
    assert "sorter-a, sorter-b" in error["message"]
    assert not one.heard and not two.heard


def test_a_model_that_does_not_sort_is_sent_to_its_own_door(settings: Settings) -> None:
    """The embeddings door's wording: what the model serves, and where to go."""
    chat = FakeDriverClient(name="chat", model_id="chatty")
    embedder = FakeDriverClient(name="embedder", model_id="nomic", supports_embeddings=True)
    with serve(settings, chat, embedder, sorter()) as client:
        on_chat = rerank(client, model="chatty")
        on_embedder = rerank(client, model="nomic")
    for response, serves, door in (
        (on_chat, "chat", "/v1/chat/completions"),
        (on_embedder, "embeddings", "/v1/embeddings"),
    ):
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["param"] == "model"
        assert f"serves {serves} and not rerank" in error["message"]
        assert door in error["message"]
    assert not chat.calls


def test_a_sorter_is_sent_to_the_rerank_door_from_the_other_two(settings: Settings) -> None:
    """`_DOORS` names it: without the entry the chat door would send a caller to
    `/v1/embeddings`, the only door it knew to offer."""
    with serve(settings, sorter()) as client:
        chat = client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )
        embeddings = client.post("/v1/embeddings", json={"model": MODEL, "input": "a"})
    for response in (chat, embeddings):
        assert response.status_code == 400, response.text
        assert "serves rerank" in response.json()["error"]["message"]
        assert "/v1/rerank" in response.json()["error"]["message"]


def test_an_unknown_model_is_a_404(settings: Settings) -> None:
    fake = sorter()
    with serve(settings, fake) as client:
        response = rerank(client, model="no-such-model")
    assert response.status_code == 404, response.text
    assert "no-such-model" in response.json()["error"]["message"]
    assert not fake.heard


def test_a_provider_accounts_model_is_asked_for_by_its_own_id(settings: Settings) -> None:
    """The driver is sent `cohere/rerank-v3.5`; the caller sees the public id."""
    account = Reranker(
        name="openrouter",
        models=["mistralai/mistral-nemo", "cohere/rerank-v3.5"],
        sorters=["cohere/rerank-v3.5"],
        account=True,
    )
    with serve(settings, account) as client:
        response = rerank(client, model="openrouter/cohere/rerank-v3.5")
    assert response.status_code == 200, response.text
    assert account.heard[0].model == "cohere/rerank-v3.5"
    assert response.json()["model"] == "openrouter/cohere/rerank-v3.5"


def test_the_listing_says_a_sorter_sorts_and_nothing_else(settings: Settings) -> None:
    chat = FakeDriverClient(name="chat", model_id="chatty")
    with serve(settings, chat, sorter()) as client:
        listed = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
        found = client.get(f"/v1/models/{MODEL}")
    assert listed[MODEL]["x_eugene_plexus"]["surfaces"] == ["rerank"]
    assert listed["chatty"]["x_eugene_plexus"]["surfaces"] == ["chat"]
    assert found.status_code == 200 and found.json() == listed[MODEL]


# --------------------------------------------------------------------------- #
# an empty list reaches no model
# --------------------------------------------------------------------------- #


def test_an_empty_documents_list_is_answered_with_no_results_and_no_call(
    settings: Settings,
) -> None:
    fake = sorter()
    with serve(settings, fake) as client:
        response = rerank(client, documents=[])
    assert response.status_code == 200, response.text
    body = response.json()
    RerankResponse.model_validate(body)
    assert body["results"] == [] and body["object"] == "list" and body["model"] == MODEL
    assert re.fullmatch(r"rerank-[0-9a-f]+", body["id"])
    assert "x_eugene_plexus" not in body and "usage" not in body
    assert not fake.heard


def test_an_empty_texts_list_is_answered_with_an_empty_array(settings: Settings) -> None:
    fake = sorter()
    with serve(settings, fake) as client:
        response = rerank(client, documents=OMIT, texts=[])
    assert response.status_code == 200 and response.json() == []
    assert not fake.heard


def test_an_empty_list_is_still_refused_for_a_model_that_is_not_there_or_not_a_sorter(
    settings: Settings,
) -> None:
    """After the model resolves: an unknown model is still a 404, and a chat
    model is still a 400 -- the caller learns of the mistake without sending
    any passages."""
    chat = FakeDriverClient(name="chat", model_id="chatty")
    with serve(settings, chat, sorter()) as client:
        unknown = rerank(client, model="no-such-model", documents=[])
        wrong = rerank(client, model="chatty", documents=[])
    assert unknown.status_code == 404, unknown.text
    assert wrong.status_code == 400, wrong.text


def test_an_empty_list_for_a_slot_alias_names_the_model_that_would_have_scored(
    settings: Settings,
) -> None:
    first, second = sorter("a", "sorter-a"), sorter("b", "sorter-b")
    slots = [{"model": "best", "targets": ["sorter-a", "sorter-b"]}]
    with serve(settings, first, second, slots=slots) as client:
        response = rerank(client, model="best", documents=[])
    assert response.status_code == 200, response.text
    assert response.json()["model"] == "sorter-a"


def test_an_empty_list_is_answered_while_the_sorter_is_asleep(settings: Settings) -> None:
    """Nothing is sent, so nothing needs to be ready; a request with passages
    meets the 503 (and no wake, as for embeddings)."""
    asleep = sorter(runtime="asleep")
    with serve(settings, asleep, runtimes=[runtime_facts("asleep", status="stopped")]) as client:
        empty = rerank(client, documents=[])
        passages = rerank(client)
    assert empty.status_code == 200, empty.text
    assert passages.status_code == 503, passages.text
    assert not asleep.heard


# --------------------------------------------------------------------------- #
# what it accepts and does not honour, and says so
# --------------------------------------------------------------------------- #

SENT = {
    "truncate": True,
    "truncation_direction": "left",
    "max_chunks_per_doc": 3,
    "max_tokens_per_doc": 512,
    "rank_fields": ["text"],
}
ORDER = "truncate, truncation_direction, max_chunks_per_doc, max_tokens_per_doc, rank_fields"


def test_the_five_ignored_settings_are_named_in_one_stable_order(settings: Settings) -> None:
    """The order is the door's, not the request's: sent here back to front."""
    backwards = dict(reversed(list(SENT.items())))
    with serve(settings, sorter()) as client:
        response = rerank(client, **backwards)
    assert response.status_code == 200, response.text
    assert response.headers[IGNORED] == ORDER


@pytest.mark.parametrize(
    "name", ["truncate", "max_chunks_per_doc", "max_tokens_per_doc", "rank_fields"]
)
def test_each_ignored_setting_that_stands_alone_is_named_alone(
    settings: Settings, name: str
) -> None:
    """`truncation_direction` does not stand alone: it is named beside a
    `truncate` that asks for a cut, and the test after next says when."""
    with serve(settings, sorter()) as client:
        response = rerank(client, **{name: SENT[name]})
    assert response.headers[IGNORED] == name


def test_truncate_false_is_what_the_door_does_and_is_not_named(settings: Settings) -> None:
    """TEI's default, `truncate: false`, asks for a passage longer than the model
    reads to be refused rather than cut, which is what this door does, so there
    is nothing ignored to name. `truncate: true` asks for a cut that is not made:
    named, and the passage reaches the driver whole."""
    long = "a very long passage " * 400
    fake = sorter()
    with serve(settings, fake) as client:
        refused_not_cut = rerank(client, documents=[BERLIN, long], truncate=False)
        asked_to_cut = rerank(client, documents=[BERLIN, long], truncate=True)
    assert refused_not_cut.status_code == 200 and IGNORED not in refused_not_cut.headers
    assert asked_to_cut.status_code == 200 and asked_to_cut.headers[IGNORED] == "truncate"
    # Never cut, whichever way it was asked.
    assert [[d.root for d in heard.documents] for heard in fake.heard] == [[BERLIN, long]] * 2


@pytest.mark.parametrize(
    ("over", "named"),
    [
        ({"truncation_direction": "left"}, None),
        ({"truncation_direction": "left", "truncate": False}, None),
        ({"truncation_direction": "left", "truncate": None}, None),
        ({"truncation_direction": "left", "truncate": True}, "truncate, truncation_direction"),
        ({"truncation_direction": None, "truncate": True}, "truncate"),
    ],
    ids=["alone", "beside false", "beside null", "beside true", "null beside true"],
)
def test_a_truncation_direction_is_named_only_beside_a_truncate_that_asks_for_a_cut(
    settings: Settings, over: dict[str, Any], named: str | None
) -> None:
    """It says which end to cut, so it means something only when a cut is asked
    for. Without that nothing is cut, and there is no direction to ignore."""
    with serve(settings, sorter()) as client:
        response = rerank(client, **over)
    assert response.status_code == 200, response.text
    assert response.headers.get(IGNORED) == named


@pytest.mark.parametrize("shape", ["documents", "texts"])
def test_a_tei_clients_own_defaults_name_nothing(settings: Settings, shape: str) -> None:
    """A client that sends TEI's defaults explicitly (`truncate: false`,
    `truncation_direction: "right"`, `raw_scores: false`, `return_text: false`)
    asks for nothing this door does not do. A header on every one of its answers
    would teach it to ignore the header."""
    with serve(settings, sorter()) as client:
        response = rerank(
            client,
            truncate=False,
            truncation_direction="right",
            raw_scores=False,
            return_text=False,
            **_in(shape),
        )
    assert response.status_code == 200, response.text
    assert IGNORED not in response.headers


@pytest.mark.parametrize("shape", ["documents", "texts"])
def test_the_header_rides_either_shape(settings: Settings, shape: str) -> None:
    with serve(settings, sorter()) as client:
        response = rerank(client, truncate=True, **_in(shape))
    assert response.headers[IGNORED] == "truncate"


def test_nothing_is_named_when_none_is_sent_or_each_is_null(settings: Settings) -> None:
    """A header on every answer would teach callers to ignore it. What is
    honoured (`top_n`, `return_documents`, `return_text`, `raw_scores`) is never
    named, and a JSON null is not a setting."""
    with serve(settings, sorter()) as client:
        plain = rerank(client)
        honoured = rerank(client, top_n=2, return_documents=True, return_text=True, raw_scores=True)
        nulls = rerank(client, **{name: None for name in SENT})
    for response in (plain, honoured, nulls):
        assert response.status_code == 200, response.text
        assert IGNORED not in response.headers


def test_an_empty_answer_names_them_too(settings: Settings) -> None:
    with serve(settings, sorter()) as client:
        response = rerank(client, documents=[], max_chunks_per_doc=2)
    assert response.status_code == 200 and response.headers[IGNORED] == "max_chunks_per_doc"


def test_a_refused_or_failed_request_names_nothing(settings: Settings) -> None:
    """The header describes an answer that was served."""
    dead = sorter(fail=httpx.ConnectError("refused"))
    with serve(settings, dead) as client:
        malformed = rerank(client, documents=OMIT, **SENT)
        unknown = rerank(client, model="no-such-model", **SENT)
        failed = rerank(client, **SENT)
    assert malformed.status_code == 400 and unknown.status_code == 404
    assert failed.status_code == 502
    for response in (malformed, unknown, failed):
        assert IGNORED not in response.headers


def test_none_of_them_reaches_the_backend(settings: Settings) -> None:
    """Honest only if nothing here quietly acts on them: the driver's request
    has no field for any, and a passage too long for the model is a 400 naming
    the limit, not a cut one."""
    fake = sorter()
    with serve(settings, fake) as client:
        rerank(client, **SENT)
    sent = fake.heard[0].model_dump(mode="json", exclude_none=True)
    assert not set(SENT) & set(sent)


# --------------------------------------------------------------------------- #
# THE RULE: one model, replicas only
# --------------------------------------------------------------------------- #


def test_a_failure_does_not_cascade_to_a_different_model(settings: Settings) -> None:
    """**The test this door exists to pass.** Tier 1 is a dead sorter, tier 2 a
    healthy one for a DIFFERENT model. For chat that cascades and is right; here
    a 200 carrying tier 2's scores would sit beside scores on another scale."""
    dead = sorter("primary", "sorter-a", fail=_failure(503, "safe"))
    other = sorter("secondary", "sorter-b")
    slots = [{"model": "sorter-a", "targets": ["sorter-a", "sorter-b"]}]
    with serve(settings, dead, other, slots=slots) as client:
        response = rerank(client, model="sorter-a")
    assert response.status_code != 200, "it sorted with the wrong model"
    assert len(dead.heard) == 1
    assert not other.heard, "the other model was contacted at all"


def test_a_slot_alias_sorts_with_its_first_model_and_never_its_second(settings: Settings) -> None:
    first, second = sorter("primary", "sorter-a"), sorter("secondary", "sorter-b")
    slots = [{"model": "best", "targets": ["sorter-a", "sorter-b"]}]
    with serve(settings, first, second, slots=slots) as client:
        response = rerank(client, model="best")
    assert response.status_code == 200, response.text
    assert len(first.heard) == 1 and not second.heard
    assert response.json()["model"] == "sorter-a"


def test_a_slot_aliass_first_model_down_is_not_replaced_by_its_second(settings: Settings) -> None:
    """An alias is its first target, and a first model that is down is down.
    Unready, not failing: `pick_rerank` finds nothing and nobody is asked."""
    first = sorter("primary", "sorter-a", runtime="asleep")
    second = sorter("secondary", "sorter-b")
    slots = [{"model": "best", "targets": ["sorter-a", "sorter-b"]}]
    runtimes = [runtime_facts("asleep", status="stopped")]
    with serve(settings, first, second, slots=slots, runtimes=runtimes) as client:
        response = rerank(client, model="best")
    assert response.status_code == 503, response.text
    assert not first.heard and not second.heard, "it sorted with the wrong model"


def test_pick_rerank_holds_one_tier_of_the_first_models_replicas() -> None:
    """The structural half: the client has no second tier to cascade into, so
    the rule cannot be lost by editing the cascade."""
    a1, a2 = sorter("a1", "sorter-a"), sorter("a2", "sorter-a")
    b = sorter("b", "sorter-b")
    chat = FakeDriverClient(name="chat", model_id="chatty")
    slots = [{"model": "best", "targets": ["sorter-a", "sorter-b"]}]
    table = make_routing_table(a1, a2, b, chat, slots=slots)
    client = table.pick_rerank(table.resolve("best"))
    assert isinstance(client, TieredClient)
    assert len(client._tiers) == 1
    assert sorted(c.name for c in client.candidates) == ["a1", "a2"]
    # A model that does not sort has no sorter's client at all.
    assert table.pick_rerank(table.resolve("chatty")) is None


def test_pick_rerank_finds_nothing_when_the_first_model_is_not_ready() -> None:
    first = sorter("a", "sorter-a", runtime="asleep")
    second = sorter("b", "sorter-b")
    slots = [{"model": "best", "targets": ["sorter-a", "sorter-b"]}]
    table = make_routing_table(
        first, second, slots=slots, runtimes=[runtime_facts("asleep", status="stopped")]
    )
    assert table.pick_rerank(table.resolve("best")) is None
    assert table.pick_rerank(table.resolve("sorter-b")) is not None


@pytest.mark.parametrize(
    ("first_fails", "cascades", "status"),
    [
        (_failure(503, "safe"), True, 200),
        (_failure(502, "safe"), True, 200),
        (httpx.ConnectError("refused"), True, 200),
        (_failure(400, "terminal", "a passage is 9000 tokens; this model takes 8192"), False, 400),
        (_failure(502, "indeterminate"), False, 502),
        (_failure(504, None), False, 504),
        (httpx.ReadTimeout("slow"), False, 504),
    ],
    ids=["503", "502", "connect", "400", "502-unsure", "504", "timeout"],
)
def test_replicas_of_the_same_model_fail_over_only_when_it_is_safe(
    settings: Settings, first_fails: Exception, cascades: bool, status: int
) -> None:
    """A 5xx the driver calls safe, and a connection that never opened, move to
    the next replica of the same model. A 4xx would be the same on every
    replica, work that may have happened is not replayed, and a replica would
    take as long as this one did."""
    first = sorter("a-replica", fail=first_fails)
    second = sorter("b-replica")
    with serve(settings, first, second) as client:
        response = rerank(client)
    assert response.status_code == status, response.text
    assert len(first.heard) == 1
    assert len(second.heard) == (1 if cascades else 0)
    if cascades:
        assert response.json()["x_eugene_plexus"]["attempts"] == 2
        assert response.json()["x_eugene_plexus"]["driver"] == "b-replica"


def test_a_backend_that_rejects_the_request_says_why_in_its_own_words(settings: Settings) -> None:
    """A passage longer than the model takes: the driver names the limit and
    the gateway carries it to the caller as the 400 it is."""
    first = sorter(
        fail=_failure(400, "terminal", "passage 2 is 9000 tokens; this model takes at most 8192")
    )
    with serve(settings, first) as client:
        response = rerank(client)
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert "at most 8192" in error["message"]


def test_every_replica_failing_is_a_502_that_names_the_model(settings: Settings) -> None:
    first = sorter("a-replica", fail=httpx.ConnectError("refused"))
    second = sorter("b-replica", fail=httpx.ConnectError("refused"))
    with serve(settings, first, second) as client:
        response = rerank(client)
    assert response.status_code == 502, response.text
    assert MODEL in response.json()["error"]["message"]
    assert len(first.heard) == len(second.heard) == 1


# --------------------------------------------------------------------------- #
# a closed tab
# --------------------------------------------------------------------------- #


async def test_a_closed_tab_cancels_the_rerank(settings: Settings) -> None:
    hang = _Hang()
    fake = sorter()
    fake.hook = hang.run
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake)
    rows: list[Any] = []
    app.state.metrics = SimpleNamespace(record=rows.append)
    gone = asyncio.Event()
    payload = json.dumps(_body()).encode()
    delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    scope = _SCOPE | {"app": app, "path": "/v1/rerank", "raw_path": b"/v1/rerank"}
    route = asyncio.ensure_future(create_rerank(Request(scope, receive)))
    await asyncio.wait_for(hang.entered.wait(), timeout=5)
    gone.set()
    answer = await asyncio.wait_for(route, timeout=2)
    assert hang.cancelled and not hang.finished
    assert answer.status_code == 499
    # A caller that left is not a backend that failed, but the row is kept.
    assert [(r.door, r.outcome) for r in rows] == [("rerank", "error")]


# --------------------------------------------------------------------------- #
# the key: allowed models, limits, and a row of its own
# --------------------------------------------------------------------------- #


@pytest.fixture
def keyed(settings: Settings, install: Any) -> Any:
    """Two sorters and a chat model behind a client key whose authority is scripted."""
    authority = Authority()
    token = install.client_key(name="Original name", jti="key-1")
    app = create_app(settings=settings)
    app.state.auth_state = install.auth_state()
    app.state.client_key_guard = authority.as_guard(ttl_seconds=0)
    a, b = sorter("a-driver", "sorter-a"), sorter("b-driver", "sorter-b")
    chat = FakeDriverClient(name="c-driver", model_id="chatty")
    app.state.routing = make_routing_table(a, b, chat)
    headers = {"Authorization": "Bearer " + token}
    operator = {"Authorization": "Bearer " + install.session(sub="operator")}
    return app, authority, a, b, chat, headers, operator


def test_a_scoped_key_sorts_only_with_the_models_it_may_use(keyed: Any) -> None:
    app, authority, a, b, _chat, headers, _operator = keyed
    authority.allowed = ["sorter-a"]
    with TestClient(app) as client:
        allowed = client.post("/v1/rerank", json=_body(model="sorter-a"), headers=headers)
        refused = client.post("/v1/rerank", json=_body(model="sorter-b"), headers=headers)
    assert allowed.status_code == 200, allowed.text
    assert refused.status_code == 404, refused.text
    assert "sorter-b" not in refused.text
    assert len(a.heard) == 1 and not b.heard
    # The reservation: checked on arrival and acquired as it leaves its line
    # (row 11) for the model named, renewed before the one attempt, and
    # released; the refused model is acquired and released and never attempted.
    assert [(c["action"], c.get("model")) for c in authority.calls] == [
        ("check", "sorter-a"),
        ("acquire", "sorter-a"),
        ("renew", "sorter-a"),
        ("release", "sorter-a"),
        ("check", "sorter-b"),
        ("acquire", "sorter-b"),
        ("release", "sorter-b"),
    ]


def test_a_left_out_model_is_chosen_among_those_the_key_may_use(keyed: Any) -> None:
    app, authority, a, b, _chat, headers, _operator = keyed
    with TestClient(app) as client:
        authority.allowed = ["sorter-a"]
        one = client.post("/v1/rerank", json=_body(model=OMIT), headers=headers)
        authority.allowed = ["chatty"]
        none = client.post("/v1/rerank", json=_body(model=OMIT), headers=headers)
        authority.allowed = ["sorter-*"]
        two = client.post("/v1/rerank", json=_body(model=OMIT), headers=headers)
    assert one.status_code == 200 and one.json()["model"] == "sorter-a"
    assert none.status_code == 404
    assert "no result sorter" in none.json()["error"]["message"]
    assert two.status_code == 400
    assert "sorter-a, sorter-b" in two.json()["error"]["message"]
    assert len(a.heard) == 1 and not b.heard


def test_a_refused_reservation_precedes_the_backend(keyed: Any) -> None:
    """The key's rate and concurrency limits, as embeddings has them: refused
    before anything is sent, with the authority's own Retry-After."""
    app, authority, a, _b, _chat, headers, operator = keyed
    authority.allowed = None
    authority.refuse_acquire = 429
    with TestClient(app) as client:
        response = client.post("/v1/rerank", json=_body(model="sorter-a"), headers=headers)
        assert response.status_code == 429 and response.headers["Retry-After"] == "7"
        assert not a.heard
        assert client.get("/v1/config", headers=operator).status_code == 200


def test_a_request_is_recorded_for_its_key(keyed: Any) -> None:
    app, authority, _a, _b, _chat, headers, operator = keyed
    authority.allowed = None
    with TestClient(app) as client:
        response = client.post("/v1/rerank", json=_body(model="sorter-a"), headers=headers)
        assert response.status_code == 200, response.text
        for _ in range(100):
            usage = client.get("/v1/metrics/clients", headers=operator).json()
            if usage["clients"]:
                break
            time.sleep(0.01)
        row = usage["clients"][0]
        history = client.get("/v1/metrics/requests", headers=operator).json()["requests"]
    assert row["clientKeyId"] == "key-1" and row["clientKeyName"] == "Verified app"
    assert row["requests"] == row["served"] == row["attempts"] == 1
    # Tokens read, none written: a rerank is not a request whose usage is unknown.
    assert (row["promptTokens"], row["completionTokens"]) == (TOKENS, 0)
    assert row["failed"] == row["incompleteUsageRequests"] == 0
    assert history[0]["clientKeyId"] == "key-1"
    assert QUERY not in json.dumps(history) + json.dumps(usage)


def test_a_local_only_key_sends_the_driver_the_policy(keyed: Any) -> None:
    app, authority, a, _b, _chat, headers, _operator = keyed
    authority.allowed = None
    authority.local_only = True
    info = a.describe().model_copy(
        update={"locality": Locality("local"), "localOnlyEnforced": True}
    )
    a.describe = lambda: info  # type: ignore[method-assign]
    install_snapshot(app.state.routing, a)
    with TestClient(app) as client:
        response = client.post("/v1/rerank", json=_body(model="sorter-a"), headers=headers)
    assert response.status_code == 200, response.text
    assert a.heard[0].localOnly is True


def test_a_local_only_key_chooses_among_the_local_sorters_only(keyed: Any) -> None:
    """A left-out model is the one sorter this key may use, and a key that
    requires local inference may not use one that is somebody else's server."""
    app, authority, a, b, _chat, headers, _operator = keyed
    authority.allowed = None
    authority.local_only = True
    describe_as(a, "local")
    describe_as(b, "external")
    install_snapshot(app.state.routing, a, b)
    with TestClient(app) as client:
        left_out = client.post("/v1/rerank", json=_body(model=OMIT), headers=headers)
        named = client.post("/v1/rerank", json=_body(model="sorter-b"), headers=headers)
    assert left_out.status_code == 200, left_out.text
    assert left_out.json()["model"] == "sorter-a"
    assert named.status_code == 403 and "local-only" in named.text
    assert not b.heard


def test_the_total_deadline_cancels_the_sort_and_frees_the_key(keyed: Any) -> None:
    """As embeddings has it (test_request_lifetime): the call in flight is
    cancelled, a replica is not asked to do it again, and the reservation is
    released."""
    app, authority, a, _b, _chat, headers, _operator = keyed
    authority.allowed = None
    replica = sorter("a2-driver", "sorter-a")
    install_snapshot(app.state.routing, a, replica)
    cancelled: list[bool] = []

    async def hang() -> None:
        try:
            await asyncio.sleep(30)
        finally:
            cancelled.append(True)

    a.hook = hang
    with TestClient(app) as client:
        original = app.state.config_store.get
        # 0.5 s, not 0.08: the budget has to outlast the route's own preparation
        # on a slow runner, and the backend still hangs for 30.
        app.state.config_store.get = lambda key: (
            0.5 if key == "requestTimeoutSeconds" else original(key)
        )
        before = time.perf_counter()
        response = client.post("/v1/rerank", json=_body(model="sorter-a"), headers=headers)
        took = time.perf_counter() - before
    assert took < 2
    assert response.status_code == 504 and "total request deadline" in response.text
    assert cancelled and not replica.heard
    assert authority.calls[-1]["action"] == "release"
    assert response.headers["x-request-id"]


# --------------------------------------------------------------------------- #
# the metrics row: its own door
# --------------------------------------------------------------------------- #


def test_a_served_request_shows_in_the_metrics_under_its_own_door(settings: Settings) -> None:
    with serve(settings, sorter()) as client:
        assert rerank(client).status_code == 200
        rows = _rows(client)
    (row,) = rows
    assert (row["door"], row["requestedModel"], row["servedModel"]) == ("rerank", MODEL, MODEL)
    assert (row["outcome"], row["attempts"], row["tier"]) == ("served", 1, 1)
    assert (row["promptTokens"], row["completionTokens"]) == (TOKENS, 0)
    assert row["tries"][0]["driver"] == "sorter" and row["tries"][0]["served"] is True
    # The driver's own measurement, on the attempt that produced it.
    assert row["tries"][0]["backendMs"] == 1


def test_a_failover_is_one_row_with_both_attempts(settings: Settings) -> None:
    first = sorter("a-replica", fail=httpx.ConnectError("refused"))
    second = sorter("b-replica")
    with serve(settings, first, second) as client:
        assert rerank(client).status_code == 200
        (row,) = _rows(client)
    assert (row["door"], row["attempts"], row["outcome"]) == ("rerank", 2, "served")
    assert [t["driver"] for t in row["tries"]] == ["a-replica", "b-replica"]


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("refused"),
        _failure(400, "terminal"),
        _failure(503, "safe"),
        httpx.ReadTimeout("slow"),
    ],
    ids=["connect", "driver-400", "driver-503", "timeout"],
)
def test_a_failed_request_shows_as_an_error_under_its_own_door(
    settings: Settings, failure: Exception
) -> None:
    """Every way a call can fail is recorded by the route, with the door: the
    middleware's fallback row knows no door, so a branch that forgot to record
    would still leave a row, and only the missing name would say so."""
    with serve(settings, sorter(fail=failure)) as client:
        assert rerank(client).status_code >= 400
        (row,) = _rows(client)
    assert (row["door"], row["outcome"], row["servedModel"]) == ("rerank", "error", None)


def test_an_empty_request_leaves_no_row_and_no_false_error(settings: Settings) -> None:
    """It reached no model, so there is nothing to retain: the middleware's
    fallback would have recorded it as an error with no served model."""
    with serve(settings, sorter()) as client:
        assert rerank(client, documents=[]).status_code == 200
        assert rerank(client).status_code == 200
        rows = _rows(client)
    assert [(r["door"], r["outcome"]) for r in rows] == [("rerank", "served")]


# --------------------------------------------------------------------------- #
# the driver's wire
# --------------------------------------------------------------------------- #


def _wired(handler: Any) -> HttpDriverClient:
    client = HttpDriverClient(name="d", base_url="http://d.invalid")
    client._client = httpx.AsyncClient(
        base_url=client.base_url, transport=httpx.MockTransport(handler)
    )
    return client


def _asked(**over: Any) -> RerankPassagesRequest:
    fields: dict[str, Any] = {
        "model": "m",
        "query": "q",
        "documents": [Document("a"), Document("b")],
        "topN": 1,
        "requestId": "req-1",
        **over,
    }
    return RerankPassagesRequest(**fields)


async def test_the_driver_client_posts_the_contracts_body_and_reads_its_answer() -> None:
    seen: dict[str, Any] = {}

    def wire(request: httpx.Request) -> httpx.Response:
        seen.update(path=request.url.path, body=json.loads(request.content))
        seen["id"] = request.headers.get("x-request-id")
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "score": 0.88, "rawScore": 2.0},
                    {"index": 0, "score": 0.12, "rawScore": -2.0},
                ],
                "scale": "logit",
                "modelId": "m",
                "usage": {"promptTokens": 9, "totalTokens": 9},
                "latencyMs": 3,
            },
        )

    answer = await _wired(wire).rerank(_asked())
    assert seen["path"] == "/v1/rerank" and seen["id"] == "req-1"
    assert seen["body"] == {
        "model": "m",
        "localOnly": False,
        "query": "q",
        "documents": ["a", "b"],
        "topN": 1,
        "requestId": "req-1",
    }
    assert [(r.index, r.score, r.rawScore) for r in answer.results] == [
        (1, 0.88, 2.0),
        (0, 0.12, -2.0),
    ]
    assert answer.scale is RerankScale.logit and answer.usage is not None


async def test_the_driver_clients_refusal_carries_the_drivers_problem() -> None:
    body = {
        "type": "about:blank",
        "title": "Backend rejected the request",
        "status": 400,
        "detail": "passage 1 is 9000 tokens; this model takes at most 8192",
        "retryDisposition": "terminal",
    }

    def wire(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"detail": body})

    with pytest.raises(DriverError) as caught:
        await _wired(wire).rerank(_asked())
    assert caught.value.status_code == 400
    assert caught.value.problem is not None and "at most 8192" in str(caught.value.problem.detail)


@pytest.mark.parametrize(
    "answer",
    [
        # A score for a passage nobody sent.
        {"results": [{"index": 2, "score": 0.5, "rawScore": 0.0}], "scale": "logit"},
        # A "0 to 1" score that is not.
        {"results": [{"index": 0, "score": 1.5, "rawScore": 1.5}], "scale": "probability"},
        # Not the contract's shape at all.
        {"scores": [0.5]},
    ],
    ids=["a passage not sent", "score outside 0 to 1", "wrong shape"],
)
async def test_an_answer_outside_the_contract_is_an_invalid_reply_not_a_500(
    answer: dict[str, Any],
) -> None:
    def wire(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=answer)

    with pytest.raises(DriverError) as caught:
        await _wired(wire).rerank(_asked())
    assert caught.value.status_code == 502
    assert caught.value.problem is not None
    assert caught.value.problem.title == "Driver returned an invalid response"


class OverTheWire(Reranker):
    """The same sorter, answering through the driver's HTTP contract: the route,
    the tiered and bound clients and `HttpDriverClient` are the real ones, and
    only the socket is a `MockTransport`.

    A component test is not a wiring test. The tests above prove the route
    against a fake client, and the ones just above prove the client against a
    canned answer; neither shows that what the route builds is what the client
    sends."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.posted: list[dict[str, Any]] = []
        self.wire = _wired(self._driver)

    def _driver(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.posted.append(body)
        # The driver reads its request with its own contract's model.
        asked = RerankPassagesRequest.model_validate(body)
        answer = self.answer(asked).model_dump(mode="json", exclude_none=True)
        return httpx.Response(200, json=answer)

    async def rerank(self, request: RerankPassagesRequest) -> RerankPassagesResponse:
        self.heard.append(request)
        return await self.wire.rerank(request)


def test_the_whole_path_works_over_the_drivers_wire(settings: Settings) -> None:
    fake = OverTheWire(name="sorter", model_id=MODEL)
    with serve(settings, fake) as client:
        asked = rerank(client, top_n=2, return_documents=True, truncate=True)
        plain = rerank(client, documents=OMIT, texts=PASSAGES)
    assert asked.status_code == 200, asked.text
    results = asked.json()["results"]
    assert [(r["index"], r["document"]["text"]) for r in results] == [(1, PARIS), (3, SEINE)]
    assert asked.headers[IGNORED] == "truncate"
    assert [r["index"] for r in plain.json()] == BEST_FIRST
    # What the driver received: the contract's body, the model as the driver
    # knows it, and nothing it was not asked for (no nulls, no ignored setting).
    common = {"model": MODEL, "localOnly": False, "query": QUERY, "documents": PASSAGES}
    assert fake.posted == [
        {**common, "topN": 2, "requestId": asked.headers["x-request-id"]},
        {**common, "requestId": plain.headers["x-request-id"]},
    ]
