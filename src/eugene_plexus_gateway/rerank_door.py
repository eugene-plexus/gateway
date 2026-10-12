"""/v1/rerank (MR6, 2026-10-12): the three public bodies, read and checked.

A result sorter scores each passage a search found against the question.
Jina's and Cohere's body names the passages `documents` (strings, or objects
whose `text` is the passage); Hugging Face TEI's names them `texts`. They are
one request here -- a question and a list of passages -- and the answer
follows the shape the caller asked in: `results` for `documents`, TEI's bare
array for `texts`.

**Read by hand, not by the generated `RerankRequest`.** The contract says a
bad request is a 400 that names the passage by position, and a body typed
`string | object` would answer FastAPI's 422 `{"detail": [...]}`, which no
Cohere or TEI client can read. So this module does what `moderation_door`
does: it reads the JSON itself and refuses in OpenAI's envelope, in words that
say which field and what was found there.

Fields this door does not know are ignored, as the other OpenAI-shaped bodies'
are. What it knows and does not honour is named, in one order, on the answer
(`IGNORED`, and `carries` for when a setting counts as sent).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ._generated.driver_models import RerankedPassage, Usage

#: Cohere's limit, and the driver's (`RerankPassagesRequest.documents`).
MAX_PASSAGES = 1000

#: What the contract takes of the query and of one passage (`maxLength`, the
#: driver's own limit). A longer one is a 400 that names its position: the body
#: is read by hand here, so it is never a 422, and never a validation error
#: raised while the request to the driver is built.
MAX_CHARACTERS = 1_048_576

#: What a rerank body may carry that this door accepts and does not act on, in
#: the order they are named on `x-eugene-plexus-ignored-settings`. None reaches
#: a backend: the driver's `RerankPassagesRequest` has no field for any of
#: them. Which of them a request counts as carrying is `carries`.
IGNORED = (
    "truncate",
    "truncation_direction",
    "max_chunks_per_doc",
    "max_tokens_per_doc",
    "rank_fields",
)

Shape = Literal["documents", "texts"]


class Refusal(ValueError):
    """A request this door refuses, naming the field."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = f"{field}: {message}"


@dataclass(frozen=True)
class RerankAsk:
    """The request, read. `model` is the caller's, or None when it was left out;
    the route then fills it in with the one sorter this key may use."""

    model: str | None
    query: str
    #: The passages as text, in the order sent: a result's `index` is a
    #: position in this list, whichever shape they arrived in.
    passages: list[str]
    #: Which key the passages came in under; the answer takes the same shape.
    shape: Shape
    top_n: int | None
    #: `return_documents` or `return_text`: either one asks for the passage.
    with_text: bool
    raw_scores: bool
    #: The ignored settings this request carried, in `IGNORED`'s order.
    ignored: list[str]


def _kind(value: Any) -> str:
    """What a JSON value is, in the words an error uses for what it found."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, int | float):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "an array"
    if isinstance(value, dict):
        return "an object"
    return type(value).__name__


def carries(body: dict[str, Any], name: str) -> bool:
    """Whether the body carries this ignored setting, so that the answer names it.

    A JSON `null` is not a setting. Past that, a setting is named when it asks
    for something this door does not do:

    * `truncate: false` is TEI's default and is what this door does: a passage
      longer than the model reads is refused, never cut. Only another value asks
      for a cut, and a cut is not made.
    * `truncation_direction` says which end to cut, so it means something only
      beside a `truncate` that asks for one. Without it nothing is cut and there
      is no direction to ignore, so it is not named.
    * The others (Cohere's `max_chunks_per_doc`, `max_tokens_per_doc` and
      `rank_fields`) ask for something whenever they are sent.
    """
    value = body.get(name)
    if value is None:
        return False
    if name == "truncate":
        return value is not False
    if name == "truncation_direction":
        return carries(body, "truncate")
    return True


def _flag(body: dict[str, Any], name: str) -> bool:
    """A true/false setting. A JSON `null` is not a setting."""
    value = body.get(name)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise Refusal(name, f"must be true or false; this one is {_kind(value)}")
    return value


def _long(where: str, text: str, of: str) -> Refusal | None:
    """A refusal when `text` is longer than the driver's request takes of `of`
    (`one passage`, `the question`)."""
    if len(text) <= MAX_CHARACTERS:
        return None
    return Refusal(
        where, f"is {len(text):,} characters; this door takes at most {MAX_CHARACTERS:,} of {of}"
    )


def _passage(field: Shape, position: int, item: Any) -> str:
    """One passage's text, or a refusal naming where it is."""
    where = f"{field}[{position}]"
    if isinstance(item, str):
        text = item
    elif field == "documents" and isinstance(item, dict):
        said = item.get("text")
        if not isinstance(said, str):
            why = 'has no "text"' if "text" not in item else f'has a "text" that is {_kind(said)}'
            raise Refusal(
                where, f'an object here must carry the passage as a string "text"; this one {why}'
            )
        text = said
    else:
        wanted = "a string"
        if field == "documents":
            wanted = 'a string or an object with a string "text"'
        raise Refusal(where, f"must be {wanted}; this one is {_kind(item)}")
    if (too_long := _long(where, text, "one passage")) is not None:
        raise too_long
    return text


def parse(body: Any) -> RerankAsk:
    """The request as an ask, or a `Refusal` naming the field at fault."""
    if not isinstance(body, dict):
        raise Refusal("body", "must be a JSON object")
    model = body.get("model")
    if model is not None and (not isinstance(model, str) or not model):
        raise Refusal(
            "model", f"must be a model id, a non-empty string; this one is {_kind(model)}"
        )
    query = body.get("query")
    if query is None:
        raise Refusal("query", "is required: it is the question the passages are scored against")
    if not isinstance(query, str):
        raise Refusal(
            "query",
            "must be a string, the question the passages are scored against; "
            f"this one is {_kind(query)}",
        )
    if not query:
        raise Refusal(
            "query", "must not be empty: it is the question the passages are scored against"
        )
    if (too_long := _long("query", query, "the question")) is not None:
        raise too_long

    documents, texts = body.get("documents"), body.get("texts")
    if documents is None and texts is None:
        raise Refusal(
            "documents",
            "send the passages as `documents` (Jina, Cohere) or `texts` (TEI); "
            "this request has neither",
        )
    if documents is not None and texts is not None:
        raise Refusal(
            "documents",
            "send `documents` or `texts`, not both: the answer is in the shape of the one sent",
        )
    shape: Shape = "documents" if documents is not None else "texts"
    sent = documents if documents is not None else texts
    if not isinstance(sent, list):
        raise Refusal(shape, f"must be an array of passages; this one is {_kind(sent)}")
    if len(sent) > MAX_PASSAGES:
        raise Refusal(
            shape,
            f"has {len(sent):,} passages; at most {MAX_PASSAGES:,} a request. "
            "Send them in more than one request",
        )
    passages = [_passage(shape, position, item) for position, item in enumerate(sent)]

    top_n = body.get("top_n")
    if top_n is not None:
        if isinstance(top_n, bool) or not isinstance(top_n, int):
            raise Refusal(
                "top_n", f"must be a whole number of at least 1; this one is {_kind(top_n)}"
            )
        if top_n < 1:
            raise Refusal("top_n", f"must be at least 1; this one is {top_n}")

    return RerankAsk(
        model=model,
        query=query,
        passages=passages,
        shape=shape,
        top_n=top_n,
        with_text=_flag(body, "return_documents") or _flag(body, "return_text"),
        raw_scores=_flag(body, "raw_scores"),
        ignored=[name for name in IGNORED if carries(body, name)],
    )


def results(ask: RerankAsk, scored: Sequence[RerankedPassage]) -> list[dict[str, Any]]:
    """The driver's results in the shape the caller asked in.

    The number is the backend's own with `raw_scores`, else the 0-to-1 one; the
    passage is the request's own text (`llama-server` answers `return_documents`
    with no documents, measured on b10948 and b11552, so it is put back here).
    The driver's order stands: best first, and by index where two are equal.
    Every `index` is one of the passages sent: `HttpDriverClient.rerank` refuses
    an answer that scores a passage nobody sent.
    """
    out: list[dict[str, Any]] = []
    for item in scored:
        score = item.rawScore if ask.raw_scores else item.score
        if ask.shape == "texts":
            entry: dict[str, Any] = {"index": item.index, "score": score}
            if ask.with_text:
                entry["text"] = ask.passages[item.index]
        else:
            entry = {"index": item.index, "relevance_score": score}
            if ask.with_text:
                entry["document"] = {"text": ask.passages[item.index]}
        out.append(entry)
    return out


def usage_of(usage: Usage | None) -> dict[str, int] | None:
    """OpenAI's embeddings usage, which a rerank shares: tokens read, no
    completion. Only what the backend said."""
    if usage is None:
        return None
    said = {
        "prompt_tokens": usage.promptTokens,
        "total_tokens": usage.totalTokens,
    }
    return {name: value for name, value in said.items() if value is not None} or None


def envelope(
    items: list[dict[str, Any]],
    *,
    model: str,
    usage: dict[str, int] | None = None,
    routing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """`RerankResponse`: Jina's and Cohere's answer, which carries its own id."""
    body: dict[str, Any] = {
        "id": f"rerank-{uuid.uuid4().hex}",
        "object": "list",
        "model": model,
        "results": items,
    }
    if usage is not None:
        body["usage"] = usage
    if routing is not None:
        body["x_eugene_plexus"] = routing
    return body


__all__ = [
    "IGNORED",
    "MAX_CHARACTERS",
    "MAX_PASSAGES",
    "Refusal",
    "RerankAsk",
    "carries",
    "envelope",
    "parse",
    "results",
    "usage_of",
]
