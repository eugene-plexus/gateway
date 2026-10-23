"""Call #2 — embeddings, gateway half.

The endpoint is mostly plumbing. What is not plumbing, and what most of
these tests are about, is the rule it turns on:

    **Failover does not cross models here.**

Everywhere else a slot is an ordered list of targets and a failure
cascades to the next. For chat that degrades gracefully. For embeddings
it is silent corruption — vectors from two models occupy different
spaces, so a fallback writes noise into the caller's store with a 200
and no marker, and the damage outlives the request in a database.
"""

from __future__ import annotations

import base64
import struct

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import Problem, Usage
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import DriverError

from .conftest import FakeDriverClient, install_snapshot, make_routing_table


def test_dimensions_are_forwarded_and_not_called_ignored_for_gemini(settings) -> None:
    from eugene_plexus_gateway._generated.driver_models import BackendKind

    driver = _embedder("gemini", "embed")
    driver.backend = BackendKind.gemini_api
    sent = []
    original = driver.embed

    async def capture(request):
        sent.append(request)
        return await original(request)

    driver.embed = capture
    with TestClient(_app(driver, settings=settings)) as client:
        response = client.post(
            "/v1/embeddings", json={"model": "embed", "input": "a", "dimensions": 4}
        )
    assert response.status_code == 200, response.text
    assert sent[0].dimensions == 4
    assert "dimensions" not in response.headers.get("x-eugene-plexus-ignored-settings", "")


def _embedder(name: str, model_id: str) -> FakeDriverClient:
    """A driver that reports the embeddings surface. Vectors derive from
    the driver name, so a test can tell WHICH backend answered from the
    numbers alone -- the only way to catch a cascade that crossed
    models."""
    return FakeDriverClient(name=name, model_id=model_id, supports_embeddings=True)


def _app(*drivers: FakeDriverClient, settings, slots=None):  # type: ignore[no-untyped-def]
    app = create_app(settings=settings)
    table = make_routing_table(slots=slots or [])
    install_snapshot(table, *drivers)
    app.state.routing = table
    return app


@pytest.fixture
def embed_client(settings):  # type: ignore[no-untyped-def]
    driver = _embedder("embedder", "nomic-embed-text")
    driver.usage = Usage(promptTokens=3, completionTokens=None, totalTokens=3)
    with TestClient(_app(driver, settings=settings)) as c:
        yield driver, c


# --------------------------------------------------------------------- #
# it works
# --------------------------------------------------------------------- #


def test_a_batch_returns_one_vector_per_input_in_order(embed_client) -> None:  # type: ignore[no-untyped-def]
    _, client = embed_client

    body = client.post(
        "/v1/embeddings", json={"model": "nomic-embed-text", "input": ["a", "b", "c"]}
    ).json()

    assert body["object"] == "list"
    assert [d["index"] for d in body["data"]] == [0, 1, 2]
    assert all(d["object"] == "embedding" for d in body["data"])
    assert body["usage"] == {"prompt_tokens": 3, "total_tokens": 3}


def test_a_bare_string_input_is_accepted(embed_client) -> None:  # type: ignore[no-untyped-def]
    """OpenAI allows either; a client that sends a string must not get a
    400 for it."""
    _, client = embed_client

    body = client.post("/v1/embeddings", json={"model": "nomic-embed-text", "input": "a"}).json()

    assert len(body["data"]) == 1


def test_base64_is_little_endian_float32(embed_client) -> None:  # type: ignore[no-untyped-def]
    """**The OpenAI SDKs ask for base64 by default**, so getting this
    wrong breaks the most common client while every hand-rolled curl
    keeps working. The encoding was verified byte-for-byte against a
    real backend's own output; this pins it."""
    _, client = embed_client

    floats = client.post("/v1/embeddings", json={"model": "nomic-embed-text", "input": "a"}).json()[
        "data"
    ][0]["embedding"]
    encoded = client.post(
        "/v1/embeddings",
        json={"model": "nomic-embed-text", "input": "a", "encoding_format": "base64"},
    ).json()["data"][0]["embedding"]

    assert isinstance(encoded, str)
    raw = base64.b64decode(encoded)
    assert list(struct.unpack(f"<{len(raw) // 4}f", raw)) == pytest.approx(floats)


def test_a_token_array_input_is_refused_rather_than_guessed(embed_client) -> None:  # type: ignore[no-untyped-def]
    """OpenAI accepts token arrays. Validating one needs a tokenizer and
    this install deliberately owns none (step 7), so it is refused
    instead of silently embedding something else.

    **422, not 400, and that is correct**: the contract types `input` as
    string-or-array-of-string, so schema validation rejects it before
    the handler runs. Asserting 400 here would have been asserting that
    a weaker check did the work."""
    _, client = embed_client

    response = client.post(
        "/v1/embeddings", json={"model": "nomic-embed-text", "input": [[1, 2, 3]]}
    )

    assert response.status_code == 422


def test_an_empty_batch_is_refused_too(embed_client) -> None:  # type: ignore[no-untyped-def]
    """Also the schema, via `minItems: 1`.

    Worth its own test because a hand-written check for this WAS added
    to the handler first, and measuring found it could never fire. It
    was removed rather than left with a test pretending otherwise."""
    _, client = embed_client

    response = client.post("/v1/embeddings", json={"model": "nomic-embed-text", "input": []})

    assert response.status_code == 422


# --------------------------------------------------------------------- #
# settings it accepts and does not honour (MR0)
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("unknown", ["input_type", "bogus"])
def test_unknown_fields_are_refused_and_named(embed_client, unknown: str) -> None:  # type: ignore[no-untyped-def]
    driver, client = embed_client
    response = client.post(
        "/v1/embeddings",
        json={"model": "nomic-embed-text", "input": "a", unknown: "private-value"},
    )
    assert response.status_code == 400
    assert unknown in response.json()["error"]["message"]
    assert "private-value" not in response.text
    assert driver.embed_calls == 0


#
# Until MR0 the contract called `dimensions` "passed through" and `user`
# "recorded", and the door dropped both without a word. A caller asking for
# 256 dimensions got the model's full width, with a 200 and no sign that
# anything had been ignored.

IGNORED = "x-eugene-plexus-ignored-settings"


def _embed(client, **extra):  # type: ignore[no-untyped-def]
    return client.post("/v1/embeddings", json={"model": "nomic-embed-text", "input": "a", **extra})


def test_dimensions_is_named_as_ignored(embed_client) -> None:  # type: ignore[no-untyped-def]
    _, client = embed_client

    response = _embed(client, dimensions=2)

    assert response.status_code == 200, response.text
    assert response.headers[IGNORED] == "dimensions"


def test_user_is_named_as_ignored(embed_client) -> None:  # type: ignore[no-untyped-def]
    _, client = embed_client

    response = _embed(client, user="person-1")

    assert response.status_code == 200, response.text
    assert response.headers[IGNORED] == "user"


def test_both_are_named_in_one_stable_order(embed_client) -> None:  # type: ignore[no-untyped-def]
    """The order is the door's, not the request's: `user` is sent first here."""
    _, client = embed_client

    response = _embed(client, user="person-1", dimensions=2)

    assert response.status_code == 200, response.text
    assert response.headers[IGNORED] == "dimensions, user"


def test_nothing_is_named_when_neither_is_sent(embed_client) -> None:  # type: ignore[no-untyped-def]
    """A header that appeared on every answer would teach callers to ignore
    it. `encoding_format` is honoured here, so it is never named either."""
    _, client = embed_client

    assert IGNORED not in _embed(client).headers
    assert IGNORED not in _embed(client, encoding_format="base64").headers


def test_a_null_is_not_a_setting(embed_client) -> None:  # type: ignore[no-untyped-def]
    """As on the Anthropic door: only a value is named. An SDK that always
    sends every field as `null` is not asking for anything."""
    _, client = embed_client

    response = _embed(client, dimensions=None, user=None)

    assert response.status_code == 200, response.text
    assert IGNORED not in response.headers


def test_the_vectors_come_back_at_the_models_own_width(embed_client) -> None:  # type: ignore[no-untyped-def]
    """**The header is only honest if nothing here quietly acts on the setting.**
    This layer does not emulate `dimensions`: a truncated vector is a wrong
    answer that looks like a right one. The fake driver's vectors are 4 wide."""
    _, client = embed_client

    floats = _embed(client, dimensions=2).json()["data"][0]["embedding"]

    assert len(floats) == 4


def test_the_header_rides_a_base64_answer_too(embed_client) -> None:  # type: ignore[no-untyped-def]
    """The SDKs ask for base64 by default, so this is the answer most callers read."""
    _, client = embed_client

    response = _embed(client, dimensions=2, encoding_format="base64")

    assert response.headers[IGNORED] == "dimensions"
    assert len(base64.b64decode(response.json()["data"][0]["embedding"])) == 4 * 4


def test_a_refused_or_failed_request_names_nothing(settings) -> None:  # type: ignore[no-untyped-def]
    """The header describes an answer that was served. A request that got none
    has nothing to be told about, and a client that retries sees it only on the
    attempt that served."""
    dead = _embedder("primary", "nomic-embed-text")
    dead.embed_error = DriverError(
        driver_name="primary",
        driver_url="http://primary",
        status_code=503,
        problem=Problem(
            type="about:blank", title="Not started", status=503, retryDisposition="safe"
        ),
        raw_body="down",
    )
    both = {"input": "a", "dimensions": 2, "user": "person-1"}
    with TestClient(_app(dead, settings=settings)) as client:
        refused = client.post("/v1/embeddings", json={"model": "no-such-model", **both})
        failed = client.post("/v1/embeddings", json={"model": "nomic-embed-text", **both})

    assert refused.status_code == 404
    assert failed.status_code != 200
    assert IGNORED not in refused.headers
    assert IGNORED not in failed.headers


# --------------------------------------------------------------------- #
# THE RULE
# --------------------------------------------------------------------- #


def test_a_failure_does_NOT_cascade_to_a_different_model(settings) -> None:  # type: ignore[no-untyped-def]
    """**The test this endpoint exists to pass.**

    Tier 1 is a dead embedder, tier 2 is a healthy one for a DIFFERENT
    model. For chat this cascades and that is correct. Here it must
    fail: a 200 carrying tier 2's vectors would be noise in the caller's
    vector store, with nothing to mark it.
    """
    dead = _embedder("primary", "nomic-embed-text")
    dead.embed_error = DriverError(
        driver_name="primary",
        driver_url="http://primary",
        status_code=503,
        problem=Problem(
            type="about:blank", title="Not started", status=503, retryDisposition="safe"
        ),
        raw_body="down",
    )
    other = _embedder("secondary", "all-minilm")

    app = _app(
        dead,
        other,
        settings=settings,
        slots=[{"model": "nomic-embed-text", "targets": ["nomic-embed-text", "all-minilm"]}],
    )
    with TestClient(app) as client:
        response = client.post("/v1/embeddings", json={"model": "nomic-embed-text", "input": "a"})

    assert response.status_code != 200, "it served vectors from the wrong model"
    assert other.embed_calls == 0, "the other model was contacted at all"


def test_replicas_of_the_SAME_model_still_fail_over(settings) -> None:  # type: ignore[no-untyped-def]
    """The other half of the rule, and the half that makes it a rule
    rather than a refusal to route. Two replicas of one model are
    interchangeable by definition, so failover between them is exactly
    as safe as it is for chat."""
    dead = _embedder("replica-a", "nomic-embed-text")
    dead.embed_error = DriverError(
        driver_name="replica-a",
        driver_url="http://a",
        status_code=503,
        problem=Problem(
            type="about:blank", title="Not started", status=503, retryDisposition="safe"
        ),
        raw_body="down",
    )
    alive = _embedder("replica-bb", "nomic-embed-text")

    app = _app(dead, alive, settings=settings)
    with TestClient(app) as client:
        response = client.post("/v1/embeddings", json={"model": "nomic-embed-text", "input": "a"})

    assert response.status_code == 200
    assert alive.embed_calls == 1
    assert response.json()["x_eugene_plexus"]["attempts"] == 2


def test_a_slot_alias_embeds_with_its_first_model(settings) -> None:  # type: ignore[no-untyped-def]
    """An alias is its first target. Matching the request's own name
    instead answered every alias 503, because no backend's id IS the
    alias -- speech had the same defect and was fixed at P3a."""
    first = _embedder("primary", "nomic-embed-text")
    second = _embedder("secondary", "all-minilm")
    app = _app(
        first,
        second,
        settings=settings,
        slots=[{"model": "vectors", "targets": ["nomic-embed-text", "all-minilm"]}],
    )
    with TestClient(app) as client:
        response = client.post("/v1/embeddings", json={"model": "vectors", "input": "a"})

    assert response.status_code == 200, response.text
    assert first.embed_calls == 1
    assert second.embed_calls == 0


def test_a_slot_aliass_first_model_down_is_not_replaced_by_its_second(settings) -> None:  # type: ignore[no-untyped-def]
    """The fix must not reopen the cascade: the alias's second target is
    a different vector space, so a dead first model fails the request."""
    dead = _embedder("primary", "nomic-embed-text")
    dead.embed_error = DriverError(
        driver_name="primary",
        driver_url="http://primary",
        status_code=503,
        problem=Problem(
            type="about:blank", title="Not started", status=503, retryDisposition="safe"
        ),
        raw_body="down",
    )
    other = _embedder("secondary", "all-minilm")
    app = _app(
        dead,
        other,
        settings=settings,
        slots=[{"model": "vectors", "targets": ["nomic-embed-text", "all-minilm"]}],
    )
    with TestClient(app) as client:
        response = client.post("/v1/embeddings", json={"model": "vectors", "input": "a"})

    assert response.status_code != 200, "it served vectors from the wrong model"
    assert dead.embed_calls == 1
    assert other.embed_calls == 0


# --------------------------------------------------------------------- #
# surfaces
# --------------------------------------------------------------------- #


def test_an_embeddings_model_is_refused_on_the_chat_surface(embed_client) -> None:  # type: ignore[no-untyped-def]
    """The coherence gap this whole call was taken to close: the library
    will discover, download and launch a dedicated embedding model, and
    before this it sat on /v1/models looking like anything else."""
    _, client = embed_client

    response = client.post(
        "/v1/chat/completions",
        json={"model": "nomic-embed-text", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "serves embeddings" in message
    assert "/v1/embeddings" in message


def test_a_chat_model_is_refused_on_the_embeddings_surface(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    response = client.post("/v1/embeddings", json={"model": fake_driver.model_id, "input": "a"})

    assert response.status_code == 400
    assert "/v1/chat/completions" in response.json()["error"]["message"]


def test_models_reports_which_surfaces_each_model_serves(embed_client) -> None:  # type: ignore[no-untyped-def]
    """OpenAI's own /v1/models does not say, which is why every RAG
    front-end makes you pick from a list of everything and find out by
    failing."""
    _, client = embed_client

    data = client.get("/v1/models").json()["data"]
    surfaces = {m["id"]: m["x_eugene_plexus"]["surfaces"] for m in data}

    assert surfaces["nomic-embed-text"] == ["embeddings"]


def test_a_chat_only_driver_still_reports_the_chat_surface(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    """**The regression this guards.** Every driver that existed before
    this surface reports nothing about embeddings, so if "no embeddings
    flag" read as "no surfaces" a re-pin would silently unroute every
    model in the install."""
    data = client.get("/v1/models").json()["data"]

    assert data[0]["x_eugene_plexus"]["surfaces"] == ["chat"]
