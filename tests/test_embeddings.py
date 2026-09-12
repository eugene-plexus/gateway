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

from eugene_plexus_gateway._generated.driver_models import Usage
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import DriverError

from .conftest import FakeDriverClient, install_snapshot, make_routing_table


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
        problem=None,
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
        problem=None,
        raw_body="down",
    )
    alive = _embedder("replica-bb", "nomic-embed-text")

    app = _app(dead, alive, settings=settings)
    with TestClient(app) as client:
        response = client.post("/v1/embeddings", json={"model": "nomic-embed-text", "input": "a"})

    assert response.status_code == 200
    assert alive.embed_calls == 1
    assert response.json()["x_eugene_plexus"]["attempts"] == 2


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
