"""Local embedding/rerank reads can fail over without duplicating paid work."""

import httpx
import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import Problem
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import DriverError
from eugene_plexus_gateway.settings import Settings

from .conftest import FakeDriverClient, make_routing_table
from .test_local_only import describe_as
from .test_rerank import Reranker


@pytest.mark.parametrize("operation", ["embed", "rerank"])
@pytest.mark.parametrize(
    ("locality", "enforced"),
    [("local", True), ("local", False), ("external", True), ("unknown", True)],
)
@pytest.mark.parametrize("failure", ["read", "driver", "timeout"])
def test_only_confirmed_local_reads_retry_an_uncertain_failure(
    settings: Settings, operation: str, locality: str, enforced: bool, failure: str
) -> None:
    error: Exception
    if failure == "driver":
        error = DriverError(
            driver_name="first",
            driver_url="http://first",
            status_code=502,
            problem=Problem(
                type="https://github.com/eugene-plexus/inference-driver#backend-error",
                title="Backend error",
                status=502,
                detail="ReadError: connection lost. Outcome unknown: work may have occurred.",
                retryDisposition="indeterminate",
            ),
            raw_body="",
        )
    else:
        error = httpx.ReadTimeout("late") if failure == "timeout" else httpx.ReadError("lost")
    if operation == "embed":
        first = FakeDriverClient(name="first", model_id="model", supports_embeddings=True)
        second = FakeDriverClient(name="second", model_id="model", supports_embeddings=True)
        first.embed_error = error
        path, body = "/v1/embeddings", {"model": "model", "input": "text"}
    else:
        first = Reranker(name="first", model_id="model", fail=error)
        second = Reranker(name="second", model_id="model")
        path, body = "/v1/rerank", {"model": "model", "query": "text", "documents": ["text"]}
    for driver in (first, second):
        describe_as(driver, locality, enforced)
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(first, second)
    with TestClient(app) as client:
        response = client.post(path, json=body)
    should_retry = locality == "local" and enforced and failure != "timeout"
    assert response.status_code == (200 if should_retry else 504 if failure == "timeout" else 502)
    calls = second.embed_calls if operation == "embed" else len(second.heard)
    assert calls == int(should_retry)


@pytest.mark.parametrize("operation", ["embed", "rerank"])
def test_exhausted_local_reads_name_the_failed_backend_without_unknown_work(
    settings: Settings, operation: str
) -> None:
    error = httpx.ReadError("connection lost")
    if operation == "embed":
        driver = FakeDriverClient(name="local-engine", model_id="model", supports_embeddings=True)
        driver.embed_error = error
        path, body = "/v1/embeddings", {"model": "model", "input": "text"}
    else:
        driver = Reranker(name="local-engine", model_id="model", fail=error)
        path, body = "/v1/rerank", {"model": "model", "query": "text", "documents": ["text"]}
    describe_as(driver, "local")
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(driver)
    with TestClient(app) as client:
        response = client.post(path, json=body)
    assert response.status_code == 502
    assert "local-engine" in response.text
    assert "connection lost" in response.text
    assert "work may have occurred" not in response.text.lower()
    assert "outcome unknown" not in response.text.lower()
