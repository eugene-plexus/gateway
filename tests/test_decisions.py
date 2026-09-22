"""The decision door: POST /v1/systemone, in the pinned TypeSafe shape.

What the door adds beyond the embeddings precedent, each with a test
that fails without it: protocol bounds run on the RAW body before any
backend work; a decision-only backend is unroutable for chat and a chat
backend unroutable for decisions, both refused naming the right door;
and a backend that advertises a decision concurrency ceiling is never
over-admitted — all-at-capacity is a 503, deliberately not a queue,
because a single-slot backend cannot shed work.
"""

from __future__ import annotations

import asyncio
import threading

from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import Problem, RetryDisposition, Usage
from eugene_plexus_gateway.app import create_app
from eugene_plexus_gateway.driver_client import DriverError

from .conftest import FakeDriverClient, install_snapshot, make_routing_table


def _decider(name: str, model_id: str, *, max_concurrent: int | None = 1) -> FakeDriverClient:
    return FakeDriverClient(
        name=name,
        model_id=model_id,
        supports_decisions=True,
        decision_max_concurrent=max_concurrent,
    )


def _app(*drivers: FakeDriverClient, settings):  # type: ignore[no-untyped-def]
    app = create_app(settings=settings)
    table = make_routing_table(slots=[])
    install_snapshot(table, *drivers)
    app.state.routing = table
    return app


def _questions() -> dict:
    return {
        "refunded": {"type": "noul", "instructions": "Was a refund issued?"},
        "route": {
            "type": "choice",
            "instructions": "Route this ticket.",
            "criteria": {"billing": "money", "shipping": "boxes"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["low", "medium", "high"],
        },
    }


# --------------------------------------------------------------------- #
# it works, in TypeSafe's own vocabulary
# --------------------------------------------------------------------- #


def test_mixed_questions_come_back_in_the_pinned_shape(settings) -> None:  # type: ignore[no-untyped-def]
    driver = _decider("kev-driver", "tickets")
    driver.usage = Usage(promptTokens=89, completionTokens=156, totalTokens=245)
    with TestClient(_app(driver, settings=settings)) as client:
        response = client.post(
            "/v1/systemone",
            json={"model": "tickets", "state": "charged twice", "questions": _questions()},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == "tickets"
    assert body["answers"]["refunded"] == {"type": "noul", "noul": 0.9}
    assert body["answers"]["route"]["choice"] == "billing"
    assert body["answers"]["urgency"]["legend"]["2"] == "high"
    # TypeSafe's usage vocabulary, not the house one.
    assert body["usage"] == {"input_tokens": 89, "output_tokens": 156}
    assert body["x_eugene_plexus"]["attempts"] == 1
    assert driver.decide_calls == 1


def test_structured_state_rides_through_untouched(settings) -> None:  # type: ignore[no-untyped-def]
    driver = _decider("kev-driver", "tickets")
    with TestClient(_app(driver, settings=settings)) as client:
        response = client.post(
            "/v1/systemone",
            json={
                "model": "tickets",
                "state": {"order": {"id": "A-1", "status": "refunded"}},
                "questions": {"q": {"type": "noul", "instructions": "?"}},
            },
        )
    assert response.status_code == 200, response.text


# --------------------------------------------------------------------- #
# the two doors refuse each other's models by name
# --------------------------------------------------------------------- #


def test_chat_against_a_decision_model_names_the_decision_door(settings) -> None:  # type: ignore[no-untyped-def]
    driver = _decider("kev-driver", "tickets")
    with TestClient(_app(driver, settings=settings)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "tickets", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 400
    assert "/v1/systemone" in response.text


def test_decisions_against_a_chat_model_name_the_chat_door(settings) -> None:  # type: ignore[no-untyped-def]
    chat = FakeDriverClient(name="qwen-driver", model_id="qwen")
    with TestClient(_app(chat, settings=settings)) as client:
        response = client.post(
            "/v1/systemone",
            json={
                "model": "qwen",
                "state": "x",
                "questions": {"q": {"type": "noul", "instructions": "?"}},
            },
        )
    assert response.status_code == 400
    assert "/v1/chat/completions" in response.text


def test_an_unknown_model_is_a_404(settings) -> None:  # type: ignore[no-untyped-def]
    driver = _decider("kev-driver", "tickets")
    with TestClient(_app(driver, settings=settings)) as client:
        response = client.post(
            "/v1/systemone",
            json={
                "model": "nowhere",
                "state": "x",
                "questions": {"q": {"type": "noul", "instructions": "?"}},
            },
        )
    assert response.status_code == 404


# --------------------------------------------------------------------- #
# bounds before backend work
# --------------------------------------------------------------------- #


def test_unknown_question_fields_are_422_and_reach_no_backend(settings) -> None:  # type: ignore[no-untyped-def]
    driver = _decider("kev-driver", "tickets")
    with TestClient(_app(driver, settings=settings)) as client:
        response = client.post(
            "/v1/systemone",
            json={
                "model": "tickets",
                "state": "x",
                "questions": {"q": {"type": "noul", "instructions": "?", "top_p": 0.9}},
            },
        )
    assert response.status_code == 422
    assert "top_p" in response.text
    assert driver.decide_calls == 0


def test_the_question_ceiling_is_enforced_before_work(settings) -> None:  # type: ignore[no-untyped-def]
    driver = _decider("kev-driver", "tickets")
    questions = {
        f"q{i}": {"type": "noul", "instructions": "?"} for i in range(33)
    }  # default ceiling is 32
    with TestClient(_app(driver, settings=settings)) as client:
        response = client.post(
            "/v1/systemone",
            json={"model": "tickets", "state": "x", "questions": questions},
        )
    assert response.status_code == 422
    assert "decisionMaxQuestions" in response.text
    assert driver.decide_calls == 0


def test_protocol_bounds_are_the_pinned_ones(settings) -> None:  # type: ignore[no-untyped-def]
    driver = _decider("kev-driver", "tickets")
    with TestClient(_app(driver, settings=settings)) as client:
        response = client.post(
            "/v1/systemone",
            json={
                "model": "tickets",
                "state": "x",
                "questions": {
                    "a": {
                        "type": "choice",
                        "instructions": "?",
                        "criteria": {str(i): None for i in range(256)},
                    },
                    "b": {"type": "score", "instructions": "?", "criteria": ["one"]},
                },
            },
        )
    assert response.status_code == 422
    assert "255" in response.text and "2-10" in response.text
    assert driver.decide_calls == 0


# --------------------------------------------------------------------- #
# a single-slot backend is never over-admitted
# --------------------------------------------------------------------- #


def test_a_backend_at_its_ceiling_is_503_not_queued(settings) -> None:  # type: ignore[no-untyped-def]
    """Kev holds one request at a time and cannot shed work. While one
    decision is in flight, a second must be told to retry — queuing it
    on a socket the backend will not read is invisible over-admission."""
    driver = _decider("kev-driver", "tickets", max_concurrent=1)
    entered = threading.Event()
    release = threading.Event()

    async def hold() -> None:
        entered.set()
        # A threading.Event, polled from the loop: both requests run on
        # the TestClient portal's one loop, so an asyncio.Event awaited
        # here could never be set by the second REQUEST thread.
        while not release.is_set():  # noqa: ASYNC110 — cross-thread signal
            await asyncio.sleep(0.01)

    driver.decide_hook = hold
    payload = {
        "model": "tickets",
        "state": "x",
        "questions": {"q": {"type": "noul", "instructions": "?"}},
    }
    with TestClient(_app(driver, settings=settings)) as client:
        first_result: dict = {}

        def first() -> None:
            first_result["response"] = client.post("/v1/systemone", json=payload)

        worker = threading.Thread(target=first)
        worker.start()
        assert entered.wait(timeout=5), "first decision never reached the backend"
        try:
            second = client.post("/v1/systemone", json=payload)
        finally:
            release.set()
            worker.join(timeout=10)

    assert second.status_code == 503, second.text
    assert "concurrency ceiling" in second.text
    assert "not queued" in second.text
    assert first_result["response"].status_code == 200
    # The backend saw exactly one request: the ceiling held.
    assert driver.decide_calls == 1


def test_replicas_of_one_decision_model_fail_over(settings) -> None:  # type: ignore[no-untyped-def]
    """A refusal before work cascades across replicas of the SAME model,
    exactly as embeddings do — and never to a different model."""
    dead = _decider("dead", "tickets", max_concurrent=None)
    # A PROVEN pre-execution refusal (A6b): only retryDisposition: safe
    # cascades. A bare 503 with no problem body is indeterminate -- work
    # may have occurred -- and deliberately does not.
    dead.decide_error = DriverError(
        driver_name="dead",
        driver_url="http://dead",
        status_code=503,
        problem=Problem(
            type="about:blank",
            title="Fixture refusal before work",
            status=503,
            retryDisposition=RetryDisposition.safe,
        ),
        raw_body="refused before work",
    )
    healthy = _decider("healthy", "tickets", max_concurrent=None)
    other_model = _decider("other", "invoices", max_concurrent=None)
    with TestClient(_app(dead, healthy, other_model, settings=settings)) as client:
        response = client.post(
            "/v1/systemone",
            json={
                "model": "tickets",
                "state": "x",
                "questions": {"q": {"type": "noul", "instructions": "?"}},
            },
        )
    assert response.status_code == 200, response.text
    assert response.json()["x_eugene_plexus"]["attempts"] == 2
    assert healthy.decide_calls == 1
    # The one question tiered failover must never answer wrongly here:
    # the other MODEL was never asked.
    assert other_model.decide_calls == 0


def test_models_listing_marks_the_decision_surface(settings) -> None:  # type: ignore[no-untyped-def]
    driver = _decider("kev-driver", "tickets")
    with TestClient(_app(driver, settings=settings)) as client:
        listing = client.get("/v1/models").json()
    entry = next(m for m in listing["data"] if m["id"] == "tickets")
    assert entry["x_eugene_plexus"]["surfaces"] == ["decisions"]
