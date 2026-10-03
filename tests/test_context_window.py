"""Step 7 — context-window honesty, gateway half.

Open call #3 was settled 2026-09-12 as **let the engine refuse**, so
there is deliberately no tokenizer here and no size check. What the
gateway does instead is report: the window that applied to the request,
and whether the backend quietly threw most of the prompt away.

The second one is the differentiator. A backend that refuses is already
honest — llama.cpp names both numbers and the driver now passes that
through as a 400. A backend that truncates says nothing at all, and the
caller gets a confident answer about input the model never saw.
"""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.driver_models import Problem, Usage
from eugene_plexus_gateway.driver_client import DriverError
from eugene_plexus_gateway.routing import _RuntimeFacts

from .conftest import FakeDriverClient, install_snapshot, make_routing_table

# 66,389 characters across six messages came back as `prompt_tokens: 86`
# on this hardware — HTTP 200, no flag anywhere. Canary-verified on
# qwen3-coder:30b: the system message and the last turn survived, the
# first and the middle did not.
_MEASURED_CHARS = 66_389
_MEASURED_PROMPT_TOKENS = 86


def _long_body(model: str, chars: int = _MEASURED_CHARS) -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "x" * chars}],
    }


def _envelope(response: object) -> dict[str, object]:
    return response.json()["x_eugene_plexus"]  # type: ignore[attr-defined]


# --------------------------------------------------------------------- #
# the backend dropped the input and said nothing
# --------------------------------------------------------------------- #


def test_a_silently_truncated_prompt_is_flagged(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    """The measured case, reproduced: a very large prompt that the
    backend reports consuming almost none of."""
    fake_driver.usage = Usage(
        promptTokens=_MEASURED_PROMPT_TOKENS, completionTokens=10, totalTokens=96
    )

    response = client.post("/v1/chat/completions", json=_long_body(fake_driver.model_id or ""))

    assert response.status_code == 200
    assert _envelope(response)["prompt_truncated"] is True


def test_a_prompt_that_arrived_intact_is_not_flagged(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    """The other half of the measurement. A real tokenizer on this input
    reported 20,560 tokens for 66,389 characters — about 3.2 characters
    per token, nowhere near the threshold."""
    fake_driver.usage = Usage(promptTokens=20_560, completionTokens=10, totalTokens=20_570)

    response = client.post("/v1/chat/completions", json=_long_body(fake_driver.model_id or ""))

    assert _envelope(response)["prompt_truncated"] is False


def test_a_backend_that_reports_no_usage_is_not_accused(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    """Absent is a different claim from `false`. Several backends report
    no usage at all, and "we could not check" must not read as "we
    checked and it was fine" — nor as an accusation."""
    fake_driver.usage = None

    response = client.post("/v1/chat/completions", json=_long_body(fake_driver.model_id or ""))

    assert _envelope(response)["prompt_truncated"] is None


def test_a_short_prompt_is_not_evaluated(client: TestClient, fake_driver: FakeDriverClient) -> None:
    """Below the floor the ratio stops meaning anything: no window in use
    anywhere is small enough for a prompt this size to be truncated, and
    short inputs are where token density varies most. A single emoji
    would otherwise be a truncation report."""
    fake_driver.usage = Usage(promptTokens=1, completionTokens=1, totalTokens=2)

    response = client.post(
        "/v1/chat/completions", json=_long_body(fake_driver.model_id or "", chars=200)
    )

    assert _envelope(response)["prompt_truncated"] is None


def test_the_detector_does_not_fire_on_a_merely_token_dense_prompt(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    """The threshold is one token per 20 characters, far below any real
    tokenizer. Long runs of whitespace are the most token-efficient input
    anyone actually sends and do not come close, so the detector cannot
    be tripped by ordinary content -- only by input that never arrived."""
    # Ten characters per token: extremely dense, and still half the
    # threshold away.
    fake_driver.usage = Usage(promptTokens=_MEASURED_CHARS // 10, completionTokens=1, totalTokens=1)

    response = client.post("/v1/chat/completions", json=_long_body(fake_driver.model_id or ""))

    assert _envelope(response)["prompt_truncated"] is False


def test_truncation_is_also_written_to_the_log(
    client: TestClient, fake_driver: FakeDriverClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The flag rides a namespaced field most clients drop on the floor.
    An operator chasing "the model keeps ignoring my files" needs a line
    somewhere they will actually look."""
    fake_driver.usage = Usage(
        promptTokens=_MEASURED_PROMPT_TOKENS, completionTokens=10, totalTokens=96
    )

    with caplog.at_level(logging.WARNING, logger="eugene_plexus_gateway.routes.inference"):
        client.post("/v1/chat/completions", json=_long_body(fake_driver.model_id or ""))

    assert any("discarded most of the prompt" in r.getMessage() for r in caplog.records)


def test_a_streamed_request_reports_truncation_on_its_final_frame(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    """A stream cannot be unsent (M10), so this can only ever be a flag —
    which is exactly why the non-streamed path flags rather than fails.
    One condition must not report two different ways depending on a
    parameter the caller picked for unrelated reasons."""
    fake_driver.usage = Usage(
        promptTokens=_MEASURED_PROMPT_TOKENS, completionTokens=10, totalTokens=96
    )
    body = _long_body(fake_driver.model_id or "")
    body["stream"] = True

    response = client.post("/v1/chat/completions", json=body)

    frames = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    ]
    final = [f for f in frames if f.get("x_eugene_plexus")]
    assert final, "no frame carried the routing envelope"
    assert final[-1]["x_eugene_plexus"]["prompt_truncated"] is True


# --------------------------------------------------------------------- #
# the window that applied to this request
# --------------------------------------------------------------------- #


def test_the_completion_reports_the_answering_backends_window(settings_app_pair) -> None:
    """Per-request, and therefore not `GET /v1/models`'s number. That one
    is the smallest across every backend serving the name, which is right
    when a caller is choosing a model and wrong once a request has
    landed: what explains a truncated answer is the window that applied."""
    small, big, client = settings_app_pair
    # `big` is the only one eligible, so it answers.
    small.generate_error = DriverError(
        driver_name=small.name,
        driver_url=small.base_url,
        status_code=500,
        problem=Problem(
            type="about:blank", title="Not started", status=503, retryDisposition="safe"
        ),
        raw_body="down",
    )

    response = client.post(
        "/v1/chat/completions",
        json={"model": "shared-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    envelope = _envelope(response)
    assert envelope["driver"] == big.name
    assert envelope["context_length"] == 131072


def test_a_backend_with_no_window_reports_none(
    client: TestClient, fake_driver: FakeDriverClient
) -> None:
    """Unknown stays unknown. A hosted provider exposes nothing to read
    and a CLI subscription has no window of its own; inventing one is the
    failure the field exists to prevent."""
    assert fake_driver.max_context_tokens is None

    response = client.post(
        "/v1/chat/completions",
        json={"model": fake_driver.model_id, "messages": [{"role": "user", "content": "hi"}]},
    )

    assert _envelope(response)["context_length"] is None


def test_a_supervised_runtimes_window_wins_over_the_drivers_own(settings) -> None:
    """The agent reads it back off the engine after it loads, which is
    the post-clamp number; the driver's probe may predate the clamp."""
    from eugene_plexus_gateway.app import create_app

    fake = FakeDriverClient(
        name="local", model_id="m", runtime="engine-1", max_context_tokens=999999
    )
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(
        fake,
        runtimes=[
            _RuntimeFacts(
                name="engine-1",
                alias=None,
                status="ready",
                node=None,
                url="http://engine",
                context_length=4096,
                parallel_slots=1,
                idle_unload_seconds=None,
                start_on_demand=False,
                stop_reason=None,
            )
        ],
    )
    with TestClient(app) as c:
        response = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert _envelope(response)["context_length"] == 4096


# --------------------------------------------------------------------- #
# the second test: a cut that keeps a near-full window (2026-10-03)
# --------------------------------------------------------------------- #

# Ollama's default context is 4,096 tokens on a card under 23 GiB, and it
# still keeps the newest turns and drops the middle (upstream drift audit,
# 2026-10-03). A 40,000-character prompt kept to a full 4k window passes
# the ratio test -- 4,000 tokens is one per ten characters, half the
# threshold -- so it went unflagged.
_WINDOW = 4096
_CUT_CHARS = 40_000
_CUT_TOKENS = 4_000


def _windowed_app(settings, **fake_args):  # type: ignore[no-untyped-def]
    from eugene_plexus_gateway.app import create_app

    fake = FakeDriverClient(name="ollama-box", model_id="m", supports_tools=True, **fake_args)
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(fake)
    return fake, app


def _truncation_flag(settings, prompt_tokens: int, chars: int, **fake_args) -> object:  # type: ignore[no-untyped-def]
    fake, app = _windowed_app(settings, **fake_args)
    fake.usage = Usage(promptTokens=prompt_tokens, completionTokens=1, totalTokens=prompt_tokens)
    with TestClient(app) as c:
        response = c.post("/v1/chat/completions", json=_long_body("m", chars=chars))
    assert response.status_code == 200, response.text
    return _envelope(response)["prompt_truncated"]


def test_a_cut_that_keeps_a_near_full_window_is_flagged(settings) -> None:  # type: ignore[no-untyped-def]
    assert _truncation_flag(settings, _CUT_TOKENS, _CUT_CHARS, max_context_tokens=_WINDOW) is True


@pytest.mark.parametrize(
    ("prompt_tokens", "chars", "window"),
    [
        # Nearly fills the window, and the characters fit it: it arrived.
        (3_900, 14_000, _WINDOW),
        # The characters would overflow the window, but far from full: the
        # backend did not stop at the window, so nothing says it cut.
        (_CUT_TOKENS, _CUT_CHARS, 8_192),
        # No window known: only the ratio test can speak.
        (_CUT_TOKENS, _CUT_CHARS, None),
    ],
    ids=["fits", "not-near-full", "no-window"],
)
def test_the_window_test_does_not_fire_without_both_of_its_conditions(  # type: ignore[no-untyped-def]
    settings, prompt_tokens: int, chars: int, window: int | None
) -> None:
    assert _truncation_flag(settings, prompt_tokens, chars, max_context_tokens=window) is False


def test_the_window_test_reads_the_window_that_applied(settings) -> None:  # type: ignore[no-untyped-def]
    """A supervised runtime's window wins over the driver's own, as for
    `context_length` beside the flag."""
    from eugene_plexus_gateway.app import create_app

    fake = FakeDriverClient(
        name="local", model_id="m", runtime="engine-1", max_context_tokens=99999
    )
    fake.usage = Usage(promptTokens=_CUT_TOKENS, completionTokens=1, totalTokens=_CUT_TOKENS)
    app = create_app(settings=settings)
    app.state.routing = make_routing_table(
        fake,
        runtimes=[
            _RuntimeFacts(
                name="engine-1",
                alias=None,
                status="ready",
                node=None,
                url="http://engine",
                context_length=_WINDOW,
                parallel_slots=1,
                idle_unload_seconds=None,
                start_on_demand=False,
                stop_reason=None,
            )
        ],
    )
    with TestClient(app) as c:
        response = c.post("/v1/chat/completions", json=_long_body("m", chars=_CUT_CHARS))
    assert _envelope(response)["prompt_truncated"] is True


def test_the_window_test_is_on_the_chat_stream_too(settings) -> None:  # type: ignore[no-untyped-def]
    fake, app = _windowed_app(settings, max_context_tokens=_WINDOW)
    fake.usage = Usage(promptTokens=_CUT_TOKENS, completionTokens=1, totalTokens=_CUT_TOKENS)
    with TestClient(app) as c:
        response = c.post(
            "/v1/chat/completions", json={**_long_body("m", _CUT_CHARS), "stream": True}
        )
    frames = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    ]
    final = [f for f in frames if f.get("x_eugene_plexus")]
    assert final[-1]["x_eugene_plexus"]["prompt_truncated"] is True


def _anthropic_body(chars: int, stream: bool) -> dict[str, object]:
    return {
        "model": "m",
        "max_tokens": 100,
        "stream": stream,
        "messages": [{"role": "user", "content": "x" * chars}],
    }


def _responses_body(chars: int, stream: bool) -> dict[str, object]:
    return {"model": "m", "input": "x" * chars, "stream": stream}


@pytest.mark.parametrize(
    ("path", "build"),
    [("/v1/messages", _anthropic_body), ("/v1/responses", _responses_body)],
    ids=["anthropic", "responses"],
)
def test_the_other_doors_report_the_window_test(  # type: ignore[no-untyped-def]
    settings, caplog: pytest.LogCaptureFixture, path: str, build
) -> None:
    """Non-streamed, on the envelope's header; streamed, where those doors
    report it -- the log line."""
    fake, app = _windowed_app(settings, max_context_tokens=_WINDOW)
    fake.usage = Usage(promptTokens=_CUT_TOKENS, completionTokens=1, totalTokens=_CUT_TOKENS)
    with TestClient(app) as c:
        response = c.post(path, json=build(_CUT_CHARS, False))
        assert response.status_code == 200, response.text
        assert response.headers["x-eugene-plexus-prompt-truncated"] == "true"
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="eugene_plexus_gateway.routes.inference"):
            streamed = c.post(path, json=build(_CUT_CHARS, True))
    assert streamed.status_code == 200
    assert any("4096-token window" in r.getMessage() for r in caplog.records)


@pytest.fixture
def settings_app_pair(settings):  # type: ignore[no-untyped-def]
    """Two drivers serving one name with different windows."""
    from eugene_plexus_gateway.app import create_app

    small = FakeDriverClient(name="small-box", model_id="shared-model", max_context_tokens=8192)
    big = FakeDriverClient(name="zz-big-box", model_id="shared-model", max_context_tokens=131072)
    app = create_app(settings=settings)
    table = make_routing_table()
    install_snapshot(table, small, big)
    app.state.routing = table
    with TestClient(app) as c:
        yield small, big, c
