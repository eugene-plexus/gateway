"""Priority-list failover for driver slots.

A driver slot is an ordered list of interchangeable backends. The slot
tries them in order and cascades to the next on a transport error or a
5xx, but fails HARD on a 4xx (the next backend would hit the same bad
request) and on a deadline that fired (the next backend would take the
same time to do the same work — R2.5). These tests pin that taxonomy.

**A read timeout used to be in the cascading bucket and this file
asserted it**, which is R2.5's instance of the pattern that produced six
of the review's findings: the defect encoded as intent, with a green
test sitting beside it. `test_timeout_is_not_a_failure.py` is where the
corrected rule lives, including the controls that keep a dead host
cascading.
"""

from __future__ import annotations

import httpx
import pytest

from eugene_plexus_gateway._generated.driver_models import (
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    Problem,
)
from eugene_plexus_gateway.driver_client import (
    DriverError,
    FailoverDriverClient,
)

from .conftest import FakeDriverClient


def _request() -> GenerateRequest:
    return GenerateRequest(messages=[{"role": "user", "content": "hi"}])


def _driver_error(status_code: int) -> DriverError:
    return DriverError(
        driver_name="slot",
        driver_url="http://backend",
        status_code=status_code,
        problem=Problem(type="about:blank", title="boom", status=status_code),
        raw_body="",
    )


def _slot(*candidates: FakeDriverClient) -> FailoverDriverClient:
    return FailoverDriverClient(name="left", candidates=list(candidates))


async def test_primary_success_does_not_touch_backup() -> None:
    primary = FakeDriverClient(name="left")
    primary.responses = ["primary reply"]
    backup = FakeDriverClient(name="left")
    backup.responses = ["backup reply"]

    resp = await _slot(primary, backup).generate(_request())

    assert resp.content == "primary reply"
    assert len(primary.calls) == 1
    assert backup.calls == []  # never reached


async def test_transport_error_cascades_to_backup() -> None:
    primary = FakeDriverClient(name="left")
    primary.generate_error = httpx.ConnectError("connection refused")
    backup = FakeDriverClient(name="left")
    backup.responses = ["backup reply"]

    resp = await _slot(primary, backup).generate(_request())

    assert resp.content == "backup reply"
    assert len(backup.calls) == 1


async def test_5xx_cascades_to_backup() -> None:
    primary = FakeDriverClient(name="left")
    primary.generate_error = _driver_error(503)
    backup = FakeDriverClient(name="left")
    backup.responses = ["backup reply"]

    resp = await _slot(primary, backup).generate(_request())

    assert resp.content == "backup reply"


async def test_a_connect_timeout_cascades_to_backup() -> None:
    """A host that never accepted the connection took no work, so the
    next backend is a rescue. Contrast the READ timeout, which is not:
    see `test_timeout_is_not_a_failure.py`."""
    primary = FakeDriverClient(name="left")
    primary.generate_error = httpx.ConnectTimeout("no route to host")
    backup = FakeDriverClient(name="left")
    backup.responses = ["backup reply"]

    resp = await _slot(primary, backup).generate(_request())

    assert resp.content == "backup reply"


async def test_4xx_fails_hard_without_cascading() -> None:
    """A 4xx is a request/auth/config bug — the next backend would hit
    it identically, so we surface it instead of masking it as 'all
    backends down'."""
    primary = FakeDriverClient(name="left")
    primary.generate_error = _driver_error(401)
    backup = FakeDriverClient(name="left")
    backup.responses = ["backup reply"]

    with pytest.raises(DriverError) as exc:
        await _slot(primary, backup).generate(_request())

    assert exc.value.status_code == 401
    assert backup.calls == []  # cascade did NOT happen


async def test_all_backends_fail_raises_last_error() -> None:
    primary = FakeDriverClient(name="left")
    primary.generate_error = httpx.ConnectError("primary down")
    backup = FakeDriverClient(name="left")
    backup.generate_error = _driver_error(502)

    with pytest.raises(DriverError) as exc:
        await _slot(primary, backup).generate(_request())

    # The LAST cascade-eligible failure propagates so the chat route's
    # existing handlers surface it as they would for a single backend.
    assert exc.value.status_code == 502


async def test_single_backend_behaves_like_passthrough() -> None:
    only = FakeDriverClient(name="left")
    only.generate_error = _driver_error(500)

    with pytest.raises(DriverError):
        await _slot(only).generate(_request())


async def test_info_failover_returns_first_reachable() -> None:
    primary = FakeDriverClient(name="left")
    primary.info_error = httpx.ConnectError("down")
    backup = FakeDriverClient(name="left", model_id="backup-model")

    info = await _slot(primary, backup).info()

    assert info.modelId == "backup-model"


async def test_empty_candidates_rejected() -> None:
    with pytest.raises(ValueError, match="at least one backend"):
        FailoverDriverClient(name="left", candidates=[])


async def test_base_url_is_primary() -> None:
    primary = FakeDriverClient(name="left", base_url="http://primary")
    backup = FakeDriverClient(name="left", base_url="http://backup")
    assert _slot(primary, backup).base_url == "http://primary"


async def test_aclose_closes_every_backend() -> None:
    closed: list[str] = []

    class _Tracking(FakeDriverClient):
        async def aclose(self) -> None:
            closed.append(self.base_url)

    primary = _Tracking(name="left", base_url="http://primary")
    backup = _Tracking(name="left", base_url="http://backup")
    await _slot(primary, backup).aclose()

    assert closed == ["http://primary", "http://backup"]


async def test_retry_walks_from_top_each_call() -> None:
    """Per-turn-attempt granularity: a transiently-down primary that
    recovers is used again on the next call rather than being stuck on
    the backup."""

    class _FlakyPrimary(FakeDriverClient):
        def __init__(self) -> None:
            super().__init__(name="left")
            self.fail_next = True

        async def generate(self, request: GenerateRequest) -> GenerateResponse:
            self.calls.append(request)
            if self.fail_next:
                self.fail_next = False
                raise httpx.ConnectError("transient")
            return GenerateResponse(
                content="primary recovered",
                finishReason=FinishReason.stop,
                backend=self.backend,
                modelId=self.model_id,
                latencyMs=1,
            )

    primary = _FlakyPrimary()
    backup = FakeDriverClient(name="left")
    backup.responses = ["backup reply"]
    slot = _slot(primary, backup)

    first = await slot.generate(_request())
    second = await slot.generate(_request())

    assert first.content == "backup reply"  # primary down -> failover
    assert second.content == "primary recovered"  # primary back -> used again


# --------------------------------------------------------------------------- #
# The commit point (M10)
# --------------------------------------------------------------------------- #
#
# Streaming breaks the assumption every test above rests on. `generate()`
# may retry freely because nothing has reached the client until it
# returns; a stream that has emitted a token has already sent part of an
# answer, and appending another model's output to it would splice two
# models together with no marker at the seam -- a wrong answer that looks
# like a right one. So the rule is: cascade until the first token, never
# after it. These two tests are the rule.


async def test_a_stream_that_fails_before_the_first_token_still_cascades() -> None:
    """Nothing has been forwarded, so this is an ordinary failure."""
    primary = FakeDriverClient(name="left")
    primary.generate_error = httpx.ConnectError("connection refused")
    backup = FakeDriverClient(name="left")
    backup.responses = ["backup reply"]

    slot = _slot(primary, backup)
    events = [event async for event in slot.stream(_request())]

    assert "".join(e.text for e in events if not e.done) == "backup reply"
    assert events[-1].done
    assert slot.served_by == "left"


async def test_a_stream_that_fails_after_the_first_token_truncates() -> None:
    """The commit point. The backup must not be reached, and the tokens
    already emitted must survive -- truncating is honest, discarding what
    the user has already seen is not."""
    primary = FakeDriverClient(name="left")
    primary.responses = ["one two three"]
    primary.stream_error_after = 1  # a token is out, then it dies
    backup = FakeDriverClient(name="left")
    backup.responses = ["backup reply"]

    slot = _slot(primary, backup)
    seen: list[str] = []
    with pytest.raises(RuntimeError):
        async for event in slot.stream(_request()):
            if not event.done:
                seen.append(event.text)

    assert seen == ["one "]
    # The whole point: no second backend was tried.
    assert backup.calls == []


async def test_a_stream_failing_before_the_first_token_is_not_the_same_as_after() -> None:
    """Same backend, same error, opposite outcomes -- decided only by
    whether a token had been emitted. Pinned as one test because the two
    behaviours are one rule, and a change that broke the pairing would
    otherwise still pass one of them."""
    for error_after, expect_cascade in ((0, True), (1, False)):
        primary = FakeDriverClient(name="left")
        primary.responses = ["one two"]
        primary.stream_error_after = error_after
        # A *cascade-eligible* error in both runs, so position is the
        # only variable. With a plain RuntimeError the first run would
        # not cascade either -- for an unrelated and correct reason --
        # and the test would prove nothing about the commit point.
        primary.stream_error = httpx.ConnectError("connection refused")
        backup = FakeDriverClient(name="left")
        backup.responses = ["backup reply"]
        slot = _slot(primary, backup)

        if expect_cascade:
            events = [e async for e in slot.stream(_request())]
            assert "".join(e.text for e in events if not e.done) == "backup reply"
            assert backup.calls != []
        else:
            with pytest.raises(httpx.ConnectError):
                async for _ in slot.stream(_request()):
                    pass
            assert backup.calls == []
