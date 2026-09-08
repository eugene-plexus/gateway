"""Priority-list failover for driver slots (v0.2.1).

A driver slot is an ordered list of interchangeable backends. The slot
tries them in order and cascades to the next on a transport error / 5xx
/ timeout, but fails HARD on a 4xx (the next backend would hit the same
bad request). These tests pin that taxonomy.
"""

from __future__ import annotations

import httpx
import pytest

from eugene_plexus_gateway._generated.hemisphere_models import (
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    Problem,
)
from eugene_plexus_gateway.driver_client import (
    DriverError,
    FailoverDriverClient,
)

from .conftest import FakeHemisphereClient


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


def _slot(*candidates: FakeHemisphereClient) -> FailoverDriverClient:
    return FailoverDriverClient(name="left", candidates=list(candidates))


async def test_primary_success_does_not_touch_backup() -> None:
    primary = FakeHemisphereClient(name="left")
    primary.responses = ["primary reply"]
    backup = FakeHemisphereClient(name="left")
    backup.responses = ["backup reply"]

    resp = await _slot(primary, backup).generate(_request())

    assert resp.content == "primary reply"
    assert len(primary.calls) == 1
    assert backup.calls == []  # never reached


async def test_transport_error_cascades_to_backup() -> None:
    primary = FakeHemisphereClient(name="left")
    primary.generate_error = httpx.ConnectError("connection refused")
    backup = FakeHemisphereClient(name="left")
    backup.responses = ["backup reply"]

    resp = await _slot(primary, backup).generate(_request())

    assert resp.content == "backup reply"
    assert len(backup.calls) == 1


async def test_5xx_cascades_to_backup() -> None:
    primary = FakeHemisphereClient(name="left")
    primary.generate_error = _driver_error(503)
    backup = FakeHemisphereClient(name="left")
    backup.responses = ["backup reply"]

    resp = await _slot(primary, backup).generate(_request())

    assert resp.content == "backup reply"


async def test_timeout_cascades_to_backup() -> None:
    primary = FakeHemisphereClient(name="left")
    primary.generate_error = httpx.ReadTimeout("timed out")
    backup = FakeHemisphereClient(name="left")
    backup.responses = ["backup reply"]

    resp = await _slot(primary, backup).generate(_request())

    assert resp.content == "backup reply"


async def test_4xx_fails_hard_without_cascading() -> None:
    """A 4xx is a request/auth/config bug — the next backend would hit
    it identically, so we surface it instead of masking it as 'all
    backends down'."""
    primary = FakeHemisphereClient(name="left")
    primary.generate_error = _driver_error(401)
    backup = FakeHemisphereClient(name="left")
    backup.responses = ["backup reply"]

    with pytest.raises(DriverError) as exc:
        await _slot(primary, backup).generate(_request())

    assert exc.value.status_code == 401
    assert backup.calls == []  # cascade did NOT happen


async def test_all_backends_fail_raises_last_error() -> None:
    primary = FakeHemisphereClient(name="left")
    primary.generate_error = httpx.ConnectError("primary down")
    backup = FakeHemisphereClient(name="left")
    backup.generate_error = _driver_error(502)

    with pytest.raises(DriverError) as exc:
        await _slot(primary, backup).generate(_request())

    # The LAST cascade-eligible failure propagates so the chat route's
    # existing handlers surface it as they would for a single backend.
    assert exc.value.status_code == 502


async def test_single_backend_behaves_like_passthrough() -> None:
    only = FakeHemisphereClient(name="left")
    only.generate_error = _driver_error(500)

    with pytest.raises(DriverError):
        await _slot(only).generate(_request())


async def test_info_failover_returns_first_reachable() -> None:
    primary = FakeHemisphereClient(name="left")
    primary.info_error = httpx.ConnectError("down")
    backup = FakeHemisphereClient(name="left", model_id="backup-model")

    info = await _slot(primary, backup).info()

    assert info.modelId == "backup-model"


async def test_empty_candidates_rejected() -> None:
    with pytest.raises(ValueError, match="at least one backend"):
        FailoverDriverClient(name="left", candidates=[])


async def test_base_url_is_primary() -> None:
    primary = FakeHemisphereClient(name="left", base_url="http://primary")
    backup = FakeHemisphereClient(name="left", base_url="http://backup")
    assert _slot(primary, backup).base_url == "http://primary"


async def test_aclose_closes_every_backend() -> None:
    closed: list[str] = []

    class _Tracking(FakeHemisphereClient):
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

    class _FlakyPrimary(FakeHemisphereClient):
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
    backup = FakeHemisphereClient(name="left")
    backup.responses = ["backup reply"]
    slot = _slot(primary, backup)

    first = await slot.generate(_request())
    second = await slot.generate(_request())

    assert first.content == "backup reply"  # primary down -> failover
    assert second.content == "primary recovered"  # primary back -> used again
