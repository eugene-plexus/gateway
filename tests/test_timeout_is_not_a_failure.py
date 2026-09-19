"""R2.5 — a backend that is still computing has not failed.

The reproduction, written before the fix (roadmap §1).

**What the cascade is for, and what it is not for.** A backend that
refused the connection, or answered 500, or is a cloud provider handing
back a 429, is one the *next* backend can rescue: it did no work and it
will do none. A backend that has not answered within the deadline is
different in kind — it is almost certainly still computing, holding a
GPU or a CPU's worth of the same prompt, and the next replica will take
the same time to do the same work. Cascading there does not rescue the
request; it triples the cost of it and then reports
``Every backend serving 'x' failed``.

Measured shape of the defect: a 30B on CPU behind a 120 s driver
deadline and a 180 s gateway one loses at 120 s, is retried, loses
again, and is declared a total failure at 240 s with both engines
having computed the answer.

These tests pin **timeouts do not cascade** at two seams: the gateway's
own read timeout onto a driver, and a 504 the driver itself returns.
`test_timeout_cascades_to_backup` in `test_failover.py` asserts the
opposite and is rewritten with this slice — the defect was encoded as
intent, which is R2.5's instance of the pattern that has produced six
of these findings.
"""

from __future__ import annotations

import httpx
import pytest

from eugene_plexus_gateway._generated.driver_models import (
    GenerateRequest,
    Problem,
)
from eugene_plexus_gateway.driver_client import DriverError, FailoverDriverClient

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


async def test_a_read_timeout_does_not_recompute_on_the_next_replica() -> None:
    primary = FakeDriverClient(name="left")
    primary.generate_error = httpx.ReadTimeout("")
    backup = FakeDriverClient(name="left")
    backup.responses = ["backup reply"]

    with pytest.raises(httpx.ReadTimeout):
        await _slot(primary, backup).generate(_request())

    assert backup.calls == [], "the same prompt was sent to a second engine"


async def test_a_504_from_the_driver_does_not_recompute_either() -> None:
    """The driver's own deadline fired: it is still computing, not broken."""
    primary = FakeDriverClient(name="left")
    primary.generate_error = _driver_error(504)
    backup = FakeDriverClient(name="left")
    backup.responses = ["backup reply"]

    with pytest.raises(DriverError):
        await _slot(primary, backup).generate(_request())

    assert backup.calls == []


async def test_a_dead_cloud_backend_still_cascades() -> None:
    """The decision that motivated failover was a real OpenRouter 429,
    which the driver maps to 502. Nothing about timeouts may weaken it."""
    primary = FakeDriverClient(name="left")
    primary.generate_error = _driver_error(502)
    backup = FakeDriverClient(name="left")
    backup.responses = ["backup reply"]

    assert (await _slot(primary, backup).generate(_request())).content == "backup reply"


async def test_a_refused_connection_still_cascades() -> None:
    primary = FakeDriverClient(name="left")
    primary.generate_error = httpx.ConnectError("connection refused")
    backup = FakeDriverClient(name="left")
    backup.responses = ["backup reply"]

    assert (await _slot(primary, backup).generate(_request())).content == "backup reply"


async def test_a_connect_timeout_is_a_dead_backend_and_does_cascade() -> None:
    """The one timeout that is NOT 'still computing'. Nothing was ever
    handed to the engine, so the next backend is a real rescue — and a
    host that is down is exactly what failover exists for."""
    primary = FakeDriverClient(name="left")
    primary.generate_error = httpx.ConnectTimeout("no route to host")
    backup = FakeDriverClient(name="left")
    backup.responses = ["backup reply"]

    assert (await _slot(primary, backup).generate(_request())).content == "backup reply"


async def test_embeddings_do_not_recompute_on_a_timeout_either() -> None:
    from eugene_plexus_gateway._generated.driver_models import EmbedRequest

    primary = FakeDriverClient(name="left", supports_embeddings=True)
    primary.embed_error = httpx.ReadTimeout("")
    backup = FakeDriverClient(name="left", supports_embeddings=True)

    with pytest.raises(httpx.ReadTimeout):
        await _slot(primary, backup).embed(EmbedRequest(input=["hi"]))

    assert backup.embed_calls == 0


def test_the_gateway_holds_the_SHORTER_of_the_two_deadlines() -> None:
    """Two deadlines sit on this path and their ORDER is the defect.

    Until R2.5 the driver's 120 s fired before the gateway's 180 s, so
    the knob the docs and the UI point an operator at governed nothing:
    raising it changed which layer reported the failure and not when the
    failure happened. The rule is that the front door owns the answer,
    which means this number must stay BELOW the driver's own
    `DEFAULT_REQUEST_TIMEOUT_SECONDS` (660, in
    `inference-driver/engines/base.py`).

    Cross-repo, so the other half of the comparison is a literal here
    and a real read in `scripts/still-computing-acceptance.sh`, which
    asks both running processes what they actually think.
    """
    from eugene_plexus_gateway.config import DEFAULT_REQUEST_TIMEOUT_SECONDS, FIELDS

    field = next(f for f in FIELDS if f.key == "requestTimeoutSeconds")
    assert field.default == DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert DEFAULT_REQUEST_TIMEOUT_SECONDS < 660.0, (
        "the driver's backstop is 660 s; a gateway deadline at or above it puts the "
        "operator's knob behind one they were never told about"
    )
    # Long enough to be about a model rather than about a network: a 30B
    # on CPU answering a long prompt is minutes, and the review's own
    # case was a request killed at 240 s with the answer computed.
    assert DEFAULT_REQUEST_TIMEOUT_SECONDS >= 300.0


def test_the_timeout_default_is_written_in_one_place() -> None:
    """It was written in three and drifted. `RoutingTable`'s signature
    default and `app.py`'s fallback both read the constant now."""
    import inspect

    from eugene_plexus_gateway.config import DEFAULT_REQUEST_TIMEOUT_SECONDS
    from eugene_plexus_gateway.routing import RoutingTable

    signature = inspect.signature(RoutingTable.__init__)
    assert (
        signature.parameters["request_timeout_seconds"].default == DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
