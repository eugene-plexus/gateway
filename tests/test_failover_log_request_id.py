"""The failover warning names the request (CX8).

Metrics shows a request's id and the console links it to the gateway's
log with `?q=<id>`. Before this the line named the slot and the backend
only, so the link found nothing.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from eugene_plexus_gateway import admission
from eugene_plexus_gateway.driver_client import TieredClient


def _cascade(caplog: pytest.LogCaptureFixture) -> str:
    slot = SimpleNamespace(name="qwen", candidates=[object(), object()])
    backend = SimpleNamespace(base_url="http://127.0.0.1:8081")
    with caplog.at_level(logging.WARNING, logger="eugene_plexus_gateway.driver_client"):
        TieredClient._log_cascade(slot, "generate", 0, backend, TimeoutError())  # type: ignore[arg-type]
    return caplog.records[-1].getMessage()


def test_the_failover_line_names_the_request(caplog: pytest.LogCaptureFixture) -> None:
    token = admission.current.set(SimpleNamespace(id="req-7f3a"))  # type: ignore[arg-type]
    try:
        line = _cascade(caplog)
    finally:
        admission.current.reset(token)
    assert "(request req-7f3a)" in line
    assert "trying backend 2/2" in line


def test_without_a_request_the_line_is_as_before(caplog: pytest.LogCaptureFixture) -> None:
    line = _cascade(caplog)
    assert "request" not in line
    assert line.endswith("trying backend 2/2")
