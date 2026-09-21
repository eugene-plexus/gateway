"""A6b: silence is not proof that a request had no effects."""

import asyncio

import httpx
import pytest

from eugene_plexus_gateway._generated.driver_models import GenerateRequest, Problem
from eugene_plexus_gateway.circuit import Circuit
from eugene_plexus_gateway.driver_client import DriverError, FailoverDriverClient, StreamEvent

from .conftest import FakeDriverClient


@pytest.mark.parametrize(
    "failure",
    [
        httpx.ReadError("connection lost after acceptance"),
        httpx.WriteError("partial body"),
        httpx.RemoteProtocolError("no final reply"),
        DriverError(
            driver_name="primary",
            driver_url="http://primary",
            status_code=503,
            problem=Problem(type="about:blank", title="unknown outcome", status=503),
            raw_body="",
        ),
    ],
)
@pytest.mark.parametrize("operation", ["generate", "stream", "embed"])
async def test_ambiguous_attempt_is_never_replayed(failure, operation):
    first = FakeDriverClient(name="primary")
    first.generate_error = first.embed_error = failure
    first.stream_error_after, first.stream_error = 0, failure
    second = FakeDriverClient(name="fallback")
    client = FailoverDriverClient(name="alias", candidates=[first, second])
    with pytest.raises(type(failure)):
        if operation == "stream":
            _ = [event async for event in client.stream(GenerateRequest(messages=[]))]
        elif operation == "embed":
            from eugene_plexus_gateway._generated.driver_models import EmbedRequest

            await client.embed(EmbedRequest(input=["act"]))
        else:
            await client.generate(GenerateRequest(messages=[{"role": "user", "content": "act"}]))
    assert second.calls == [] and second.embed_calls == 0


def refusal(delay=30):
    return DriverError(
        driver_name="primary",
        driver_url="http://primary",
        status_code=502,
        problem=Problem(
            type="about:blank",
            title="refused before execution",
            status=502,
            retryDisposition="safe",
            retryAfterSeconds=delay,
        ),
        raw_body="",
    )


@pytest.mark.parametrize("operation", ["generate", "stream", "embed"])
async def test_cooldown_skips_primary_and_recovers_with_two_probes(monkeypatch, operation):
    from eugene_plexus_gateway import circuit as module
    from eugene_plexus_gateway._generated.driver_models import EmbedRequest

    now = [100.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: now[0])
    first, second = FakeDriverClient(name="primary"), FakeDriverClient(name="backup")
    first.circuit = Circuit()
    first.generate_error = first.embed_error = refusal()
    if operation == "stream":
        first.stream_error_after, first.stream_error = 0, refusal()
    request = (
        EmbedRequest(input=["act"])
        if operation == "embed"
        else GenerateRequest(messages=[{"role": "user", "content": "act"}])
    )

    async def invoke():
        client = FailoverDriverClient(name="alias", candidates=[first, second])
        if operation == "stream":
            return [event async for event in client.stream(request)]
        return await getattr(client, operation)(request)

    await invoke()
    assert first.circuit.until == 130
    calls = len(first.calls) + first.embed_calls
    await invoke()
    assert len(first.calls) + first.embed_calls == calls
    now[0] = 130
    assert first.circuit.acquire()
    await invoke()  # another request cannot steal the half-open probe
    assert len(first.calls) + first.embed_calls == calls
    first.circuit.probing = False
    first.generate_error = first.embed_error = first.stream_error_after = None
    await invoke()
    assert first.circuit.failures and first.circuit.until == 131
    await invoke()  # still cooling between successful probes
    now[0] = 131
    await invoke()
    assert first.circuit.failures == 0


@pytest.mark.parametrize(
    "fragment",
    [
        StreamEvent(text="partial"),
        StreamEvent(tool_calls=[{"index": 0, "function": {"arguments": "{"}}]),
    ],
)
async def test_no_fallback_after_text_or_tool_fragment_even_with_safe_error(fragment):
    first, second = FakeDriverClient(name="primary"), FakeDriverClient(name="backup")

    async def interrupted(request):
        yield fragment
        raise refusal()

    first.stream = interrupted
    events = []
    with pytest.raises(DriverError):
        async for event in FailoverDriverClient(name="alias", candidates=[first, second]).stream(
            GenerateRequest(messages=[])
        ):
            events.append(event)
    assert events == [fragment] and not second.calls


def test_retry_hints_are_bounded_and_malformed_hints_do_not_disable_cooldown(monkeypatch):
    from eugene_plexus_gateway import circuit as module

    monkeypatch.setattr(module.time, "perf_counter", lambda: 100)
    c = Circuit()
    c.finish(failed=True, retry_after=999999)
    assert c.until == 400
    c.finish(failed=True, retry_after=float("nan"))
    assert c.until == 400  # another failure cannot shorten an existing provider delay


def test_inflight_successes_cannot_clear_a_newer_failure_cooldown(monkeypatch):
    from eugene_plexus_gateway import circuit as module

    monkeypatch.setattr(module.time, "perf_counter", lambda: 100)
    c = Circuit()
    assert c.acquire() and c.acquire()  # two requests already running normally
    c.finish(failed=True, retry_after=30)
    c.finish(failed=False)
    c.finish(failed=False)
    assert c.until == 130 and c.failures == 1
    assert not c.acquire()


def test_new_failure_cannot_admit_a_second_probe_until_the_old_one_finishes(monkeypatch):
    from eugene_plexus_gateway import circuit as module

    now = [100]
    monkeypatch.setattr(module.time, "perf_counter", lambda: now[0])
    c = Circuit()
    c.finish(failed=True)
    now[0] = 102
    assert c.acquire()
    owner = c.epoch
    c.finish(failed=True, retry_after=30)  # older regular request fails during the probe
    now[0] = 140
    assert not c.acquire()  # even after the new cooldown, the first probe still owns the slot
    c.finish(failed=False, probe_epoch=owner)
    assert c.successes == 0 and not c.probing  # stale success releases, never restores traffic
    assert c.acquire()


@pytest.mark.parametrize("operation", ["generate", "stream", "embed", "close-stream"])
async def test_client_cancellation_does_not_cool_a_healthy_backend(operation):
    from eugene_plexus_gateway._generated.driver_models import EmbedRequest

    first = FakeDriverClient(name="primary")
    first.circuit = Circuit()
    client = FailoverDriverClient(name="model", candidates=[first])
    if operation == "close-stream":
        stream = client.stream(GenerateRequest(messages=[]))
        await anext(stream)
        await stream.aclose()
    else:
        first.generate_error = first.embed_error = asyncio.CancelledError()
        first.stream_error_after, first.stream_error = 0, asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            if operation == "stream":
                _ = [event async for event in client.stream(GenerateRequest(messages=[]))]
            elif operation == "embed":
                await client.embed(EmbedRequest(input=["cancel me"]))
            else:
                await client.generate(GenerateRequest(messages=[]))
    assert first.circuit.failures == 0
    first.generate_error = None
    assert await FailoverDriverClient(name="model", candidates=[first]).generate(
        GenerateRequest(messages=[])
    )


async def test_cancelled_recovery_probe_is_neither_success_nor_failure(monkeypatch):
    from eugene_plexus_gateway import circuit as module

    now = [100.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: now[0])
    first = FakeDriverClient(name="primary")
    first.circuit = Circuit()
    first.circuit.finish(failed=True, retry_after=30)
    now[0] = 131
    first.generate_error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await FailoverDriverClient(name="model", candidates=[first]).generate(
            GenerateRequest(messages=[])
        )
    assert not first.circuit.probing
    assert first.circuit.failures == 1 and first.circuit.successes == 0
    assert first.circuit.until == 130
    first.generate_error = None
    await FailoverDriverClient(name="model", candidates=[first]).generate(
        GenerateRequest(messages=[])
    )
    assert first.circuit.failures == 1 and first.circuit.successes == 1
