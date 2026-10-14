"""Expire an idle socket before common upstream keep-alive deadlines."""

import asyncio
from types import SimpleNamespace
from typing import Any

import httpcore._async.http11
import httpx
import pytest

from eugene_plexus_gateway import _http


@pytest.mark.parametrize("factory", ["internal_client", "egress_client", "client_for"])
async def test_post_after_idle_uses_a_fresh_connection(
    monkeypatch: pytest.MonkeyPatch, factory: str
) -> None:
    # Advance only HTTP/1.1's idle clock, without sleeping or changing the
    # event loop's clock. The server keeps the first socket open: the client
    # must retire it before a server's two-second idle close can race a POST.
    clock = [0.0]
    monkeypatch.setattr(httpcore._async.http11, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    connections = 0
    tasks: list[asyncio.Task[None]] = []

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal connections
        connections += 1
        number = connections
        task = asyncio.current_task()
        assert task is not None
        tasks.append(task)
        try:
            while True:
                await reader.readuntil(b"\r\n\r\n")
                body = str(number).encode()
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Length: "
                    + str(len(body)).encode()
                    + b"\r\n\r\n"
                    + body
                )
                await writer.drain()
        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    make = getattr(_http, factory)
    args = (url,) if factory == "client_for" else ()
    try:
        async with make(*args, trust_env=False) as client:
            assert (await client.post(url)).text == "1"
            clock[0] = 1.0
            assert (await client.post(url)).text == "1", "active connections should be reused"
            clock[0] = 2.6
            assert (await client.post(url)).text == "2", "idle connection was reused after 1.6 s"
    finally:
        server.close()
        await server.wait_closed()
        await asyncio.gather(*tasks)


def test_sync_pool_expires_before_two_seconds_and_explicit_limits_are_kept() -> None:
    with _http.sync_client_for("http://localhost") as client:
        pool: Any = client._transport._pool
        assert 0 < pool._keepalive_expiry < 2
    limits = httpx.Limits(max_connections=7, keepalive_expiry=0.5)
    with _http.sync_client_for("http://localhost", limits=limits) as client:
        pool = client._transport._pool
        assert pool._max_connections == 7
        assert pool._keepalive_expiry == 0.5
