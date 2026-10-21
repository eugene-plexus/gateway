"""TLS between machines (LT1): the machine table and the pinned transport.

A real TLS server on loopback with a P-256 key made here, the way a node's
agent makes its machine key. The table says which key it may present.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import ssl
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from eugene_plexus_gateway import _http


def _key_and_cert(tmp: Path, name: str) -> tuple[Path, Path, str]:
    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "eugene-plexus-machine")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(key, hashes.SHA256())
    )
    key_path, cert_path = tmp / f"{name}.key", tmp / f"{name}.crt"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return key_path, cert_path, _http.spki_pin(cert.public_bytes(serialization.Encoding.DER))


class Server:
    """A TLS server that answers every request with its own path, and keeps
    every byte of application data it was sent."""

    def __init__(self, key: Path, cert: Path) -> None:
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.context.load_cert_chain(cert, key)
        self.received = bytearray()
        self.port = 0
        self._server: asyncio.Server | None = None

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                self.received.extend(head)
                path = head.split(b" ", 2)[1]
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n%s" % (len(path), path))
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError, ssl.SSLError):
            pass
        finally:
            writer.close()

    async def __aenter__(self) -> Server:
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0, ssl=self.context)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()


@pytest.fixture
def keys(tmp_path: Path) -> dict[str, tuple[Path, Path, str]]:
    return {name: _key_and_cert(tmp_path, name) for name in ("right", "wrong")}


@pytest.fixture
async def server(keys: dict[str, tuple[Path, Path, str]]) -> AsyncIterator[Server]:
    key, cert, _ = keys["right"]
    async with Server(key, cert) as running:
        yield running


@pytest.fixture(autouse=True)
def _no_table() -> AsyncIterator[None]:
    yield  # type: ignore[misc]
    _http.set_machine_table(None)


def table(port: int, pins: list[str], *, required: bool = False, extra=None) -> dict:
    return {
        "machines": {f"127.0.0.1:{port}": pins},
        "routes": {"http://10.9.9.9:8079": f"https://127.0.0.1:{port}/agent", **(extra or {})},
        "required": required,
        "names": {f"127.0.0.1:{port}": "worker-1"},
    }


async def test_a_plain_request_for_another_machine_is_carried_over_pinned_tls(
    server: Server, keys: dict
) -> None:
    _http.set_machine_table(_http.parse_machine_table(table(server.port, [keys["right"][2]])))
    async with _http.internal_client(timeout=5) as client:
        answer = await client.get("http://10.9.9.9:8079/v1/components?x=1")
    assert answer.status_code == 200
    assert answer.text == "/agent/v1/components?x=1"
    assert f"Host: 127.0.0.1:{server.port}".encode() in server.received
    assert _http.machine_route("http://10.9.9.9:8079/v1/node") == (
        f"https://127.0.0.1:{server.port}/agent/v1/node"
    )


async def test_a_wrong_key_is_refused_before_a_byte_is_sent(server: Server, keys: dict) -> None:
    _http.set_machine_table(_http.parse_machine_table(table(server.port, [keys["wrong"][2]])))
    async with _http.internal_client(timeout=5) as client:
        with pytest.raises(httpx.ConnectError) as caught:
            await client.get(f"https://127.0.0.1:{server.port}/agent/v1/node", headers={"x": "y"})
    assert _http.refused_pin(caught.value) == keys["right"][2]
    assert "worker-1's machine key doesn't match" in str(caught.value)
    await asyncio.sleep(0.1)
    assert bytes(server.received) == b"", "a request byte reached a machine whose key was refused"


async def test_an_unlisted_https_address_is_verified_normally(server: Server, keys: dict) -> None:
    _http.set_machine_table(_http.parse_machine_table(table(server.port + 1, [keys["right"][2]])))
    async with _http.internal_client(timeout=5) as client:
        with pytest.raises(httpx.ConnectError) as caught:
            await client.get(f"https://127.0.0.1:{server.port}/")
    assert "CERTIFICATE_VERIFY_FAILED" in str(caught.value)
    assert _http.refused_pin(caught.value) is None


async def test_plain_http_not_in_the_table_is_untouched() -> None:
    got: list[bytes] = []

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        got.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()
        writer.close()

    plain = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = plain.sockets[0].getsockname()[1]
    _http.set_machine_table(
        _http.parse_machine_table(
            table(1, ["x"], required=True, extra={"http://10.9.9.8:8079": None})
        )
    )
    try:
        async with _http.internal_client(timeout=5) as client:
            assert (await client.get(f"http://127.0.0.1:{port}/plain")).text == "ok"
    finally:
        plain.close()
    assert got and got[0].startswith(b"GET /plain ")


async def test_required_refuses_a_machine_with_no_key_and_sends_nothing() -> None:
    _http.set_machine_table(
        _http.parse_machine_table(
            {
                "machines": {},
                "routes": {"http://10.9.9.8:8079": None},
                "required": True,
                "names": {"http://10.9.9.8:8079": "old-box"},
            }
        )
    )
    async with _http.internal_client(timeout=2) as client:
        with pytest.raises(_http.MachineRefused) as caught:
            await client.get("http://10.9.9.8:8079/v1/node")
    assert "old-box has no machine key on record" in str(caught.value)
    # Not required: the same request goes plain (mixed mode), so it is only
    # unreachable here.
    _http.set_machine_table(
        _http.parse_machine_table({"routes": {"http://10.9.9.8:8079": None}, "required": False})
    )
    assert _http.machine_table().carry(httpx.URL("http://10.9.9.8:8079/v1/node")) is None


def test_required_lets_a_loopback_machine_go_plain() -> None:
    """gateway#17: two agents on one host advertise loopback, so neither has a
    machine key. Traffic to loopback never leaves the host; "required" has
    nothing to protect there, and refusing it left the second agent's
    drivers out."""
    table = _http.parse_machine_table(
        {
            "machines": {},
            "routes": {
                "http://127.0.0.1:19101": None,
                "http://[::1]:19102": None,
                "http://10.9.9.8:8079": None,
            },
            "required": True,
        }
    )
    assert table.carry(httpx.URL("http://127.0.0.1:19101/v1/node")) is None
    assert table.carry(httpx.URL("http://[::1]:19102/v1/node")) is None
    with pytest.raises(_http.MachineRefused):
        table.carry(httpx.URL("http://10.9.9.8:8079/v1/node"))


async def test_a_route_to_an_unpinned_machine_is_dropped() -> None:
    parsed = _http.parse_machine_table(
        {"machines": {}, "routes": {"http://10.9.9.9:8079": "https://10.9.9.9:8083/agent"}}
    )
    assert parsed.routes == {"http://10.9.9.9:8079": None}


async def test_the_sync_client_pins_too(server: Server, keys: dict) -> None:
    def fetch(pin: str) -> httpx.Response:
        _http.set_machine_table(_http.parse_machine_table(table(server.port, [pin])))
        with _http.sync_client_for("http://10.9.9.9:8079", timeout=5) as client:
            return client.get("http://10.9.9.9:8079/v1/node")

    answer = await asyncio.to_thread(fetch, keys["right"][2])
    assert answer.text == "/agent/v1/node"
    server.received.clear()
    with pytest.raises(httpx.ConnectError) as caught:
        await asyncio.to_thread(fetch, keys["wrong"][2])
    assert _http.refused_pin(caught.value) == keys["right"][2]
    await asyncio.sleep(0.1)
    assert bytes(server.received) == b""


async def test_the_table_file_is_read_and_reread(
    tmp_path: Path, server: Server, keys: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "machines.json"
    monkeypatch.setenv(_http.MACHINE_PINS_ENV, str(path))
    monkeypatch.setattr(_http, "_TABLE_RECHECK_SECONDS", 0.0)
    assert _http.machine_table().empty
    path.write_text(json.dumps(table(server.port, [keys["right"][2]])), encoding="utf-8")
    assert _http.machine_pinned(f"https://127.0.0.1:{server.port}")
    path.write_text("{not json", encoding="utf-8")
    assert _http.machine_pinned(f"https://127.0.0.1:{server.port}"), "a broken file kept the last"
    path.unlink()
    assert _http.machine_table().empty


def test_the_private_pool_attribute_is_still_there() -> None:
    # The pinned pool replaces httpx's transport `_pool`. If httpx moves it,
    # this fails, rather than machines quietly being dialled unpinned.
    assert hasattr(httpx.AsyncHTTPTransport(), "_pool")
    assert hasattr(httpx.HTTPTransport(), "_pool")


def test_a_client_with_its_own_transport_keeps_it() -> None:
    mock = httpx.MockTransport(lambda request: httpx.Response(204))
    client = _http.internal_client(transport=mock)
    assert client._transport is mock
    assert isinstance(_http.internal_client()._transport, _http.MachineTransport)
    # A seam left unset is no transport: these went plain before the capture caught it.
    assert isinstance(_http.internal_client(transport=None)._transport, _http.MachineTransport)
    assert not isinstance(_http.egress_client()._transport, _http.MachineTransport)


def test_threads_reading_the_table_agree() -> None:
    _http.set_machine_table(_http.parse_machine_table(table(1, ["p"])))
    seen: list[bool] = []
    threads = [
        threading.Thread(target=lambda: seen.append(_http.machine_pinned("https://127.0.0.1:1")))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert seen == [True] * 8


def test_the_router_keeps_a_loopback_driver_under_required() -> None:
    """gateway#17, the routing half: a driver whose agent advertises loopback
    is reached plain under "required"; one on another machine with no key
    is still left out."""
    from eugene_plexus_gateway import routing

    _http.set_machine_table(_http.parse_machine_table({"required": True}))
    try:
        here = {"advertiseUrl": "http://127.0.0.1:19105"}
        assert routing._remote_url("same-host", here, same_host=True) == here["advertiseUrl"]
        local = {"advertiseUrl": "http://localhost:19106"}
        assert routing._remote_url("named", local, same_host=True) == local["advertiseUrl"]
        far = {"advertiseUrl": "http://10.9.9.8:8091"}
        assert routing._remote_url("far", far, same_host=True) is None
        # Another machine's agent saying "loopback" is not this host's loopback.
        assert routing._remote_url("lying", here, same_host=False) is None
        spelled = {"advertiseUrl": "http://127.0.0.1.evil.example:8091"}
        assert routing._remote_url("spelled", spelled, same_host=True) is None
    finally:
        _http.set_machine_table(_http.parse_machine_table({}))
