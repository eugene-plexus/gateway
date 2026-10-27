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
from typing import Any

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


async def test_a_route_that_does_not_hold_together_is_refused_not_plain() -> None:
    """agent#38 review: a TLS address the table pins no key for, or one two
    machines claim (`false`), is untrusted, never an older version. Nothing
    goes there, plain or encrypted, token or not."""
    parsed = _http.parse_machine_table(
        {
            "machines": {},
            "routes": {
                "http://10.9.9.9:8079": "https://10.9.9.9:8083/agent",
                "http://10.9.9.6:8079": False,
            },
        }
    )
    assert parsed.routes == {}
    assert parsed.invalid == {"http://10.9.9.9:8079", "http://10.9.9.6:8079"}
    _http.set_machine_table(parsed)
    sent: list[httpx.Request] = []
    async with _recording_client(sent) as client:
        for url in ("http://10.9.9.9:8079/v1/node", "http://10.9.9.6:8079/healthz"):
            with pytest.raises(_http.MachineRefused) as caught:
                await client.get(url)
            assert "does not trust" in str(caught.value)
    assert sent == []


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


def _ep_token() -> str:
    """A token of this install in form: a JWS whose header says `typ: ep-...`."""
    import base64

    def part(value: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'EdDSA', 'typ': 'ep-service+jwt'})}.{part({'sub': 'x'})}.c2ln"


def _recording_client(sent: list[httpx.Request], **options: Any) -> httpx.AsyncClient:
    """An internal client with the real machine transport, whose plain leg
    records what it would put on the wire instead of sending it. (A mock
    handed to `internal_client` as its transport would bypass the rule.)"""

    def plain(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={})

    client = _http.internal_client(timeout=2, **options)
    transport = client._transport
    assert isinstance(transport, _http.MachineTransport)
    transport._plain = httpx.MockTransport(plain)  # type: ignore[assignment]
    return client


@pytest.mark.parametrize(
    "known",
    [
        {},  # the table names no machine yet (the root before its first worker)
        {"http://10.9.9.5:8079": "https://10.9.9.5:8083/agent"},  # names others only
    ],
)
async def test_automatic_never_sends_a_token_in_plain_to_a_machine_not_named_yet(
    known: dict,
) -> None:
    """agent#38: in automatic mode the root's gateway read a worker that had
    just joined in plain HTTP, with its token, because its table (written by
    the root's agent a moment after the root recorded the worker's key) did
    not name the worker yet. Such a request is refused, with nothing sent."""
    machines = {"10.9.9.5:8083": ["p"]} if known else {}
    _http.set_machine_table(
        _http.parse_machine_table({"machines": machines, "routes": known, "required": False})
    )
    sent: list[httpx.Request] = []
    async with _recording_client(sent) as client:
        for path in ("/v1/components", "/v1/runtimes"):
            with pytest.raises(_http.MachineRefused) as caught:
                await client.get(
                    f"http://10.9.9.7:8079{path}",
                    headers={"Authorization": f"Bearer {_ep_token()}"},
                )
            assert "does not know yet whether 10.9.9.7 takes TLS" in str(caught.value)
        with pytest.raises(_http.MachineRefused):
            await client.get("http://10.9.9.7:8090/v1/info", headers={"x-api-key": _ep_token()})
    assert sent == [], "a token went to a machine in plain HTTP"


async def test_automatic_still_reaches_what_it_should_in_plain() -> None:
    """The rule is about tokens to machines not known to speak plain HTTP
    only. A machine that runs an older version (its route is null) is reached
    in plain HTTP, its drivers too, as LT1 allows for mixed versions; a
    request with no token of this install goes as before; and so does this
    machine under its own LAN address."""
    _http.set_machine_table(
        _http.parse_machine_table(
            {
                "routes": {"http://10.9.9.8:8079": None},
                "required": False,
                "local": ["http://10.9.9.2:8079"],
            }
        )
    )
    token = {"Authorization": f"Bearer {_ep_token()}"}
    sent: list[httpx.Request] = []
    async with _recording_client(sent, machine_listed_by="http://10.9.9.8:8079") as client:
        await client.get("http://10.9.9.8:8091/v1/info", headers=token)  # a driver it listed
    async with _recording_client(sent) as client:
        await client.get("http://10.9.9.8:8079/v1/components", headers=token)
        await client.get("http://10.9.9.7:8000/v1/models")  # no token: a model server
        await client.get("http://10.9.9.7:8079/v1/node", headers={"Authorization": "Bearer sk-1"})
        await client.get("http://10.9.9.2:8079/v1/node", headers=token)  # this machine
        await client.get("http://127.0.0.1:8081/v1/info", headers=token)
    assert [str(r.url) for r in sent] == [
        "http://10.9.9.8:8091/v1/info",
        "http://10.9.9.8:8079/v1/components",
        "http://10.9.9.7:8000/v1/models",
        "http://10.9.9.7:8079/v1/node",
        "http://10.9.9.2:8079/v1/node",
        "http://127.0.0.1:8081/v1/info",
    ]
    # Under required the older machine is refused, as before.
    _http.set_machine_table(
        _http.parse_machine_table({"routes": {"http://10.9.9.8:8079": None}, "required": True})
    )
    sent.clear()
    async with _recording_client(sent) as client:
        for url in ("http://10.9.9.8:8079/v1/components", "http://10.9.9.8:8091/v1/info"):
            with pytest.raises(_http.MachineRefused):
                await client.get(url, headers=token)
    assert sent == []


async def test_with_no_table_at_all_nothing_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process no machine-port agent started has no table: as before LT1."""
    monkeypatch.delenv(_http.MACHINE_PINS_ENV, raising=False)
    _http.set_machine_table(None)
    sent: list[httpx.Request] = []
    async with _recording_client(sent) as client:
        await client.get(
            "http://10.9.9.7:8079/v1/node", headers={"Authorization": f"Bearer {_ep_token()}"}
        )
    assert len(sent) == 1


async def test_a_table_file_named_but_not_written_yet_refuses_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agent names the file for its children; until it holds a machine,
    that machine is not known, so no token goes to it in plain HTTP."""
    monkeypatch.setenv(_http.MACHINE_PINS_ENV, str(tmp_path / "machine_pins.json"))
    monkeypatch.setattr(_http, "_TABLE_RECHECK_SECONDS", 0.0)
    _http.set_machine_table(None)
    sent: list[httpx.Request] = []
    async with _recording_client(sent) as client:
        with pytest.raises(_http.MachineRefused):
            await client.get(
                "http://10.9.9.7:8079/v1/node",
                headers={"Authorization": f"Bearer {_ep_token()}"},
            )
    assert sent == []


async def test_an_older_machine_is_reached_in_plain_at_its_own_address_only() -> None:
    """agent#38 review: a null route lets plain HTTP through to that address,
    not to every port on its host, which may be another machine behind the
    same name. A driver of that machine is reached because the client that
    dials it says the older machine's own agent listed it."""
    _http.set_machine_table(
        _http.parse_machine_table(
            {
                "machines": {"10.9.9.5:8083": ["p"]},
                "routes": {
                    "http://nat.example:8079": None,
                    "http://10.9.9.5:8079": "https://10.9.9.5:8083/agent",
                },
            }
        )
    )
    token = {"Authorization": f"Bearer {_ep_token()}"}
    sent: list[httpx.Request] = []
    async with _recording_client(sent) as client:
        await client.get("http://nat.example:8079/v1/components", headers=token)
        for url in ("http://nat.example:9079/v1/node", "http://nat.example:8091/v1/info"):
            with pytest.raises(_http.MachineRefused):
                await client.get(url, headers=token)
    async with _recording_client(sent, machine_listed_by="http://nat.example:8079") as client:
        await client.get("http://nat.example:8091/v1/info", headers=token)
        with pytest.raises(_http.MachineRefused):  # listed on another host
            await client.get("http://10.9.9.4:8091/v1/info", headers=token)
    async with _recording_client(sent, machine_listed_by="http://10.9.9.5:8079") as client:
        with pytest.raises(_http.MachineRefused):  # its lister takes TLS
            await client.get("http://10.9.9.5:8091/v1/info", headers=token)
    assert [str(r.url) for r in sent] == [
        "http://nat.example:8079/v1/components",
        "http://nat.example:8091/v1/info",
    ]


async def test_a_machine_hop_goes_by_its_route_alone() -> None:
    """agent#38 review: the console's proxy forwards what a person sent, a
    passphrase in a body or a cookie, not only tokens. A machine-hop client
    sends nothing in plain HTTP to a machine the table does not name, token
    or not; an older machine is still reached; loopback is not its business."""
    _http.set_machine_table(
        _http.parse_machine_table({"routes": {"http://10.9.9.8:8079": None}, "required": False})
    )
    login = {"json": {"passphrase": "hunter2"}, "headers": {"Cookie": "ep_session=x"}}
    sent: list[httpx.Request] = []
    async with _recording_client(sent, machine_hop=True) as client:
        with pytest.raises(_http.MachineRefused) as caught:
            await client.post("http://10.9.9.7:8079/v1/auth/login", **login)
        assert "carries what a person sent" in str(caught.value)
        await client.post("http://10.9.9.8:8079/v1/auth/login", **login)
        await client.post("http://127.0.0.1:8079/v1/auth/login", **login)
    async with _recording_client(sent) as client:  # not a hop: no token, no rule
        await client.post("http://10.9.9.7:8079/v1/auth/login", **login)
    assert [str(r.url) for r in sent] == [
        "http://10.9.9.8:8079/v1/auth/login",
        "http://127.0.0.1:8079/v1/auth/login",
        "http://10.9.9.7:8079/v1/auth/login",
    ]


def test_a_machine_in_the_table_gets_the_machine_transport_whatever_its_name() -> None:
    """agent#38 review: a root reached as `root.lan` (a dotted name, so not
    "internal") got the egress client, and its Bearer went out unrouted."""
    _http.set_machine_table(
        _http.parse_machine_table({"routes": {"http://root.lan:8083": None}, "required": True})
    )
    client = _http.client_for("http://root.lan:8083", timeout=2)
    assert isinstance(client._transport, _http.MachineTransport)
    assert isinstance(
        _http.sync_client_for("http://root.lan:8083")._transport, _http.SyncMachineTransport
    )
    assert not isinstance(
        _http.client_for("http://elsewhere.example")._transport, _http.MachineTransport
    )


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


async def test_the_router_names_the_agent_that_listed_each_driver() -> None:
    """agent#38 review: a driver on a machine with no machine port is reached
    in plain HTTP only because its own agent listed it; the gateway's driver
    and search clients say which agent that was."""
    from eugene_plexus_gateway.routing import RoutingTable

    table = RoutingTable(agent_url="http://127.0.0.1:1")
    table._listing_agents = {"old-box": "http://10.9.9.8:8079"}
    try:
        await table._probe("old-box", "llm", "http://127.0.0.1:1")
        await table._probe_tools([("old-box", "search", "http://127.0.0.1:1")])
        driver = table._clients[("old-box", "llm", "http://127.0.0.1:1")]
        tool = table._tool_clients[("old-box", "search", "http://127.0.0.1:1")]
        for client in (driver._client, tool._client):
            transport = client._transport
            assert isinstance(transport, _http.MachineTransport)
            assert transport.listed_by == "http://10.9.9.8:8079"
    finally:
        await table.aclose()
