"""Row 48b: saved settings survive an outage and a process restart."""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_gateway._generated.models import ConfigUpdateRequest
from eugene_plexus_gateway.config import ConfigStore
from eugene_plexus_gateway.control_tokens import SpecialTokenSpellings

from .test_profile_defaults import Library, resolver


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500])
async def test_a_refused_read_keeps_saved_values(status: int) -> None:
    """A refusal (a rejected credential, a Library error) says nothing about
    the model's settings: only a successful answer replaces them."""
    library = Library()
    now = [0.0]
    profiles = resolver(library, clock=lambda: now[0])
    try:
        saved = await profiles.get("/models/a.gguf")
        assert saved
        library.status = status
        now[0] = 400
        assert await profiles.get("/models/a.gguf") == saved
    finally:
        await profiles.aclose()


async def test_a_stop_writes_what_changed_in_the_last_minute(tmp_path: Path) -> None:
    from eugene_plexus_gateway.library_cache import LibraryCache

    now = [0.0]
    path = tmp_path / "library-cache.json"
    cache = LibraryCache(path, clock=lambda: now[0])
    cache.update("profiles", "/models/a.gguf", {"temperature": 0.2})
    await cache.flush()
    now[0] = 10
    cache.update("profiles", "/models/a.gguf", {"temperature": 0.3})
    await cache.aclose()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["profiles"]["/models/a.gguf"]["value"] == {"temperature": 0.3}


@pytest.mark.parametrize("limit", [0, 12])
async def test_explicit_stale_limit_is_kept(limit: int) -> None:
    library = Library()
    now = [0.0]
    profiles = resolver(library, {"profileMaxStaleSeconds": limit}, clock=lambda: now[0])
    try:
        assert await profiles.get("/models/a.gguf")
        library.status = 502
        now[0] = 31
        assert bool(await profiles.get("/models/a.gguf")) is (limit > 0)
        now[0] = 43
        assert await profiles.get("/models/a.gguf") == {}
    finally:
        await profiles.aclose()


def test_default_outage_limit_is_empty_and_patch_keeps_expert_choice(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "gateway.yaml")
    assert store.get("profileMaxStaleSeconds") is None
    for limit in (0, 17, None):
        result = store.apply_patch(
            ConfigUpdateRequest.model_validate({"profileMaxStaleSeconds": limit})
        )
        assert not result.rejected
        assert store.get("profileMaxStaleSeconds") == limit
        reloaded = ConfigStore(tmp_path / "gateway.yaml")
        reloaded.load()
        assert reloaded.get("profileMaxStaleSeconds") == limit


async def test_profile_and_tokens_survive_restart_as_stale(tmp_path: Path) -> None:
    from eugene_plexus_gateway.library_cache import LibraryCache

    library = Library()
    path = "/models/a.gguf"

    async def request(req: httpx.Request) -> httpx.Response:
        if library.status == 200 and req.url.path.endswith("/special-tokens"):
            return httpx.Response(200, json={"tokens": ["<|end|>"]})
        return await library.request(req)

    clock = [0.0]
    cache = LibraryCache(tmp_path / "library-cache.json", clock=lambda: clock[0])
    profiles = resolver(library, saved=cache)
    tokens = SpecialTokenSpellings(
        "http://agent.invalid",
        "gateway-service",
        transport=httpx.MockTransport(request),
        saved=cache,
    )
    try:
        assert (await profiles.get(path))["temperature"] == 0
        assert (await tokens.get(path)) is not None
        await cache.flush()
        clock[0] = 60
        await cache.flush()
    finally:
        await profiles.aclose()
        await tokens.aclose()
        await cache.aclose()
    payload = json.loads((tmp_path / "library-cache.json").read_text(encoding="utf-8"))
    assert path in payload["profiles"] and path in payload["specialTokens"]
    library.status = 503
    loaded = LibraryCache(tmp_path / "library-cache.json")
    profiles = resolver(library, saved=loaded)
    tokens = SpecialTokenSpellings(
        "http://agent.invalid",
        "gateway-service",
        transport=httpx.MockTransport(request),
        saved=loaded,
    )
    try:
        assert loaded.reachable is None
        calls = len(library.calls)
        assert (await profiles.get(path))["temperature"] == 0
        assert len(library.calls) > calls, "loaded defaults must be checked, never fresh"
        spellings = await tokens.get(path)
        assert spellings is not None and "<|end|>" in spellings.words
        assert loaded.reachable is False
        library.status = 200
        library.values[path] = {}
        profiles._cache.clear()
        assert await profiles.get(path) == {}
        assert loaded.reachable is True and loaded.since is None and loaded.cause is None
    finally:
        await profiles.aclose()
        await tokens.aclose()
        await loaded.aclose()


def test_info_reports_observed_outage_and_only_effective_saved_settings(app: FastAPI) -> None:
    library = Library()
    now = [0.0]
    config: dict[str, Any] = {"profileCacheSeconds": 30}
    profiles = resolver(library, config, clock=lambda: now[0])
    app.state.profile_defaults = profiles
    with TestClient(app) as client:
        before = client.get("/v1/info")
        assert before.status_code == 200
        assert before.json()["library"]["reachable"] is None
        client.portal.call(profiles.get, "/models/a.gguf")
        library.status = 503
        now[0] = 400
        client.portal.call(profiles.get, "/models/a.gguf")
        failed = client.get("/v1/info").json()["library"]
        assert failed["reachable"] is False
        assert failed["since"] and "503" in failed["cause"]
        assert failed["modelsOnSavedSettings"] == 1
        config["profileMaxStaleSeconds"] = 0
        assert client.get("/v1/info").json()["library"]["modelsOnSavedSettings"] == 0
        client.portal.call(profiles.aclose)


async def test_transport_timeout_keeps_values_past_400_seconds() -> None:
    library = Library()
    now = [0.0]
    profiles = resolver(library, clock=lambda: now[0])
    try:
        assert await profiles.get("/models/a.gguf")

        async def timeout(path: str) -> dict[str, Any]:
            raise httpx.ReadTimeout("the Library check timed out")

        profiles._read = timeout  # type: ignore[method-assign]
        now[0] = 400
        assert (await profiles.get("/models/a.gguf"))["temperature"] == 0
    finally:
        await profiles.aclose()


async def test_saved_answers_write_only_when_changed_and_at_most_once_a_minute(
    tmp_path: Path,
) -> None:
    from eugene_plexus_gateway.library_cache import LibraryCache

    now = [0.0]
    saved = LibraryCache(tmp_path / "library-cache.json", clock=lambda: now[0])
    writes: list[str] = []
    saved._write = writes.append  # type: ignore[method-assign]
    saved.update("profiles", "/a", {"temperature": 0})
    await saved.flush()
    assert len(writes) == 1
    saved.update("specialTokens", "/a", ["<end>"])
    now[0] = 59
    await saved.flush()
    assert len(writes) == 1
    now[0] = 60
    await saved.flush()
    assert len(writes) == 2 and "<end>" in writes[-1]
    now[0] = 120
    saved.update("specialTokens", "/a", ["<end>"])
    await saved.flush()
    assert len(writes) == 2
    await saved.aclose()


@pytest.mark.parametrize("raw", ["broken", '{"profiles":{"/a":null},"specialTokens":{"/a":7}}'])
async def test_a_broken_saved_cache_never_prevents_a_live_answer(tmp_path: Path, raw: str) -> None:
    from eugene_plexus_gateway.library_cache import LibraryCache

    (tmp_path / "library-cache.json").write_text(raw, encoding="utf-8")
    saved = LibraryCache(tmp_path / "library-cache.json")
    profiles = resolver(Library(), saved=saved)
    try:
        assert (await profiles.get("/models/a.gguf"))["temperature"] == 0
    finally:
        await profiles.aclose()
        await saved.aclose()
