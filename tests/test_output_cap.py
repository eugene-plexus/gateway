"""A length stop can name the setting that made it (2026-10-10).

A Strata answer stopped mid-thought at 4,096 tokens, and Workbench told Troy
to raise his chat's own setting. The cap was his install's
`defaultMaxTokens`, saved long before. The chat door now says what it sent
and where it came from (`x_eugene_plexus.output_cap`), for the backend that
answered: a fallback is capped by its own model's profile.
"""

import json
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from .conftest import FakeDriverClient
from .test_profile_defaults import Library, setup_routes


@pytest.fixture
def library() -> Library:
    return Library()


def _ask(client: TestClient, stream: bool, **extra: Any) -> dict[str, Any]:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "alias",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": stream,
            **extra,
        },
    )
    assert response.status_code == 200, response.text
    if not stream:
        return response.json()["x_eugene_plexus"]
    frames = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    return next(
        f["x_eugene_plexus"]
        for f in frames
        if f.get("choices") and f["choices"][0].get("finish_reason")
    )


def _one(app: FastAPI, library: Library) -> tuple[FakeDriverClient, Any]:
    driver = FakeDriverClient(
        name="driver", base_url="http://driver.invalid", model_id="alias", runtime="run"
    )
    return driver, setup_routes(app, library, driver)


@pytest.mark.parametrize("stream", [False, True])
def test_the_models_profile(app: FastAPI, library: Library, stream: bool) -> None:
    _driver, profiles = _one(app, library)
    with TestClient(app) as client:
        assert _ask(client, stream)["output_cap"] == {"tokens": 91, "source": "profile"}
        client.portal.call(profiles.aclose)


@pytest.mark.parametrize("stream", [False, True])
def test_the_callers_own(app: FastAPI, library: Library, stream: bool) -> None:
    _driver, profiles = _one(app, library)
    with TestClient(app) as client:
        said = _ask(client, stream, max_tokens=17)
        assert said["output_cap"] == {"tokens": 17, "source": "request"}
        client.portal.call(profiles.aclose)


@pytest.mark.parametrize("stream", [False, True])
def test_the_install_default(app: FastAPI, library: Library, stream: bool) -> None:
    # THE BUG: no profile cap, the install's 4,096 sent.
    driver, profiles = _one(app, library)
    library.values.clear()
    with TestClient(app) as client:
        assert client.patch("/v1/config", json={"defaultMaxTokens": 4096}).status_code == 200
        said = _ask(client, stream)
        assert driver.calls[-1].maxTokens == 4096
        assert said["output_cap"] == {"tokens": 4096, "source": "install"}
        client.portal.call(profiles.aclose)


@pytest.mark.parametrize("stream", [False, True])
def test_no_cap_sent_says_none(app: FastAPI, library: Library, stream: bool) -> None:
    driver, profiles = _one(app, library)
    library.values.clear()
    with TestClient(app) as client:
        said = _ask(client, stream)
        assert driver.calls[-1].maxTokens is None
        assert said.get("output_cap") is None
        client.portal.call(profiles.aclose)


@pytest.mark.parametrize("stream", [False, True])
def test_a_fallback_says_its_own_profiles_cap(app: FastAPI, library: Library, stream: bool) -> None:
    first = FakeDriverClient(
        name="d1", base_url="http://one.invalid", model_id="first", runtime="one"
    )
    second = FakeDriverClient(
        name="d2", base_url="http://two.invalid", model_id="second", runtime="two"
    )
    first.generate_error = httpx.ConnectError("offline")
    first.stream_error_after = 0
    first.stream_error = httpx.ConnectError("offline")
    profiles = setup_routes(app, library, first, second)
    app.state.routing._slots = lambda: [{"model": "alias", "targets": ["first", "second"]}]
    with TestClient(app) as client:
        said = _ask(client, stream)
        assert second.calls[-1].maxTokens == 73
        assert said["output_cap"] == {"tokens": 73, "source": "profile"}
        client.portal.call(profiles.aclose)
