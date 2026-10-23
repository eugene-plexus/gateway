"""Compaction is opaque, authenticated, stateless, and uses normal admission."""

from pathlib import Path

import pytest

from eugene_plexus_gateway._generated.driver_models import Usage
from eugene_plexus_gateway.settings import Settings

from .test_responses import MODEL, sent, serve
from .test_responses import driver as response_driver


def driver():
    fake = response_driver()
    fake.usage = Usage(promptTokens=20, completionTokens=5, totalTokens=25)
    return fake


def compact(client):
    result = client.post(
        "/v1/responses/compact",
        json={
            "model": MODEL,
            "input": "Remember the file hello.txt",
            "instructions": "Keep filenames",
        },
    )
    assert result.status_code == 200, result.text
    return result.json()


def test_compact_returns_an_item_that_expands_next_time(settings: Settings):
    fake = driver()
    fake.responses = ["Summary: edited hello.txt; tests pending."]
    fake.prompt_tokens = 12
    with serve(settings, fake) as client:
        result = compact(client)
        assert result["object"] == "response.compaction"
        item = result["output"][-1]
        assert item["type"] == "compaction"
        assert "hello.txt" not in item["encrypted_content"]
        assert fake.calls[-1].maxTokens == 4000
        assert "Keep filenames" in str(sent(fake))
        followup = client.post(
            "/v1/responses",
            json={"model": MODEL, "input": [item, {"role": "user", "content": "continue"}]},
        )
        assert followup.status_code == 200, followup.text
        # The public translation is developer; the existing driver contract
        # normalizes every developer message to system for local chat templates.
        from eugene_plexus_gateway import compaction, responses

        translated = responses.translate_request(
            {"model": MODEL, "input": [item]},
            sealer=compaction.Sealer(settings.config_file.parent / "compaction.key"),
        )
        assert translated.request.messages[0].role == "developer"
        assert sent(fake)[0]["role"] == "system"
        assert sent(fake)[0]["content"] == "Summary: edited hello.txt; tests pending."
        counted = client.post("/v1/responses/input_tokens", json={"model": MODEL, "input": [item]})
        assert counted.status_code == 200, counted.text
    with serve(settings, fake) as restarted:
        result = restarted.post("/v1/responses", json={"model": MODEL, "input": [item]})
        assert result.status_code == 200, result.text
    key = settings.config_file.parent / "compaction.key"
    assert key.is_file()
    assert "hello.txt" not in key.read_text()


@pytest.mark.parametrize("token", ["foreign-openai-content", "", None])
def test_a_tampered_item_is_refused(settings: Settings, token):
    fake = driver()
    with serve(settings, fake) as client:
        result = compact(client)
        item = result["output"][-1]
        if token == "":
            encrypted = item["encrypted_content"]
            token = encrypted[:25] + ("A" if encrypted[25] != "A" else "B") + encrypted[26:]
        item["encrypted_content"] = token
        before = len(fake.calls)
        bad = client.post("/v1/responses", json={"model": MODEL, "input": [item]})
    assert bad.status_code == 400
    assert (
        "This compaction was made elsewhere; send the conversation instead."
        in bad.json()["error"]["message"]
    )
    assert len(fake.calls) == before


def test_key_failure_names_observed_cause(settings: Settings):
    key = settings.config_file.parent / "compaction.key"
    key.mkdir()
    with serve(settings, driver()) as client:
        result = client.post("/v1/responses/compact", json={"model": MODEL, "input": "summarize"})
    assert result.status_code == 503
    assert "compaction.key" in result.json()["error"]["message"]


@pytest.mark.parametrize("extra", [{"stream": True}, {"previous_response_id": "resp_foreign"}])
def test_compact_refuses_unknown_fields_and_unavailable_chains(settings: Settings, extra):
    fake = driver()
    with serve(settings, fake) as client:
        result = client.post("/v1/responses/compact", json={"model": MODEL, "input": "go", **extra})
    assert result.status_code == 400
    assert next(iter(extra)) in result.json()["error"]["message"]
    assert not fake.calls


def test_compaction_from_another_gateway_is_refused(settings: Settings):
    with serve(settings, driver()) as client:
        item = compact(client)["output"][-1]
    other = settings.model_copy(
        update={"config_file": settings.config_file.parent / "other" / "gateway.yaml"}
    )
    Path(other.config_file).parent.mkdir()
    with serve(other, driver()) as client:
        result = client.post("/v1/responses", json={"model": MODEL, "input": [item]})
    assert result.status_code == 400


@pytest.mark.parametrize("reason", ["length", "content_filter"])
def test_an_unfinished_summary_is_not_sealed(settings: Settings, reason):
    from eugene_plexus_gateway._generated.driver_models import FinishReason

    fake = driver()
    fake.finish_reason = FinishReason(reason)
    with serve(settings, fake) as client:
        result = client.post("/v1/responses/compact", json={"model": MODEL, "input": "go"})
    assert result.status_code == 502
    assert reason in result.json()["error"]["message"]


def test_compact_does_not_invent_missing_token_usage(settings: Settings):
    fake = driver()
    fake.usage = None
    with serve(settings, fake) as client:
        result = client.post("/v1/responses/compact", json={"model": MODEL, "input": "go"})
    assert result.status_code == 502
    assert "did not report token usage" in result.json()["error"]["message"]
