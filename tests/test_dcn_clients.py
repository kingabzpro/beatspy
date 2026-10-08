"""Provider wire contracts, response parsing, and retry behaviour.

No network: `httpx.MockTransport` stands in for every provider, so these tests
pin the exact request bodies and the exact response shapes we accept. If a
provider changes its envelope, this fails loudly instead of silently producing
missing answers.
"""

from __future__ import annotations

import json

import httpx
import pytest

from beatspy.config import DECISION_MODELS, Settings, decision_api_key
from beatspy.dcn.clients import (
    CloudflareClefClient,
    DcnError,
    OpenAiDecisionsClient,
    TypeSafeClient,
    make_client,
)
from beatspy.dcn.questions import build_questions
from beatspy.schemas import parse_answers


def _questions():
    return build_questions("twin", ["AAPL"], "SPY", 20).questions


def _capture(payload: dict, status: int = 200, calls: list | None = None):
    """Build a MockTransport that records requests and returns one payload."""

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


def _client(transport, account_id: str | None = None):
    return httpx.AsyncClient(transport=transport)


# --------------------------------------------------------------------------- #
# Request shapes
# --------------------------------------------------------------------------- #


def test_openai_sends_a_responses_request_with_a_decision_block():
    import asyncio

    calls: list[httpx.Request] = []
    payload = {
        "decision": {"answers": {"noul_AAPL": {"type": "noul", "noul": 0.62}}},
        "usage": {"input_tokens": 120, "output_tokens": 0},
    }
    transport = _capture(payload, calls=calls)
    client = OpenAiDecisionsClient(
        model="gpt-6-luna",
        endpoint="https://api.openai.com/v1",
        api_key="sk-test",
        client=_client(transport),
    )
    result = asyncio.run(client.decide({"as_of": "2022-02-15"}, _questions()))

    request = calls[0]
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body["model"] == "gpt-6-luna"
    assert body["input"][0]["content"][0]["type"] == "input_text"
    assert "2022-02-15" in body["input"][0]["content"][0]["text"]
    assert body["decision"]["questions"]["noul_AAPL"]["type"] == "noul"
    assert body["decision"]["questions"]["choice_AAPL"]["criteria"] == {
        "outperform": "Total return above the benchmark over the horizon",
        "underperform": "Total return at or below the benchmark over the horizon",
    }
    assert result.input_tokens == 120
    assert result.answers["noul_AAPL"].noul == pytest.approx(0.62)


def test_openai_also_reads_answers_nested_in_output_content():
    import asyncio

    payload = {
        "output": [
            {"content": [{"type": "output_json", "json": {"answers": {"noul_AAPL": {"type": "noul", "noul": 0.4}}}}]}
        ]
    }
    client = OpenAiDecisionsClient(
        model="gpt-6-luna",
        endpoint="https://api.openai.com/v1",
        api_key="sk",
        client=_client(_capture(payload)),
    )
    result = asyncio.run(client.decide({"as_of": "x"}, _questions()))
    assert result.answers["noul_AAPL"].noul == pytest.approx(0.4)
    assert "choice_AAPL" in result.missing_ids


def test_typesafe_posts_state_and_questions_verbatim():
    import asyncio

    calls: list[httpx.Request] = []
    payload = {
        "model": "jev-1.13.0",
        "answers": {"noul_AAPL": {"type": "noul", "noul": 0.55}},
        "usage": {"input_tokens": 296, "output_tokens": 20},
    }
    client = TypeSafeClient(
        model="jev-latest",
        endpoint="https://api.typesafe.ai/v1/systemone",
        api_key="ts-test",
        client=_client(_capture(payload, calls=calls)),
    )
    state = {"as_of": "2022-02-15", "market": []}
    result = asyncio.run(client.decide(state, _questions()))

    body = json.loads(calls[0].content)
    assert str(calls[0].url) == "https://api.typesafe.ai/v1/systemone"
    assert calls[0].headers["authorization"] == "Bearer ts-test"
    assert body["state"] == state  # the state is passed through unmodified
    assert body["model"] == "jev-latest"
    assert body["questions"]["noul_AAPL"]["type"] == "noul"
    assert result.input_tokens == 296
    assert result.output_tokens == 20


def test_cloudflare_puts_the_model_and_account_in_the_url():
    import asyncio

    calls: list[httpx.Request] = []
    payload = {
        "result": {
            "answers": {
                "noul_AAPL": {"type": "noul", "noul": 0.71},
                "choice_AAPL": {
                    "type": "choice",
                    "choice": "outperform",
                    "probabilities": {"outperform": 0.71, "underperform": 0.29},
                },
            },
            "usage": {"input_tokens": 400},
        },
        "success": True,
    }
    client = CloudflareClefClient(
        model="clef-flash",
        endpoint="https://api.cloudflare.com/client/v4/accounts",
        api_key="cf-token",
        account_id="acct123",
        client=_client(_capture(payload, calls=calls)),
    )
    result = asyncio.run(client.decide({"as_of": "2022-02-15"}, _questions()))

    assert str(calls[0].url).endswith("/acct123/ai/run/@cf/cloudflare/clef-flash")
    body = json.loads(calls[0].content)
    assert body["model"] == "clef-flash"
    # Structured state, not a stringified blob: Workers AI truncates long text.
    assert isinstance(body["state"], dict)
    assert result.input_tokens == 400
    assert result.missing_ids == []
    assert result.answers["choice_AAPL"].probabilities["outperform"] == pytest.approx(0.71)


# --------------------------------------------------------------------------- #
# Retry behaviour
# --------------------------------------------------------------------------- #


def test_transient_statuses_are_retried_then_succeed(monkeypatch):
    import asyncio

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, json={"error": "slow down"})
        if attempts["n"] == 2:
            return httpx.Response(529, json={"error": "overloaded"})
        return httpx.Response(200, json={"answers": {"noul_AAPL": {"type": "noul", "noul": 0.5}}})

    async def no_sleep(_seconds):
        return None

    client = TypeSafeClient(
        model="jev-latest",
        endpoint="https://api.typesafe.ai/v1/systemone",
        api_key="k",
        max_retries=3,
        client=_client(httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(client, "_sleep", no_sleep)
    result = asyncio.run(client.decide({"as_of": "x"}, {"noul_AAPL": _questions()["noul_AAPL"]}))
    assert attempts["n"] == 3
    assert result.attempts == 3


def test_a_real_client_error_is_not_retried():
    import asyncio

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(422, json={"error": "bad question schema"})

    client = TypeSafeClient(
        model="jev-latest",
        endpoint="https://api.typesafe.ai/v1/systemone",
        api_key="k",
        max_retries=3,
        client=_client(httpx.MockTransport(handler)),
    )
    with pytest.raises(DcnError) as excinfo:
        asyncio.run(client.decide({"as_of": "x"}, _questions()))
    assert attempts["n"] == 1  # retrying a 422 would only burn quota
    assert "422" in str(excinfo.value)


def test_retries_are_exhausted_and_surface_the_last_error():
    import asyncio

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    client = TypeSafeClient(
        model="jev-latest",
        endpoint="https://api.typesafe.ai/v1/systemone",
        api_key="k",
        max_retries=2,
        client=_client(httpx.MockTransport(handler)),
    )
    client._sleep = _no_sleep
    with pytest.raises(DcnError) as excinfo:
        asyncio.run(client.decide({"as_of": "x"}, _questions()))
    assert "503" in str(excinfo.value)


def test_a_non_json_body_is_an_error_not_a_missing_answer():
    import asyncio

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    client = TypeSafeClient(
        model="jev-latest",
        endpoint="https://api.typesafe.ai/v1/systemone",
        api_key="k",
        client=_client(httpx.MockTransport(handler)),
    )
    with pytest.raises(DcnError):
        asyncio.run(client.decide({"as_of": "x"}, _questions()))


async def _no_sleep(_seconds):
    return None


def test_network_errors_are_retried(monkeypatch):
    import asyncio

    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(200, json={"answers": {}})

    client = TypeSafeClient(
        model="jev-latest",
        endpoint="https://api.typesafe.ai/v1/systemone",
        api_key="k",
        max_retries=3,
        client=_client(httpx.MockTransport(handler)),
    )
    client._sleep = _no_sleep
    result = asyncio.run(client.decide({"as_of": "x"}, _questions()))
    assert attempts["n"] == 3
    assert set(result.missing_ids) == {"noul_AAPL", "choice_AAPL"}


# --------------------------------------------------------------------------- #
# Answer parsing
# --------------------------------------------------------------------------- #


def test_answers_that_fail_validation_become_missing_not_fatal():
    payload = {
        "answers": {
            "good": {"type": "noul", "noul": 0.4},
            "out_of_range": {"type": "noul", "noul": 1.4},
            "bad_type": {"type": "wat", "value": 1},
            "not_an_object": "nope",
            "distribution_wrong": {
                "type": "choice",
                "choice": "outperform",
                "probabilities": {"outperform": 0.4, "underperform": 0.4},
            },
        }
    }
    answers, missing = parse_answers(payload, ["good", "out_of_range", "bad_type", "not_an_object", "distribution_wrong"])
    assert set(answers) == {"good"}
    assert set(missing) == {"out_of_range", "bad_type", "not_an_object", "distribution_wrong"}


def test_a_choice_distribution_tolerates_rounding_but_not_a_real_error():
    ok, _ = parse_answers(
        {"answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0, "b": 0.005}}}},
        ["q"],
    )
    assert "q" in ok
    bad, missing = parse_answers(
        {"answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 0.5, "b": 0.2}}}},
        ["q"],
    )
    assert not bad
    assert missing == ["q"]


def test_a_missing_answers_key_makes_everything_missing():
    answers, missing = parse_answers({"model": "x"}, ["a", "b"])
    assert answers == {}
    assert missing == ["a", "b"]


def test_score_answers_pass_through_unchanged():
    answers, missing = parse_answers(
        {"answers": {"q": {"type": "score", "score": 1.05, "legend": {"0": "Calm", "1": "Frustrated"}}}},
        ["q"],
    )
    assert missing == []
    assert answers["q"].score == pytest.approx(1.05)
    assert answers["q"].legend["1"] == "Frustrated"


# --------------------------------------------------------------------------- #
# Client construction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(DECISION_MODELS))
def test_make_client_builds_every_configured_model(name):
    settings = Settings()
    settings.decision.decision_model = name
    client = make_client(settings)
    spec = DECISION_MODELS[name]
    assert client.provider == spec["provider"]
    assert client.model == spec["model"]
    assert client.endpoint == spec["base_url"]


def test_unknown_model_is_rejected_at_construction():
    from pydantic import ValidationError

    from beatspy.config import DecisionConfig

    # Mutable pydantic models do not re-validate on assignment, so the guard that
    # matters is the one at construction time.
    with pytest.raises(ValidationError):
        DecisionConfig(decision_model="not-a-model")


def test_every_registered_model_has_a_key_environment_variable():
    for name, spec in DECISION_MODELS.items():
        assert spec["api_key_env"], f"{name} has no api_key_env"
        assert spec["provider"] in ("openai_decisions", "typesafe", "cloudflare")
        assert spec["cost_per_m_input"] is not None
        assert spec["cost_per_m_output"] == 0.0  # decision models bill input only


def test_api_key_resolution_prefers_the_beatspy_prefixed_variable(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("BEATSPY_TYPESAFE_API_KEY", "prefixed")
    assert decision_api_key("jev-latest") == "prefixed"
    monkeypatch.setenv("TYPESAFE_API_KEY", "plain")
    assert decision_api_key("jev-latest") == "prefixed"  # the prefixed name wins first
    monkeypatch.delenv("BEATSPY_TYPESAFE_API_KEY")
    assert decision_api_key("jev-latest") == "plain"


def test_api_key_absent_is_none_not_an_error(monkeypatch):
    for env in ("TYPESAFE_API_KEY", "BEATSPY_TYPESAFE_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    assert decision_api_key("jev-latest") is None