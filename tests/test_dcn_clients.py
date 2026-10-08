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


def test_openai_posts_to_the_dedicated_decisions_endpoint():
    """OpenAI does not share the System One envelope: /v1/decisions, array questions."""
    import asyncio

    calls: list[httpx.Request] = []
    payload = {
        "answers": [{"type": "predicate", "name": "noul_AAPL", "probability": 0.62}],
        "usage": {"input_tokens": 120, "output_tokens": 0},
    }
    transport = _capture(payload, calls=calls)
    client = OpenAiDecisionsClient(
        model="gpt-6-luna",
        endpoint="https://api.openai.com/v1",
        api_key="sk-test",
        client=_client(transport),
    )
    state = {"as_of": "2022-02-15", "market": [{"ticker": "AAPL"}]}
    result = asyncio.run(client.decide(state, _questions()))

    request = calls[0]
    assert str(request.url) == "https://api.openai.com/v1/decisions"
    assert request.headers["authorization"] == "Bearer sk-test"
    body = json.loads(request.content)
    assert body["model"] == "gpt-6-luna"
    # A single text input, not a Responses-style message array.
    assert isinstance(body["input"], str)
    assert "2022-02-15" in body["input"]
    # Questions are an array, each named, and the yes/no type is `predicate`.
    assert isinstance(body["questions"], list)
    by_name = {q["name"]: q for q in body["questions"]}
    assert by_name["noul_AAPL"]["type"] == "predicate"
    assert by_name["choice_AAPL"]["type"] == "choice"
    assert by_name["choice_AAPL"]["choices"] == [
        {"value": "outperform", "description": "Total return above the benchmark over the horizon"},
        {"value": "underperform", "description": "Total return at or below the benchmark over the horizon"},
    ]
    assert result.input_tokens == 120
    assert result.answers["noul_AAPL"].noul == pytest.approx(0.62)
    assert result.missing_ids == ["choice_AAPL"]


def test_openai_reads_list_probabilities_and_a_score_levels():
    import asyncio

    payload = {
        "answers": [
            {
                "type": "choice",
                "name": "choice_AAPL",
                "choice": "outperform",
                "probabilities": [
                    {"value": "outperform", "probability": 0.7},
                    {"value": "underperform", "probability": 0.3},
                ],
                "confidence": 0.8,
            },
            {
                "type": "score",
                "name": "rank_AAPL",
                "score": 1.1,
                "probabilities": [
                    {"value": 0, "label": "Cosmetic", "probability": 0.1},
                    {"value": 1, "label": "Workaround", "probability": 0.7},
                    {"value": 2, "label": "Blocked", "probability": 0.2},
                ],
                "confidence": 0.55,
            },
        ]
    }
    client = OpenAiDecisionsClient(
        model="gpt-6-luna",
        endpoint="https://api.openai.com/v1",
        api_key="sk",
        client=_client(_capture(payload)),
    )
    result = asyncio.run(client.decide({"as_of": "x"}, {"choice_AAPL": _questions()["choice_AAPL"],
                                                          "rank_AAPL": _questions()["noul_AAPL"]}))
    assert result.answers["choice_AAPL"].probabilities == {"outperform": 0.7, "underperform": 0.3}
    assert result.answers["rank_AAPL"].score == pytest.approx(1.1)
    assert result.answers["rank_AAPL"].legend["2"] == "Blocked"


def test_openai_normalises_a_distribution_that_does_not_sum_to_one():
    import asyncio

    payload = {
        "answers": [
            {
                "type": "choice",
                "name": "choice_AAPL",
                "probabilities": [
                    {"value": "outperform", "probability": 0.6},
                    {"value": "underperform", "probability": 0.6},
                ],
            }
        ]
    }
    client = OpenAiDecisionsClient(
        model="gpt-6-luna", endpoint="https://api.openai.com/v1", api_key="sk",
        client=_client(_capture(payload)),
    )
    result = asyncio.run(client.decide({"as_of": "x"}, {"choice_AAPL": _questions()["choice_AAPL"]}))
    assert result.answers["choice_AAPL"].probabilities == {"outperform": 0.5, "underperform": 0.5}


def test_openai_refusal_is_missing_coverage_not_a_crash():
    """A refusal has no value, so it must count as an unanswered question."""
    import asyncio

    payload = {
        "answers": [
            {"type": "refusal", "name": "noul_AAPL"},
            {"type": "predicate", "name": "choice_AAPL", "probability": 0.4},
        ]
    }
    client = OpenAiDecisionsClient(
        model="gpt-6-luna", endpoint="https://api.openai.com/v1", api_key="sk",
        client=_client(_capture(payload)),
    )
    result = asyncio.run(client.decide({"as_of": "x"}, _questions()))
    assert "noul_AAPL" not in result.answers
    assert result.missing_ids == ["noul_AAPL"]
    assert result.raw["_refused"] == ["noul_AAPL"]


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
def test_make_client_builds_every_configured_model(name, monkeypatch):
    spec = DECISION_MODELS[name]
    # Supply whatever this provider needs so construction is what is under test.
    monkeypatch.setenv(spec["api_key_env"], "test-key")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    settings = Settings()
    settings.decision.decision_model = name
    client = make_client(settings)
    assert client.provider == spec["provider"]
    assert client.model == spec["model"]
    assert client.endpoint == spec["base_url"]


def test_a_missing_credential_fails_fast_with_the_variable_name(monkeypatch):
    """An absent key must name itself, not surface as a transport header error."""
    for env in ("TYPESAFE_API_KEY", "BEATSPY_TYPESAFE_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    settings = Settings()
    settings.decision.decision_model = "jev-latest"
    with pytest.raises(DcnError) as excinfo:
        make_client(settings)
    message = str(excinfo.value)
    assert "TYPESAFE_API_KEY" in message
    assert ".env" in message


def test_cloudflare_needs_both_the_token_and_the_account_id(monkeypatch):
    from beatspy.dcn.clients import missing_credentials

    settings = Settings()
    settings.decision.decision_model = "clef-flash"
    monkeypatch.setenv("CLOUDFLARE_AUTH_TOKEN", "tok")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("BEATSPY_CLOUDFLARE_ACCOUNT_ID", raising=False)
    assert missing_credentials(settings) == ["CLOUDFLARE_ACCOUNT_ID"]

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    assert missing_credentials(settings) == []

    monkeypatch.delenv("CLOUDFLARE_AUTH_TOKEN")
    assert missing_credentials(settings) == ["CLOUDFLARE_AUTH_TOKEN"]


def test_missing_credentials_is_empty_for_a_non_cloudflare_provider(monkeypatch):
    from beatspy.dcn.clients import missing_credentials

    settings = Settings()
    settings.decision.decision_model = "gpt-6-luna"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("BEATSPY_CLOUDFLARE_ACCOUNT_ID", raising=False)
    assert missing_credentials(settings) == []


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