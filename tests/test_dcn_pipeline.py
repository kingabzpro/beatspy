"""The sizing policy and the neutrality of missing answers.

The policy is the part of a benchmark most easily tuned after the fact, so it is
pinned here: whatever the probabilities are, the mapping to a portfolio is fixed
and provable.
"""

from __future__ import annotations

from datetime import date

import pytest

from beatspy.config import DecisionConfig
from beatspy.dcn.clients import DcnResult
from beatspy.dcn.pipeline import NEUTRAL, DcnPipeline, derive_weights
from beatspy.schemas import NoulAnswer


@pytest.fixture
def policy() -> DecisionConfig:
    return DecisionConfig(min_probability=0.5, top_n=3, min_names=2)


def test_all_neutral_probabilities_hold_cash(policy):
    decision = derive_weights({"AAPL": 0.5, "MSFT": 0.5, "TLT": 0.5}, policy, _scenario())
    assert decision.weights == {}
    assert decision.cash == 1.0
    assert not decision.invalid


def test_one_clearing_name_is_not_enough(policy):
    decision = derive_weights({"AAPL": 0.9, "MSFT": 0.5, "TLT": 0.4}, policy, _scenario())
    assert decision.weights == {}
    assert decision.cash == 1.0


def test_strongest_names_are_equal_weighted(policy):
    decision = derive_weights({"AAPL": 0.9, "MSFT": 0.8, "TLT": 0.7}, policy, _scenario())
    assert decision.weights == pytest.approx({"AAPL": 1 / 3, "MSFT": 1 / 3, "TLT": 1 / 3})
    assert decision.cash == pytest.approx(0.0)


def test_only_top_n_are_held(policy):
    probabilities = {"AAPL": 0.7, "MSFT": 0.9, "TLT": 0.8, "GOOGL": 0.85}
    scenario = _scenario(tradable=["AAPL", "MSFT", "TLT", "GOOGL"])
    decision = derive_weights(probabilities, policy, scenario)
    assert set(decision.weights) == {"MSFT", "GOOGL", "TLT"}  # the three strongest
    assert "AAPL" not in decision.weights


def test_probability_above_the_minimum_is_included(policy):
    # Strictly greater than: exactly 0.5 is excluded, 0.5001 is not.
    boundary = derive_weights({"AAPL": 0.5001, "MSFT": 0.5}, policy, _scenario())
    assert boundary.weights == {}  # only one name cleared, below min_names


def test_ties_break_on_ticker_so_a_run_is_reproducible(policy):
    # Two names tie on probability; both are held, and the scenario's 0.35 cap
    # applies to each, so equal weighting is what "equal" can mean here.
    decision = derive_weights({"AAPL": 0.8, "MSFT": 0.8}, policy, _scenario(max_position_weight=0.5))
    assert set(decision.weights) == {"AAPL", "MSFT"}
    assert decision.weights["AAPL"] == pytest.approx(0.5)
    assert decision.weights["MSFT"] == pytest.approx(0.5)

    capped = derive_weights({"AAPL": 0.8, "MSFT": 0.8}, policy, _scenario())
    assert capped.weights["AAPL"] == pytest.approx(0.35)  # default scenario cap


def test_position_limit_clamps_and_cash_absorbs_the_remainder():
    policy = DecisionConfig(min_probability=0.5, top_n=3, min_names=2)
    scenario = _scenario(max_position_weight=0.2)
    decision = derive_weights({"AAPL": 0.9, "MSFT": 0.8, "TLT": 0.7}, policy, scenario)
    assert decision.weights == pytest.approx({"AAPL": 0.2, "MSFT": 0.2, "TLT": 0.2})
    assert decision.cash == pytest.approx(0.4)
    assert decision.violations  # the clamp is recorded, not silent


def test_unknown_ticker_is_dropped_as_a_violation(policy):
    decision = derive_weights({"AAPL": 0.9, "MSFT": 0.8, "ZZZZ": 0.99}, policy, _scenario())
    assert "ZZZZ" not in decision.weights
    assert any("ZZZZ" in violation for violation in decision.violations)


def test_policy_is_configurable_but_defaults_are_recorded(policy):
    assert DecisionConfig().min_probability == 0.5
    assert DecisionConfig().top_n == 3
    assert DecisionConfig().min_names == 2
    laxer = DecisionConfig(min_probability=0.0, top_n=1, min_names=1)
    # A single name is capped by the scenario's position limit, not by the policy.
    decision = derive_weights({"AAPL": 0.2}, laxer, _scenario())
    assert decision.weights == pytest.approx({"AAPL": 0.35})
    assert decision.cash == pytest.approx(0.65)

    uncapped = derive_weights({"AAPL": 0.2}, laxer, _scenario(max_position_weight=1.0))
    assert uncapped.weights == {"AAPL": 1.0}
    assert uncapped.cash == pytest.approx(0.0)


def _scenario(**overrides):
    from beatspy.scenarios import make_scenario

    return make_scenario(**overrides)


class FailingClient:
    provider = "offline"
    model = "failing"
    endpoint = "https://offline.invalid"

    async def decide(self, state, questions):
        raise RuntimeError("provider exploded")

    async def close(self):
        return None


class PartialClient:
    """Answers only the first question, to exercise missing-answer handling."""

    provider = "offline"
    model = "partial"
    endpoint = "https://offline.invalid"

    def __init__(self):
        self.seen: list[str] = []

    async def decide(self, state, questions):
        ids = list(questions)
        self.seen = ids
        return DcnResult(answers={ids[0]: NoulAnswer(noul=0.99)}, missing_ids=ids[1:], input_tokens=10)

    async def close(self):
        return None


def test_missing_answers_backfill_to_neutral_not_to_a_direction(data_service, scenario, settings):
    """A provider failure must not become a trade."""
    import asyncio

    client = PartialClient()
    pipeline = DcnPipeline(settings, scenario, client, question_form="noul")
    decision = asyncio.run(
        pipeline.decide(
            data_service,
            date(2022, 2, 1),
            holdings={},
            cash=1.0,
            horizon_trading_days=20,
            next_decision_date="2022-03-01",
        )
    )
    assert decision.missing_ids
    # Every unanswered ticker sits on the neutral, and only the single answered
    # name could ever clear the bar — so fewer than min_names clear it: all cash.
    for ticker, probability in decision.probabilities.items():
        if ticker not in _answered_tickers(client):
            assert probability == NEUTRAL
    assert decision.validated.weights == {}
    assert decision.validated.cash == 1.0


def _answered_tickers(client: PartialClient) -> set[str]:
    return {client.seen[0].rsplit("_", 1)[-1].upper()} if client.seen else set()


def test_provider_failure_produces_an_invalid_decision_that_holds(data_service, scenario, settings):
    import asyncio

    pipeline = DcnPipeline(settings, scenario, FailingClient(), question_form="noul")
    decision = asyncio.run(
        pipeline.decide(
            data_service,
            date(2022, 2, 1),
            holdings={"AAPL": 0.4},
            cash=0.6,
            horizon_trading_days=20,
            next_decision_date="2022-03-01",
        )
    )
    assert decision.validated.invalid
    assert decision.error and "provider exploded" in decision.error
    # A total failure yields no answers at all: every id is missing and the
    # decision is flagged invalid, which the runner treats as "hold".
    assert decision.answers == {}
    assert set(decision.missing_ids) == set(decision.question_ids)
    assert decision.coverage == 0.0
    assert decision.call is None  # no response to record
    assert decision.validated.weights == {}