"""The sizing policy and the neutrality of missing answers.

Two policies are tested here.

- **graded** (the default): fully invested across the whole universe, each weight
  proportional to the probability above the prior. A weak forecast becomes a mild
  tilt instead of a binary switch. This replaced a top-N cut that could only emit
  0% or 1/top_n%, which rewrote the whole book every week.
- **top_n** (legacy): hold only `top_n` names above `min_probability`, cash if
  fewer than `min_names` qualify. Kept because earlier runs recorded it.

The policy is the part of a benchmark most easily tuned after the fact, so the
mapping from probabilities to portfolio is pinned here.
"""

from __future__ import annotations

from datetime import date

import pytest

from beatspy.config import DecisionConfig
from beatspy.dcn.clients import DcnResult
from beatspy.dcn.pipeline import NEUTRAL, DcnPipeline, derive_weights
from beatspy.schemas import NoulAnswer


def _scenario(**overrides):
    """A scenario wide enough for a graded book.

    The default test scenario has three tickers, so a uniform spread would give
    each 33% and every test would be measuring the position cap instead of the
    policy.
    """
    from beatspy.scenarios import make_scenario

    payload = {
        "name": "policy-test",
        "title": "Policy test",
        "start": "2022-01-03",
        "end": "2022-03-31",
        "tradable": ["AAPL", "MSFT", "TLT", "GOOGL", "AMZN", "JPM"],
        # Uncapped here: the blend maths is what these tests are about. The cap has
        # its own test, and a 0.35 default would clamp every concentrated case and
        # hide the behaviour being asserted.
        "max_position_weight": 1.0,
    }
    payload.update(overrides)
    return make_scenario(**payload)


def _six(probabilities: dict[str, float]) -> dict[str, float]:
    """Pad a probability map to the full six-asset scenario at the neutral."""
    base = {ticker: 0.5 for ticker in ("AAPL", "MSFT", "TLT", "GOOGL", "AMZN", "JPM")}
    base.update(probabilities)
    return base


@pytest.fixture
def graded() -> DecisionConfig:
    return DecisionConfig(selection="graded", min_probability=0.5)


@pytest.fixture
def legacy() -> DecisionConfig:
    return DecisionConfig(selection="top_n", min_probability=0.5, top_n=3, min_names=2)


# --------------------------------------------------------------------------- #
# Graded (default)
# --------------------------------------------------------------------------- #


def test_graded_weights_blend_an_even_base_with_conviction(graded):
    """Half the book is even across the universe, half follows the tilt.

    With six assets: even base 0.5/6 each, plus 0.5 split 0.3:0.1 between the two
    that have a view. Every asset keeps some weight, so the book is never a
    two-name bet.
    """
    decision = derive_weights(_six({"AAPL": 0.8, "MSFT": 0.6}), graded, _scenario())
    assert decision.weights["AAPL"] == pytest.approx(0.5 / 6 + 0.5 * 0.75)
    assert decision.weights["MSFT"] == pytest.approx(0.5 / 6 + 0.5 * 0.25)
    # The four assets with no view still carry the even base, not zero.
    for ticker in ("TLT", "GOOGL", "AMZN", "JPM"):
        assert decision.weights[ticker] == pytest.approx(0.5 / 6)
    assert sum(decision.weights.values()) == pytest.approx(1.0)


def test_graded_is_always_fully_invested_when_there_is_a_view(graded):
    """The old policy could sit in cash at will; a graded book always participates."""
    for probabilities in (
        {"AAPL": 0.51},
        {"AAPL": 0.99, "MSFT": 0.51},
        {"AAPL": 0.52, "MSFT": 0.51, "TLT": 0.5},
    ):
        decision = derive_weights(_six(probabilities), graded, _scenario())
        assert sum(decision.weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_graded_spreads_the_universe_evenly_when_there_is_no_view(graded):
    """All-neutral answers mean "no opinion", which is an even spread, not cash.

    Sitting in cash would make the run a non-participant and quietly flatter
    itself in a falling market.
    """
    decision = derive_weights(_six({}), graded, _scenario())
    assert decision.weights == pytest.approx({ticker: 1 / 6 for ticker in _six({})})
    assert decision.cash == pytest.approx(0.0)


def test_graded_keeps_every_asset_in_play(graded):
    """A single decision per asset means every asset holds weight, always."""
    probabilities = _six({"AAPL": 0.9, "MSFT": 0.8, "TLT": 0.7, "GOOGL": 0.6})
    decision = derive_weights(probabilities, graded, _scenario())
    assert set(decision.weights) == set(probabilities)
    assert sum(decision.weights.values()) == pytest.approx(1.0)
    # The no-view assets are still held, which is the point of the even base.
    assert decision.weights["AMZN"] > 0


def test_graded_never_oscillates_between_zero_and_a_full_slot(graded):
    """The defect that motivated graded: weights moved 0% -> 33% -> 0% every week.

    Conviction moves weight smoothly. Note the shape: with two assets relative
    conviction sets the split, so the same 0.6 answer yields one weight when it is
    the strongest view and another when something else is stronger. What matters is
    that neither 0% nor 100% is reachable, and that no asset ever leaves the book.
    """
    weak = derive_weights(_six({"AAPL": 0.52, "MSFT": 0.51}), graded, _scenario())
    strong = derive_weights(_six({"AAPL": 0.9, "MSFT": 0.51}), graded, _scenario())
    assert 0.0 < weak.weights["AAPL"] < strong.weights["AAPL"] < 1.0
    assert 0.0 < weak.weights["AAPL"] < 0.45  # a mild view stays a mild weight
    # Every asset keeps weight in both books; nothing oscillates to zero.
    for ticker in ("AAPL", "MSFT", "TLT", "GOOGL", "AMZN", "JPM"):
        assert weak.weights[ticker] > 0 and strong.weights[ticker] > 0
    # And the book is always fully invested, never partly in cash.
    assert sum(weak.weights.values()) == pytest.approx(1.0)
    assert sum(strong.weights.values()) == pytest.approx(1.0)


def test_graded_position_limit_is_enforced_and_cash_absorbs_the_remainder(graded):
    scenario = _scenario(max_position_weight=0.4)
    decision = derive_weights(_six({"AAPL": 0.9, "MSFT": 0.51}), graded, scenario)
    assert decision.weights["AAPL"] == pytest.approx(0.4)
    assert decision.cash == pytest.approx(1.0 - sum(decision.weights.values()))
    assert decision.violations  # the clamp is recorded, not silent


def test_graded_scores_only_the_scenario_universe(graded):
    """The universe is the allowlist: an out-of-universe ticker cannot be scored."""
    decision = derive_weights(_six({"AAPL": 0.9, "ZZZZ": 0.99}), graded, _scenario())
    assert "ZZZZ" not in decision.weights
    # AAPL takes the conviction share plus its even base; the rest stay even.
    assert decision.weights["AAPL"] == pytest.approx(0.5 / 6 + 0.5)
    assert decision.weights["MSFT"] == pytest.approx(0.5 / 6)


def test_graded_never_buys_the_benchmark(graded):
    """SPY is comparison-only. It is not in `tradable`, so it can never be held."""
    scenario = _scenario(tradable=["AAPL", "MSFT"])
    decision = derive_weights({"AAPL": 0.9, "MSFT": 0.6}, graded, scenario, benchmark="SPY")
    assert "SPY" not in decision.weights
    assert set(decision.weights) <= {"AAPL", "MSFT"}


# --------------------------------------------------------------------------- #
# Legacy top-N (kept for reproducibility of earlier runs)
# --------------------------------------------------------------------------- #


def test_legacy_holds_cash_when_nothing_clears_the_bar(legacy):
    decision = derive_weights({"AAPL": 0.5, "MSFT": 0.5, "TLT": 0.5}, legacy, _scenario())
    assert decision.weights == {}
    assert decision.cash == 1.0


def test_legacy_needs_min_names(legacy):
    decision = derive_weights({"AAPL": 0.9, "MSFT": 0.5}, legacy, _scenario())
    assert decision.weights == {}


def test_legacy_equal_weights_the_strongest(legacy):
    decision = derive_weights({"AAPL": 0.9, "MSFT": 0.8, "TLT": 0.7}, legacy, _scenario())
    assert decision.weights == pytest.approx({"AAPL": 1 / 3, "MSFT": 1 / 3, "TLT": 1 / 3})


def test_legacy_holds_only_top_n(legacy):
    probabilities = {"AAPL": 0.7, "MSFT": 0.9, "TLT": 0.8, "GOOGL": 0.85}
    scenario = _scenario(tradable=["AAPL", "MSFT", "TLT", "GOOGL"])
    decision = derive_weights(probabilities, legacy, scenario)
    assert set(decision.weights) == {"MSFT", "GOOGL", "TLT"}


def test_defaults_are_graded_and_recorded():
    assert DecisionConfig().selection == "graded"
    assert DecisionConfig().min_probability == 0.5


# --------------------------------------------------------------------------- #
# Missing answers
# --------------------------------------------------------------------------- #


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


def test_missing_answers_backfill_to_neutral_and_only_the_answered_name_gets_weight(
    data_service, scenario, settings
):
    """A provider failure must not become a directional bet on the unanswered names."""
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
    answered = {client.seen[0].rsplit("_", 1)[-1].upper()}
    for ticker, probability in decision.probabilities.items():
        if ticker not in answered:
            assert probability == NEUTRAL

    if settings.decision.selection == "graded":
        # The unanswered names keep the even base; the answered name also carries
        # the conviction share and hits the position limit.
        assert answered <= set(decision.validated.weights)
        assert decision.validated.weights[list(answered)[0]] > 1.0 / len(scenario.tradable)
    else:
        assert decision.validated.weights == {}


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
    assert decision.answers == {}
    assert set(decision.missing_ids) == set(decision.question_ids)
    assert decision.coverage == 0.0
    assert decision.call is None  # no response to record
    assert decision.validated.weights == {}


def test_per_asset_asks_about_every_investable_asset(data_service, scenario, settings):
    """The default form: one question per asset, for the whole universe."""
    import asyncio

    from conftest import OfflineDcnClient

    pipeline = DcnPipeline(settings, scenario, OfflineDcnClient(), question_form="per_asset")
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
    assert sorted(decision.tickers) == sorted(t.upper() for t in scenario.tradable)
    # Exactly one question per asset, all of the sizing form.
    assert len(decision.question_ids) == len(scenario.tradable)
    assert all(qid.startswith("noul_") for qid in decision.question_ids)
    assert set(decision.probabilities) == set(decision.tickers)