"""Point-in-time state: no leakage, deterministic, and inside the token budget.

The no-leakage test is the important one. It builds the same decision from two
price series that are identical up to the decision date and completely different
after it, and requires byte-identical state. If any accessor ever reaches forward,
this fails.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from beatspy.data.service import DataService
from beatspy.dcn.state import (
    build_state,
    candidates,
    render_state,
    state_token_estimate,
)


def _truncated(prices: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    return prices.loc[prices.date <= pd.Timestamp(cutoff)].reset_index(drop=True)


def _split(prices: pd.DataFrame, as_of: str, seed_after: int) -> pd.DataFrame:
    """Keep everything up to `as_of`; replace every later close with noise."""
    import numpy as np

    frame = prices.copy()
    later = frame.date > pd.Timestamp(as_of)
    rng = np.random.default_rng(seed_after)
    frame.loc[later, "close"] = frame.loc[later, "close"].to_numpy() * rng.uniform(0.5, 2.0, int(later.sum()))
    return frame


@pytest.fixture
def prices() -> pd.DataFrame:
    from conftest import make_prices

    return make_prices(["AAPL", "MSFT", "TLT"])


def _state(prices: pd.DataFrame, scenario, as_of: date, **overrides) -> dict:
    data = DataService(prices, benchmark=scenario.benchmark)
    return build_state(
        data,
        scenario,
        as_of,
        holdings=overrides.get("holdings", {}),
        cash=overrides.get("cash", 1.0),
        question_tickers=overrides.get("question_tickers", ["AAPL", "MSFT"]),
        horizon_trading_days=overrides.get("horizon_trading_days", 20),
        next_decision_date=overrides.get("next_decision_date", "2022-03-01"),
    )


def test_state_is_identical_when_only_future_prices_differ(prices, scenario):
    """The decisive no-leakage test: future data cannot change today's state."""
    as_of = date(2022, 2, 15)
    honest = _state(prices, scenario, as_of)
    tampered = _state(_split(prices, "2022-02-15", seed_after=99), scenario, as_of)
    assert honest == tampered


def test_state_is_identical_when_the_future_is_removed_entirely(prices, scenario):
    as_of = date(2022, 2, 15)
    honest = _state(prices, scenario, as_of)
    truncated = _state(_truncated(prices, "2022-02-15"), scenario, as_of)
    assert honest == truncated


def test_state_never_mentions_a_future_date(prices, scenario):
    as_of = date(2022, 2, 15)
    state = _state(prices, scenario, as_of)
    assert state["as_of"] == "2022-02-15"
    for row in state["market"]:
        # last_close is dated by the source frame; nothing may exceed as_of.
        assert row["ticker"]
    rendered = render_state(state)
    assert "2022-02-16" not in rendered
    assert "2022-03" not in rendered.replace('"2022-03-01"', "")  # only next_decision_date


def test_state_is_deterministic(prices, scenario):
    as_of = date(2022, 2, 15)
    assert render_state(_state(prices, scenario, as_of)) == render_state(_state(prices, scenario, as_of))


def test_state_stays_within_the_workers_ai_truncation_budget(prices, scenario):
    """Clef silently drops state past ~2K tokens, so the budget is enforced."""
    state = _state(prices, scenario, date(2022, 2, 15), question_tickers=scenario.tradable)
    estimate = state_token_estimate(render_state(state))
    assert estimate < 900  # three tickers; the full 23-name universe is checked below


def test_state_stays_compact_for_a_large_universe(scenario):
    from conftest import make_prices

    tickers = [f"T{index:02d}" for index in range(23)]
    prices = make_prices(tickers, start="2021-01-01", end="2022-04-01")
    big = scenario.model_copy(update={"tradable": tickers})
    state = _state(prices, big, date(2022, 3, 1), question_tickers=tickers)
    estimate = state_token_estimate(render_state(state))
    assert estimate < 1800, f"state would be truncated by Workers AI at ~2K tokens (est {estimate})"


BIG_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "HD", "WMT", "PG", "COST",
    "JPM", "V", "LLY", "JNJ", "UNH", "XOM", "CVX", "CAT", "GE", "NEE",
    "LIN", "PLD", "AMT", "QQQ", "TLT",
]


def _full_book_state(prices, scenario, tickers, as_of: date) -> dict:
    data = DataService(prices, benchmark=scenario.benchmark)
    holdings = {ticker: 1.0 / len(tickers) for ticker in tickers}
    recent = [
        {"date": f"2022-0{month}-01", "held": ", ".join(f"{ticker} 4%" for ticker in tickers)}
        for month in (1, 2, 3)
    ]
    return build_state(
        data,
        scenario,
        as_of,
        holdings=holdings,
        cash=0.0,
        question_tickers=tickers,
        horizon_trading_days=5,
        next_decision_date="2022-02-28",
        recent=recent,
    )


def test_a_fully_invested_book_exceeds_the_budget_before_enforcement(scenario):
    """The published runs grew past the budget once holdings and history were
    included, which is why enforcement exists rather than a comment."""
    from beatspy.dcn.state import enforce_budget
    from conftest import make_prices

    big = scenario.model_copy(update={"tradable": BIG_UNIVERSE})
    state = _full_book_state(make_prices(big.universe), big, BIG_UNIVERSE, date(2022, 2, 15))
    assert state_token_estimate(render_state(state)) > 1800  # the raw state is over
    trimmed_state, trimmed = enforce_budget(state, 1800)
    assert state_token_estimate(render_state(trimmed_state)) <= 1800
    assert "recent_decisions" in trimmed
    # Deterministic: the same input trims the same way every time.
    again, trimmed_again = enforce_budget(state, 1800)
    assert render_state(again) == render_state(trimmed_state)
    assert trimmed_again == trimmed


def test_enforce_budget_fails_loudly_when_nothing_more_can_be_trimmed(prices, scenario):
    from beatspy.dcn.state import enforce_budget

    state = _state(prices, scenario, date(2022, 2, 15))
    with pytest.raises(ValueError, match="token budget"):
        enforce_budget(state, 20)


def test_the_pipeline_enforces_the_budget_on_a_full_book(scenario):
    """The real decision path sends a state inside the configured budget."""
    import asyncio

    from beatspy.config import Settings
    from beatspy.dcn.pipeline import DcnPipeline
    from conftest import OfflineDcnClient, make_prices, make_scenario

    big = make_scenario(tradable=BIG_UNIVERSE)
    data = DataService(make_prices(big.universe), benchmark=big.benchmark)
    pipeline = DcnPipeline(Settings(), big, OfflineDcnClient(), question_form="per_asset")
    holdings = {ticker: 1.0 / len(BIG_UNIVERSE) for ticker in BIG_UNIVERSE}
    # Same shape the runner writes into `recent`: every holding, spelled out.
    recent = [
        {"date": f"2022-0{month}-01", "held": ", ".join(f"{ticker} 4%" for ticker in BIG_UNIVERSE)}
        for month in (1, 2, 3)
    ]
    decision = asyncio.run(
        pipeline.decide(
            data,
            date(2022, 2, 15),
            holdings=holdings,
            cash=0.0,
            horizon_trading_days=5,
            next_decision_date="2022-02-28",
            recent=recent,
        )
    )
    assert decision.state_trimmed, "a full 25-name book should have needed trimming"
    assert state_token_estimate(decision.state_text) <= Settings().decision.max_state_tokens
    assert decision.state["as_of"] == "2022-02-15"


def test_state_carries_a_point_in_time_base_rate(prices, scenario):
    """A probability needs the frequency it is estimating, so the state carries it."""
    state = _state(prices, scenario, date(2022, 2, 15))
    base = state["base_rate"]
    assert base["window_trading_days"] == 63
    assert 0.0 <= base["share_beat_benchmark_pct"] <= 100.0
    assert base["median_excess_pct"] is not None

    # Point-in-time: tampering with later prices cannot move it.
    frame = prices.copy()
    later = frame["date"] > pd.Timestamp("2022-02-15")
    frame.loc[later, "close"] = frame.loc[later, "close"] * 4.0
    tampered = _state(frame, scenario, date(2022, 2, 15))
    assert tampered["base_rate"] == base


def test_select_universe_defaults_to_the_whole_pool(prices, scenario):
    from beatspy.dcn.state import select_universe

    data = DataService(prices, benchmark=scenario.benchmark)
    assert select_universe(data, scenario, date(2022, 2, 15)) == [t.upper() for t in scenario.tradable]


def test_select_universe_keeps_the_strongest_trailing_performers(prices, scenario):
    """A selection scenario trades only the pool's best trailing performers."""
    from beatspy.dcn.state import select_universe

    data = DataService(prices, benchmark=scenario.benchmark)
    as_of = date(2022, 2, 15)
    lookback = 126
    benchmark_return = data.trailing_return_pct(scenario.benchmark, as_of, lookback) or 0.0
    ranked = sorted(
        ((data.trailing_return_pct(ticker, as_of, lookback) or -1e9) - benchmark_return, ticker)
        for ticker in scenario.tradable
    )
    expected = sorted(ticker for _, ticker in ranked[-2:])

    selected = scenario.model_copy(update={"selection_top": 2})
    assert select_universe(data, selected, as_of) == expected


def test_select_universe_uses_only_data_up_to_the_decision_date(prices, scenario):
    """The ranking is point-in-time: tampering with later prices cannot move it."""
    from beatspy.dcn.state import select_universe

    selected = scenario.model_copy(update={"selection_top": 2})
    as_of = date(2022, 2, 15)
    honest = select_universe(DataService(prices, benchmark=scenario.benchmark), selected, as_of)

    frame = prices.copy()
    later = frame["date"] > pd.Timestamp(as_of)
    frame.loc[later, "close"] = frame.loc[later, "close"] * 3.0
    tampered = select_universe(DataService(frame, benchmark=scenario.benchmark), selected, as_of)
    assert honest == tampered


def test_candidates_are_deterministic_and_bounded(prices, scenario):
    data = DataService(prices, benchmark=scenario.benchmark)
    as_of = date(2022, 2, 15)
    first = candidates(data, scenario, as_of, extra=[], max_candidates=2)
    second = candidates(data, scenario, as_of, extra=[], max_candidates=2)
    assert first == second
    assert len(first) == 2
    assert set(first) <= set(scenario.tradable)


def test_holdings_are_always_asked_about(prices, scenario):
    """The model must be able to act on what it already owns."""
    data = DataService(prices, benchmark=scenario.benchmark)
    as_of = date(2022, 2, 15)
    # A holding takes the first slot even though TLT is the weakest mover here,
    # because the provider limit counts every question the decision will send.
    assert candidates(data, scenario, as_of, extra=["TLT"], max_candidates=1) == ["TLT"]
    roomy = candidates(data, scenario, as_of, extra=["TLT"], max_candidates=2)
    assert "TLT" in roomy and len(roomy) == 2


def test_candidates_ignore_untradable_extras(prices, scenario):
    data = DataService(prices, benchmark=scenario.benchmark)
    picked = candidates(data, scenario, date(2022, 2, 15), extra=["SPY", "NVDA"], max_candidates=2)
    assert "SPY" not in picked
    assert "NVDA" not in picked


def test_state_carries_the_decide_on_list_and_constraints(prices, scenario):
    state = _state(prices, scenario, date(2022, 2, 15), question_tickers=["AAPL", "MSFT"])
    assert state["decide_on"] == ["AAPL", "MSFT"]
    assert state["constraints"]["long_only"] is True
    assert state["portfolio"]["cash_weight_pct"] == 100.0
    assert state["benchmark"]["ticker"] == "SPY"


def test_state_reports_relative_performance_against_the_benchmark(prices, scenario):
    state = _state(prices, scenario, date(2022, 2, 15))
    row = next(entry for entry in state["market"] if entry["ticker"] == "AAPL")
    assert "return_90d_vs_benchmark_pct" in row
    assert row["return_90d_vs_benchmark_pct"] is not None


def test_missing_history_is_omitted_rather_than_zero_filled(scenario):
    """A ticker with no bars must be absent, not reported as a 0% return."""
    from conftest import make_prices

    prices = make_prices(["AAPL"])
    data = DataService(prices, benchmark="SPY")
    state = build_state(
        data,
        scenario,
        date(2022, 2, 15),
        holdings={},
        cash=1.0,
        question_tickers=["AAPL", "MSFT"],
        horizon_trading_days=20,
        next_decision_date=None,
    )
    tickers = {entry["ticker"] for entry in state["market"]}
    assert "MSFT" not in tickers
    assert "AAPL" in tickers