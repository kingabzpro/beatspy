"""No look-ahead, enforced on every commit, across the whole decision chain.

`test_dcn_state.py` proves that one decision is unaffected by future prices.
This proves the same for the chained path the runner actually walks: holdings and
recent history flow from decision to decision, so a leak anywhere in that chain
would surface here. The test also proves it has power, by requiring later
decisions to change and by requiring the pre-cut decisions to differ once their
label horizon crosses the cut.
"""

from __future__ import annotations

import asyncio
from datetime import date

import numpy as np
import pandas as pd
import pytest

from beatspy.config import Settings
from beatspy.data.service import DataService
from beatspy.dcn.clients import DcnResult
from beatspy.dcn.pipeline import DcnPipeline
from beatspy.schemas import NoulAnswer
from conftest import make_prices, make_scenario


class StateSensitiveClient:
    """Answers as a deterministic function of the state, so a leak propagates.

    A client that hashes question ids would answer identically even if the state
    changed; this one reads the state's own numbers.
    """

    provider = "offline"
    model = "state-sensitive"
    endpoint = "https://offline.invalid"

    async def decide(self, state, questions):
        market = {row["ticker"]: row for row in state["market"]}
        answers = {}
        for question_id in questions:
            row = market.get(question_id.split("_", 1)[-1]) or {}
            five_day = float(row.get("last_5d_pct") or 0.0)
            probability = 0.5 + five_day / 50.0
            answers[question_id] = NoulAnswer(noul=round(min(max(probability, 0.01), 0.99), 4))
        return DcnResult(answers=answers, input_tokens=10, attempts=1)

    async def close(self):
        return None


def _tamper_after(prices: pd.DataFrame, cut: date) -> pd.DataFrame:
    frame = prices.copy()
    later = frame["date"] > pd.Timestamp(cut)
    rng = np.random.default_rng(11)
    factor = rng.uniform(0.5, 1.5, int(later.sum()))
    for column in ("open", "high", "low", "close"):
        frame.loc[later, column] = frame.loc[later, column].to_numpy() * factor
    frame.loc[later, "volume"] = frame.loc[later, "volume"] * 3.0
    return frame


def _run_chain(data: DataService, scenario) -> list:
    """Walk every decision exactly as the runner does, chaining holdings."""
    start, end = date.fromisoformat(scenario.start), date.fromisoformat(scenario.end)
    decision_dates = data.decision_dates(start, end, scenario.frequency, max_decisions=scenario.max_decisions)

    async def walk():
        pipeline = DcnPipeline(Settings(), scenario, StateSensitiveClient(), question_form="per_asset")
        holdings, recent, decisions = {}, [], []
        for day in decision_dates:
            period = data.decision_period(day, decision_dates, end)
            decision = await pipeline.decide(
                data,
                day,
                holdings=holdings,
                cash=1.0 - sum(holdings.values()),
                horizon_trading_days=period["horizon_trading_days"],
                next_decision_date=period["next_decision_date"],
                recent=list(recent[-3:]),
            )
            decisions.append(decision)
            holdings = dict(decision.validated.weights)
            held = ", ".join(f"{ticker} {weight:.0%}" for ticker, weight in sorted(holdings.items()))
            recent.append({"date": day.isoformat(), "held": held or "all cash"})
        return decisions

    return asyncio.run(walk())


@pytest.fixture
def weekly_scenario():
    """Weekly decisions over three months, so several fall either side of the cut."""
    return make_scenario(
        name="leak-test",
        start="2022-01-03",
        end="2022-03-31",
        frequency="weekly",
        tradable=["AAPL", "MSFT", "TLT"],
        # The selection rule is part of the path under test: it ranks on trailing
        # returns, so a leak there would show up as a changed pre-cut decision.
        selection_top=2,
        selection_lookback_days=63,
    )


def test_future_prices_cannot_change_a_past_decision(weekly_scenario):
    cut = date(2022, 2, 15)
    prices = make_prices(weekly_scenario.universe)
    honest = _run_chain(DataService(prices, benchmark="SPY"), weekly_scenario)
    cheater = _run_chain(DataService(_tamper_after(prices, cut), benchmark="SPY"), weekly_scenario)

    before = [d for d in honest if d.date <= cut]
    after = [d for d in honest if d.date > cut]
    # Guard against a vacuous test: both sides of the cut must exist.
    assert len(before) >= 3 and len(after) >= 3

    for original, tampered in zip(honest, cheater, strict=True):
        if original.date > cut:
            continue
        assert original.state_text == tampered.state_text, f"state leaked future data at {original.date}"
        assert original.question_ids == tampered.question_ids
        assert original.probabilities == tampered.probabilities
        assert original.validated.weights == tampered.validated.weights

    # Power: every later decision must see the tampered data, or the test above
    # proves nothing.
    changed = sum(
        1
        for original, tampered in zip(honest, cheater, strict=True)
        if original.date > cut and original.state_text != tampered.state_text
    )
    assert changed == len(after), "tampering did not reach the post-cut decisions"


def test_label_horizons_read_the_realized_future(weekly_scenario):
    """Labels must use data after the decision date, or calibration measures nothing."""
    cut = date(2022, 2, 15)
    prices = make_prices(weekly_scenario.universe)
    honest = DataService(prices, benchmark="SPY")
    cheater = DataService(_tamper_after(prices, cut), benchmark="SPY")
    decision_dates = honest.decision_dates(
        date.fromisoformat(weekly_scenario.start),
        date.fromisoformat(weekly_scenario.end),
        weekly_scenario.frequency,
        max_decisions=weekly_scenario.max_decisions,
    )

    changed = 0
    for day in (d for d in decision_dates if d <= cut):
        horizon = honest.decision_period(day, decision_dates, date.fromisoformat(weekly_scenario.end))[
            "horizon_trading_days"
        ]
        if honest.close_on_offset("SPY", day, horizon) != cheater.close_on_offset("SPY", day, horizon):
            changed += 1
    assert changed >= 1, "no label read beyond the cut; the tamper did not reach the label window"
