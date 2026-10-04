"""Shared test fixtures. All tests are hermetic: no network, no real models."""

from __future__ import annotations

import json
from datetime import date

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from beatspy.agents.pipeline import AgentOutcome
from beatspy.config import Settings
from beatspy.data.service import DataService
from beatspy.schemas import Scenario
from beatspy.tools.context import ToolContext
from beatspy.tools.forecast import get_forecast_fn

TICKERS = ["AAPL", "MSFT", "TLT"]


def make_prices(
    tickers: list[str] | None = None,
    start: str = "2021-09-01",
    end: str = "2022-07-05",
    seed: int = 7,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """Deterministic synthetic OHLCV covering the scenario window plus lookback."""
    tradable = [t for t in (tickers or TICKERS) if t != benchmark]
    tickers = [benchmark, *tradable]
    rng = np.random.default_rng(seed)
    calendar = xcals.get_calendar("XNYS", start=start, end=end)
    days = calendar.sessions
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    rows = []
    for ticker in tickers:
        n = len(days)
        drift = -0.002 if ticker == benchmark else float(rng.normal(0.0, 0.001))
        close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
        open_ = close * (1.0 + rng.normal(0.0, 0.003, n))
        for i, day in enumerate(days):
            rows.append(
                (
                    day.date().isoformat(),
                    ticker,
                    float(open_[i]),
                    float(close[i]) * 1.01,
                    float(close[i]) * 0.99,
                    float(close[i]),
                    1_000_000.0,
                )
            )
    return pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "volume"]).assign(
        date=lambda df: pd.to_datetime(df["date"])
    )


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(
        name="test-scenario",
        title="Test Scenario",
        description="synthetic",
        start="2022-01-03",
        end="2022-03-31",
        frequency="monthly",
        benchmark="SPY",
        tradable=TICKERS,
        initial_capital=100_000.0,
    )


@pytest.fixture
def data_service(scenario: Scenario) -> DataService:
    return DataService(make_prices(scenario.universe), benchmark=scenario.benchmark)


@pytest.fixture
def settings() -> Settings:
    return Settings()


def make_tctx(
    data_service: DataService, scenario: Scenario, as_of: date | None = None, budget: int = 10
) -> ToolContext:
    return ToolContext(
        as_of=as_of or date(2022, 2, 1),
        data=data_service,
        scenario=scenario,
        forecast_fn=get_forecast_fn("baseline"),
        tool_budget_per_agent=budget,
    )


# --------------------------------------------------------------------------- #
# Canned agent outputs
# --------------------------------------------------------------------------- #

RESEARCH_JSON = json.dumps(
    {
        "summary": "Fed tightening is pressuring equities; energy strong.",
        "key_events": [{"date": "2022-02-24", "headline": "Russia invades Ukraine", "relevance": "risk-off"}],
        "sentiment": {"AAPL": -0.3, "MSFT": -0.2, "TLT": 0.2},
        "risks": ["inflation"],
        "confidence": 0.6,
    }
)

ANALYST_JSON = json.dumps(
    {
        "trend": {
            "AAPL": {"trend": "down", "momentum_30d_pct": -5.0, "volatility_regime": "high", "notes": ""},
            "SPY": {"trend": "down", "momentum_30d_pct": -6.0, "volatility_regime": "high", "notes": ""},
        },
        "opportunities": ["energy strength"],
        "risks": ["rate hikes"],
        "confidence": 0.55,
    }
)

FORECAST_JSON = json.dumps(
    {
        "forecasts": [
            {
                "ticker": "SPY",
                "direction": "down",
                "expected_return_pct": -2.0,
                "p10_pct": -8.0,
                "p90_pct": 4.0,
                "confidence": 0.5,
            }
        ],
        "method_notes": "tool-based",
    }
)

CRITIC_JSON = json.dumps(
    {
        "issues": [{"agent": "forecaster", "claim": "SPY -2%", "problem": "ignores event risk", "severity": "medium"}],
        "contradictions": [],
        "overall_assessment": "reasonable but uncertain",
        "adjusted_confidence": 0.5,
    }
)

PM_JSON = json.dumps(
    {
        "allocations": [{"ticker": "SPY", "weight": 0.30}, {"ticker": "TLT", "weight": 0.20}],
        "cash_weight": 0.50,
        "expected_direction": "up",
        "expected_return_pct": 1.0,
        "rationale": "Defensive tilt.",
    }
)


DEFAULT_OUTPUTS = {
    "research": RESEARCH_JSON,
    "analyst": ANALYST_JSON,
    "forecaster": FORECAST_JSON,
    "critic": CRITIC_JSON,
    "portfolio_manager": PM_JSON,
}


class FakeExecutor:
    """Scripted AgentExecutor: role -> output text (or callable(decision_index))."""

    def __init__(self, outputs: dict[str, object] | None = None):
        self.outputs = dict(outputs or DEFAULT_OUTPUTS)
        self.calls: list[str] = []
        self.index = 0

    async def run(self, agent, input_text: str, tctx: ToolContext) -> AgentOutcome:
        self.calls.append(agent.name)
        spec = self.outputs.get(agent.name, "{}")
        text = spec(self.index) if callable(spec) else spec
        return AgentOutcome(role=agent.name, final_output=text, input_tokens=100, output_tokens=50, requests=1)
