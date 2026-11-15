"""BetterTradingAgents' Fast profile, deterministic replay.

The indicator values and the three mock agent outputs are pinned to a run of
BetterTradingAgents' own code on the same 260-bar series. This is the Fast
depth profile's rule replay, not the live LLM agents.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta

import pandas as pd
import pytest

from beatspy.config import DecisionConfig, Settings
from beatspy.data.service import DataService
from beatspy.dcn import bettertrading as bt
from beatspy.dcn.questions import build_questions
from conftest import run_offline

REFERENCE_INDICATORS = {
    "price": 93.6,
    "sma_20": 92.38,
    "sma_50": 91.67,
    "rsi_14": 59.1,
    "momentum_10d_pct": 1.79,
    "volatility_annualized_pct": 9.66,
    "change_1d_pct": 1.3,
    "change_5d_pct": 0.43,
    "change_21d_pct": 1.3,
    "change_63d_pct": 3.65,
    "price_above_sma20": True,
    "sma20_above_sma50": True,
    "history_days": 260,
    "macd": 0.353,
    "macd_signal": 0.338,
    "macd_histogram": 0.015,
    "bollinger_percent_b": 1.022,
    "bollinger_width_pct": 2.53,
    "atr_14": 0.85,
    "atr_pct_of_price": 0.91,
    "avg_volume_20d": 1100000,
    "relative_volume": 2.73,
    "forecast_price_5d": 92.72,
    "forecast_change_5d_pct": -0.94,
    "forecast_trend_r2": 0.998,
    "forecast_window_days": 120,
    "forecast_horizon_days": 5,
    "forecast_path": [93.37, 93.57, 92.62, 92.67, 92.72],
    "forecast_method": "ridge_lag",
}

REFERENCE_TECHNICAL = {
    "ticker": "TEST",
    "signal": "bullish",
    "confidence": 0.75,
    "summary": (
        "[mock] Price 93.6 vs SMA20 92.38 / SMA50 91.67, RSI 59.1, 10d momentum "
        "1.79%, 5d forecast -0.94% (trend R2 0.998, ridge_lag) -> bullish."
    ),
}
REFERENCE_FORECAST = {
    "ticker": "TEST",
    "signal": "neutral",
    "confidence": 0.45,
    "summary": (
        "[mock] 5-day forecast -0.94% sits inside the ~1.4% 5-day noise band "
        "via ridge_lag -> neutral."
    ),
}
REFERENCE_MANAGER = {
    "ticker": "TEST",
    "decision": "BUY",
    "confidence": 0.95,
    "probability_beat_spy": 0.95,
    "summary": "[mock] Research net (no debate) -> BUY.",
    "bull_case": "[mock] See bull researcher summary.",
    "bear_case": "[mock] See bear researcher summary.",
    "would_upgrade_if": (
        "[mock] The bear case weakens below 0.30 or fresh research flips the net score."
    ),
    "would_downgrade_if": (
        "[mock] The bull case weakens below 0.30 or fresh research flips the net score."
    ),
}


def _reference_frame(bars: int = 260) -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-02")
    last_close = 80.0
    for i in range(bars):
        base = 80.0 + 0.05 * i
        wave = 0.5 if i % 7 < 3 else -0.5
        price = last_close + 1.2 if i == bars - 1 else base + wave
        rows.append(
            {
                "date": start + pd.Timedelta(days=i),
                "open": price - 0.05,
                "high": price + 0.3,
                "low": price - 0.35,
                "close": price,
                "volume": 3_000_000.0 if i == bars - 1 else 1_000_000.0,
            }
        )
        last_close = price
    return pd.DataFrame(rows).set_index("date")


def _prices(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reset_index().rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = "TEST"
    return out[["date", "ticker", "open", "high", "low", "close", "volume"]]


def _reference_service() -> DataService:
    return DataService(_prices(_reference_frame()), benchmark="TEST")


def _indicators(frame: pd.DataFrame | None = None) -> dict:
    frame = _reference_frame() if frame is None else frame
    return bt.compute_indicators(
        [float(v) for v in frame["close"]],
        [float(v) for v in frame["high"]],
        [float(v) for v in frame["low"]],
        [float(v) for v in frame["volume"]],
    )


def _decide(client: bt.BetterTradingClient, as_of: date):
    questions = build_questions("all", ["TEST"], "SPY", 5).questions
    return asyncio.run(client.decide({"as_of": as_of.isoformat()}, questions))


def test_indicators_match_bettertradingagents():
    assert _indicators() == REFERENCE_INDICATORS


def test_fast_mock_agents_match_bettertradingagents():
    indicators = _indicators()
    technical = bt.technical_mock("TEST", indicators)
    assert technical == REFERENCE_TECHNICAL
    payload = bt.forecast_payload(indicators, len(_reference_frame()))
    assert payload["noise_band_pct"] == 1.36
    assert payload["local_trend_forecast"]["forecast_price_5d"] == 92.72
    forecast = bt.forecast_mock("TEST", payload)
    assert forecast == REFERENCE_FORECAST
    manager = bt.manager_mock(
        "TEST", {"ticker": "TEST", "technical": technical, "forecast": forecast}
    )
    assert manager == REFERENCE_MANAGER


def test_the_direction_specific_probability_is_inverted_for_sell():
    assert bt.beat_probability("BUY", 0.95) == pytest.approx(0.95)
    assert bt.beat_probability("SELL", 0.95) == pytest.approx(0.05)
    assert bt.beat_probability("HOLD", None) == pytest.approx(0.5)


def test_decision_for_the_reference_series():
    result = bt.decision_for(_reference_frame())
    assert result is not None
    assert result.decision == "BUY"
    assert result.probability == pytest.approx(0.95)


def test_client_answers_every_question_form():
    frame = _reference_frame()
    client = bt.BetterTradingClient(_reference_service())
    result = _decide(client, frame.index[-1].date())

    assert result.missing_ids == []
    assert result.input_tokens == 0
    assert result.requests == 0
    assert result.answers["noul_TEST"].noul == pytest.approx(0.95)
    assert result.answers["choice_TEST"].probabilities["outperform"] == pytest.approx(0.95)
    assert result.answers["rank_TEST"].score == pytest.approx(3.8)


def test_client_only_reads_bars_up_to_the_decision_date():
    base = _reference_frame()
    decision = base.index[-1].date()
    future = pd.DataFrame(
        {
            "date": [base.index[-1] + timedelta(days=i) for i in range(1, 41)],
            "open": 50.0,
            "high": 51.0,
            "low": 49.0,
            "close": 50.0,
            "volume": 1_000_000.0,
        }
    ).set_index("date")

    full_service = DataService(_prices(pd.concat([base, future])), benchmark="TEST")
    truncated_service = DataService(_prices(base), benchmark="TEST")
    full = _decide(bt.BetterTradingClient(full_service), decision)
    truncated = _decide(bt.BetterTradingClient(truncated_service), decision)

    assert full.answers["noul_TEST"].noul == truncated.answers["noul_TEST"].noul


def test_a_run_records_the_bettertradingagents_model(tmp_path, scenario, frozen_data):
    settings = Settings(decision=DecisionConfig(decision_model="bettertradingagents"))
    client = bt.BetterTradingClient(frozen_data[0])

    run_dir = run_offline(tmp_path, scenario, frozen_data, client, settings=settings)

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["decision"]["decision_model"] == "bettertradingagents"
    assert meta["decision"]["provider"] == "bta"
    assert meta["decision"]["category"] == "agent"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["decisions"] >= 1
    assert metrics["answer_coverage"] == 1.0
