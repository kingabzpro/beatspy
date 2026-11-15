"""FastTrade's model signal, adapted to BeatSPY's daily bars.

FastTrade normally runs on hourly candles. This port applies the same feature
list to daily bars and asks FastTrade's own BUY/HOLD/SELL question through
TypeSafe Jev. The feature values are pinned to FastTrade's own implementation
on a fixed 260-bar series.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta

import pandas as pd
import pytest

from beatspy.config import DecisionConfig, Settings
from beatspy.data.service import DataService
from beatspy.dcn import fasttrade
from beatspy.dcn.clients import DcnResult
from beatspy.dcn.questions import build_questions
from beatspy.schemas import ChoiceAnswer
from conftest import run_offline

# Printed by FastTrade's `fasttrade.indicators.features` on this exact daily
# series (America/New_York), so a drift in any indicator fails loudly.
FASTTRADE_REFERENCE = {
    "price": 93.60000000000001,
    "ema_9": 92.68234204175509,
    "ema_21": 92.37172018548345,
    "ema_50": 91.64924611634449,
    "rsi_14": 64.91599536390277,
    "macd": 0.3528487043681139,
    "macd_signal": 0.3378794084034129,
    "vwap": 93.58333333333336,
    "vwap_delta": 0.0001780943900264731,
    "atr_14": 0.8695353163847865,
    "volume_ratio": 2.727272727272727,
    "return_1": 0.012987012987013102,
    "return_3": 0.014084507042253724,
    "return_5": 0.0042918454935623185,
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


def _prices(frame: pd.DataFrame, ticker: str = "TEST") -> pd.DataFrame:
    out = frame.reset_index().rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = ticker
    return out[["date", "ticker", "open", "high", "low", "close", "volume"]]


def _reference_service() -> DataService:
    frame = _reference_frame()
    return DataService(_prices(frame), benchmark="TEST")


class FakeFastTradeProvider:
    """Stands in for the TypeSafe call; returns one choice answer per state."""

    provider = "typesafe"
    model = "jev-latest"
    endpoint = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, probabilities: dict | None = None, fail: bool = False):
        self.probabilities = probabilities or {"BUY": 0.6, "HOLD": 0.3, "SELL": 0.1}
        self.fail = fail
        self.states: list[dict] = []

    async def decide(self, state: dict, questions: dict) -> DcnResult:
        self.states.append(state)
        if self.fail:
            raise RuntimeError("provider down")
        choice = max(self.probabilities, key=self.probabilities.get)
        answer = ChoiceAnswer(choice=choice, probabilities=dict(self.probabilities))
        return DcnResult(answers={"action": answer}, input_tokens=120, requests=1, latency_s=0.01)

    async def close(self) -> None:
        return None


def _decide(client: fasttrade.FastTradeClient, as_of: date, ticker: str = "TEST"):
    questions = build_questions("all", [ticker], "SPY", 5).questions
    return asyncio.run(client.decide({"as_of": as_of.isoformat()}, questions))


def test_features_match_fasttrade():
    features = fasttrade.fasttrade_features(_reference_frame())
    assert features is not None
    for key, expected in FASTTRADE_REFERENCE.items():
        assert features[key] == pytest.approx(expected, abs=1e-9), key
    assert features["spread"] is None
    assert features["liquidity"] is None


def test_short_history_has_no_features():
    assert fasttrade.fasttrade_features(_reference_frame().tail(fasttrade.MIN_BARS - 1)) is None


def test_probability_is_the_expected_action_score():
    assert fasttrade.beat_probability({"BUY": 0.6, "HOLD": 0.3, "SELL": 0.1}) == pytest.approx(0.75)
    assert fasttrade.beat_probability({"BUY": 0.0, "HOLD": 1.0, "SELL": 0.0}) == pytest.approx(0.5)
    assert fasttrade.beat_probability({"BUY": 0.0, "HOLD": 0.0, "SELL": 1.0}) == pytest.approx(0.0)
    assert fasttrade.beat_probability({"BUY": 0.6, "HOLD": 0.4}) is None


def test_client_answers_every_question_form():
    frame = _reference_frame()
    provider = FakeFastTradeProvider()
    client = fasttrade.FastTradeClient(_reference_service(), provider, concurrency=2)
    result = _decide(client, frame.index[-1].date())

    assert result.missing_ids == []
    assert result.input_tokens == 120
    assert result.requests == 1
    assert result.answers["noul_TEST"].noul == pytest.approx(0.75)
    assert result.answers["choice_TEST"].probabilities["outperform"] == pytest.approx(0.75)
    assert result.answers["rank_TEST"].score == pytest.approx(3.0)
    state = provider.states[0]
    assert state["timeframe"] == "1d"
    assert state["symbol"] == "TEST"
    assert state["headlines"] == []
    assert state["features"]["ema_9"] == pytest.approx(FASTTRADE_REFERENCE["ema_9"])


def test_client_reports_missing_when_the_provider_fails():
    frame = _reference_frame()
    provider = FakeFastTradeProvider(fail=True)
    client = fasttrade.FastTradeClient(_reference_service(), provider, concurrency=2)
    result = _decide(client, frame.index[-1].date())

    assert result.answers == {}
    questions = build_questions("all", ["TEST"], "SPY", 5).questions
    assert set(result.missing_ids) == set(questions)


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

    full = pd.concat([base, future])
    full_provider = FakeFastTradeProvider()
    truncated_provider = FakeFastTradeProvider()
    full_client = fasttrade.FastTradeClient(
        DataService(_prices(full), benchmark="TEST"), full_provider, concurrency=2
    )
    truncated_client = fasttrade.FastTradeClient(
        DataService(_prices(base), benchmark="TEST"), truncated_provider, concurrency=2
    )

    full_result = _decide(full_client, decision)
    truncated_result = _decide(truncated_client, decision)

    assert full_result.answers["noul_TEST"].noul == truncated_result.answers["noul_TEST"].noul
    assert full_provider.states[0]["candle_timestamp"] == truncated_provider.states[0]["candle_timestamp"]
    assert full_provider.states[0]["features"] == truncated_provider.states[0]["features"]


def test_a_run_records_the_fasttrade_model(tmp_path, scenario, frozen_data):
    settings = Settings(decision=DecisionConfig(decision_model="fasttrade"))
    client = fasttrade.FastTradeClient(frozen_data[0], FakeFastTradeProvider())

    run_dir = run_offline(tmp_path, scenario, frozen_data, client, settings=settings)

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["decision"]["decision_model"] == "fasttrade"
    assert meta["decision"]["provider"] == "fasttrade"
    assert meta["decision"]["category"] == "ai"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["decisions"] >= 1
    assert metrics["answer_coverage"] == 1.0
