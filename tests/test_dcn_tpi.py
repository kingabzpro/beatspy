"""The deterministic TPI-RS baseline.

The port is checked against indicator values and a composite score produced by
the Rust implementation on a fixed 260-bar series. That keeps the Python adapter
honest: if RSI, ATR, ADX, MACD, the channels or the EMA stack drift, one of the
pinned values moves.

No network, no credentials, and no real models: the client reads the same frozen
`DataService` the benchmark harness already holds.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta

import pandas as pd
import pytest

from beatspy.config import DecisionConfig, Settings
from beatspy.data.service import DataService
from beatspy.dcn import tpi
from beatspy.dcn.questions import build_questions
from conftest import run_offline

# Reference bars from the TPI-RS unit test `breakout_in_uptrend_is_buy_with_stop_below_entry`,
# evaluated at the last bar with the short horizon. The indicator values and the
# score were printed by the Rust implementation on this exact series.
RUST_REFERENCE = {
    "close": 93.60000000000001,
    "ema20": 92.39670780037417,
    "ema50": 91.64924611634449,
    "ema200": 88.31857186790647,
    "ema50_slope_pct": 0.5449440235014246,
    "rsi14": 64.91599526590247,
    "atr14": 0.8695353170076665,
    "atr_pct": 0.928990723298789,
    "adx14": 15.32454595001902,
    "plus_di": 26.523864556507736,
    "minus_di": 14.334885960361552,
    "don20_high": 93.5,
    "don40_high": 93.5,
    "don40_low": 90.15,
    "macd_hist": 0.01496929596470098,
    "volume_ratio": 2.727272727272727,
    "dist_ema20_atr": 1.3838336132990283,
}
RUST_REFERENCE_SCORE = 96.0


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


def _prices(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = frame.reset_index().rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = ticker
    return out[["date", "ticker", "open", "high", "low", "close", "volume"]]


def _benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": frame.index,
            "ticker": "SPY",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000.0,
        }
    )
    return out


def _reference_service() -> DataService:
    frame = _reference_frame()
    prices = pd.concat([_prices(frame, "TEST"), _benchmark(frame)], ignore_index=True)
    return DataService(prices, benchmark="SPY")


def _decide(client: tpi.TpiClient, as_of: date, ticker: str = "TEST"):
    questions = build_questions("noul", [ticker], "SPY", 5).questions
    return asyncio.run(client.decide({"as_of": as_of.isoformat()}, questions))


def test_features_match_the_rust_implementation():
    features = tpi._features(_reference_frame())
    assert features is not None
    for name, expected in RUST_REFERENCE.items():
        assert getattr(features, name) == pytest.approx(expected, abs=1e-9), name
    # The Rust momentum component proves RSI is above its value three bars back.
    assert features.rsi14_prev3 < features.rsi14


def test_score_matches_the_rust_implementation():
    signal = tpi.signal(_reference_frame())
    assert signal is not None
    assert signal.buy is True
    assert signal.score == pytest.approx(RUST_REFERENCE_SCORE, abs=1e-9)


def test_no_breakout_is_not_a_buy():
    frame = _reference_frame().copy()
    # Close below the prior 20-day high, with consistent OHLC. The breakout gate
    # fails even though most of the score survives.
    frame.loc[frame.index[-1], ["high", "low", "close"]] = [93.3, 92.7, 93.0]
    signal = tpi.signal(frame)
    assert signal is not None
    assert signal.buy is False
    assert signal.score < RUST_REFERENCE_SCORE


def test_short_history_produces_no_signal():
    assert tpi.signal(_reference_frame().tail(tpi.MIN_BARS - 1)) is None


def test_probability_tracks_the_score_without_a_gate_step():
    assert tpi.probability(50.0) == pytest.approx(0.5)
    assert tpi.probability(70.0) > tpi.probability(50.0) > tpi.probability(30.0)
    assert 0.25 < tpi.probability(0.0) < tpi.probability(100.0) < 0.75
    # A score above the old BUY threshold no longer flips sign at the gate.
    assert tpi.probability(62.0) > 0.5
    assert tpi.probability(61.0) > 0.5


def test_client_reports_a_warmup_as_missing(data_service):
    client = tpi.TpiClient(data_service)
    questions = build_questions("all", ["AAPL", "MSFT"], "SPY", 5).questions
    result = asyncio.run(client.decide({"as_of": "2022-02-01"}, questions))

    # The fixture has only ~85 bars before the decision, below the 220-bar warmup.
    # A missing answer is coverage, not a fabricated coin flip.
    assert result.answers == {}
    assert set(result.missing_ids) == set(questions)


def test_client_answers_every_question_form_with_a_real_signal():
    frame = _reference_frame()
    client = tpi.TpiClient(_reference_service())
    questions = build_questions("all", ["TEST"], "SPY", 5).questions
    result = asyncio.run(
        client.decide({"as_of": frame.index[-1].date().isoformat()}, questions)
    )

    assert set(result.answers) == set(questions)
    assert result.missing_ids == []
    noul = result.answers["noul_TEST"].noul
    assert 0.7 < noul < 0.75
    choice = result.answers["choice_TEST"]
    assert choice.choice == "outperform"
    assert choice.probabilities["outperform"] == pytest.approx(noul)
    assert sum(choice.probabilities.values()) == pytest.approx(1.0)
    assert result.answers["rank_TEST"].score == pytest.approx(4.0 * noul)


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

    full_prices = pd.concat(
        [_prices(pd.concat([base, future]), "TEST"), _benchmark(pd.concat([base, future]))],
        ignore_index=True,
    )
    truncated_prices = pd.concat(
        [_prices(base, "TEST"), _benchmark(base)],
        ignore_index=True,
    )

    full = _decide(tpi.TpiClient(DataService(full_prices, benchmark="SPY")), decision)
    truncated = _decide(tpi.TpiClient(DataService(truncated_prices, benchmark="SPY")), decision)

    assert full.answers["noul_TEST"].noul > 0.7
    assert full.answers["noul_TEST"].noul == truncated.answers["noul_TEST"].noul


def test_a_run_records_the_baseline_and_missing_warmup(tmp_path, scenario, frozen_data):
    settings = Settings(decision=DecisionConfig(decision_model="tpi-rs"))
    client = tpi.TpiClient(frozen_data[0])

    run_dir = run_offline(tmp_path, scenario, frozen_data, client, settings=settings)

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["decision"]["decision_model"] == "tpi-rs"
    assert meta["decision"]["provider"] == "local"
    assert meta["decision"]["pricing"]["cost_per_m_input"] == 0.0

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["decisions"] >= 1
    # The fixture's 90-day lookback never reaches 220 bars, so the run must record
    # zero coverage instead of scoring neutral answers as if they were real.
    assert metrics["answer_coverage"] == 0.0
