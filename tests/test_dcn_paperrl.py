"""PaperRL's five-factor signal.

The indicator values, action and confidence are pinned to PaperRL's own
`recommend` on the same 260-bar series. The per-symbol PPO overlay is not used.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta

import pandas as pd
import pytest

from beatspy.config import DecisionConfig, Settings
from beatspy.data.service import DataService
from beatspy.dcn import paperrl
from beatspy.dcn.questions import build_questions
from conftest import run_offline

REFERENCE_INDICATORS = {
    "price": 93.60000000000001,
    "change_pct": 1.2987012987013102,
    "ema20": 92.39670780037082,
    "ema50": 91.64919384199793,
    "rsi14": 64.91599526590247,
    "macd_hist": 0.014969294969436553,
    "ret5": 0.0042918454935623185,
    "ret20": 0.012439156300703136,
}
REFERENCE_CONFIDENCE = 0.723310932238


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


def _decide(client: paperrl.PaperRLClient, as_of: date):
    questions = build_questions("all", ["TEST"], "SPY", 5).questions
    return asyncio.run(client.decide({"as_of": as_of.isoformat()}, questions))


def test_recommendation_matches_paperrl():
    signal = paperrl.recommend(_reference_frame())
    assert signal is not None
    assert signal.action == "Buy"
    assert signal.confidence == pytest.approx(REFERENCE_CONFIDENCE, abs=1e-12)
    assert signal.probability == pytest.approx(REFERENCE_CONFIDENCE, abs=1e-12)
    for key, expected in REFERENCE_INDICATORS.items():
        assert signal.indicators[key] == pytest.approx(expected, abs=1e-12), key
    assert signal.reasons == [
        "Price 93.60 is above the 50-day EMA (91.65)",
        "EMA20 is above EMA50 (uptrend)",
        "MACD histogram is positive",
    ]


def test_ppo_table_covers_the_scored_windows():
    table = paperrl.load_ppo_table()
    assert set(table) == {"2025-10-6m", "2026-04-6m", "2026-6m"}
    assert table["2025-10-6m"]["weights"]["AAPL"]
    assert table["2026-6m"]["weights"]["NVDA"]


def test_ppo_overlay_adds_the_bonus():
    base = paperrl.recommend(_reference_frame(), "TEST")
    signal = paperrl.recommend(_reference_frame(), "TEST", 0.75)
    assert base is not None and signal is not None
    assert signal.ppo_weight == 0.75
    assert signal.score == pytest.approx(base.score + 0.35, abs=1e-12)
    assert signal.probability == pytest.approx(base.probability + 0.175, abs=1e-12)
    assert "PPO model (seed 42) targets 75% of equity" in signal.reasons


def test_ppo_overlay_subtracts_the_penalty():
    base = paperrl.recommend(_reference_frame(), "TEST")
    signal = paperrl.recommend(_reference_frame(), "TEST", 0.2)
    assert base is not None and signal is not None
    assert signal.score == pytest.approx(base.score - 0.35, abs=1e-12)
    assert signal.probability == pytest.approx(base.probability - 0.175, abs=1e-12)
    assert "PPO model (seed 42) stays near cash (20% weight)" in signal.reasons


def test_ppo_overlay_stays_neutral_between_the_thresholds():
    base = paperrl.recommend(_reference_frame(), "TEST")
    signal = paperrl.recommend(_reference_frame(), "TEST", 0.5)
    assert base is not None and signal is not None
    assert signal.score == pytest.approx(base.score, abs=1e-12)
    assert "PPO model (seed 42) is neutral (50% weight)" in signal.reasons


def test_client_applies_a_table_weight():
    frame = _reference_frame()
    last_date = frame.index[-1].date().isoformat()
    table = {
        "test": {
            "start": "2024-01-01",
            "end": "2024-12-31",
            "weights": {"TEST": {last_date: 0.75}},
        }
    }
    client = paperrl.PaperRLClient(_reference_service(), ppo_table=table)
    result = _decide(client, frame.index[-1].date())
    base = paperrl.recommend(frame, "TEST")

    assert result.missing_ids == []
    assert base is not None
    assert result.answers["noul_TEST"].noul == pytest.approx(base.probability + 0.175, abs=1e-12)


def test_short_history_has_no_signal():
    assert paperrl.recommend(_reference_frame().tail(paperrl.MIN_BARS - 1)) is None


def test_client_answers_every_question_form():
    frame = _reference_frame()
    client = paperrl.PaperRLClient(_reference_service())
    result = _decide(client, frame.index[-1].date())

    assert result.missing_ids == []
    assert result.input_tokens == 0
    assert result.answers["noul_TEST"].noul == pytest.approx(REFERENCE_CONFIDENCE, abs=1e-12)
    assert result.answers["choice_TEST"].probabilities["outperform"] == pytest.approx(
        REFERENCE_CONFIDENCE, abs=1e-12
    )
    assert result.answers["rank_TEST"].score == pytest.approx(
        4.0 * result.answers["noul_TEST"].noul, abs=1e-12
    )


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
    full = _decide(paperrl.PaperRLClient(full_service), decision)
    truncated = _decide(paperrl.PaperRLClient(truncated_service), decision)

    assert full.answers["noul_TEST"].noul == truncated.answers["noul_TEST"].noul


def test_a_run_records_the_paperrl_model(tmp_path, scenario, frozen_data):
    settings = Settings(decision=DecisionConfig(decision_model="paperrl"))
    client = paperrl.PaperRLClient(frozen_data[0])

    run_dir = run_offline(tmp_path, scenario, frozen_data, client, settings=settings)

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["decision"]["decision_model"] == "paperrl"
    assert meta["decision"]["provider"] == "paperrl"
    assert meta["decision"]["category"] == "rl"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["decisions"] >= 1
    assert metrics["answer_coverage"] == 1.0
