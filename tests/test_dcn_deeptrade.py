"""DeepTrade: dossier and one structured decision per ticker.

The LLM is faked here; one real call is made only by the operator when
generating a run. The dossier is built from the same verified TPI port.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pandas as pd
import pytest

from beatspy.config import DecisionConfig, Settings
from beatspy.data.service import DataService
from beatspy.dcn import deeptrade
from beatspy.dcn.questions import build_questions
from conftest import run_offline


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int


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


def _service() -> DataService:
    return DataService(_prices(_reference_frame()), benchmark="TEST")


def _client(monkeypatch) -> deeptrade.DeepTradeClient:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    client = deeptrade.DeepTradeClient(_service())
    client._openai = lambda: object()
    return client


def _decide(client: deeptrade.DeepTradeClient):
    frame = _reference_frame()
    questions = build_questions("all", ["TEST"], "SPY", 5).questions
    return asyncio.run(client.decide({"as_of": frame.index[-1].date().isoformat()}, questions))


def test_dossier_carries_tpi_and_statistics():
    frame = _reference_frame()
    doc = deeptrade.dossier(
        frame,
        "TEST",
        frame.index[-1].date(),
        base=55.0,
        cross_section={
            "tpi_percentile": 50.0,
            "return_20d_percentile": 50.0,
            "tickers_in_slice": 1,
        },
    )
    assert doc is not None
    assert doc["tpi"]["score"] == pytest.approx(96.0, abs=1e-9)
    assert doc["tpi"]["buy"] is True
    assert doc["returns_pct"]["20d"] is not None
    assert doc["base_rate_pct"] == 55.0
    assert doc["cross_section"]["tickers_in_slice"] == 1


def test_beat_probability_maps_the_direction():
    def decision(name: str, probability: float | None):
        return deeptrade.Decision(
            ticker="T", decision=name, confidence=0.7, probability_beat_spy=probability
        )

    assert deeptrade.beat_probability(decision("BUY", 0.8)) == pytest.approx(0.8)
    assert deeptrade.beat_probability(decision("SELL", 0.8)) == pytest.approx(0.2)
    assert deeptrade.beat_probability(decision("HOLD", None)) == pytest.approx(0.5)


def test_client_uses_one_decision_per_ticker(monkeypatch):
    client = _client(monkeypatch)
    usage = Usage(120, 20)

    async def fake_one(_client, doc, horizon):
        return (
            deeptrade.Decision(
                ticker=doc["ticker"],
                decision="BUY",
                confidence=0.8,
                probability_beat_spy=0.75,
                reason="test",
            ),
            usage,
            0.01,
        )

    client._one = fake_one
    result = _decide(client)
    asyncio.run(client.close())

    assert result.answers["noul_TEST"].noul == pytest.approx(0.75)
    assert result.answers["choice_TEST"].probabilities["outperform"] == pytest.approx(0.75)
    assert result.answers["rank_TEST"].score == pytest.approx(3.0)
    assert result.input_tokens == 120
    assert result.output_tokens == 20
    assert result.requests == 1


def test_client_reports_missing_on_failure(monkeypatch):
    client = _client(monkeypatch)

    async def boom(_client, doc, horizon):
        raise RuntimeError("provider down")

    client._one = boom
    result = _decide(client)
    asyncio.run(client.close())

    assert result.answers == {}
    assert result.missing_ids
    assert all(question_id.startswith(("noul_", "choice_", "rank_")) for question_id in result.missing_ids)


def test_a_run_records_the_deeptrade_model(tmp_path, scenario, frozen_data, monkeypatch):
    settings = Settings(decision=DecisionConfig(decision_model="deeptrade"))
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    client = deeptrade.DeepTradeClient(frozen_data[0])
    client._openai = lambda: object()

    run_dir = run_offline(tmp_path, scenario, frozen_data, client, settings=settings)

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["decision"]["decision_model"] == "deeptrade"
    assert meta["decision"]["provider"] == "deeptrade"
    assert meta["decision"]["category"] == "hybrid"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["decisions"] >= 1
    # The fixture has fewer than the 220 bars DeepTrade needs.
    assert metrics["answer_coverage"] == 0.0
