"""BetterTradingAgents' Fast profile with live model calls.

The HTTP transport is mocked here; one real call is made only by the operator
when generating a run. The live client is the same three-agent pipeline as the
rule replay, but technical, forecast and manager are model calls.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pandas as pd
import pytest

from beatspy.config import DecisionConfig, Settings
from beatspy.data.service import DataService
from beatspy.dcn import bettertrading_live as live
from beatspy.dcn.clients import DcnError
from beatspy.dcn.questions import build_questions
from conftest import run_offline


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


def _client(monkeypatch, chat) -> live.BetterTradingLiveClient:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    client = live.BetterTradingLiveClient(_service())
    client._chat = chat
    return client


def _chat(manager: dict, technical: dict | None = None, forecast: dict | None = None, calls: list | None = None):
    async def fake(config, system, user):
        if calls is not None:
            calls.append((system, user))
        if "Technical Analyst" in system:
            data = technical or {"signal": "bullish", "confidence": 0.8, "summary": "tech"}
            return data, {"prompt_tokens": 100, "completion_tokens": 10}, 1
        if "Forecast Analyst" in system:
            data = forecast or {"signal": "neutral", "confidence": 0.5, "summary": "forecast"}
            return data, {"prompt_tokens": 100, "completion_tokens": 10}, 1
        return manager, {"prompt_tokens": 300, "completion_tokens": 30}, 1

    return fake


def _decide(client: live.BetterTradingLiveClient):
    frame = _reference_frame()
    questions = build_questions("noul", ["TEST"], "SPY", 5).questions
    return asyncio.run(client.decide({"as_of": frame.index[-1].date().isoformat()}, questions))


def test_parse_json_handles_fences_and_prose():
    assert live._parse_json('```json\n{"ok": true}\n```') == {"ok": True}
    assert live._parse_json('Here is the result: {"ok": true} thanks') == {"ok": True}
    with pytest.raises(ValueError):
        live._parse_json("not json")


def test_manager_prompt_uses_the_five_day_horizon():
    prompt = live.manager_prompt("AAPL", {"ticker": "AAPL"})
    assert "5 trading days" in prompt
    assert "21 days" not in prompt


def test_buy_uses_the_manager_probability(monkeypatch):
    manager = {"decision": "BUY", "confidence": 0.7, "probability_beat_spy": 0.8}
    client = _client(monkeypatch, _chat(manager))
    result = _decide(client)
    asyncio.run(client.close())

    assert result.answers["noul_TEST"].noul == pytest.approx(0.8)
    assert result.input_tokens == 500
    assert result.output_tokens == 50
    assert result.requests == 3


def test_sell_inverts_the_manager_probability(monkeypatch):
    manager = {"decision": "SELL", "confidence": 0.7, "probability_beat_spy": 0.8}
    client = _client(monkeypatch, _chat(manager))
    result = _decide(client)
    asyncio.run(client.close())

    assert result.answers["noul_TEST"].noul == pytest.approx(0.2)


def test_hold_is_neutral(monkeypatch):
    manager = {"decision": "HOLD", "confidence": 0.6, "probability_beat_spy": None}
    client = _client(monkeypatch, _chat(manager))
    result = _decide(client)
    asyncio.run(client.close())

    assert result.answers["noul_TEST"].noul == pytest.approx(0.5)


def test_an_invalid_manager_decision_is_missing_coverage(monkeypatch):
    manager = {"decision": "MAYBE", "confidence": 0.6, "probability_beat_spy": 0.7}
    client = _client(monkeypatch, _chat(manager))
    result = _decide(client)
    asyncio.run(client.close())

    assert result.answers == {}
    assert result.missing_ids == ["noul_TEST"]


def test_the_manager_sees_the_technical_and_forecast_results(monkeypatch):
    calls: list = []
    manager = {"decision": "BUY", "confidence": 0.7, "probability_beat_spy": 0.8}
    client = _client(monkeypatch, _chat(manager, calls=calls))
    _decide(client)
    asyncio.run(client.close())

    manager_prompt = calls[-1][1]
    assert '"technical"' in manager_prompt
    assert '"forecast"' in manager_prompt
    assert "short_term" in manager_prompt


def test_the_client_needs_an_api_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY_ANALYSTS", raising=False)
    monkeypatch.delenv("LLM_API_KEY_MANAGER", raising=False)
    with pytest.raises(DcnError, match="LLM_API_KEY"):
        live.BetterTradingLiveClient(_service())


def test_the_http_transport_parses_a_chat_completion(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    client = live.BetterTradingLiveClient(_service())
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"signal": "bullish"}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    asyncio.run(client._client.aclose())
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    data, usage, attempts = asyncio.run(client._chat(client._analysts, "system", "user"))
    asyncio.run(client.close())

    assert captured["url"] == "https://example.invalid/v1/chat/completions"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["temperature"] == 0.0
    assert data == {"signal": "bullish"}
    assert usage["prompt_tokens"] == 12
    assert attempts == 1


def test_the_http_transport_adds_v1_to_a_bare_server_root(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    client = live.BetterTradingLiveClient(_service())
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"signal": "bullish"}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    asyncio.run(client._client.aclose())
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    asyncio.run(client._chat(client._analysts, "system", "user"))
    asyncio.run(client.close())

    assert captured["url"] == "https://example.invalid/v1/chat/completions"


def test_a_run_records_the_live_model(tmp_path, scenario, frozen_data, monkeypatch):
    settings = Settings(decision=DecisionConfig(decision_model="bettertradingagents-live"))
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    client = live.BetterTradingLiveClient(frozen_data[0])
    manager = {"decision": "BUY", "confidence": 0.7, "probability_beat_spy": 0.8}
    client._chat = _chat(manager)

    run_dir = run_offline(tmp_path, scenario, frozen_data, client, settings=settings)
    asyncio.run(client.close())

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["decision"]["decision_model"] == "bettertradingagents-live"
    assert meta["decision"]["provider"] == "bta-live"
    assert meta["decision"]["category"] == "agent"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["decisions"] >= 1
    assert metrics["answer_coverage"] == 1.0
