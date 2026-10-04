"""Market tool integration tests through the real SDK loop with mocked providers."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from datetime import date

import pytest
from agents import Agent, ModelSettings, RunConfig, Runner, set_tracing_disabled
from agents.items import ToolCallOutputItem

from beatspy.tools.context import ToolContext
from beatspy.tools.forecast import chronos_forecast, get_forecast_fn, timegpt_forecast
from beatspy.tools.market import tools_for_role
from conftest import make_tctx
from test_tools import ScriptedModel, StaticProvider

set_tracing_disabled(True)


def run_scripted(turns, agent, tctx):
    config = RunConfig(
        model_provider=StaticProvider(ScriptedModel(turns)),
        model_settings=ModelSettings(temperature=0.0),
        tracing_disabled=True,
    )
    return asyncio.run(Runner.run(agent, "go", context=tctx, max_turns=5, run_config=config))


def tool_outputs(result) -> list[dict]:
    payloads = []
    for item in result.new_items:
        if isinstance(item, ToolCallOutputItem):
            payloads.append(json.loads(str(item.output)))
    return payloads


def news_tctx(data_service, scenario, **kwargs) -> ToolContext:
    tctx = make_tctx(data_service, scenario, as_of=date(2022, 2, 1))
    for key, value in kwargs.items():
        setattr(tctx, key, value)
    return tctx


class TestCompanyNews:
    def test_future_news_is_filtered_and_newest_first(self, data_service, scenario, monkeypatch):
        captured = {}

        def fake_news(api_key, symbol, from_date, to_date, cache_dir=None):
            captured["window"] = (from_date, to_date)
            return [
                {"datetime": 1735689600, "headline": "future headline"},  # 2025-01-01
                {"datetime": 1642723200, "headline": "jan 21 headline"},  # 2022-01-21
                {"datetime": 1643673600, "headline": "feb 1 headline"},  # 2022-02-01
            ]

        monkeypatch.setattr("beatspy.tools.market.web.finnhub_company_news", fake_news)
        tctx = news_tctx(data_service, scenario, finnhub_api_key="k")
        agent = Agent(
            name="research", instructions="news", model="research", tools=tools_for_role("research", web_enabled=False)
        )
        result = run_scripted(
            [("tool", "get_company_news", {"ticker": "AAPL", "days_back": 30}), ("final", "{}")], agent, tctx
        )

        payload = tool_outputs(result)[0]
        assert captured["window"] == ("2022-01-02", "2022-02-01")  # clamped to as_of
        assert [n["headline"] for n in payload["news"]] == ["feb 1 headline", "jan 21 headline"]
        assert all(n["date"] <= "2022-02-01" for n in payload["news"])

    def test_without_key_reports_unconfigured(self, data_service, scenario):
        tctx = news_tctx(data_service, scenario)
        agent = Agent(
            name="research", instructions="news", model="research", tools=tools_for_role("research", web_enabled=False)
        )
        result = run_scripted([("tool", "get_company_news", {"ticker": "AAPL"}), ("final", "{}")], agent, tctx)
        payload = tool_outputs(result)[0]
        assert "not configured" in payload["error"]


class TestFundamentals:
    def test_flags_future_data_risk(self, data_service, scenario, monkeypatch):
        monkeypatch.setattr(
            "beatspy.tools.market.web.finnhub_metrics",
            lambda *a, **k: {"metric": {"peTTM": 25.4, "marketCapitalization": 900000}},
        )
        tctx = news_tctx(data_service, scenario, finnhub_api_key="k")
        agent = Agent(
            name="research",
            instructions="fundamentals",
            model="research",
            tools=tools_for_role("research", web_enabled=False),
        )
        result = run_scripted([("tool", "get_fundamentals", {"ticker": "AAPL"}), ("final", "{}")], agent, tctx)

        payload = tool_outputs(result)[0]
        assert payload["fundamentals"]["peTTM"] == 25.4
        assert "look-ahead bias" in payload["warning"]
        assert any(v["kind"] == "future_data_risk" for v in tctx.violations)


class TestWebSearch:
    def test_passes_through_with_warning_and_date_scoped_cache(self, data_service, scenario, monkeypatch):
        captured = {}

        def fake_answers(api_key, task, cache_dir=None):
            captured["cache_dir"] = str(cache_dir)
            return {"result": "synthesized answer", "sources": ["https://example.com/a"]}

        monkeypatch.setattr("beatspy.tools.market.web.olostep_answers", fake_answers)
        tctx = news_tctx(data_service, scenario, olostep_api_key="k")
        agent = Agent(
            name="research", instructions="search", model="research", tools=tools_for_role("research", web_enabled=True)
        )
        result = run_scripted(
            [("tool", "search_web", {"query": "Fed outlook February 2022"}), ("final", "{}")], agent, tctx
        )

        payload = tool_outputs(result)[0]
        assert payload["answer"] == "synthesized answer"
        assert "2022-02-01" in payload["warning"]
        assert "2022-02-01" in captured["cache_dir"]  # cache key includes the decision date


class TestForecastProviderErrors:
    def test_timegpt_without_key_raises_with_hint(self):
        with pytest.raises(RuntimeError, match="BEATSPY_TIMEGPT_API_KEY"):
            timegpt_forecast("SPY", [100.0 + i for i in range(60)], 20, api_key=None)

    def test_chronos_without_extra_raises_with_hint(self):
        if importlib.util.find_spec("chronos") is not None:
            pytest.skip("chronos is installed; the missing-extra path cannot be exercised")
        with pytest.raises(RuntimeError, match="--extra chronos"):
            chronos_forecast("SPY", [100.0 + i for i in range(60)], 20)

    def test_unknown_provider_name_rejected_at_resolution(self):
        with pytest.raises(ValueError, match="baseline, naive, chronos"):
            get_forecast_fn("quantum")


def test_chronos_quantiles_use_terminal_values_across_sample_paths(monkeypatch):
    import sys
    from types import SimpleNamespace

    import numpy as np

    from beatspy.tools import forecast

    class Tensor:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, key):
            return Tensor(self.values[key])

        def numpy(self):
            return self.values

    samples = np.full((1, 50, 3), 1000.0)
    samples[0, :, -1] = np.arange(110, 160)
    pipeline = SimpleNamespace(predict=lambda **kwargs: Tensor(samples))
    monkeypatch.setattr(forecast, "_chronos_pipeline", pipeline)
    monkeypatch.setitem(sys.modules, "chronos", SimpleNamespace(ChronosPipeline=object))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            manual_seed=lambda seed: None,
            tensor=lambda values, **kwargs: values,
            float32=None,
        ),
    )
    result = chronos_forecast("SPY", [100.0] * 60, 3)
    assert (result.p10_pct, result.p50_pct, result.p90_pct) == (14.9, 34.5, 54.1)
