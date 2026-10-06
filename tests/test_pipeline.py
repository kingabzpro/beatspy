"""Decision pipeline tests with scripted executors (no network, no SDK internals)."""

from __future__ import annotations

import pytest

from beatspy.agents.pipeline import DecisionPipeline
from conftest import FakeExecutor, make_tctx


@pytest.fixture
def pipeline(settings, scenario) -> DecisionPipeline:
    return DecisionPipeline(settings, scenario, provider=None, executor=FakeExecutor())


def test_full_decision(pipeline, data_service, scenario):
    tctx = make_tctx(data_service, scenario)
    record = asyncio_run(pipeline.decide(tctx, {}, 1.0, None))

    assert set(record.outcomes) == {"research", "analyst", "forecaster", "critic", "portfolio_manager"}
    assert record.parse_errors == []
    assert record.validated.weights == {"MSFT": pytest.approx(0.30), "TLT": pytest.approx(0.20)}
    assert record.validated.cash == pytest.approx(0.50)
    assert record.predicted_direction == "up"
    assert record.artifacts["research"]["summary"].startswith("Fed")
    totals = record.totals()
    assert totals["input_tokens"] == 500  # 5 agents x 100
    assert totals["requests"] == 5


def test_single_call_has_no_tools_and_keeps_spy_guard(settings, data_service, scenario):
    executor = FakeExecutor(
        {
            "portfolio_manager": '{"allocations":[{"ticker":"SPY","weight":0.3},'
            '{"ticker":"MSFT","weight":0.3}],"cash_weight":0.4}'
        }
    )
    pipeline = DecisionPipeline(settings, scenario, provider=None, executor=executor, protocol_version=6)
    record = asyncio_run(pipeline.decide(make_tctx(data_service, scenario), {}, 1.0, None))
    assert executor.calls == ["portfolio_manager"]
    assert pipeline.agents["portfolio_manager"].tools == []
    assert set(record.outcomes) == {"portfolio_manager"}
    assert record.artifacts == {}
    assert record.validated.weights == {"MSFT": 0.3}
    assert any("SPY" in violation for violation in record.validated.violations)


def test_feedback_uses_current_and_previous_values_only(settings, data_service, scenario):
    from datetime import timedelta

    pipeline = DecisionPipeline(settings, scenario, provider=None, executor=FakeExecutor(), protocol_version=7)
    tctx = make_tctx(data_service, scenario)
    first = asyncio_run(pipeline.decide(tctx, {}, 1, None, portfolio_equity=100000))
    tctx.as_of += timedelta(days=7)
    second = asyncio_run(pipeline.decide(tctx, {}, 1, None, portfolio_equity=102000))
    feedback = second.market_brief["performance_feedback"]
    assert first.market_brief["performance_feedback"]["since_start"]["portfolio_return_pct"] == 0
    assert feedback["previous_window"]["portfolio_return_pct"] == 2
    assert feedback["previous_window"]["start"] == first.date.isoformat()
    start_price = data_service.last_close(scenario.benchmark, first.date)[1]
    current_price = data_service.last_close(scenario.benchmark, second.date)[1]
    assert feedback["previous_window"]["benchmark_return_pct"] == round((current_price / start_price - 1) * 100, 3)


def test_large_company_core_is_fixed_and_cannot_include_spy():
    from beatspy.engine.backtest import large_company_core_fn
    from beatspy.scenarios import load_scenario

    scenario = load_scenario("2026-ytd")
    weights = large_company_core_fn(scenario)(None, None)["weights"]
    assert set(weights) == {"AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"}
    assert all(weight == pytest.approx(1 / 6) for weight in weights.values())
    assert sum(weights.values()) == pytest.approx(1)
    assert max(weights.values()) <= scenario.max_position_weight


def test_invalid_pm_output_holds_portfolio(pipeline, data_service, scenario):
    pipeline.executor.outputs["portfolio_manager"] = "I cannot decide right now, sorry."
    tctx = make_tctx(data_service, scenario)
    record = asyncio_run(pipeline.decide(tctx, {"MSFT": 0.5}, 0.5, None))
    assert record.validated.invalid
    assert record.validated.weights == {}
    assert any("invalid" in e for e in record.parse_errors)


def test_overweight_is_clamped_with_violation(settings, data_service, scenario):
    outputs = {
        "portfolio_manager": '{"allocations": [{"ticker": "MSFT", "weight": 0.9}], "cash_weight": 0.1, "expected_direction": "up", "rationale": "yolo"}'
    }
    pipeline = DecisionPipeline(settings, scenario, provider=None, executor=FakeExecutor(outputs))
    tctx = make_tctx(data_service, scenario)
    record = asyncio_run(pipeline.decide(tctx, {}, 1.0, None))
    assert record.validated.weights["MSFT"] == pytest.approx(scenario.max_position_weight)
    assert any("clamped" in v for v in record.validated.violations)


def test_malformed_research_degrades_gracefully(pipeline, data_service, scenario):
    pipeline.executor.outputs["research"] = "totally not json"
    tctx = make_tctx(data_service, scenario)
    record = asyncio_run(pipeline.decide(tctx, {}, 1.0, None))
    assert any("research" in e for e in record.parse_errors)
    # the PM still received the remaining artifacts and produced its usual decision
    assert record.validated.weights == {"MSFT": pytest.approx(0.30), "TLT": pytest.approx(0.20)}


def test_tool_context_receives_agent_budget(pipeline, data_service, scenario):
    tctx = make_tctx(data_service, scenario, budget=3)
    asyncio_run(pipeline.decide(tctx, {}, 1.0, None))
    # FakeExecutor never calls tools, so nothing was consumed
    assert tctx.used == {}


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


def test_pipeline_brief_has_horizon_costs_reference_and_no_unavailable_news(pipeline, data_service, scenario):
    tctx = make_tctx(data_service, scenario)
    tctx.horizon_days = 7
    record = asyncio_run(pipeline.decide(tctx, {}, 1.0, None))
    brief = record.market_brief
    assert brief["horizon_trading_days"] == 7
    assert brief["transaction_cost_bps_each_way"] == scenario.fee_bps + scenario.slippage_bps
    assert sum(brief["reference_allocation"]["weights"].values()) <= 1
    assert all(w <= scenario.max_position_weight for w in brief["reference_allocation"]["weights"].values())
    assert [tool.name for tool in pipeline.agents["research"].tools] == ["get_market_events"]


def test_compaction_preserves_complete_structured_reports():
    import json

    from beatspy.agents.prompts import _compact

    report = {"forecasts": [{"ticker": str(i), "method_notes": "x" * 1000} for i in range(9)]}
    assert json.loads(_compact(report)) == report


def test_benchmark_is_visible_but_not_investable(settings, data_service, scenario):
    import json

    output = json.dumps(
        {"allocations": [{"ticker": "SPY", "weight": 0.3}, {"ticker": "MSFT", "weight": 0.3}], "cash_weight": 0.4}
    )
    pipeline = DecisionPipeline(settings, scenario, provider=None, executor=FakeExecutor({"portfolio_manager": output}))
    record = asyncio_run(pipeline.decide(make_tctx(data_service, scenario), {}, 1.0, None))
    assert "SPY" in {row["ticker"] for row in record.market_brief["universe_snapshot"]}
    assert "SPY" not in record.market_brief["tradable_tickers"]
    assert record.market_brief["benchmark_role"] == "comparison_only"
    assert "SPY" not in record.market_brief["reference_allocation"]["weights"]
    assert record.validated.weights == {"MSFT": 0.3}
    assert record.validated.cash == pytest.approx(0.7)
    assert any("SPY" in violation for violation in record.validated.violations)


def test_protocol_3_keeps_original_benchmark_allocation_for_replay(settings, data_service, scenario):
    output = '{"allocations":[{"ticker":"SPY","weight":0.3}],"cash_weight":0.7}'
    pipeline = DecisionPipeline(
        settings, scenario, provider=None, executor=FakeExecutor({"portfolio_manager": output}), protocol_version=3
    )
    record = asyncio_run(pipeline.decide(make_tctx(data_service, scenario), {}, 1.0, None))
    assert record.validated.weights == {"SPY": 0.3}
    assert "SPY" in record.market_brief["tradable_tickers"]
    assert "benchmark_role" not in record.market_brief
