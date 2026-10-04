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
    assert record.validated.weights == {"SPY": pytest.approx(0.30), "TLT": pytest.approx(0.20)}
    assert record.validated.cash == pytest.approx(0.50)
    assert record.predicted_direction == "up"
    assert record.artifacts["research"]["summary"].startswith("Fed")
    totals = record.totals()
    assert totals["input_tokens"] == 500  # 5 agents x 100
    assert totals["requests"] == 5


def test_invalid_pm_output_holds_portfolio(pipeline, data_service, scenario):
    pipeline.executor.outputs["portfolio_manager"] = "I cannot decide right now, sorry."
    tctx = make_tctx(data_service, scenario)
    record = asyncio_run(pipeline.decide(tctx, {"SPY": 0.5}, 0.5, None))
    assert record.validated.invalid
    assert record.validated.weights == {}
    assert any("invalid" in e for e in record.parse_errors)


def test_overweight_is_clamped_with_violation(settings, data_service, scenario):
    outputs = {
        "portfolio_manager": '{"allocations": [{"ticker": "SPY", "weight": 0.9}], "cash_weight": 0.1, "expected_direction": "up", "rationale": "yolo"}'
    }
    pipeline = DecisionPipeline(settings, scenario, provider=None, executor=FakeExecutor(outputs))
    tctx = make_tctx(data_service, scenario)
    record = asyncio_run(pipeline.decide(tctx, {}, 1.0, None))
    assert record.validated.weights["SPY"] == pytest.approx(scenario.max_position_weight)
    assert any("clamped" in v for v in record.validated.violations)


def test_malformed_research_degrades_gracefully(pipeline, data_service, scenario):
    pipeline.executor.outputs["research"] = "totally not json"
    tctx = make_tctx(data_service, scenario)
    record = asyncio_run(pipeline.decide(tctx, {}, 1.0, None))
    assert any("research" in e for e in record.parse_errors)
    # the PM still received the remaining artifacts and produced its usual decision
    assert record.validated.weights == {"SPY": pytest.approx(0.30), "TLT": pytest.approx(0.20)}


def test_tool_context_receives_agent_budget(pipeline, data_service, scenario):
    tctx = make_tctx(data_service, scenario, budget=3)
    asyncio_run(pipeline.decide(tctx, {}, 1.0, None))
    # FakeExecutor never calls tools, so nothing was consumed
    assert tctx.used == {}


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
