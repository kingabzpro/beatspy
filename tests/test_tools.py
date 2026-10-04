"""Tool-layer tests: budgets, forecasters, and tools driven through the real SDK loop."""

from __future__ import annotations

import asyncio
import json

import pytest
from agents import (
    Agent,
    ModelProvider,
    ModelResponse,
    ModelSettings,
    RunConfig,
    Usage,
    set_tracing_disabled,
)
from agents.items import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from beatspy.agents.pipeline import SdkExecutor
from beatspy.tools.forecast import baseline_forecast, get_forecast_fn
from beatspy.tools.market import tools_for_role
from conftest import make_tctx

set_tracing_disabled(True)


# --------------------------------------------------------------------------- #
# Scripted SDK model: replays tool calls and a final message through Runner
# --------------------------------------------------------------------------- #


class ScriptedModel:
    def __init__(self, turns: list):
        self.turns = list(turns)
        self._calls = 0

    async def get_response(
        self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing, **kwargs
    ):
        turn = self.turns.pop(0)
        usage = Usage(requests=1, input_tokens=50, output_tokens=20)
        if turn[0] == "tool":
            self._calls += 1
            _, name, args = turn
            call = ResponseFunctionToolCall(
                name=name,
                arguments=json.dumps(args),
                call_id=f"call{self._calls}",
                type="function_call",
                id=f"fc{self._calls}",
            )
            return ModelResponse(output=[call], usage=usage, response_id=f"r{self._calls}")
        message = ResponseOutputMessage(
            id="m1",
            role="assistant",
            type="message",
            status="completed",
            content=[ResponseOutputText(text=turn[1], annotations=[], type="output_text")],
        )
        return ModelResponse(output=[message], usage=usage, response_id="r2")

    def get_retry_advice(self, *args, **kwargs):
        return None


class StaticProvider(ModelProvider):
    def __init__(self, model):
        self.model = model

    def get_model(self, model_name):
        return self.model


def make_executor(turns: list) -> SdkExecutor:
    provider = StaticProvider(ScriptedModel(turns))
    return SdkExecutor(
        run_config=RunConfig(
            model_provider=provider, model_settings=ModelSettings(temperature=0.0), tracing_disabled=True
        ),
        max_turns=6,
    )


class TestSdkToolLoop:
    def test_concurrent_agents_have_independent_budgets(self, data_service, scenario):
        # regression: the parallel trio shared one mutable `agent` field, so all
        # tool calls landed on a single budget key
        from dataclasses import replace as dc_replace

        base = make_tctx(data_service, scenario, as_of=__import__("datetime").date(2022, 2, 1), budget=1)
        first = dc_replace(base, agent="research")
        second = dc_replace(base, agent="analyst")

        assert first.spend("research", "get_market_events") is None
        assert second.spend("analyst", "get_technical_indicators") is None
        assert second.spend("analyst", "get_technical_indicators") is not None  # analyst's own budget
        assert base.used == {"research": 1, "analyst": 1}
        assert len([e for e in base.events if e["type"] == "budget_exhausted"]) == 1

    def test_verify_price_clamps_future_dates(self, data_service, scenario):
        tctx = make_tctx(data_service, scenario, as_of=__import__("datetime").date(2022, 2, 1))
        agent = Agent(
            name="critic",
            instructions="Call verify_price for SPY on 2030-01-01, then answer.",
            model="critic",
            tools=tools_for_role("critic", web_enabled=False),
        )
        executor = make_executor(
            [
                ("tool", "verify_price", {"ticker": "SPY", "on_date": "2030-01-01"}),
                (
                    "final",
                    '{"issues": [], "contradictions": [], "overall_assessment": "ok", "adjusted_confidence": 0.5}',
                ),
            ]
        )
        outcome = asyncio.run(executor.run(agent, "check", tctx))
        assert outcome.error is None
        assert tctx.used["critic"] == 1
        assert any(v["kind"] == "future_date_request" for v in tctx.violations)

    def test_budget_exhaustion_stops_data_but_not_the_run(self, data_service, scenario):
        tctx = make_tctx(data_service, scenario, as_of=__import__("datetime").date(2022, 2, 1), budget=1)
        agent = Agent(
            name="research",
            instructions="Call get_market_events twice, then answer.",
            model="research",
            tools=tools_for_role("research", web_enabled=False),
        )
        executor = make_executor(
            [
                ("tool", "get_market_events", {"days_back": 5}),
                ("tool", "get_market_events", {"days_back": 5}),
                (
                    "final",
                    '{"summary": "worked with what I had", "key_events": [], "sentiment": {}, "risks": [], "confidence": 0.5}',
                ),
            ]
        )
        outcome = asyncio.run(executor.run(agent, "go", tctx))
        assert outcome.error is None
        assert tctx.used["research"] == 1
        assert len([e for e in tctx.events if e["type"] == "budget_exhausted"]) == 1
        assert "worked with what I had" in outcome.final_output


# --------------------------------------------------------------------------- #
# Forecaster determinism
# --------------------------------------------------------------------------- #


class TestForecasters:
    def test_baseline_deterministic_and_ordered(self):
        closes = [100.0 * (1.0 + i * 0.001) for i in range(60)]
        first = baseline_forecast("SPY", closes, 20)
        second = baseline_forecast("SPY", closes, 20)
        assert first.as_dict() == second.as_dict()
        assert first.p10_pct <= first.p50_pct <= first.p90_pct
        assert first.direction == "up"

    def test_naive_is_flat(self):
        closes = [100.0 * (1.0 + i * 0.001) for i in range(60)]
        result = baseline_forecast("SPY", closes, 20, method="naive")
        assert result.expected_return_pct == 0.0
        assert result.direction == "flat"

    def test_insufficient_history_returns_none(self):
        assert baseline_forecast("SPY", [100.0, 101.0], 20) is None

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            get_forecast_fn("quantum")

    def test_forecast_tool_uses_clamped_data(self, data_service, scenario):
        tctx = make_tctx(data_service, scenario, as_of=__import__("datetime").date(2022, 2, 1))
        closes = tctx.data.closes("SPY", tctx.as_of, 260)
        assert closes
        result = tctx.forecast_fn("SPY", closes, 20)
        assert result is not None
        assert result.ticker == "SPY"
