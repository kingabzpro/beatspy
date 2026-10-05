"""Backtest accounting and metric tests, with hand-computed expectations."""

from __future__ import annotations

import asyncio
from datetime import date

import numpy as np
import pandas as pd
import pytest

from beatspy.engine.backtest import (
    PortfolioState,
    execute_target,
    run_backtest,
)
from beatspy.engine.metrics import (
    annualized_volatility,
    cagr,
    directional_accuracy,
    max_drawdown,
    outperformance_hit_rate,
    sharpe_ratio,
    total_return,
    window_returns,
)


class TestExecuteTarget:
    def test_full_allocation_respects_fees(self, data_service, scenario):
        state = PortfolioState(cash=100_000.0)
        exec_date = date(2022, 2, 1)
        price = data_service.open_on("SPY", exec_date)
        fee_rate = scenario.total_fee_rate

        result = execute_target(state, {"SPY": 1.0}, date(2022, 1, 31), exec_date, data_service, fee_rate)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.action == "buy"
        assert trade.price == pytest.approx(price)
        # the fee-adjusted target keeps cash at (or a hair above) zero, never negative
        assert state.cash == pytest.approx(0.0, abs=0.02)
        assert state.positions["SPY"] == pytest.approx((100_000.0 / price) / (1 + fee_rate), rel=1e-6)

    def test_rebalance_sells_before_buys(self, data_service, scenario):
        state = PortfolioState(cash=100_000.0)
        exec_date = date(2022, 2, 1)
        execute_target(state, {"SPY": 1.0}, date(2022, 1, 31), exec_date, data_service, scenario.total_fee_rate)
        assert "SPY" in state.positions

        result = execute_target(
            state, {"TLT": 0.5, "MSFT": 0.5}, date(2022, 2, 2), date(2022, 2, 3), data_service, scenario.total_fee_rate
        )
        actions = [t.action for t in result.trades]
        assert actions[0] == "sell"  # proceeds available before buying
        assert set(state.positions) == {"TLT", "MSFT"}
        assert result.turnover > 0

    def test_turnover_is_one_sided(self, data_service, scenario):
        state = PortfolioState(cash=100_000.0)
        exec_date = date(2022, 2, 1)
        result = execute_target(state, {"SPY": 0.5}, date(2022, 1, 31), exec_date, data_service, 0.0)
        expected_notional = 0.5 * 100_000.0
        assert result.turnover == pytest.approx(expected_notional / (2 * 100_000.0))

    def test_unknown_ticker_target_is_skipped(self, data_service, scenario):
        state = PortfolioState(cash=100_000.0)
        execute_target(state, {"SPY": 1.0}, date(2022, 1, 31), date(2022, 2, 1), data_service, 0.0)
        result = execute_target(state, {"ZZZZ": 0.5, "SPY": 0.5}, date(2022, 2, 2), date(2022, 2, 3), data_service, 0.0)
        traded = {t.ticker for t in result.trades}
        assert "ZZZZ" not in traded  # no data: no trade, no position
        assert "SPY" in traded
        assert "ZZZZ" not in state.positions

    def test_empty_target_liquidates(self, data_service, scenario):
        state = PortfolioState(cash=100_000.0)
        execute_target(state, {"SPY": 1.0}, date(2022, 1, 31), date(2022, 2, 1), data_service, 0.0)
        assert "SPY" in state.positions
        result = execute_target(state, {}, date(2022, 2, 2), date(2022, 2, 3), data_service, 0.0)
        assert state.positions == {}
        assert any(t.action == "sell" for t in result.trades)


class TestRunBacktest:
    @pytest.mark.parametrize("legacy", [False, True])
    def test_open_fills_precede_same_session_close_decisions(self, data_service, scenario, legacy):
        observed = {}

        def weights(day, state):
            observed[day] = dict(state.positions)
            return {"weights": {"SPY": 0.5}}

        asyncio.run(run_backtest(data_service, [date(2022, 1, 3), date(2022, 1, 4)], weights, scenario, legacy=legacy))
        assert bool(observed[date(2022, 1, 4)]) is not legacy

    def test_agent_style_async_weights(self, data_service, scenario):
        days = data_service.decision_dates(date(2022, 1, 3), date(2022, 3, 31), "monthly")
        assert len(days) == 3

        async def weight_fn(day, state):
            return {"weights": {"SPY": 1.0}}

        result = asyncio.run(run_backtest(data_service, days, weight_fn, scenario))
        assert result.equity.index[0] == days[0]
        assert result.equity.iloc[0] == pytest.approx(100_000.0)
        assert len(result.trades) >= 1
        assert all(t.exec_date > t.decision_date for t in result.trades)
        # every decision has exactly one turnover entry
        assert len(result.turnover) == len(days)

    def test_sync_weights_for_baselines(self, data_service, scenario):
        days = data_service.decision_dates(date(2022, 1, 3), date(2022, 3, 31), "monthly")

        def weight_fn(day, state):
            return {"weights": {"SPY": 0.6}}

        result = asyncio.run(run_backtest(data_service, days, weight_fn, scenario))
        assert len(result.equity) > 10
        assert result.equity.index[-1] == date.fromisoformat(scenario.end)

    def test_last_decision_without_next_day_is_recorded_unexecuted(self, data_service, scenario):
        last_day = data_service.trading_days()[-1]
        scenario = scenario.model_copy(update={"end": last_day.isoformat()})
        result = asyncio.run(run_backtest(data_service, [last_day], lambda d, s: {"weights": {"SPY": 1.0}}, scenario))
        assert result.trades == []
        assert result.turnover == [(last_day, 0.0)]
        assert result.equity.iloc[-1] == pytest.approx(100_000.0)  # stayed all cash


class TestMetrics:
    def test_total_return_and_drawdown(self):
        equity = pd.Series([100.0, 110.0, 99.0, 121.0])
        assert total_return(equity) == pytest.approx(0.21)
        assert max_drawdown(equity) == pytest.approx(99.0 / 110.0 - 1.0)

    def test_cagr_annualizes_by_252(self):
        equity = pd.Series([100.0] + [100.0] * 251 + [200.0])  # 252 days, 2x
        assert cagr(equity) == pytest.approx(1.0)  # doubling in one year

    def test_sharpe_matches_formula(self):
        returns = pd.Series([0.001, -0.002, 0.003, 0.001, -0.001] * 20)
        expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
        assert sharpe_ratio(returns) == pytest.approx(expected)

    def test_sharpe_with_risk_free(self):
        returns = pd.Series([0.0] * 10)
        assert sharpe_ratio(returns) == 0.0

    def test_annualized_volatility(self):
        returns = pd.Series([0.01, -0.01] * 50)
        expected = returns.std(ddof=1) * np.sqrt(252)
        assert annualized_volatility(returns) == pytest.approx(expected)

    def test_window_returns(self):
        equity = pd.Series({"d1": 100.0, "d2": 110.0, "d3": 99.0})
        windows = window_returns(equity, ["d1", "d2"])
        assert windows == [("d1", pytest.approx(0.1)), ("d2", pytest.approx(99.0 / 110.0 - 1))]

    def test_window_returns_survives_timestamp_index(self):
        # regression: pandas may convert date keys to Timestamps; hashing differs
        from datetime import date as D

        equity = pd.Series({D(2022, 1, 3): 100.0, D(2022, 1, 31): 110.0, D(2022, 2, 28): 121.0})
        windows = window_returns(equity, [D(2022, 1, 3), D(2022, 1, 31)])
        assert len(windows) == 2
        assert windows[0][1] == pytest.approx(0.1)

    def test_directional_accuracy(self):
        windows = [("d1", 0.02), ("d2", -0.03), ("d3", 0.001)]
        predicted = {"d1": "up", "d2": "up", "d3": "flat"}
        assert directional_accuracy(windows, predicted) == pytest.approx(2 / 3)

    def test_hit_rate(self):
        port = [("d1", 0.03), ("d2", -0.01)]
        bench = [("d1", 0.01), ("d2", 0.0)]
        assert outperformance_hit_rate(port, bench) == pytest.approx(0.5)


class TestPortfolioState:
    def test_weights_and_equity(self, data_service):
        day = date(2022, 2, 1)
        spy_close = data_service.close_on("SPY", day)
        state = PortfolioState(cash=500.0, positions={"SPY": 10.0})
        assert state.equity(day, data_service) == pytest.approx(500.0 + 10.0 * spy_close)
        assert state.weights(day, data_service)["SPY"] == pytest.approx(10.0 * spy_close / (500 + 10 * spy_close))


def test_forbidden_benchmark_order_is_rejected_before_execution(data_service, scenario):
    state = PortfolioState(cash=100_000)
    with pytest.raises(ValueError, match="non-tradable"):
        execute_target(
            state,
            {"SPY": 0.3, "MSFT": 0.3},
            date(2022, 1, 31),
            date(2022, 2, 1),
            data_service,
            scenario.total_fee_rate,
            allowed_tickers=scenario.tradable,
        )
    assert state.positions == {} and state.cash == 100_000


def test_backtest_rejects_benchmark_allocation_that_bypasses_pipeline(data_service, scenario):
    with pytest.raises(ValueError, match="non-tradable"):
        asyncio.run(
            run_backtest(
                data_service,
                [date(2022, 1, 3)],
                lambda day, state: {"weights": {"SPY": 0.3}},
                scenario,
                allowed_tickers=scenario.tradable,
            )
        )
