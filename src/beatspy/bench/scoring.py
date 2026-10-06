"""Deterministic scoring shared by execution and submission replay."""

from __future__ import annotations

import statistics
from datetime import date

import pandas as pd

from ..engine.backtest import (
    equal_weight_fn,
    large_company_core_fn,
    momentum_weight_fn,
    run_backtest,
    sixty_forty_weight_fn,
)
from ..engine.metrics import (
    annualized_volatility,
    cagr,
    directional_accuracy,
    max_drawdown,
    outperformance_hit_rate,
    sharpe_ratio,
    total_return,
    window_returns,
)
from ..schemas import RunMetrics


def estimate_cost(settings, input_tokens, output_tokens):
    m = settings.model
    if m.cost_per_m_input is None and m.cost_per_m_output is None:
        return None
    return round((input_tokens * (m.cost_per_m_input or 0) + output_tokens * (m.cost_per_m_output or 0)) / 1e6, 4)


async def score_run(
    data, scenario, settings, result, decision_dates, decisions, harness_violations, *, legacy=False, protocol_version=5
):
    # SPY buy-and-hold from the same start value, through the same price data.
    spy_values: dict[date, float] = {}
    base_close = None
    for day in result.equity.index:
        close = data.close_on(scenario.benchmark, day)
        if close is None:
            continue
        if base_close is None:
            base_close = close
        spy_values[day] = scenario.initial_capital * close / base_close
    spy_equity = pd.Series(spy_values).sort_index()

    baselines = {}
    for name, fn in [
        ("equal_weight", equal_weight_fn(scenario)),
        ("sixty_forty", sixty_forty_weight_fn(scenario)),
        ("momentum_12_1", momentum_weight_fn(scenario, data)),
    ]:
        baselines[name] = (await run_backtest(data, decision_dates, fn, scenario, legacy=legacy)).equity
    # Protocol 8 historically implied the fixed stock core; protocol 9 lets the
    # scenario declare it, so archived runs keep scoring the baseline they always did.
    if protocol_version >= 8 or scenario.reference_kind == "core":
        baselines["large_company_core"] = (
            await run_backtest(
                data, decision_dates, large_company_core_fn(scenario), scenario, allowed_tickers=scenario.tradable
            )
        ).equity
    # Non-tradable assets scored alongside SPY: buy-and-hold from the same start
    # value through the same frozen prices. This makes a hedge sleeve's own market
    # return a first-class, replayable number instead of a claim in a report.
    asset_curves: dict[str, pd.Series] = {}
    for ticker in scenario.score_assets:
        values: dict[date, float] = {}
        base = None
        for day in result.equity.index:
            close = data.close_on(ticker, day)
            if close is None:
                continue
            if base is None:
                base = close
            values[day] = scenario.initial_capital * close / base
        if values:
            asset_curves[ticker] = pd.Series(values).sort_index()

    daily_returns = result.equity.pct_change().dropna()
    p_windows = window_returns(result.equity, decision_dates)
    b_windows = window_returns(spy_equity, decision_dates)
    predicted = {rec.date: rec.predicted_direction for rec in decisions}
    input_tokens = sum(o.input_tokens for rec in decisions for o in rec.outcomes.values())
    output_tokens = sum(o.output_tokens for rec in decisions for o in rec.outcomes.values())
    requests = sum(o.requests for rec in decisions for o in rec.outcomes.values())
    invalid_outputs = sum(1 for rec in decisions if rec.validated.invalid)
    risk_violations = sum(len(rec.validated.violations) for rec in decisions) + len(harness_violations)
    turnover_values = [t for _, t in result.turnover]

    metrics = RunMetrics(
        total_return=round(total_return(result.equity), 6),
        cagr=round(cagr(result.equity), 6),
        annual_volatility=round(annualized_volatility(daily_returns), 6),
        sharpe=round(sharpe_ratio(daily_returns, settings.bench.risk_free_annual), 4),
        max_drawdown=round(max_drawdown(result.equity), 6),
        spy_total_return=round(total_return(spy_equity), 6),
        excess_return_vs_spy=round(total_return(result.equity) - total_return(spy_equity), 6),
        outperformance_hit_rate=round(outperformance_hit_rate(p_windows, b_windows), 4),
        directional_accuracy=round(directional_accuracy(p_windows, predicted), 4),
        decisions=len(decisions),
        avg_turnover=round(statistics.mean(turnover_values), 6) if turnover_values else 0.0,
        trades=len(result.trades),
        invalid_outputs=invalid_outputs,
        risk_violations=risk_violations,
        tool_calls=sum(rec.totals()["tool_calls"] for rec in decisions),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        requests=requests,
        estimated_cost_usd=estimate_cost(settings, input_tokens, output_tokens),
        baselines={name: round(total_return(eq), 6) for name, eq in baselines.items()}
        | {ticker: round(total_return(eq), 6) for ticker, eq in asset_curves.items()},
    )

    equity_df = pd.DataFrame(
        {
            "portfolio": result.equity,
            "spy_buy_hold": spy_equity,
            "equal_weight": baselines["equal_weight"],
            "sixty_forty": baselines["sixty_forty"],
            "momentum_12_1": baselines["momentum_12_1"],
        }
    )
    if protocol_version >= 8 or scenario.reference_kind == "core":
        equity_df["large_company_core"] = baselines["large_company_core"]
    for ticker, curve in asset_curves.items():
        equity_df[ticker] = curve
    return metrics, equity_df
