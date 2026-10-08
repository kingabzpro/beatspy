"""Deterministic scoring for a DCN run: portfolio metrics plus calibration.

Half of what this benchmark measures is not a return at all. The decision model
returns probabilities, so the run is scored twice:

- **Portfolio** — the usual accounting engine, against SPY and three deterministic
  references on the same frozen prices.
- **Calibration** — Brier, log loss, AUC, and reliability bins over every
  probability the model produced, labeled by what actually happened.
"""

from __future__ import annotations

import statistics
from datetime import date

import pandas as pd

from ..engine.backtest import equal_weight_fn, momentum_weight_fn, run_backtest, sixty_forty_weight_fn
from ..engine.metrics import (
    annualized_volatility,
    cagr,
    max_drawdown,
    outperformance_hit_rate,
    sharpe_ratio,
    total_return,
    window_returns,
)
from ..schemas import RunMetrics
from .calibration import build_report

BASELINES = ("equal_weight", "sixty_forty", "momentum_12_1")


def estimate_cost(settings, input_tokens: int, output_tokens: int) -> float | None:
    """Decision models bill input only, but an output rate is honoured if set."""
    decision = settings.decision
    if decision.cost_per_m_input is None and decision.cost_per_m_output is None:
        return None
    total = input_tokens * (decision.cost_per_m_input or 0.0) + output_tokens * (decision.cost_per_m_output or 0.0)
    return round(total / 1e6, 6)


async def score_run(data, scenario, settings, result, decision_dates, decisions, observations):
    """Score one DCN run. `decisions` are DcnDecision records; no model calls here."""
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

    baseline_equity = {}
    for name, fn in (
        ("equal_weight", equal_weight_fn(scenario)),
        ("sixty_forty", sixty_forty_weight_fn(scenario)),
        ("momentum_12_1", momentum_weight_fn(scenario, data)),
    ):
        baseline_equity[name] = (await run_backtest(data, decision_dates, fn, scenario)).equity

    daily_returns = result.equity.pct_change().dropna()
    p_windows = window_returns(result.equity, decision_dates)
    b_windows = window_returns(spy_equity, decision_dates)

    calls = [decision.call for decision in decisions if decision.call is not None]
    input_tokens = sum(call.input_tokens for call in calls)
    output_tokens = sum(call.output_tokens for call in calls)
    requests = sum(call.requests for call in calls)
    latencies = [call.latency_s for call in calls]
    invalid_outputs = sum(1 for decision in decisions if decision.error or decision.validated.invalid)
    risk_violations = sum(len(decision.validated.violations) for decision in decisions)
    coverages = [decision.coverage for decision in decisions]
    turnover_values = [value for _, value in result.turnover]

    calibration = build_report(
        list(observations),
        coverage=statistics.mean(coverages) if coverages else 0.0,
        horizons=len(decisions),
        note="labels are benchmark-relative: did the ticker beat the benchmark over the same horizon",
    )

    metrics = RunMetrics(
        total_return=round(total_return(result.equity), 6),
        cagr=round(cagr(result.equity), 6),
        annual_volatility=round(annualized_volatility(daily_returns), 6),
        sharpe=round(sharpe_ratio(daily_returns, settings.bench.risk_free_annual), 4),
        max_drawdown=round(max_drawdown(result.equity), 6),
        spy_total_return=round(total_return(spy_equity), 6),
        excess_return_vs_spy=round(total_return(result.equity) - total_return(spy_equity), 6),
        outperformance_hit_rate=round(outperformance_hit_rate(p_windows, b_windows), 4),
        decisions=len(decisions),
        avg_turnover=round(statistics.mean(turnover_values), 6) if turnover_values else 0.0,
        trades=len(result.trades),
        invalid_outputs=invalid_outputs,
        risk_violations=risk_violations,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        requests=requests,
        calibration=calibration,
        answer_coverage=round(statistics.mean(coverages), 4) if coverages else 0.0,
        mean_latency_s=round(statistics.mean(latencies), 3) if latencies else 0.0,
        estimated_cost_usd=estimate_cost(settings, input_tokens, output_tokens),
        baselines={name: round(total_return(equity), 6) for name, equity in baseline_equity.items()},
    )

    equity_df = pd.DataFrame(
        {
            "portfolio": result.equity,
            "spy_buy_hold": spy_equity,
            "equal_weight": baseline_equity["equal_weight"],
            "sixty_forty": baseline_equity["sixty_forty"],
            "momentum_12_1": baseline_equity["momentum_12_1"],
        }
    )
    return metrics, equity_df