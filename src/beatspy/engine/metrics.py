"""Deterministic metric computation. There is no LLM judge anywhere in BeatSPY.

Conventions:
- 252 trading days per year; sample standard deviation (ddof=1).
- Sharpe uses annualized excess returns over the configured risk-free rate.
- Directional accuracy scores the Portfolio Manager's stated direction against
  the realized portfolio return sign (flat counts as correct within a 0.5%
  band), keeping forecasting accuracy separate from profitability.
- Outperformance hit rate is the share of decision windows where the portfolio
  beat SPY.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd

TRADING_DAYS = 252
FLAT_BAND = 0.005


def total_return(equity) -> float:
    if len(equity) < 2:
        return 0.0
    first = float(equity.iloc[0])
    last = float(equity.iloc[-1])
    if first <= 0:
        return 0.0
    return last / first - 1.0


def cagr(equity) -> float:
    periods = len(equity) - 1
    growth = 1.0 + total_return(equity)
    if periods <= 0:
        return 0.0
    if growth <= 0:
        return -1.0
    return growth ** (TRADING_DAYS / periods) - 1.0


def annualized_volatility(daily_returns) -> float:
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std(ddof=1) * math.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_returns, risk_free_annual: float = 0.0) -> float:
    if len(daily_returns) < 3:
        return 0.0
    excess = daily_returns - risk_free_annual / TRADING_DAYS
    std = float(excess.std(ddof=1))
    if std == 0:
        return 0.0
    return float(excess.mean() / std * math.sqrt(TRADING_DAYS))


def max_drawdown(equity) -> float:
    if len(equity) < 2:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _as_date_key(key):
    """Normalize index keys so lookups hash consistently across pandas index types."""
    if isinstance(key, pd.Timestamp):
        return key.date()
    if isinstance(key, dt.datetime):
        return key.date()
    return key  # datetime.date or test-only string keys pass through


def window_returns(equity, decision_dates) -> list[tuple]:
    """Return [(start_date, window_return)] for each decision window.

    Keys are normalized on both sides: pandas may hand back a DatetimeIndex
    (Timestamp keys) or an object index (date keys), and hash() differs between
    the two even though == holds.
    """
    values = {_as_date_key(ts): float(v) for ts, v in equity.items()}
    dates = sorted(values)
    starts = [_as_date_key(d) for d in decision_dates]
    out = []
    for i, start in enumerate(starts):
        if start not in values:
            continue
        end = starts[i + 1] if i + 1 < len(starts) else dates[-1]
        if end not in values or end <= start:
            continue
        out.append((start, values[end] / values[start] - 1.0))
    return out


def directional_accuracy(windows: list[tuple], predicted: dict) -> float:
    """Fraction of decisions where the stated direction matched the realized sign."""
    scored = 0
    correct = 0
    for start, realized in windows:
        prediction = predicted.get(start)
        if prediction is None:
            continue
        scored += 1
        if (
            prediction == "up"
            and realized > 0
            or prediction == "down"
            and realized < 0
            or prediction == "flat"
            and abs(realized) <= FLAT_BAND
        ):
            correct += 1
    return correct / scored if scored else 0.0


def outperformance_hit_rate(portfolio_windows: list[tuple], benchmark_windows: list[tuple]) -> float:
    bench = dict(benchmark_windows)
    pairs = [(p, bench.get(d)) for d, p in portfolio_windows if bench.get(d) is not None]
    if not pairs:
        return 0.0
    return sum(1 for p, b in pairs if p > b) / len(pairs)


def annualized_return_from_cagr(cagr_value: float) -> float:
    """Kept for clarity in reports; CAGR already is annualized."""
    return cagr_value
