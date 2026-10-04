"""Portfolio accounting shared by agent runs and non-AI baselines.

Execution rules (identical for every participant):
- Decisions are made on decision-date closes; orders fill at the next trading
  day's open.
- fee_bps + slippage_bps are charged on traded notional each way.
- Fractional shares, no leverage, remainder weight sits in cash.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from ..data.service import DataService
from ..schemas import Scenario

log = logging.getLogger(__name__)


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)  # ticker -> shares

    def price_on(self, ticker: str, day: date, data: DataService) -> float:
        close = data.close_on(ticker, day)
        if close is not None:
            return close
        found = data.last_close(ticker, day)
        return found[1] if found else 0.0

    def equity(self, day: date, data: DataService) -> float:
        total = self.cash
        for ticker, shares in self.positions.items():
            total += shares * self.price_on(ticker, day, data)
        return total

    def weights(self, day: date, data: DataService) -> dict[str, float]:
        equity = self.equity(day, data)
        if equity <= 0:
            return {}
        return {ticker: shares * self.price_on(ticker, day, data) / equity for ticker, shares in self.positions.items()}


@dataclass
class Trade:
    decision_date: date
    exec_date: date
    ticker: str
    action: str
    shares: float
    price: float
    notional: float
    fee: float

    def as_row(self) -> dict:
        return {
            "decision_date": self.decision_date.isoformat(),
            "exec_date": self.exec_date.isoformat(),
            "ticker": self.ticker,
            "action": self.action,
            "shares": round(self.shares, 6),
            "price": round(self.price, 4),
            "notional": round(self.notional, 2),
            "fee": round(self.fee, 2),
        }


@dataclass
class StepResult:
    trades: list[Trade] = field(default_factory=list)
    turnover: float = 0.0


def execute_target(
    state: PortfolioState,
    target_weights: dict[str, float],
    decision_date: date,
    exec_date: date,
    data: DataService,
    fee_rate: float,
) -> StepResult:
    """Rebalance `state` to target weights, filling at exec_date's open."""
    equity = state.equity(decision_date, data)
    if equity <= 0:
        return StepResult()

    targets: dict[str, float] = {}
    for ticker, weight in target_weights.items():
        price = data.open_on(ticker, exec_date)
        if not _valid_price(price):
            log.warning("no open price for %s on %s; keeping current position", ticker, exec_date)
            continue
        # Reserve the fee so a full-allocation buy never drives cash negative.
        targets[ticker] = (weight * equity) / (price * (1.0 + fee_rate))

    trades: list[Trade] = []
    traded_notional = 0.0
    # Sell before buying so proceeds are available; within a side, alphabetical.
    tickers = sorted(
        set(state.positions) | set(targets),
        key=lambda t: ((targets.get(t, 0.0) - state.positions.get(t, 0.0)) >= 0, t),
    )
    for ticker in tickers:
        price = data.open_on(ticker, exec_date)
        if not _valid_price(price):
            continue
        current = state.positions.get(ticker, 0.0)
        delta = targets.get(ticker, 0.0) - current
        if abs(delta * price) < 0.01:  # dust
            continue
        notional = delta * price
        fee = abs(notional) * fee_rate
        state.cash -= notional + fee
        state.positions[ticker] = current + delta
        if abs(state.positions[ticker]) < 1e-9:
            del state.positions[ticker]
        traded_notional += abs(notional)
        trades.append(
            Trade(
                decision_date=decision_date,
                exec_date=exec_date,
                ticker=ticker,
                action="buy" if delta > 0 else "sell",
                shares=abs(delta),
                price=price,
                notional=abs(notional),
                fee=fee,
            )
        )
    return StepResult(trades=trades, turnover=traded_notional / (2.0 * equity))


def _valid_price(price: float | None) -> bool:
    # NaN fails every comparison, so guard it explicitly
    return price is not None and math.isfinite(price) and price > 0


@dataclass
class BacktestResult:
    equity: pd.Series  # indexed by date
    trades: list[Trade] = field(default_factory=list)
    turnover: list[tuple[date, float]] = field(default_factory=list)
    decisions: list[date] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


WeightFn = object  # (decision_date, state) -> dict | awaitable[dict]


async def run_backtest(
    data: DataService,
    decision_dates: list[date],
    weight_fn: WeightFn,
    scenario: Scenario,
) -> BacktestResult:
    """Drive a portfolio through decision dates with a weight function.

    weight_fn(decision_date, state) returns {"weights": {ticker: weight}, ...};
    it may be async (the agent pipeline) or plain (baselines). Unexecuted final
    decisions (no next trading day inside the data) are recorded with zero
    turnover and no trades.
    """
    if not decision_dates:
        raise ValueError("no decision dates provided")
    all_days = data.trading_days()
    cutoff = min(all_days[-1], date.fromisoformat(scenario.end))
    days = [d for d in all_days if decision_dates[0] <= d <= cutoff]
    if not days or any(d < days[0] or d > cutoff for d in decision_dates):
        raise ValueError("decision dates outside evaluation window")
    last_day = days[-1] if days else decision_dates[0]

    state = PortfolioState(cash=scenario.initial_capital)
    equity: dict[date, float] = {}
    trades: list[Trade] = []
    turnover: list[tuple[date, float]] = []
    violations: list[str] = []
    pending: dict[date, tuple[date, dict]] = {}  # exec_date -> (decision_date, weights)

    idx = 0
    for day in days:
        if idx < len(decision_dates) and day == decision_dates[idx]:
            out = weight_fn(day, state)
            if asyncio.iscoroutine(out):
                out = await out
            violations.extend(out.get("violations", []))
            exec_date = data.next_trading_day(day)
            if exec_date is not None and exec_date <= last_day:
                pending[exec_date] = (day, dict(out.get("weights", {})))
            else:
                turnover.append((day, 0.0))
            idx += 1
        if day in pending:
            decision_day, weights = pending.pop(day)
            step = execute_target(state, weights, decision_day, day, data, scenario.total_fee_rate)
            trades.extend(step.trades)
            turnover.append((decision_day, step.turnover))
        equity[day] = state.equity(day, data)

    return BacktestResult(
        equity=pd.Series(equity).sort_index(),
        trades=trades,
        turnover=turnover,
        decisions=decision_dates,
        violations=violations,
    )


# --------------------------------------------------------------------------- #
# Non-AI baselines (same accounting engine, deterministic weight functions)
# --------------------------------------------------------------------------- #


def equal_weight_fn(scenario: Scenario):
    def fn(day: date, state: PortfolioState) -> dict:
        tickers = [t for t in scenario.tradable if t != scenario.benchmark] or [scenario.benchmark]
        weight = 1.0 / len(tickers)
        return {"weights": {t: weight for t in tickers}}

    return fn


def sixty_forty_weight_fn(scenario: Scenario):
    def fn(day: date, state: PortfolioState) -> dict:
        return {"weights": {scenario.benchmark: 0.6}}

    return fn


def momentum_weight_fn(scenario: Scenario, data: DataService, top: int = 3):
    """12-1 momentum: rank by return from ~252 trading days ago to ~21 days ago."""

    def fn(day: date, state: PortfolioState) -> dict:
        scored = []
        for ticker in scenario.tradable:
            if ticker == scenario.benchmark:
                continue
            closes = data.closes(ticker, day, 253)
            if len(closes) >= 200:
                scored.append((ticker, closes[-22] / closes[0] - 1.0))
        if not scored:
            return {"weights": {scenario.benchmark: 1.0}}
        picked = sorted(scored, key=lambda kv: kv[1], reverse=True)[:top]
        weight = 1.0 / len(picked)
        return {"weights": {t: weight for t, _ in picked}}

    return fn
