"""Build the point-in-time state a decision model reads.

Two rules govern everything here:

1. **No leakage.** Only data dated on or before the decision date enters the
   state. `DataService` is the single source, and it never serves a bar later
   than `as_of`.
2. **Compact.** Workers AI truncates long text state to roughly its first 2K
   tokens, so a state that overflows would be silently cut in half for Clef and
   read in full by the other providers. The state is therefore budgeted to stay
   inside `DecisionConfig.max_state_tokens` (~1,800) for everyone, which keeps
   the comparison like-for-like and the failure mode impossible.

The state says nothing about what to buy. Allocation rules belong in the
question set and the sizing policy, not in a preamble that could leak a
reference allocation into the benchmark.
"""

from __future__ import annotations

import json
from datetime import date

BENCHMARK_NOTE = (
    "All figures are computed from prices dated on or before as_of. "
    "The benchmark column is the comparison index itself; it is not investable."
)


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


def _position(weights: dict[str, float], cash: float) -> dict:
    rows = [
        {"ticker": ticker, "weight_pct": _round(weight * 100.0, 1)}
        for ticker, weight in sorted(weights.items())
        if weight > 0
    ]
    return {"cash_weight_pct": _round(cash * 100.0, 1), "holdings": rows}


def _ticker_row(data, ticker: str, as_of: date, benchmark_return: float | None) -> dict | None:
    window = data.window(ticker, as_of, 5)
    if window is None or window.empty:
        return None
    close = window["close"]
    last = float(close.iloc[-1])
    prior = float(close.iloc[0]) if len(close) > 1 else last
    row = {
        "ticker": ticker.upper(),
        "last_close": _round(last),
        "last_5d_pct": _round((last / prior - 1.0) * 100.0) if prior else None,
        "return_30d_pct": data.trailing_return_pct(ticker, as_of, 21),
        "return_90d_pct": data.trailing_return_pct(ticker, as_of, 63),
        "return_90d_vs_benchmark_pct": None,
    }
    own = row["return_90d_pct"]
    if own is not None and benchmark_return is not None:
        row["return_90d_vs_benchmark_pct"] = _round(own - benchmark_return)
    return row


def benchmark_row(data, benchmark: str, as_of: date) -> dict:
    return {
        "ticker": benchmark.upper(),
        "return_30d_pct": data.trailing_return_pct(benchmark, as_of, 21),
        "return_90d_pct": data.trailing_return_pct(benchmark, as_of, 63),
    }


def candidates(data, scenario, as_of: date, *,
               extra: list[str] | None = None,
               max_candidates: int | None = None) -> list[str]:
    """The tickers a decision asks about: the strongest trailing 90-day movers.

    Deterministic and point-in-time — ranked on data available at `as_of`, ties
    broken alphabetically — so the candidate set is reproducible and never
    depends on how a previous decision went. Holdings are always included so the
    model can act on what it owns.
    """
    limit = max_candidates or scenario.max_candidates
    ranked = []
    for ticker in scenario.tradable:
        trailing = data.trailing_return_pct(ticker, as_of, 63)
        if trailing is None:
            continue
        ranked.append((abs(trailing), ticker.upper()))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    chosen = [ticker for _, ticker in ranked[:limit]]
    for ticker in extra or []:
        name = ticker.upper()
        if name in scenario.tradable and name not in chosen:
            chosen.append(name)
    return chosen


def build_state(
    data,
    scenario,
    as_of: date,
    *,
    holdings: dict[str, float],
    cash: float,
    question_tickers: list[str],
    horizon_trading_days: int,
    next_decision_date: str | None,
    recent: list[dict] | None = None,
) -> dict:
    """Assemble the state dict for one decision. Deterministic for a given input."""
    benchmark = scenario.benchmark.upper()
    bench = benchmark_row(data, benchmark, as_of)
    universe = [
        row
        for row in (
            _ticker_row(data, ticker, as_of, bench.get("return_90d_pct")) for ticker in scenario.universe
        )
        if row is not None
    ]
    return {
        "as_of": as_of.isoformat(),
        "scenario": scenario.name,
        "horizon_trading_days": horizon_trading_days,
        "next_decision_date": next_decision_date,
        "benchmark": bench,
        "costs": {
            "fee_bps": scenario.fee_bps,
            "slippage_bps": scenario.slippage_bps,
            "max_position_weight_pct": _round(scenario.max_position_weight * 100.0, 1),
        },
        "portfolio": _position(holdings, cash),
        "market": universe,
        "decide_on": [ticker.upper() for ticker in question_tickers],
        "constraints": {
            "long_only": True,
            "notes": BENCHMARK_NOTE,
        },
        "recent_decisions": recent or [],
    }


def render_state(state: dict) -> str:
    """Serialize the state for the wire: compact, stable, and reproducible."""
    return json.dumps(state, separators=(",", ":"), sort_keys=True)


def state_token_estimate(text: str) -> int:
    """Cheap upper bound on tokens; ~3 chars/token is conservative for JSON numbers."""
    return (len(text) + 2) // 3