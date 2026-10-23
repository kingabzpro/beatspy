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

# Sector labels for the scenario pools, so a model can see what a name is, not
# only what it did. Short strings keep the state inside its token budget; a
# ticker without an entry simply carries no sector field.
SECTORS = {
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "GOOGL": "Tech", "META": "Tech",
    "AMZN": "Consumer Disc.", "HD": "Consumer Disc.",
    "WMT": "Consumer Staples", "PG": "Consumer Staples", "COST": "Consumer Staples",
    "JPM": "Financials", "V": "Financials",
    "LLY": "Healthcare", "JNJ": "Healthcare", "UNH": "Healthcare",
    "XOM": "Energy", "CVX": "Energy",
    "CAT": "Industrials", "GE": "Industrials",
    "NEE": "Utilities", "LIN": "Materials",
    "PLD": "Real Estate", "AMT": "Real Estate",
    "QQQ": "Broad equity ETF", "TLT": "Long Treasury ETF",
}


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


def _position(weights: dict[str, float], cash: float) -> dict:
    rows = [
        {"ticker": ticker, "weight_pct": _round(weight * 100.0, 1)}
        for ticker, weight in sorted(weights.items())
        if weight > 0
    ]
    return {"cash_weight_pct": _round(cash * 100.0, 1), "holdings": rows}


def _volume_ratio(data, ticker: str, as_of: date) -> float | None:
    """Last 10 sessions of volume against the 50 before them; ~1 is normal attention."""
    sub = data.window(ticker, as_of, 60)
    if sub is None or len(sub) < 60 or "volume" not in sub:
        return None
    recent = float(sub["volume"].tail(10).mean())
    prior = float(sub["volume"].iloc[:-10].mean())
    return _round(recent / prior, 2) if prior else None


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
    # Protocol 6: risk and context, so the model has more than momentum to read.
    # RSI adds reversal pressure, realized volatility adds risk, the 52-week-high
    # distance adds level, the volume ratio adds attention, the sector adds what
    # the name is. Fields that cannot be computed point-in-time are omitted.
    indicators = data.indicators(ticker, as_of) or {}
    extras = {
        "rsi_14": indicators.get("rsi_14"),
        "annualized_vol_30d_pct": indicators.get("annualized_vol_30d_pct"),
        "off_52w_high_pct": indicators.get("off_52w_high_pct"),
        "volume_ratio_10_50": _volume_ratio(data, ticker, as_of),
        "sector": SECTORS.get(row["ticker"]),
    }
    row.update({key: value for key, value in extras.items() if value is not None})
    return row


def benchmark_row(data, benchmark: str, as_of: date) -> dict:
    return {
        "ticker": benchmark.upper(),
        "return_30d_pct": data.trailing_return_pct(benchmark, as_of, 21),
        "return_90d_pct": data.trailing_return_pct(benchmark, as_of, 63),
    }


def candidates(data, scenario, as_of: date, *,
               extra: list[str] | None = None,
               max_candidates: int | None = None,
               pool: list[str] | None = None) -> list[str]:
    """The tickers a decision asks about: the strongest trailing 90-day movers.

    Deterministic and point-in-time (ranked on data available at `as_of`, ties
    broken alphabetically), so the candidate set is reproducible and never
    depends on how a previous decision went. Holdings come first, so the model
    can always act on what it owns, then the strongest movers up to the limit.
    """
    limit = max_candidates or scenario.max_candidates
    allowed = [ticker.upper() for ticker in (pool if pool is not None else scenario.tradable)]
    held = [ticker.upper() for ticker in (extra or []) if ticker.upper() in allowed]
    ranked = []
    for ticker in allowed:
        trailing = data.trailing_return_pct(ticker, as_of, 63)
        if trailing is None:
            continue
        ranked.append((abs(trailing), ticker))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    chosen = list(dict.fromkeys(held))
    for _, ticker in ranked:
        if len(chosen) >= limit:
            break
        if ticker not in chosen:
            chosen.append(ticker)
    return chosen[: max(limit, len(held))]


def select_universe(data, scenario, as_of: date) -> list[str]:
    """The investable set at `as_of`: the pool, or its strongest trailing performers.

    A scenario may declare `selection_top`: each decision then trades only that
    many names, ranked by their trailing return versus the benchmark over
    `selection_lookback_days`. The ranking reads closes dated on or before
    `as_of` through `DataService`, so a name qualifies on what it already did,
    never on what happens next. Ties break alphabetically, so the set is
    reproducible.

    With no rule declared the whole pool is investable, which is what the other
    scenarios do.
    """
    pool = [ticker.upper() for ticker in scenario.tradable]
    top = getattr(scenario, "selection_top", None)
    if not top or top >= len(pool):
        return pool
    lookback = int(getattr(scenario, "selection_lookback_days", 126))
    benchmark_return = data.trailing_return_pct(scenario.benchmark, as_of, lookback) or 0.0
    ranked = []
    for ticker in pool:
        trailing = data.trailing_return_pct(ticker, as_of, lookback)
        if trailing is None:
            continue
        ranked.append((-(trailing - benchmark_return), ticker))
    ranked.sort()
    return sorted(ticker for _, ticker in ranked[:top])


def base_rate_row(universe: list[dict]) -> dict:
    """The pool's realized hit rate against the benchmark, as a calibration anchor.

    A probability is only meaningful next to the frequency it is trying to
    estimate. If a third of the pool beat the benchmark over the last quarter,
    then 0.5 is an optimistic answer for a typical name, and the model should be
    able to see that. Everything here is computed from the same point-in-time
    rows the state already carries, so it leaks nothing.
    """
    excesses = [
        row["return_90d_vs_benchmark_pct"]
        for row in universe
        if row.get("return_90d_vs_benchmark_pct") is not None
    ]
    if not excesses:
        return {"window_trading_days": 63, "share_beat_benchmark_pct": None, "median_excess_pct": None}
    ordered = sorted(excesses)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "window_trading_days": 63,
        "share_beat_benchmark_pct": _round(100.0 * sum(1 for value in excesses if value > 0) / len(excesses), 1),
        "median_excess_pct": _round(median, 2),
    }


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
        # What the pool has actually been doing, so a probability has something
        # to be calibrated against.
        "base_rate": base_rate_row(universe),
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


def enforce_budget(state: dict, max_tokens: int) -> tuple[dict, list[str]]:
    """Trim the state until it fits ``max_tokens``, deterministically.

    Workers AI truncates long text state to roughly its first 2K tokens, which
    would silently give Clef a different question from the one every other
    provider reads. Rather than let that happen, the least load-bearing content
    is dropped in a fixed order:

    1. the recent-decision history, oldest entry first;
    2. market rows the decision is not asking about.

    If the state still does not fit, this raises instead of sending a state one
    provider would truncate. Returns the (possibly trimmed) state and the list
    of what was dropped, for the record.
    """
    state = {
        **state,
        "recent_decisions": list(state.get("recent_decisions") or []),
        "market": list(state.get("market") or []),
    }
    trimmed: list[str] = []
    if state_token_estimate(render_state(state)) <= max_tokens:
        return state, trimmed
    while state["recent_decisions"] and state_token_estimate(render_state(state)) > max_tokens:
        state["recent_decisions"].pop(0)
        if "recent_decisions" not in trimmed:
            trimmed.append("recent_decisions")
    asked = {str(ticker).upper() for ticker in state.get("decide_on") or []}
    kept = [row for row in state["market"] if str(row.get("ticker", "")).upper() in asked]
    if len(kept) != len(state["market"]):
        trimmed.append(f"market:{len(state['market']) - len(kept)}")
        state["market"] = kept
    estimate = state_token_estimate(render_state(state))
    if estimate > max_tokens:
        raise ValueError(
            f"state needs about {estimate} tokens after trimming, over the {max_tokens}-token budget; "
            "shrink the universe or raise max_state_tokens"
        )
    return state, trimmed