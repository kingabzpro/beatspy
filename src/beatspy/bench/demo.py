"""Synthetic demo runs, clearly labeled SYNTHETIC.

Lets anyone evaluate the dashboard, compare, and report commands without a
model, an API key, or even internet access. Demo runs carry synthetic=true and
are excluded from `compare` by default.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..config import results_dir
from .runner import DISCLAIMER, write_run_artifacts

log = logging.getLogger(__name__)

DEMO_MODELS = [
    # (name, daily edge over SPY, daily extra vol)
    ("demo-alpha", 0.0008, 0.004),
    ("demo-beta", -0.0004, 0.006),
    ("demo-gamma", 0.0000, 0.010),
]
TICKERS = ["SPY", "AAPL", "MSFT", "TLT", "GLD"]
ROLES = ["research", "analyst", "forecaster", "critic", "portfolio_manager"]
DIRECTIONS = ["up", "down", "flat"]


def _decision_dates(days: list) -> list:
    out = [days[0]]
    month_ends: dict[tuple, object] = {}
    for day in days:
        month_ends[(day.year, day.month)] = day
    first_month = (days[0].year, days[0].month)
    out += [day for key, day in month_ends.items() if key != first_month]
    return out


def generate_demo_runs(seed: int = 42, out_root: Path | None = None) -> list[Path]:
    rng = np.random.default_rng(seed)
    days = [d.date() for d in pd.bdate_range("2022-01-03", "2022-06-30")]
    decisions_dates = _decision_dates(days)

    spy_logret = rng.normal(-0.0013, 0.012, size=len(days))

    out_root = out_root or results_dir()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    created = datetime.now(UTC).isoformat(timespec="seconds")
    run_dirs: list[Path] = []

    for name, edge, extra_vol in DEMO_MODELS:
        port_logret = spy_logret + rng.normal(edge, extra_vol, size=len(days))
        equity = 100_000.0 * np.exp(np.cumsum(port_logret))
        spy_line = 100_000.0 * np.exp(np.cumsum(spy_logret))

        equity_df = pd.DataFrame(
            {
                "portfolio": equity,
                "spy_buy_hold": spy_line,
                "equal_weight": 100_000.0 * np.exp(np.cumsum(rng.normal(-0.0008, 0.013, len(days)))),
                "sixty_forty": 100_000.0 * np.exp(np.cumsum(rng.normal(-0.0006, 0.008, len(days)))),
                "momentum_12_1": 100_000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, len(days)))),
            },
            index=pd.Index(days, name="date"),
        )

        decision_lines = []
        event_lines = []
        trade_rows = []
        predicted = {}
        for day in decisions_dates:
            count = int(rng.integers(2, 5))
            picks = list(rng.choice(TICKERS, size=count, replace=False))
            raw = rng.dirichlet(np.ones(count)) * float(rng.uniform(0.6, 1.0))
            weights = {t: round(min(w, 0.35), 4) for t, w in zip(picks, raw, strict=True)}
            direction = str(rng.choice(DIRECTIONS))
            predicted[day.isoformat()] = direction
            decision_lines.append(
                {
                    "date": day.isoformat(),
                    "decision": {
                        "allocations": [{"ticker": t, "weight": w} for t, w in weights.items()],
                        "expected_direction": direction,
                        "rationale": "Synthetic demo decision; no model was run.",
                    },
                    "validated": {
                        "weights": weights,
                        "cash": round(1 - sum(weights.values()), 4),
                        "violations": [],
                        "invalid": False,
                    },
                    "predicted_direction": direction,
                    "rationale": "Synthetic demo decision; no model was run.",
                    "usage": {
                        role: {
                            "input_tokens": int(rng.integers(1500, 5000)),
                            "output_tokens": int(rng.integers(200, 900)),
                            "requests": int(rng.integers(1, 4)),
                            "tool_calls": int(rng.integers(0, 8)),
                        }
                        for role in ROLES
                    },
                    "agent_runs": {},
                }
            )
            for ticker in picks[:3]:
                trade_rows.append(
                    {
                        "decision_date": day.isoformat(),
                        "exec_date": (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat(),
                        "ticker": ticker,
                        "action": str(rng.choice(["buy", "sell"])),
                        "shares": round(float(rng.uniform(10, 300)), 4),
                        "price": round(float(rng.uniform(80, 320)), 2),
                        "notional": round(float(rng.uniform(1000, 20000)), 2),
                        "fee": round(float(rng.uniform(1, 10)), 2),
                    }
                )
            event_lines.append({"type": "decision", "date": day.isoformat(), "weights": weights})
            event_lines.append(
                {"type": "tool_call", "agent": "research", "tool": "get_market_events", "date": day.isoformat()}
            )
            event_lines.append(
                {"type": "tool_call", "agent": "analyst", "tool": "get_technical_indicators", "date": day.isoformat()}
            )

        total_in = sum(d["usage"][r]["input_tokens"] for d in decision_lines for r in ROLES)
        total_out = sum(d["usage"][r]["output_tokens"] for d in decision_lines for r in ROLES)

        rets = equity_df["portfolio"].pct_change().dropna()
        run_id = f"{stamp}_2022-bear_{name}"
        run_meta = {
            "run_id": run_id,
            "label": "demo",
            "beatspy_version": __version__,
            "created_utc": created,
            "synthetic": True,
            "disclaimer": DISCLAIMER,
            "model": {"base_url": "synthetic://demo", "model": name, "temperature": 0.0},
            "scenario": {"name": "2022-bear", "start": "2022-01-03", "end": "2022-06-30"},
            "requested": {"frequency": "monthly", "start": "2022-01-03", "end": "2022-06-30"},
            "data": {"snapshot_id": "synthetic", "tickers": TICKERS},
            "bench": {},
            "note": "Synthetic demo run generated by `beatspy demo`. Not a real benchmark result.",
        }
        from ..engine.metrics import total_return

        metrics = {
            "total_return": round(float(equity_df["portfolio"].iloc[-1] / 100_000.0 - 1), 6),
            "spy_total_return": round(float(spy_line[-1] / 100_000.0 - 1), 6),
            "excess_return_vs_spy": round(
                float(equity_df["portfolio"].iloc[-1] / equity_df["portfolio"].iloc[0] - 1)
                - float(spy_line[-1] / spy_line[0] - 1),
                6,
            ),
            "cagr": round(float((equity_df["portfolio"].iloc[-1] / 100_000.0) ** (252 / len(days)) - 1), 6),
            "annual_volatility": round(float(rets.std(ddof=1) * np.sqrt(252)), 6),
            "sharpe": round(float(rets.mean() / rets.std(ddof=1) * np.sqrt(252)), 4),
            "max_drawdown": round(float((equity_df["portfolio"] / equity_df["portfolio"].cummax() - 1).min()), 6),
            "outperformance_hit_rate": 0.5,
            "directional_accuracy": 0.5,
            "decisions": len(decision_lines),
            "avg_turnover": 0.12,
            "trades": len(trade_rows),
            "invalid_outputs": 0,
            "risk_violations": 0,
            "tool_calls": sum(d["usage"][r]["tool_calls"] for d in decision_lines for r in ROLES),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "requests": sum(d["usage"][r]["requests"] for d in decision_lines for r in ROLES),
            "estimated_cost_usd": None,
            "baselines": {
                "equal_weight": round(total_return(equity_df["equal_weight"]), 6),
                "sixty_forty": round(total_return(equity_df["sixty_forty"]), 6),
                "momentum_12_1": round(total_return(equity_df["momentum_12_1"]), 6),
            },
        }
        out_dir = out_root / run_id
        write_run_artifacts(out_dir, run_meta, metrics, equity_df, trade_rows, decision_lines, event_lines)
        run_dirs.append(out_dir)
        log.info("generated synthetic demo run %s", out_dir)
    return run_dirs
