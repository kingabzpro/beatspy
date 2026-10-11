"""Adversarial look-ahead audit: can future prices change a past decision?

    uv run python examples/leakage_audit.py [run_dir]

Takes a finished run (default: the most recent one under ``results/``), reads the
frozen snapshot it was priced from, and runs the real pipeline twice over it. In
the second copy, every price after a cut date is replaced with different,
deterministic, plausible data. Then it requires:

1. each decision on or before the cut to be byte-identical (state, questions,
   probabilities, weights), through the full pipeline with chained holdings;
2. every decision after the cut to differ, or the tampering was inert and the
   audit proves nothing;
3. the realized labels of at least one pre-cut decision to change, where the
   horizon crosses the cut, which proves labels do read the realized future (as
   designed) and that the tampered data is real data;
4. the run's own first recorded state hash to reproduce from the snapshot alone.

Exit code 0 means no path lets data after a decision date reach that decision.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from beatspy.config import Settings
from beatspy.data.calendar import resolve_scenario
from beatspy.data.service import DataService
from beatspy.dcn.clients import DcnResult
from beatspy.dcn.pipeline import DcnPipeline
from beatspy.dcn.runner import read_run
from beatspy.dcn.state import build_state, enforce_budget, render_state, select_universe
from beatspy.scenarios import load_scenario
from beatspy.schemas import NoulAnswer


class StateSensitiveClient:
    """Answers as a deterministic function of the state, so any leak propagates.

    The scripted test client hashes question ids, which would hide a state
    difference. This one reads the state's own numbers, so a leaked future bar
    would change the probability and therefore the weights.
    """

    provider = "audit"
    model = "audit"
    endpoint = "https://audit.invalid"

    async def decide(self, state, questions):
        market = {row["ticker"]: row for row in state["market"]}
        answers = {}
        for question_id in questions:
            row = market.get(question_id.split("_", 1)[-1]) or {}
            five_day = float(row.get("last_5d_pct") or 0.0)
            ninety_day = float(row.get("return_90d_pct") or 0.0)
            probability = 0.5 + five_day / 50.0 + ninety_day / 500.0
            answers[question_id] = NoulAnswer(noul=round(min(max(probability, 0.01), 0.99), 4))
        return DcnResult(answers=answers, input_tokens=100, latency_s=0.0, attempts=1)

    async def close(self):
        return None


def tamper(prices: pd.DataFrame, cut: date) -> pd.DataFrame:
    """Different, deterministic, plausible prices after `cut`."""
    frame = prices.copy()
    later = frame["date"] > pd.Timestamp(cut)
    count = int(later.sum())
    rng = np.random.default_rng(20261011)
    factor = rng.uniform(0.5, 1.5, count)
    for column in ("open", "high", "low", "close"):
        frame.loc[later, column] = frame.loc[later, column].to_numpy() * factor
    frame.loc[later, "volume"] = frame.loc[later, "volume"] * 3.0
    return frame


def latest_run(repo: Path) -> Path:
    runs = [p for p in (repo / "results").glob("*") if (p / "metrics.json").is_file()]
    if not runs:
        raise SystemExit("no finished run under results/; run `beatspy run` first")
    return max(runs, key=lambda p: p.stat().st_mtime)


async def main(argv: list[str]) -> int:
    repo = Path(__file__).resolve().parents[1]
    run_dir = Path(argv[1]).resolve() if len(argv) > 1 else latest_run(repo)
    meta, _, lines = read_run(run_dir)
    scenario = resolve_scenario(load_scenario(meta["scenario"]["name"]))
    start, end = date.fromisoformat(scenario.start), date.fromisoformat(scenario.end)
    cut = start + timedelta(days=(end - start).days // 2)
    print(f"run      {run_dir.name}")
    print(f"window   {start} → {end} · cut {cut}")

    prices = pd.read_csv(run_dir / "snapshot" / "prices.csv", parse_dates=["date"])
    original = DataService(prices, benchmark=scenario.benchmark)
    tampered = DataService(tamper(prices, cut), benchmark=scenario.benchmark)
    decision_dates = original.decision_dates(
        start, end, scenario.frequency, max_decisions=scenario.max_decisions
    )
    before = [d for d in decision_dates if d <= cut]
    after = [d for d in decision_dates if d > cut]
    print(f"decisions {len(decision_dates)}: {len(before)} on/before the cut, {len(after)} after\n")

    failures: list[str] = []

    def state_for(bench, day: date, period: dict) -> str:
        """The rendered state one service would build for one decision."""
        return render_state(
            build_state(
                bench,
                scenario,
                day,
                holdings={},
                cash=1.0,
                question_tickers=select_universe(bench, scenario, day),
                horizon_trading_days=period["horizon_trading_days"],
                next_decision_date=period["next_decision_date"],
            )
        )

    # 1. Direct state comparison for every pre-cut decision. Each side uses its
    #    own investable set, so a selection rule that peeked ahead would differ.
    for day in before:
        period = original.decision_period(day, decision_dates, end)
        if state_for(original, day, period) != state_for(tampered, day, period):
            failures.append(f"state differs at {day}")
    print(f"[1] states byte-identical for all {len(before)} pre-cut decisions: {not failures}")

    # 2. Full pipeline, chained holdings, on both copies.
    async def run_pipeline(data, *, tolerant=False):
        pipeline = DcnPipeline(Settings(), scenario, StateSensitiveClient(), question_form="per_asset")
        holdings, recent, records = {}, [], []
        for day in decision_dates:
            period = data.decision_period(day, decision_dates, end)
            try:
                decision = await pipeline.decide(
                    data,
                    day,
                    holdings=holdings,
                    cash=1.0 - sum(holdings.values()),
                    horizon_trading_days=period["horizon_trading_days"],
                    next_decision_date=period["next_decision_date"],
                    recent=list(recent[-3:]),
                )
            except Exception as exc:  # tolerated after the cut: divergence is the point
                if tolerant and day > cut:
                    records.append((day, f"RAISED {type(exc).__name__}"))
                    break
                raise
            records.append(decision)
            holdings = dict(decision.validated.weights)
            # Same shape the runner writes into `recent`, so the history channel
            # is exercised with real content rather than a constant.
            held = ", ".join(f"{ticker} {weight:.0%}" for ticker, weight in sorted(holdings.items()))
            recent.append({"date": day.isoformat(), "held": held or "all cash"})
        return records

    honest = await run_pipeline(original)
    cheater = await run_pipeline(tampered, tolerant=True)
    mismatches = []
    # strict=False: the tampered run may stop early, which is itself evidence.
    for a, b in zip(honest, cheater, strict=False):
        day = a if isinstance(a, date) else a.date
        if isinstance(b, tuple):
            if day <= cut:
                mismatches.append(f"tampered run raised pre-cut at {day}")
            continue
        if day <= cut and (
            a.state_text != b.state_text
            or a.probabilities != b.probabilities
            or a.validated.weights != b.validated.weights
            or a.question_ids != b.question_ids
        ):
            mismatches.append(day.isoformat())
    print(f"[2] pipeline state/answers/weights identical pre-cut: {not mismatches} {mismatches or ''}")
    failures.extend(mismatches)

    # 3. Power: post-cut decisions must differ.
    changed = sum(
        1
        for a, b in zip(honest, cheater, strict=False)
        if (a if isinstance(a, date) else a.date) > cut and (isinstance(b, tuple) or a.state_text != b.state_text)
    )
    print(f"[3] post-cut decisions that changed under tampering: {changed}/{len(after)}")
    if changed != len(after):
        failures.append("tampering did not change post-cut states")

    # 4. Labels do read the realized future, where the horizon crosses the cut.
    def realized(bench: DataService, day: date) -> float | None:
        horizon = bench.decision_period(day, decision_dates, end)["horizon_trading_days"]
        return bench.close_on_offset(scenario.benchmark, day, horizon)

    label_changes = sum(1 for day in before if realized(original, day) != realized(tampered, day))
    print(f"[4] pre-cut decisions whose realized label changed: {label_changes}")
    if label_changes == 0:
        failures.append("labels never changed; tamper did not reach the label window")

    # 5. The run's own first state must reproduce from the snapshot alone.
    recorded = lines[0]
    day = date.fromisoformat(recorded["date"])
    period = original.decision_period(day, decision_dates, end)
    rebuilt, trimmed = enforce_budget(
        build_state(
            original,
            scenario,
            day,
            holdings={},
            cash=1.0,
            question_tickers=select_universe(original, scenario, day),
            horizon_trading_days=period["horizon_trading_days"],
            next_decision_date=period["next_decision_date"],
        ),
        Settings().decision.max_state_tokens,
    )
    digest = hashlib.sha256(render_state(rebuilt).encode("utf-8")).hexdigest()
    print(f"[5] rebuilt first decision state matches the record: {digest == recorded['state_sha256']}")
    if digest != recorded["state_sha256"]:
        failures.append("recorded state hash does not reproduce from the snapshot")

    print()
    if failures:
        print("AUDIT FAILED:")
        for item in failures:
            print("  -", item)
        return 1
    print("AUDIT PASSED: no path lets data after a decision date reach that decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv)))
