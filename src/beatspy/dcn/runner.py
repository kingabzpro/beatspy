"""End-to-end DCN runs: decision loop, execution, calibration, metrics, artifacts.

The loop is the same one the benchmark has always used — resolve dates, walk them
in order, execute at the next session's open, score through the cutoff — with the
five-agent relay replaced by a single decision-model call per decision.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import statistics
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .. import __version__
from ..artifacts import RUN_FILES, atomic_json, sha256
from ..config import Settings, results_dir
from ..data.calendar import resolve_scenario
from ..data.freeze import load_snapshot
from ..data.service import DataService
from ..engine.backtest import BacktestResult, run_backtest
from ..schemas import CalibrationReport
from . import DCN_PROTOCOL
from .calibration import Observation, build_report
from .clients import DcnClient, make_client
from .pipeline import DcnDecision, DcnPipeline, label_observations

log = logging.getLogger(__name__)

DISCLAIMER = (
    "Historical decision-model benchmark. The model receives frozen point-in-time market state, "
    "returns typed probabilities, and is scored on those probabilities and on a portfolio built from them."
)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:48] or "run"


def ensure_data(scenario, root: Path | None = None, refresh: bool = False) -> tuple[DataService, dict]:
    prices, manifest = load_snapshot(scenario, root=root, refresh=refresh)
    return DataService(prices, benchmark=scenario.benchmark), manifest


def write_run_artifacts(
    out_dir: Path,
    run_meta: dict,
    metrics: dict,
    equity_df: pd.DataFrame,
    trade_rows: list[dict],
    decision_lines: list[dict],
    event_lines: list[dict],
) -> None:
    """Write the standard artifact set for one run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2, default=str), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    equity = equity_df.copy()
    equity.index.name = "date"
    equity.to_csv(out_dir / "equity_curve.csv")
    trades = pd.DataFrame(
        trade_rows,
        columns=["decision_date", "exec_date", "ticker", "action", "shares", "price", "notional", "fee"],
    )
    trades.to_csv(out_dir / "trades.csv", index=False)
    with (out_dir / "decisions.jsonl").open("w", encoding="utf-8") as fh:
        for line in decision_lines:
            fh.write(json.dumps(line, default=str) + "\n")
    with (out_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        for line in event_lines:
            fh.write(json.dumps(line, default=str) + "\n")


def _recent_summary(day: date, weights: dict[str, float]) -> dict:
    """A small, honest history entry: what we held and when. No commentary."""
    held = ", ".join(f"{ticker} {weight:.0%}" for ticker, weight in sorted(weights.items()))
    return {"date": day.isoformat(), "held": held or "all cash"}


def decision_line(decision: DcnDecision) -> dict:
    return {
        "date": decision.date.isoformat(),
        "question_form": decision.question_form,
        "tickers": decision.tickers,
        "state_sha256": decision.state_sha256,
        "state_chars": len(decision.state_text),
        "questions": {key: value.model_dump() for key, value in decision.questions.items()},
        "answers": {key: value.model_dump() for key, value in decision.answers.items()},
        "missing_ids": decision.missing_ids,
        "backfilled": decision.backfilled,
        "probabilities": decision.probabilities,
        "choice_probabilities": decision.choice_probabilities,
        "rank_probabilities": decision.rank_probabilities,
        "validated": {
            "weights": decision.validated.weights,
            "cash": round(decision.validated.cash, 4),
            "violations": decision.validated.violations,
            "invalid": decision.validated.invalid,
        },
        "coverage": round(decision.coverage, 4),
        "observed": [
            {
                "ticker": obs.ticker,
                "noul": obs.noul,
                "choice": obs.choice,
                "rank": obs.rank,
                "outcome": obs.outcome,
                "excess_return": obs.excess_return,
            }
            for obs in decision.observed
        ],
        "call": decision.call.as_dict() if decision.call else None,
        "error": decision.error,
    }


async def run_dcn(
    scenario,
    settings: Settings,
    *,
    freq: str | None = None,
    start: str | None = None,
    end: str | None = None,
    label: str | None = None,
    refresh_data: bool = False,
    out_root: Path | None = None,
    data_root: Path | None = None,
    progress: Callable[[int, int, date], None] | None = None,
    prepared_data: tuple[DataService, dict] | None = None,
    client: DcnClient | None = None,
    question_form: str = "per_asset",
) -> Path:
    """Run the full DCN benchmark for one decision model on one scenario."""
    scenario = resolve_scenario(scenario)
    scenario = scenario.model_copy(
        update={"start": start or scenario.start, "end": end or scenario.end, "frequency": freq or scenario.frequency}
    )
    data, manifest = prepared_data or await asyncio.to_thread(ensure_data, scenario, data_root, refresh_data)
    owns_client = client is None
    dcn_client = client or make_client(settings)
    try:
        pipeline = DcnPipeline(settings, scenario, dcn_client, question_form=question_form)
        start_day = date.fromisoformat(start or scenario.start)
        end_day = date.fromisoformat(end or scenario.end)
        decision_dates = data.decision_dates(
            start_day, end_day, freq or scenario.frequency, max_decisions=scenario.max_decisions
        )
        if not decision_dates:
            raise RuntimeError("No decision dates in the requested range; check scenario start/end.")

        decisions: list[DcnDecision] = []
        event_lines: list[dict] = []
        observations: list[Observation] = []
        recent: list[dict] = []

        async def weight_fn(day: date, state) -> dict:
            invested = state.weights(day, data)
            period = data.decision_period(day, decision_dates, end_day)
            if progress is not None:
                progress(len(decisions) + 1, len(decision_dates), day)
            decision = await pipeline.decide(
                data,
                day,
                holdings=invested,
                cash=1.0 - sum(invested.values()),
                horizon_trading_days=period["horizon_trading_days"],
                next_decision_date=period["next_decision_date"],
                recent=list(recent[-3:]),
            )
            decision.observed = label_observations(data, scenario, decision)
            observations.extend(decision.observed)
            decisions.append(decision)
            event_lines.append(
                {
                    "type": "decision",
                    "date": day.isoformat(),
                    "weights": decision.validated.weights,
                    "cash": round(decision.validated.cash, 4),
                    "coverage": round(decision.coverage, 4),
                    "violations": decision.validated.violations,
                    "error": decision.error,
                }
            )
            # An unusable answer set means "hold", never "liquidate".
            target = invested if (decision.validated.invalid or decision.error) else decision.validated.weights
            recent.append(_recent_summary(day, target))
            return {"weights": target}

        result: BacktestResult = await run_backtest(
            data, decision_dates, weight_fn, scenario, allowed_tickers=scenario.tradable
        )

        from .scoring import score_run

        metrics, equity_df = await score_run(
            data,
            scenario,
            settings,
            result,
            decision_dates,
            decisions,
            observations,
        )

        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}_{_slug(scenario.name)}_{_slug(settings.decision.decision_model)}_{uuid4().hex[:10]}"
        out_dir = (out_root or results_dir()) / run_id

        run_meta = {
            "schema_version": 3,
            "benchmark": "dcn",
            "dcn_protocol": DCN_PROTOCOL,
            "source_revision": source_revision(),
            "run_id": run_id,
            "label": label,
            "beatspy_version": __version__,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "synthetic": False,
            "disclaimer": DISCLAIMER,
            "decision": {
                "decision_model": settings.decision.decision_model,
                "provider": settings.decision.provider,
                "label": settings.decision.label,
                "endpoint": settings.decision.endpoint,
                "question_form": question_form,
                "pricing": {
                    "cost_per_m_input": settings.decision.cost_per_m_input,
                    "cost_per_m_output": settings.decision.cost_per_m_output,
                },
                "policy": {
                    "selection": settings.decision.selection,
                    "min_probability": settings.decision.min_probability,
                    "top_n": settings.decision.top_n,
                    "min_names": settings.decision.min_names,
                },
            },
            "scenario": scenario.model_dump(),
            "requested": {
                "frequency": freq or scenario.frequency,
                "start": start_day.isoformat(),
                "end": end_day.isoformat(),
            },
            "data": {
                "snapshot_id": manifest.get("snapshot_id"),
                "tickers": manifest.get("tickers"),
                "start": manifest.get("start"),
                "end": manifest.get("end"),
                "rows": manifest.get("rows"),
                "fetched_at_utc": manifest.get("fetched_at_utc"),
                "source": manifest.get("source"),
                "prices_sha256": manifest.get("prices_sha256"),
            },
            "bench": settings.bench.model_dump(),
        }

        write_run_artifacts(
            out_dir,
            run_meta,
            metrics.model_dump(),
            equity_df,
            [trade.as_row() for trade in result.trades],
            [decision_line(decision) for decision in decisions],
            event_lines,
        )
        import shutil

        shutil.copytree(Path(manifest["path"]), out_dir / "snapshot")
        run_meta["artifact_hashes"] = {name: sha256(out_dir / name) for name in RUN_FILES}
        run_meta["artifact_hashes"]["snapshot/MANIFEST.json"] = sha256(out_dir / "snapshot/MANIFEST.json")
        run_meta["artifact_hashes"]["snapshot/prices.csv"] = sha256(out_dir / "snapshot/prices.csv")
        atomic_json(out_dir / "run.json", run_meta)
        return out_dir
    finally:
        if owns_client:
            await dcn_client.close()


def source_revision() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_run(run_dir: Path) -> tuple[dict, dict, list[dict]]:
    """Load a recorded run: run.json, metrics.json, and the decision lines."""
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    lines = [
        json.loads(line)
        for line in (run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return meta, metrics, lines


def summarize_decisions(decision_lines: list[dict]) -> dict:
    """Aggregate the per-decision records into the numbers the report shows."""
    coverages = [float(line.get("coverage", 0.0)) for line in decision_lines]
    latencies = [
        float(line["call"]["latency_s"])
        for line in decision_lines
        if isinstance(line.get("call"), dict) and line["call"].get("latency_s") is not None
    ]
    errors = [line["date"] for line in decision_lines if line.get("error")]
    return {
        "coverage": round(statistics.mean(coverages), 4) if coverages else 0.0,
        "mean_latency_s": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "failed_decisions": errors,
    }


def calibration_from_decisions(decision_lines: list[dict]) -> CalibrationReport:
    """Rebuild the calibration report from recorded decisions alone.

    Replay uses this so it never needs the network and never re-derives a label
    differently from the original run.
    """
    observations: list[Observation] = []
    for line in decision_lines:
        for row in line.get("observed") or []:
            observations.append(Observation(**row))
    summary = summarize_decisions(decision_lines)
    return build_report(
        observations,
        coverage=summary["coverage"],
        horizons=len(decision_lines),
        note="rebuilt offline from decisions.jsonl",
    )