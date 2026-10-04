"""End-to-end benchmark runs: decision loop, execution, metrics, artifacts."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from .. import __version__
from ..agents.pipeline import AgentExecutor, DecisionPipeline
from ..artifacts import RUN_FILES, atomic_json, sha256
from ..config import Settings, model_api_key, provider_api_key, results_dir
from ..data.calendar import resolve_scenario
from ..data.freeze import load_snapshot
from ..data.service import DataService
from ..engine.backtest import (
    BacktestResult,
    run_backtest,
)
from ..models.provider import BeatSpyModelProvider
from ..scenarios import load_events
from ..schemas import Scenario
from ..tools.context import ToolContext
from ..tools.forecast import get_forecast_fn

log = logging.getLogger(__name__)

DISCLAIMER = "Historical market benchmark. Trades are simulated and scored against frozen market prices."


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:48] or "run"


def ensure_data(scenario: Scenario, root: Path | None = None, refresh: bool = False) -> tuple[DataService, dict]:
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
    """Write the standard artifact set for one run (used by runner and demo)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(run_meta, indent=2, default=str), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    equity = equity_df.copy()
    equity.index.name = "date"
    equity.to_csv(out_dir / "equity_curve.csv")
    trades = pd.DataFrame(
        trade_rows,
        columns=[
            "decision_date",
            "exec_date",
            "ticker",
            "action",
            "shares",
            "price",
            "notional",
            "fee",
        ],
    )
    trades.to_csv(out_dir / "trades.csv", index=False)
    with (out_dir / "decisions.jsonl").open("w", encoding="utf-8") as fh:
        for line in decision_lines:
            fh.write(json.dumps(line, default=str) + "\n")
    with (out_dir / "events.jsonl").open("w", encoding="utf-8") as fh:
        for line in event_lines:
            fh.write(json.dumps(line, default=str) + "\n")


async def run_benchmark(
    scenario: Scenario,
    settings: Settings,
    *,
    freq: str | None = None,
    start: str | None = None,
    end: str | None = None,
    label: str | None = None,
    refresh_data: bool = False,
    executor: AgentExecutor | None = None,
    out_root: Path | None = None,
    data_root: Path | None = None,
    progress: Callable[[int, int, date], None] | None = None,
    prepared_data: tuple[DataService, dict] | None = None,
    agent_limit: asyncio.Semaphore | None = None,
    external_limit: asyncio.Semaphore | None = None,
    http_client: object = None,
) -> Path:
    """Run the full benchmark for one model on one scenario; returns the run dir.

    progress(i, total, decision_date) is called before each decision so callers
    can surface live feedback.
    """
    scenario = resolve_scenario(scenario)
    scenario = scenario.model_copy(
        update={"start": start or scenario.start, "end": end or scenario.end, "frequency": freq or scenario.frequency}
    )
    scenario = Scenario.model_validate(scenario.model_dump())
    external_limit = external_limit or asyncio.Semaphore(6)
    data, manifest = prepared_data or await asyncio.to_thread(ensure_data, scenario, data_root, refresh_data)
    provider = BeatSpyModelProvider(settings.model.base_url, model_api_key(settings))
    try:
        web_enabled = scenario.allow_web_search and provider_api_key(settings, "olostep") is not None
        forecast_fn = get_forecast_fn(
            scenario.forecast_provider or settings.tools.forecast_provider,
            provider_api_key(settings, "timegpt"),
        )
        events = load_events(scenario.name)
        pipeline = DecisionPipeline(
            settings,
            scenario,
            provider,
            executor=executor,
            web_tools=web_enabled,
            agent_limit=agent_limit,
        )

        start_day = date.fromisoformat(start or scenario.start)
        end_day = date.fromisoformat(end or scenario.end)
        decision_dates = data.decision_dates(start_day, end_day, freq or scenario.frequency)
        if not decision_dates:
            raise RuntimeError("No decision dates in the requested range; check scenario start/end.")

        decisions = []
        event_lines: list[dict] = []
        recent_summary: str | None = None
        harness_violations: list[dict] = []

        done = {"count": 0}

        async def weight_fn(day: date, state) -> dict:
            nonlocal recent_summary
            done["count"] += 1
            if progress is not None:
                progress(done["count"], len(decision_dates), day)
            tctx = ToolContext(
                as_of=day,
                data=data,
                scenario=scenario,
                events_feed=events,
                forecast_fn=forecast_fn,
                finnhub_api_key=(provider_api_key(settings, "finnhub") if scenario.allow_finnhub else None),
                olostep_api_key=(provider_api_key(settings, "olostep") if web_enabled else None),
                tool_budget_per_agent=settings.bench.tool_budget_per_agent,
                external_limit=external_limit,
                http_client=http_client,
            )
            invested = state.weights(day, data)
            record = await pipeline.decide(tctx, invested, 1.0 - sum(invested.values()), recent_summary)
            decisions.append(record)
            # An unparseable decision means "hold", never "liquidate": a model
            # failure must not turn into an unwanted trade.
            target_weights = invested if record.validated.invalid else record.validated.weights
            event_lines.append(
                {
                    "type": "decision",
                    "date": day.isoformat(),
                    "weights": record.validated.weights,
                    "cash": round(record.validated.cash, 4),
                    "predicted_direction": record.predicted_direction,
                    "violations": record.validated.violations,
                    "parse_errors": record.parse_errors,
                }
            )
            event_lines.extend(tctx.events)
            harness_violations.extend(tctx.violations)
            weights_text = ", ".join(f"{t} {w:.0%}" for t, w in sorted(target_weights.items())) or "all cash"
            recent_summary = f"{day.isoformat()}: allocated {weights_text}; {record.rationale[:200]}"
            return {"weights": target_weights}

        log.info("Running %d decisions for %s on %s", len(decision_dates), scenario.name, settings.model.model)
        result: BacktestResult = await run_backtest(data, decision_dates, weight_fn, scenario)

        from .scoring import score_run

        metrics, equity_df = await score_run(
            data, scenario, settings, result, decision_dates, decisions, harness_violations
        )

        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}_{_slug(scenario.name)}_{_slug(settings.model.model)}_{uuid4().hex[:10]}"
        out_dir = (out_root or results_dir()) / run_id

        run_meta = {
            "schema_version": 2,
            "protocol_version": 2,
            "source_revision": source_revision(),
            "run_id": run_id,
            "label": label,
            "beatspy_version": __version__,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "synthetic": False,
            "disclaimer": DISCLAIMER,
            "model": settings.model.model_dump(),
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
            "capabilities": {
                "web_search_tools": web_enabled,
                "finnhub_tools": scenario.allow_finnhub and provider_api_key(settings, "finnhub") is not None,
                "forecast_provider": scenario.forecast_provider or settings.tools.forecast_provider,
            },
            "bench": settings.bench.model_dump(),
            "event_feed": {
                "count": len(events),
                "through": max((e.date for e in events), default=None),
                "note": "Sparse curated events, not a complete news history; coverage may lag the data cutoff.",
            },
        }

        decision_lines = [
            {
                "date": rec.date.isoformat(),
                "artifacts": rec.artifacts,
                "decision": rec.raw_decision,
                "validated": {
                    "weights": rec.validated.weights,
                    "cash": round(rec.validated.cash, 4),
                    "violations": rec.validated.violations,
                    "invalid": rec.validated.invalid,
                },
                "predicted_direction": rec.predicted_direction,
                "rationale": rec.rationale,
                "usage": rec.usage_by_role(),
                "totals": rec.totals(),
                "agent_runs": {role: o.as_dict() for role, o in rec.outcomes.items()},
                "parse_errors": rec.parse_errors,
            }
            for rec in decisions
        ]
        trade_rows = [t.as_row() for t in result.trades]

        write_run_artifacts(out_dir, run_meta, metrics.model_dump(), equity_df, trade_rows, decision_lines, event_lines)
        import shutil

        shutil.copytree(Path(manifest["path"]), out_dir / "snapshot")
        run_meta["artifact_hashes"] = {name: sha256(out_dir / name) for name in RUN_FILES}
        run_meta["artifact_hashes"]["snapshot/MANIFEST.json"] = sha256(out_dir / "snapshot/MANIFEST.json")
        run_meta["artifact_hashes"]["snapshot/prices.csv"] = sha256(out_dir / "snapshot/prices.csv")
        atomic_json(out_dir / "run.json", run_meta)

        return out_dir
    finally:
        await provider.client.close()


def source_revision() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
