"""Offline replay validates scores, not the claimed model's identity."""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import fields
from datetime import date
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from ..agents.pipeline import ROLES, AgentOutcome, DecisionPipeline
from ..artifacts import RUN_FILES, sha256
from ..config import Settings
from ..data.calendar import completed_session
from ..data.freeze import read_snapshot
from ..data.service import DataService
from ..engine.backtest import run_backtest
from ..schemas import RunMetrics, Scenario
from ..tools.context import ToolContext
from .scoring import score_run

SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")


def read_json(path: Path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 25_000_000:
        raise ValueError(f"missing, oversized, or unsafe JSON artifact: {path.name}")

    def invalid(value):
        raise ValueError(f"invalid JSON number: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_numbers(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number in artifacts")
    if isinstance(value, dict):
        for child in value.values():
            check_numbers(child)
    elif isinstance(value, list):
        for child in value:
            check_numbers(child)


def snapshot_directory(run_dir: Path, meta: dict) -> Path:
    ident = meta["data"]["snapshot_id"]
    if not re.fullmatch(r"[a-f0-9]{64}", ident):
        raise ValueError("invalid snapshot id")
    local = run_dir / "snapshot"
    directory = local if local.exists() else run_dir.parent.parent / "snapshots" / ident
    if directory.is_symlink() or directory.parent.is_symlink():
        raise ValueError("snapshot symlinks are not allowed")
    return directory


def validate_run(run_dir: Path) -> dict:
    return asyncio.run(replay_run(run_dir))


async def replay_run(run_dir: Path) -> dict:
    if run_dir.is_symlink() or run_dir.parent.is_symlink():
        raise ValueError("run symlinks are not allowed")
    meta = read_json(run_dir / "run.json")
    if meta.get("synthetic") is not False:
        raise ValueError("synthetic or incomplete results cannot be submitted")
    if meta.get("schema_version") != 2 or meta.get("protocol_version") not in (2, 3, 4, 5):
        raise ValueError("only complete protocol 2, 3, 4, or 5 runs support submission; rerun legacy results")
    legacy = meta["protocol_version"] == 2
    if not SAFE_ID.fullmatch(meta["run_id"]) or meta["run_id"] != run_dir.name:
        raise ValueError("invalid or mismatched run id")
    if "trust" in meta or "maintainer_run" in meta:
        raise ValueError("submission cannot assign its own trust")
    check_numbers(meta)
    scenario = Scenario.model_validate(meta["scenario"])
    if not scenario.start or not scenario.end:
        raise ValueError("submission dates must be resolved")
    if date.fromisoformat(scenario.end) > completed_session():
        raise ValueError("submission includes a future or incomplete session")
    requested = meta["requested"]
    if (requested["start"], requested["end"], requested["frequency"]) != (
        scenario.start,
        scenario.end,
        scenario.frequency,
    ):
        raise ValueError("requested window does not match scenario")
    settings = Settings.model_validate({"model": meta["model"], "bench": meta["bench"]})
    snapshot = snapshot_directory(run_dir, meta)
    required = {name: run_dir / name for name in RUN_FILES}
    required.update(
        {"snapshot/prices.csv": snapshot / "prices.csv", "snapshot/MANIFEST.json": snapshot / "MANIFEST.json"}
    )
    if set(meta["artifact_hashes"]) != set(required):
        raise ValueError("incomplete artifact hashes")
    for name, path in required.items():
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 25_000_000:
            raise ValueError(f"missing, oversized, or unsafe artifact: {name}")
        if sha256(path) != meta["artifact_hashes"][name]:
            raise ValueError(f"artifact hash mismatch: {name}")
    prices, manifest = read_snapshot(snapshot, scenario)
    if manifest["snapshot_id"] != meta["data"]["snapshot_id"]:
        raise ValueError("snapshot id mismatch")
    if manifest["prices_sha256"] != meta["data"]["prices_sha256"]:
        raise ValueError("price hash mismatch")
    for key in ("tickers", "start", "end", "rows", "source", "fetched_at_utc"):
        if meta["data"].get(key) != manifest.get(key):
            raise ValueError(f"run data metadata mismatch: {key}")
    if prices.date.max().date() > date.fromisoformat(scenario.end):
        raise ValueError("snapshot includes prices beyond the evaluation cutoff")
    data = DataService(prices, scenario.benchmark)
    decision_dates = data.decision_dates(
        date.fromisoformat(scenario.start),
        date.fromisoformat(scenario.end),
        scenario.frequency,
        legacy=legacy,
        max_decisions=scenario.max_decisions,
    )
    submitted = lines(run_dir / "decisions.jsonl")
    events = lines(run_dir / "events.jsonl")
    check_numbers(submitted)
    check_numbers(events)
    if [row["date"] for row in submitted] != [day.isoformat() for day in decision_dates]:
        raise ValueError("decision dates do not match benchmark schedule")
    for event in events:
        if event.get("date") not in {day.isoformat() for day in decision_dates}:
            raise ValueError("event outside decision schedule")
        if event.get("type") in ("tool_call", "budget_exhausted") and event.get("agent") not in ROLES:
            raise ValueError("unknown agent in tool event")
    lookup = {row["date"]: row for row in submitted}

    class ReplayExecutor:
        async def run(self, agent, input_text, tctx):
            raw = lookup[tctx.as_of.isoformat()]["agent_runs"][agent.name]
            for name in ("input_tokens", "output_tokens", "requests"):
                if type(raw[name]) is not int or raw[name] < 0:
                    raise ValueError("invalid agent usage")
            values = {f.name: raw[f.name] for f in fields(AgentOutcome) if f.name in raw}
            values["role"] = agent.name
            return AgentOutcome(**values)

    pipeline = DecisionPipeline(
        settings,
        scenario,
        provider=None,
        executor=ReplayExecutor(),
        legacy_parsing=legacy,
        protocol_version=meta["protocol_version"],
    )
    records = []
    recent_summary = None

    async def weights(day, state):
        nonlocal recent_summary
        row = lookup[day.isoformat()]
        if set(row["agent_runs"]) != set(ROLES):
            raise ValueError("incomplete five-agent outputs")
        for role in ROLES:
            calls = [
                event
                for event in events
                if event.get("date") == day.isoformat()
                and event.get("agent") == role
                and event.get("type") == "tool_call"
            ]
            if len(calls) > settings.bench.tool_budget_per_agent:
                raise ValueError("per-agent tool budget exceeded")
        invested = state.weights(day, data)
        period = data.decision_period(day, decision_dates, date.fromisoformat(scenario.end))
        context = ToolContext(
            day,
            data,
            scenario,
            tool_budget_per_agent=settings.bench.tool_budget_per_agent,
            horizon_days=period["horizon_trading_days"],
            next_decision_date=period["next_decision_date"],
            finnhub_api_key="replay-enabled" if meta["capabilities"].get("finnhub_tools") else None,
            olostep_api_key="replay-enabled" if meta["capabilities"].get("web_search_tools") else None,
        )
        record = await pipeline.decide(context, invested, 1 - sum(invested.values()), recent_summary)
        if not legacy and row.get("market_brief") != record.market_brief:
            raise ValueError(f"market brief replay mismatch on {day}")
        for name, expected in {
            "decision": record.raw_decision,
            "artifacts": record.artifacts,
            "usage": record.usage_by_role(),
            "totals": record.totals(),
            "predicted_direction": record.predicted_direction,
            "rationale": record.rationale,
            "parse_errors": record.parse_errors,
            "validated": {
                "weights": record.validated.weights,
                "cash": round(record.validated.cash, 4),
                "violations": record.validated.violations,
                "invalid": record.validated.invalid,
            },
        }.items():
            if row.get(name) != expected:
                raise ValueError(f"decision replay mismatch: {name} on {day}")
        records.append(record)
        target = invested if record.validated.invalid else record.validated.weights
        weights_text = ", ".join(f"{t} {w:.0%}" for t, w in sorted(target.items())) or "all cash"
        recent_summary = f"{day.isoformat()}: allocated {weights_text}; {record.rationale[:200]}"
        return {"weights": target}

    result = await run_backtest(
        data,
        decision_dates,
        weights,
        scenario,
        legacy=legacy,
        allowed_tickers=scenario.tradable if meta["protocol_version"] >= 4 else None,
    )
    violations = [event for event in events if event.get("type") == "violation"]
    expected_metrics, expected_equity = await score_run(
        data, scenario, settings, result, decision_dates, records, violations, legacy=legacy
    )
    metrics = read_json(run_dir / "metrics.json")
    check_numbers(metrics)
    RunMetrics.model_validate(metrics)
    if metrics != expected_metrics.model_dump():
        raise ValueError("metrics replay mismatch")
    equity = pd.read_csv(run_dir / "equity_curve.csv", index_col="date", parse_dates=True)
    equity.index = equity.index.date
    equity.index.name = None
    try:
        assert_frame_equal(equity, expected_equity, check_dtype=False, atol=1e-6, rtol=1e-9)
        trades = pd.read_csv(run_dir / "trades.csv")
        expected_trades = pd.DataFrame([trade.as_row() for trade in result.trades], columns=trades.columns)
        if list(trades.columns) != [
            "decision_date",
            "exec_date",
            "ticker",
            "action",
            "shares",
            "price",
            "notional",
            "fee",
        ]:
            raise ValueError("invalid trade columns")
        assert_frame_equal(trades, expected_trades, check_dtype=False, atol=1e-6, rtol=1e-9)
    except AssertionError as exc:
        raise ValueError("trades or equity replay mismatch") from exc
    return meta
