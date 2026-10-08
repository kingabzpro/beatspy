"""Offline replay for DCN runs: integrity, schedule, and a full re-score.

A DCN run records everything needed to re-derive its own result: the frozen
snapshot it was priced from, the decision schedule, and the weights each decision
produced. This module re-reads those artifacts and re-runs the accounting engine
over the *recorded* weights — no network, no model call — then demands the same
metrics, equity curve, and trade blotter the run published.

This is an integrity check, not an identity check. It does not claim to know
which model produced a run, and it deliberately has no notion of signatures,
public keys, or trust: what it proves is narrower and still useful — that the
numbers in ``metrics.json`` follow from the recorded decisions and the frozen
prices, so a run cannot ship a result its own artifacts do not support.

Every rejection is a ``ValueError`` naming the artifact and the reason.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

from ..artifacts import RUN_FILES, sha256
from ..config import Settings
from ..data.calendar import completed_session
from ..data.freeze import read_snapshot
from ..data.service import DataService
from ..engine.backtest import run_backtest
from ..schemas import RunMetrics, Scenario
from . import DCN_PROTOCOL
from .calibration import Observation
from .scoring import score_run

SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")

# No artifact in a run is legitimately this large; refusing before hashing keeps a
# hostile run directory from being read into memory.
MAX_ARTIFACT_BYTES = 25_000_000

TRADE_COLUMNS = ["decision_date", "exec_date", "ticker", "action", "shares", "price", "notional", "fee"]

# The exact artifact set a DCN run must hash, and nothing else.
HASHED_ARTIFACTS = (*RUN_FILES, "snapshot/MANIFEST.json", "snapshot/prices.csv")

# run.json carries the recorded data block; every key must agree with the snapshot.
DATA_BLOCK_KEYS = ("snapshot_id", "prices_sha256", "tickers", "start", "end", "rows", "source", "fetched_at_utc")


def _is_link(path: Path) -> bool:
    """True for symlinks and for Windows directory junctions.

    A junction is a reparse point that redirects a directory just as a symlink
    does, so it is rejected the same way: a run must not point at artifacts that
    live somewhere else.
    """
    if path.is_symlink():
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))


# --------------------------------------------------------------------------- #
# Shared safety helpers
# --------------------------------------------------------------------------- #


def read_json(path: Path):
    """Read one JSON artifact, refusing symlinks, oversized files, and non-finite numbers."""
    if _is_link(path) or not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"missing, oversized, or unsafe JSON artifact: {path.name}")

    def invalid(value):
        raise ValueError(f"invalid JSON number: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_numbers(value):
    """Reject NaN/Infinity anywhere in an artifact tree."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number in artifacts")
    if isinstance(value, dict):
        for child in value.values():
            check_numbers(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            check_numbers(child)


def snapshot_directory(run_dir: Path, meta: dict) -> Path:
    """The snapshot a run was priced from: its own copy, else the shared cache."""
    ident = (meta.get("data") or {}).get("snapshot_id")
    if not isinstance(ident, str) or not re.fullmatch(r"[a-f0-9]{64}", ident):
        raise ValueError("invalid snapshot id")
    local = run_dir / "snapshot"
    directory = local if local.exists() else run_dir.parent.parent / "snapshots" / ident
    if _is_link(directory) or _is_link(directory.parent):
        raise ValueError("snapshot symlinks are not allowed")
    return directory


# --------------------------------------------------------------------------- #
# Recorded-decision stand-ins: the subset of a DcnDecision that scoring reads
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _ReplayCall:
    input_tokens: int
    output_tokens: int
    requests: int
    latency_s: float


@dataclass(frozen=True)
class _ReplayValidated:
    weights: dict[str, float]
    cash: float
    violations: list[str]
    invalid: bool


@dataclass(frozen=True)
class _ReplayDecision:
    date: date
    call: _ReplayCall | None
    error: str | None
    validated: _ReplayValidated
    coverage: float


def _number(day: str, ticker: str, field: str, value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{day}/{ticker}: observed {field} is not a finite number: {value!r}")
    return float(value)


def _probability(day: str, ticker: str, field: str, value) -> float | None:
    number = _number(day, ticker, field, value)
    if number is not None and not 0.0 <= number <= 1.0:
        raise ValueError(f"{day}/{ticker}: observed {field} is outside [0, 1]: {value!r}")
    return number


def _observation(day: str, row) -> Observation:
    if not isinstance(row, dict):
        raise ValueError(f"{day}: every observed entry must be an object")
    ticker = row.get("ticker")
    if not isinstance(ticker, str) or not ticker:
        raise ValueError(f"{day}: an observed entry has no ticker")
    outcome = row.get("outcome")
    if outcome is not None and (type(outcome) is not int or outcome not in (0, 1)):
        raise ValueError(f"{day}/{ticker}: observed outcome must be 0, 1, or null: {outcome!r}")
    return Observation(
        ticker=ticker,
        noul=_probability(day, ticker, "noul", row.get("noul")),
        choice=_probability(day, ticker, "choice", row.get("choice")),
        rank=_probability(day, ticker, "rank", row.get("rank")),
        outcome=outcome,
        excess_return=_number(day, ticker, "excess_return", row.get("excess_return")),
    )


def _read_decision(day: str, line: dict) -> tuple[_ReplayDecision, dict[str, float], list[Observation]]:
    """Rebuild one recorded decision, its target weights, and its observations."""
    validated = line.get("validated")
    if not isinstance(validated, dict):
        raise ValueError(f"{day}: decision is missing its validated block")

    raw_weights = validated.get("weights")
    if not isinstance(raw_weights, dict):
        raise ValueError(f"{day}: recorded decision has no weight map")
    weights: dict[str, float] = {}
    for ticker, weight in raw_weights.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
            raise ValueError(f"{day}: invalid recorded weight for {ticker!r}: {weight!r}")
        value = float(weight)
        if value < 0:
            raise ValueError(f"{day}: negative recorded weight for {ticker!r}: {value}")
        if value:
            weights[str(ticker).upper()] = value
    exposure = sum(weights.values())
    if exposure > 1.0 + 1e-6:
        raise ValueError(f"{day}: recorded weights invest {exposure:.4f} of the portfolio")

    cash = validated.get("cash")
    if isinstance(cash, bool) or not isinstance(cash, (int, float)) or not math.isfinite(float(cash)):
        raise ValueError(f"{day}: recorded cash is not a finite number: {cash!r}")

    violations = validated.get("violations") or []
    if not isinstance(violations, list) or not all(isinstance(item, str) for item in violations):
        raise ValueError(f"{day}: recorded violations must be a list of strings")

    invalid = validated.get("invalid")
    if not isinstance(invalid, bool):
        raise ValueError(f"{day}: recorded 'invalid' flag must be a boolean")

    coverage = line.get("coverage")
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, (int, float))
        or not math.isfinite(float(coverage))
        or not 0.0 <= float(coverage) <= 1.0
    ):
        raise ValueError(f"{day}: recorded coverage is not a number in [0, 1]: {coverage!r}")

    error = line.get("error")
    if error is not None and not isinstance(error, str):
        raise ValueError(f"{day}: recorded error must be a string or null")

    call = line.get("call")
    replay_call: _ReplayCall | None = None
    if call is not None:
        if not isinstance(call, dict):
            raise ValueError(f"{day}: recorded call is not an object")
        tokens: list[int] = []
        for key in ("input_tokens", "output_tokens", "requests"):
            value = call.get(key, 0)
            if type(value) is not int or value < 0:
                raise ValueError(f"{day}: recorded call.{key} must be a non-negative integer: {value!r}")
            tokens.append(value)
        latency = call.get("latency_s", 0.0)
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(float(latency)):
            raise ValueError(f"{day}: recorded call.latency_s is not a finite number: {latency!r}")
        if latency < 0:
            raise ValueError(f"{day}: recorded call.latency_s is negative: {latency!r}")
        replay_call = _ReplayCall(*tokens, float(latency))

    rows = line.get("observed")
    if not isinstance(rows, list):
        raise ValueError(f"{day}: recorded decision has no observed list")
    observations = [_observation(day, row) for row in rows]

    record = _ReplayDecision(
        date=date.fromisoformat(day),
        call=replay_call,
        error=error,
        validated=_ReplayValidated(weights=weights, cash=float(cash), violations=list(violations), invalid=invalid),
        coverage=float(coverage),
    )
    return record, weights, observations


def _read_lines(path: Path, label: str) -> list[dict]:
    if _is_link(path) or not path.is_file():
        raise ValueError(f"missing, oversized, or unsafe artifact: {label}")
    try:
        rows = lines(path)
    except (ValueError, OSError) as exc:
        raise ValueError(f"{label} is not valid JSONL: {exc}") from exc
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {index} is not a JSON object")
    return rows


def _differences(stored, expected, prefix="") -> list[str]:
    """Human-readable paths where the stored metrics disagree with the replayed ones."""
    out: list[str] = []
    if isinstance(stored, dict) and isinstance(expected, dict):
        for key in sorted(set(stored) | set(expected)):
            if key not in stored:
                out.append(f"{prefix}{key}: missing from metrics.json")
            elif key not in expected:
                out.append(f"{prefix}{key}: present in metrics.json but not replayed")
            else:
                out.extend(_differences(stored[key], expected[key], f"{prefix}{key}."))
    elif stored != expected:
        out.append(f"{prefix.rstrip('.') or 'metrics'}: stored {stored!r} != replayed {expected!r}")
    return out


# --------------------------------------------------------------------------- #
# The replay itself
# --------------------------------------------------------------------------- #


def validate_run(run_dir: Path) -> dict:
    """Validate and replay one run directory. Returns the run's metadata."""
    return asyncio.run(replay_run(run_dir))


async def replay_run(run_dir: Path) -> dict:
    """Re-derive a DCN run's metrics, equity curve, and trades from its own artifacts."""
    run_dir = Path(run_dir)
    if _is_link(run_dir) or _is_link(run_dir.parent):
        raise ValueError("run symlinks are not allowed")

    meta = read_json(run_dir / "run.json")

    benchmark = meta.get("benchmark")
    if benchmark is None:
        raise ValueError("run.json is missing its 'benchmark' field; expected 'dcn'")
    if benchmark != "dcn":
        raise ValueError(f"not a DCN run: benchmark is {benchmark!r}, expected 'dcn'")
    if meta.get("schema_version") != 3:
        raise ValueError(f"unsupported run schema version {meta.get('schema_version')!r}; expected 3")
    if meta.get("synthetic") is not False:
        raise ValueError("synthetic runs cannot be verified: they are fabricated data, not a recorded benchmark")
    if meta.get("dcn_protocol") != DCN_PROTOCOL:
        raise ValueError(
            f"run was recorded under DCN protocol {meta.get('dcn_protocol')!r}; "
            f"this build verifies protocol {DCN_PROTOCOL}, so the recorded rules do not apply"
        )

    run_id = meta.get("run_id")
    if not isinstance(run_id, str) or not SAFE_ID.fullmatch(run_id) or run_id != run_dir.name:
        raise ValueError(f"invalid or mismatched run id {run_id!r} for directory {run_dir.name!r}")

    for key in ("trust", "maintainer_run", "verification", "verified"):
        if key in meta:
            raise ValueError(f"a run cannot assign its own trust: run.json carries {key!r}")

    check_numbers(meta)

    try:
        scenario = Scenario.model_validate(meta["scenario"])
    except KeyError as exc:
        raise ValueError("run.json has no scenario block") from exc
    except ValidationError as exc:
        raise ValueError(f"run.json carries an invalid scenario: {exc}") from exc

    if not scenario.start or not scenario.end:
        raise ValueError("scenario dates must be resolved before a run can be verified")
    last_session = completed_session()
    if date.fromisoformat(scenario.end) > last_session:
        raise ValueError(
            f"scenario ends {scenario.end}, after the last completed session {last_session}; "
            "the evaluation window is not fully observed"
        )

    requested = meta.get("requested")
    if not isinstance(requested, dict):
        raise ValueError("run.json is missing its requested window")
    for key, expected in (("start", scenario.start), ("end", scenario.end), ("frequency", scenario.frequency)):
        if requested.get(key) != expected:
            raise ValueError(
                f"requested window does not match the scenario: requested.{key}={requested.get(key)!r}, "
                f"scenario.{key}={expected!r}"
            )

    decision = meta.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("run.json is missing its decision block")
    policy = decision.get("policy") or {}
    try:
        settings = Settings.model_validate(
            {
                "decision": {
                    "decision_model": decision.get("decision_model"),
                    "min_probability": policy.get("min_probability", 0.5),
                    "top_n": policy.get("top_n", 3),
                    "min_names": policy.get("min_names", 2),
                },
                "bench": meta.get("bench") or {},
            }
        )
    except ValidationError as exc:
        raise ValueError(f"run.json carries an unusable decision configuration: {exc}") from exc

    pricing = decision.get("pricing") or {}
    for key, current in (
        ("cost_per_m_input", settings.decision.cost_per_m_input),
        ("cost_per_m_output", settings.decision.cost_per_m_output),
    ):
        recorded = pricing.get(key)
        if recorded is None:
            continue
        if isinstance(recorded, bool) or not isinstance(recorded, (int, float)) or not math.isfinite(float(recorded)):
            raise ValueError(f"run.json carries an invalid pricing value for {key}: {recorded!r}")
        if current is not None and abs(float(recorded) - float(current)) > 1e-12:
            raise ValueError(
                f"recorded decision pricing {key}={recorded!r} disagrees with the current table ({current!r}); "
                "re-scoring would not be comparable"
            )

    # ---- artifacts ------------------------------------------------------- #
    snapshot = snapshot_directory(run_dir, meta)
    required = {name: run_dir / name for name in RUN_FILES}
    for name in HASHED_ARTIFACTS:
        if name.startswith("snapshot/"):
            required[name] = snapshot / Path(name).name

    hashes = meta.get("artifact_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("run.json is missing its artifact_hashes map")
    missing = sorted(set(required) - set(hashes))
    unexpected = sorted(set(hashes) - set(required))
    if missing or unexpected:
        raise ValueError(f"artifact hash set does not match the run format; missing {missing}, unexpected {unexpected}")
    for name, path in required.items():
        if _is_link(path) or not path.is_file():
            raise ValueError(f"missing, oversized, or unsafe artifact: {name}")
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError(f"oversized artifact: {name}")
        digest = sha256(path)
        if digest != hashes[name]:
            raise ValueError(f"artifact hash mismatch: {name} (recorded {hashes[name]}, found {digest})")

    try:
        prices, manifest = read_snapshot(snapshot, scenario)
    except (ValueError, RuntimeError, OSError) as exc:
        raise ValueError(f"frozen snapshot failed validation: {exc}") from exc

    recorded_data = meta.get("data")
    if not isinstance(recorded_data, dict):
        raise ValueError("run.json is missing its data block")
    for key in DATA_BLOCK_KEYS:
        if recorded_data.get(key) != manifest.get(key):
            raise ValueError(
                f"recorded data block disagrees with the snapshot manifest: "
                f"run.json data.{key}={recorded_data.get(key)!r}, manifest.{key}={manifest.get(key)!r}"
            )

    latest_price = prices.date.max().date()
    if latest_price > last_session:
        raise ValueError(
            f"snapshot contains prices through {latest_price}, after the last completed session {last_session}"
        )

    data = DataService(prices, scenario.benchmark)

    # ---- decision schedule ----------------------------------------------- #
    decision_dates = data.decision_dates(
        date.fromisoformat(scenario.start),
        date.fromisoformat(scenario.end),
        scenario.frequency,
        max_decisions=scenario.max_decisions,
    )
    if not decision_dates:
        raise ValueError("scenario produces no decision dates inside the frozen snapshot")

    submitted = _read_lines(run_dir / "decisions.jsonl", "decisions.jsonl")
    events = _read_lines(run_dir / "events.jsonl", "events.jsonl")
    check_numbers(submitted)
    check_numbers(events)

    expected_dates = [day.isoformat() for day in decision_dates]
    recorded_dates = [line.get("date") for line in submitted]
    if recorded_dates != expected_dates:
        raise ValueError(
            f"recorded decision dates do not match the scenario schedule: recorded {recorded_dates}, "
            f"expected {expected_dates}"
        )

    schedule = set(expected_dates)
    for index, event in enumerate(events, start=1):
        if event.get("date") not in schedule:
            raise ValueError(f"events.jsonl line {index} is outside the decision schedule: {event.get('date')!r}")

    # ---- rebuild the decision records ------------------------------------ #
    records: list[_ReplayDecision] = []
    observations: list[Observation] = []
    tradable = {ticker.upper() for ticker in scenario.tradable}
    for line in submitted:
        day = line["date"]
        record, weights, observed = _read_decision(day, line)
        outside = sorted(set(weights) - tradable)
        if outside:
            raise ValueError(f"{day}: recorded weights hold tickers outside the scenario universe: {outside}")
        records.append(record)
        observations.extend(observed)

    by_date = {record.date.isoformat(): record for record in records}

    # ---- replay the portfolio from the recorded weights ------------------ #
    async def replay_weights(day: date, state) -> dict:
        record = by_date[day.isoformat()]
        # The recording means "hold", never liquidate: an unusable answer set must
        # not be re-read as a decision to sell everything.
        target = state.weights(day, data) if (record.validated.invalid or record.error) else record.validated.weights
        return {"weights": dict(target)}

    result = await run_backtest(data, decision_dates, replay_weights, scenario, allowed_tickers=scenario.tradable)
    expected_metrics, expected_equity = await score_run(
        data, scenario, settings, result, decision_dates, records, observations
    )

    # ---- metrics --------------------------------------------------------- #
    stored_metrics = read_json(run_dir / "metrics.json")
    check_numbers(stored_metrics)
    try:
        RunMetrics.model_validate(stored_metrics)
    except ValidationError as exc:
        raise ValueError(f"metrics.json is not a valid metrics document: {exc}") from exc
    if stored_metrics != expected_metrics.model_dump():
        detail = _differences(stored_metrics, expected_metrics.model_dump())[:6]
        raise ValueError("metrics replay mismatch: " + "; ".join(detail))

    # ---- equity curve ---------------------------------------------------- #
    try:
        stored_equity = pd.read_csv(run_dir / "equity_curve.csv", index_col="date", parse_dates=True)
    except (ValueError, OSError) as exc:
        raise ValueError(f"equity_curve.csv could not be read: {exc}") from exc
    stored_equity.index = stored_equity.index.date
    stored_equity.index.name = None
    if list(stored_equity.columns) != list(expected_equity.columns):
        raise ValueError(
            f"equity_curve.csv columns {list(stored_equity.columns)} do not match {list(expected_equity.columns)}"
        )
    try:
        assert_frame_equal(stored_equity, expected_equity, check_dtype=False, atol=1e-6, rtol=1e-9)
    except AssertionError as exc:
        raise ValueError(f"equity curve replay mismatch: {exc}") from exc

    # ---- trades ---------------------------------------------------------- #
    try:
        stored_trades = pd.read_csv(run_dir / "trades.csv")
    except (ValueError, OSError) as exc:
        raise ValueError(f"trades.csv could not be read: {exc}") from exc
    if list(stored_trades.columns) != TRADE_COLUMNS:
        raise ValueError(f"trades.csv columns {list(stored_trades.columns)} do not match {TRADE_COLUMNS}")
    expected_trades = pd.DataFrame([trade.as_row() for trade in result.trades], columns=TRADE_COLUMNS)
    try:
        assert_frame_equal(stored_trades, expected_trades, check_dtype=False, atol=1e-6, rtol=1e-9)
    except AssertionError as exc:
        raise ValueError(f"trades replay mismatch: {exc}") from exc

    return meta


__all__ = [
    "HASHED_ARTIFACTS",
    "MAX_ARTIFACT_BYTES",
    "SAFE_ID",
    "TRADE_COLUMNS",
    "check_numbers",
    "lines",
    "read_json",
    "replay_run",
    "snapshot_directory",
    "validate_run",
]
