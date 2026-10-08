"""Static result catalog for decision-model (DCN) runs.

The signed submission path went away with the five-agent benchmark, so this
module only *reads* recorded runs. A catalog row is a flat, comparable summary of
one run: the decision model and provider that answered, the typed question form it
answered, the portfolio and calibration numbers from ``metrics.json``, and a
``comparison_group`` hash that marks runs as comparable.

Nothing here imports the replay validator: the catalog has to stay readable while
a run is only partially written, and readers must not depend on an in-flight
module.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..artifacts import atomic_json

# Run ids double as directory names, so they are limited to a safe subset.
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")

# The noul signal is the one that sizes the portfolio, so it supplies the
# headline calibration numbers in a catalog row. The other signals stay in
# ``metrics.calibration`` and in the per-signal breakdown.
PRIMARY_SIGNAL = "noul"

# Files a catalogued run directory has to carry. ``run.json`` is added separately.
CATALOG_FILES = ("metrics.json",)

MAX_ARTIFACT_BYTES = 25_000_000


def read_json(path: Path):
    """Read a JSON artifact, rejecting symlinks, oversized files and NaN."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"missing, oversized, or unsafe JSON artifact: {path.name}")

    def invalid(value):
        raise ValueError(f"invalid JSON number: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def validate_run_meta(run_dir: Path) -> dict:
    """Structural check of a recorded DCN run directory; returns its run.json.

    This is deliberately not a replay check: it proves the directory is a
    well-formed schema-3 DCN run, nothing about its scores.
    """
    run_dir = Path(run_dir)
    if run_dir.is_symlink() or run_dir.parent.is_symlink():
        raise ValueError("run symlinks are not allowed")
    meta = read_json(run_dir / "run.json")
    if meta.get("benchmark") != "dcn":
        raise ValueError("only dcn runs can be catalogued")
    if meta.get("schema_version") != 3:
        raise ValueError("only schema_version 3 dcn runs can be catalogued")
    run_id = meta.get("run_id")
    if not isinstance(run_id, str) or not SAFE_ID.fullmatch(run_id):
        raise ValueError("invalid run id")
    if run_id != run_dir.name:
        raise ValueError("run id does not match its directory name")
    for name in CATALOG_FILES:
        read_json(run_dir / name)
    return meta


def comparison_group(meta) -> str:
    """Hash of everything two runs must share before their scores are comparable.

    The decision model is deliberately absent: comparing models under one frozen
    scenario is the point of the group.
    """
    scenario = {k: v for k, v in (meta.get("scenario") or {}).items() if k not in ("title", "description")}
    decision = meta.get("decision") or {}
    payload = {
        "scenario": scenario,
        "requested": meta.get("requested"),
        "bench": meta.get("bench"),
        "snapshot": (meta.get("data") or {}).get("snapshot_id"),
        "protocol": meta.get("dcn_protocol", meta.get("protocol_version", "legacy")),
        "synthetic": meta.get("synthetic", False),
        "question_form": decision.get("question_form"),
        "policy": decision.get("policy"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def calibration_row(metrics) -> dict:
    """Pull the headline calibration numbers out of a metrics dict."""
    calibration = (metrics or {}).get("calibration") or {}
    signals = calibration.get("signals") or {}
    primary = signals.get(PRIMARY_SIGNAL) or {}
    row = {
        "brier": primary.get("brier"),
        "brier_skill_score": primary.get("brier_skill_score"),
        "auc": primary.get("auc"),
        "ece": primary.get("expected_calibration_error"),
        "max_calibration_error": primary.get("max_calibration_error"),
        "calibration_gap": calibration.get("miscalibration_gap"),
    }
    for name, stats in signals.items():
        stats = stats or {}
        row[f"{name}_brier"] = stats.get("brier")
        row[f"{name}_ece"] = stats.get("expected_calibration_error")
    return row


def summary(meta, metrics, directory, trust="replayed") -> dict:
    """Flatten one recorded run into a catalog row."""
    scenario = meta.get("scenario") or {}
    requested = meta.get("requested") or {}
    decision = meta.get("decision") or {}
    metrics = metrics or {}
    start = requested.get("start") or scenario.get("start", "")
    end = requested.get("end") or scenario.get("end", "")
    row = {
        "run_id": meta["run_id"],
        "dir": directory,
        "model": decision.get("decision_model", "?"),
        "model_label": decision.get("label"),
        "provider": decision.get("provider", "?"),
        "question_form": decision.get("question_form", "?"),
        "policy": decision.get("policy") or {},
        "pricing": decision.get("pricing") or {},
        "scenario": scenario.get("name", "?"),
        "start": start,
        "end": end,
        "year": end[:4],
        "data_cutoff": (meta.get("data") or {}).get("end", end),
        "snapshot_id": (meta.get("data") or {}).get("snapshot_id"),
        "created_utc": meta.get("created_utc", ""),
        "synthetic": bool(meta.get("synthetic")),
        "trust": "synthetic" if meta.get("synthetic") else trust,
        "label": meta.get("label"),
        "group": comparison_group(meta),
        "decisions": metrics.get("decisions"),
        "invalid_outputs": metrics.get("invalid_outputs"),
        "answer_coverage": metrics.get("answer_coverage"),
        "mean_latency_s": metrics.get("mean_latency_s"),
        "estimated_cost_usd": metrics.get("estimated_cost_usd"),
        "metrics": metrics,
    }
    row.update(calibration_row(metrics))
    return row


def build_catalog(data_root: Path, *, validate: bool = True):
    """Rebuild ``index.json`` for every run under ``data_root/runs``."""
    data_root = Path(data_root)
    if any((data_root / name).is_symlink() for name in ("runs", "snapshots")):
        raise ValueError("catalog directories must not be symlinks")
    runs = []
    for directory in sorted((data_root / "runs").glob("*")):
        if not directory.is_dir():
            continue
        meta = validate_run_meta(directory) if validate else read_json(directory / "run.json")
        metrics = read_json(directory / "metrics.json")
        runs.append(summary(meta, metrics, f"runs/{directory.name}", "replayed"))
    atomic_json(data_root / "index.json", {"schema_version": 1, "default_trust": "all", "runs": runs})
    return runs


def sanitized(value, secrets):
    if isinstance(value, dict):
        return {
            key: sanitized(child, secrets)
            for key, child in value.items()
            if key.lower() not in ("api_key", "authorization", "password", "secrets", "secret", "access_token")
        }
    if isinstance(value, list):
        return [sanitized(child, secrets) for child in value]
    if isinstance(value, str):
        for secret in secrets:
            if len(secret) >= 6:
                value = value.replace(secret, "[REDACTED]")
    return value