"""Prepare a standalone static dashboard; no HTML template or hosted backend.

The site is unchanged: the same static ``dashboard/`` assets, the same
"local reports cannot overwrite the public dashboard" guard. Only the payload
changed, because this benchmark is about decision-model calibration.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

from ..artifacts import RUN_FILES, atomic_json
from .catalog import summary

log = logging.getLogger(__name__)

SNAPSHOT_ID = re.compile(r"[a-f0-9]{64}")
COPIED_FILES = ("run.json", *RUN_FILES)


def static_dir() -> Path:
    packaged = Path(__file__).parent / "static"
    return packaged if (packaged / "index.html").exists() else Path(__file__).resolve().parents[3] / "dashboard"


def collect_runs(results_root: Path) -> list[dict]:
    """Every readable DCN run directory under ``results_root``, as catalog rows."""
    runs = []
    if not results_root.exists():
        return runs
    for directory in sorted(results_root.iterdir()):
        if not directory.is_dir() or not (directory / "metrics.json").is_file():
            continue
        try:
            meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            if meta.get("benchmark") != "dcn":
                log.warning("skipping non-dcn run %s", directory.name)
                continue
            meta.setdefault("run_id", directory.name)
            row = summary(meta, metrics, directory.name, "local")
            row.update(meta=meta, endpoint=(meta.get("decision") or {}).get("endpoint", "?"))
            runs.append(row)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            log.warning("skipping unreadable run %s: %s", directory.name, exc)
    return runs


def _copy_json(source: Path, target: Path, secrets) -> None:
    from .catalog import sanitized

    atomic_json(target, sanitized(json.loads(source.read_text(encoding="utf-8")), secrets))


def _copy_decision_lines(source: Path, target: Path, secrets) -> None:
    from .catalog import sanitized

    rows = [
        sanitized(json.loads(line), secrets)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def render_dashboard(results_root: Path, run_id: str | None = None, out_path: Path | None = None) -> Path:
    runs = collect_runs(results_root)
    if not runs:
        raise RuntimeError("No runs found; record a DCN run first (`beatspy run`)")
    if run_id and not any(row["run_id"] == run_id for row in runs):
        raise ValueError(f"unknown run: {run_id}")
    destination = out_path.parent if out_path else results_root / "dashboard"
    if destination.resolve() == static_dir().resolve():
        raise ValueError("local reports cannot overwrite the public dashboard")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copyfile(static_dir() / name, destination / name)
    for name in ("favicon.ico", "site.webmanifest", "robots.txt", "sitemap.xml"):
        shutil.copyfile(static_dir() / name, destination / name)
    shutil.copytree(static_dir() / "assets", destination / "assets", dirs_exist_ok=True)
    from ..config import secret_values

    secrets = secret_values()
    for row in runs:
        source = results_root / row["dir"]
        directory = destination / "data/runs" / row["dir"]
        directory.mkdir(parents=True, exist_ok=True)
        for name in COPIED_FILES:
            if not (source / name).is_file():
                continue
            if name.endswith(".json"):
                _copy_json(source / name, directory / name, secrets)
            elif name.endswith(".jsonl"):
                _copy_decision_lines(source / name, directory / name, secrets)
            else:
                shutil.copyfile(source / name, directory / name)
        snapshot_id = row.get("snapshot_id")
        snapshot = source / "snapshot"
        if snapshot.is_dir() and isinstance(snapshot_id, str) and SNAPSHOT_ID.fullmatch(snapshot_id):
            target = destination / "data" / "snapshots" / snapshot_id
            if not target.exists():
                shutil.copytree(snapshot, target)
            row["snapshot_dir"] = f"snapshots/{snapshot_id}"
        row["dir"] = f"runs/{row['dir']}"
        row.pop("meta", None)
        row.pop("endpoint", None)
    atomic_json(
        destination / "data/index.json", {"schema_version": 1, "default_trust": "all", "selected": run_id, "runs": runs}
    )
    return destination / "index.html"