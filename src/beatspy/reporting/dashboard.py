"""Prepare a standalone static dashboard; no HTML template or hosted backend."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from ..artifacts import RUN_FILES, atomic_json
from .catalog import comparison_group, sanitized, summary

log = logging.getLogger(__name__)


def static_dir() -> Path:
    packaged = Path(__file__).parent / "static"
    return packaged if (packaged / "index.html").exists() else Path(__file__).resolve().parents[3] / "dashboard"


def collect_runs(results_root: Path) -> list[dict]:
    runs = []
    if not results_root.exists():
        return runs
    for directory in sorted(results_root.iterdir()):
        if not directory.is_dir() or not (directory / "metrics.json").is_file():
            continue
        try:
            meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            meta.setdefault("run_id", directory.name)
            row = summary(meta, metrics, directory.name, "local")
            row.update(meta=meta, base_url=(meta.get("model") or {}).get("base_url", "?"))
            runs.append(row)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            log.warning("skipping unreadable run %s: %s", directory.name, exc)
    return runs


def render_dashboard(results_root: Path, run_id: str | None = None, out_path: Path | None = None) -> Path:
    runs = collect_runs(results_root)
    if not runs:
        raise RuntimeError("No runs found; run a benchmark or `beatspy demo` first")
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
        for name in ("run.json", *RUN_FILES):
            if (source / name).is_file():
                if name.endswith(".json"):
                    atomic_json(
                        directory / name, sanitized(json.loads((source / name).read_text(encoding="utf-8")), secrets)
                    )
                elif name.endswith(".jsonl"):
                    rows = [
                        sanitized(json.loads(line), secrets)
                        for line in (source / name).read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    (directory / name).write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
                else:
                    shutil.copyfile(source / name, directory / name)
        row["dir"] = f"runs/{row['dir']}"
        row["group"] = comparison_group(row.pop("meta"))
        row.pop("base_url", None)
    atomic_json(
        destination / "data/index.json", {"schema_version": 1, "default_trust": "all", "selected": run_id, "runs": runs}
    )
    return destination / "index.html"
