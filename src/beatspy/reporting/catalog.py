"""Static result catalog and local PR preparation."""

import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from ..artifacts import RUN_FILES, atomic_json, sha256
from ..bench.validation import SAFE_ID, read_json, snapshot_directory, validate_run
from ..config import secret_values


def comparison_group(meta):
    scenario = {k: v for k, v in meta.get("scenario", {}).items() if k not in ("title", "description")}
    payload = {
        "scenario": scenario,
        "requested": meta.get("requested"),
        "bench": meta.get("bench"),
        "snapshot": meta.get("data", {}).get("snapshot_id"),
        "capabilities": meta.get("capabilities"),
        "protocol": meta.get("protocol_version", "legacy"),
        "synthetic": meta.get("synthetic", False),
        "temperature": (
            float(meta["model"]["temperature"]) if meta.get("model", {}).get("temperature") is not None else None
        ),
        "max_turns": meta.get("model", {}).get("max_turns"),
    }
    if (effort := meta.get("model", {}).get("reasoning_effort")) is not None:
        payload["reasoning_effort"] = effort
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def summary(meta, metrics, directory, trust="replayed"):
    scenario = meta.get("scenario") or {}
    requested = meta.get("requested") or {}
    start = requested.get("start") or scenario.get("start", "")
    end = requested.get("end") or scenario.get("end", "")
    result = {
        "run_id": meta["run_id"],
        "dir": directory,
        "model": (meta.get("model") or {}).get("model", "?"),
        "scenario": scenario.get("name", "?"),
        "start": start,
        "end": end,
        "year": end[:4],
        "data_cutoff": meta.get("data", {}).get("end", end),
        "snapshot_id": meta.get("data", {}).get("snapshot_id"),
        "created_utc": meta.get("created_utc", ""),
        "synthetic": bool(meta.get("synthetic")),
        "trust": "synthetic" if meta.get("synthetic") else trust,
        "label": meta.get("label"),
        "group": comparison_group(meta),
        "metrics": metrics,
    }
    if (effort := meta.get("model", {}).get("reasoning_effort")) is not None:
        result["reasoning_effort"] = effort
    # Six-month family context: lets the dashboard group versions without
    # inferring them from the scenario name.
    if scenario.get("require_min_names", 1) > 1:
        result["min_names"] = scenario["require_min_names"]
    if scenario.get("first_only"):
        result["first_only"] = True
    if groups := scenario.get("universe_groups"):
        result["universe_groups"] = {name: len(tickers) for name, tickers in groups.items()}
    if assets := scenario.get("score_assets"):
        result["score_assets"] = assets
    return result


def build_catalog(data_root: Path, *, public_key: Path | None = None, validate=True):
    if any((data_root / name).is_symlink() for name in ("runs", "snapshots")):
        raise ValueError("catalog directories must not be symlinks")
    from ..bench.verification import verify_result

    public_key = public_key or Path(".github/verification-key.pem")
    runs = []
    identities = set()
    for directory in sorted((data_root / "runs").glob("*")):
        if not directory.is_dir():
            continue
        meta = validate_run(directory) if validate else read_json(directory / "run.json")
        metrics = read_json(directory / "metrics.json")
        row = summary(meta, metrics, f"runs/{directory.name}", "replayed")
        if (directory / "verification.json").exists():
            verified = verify_result(directory, public_key.read_bytes())
            if verified["verification_id"] in identities:
                raise ValueError("duplicate verification ID")
            identities.add(verified["verification_id"])
            row.update(trust="verified", verification_id=verified["verification_id"])
        runs.append(row)
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


def submit_run(run_dir: Path, data_root: Path) -> Path:
    meta = validate_run(run_dir)
    if not SAFE_ID.fullmatch(meta["run_id"]):
        raise ValueError("unsafe run id")
    target = data_root / "runs" / meta["run_id"]
    if target.exists():
        raise ValueError("run already submitted; existing public artifacts are immutable")
    secrets = secret_values()
    source_snapshot = snapshot_directory(run_dir, meta)
    snapshot = data_root / "snapshots" / meta["data"]["snapshot_id"]
    if not snapshot.exists():
        shutil.copytree(source_snapshot, snapshot)
    target.mkdir(parents=True)
    try:
        for name in RUN_FILES:
            if name.endswith(".json"):
                atomic_json(target / name, sanitized(read_json(run_dir / name), secrets))
            elif name.endswith(".jsonl"):
                rows = [
                    sanitized(json.loads(line), secrets)
                    for line in (run_dir / name).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                (target / name).write_text(
                    "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8"
                )
            else:
                shutil.copyfile(run_dir / name, target / name)
        url = urlsplit(meta["model"]["base_url"])
        host = url.hostname or ""
        if url.port:
            host += f":{url.port}"
        meta["model"]["base_url"] = urlunsplit((url.scheme, host, url.path, "", ""))
        meta = sanitized(meta, secrets)
        meta["artifact_hashes"] = {name: sha256(target / name) for name in RUN_FILES}
        meta["artifact_hashes"].update(
            {f"snapshot/{name}": sha256(snapshot / name) for name in ("prices.csv", "MANIFEST.json")}
        )
        atomic_json(target / "run.json", meta)
        validate_run(target)
        build_catalog(data_root)
    except Exception:
        shutil.rmtree(target)
        raise
    return target
