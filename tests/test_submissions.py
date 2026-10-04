"""Replay and public trust boundaries, including rehashed tampering."""

import asyncio
import json

import pandas as pd
import pytest

from beatspy.artifacts import atomic_json, sha256
from beatspy.bench.runner import run_benchmark
from beatspy.bench.validation import validate_run
from beatspy.reporting.catalog import build_catalog, submit_run
from conftest import FakeExecutor, make_prices


@pytest.fixture
def run(tmp_path, monkeypatch, scenario, settings):
    monkeypatch.setenv("BEATSPY_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("beatspy.data.freeze.download_ohlc", lambda *a: make_prices(scenario.universe))
    return asyncio.run(
        run_benchmark(
            scenario, settings, executor=FakeExecutor(), out_root=tmp_path / "results", data_root=tmp_path / "cache"
        )
    )


def rehash(run, filename):
    meta = json.loads((run / "run.json").read_text())
    meta["artifact_hashes"][filename] = sha256(run / filename)
    atomic_json(run / "run.json", meta)


def test_complete_run_replays_and_stops_at_cutoff(run):
    meta = validate_run(run)
    assert meta["protocol_version"] == 3
    equity = pd.read_csv(run / "equity_curve.csv")
    trades = pd.read_csv(run / "trades.csv")
    assert equity.date.max() == meta["requested"]["end"]
    assert trades.exec_date.max() <= meta["requested"]["end"]


def test_rehashed_market_brief_tampering_fails(run):
    path = run / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["market_brief"]["horizon_trading_days"] = 999
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    rehash(run, "decisions.jsonl")
    with pytest.raises(ValueError, match="market brief replay mismatch"):
        validate_run(run)


@pytest.mark.parametrize(
    "filename", ["metrics.json", "trades.csv", "equity_curve.csv", "decisions.jsonl", "snapshot/prices.csv"]
)
def test_hash_tampering_fails(run, filename):
    with (run / filename).open("a") as stream:
        stream.write(" \n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_run(run)


@pytest.mark.parametrize("filename", ["metrics.json", "trades.csv", "equity_curve.csv", "decisions.jsonl"])
def test_rehashed_tampering_still_fails_replay(run, filename):
    path = run / filename
    if filename == "metrics.json":
        metrics = json.loads(path.read_text())
        metrics["total_return"] += 0.1
        atomic_json(path, metrics)
    elif filename == "decisions.jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["decision"]["allocations"][0]["weight"] = 0.99
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        frame = pd.read_csv(path)
        frame.loc[0, "price" if filename == "trades.csv" else "portfolio"] *= 2
        frame.to_csv(path, index=False)
    rehash(run, filename)
    with pytest.raises(ValueError, match="replay mismatch"):
        validate_run(run)


def test_synthetic_and_forged_trust_rejected(run):
    meta = json.loads((run / "run.json").read_text())
    meta["trust"] = "maintainer"
    atomic_json(run / "run.json", meta)
    with pytest.raises(ValueError, match="own trust"):
        validate_run(run)
    del meta["trust"]
    meta["synthetic"] = True
    atomic_json(run / "run.json", meta)
    with pytest.raises(ValueError, match="synthetic"):
        validate_run(run)


def test_submit_is_community_and_registry_is_bound_to_hash(run, tmp_path):
    root = tmp_path / "public"
    target = submit_run(run, root)
    rows = build_catalog(root)
    assert rows[0]["trust"] == "community"
    atomic_json(root / "maintainer-runs.json", {target.name: sha256(target / "run.json")})
    assert build_catalog(root)[0]["trust"] == "maintainer"
    with (target / "run.json").open("a") as stream:
        stream.write("\n")
    assert build_catalog(root)[0]["trust"] == "community"
    with pytest.raises(ValueError, match="already submitted"):
        submit_run(run, root)
    assert not (target / "snapshot").exists()
    assert (root / "snapshots").is_dir()


@pytest.mark.parametrize("key_env", ["BEATSPY_API_KEY", "CUSTOM_MODEL_TOKEN"])
def test_credentials_are_excluded_from_export(run, tmp_path, monkeypatch, key_env):
    monkeypatch.setenv("BEATSPY_API_KEY_ENV", key_env)
    monkeypatch.setenv(key_env, "sensitive-test-key")
    meta = json.loads((run / "run.json").read_text())
    meta["model"]["base_url"] = "https://user:sensitive-test-key@example.org/v1?token=sensitive-test-key"
    meta["label"] = "sensitive-test-key"
    atomic_json(run / "run.json", meta)
    target = submit_run(run, tmp_path / "public")
    exported = (target / "run.json").read_text()
    assert "sensitive-test-key" not in exported and "user:" not in exported
    assert "sensitive-test-key" in (run / "run.json").read_text()


def test_empty_public_catalog(tmp_path):
    assert build_catalog(tmp_path) == []
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["runs"] == [] and index["default_trust"] == "maintainer"


def test_oversized_metadata_rejected_before_parsing(run):
    with (run / "run.json").open("wb") as stream:
        stream.truncate(25_000_001)
    with pytest.raises(ValueError, match="oversized"):
        validate_run(run)
