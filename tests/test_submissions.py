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


def test_capped_run_replays_full_period_and_rejects_schedule_tampering(tmp_path, monkeypatch, scenario, settings):
    monkeypatch.setenv("BEATSPY_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("beatspy.data.freeze.download_ohlc", lambda *a: make_prices(scenario.universe))
    scenario.max_decisions = 2
    directory = asyncio.run(
        run_benchmark(
            scenario, settings, executor=FakeExecutor(), out_root=tmp_path / "results", data_root=tmp_path / "cache"
        )
    )
    meta = validate_run(directory)
    assert meta["scenario"]["max_decisions"] == 2
    assert len((directory / "decisions.jsonl").read_text().splitlines()) == 2
    assert pd.read_csv(directory / "equity_curve.csv").date.max() == scenario.end
    meta["scenario"]["max_decisions"] = 3
    atomic_json(directory / "run.json", meta)
    with pytest.raises(ValueError, match="schedule"):
        validate_run(directory)


@pytest.mark.parametrize(
    "author,artifact,expected", [("owner", False, "true"), ("visitor", False, None), ("owner", True, "false")]
)
def test_workflow_reset_cannot_accept_unsigned_artifacts(tmp_path, monkeypatch, author, artifact, expected):
    import re
    import textwrap
    from pathlib import Path

    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/results.yml").read_text()
    block = re.search(r"          python - <<'PY'\r?\n(.*?)          PY", workflow, re.DOTALL)
    code = compile(textwrap.dedent(block[1]), "owner-reset", "exec")
    root = tmp_path / "submitted/dashboard/data"
    root.mkdir(parents=True)
    atomic_json(root / "index.json", {"schema_version": 1, "default_trust": "all", "runs": []})
    if artifact:
        directory = root / "runs/unsigned"
        directory.mkdir(parents=True)
        atomic_json(directory / "run.json", {})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REQUEST_AUTHOR", author)
    monkeypatch.setenv("REPOSITORY_OWNER", "owner")
    output = tmp_path / "workflow-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    if expected is None:
        with pytest.raises(SystemExit, match="Only the repository owner"):
            exec(code, {})
        assert not output.exists()
    else:
        exec(code, {})
        assert output.read_text().strip() == f"is_reset={expected}"


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


def test_unsigned_export_never_claims_verified(run, tmp_path):
    root = tmp_path / "public"
    target = submit_run(run, root)
    assert build_catalog(root)[0]["trust"] == "replayed"
    with pytest.raises(ValueError, match="already submitted"):
        submit_run(run, root)
    assert not (target / "snapshot").exists()
    assert (root / "snapshots").is_dir()


def test_signature_binds_release_metadata_and_all_artifacts(run, tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from beatspy.bench.verification import code_digest, sign_result, verify_result

    root = tmp_path / "public"
    target = submit_run(run, root)
    key = Ed25519PrivateKey.generate()
    private = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    public_path = tmp_path / "public.pem"
    public_path.write_bytes(public)
    envelope = sign_result(target, private, "https://github.com/example/actions/runs/1")
    payload = verify_result(target, public)
    assert payload == envelope["payload"]
    assert build_catalog(root, public_key=public_path)[0]["trust"] == "verified"
    with pytest.raises(ValueError, match="unapproved benchmark"):
        verify_result(target, public, release=("fake", code_digest()))
    other = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    with pytest.raises(ValueError, match="invalid result signature"):
        verify_result(target, other)
    meta = json.loads((target / "run.json").read_text())
    meta["model"]["model"] = "forged-model"
    atomic_json(target / "run.json", meta)
    with pytest.raises(ValueError, match="run hash"):
        verify_result(target, public)
    envelope["payload"]["code_sha256"] = "forged"
    atomic_json(target / "verification.json", envelope)
    with pytest.raises(ValueError, match="invalid result signature"):
        verify_result(target, public)


def test_local_process_cannot_use_trusted_signer(monkeypatch):
    from beatspy.bench.trusted import request_model, require_trusted_checkout

    monkeypatch.delenv("GITHUB_WORKFLOW_REF", raising=False)
    with pytest.raises(ValueError, match="trusted main-branch"):
        require_trusted_checkout()
    with pytest.raises(ValueError, match="JSON benchmark request"):
        request_model({"issue": {"body": "execute arbitrary commands"}})


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
    assert index["runs"] == [] and index["default_trust"] == "all"


def test_oversized_metadata_rejected_before_parsing(run):
    with (run / "run.json").open("wb") as stream:
        stream.truncate(25_000_001)
    with pytest.raises(ValueError, match="oversized"):
        validate_run(run)
