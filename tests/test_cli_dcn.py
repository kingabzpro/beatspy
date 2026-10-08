"""CLI tests for the DCN benchmark.

Every test here is hermetic: no network, no live provider. Two arguments make
that possible — `--offline-fixture` (a scripted client plus generated prices) and
the `dcn_run_dir` fixture from conftest.py (a complete run produced by the real
runner offline).
"""

from __future__ import annotations

import json
import re
import shutil

import pytest

from beatspy.cli import _local_verify, build_parser, main
from beatspy.config import DECISION_MODELS, load_settings
from beatspy.dcn.clients import DcnError, DcnResult
from beatspy.schemas import NoulAnswer


class _FakeProbeClient:
    """Stands in for a provider during `doctor --probe`; never opens a socket."""

    provider = "openai_decisions"
    model = "fake-probe"
    endpoint = "https://fake.invalid"

    def __init__(self, *, result: DcnResult | None = None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.probes = 0

    async def decide(self, state, questions):
        self.probes += 1
        if self.error is not None:
            raise self.error
        return self.result

    async def close(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# setup / doctor / models / scenarios
# --------------------------------------------------------------------------- #


def test_models_lists_every_decision_model(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    for name, spec in DECISION_MODELS.items():
        assert name in out
        assert spec["provider"] in out
        assert f"{spec['cost_per_m_input']:.3f}" in out
    assert "input tokens only" in out


def test_doctor_offline_exits_zero_without_network(monkeypatch, capsys):
    def explode(*args, **kwargs):  # pragma: no cover - only runs if a probe escaped
        raise AssertionError("doctor without --probe must not build a client")

    monkeypatch.setattr("beatspy.dcn.clients.make_client", explode)
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    for name, spec in DECISION_MODELS.items():
        assert name in out
        assert spec["label"] in out
        assert spec["base_url"] in out
    # Report every model either way, but never assert WHICH way: whether a key
    # resolves depends on the machine, and a test that hard-codes the credential
    # state fails the moment someone configures one. Status lines carry ANSI
    # colour codes, including inside the brackets, so strip them before matching.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", out)
    reported = re.findall(r"\[(?:PASS|WARN)\] model ", plain)
    assert len(reported) == len(DECISION_MODELS)
    assert "No network calls were made" in plain


def test_doctor_probe_reports_the_probability(monkeypatch, capsys):
    client = _FakeProbeClient(
        result=DcnResult(answers={"noul_AAPL": NoulAnswer(noul=0.62)}, input_tokens=42, latency_s=0.5)
    )
    monkeypatch.setattr("beatspy.dcn.clients.make_client", lambda *a, **k: client)
    assert main(["doctor", "--probe"]) == 0
    out = capsys.readouterr().out
    assert out.count("probe ") == len(DECISION_MODELS)
    assert "noul_AAPL=0.6200" in out
    assert client.probes == len(DECISION_MODELS)


def test_doctor_probe_reports_the_exact_error(monkeypatch, capsys):
    client = _FakeProbeClient(error=DcnError("HTTP 401: invalid api key for clef-flash"))
    monkeypatch.setattr("beatspy.dcn.clients.make_client", lambda *a, **k: client)
    assert main(["doctor", "--probe"]) == 1
    err = capsys.readouterr().out
    assert "HTTP 401: invalid api key for clef-flash" in err
    assert "FAIL" in err


def test_doctor_probe_flags_a_missing_answer(monkeypatch, capsys):
    client = _FakeProbeClient(result=DcnResult(answers={}, missing_ids=["noul_AAPL"]))
    monkeypatch.setattr("beatspy.dcn.clients.make_client", lambda *a, **k: client)
    assert main(["doctor", "--probe"]) == 1
    assert "no answer for noul_AAPL" in capsys.readouterr().out


def test_setup_persists_the_decision_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BEATSPY_HOME", str(tmp_path / "home"))
    assert main(["setup", "--model", "clef-flash", "--api-key", "test-key"]) == 0
    out = capsys.readouterr().out
    assert "clef-flash" in out

    config = tmp_path / "home" / "config.toml"
    assert config.is_file()
    assert 'decision_model = "clef-flash"' in config.read_text(encoding="utf-8")
    assert load_settings().decision.decision_model == "clef-flash"
    assert "CLOUDFLARE_AUTH_TOKEN=test-key" in (tmp_path / "home" / "secrets.env").read_text(encoding="utf-8")


def test_setup_rejects_unknown_model(capsys):
    assert main(["setup", "--model", "not-a-model"]) == 1
    assert "unknown decision model" in capsys.readouterr().err


def test_scenarios_lists_builtins(capsys):
    assert main(["scenarios"]) == 0
    out = capsys.readouterr().out
    assert "2022-bear" in out
    assert "2020-covid" in out
    assert "2023-recovery" in out
    assert "benchmark" in out


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def test_run_rejects_unknown_model(capsys):
    assert main(["run", "--model", "not-a-model"]) == 1
    assert "unknown decision model" in capsys.readouterr().err


def test_run_offline_fixture_writes_a_complete_dcn_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--offline-fixture", "--scenario", "test-scenario", "--label", "fixture"]) == 0
    out = capsys.readouterr().out
    assert "no network" in out
    assert "Run complete" in out

    run_dirs = [p for p in (tmp_path / "results").iterdir() if (p / "run.json").is_file()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    for name in ("run.json", "metrics.json", "decisions.jsonl", "equity_curve.csv", "trades.csv", "events.jsonl"):
        assert (run_dir / name).is_file(), name
    assert (run_dir / "snapshot" / "prices.csv").is_file()

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == 3
    assert meta["benchmark"] == "dcn"
    assert meta["synthetic"] is False
    assert meta["decision"]["decision_model"] == "clef-flash"
    assert meta["decision"]["question_form"] == "twin"
    assert meta["label"] == "fixture"
    assert meta["scenario"]["name"] == "test-scenario"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["calibration"]["signals"]["noul"]["n"] > 0
    assert metrics["calibration"]["signals"]["choice"]["n"] > 0
    assert metrics["decisions"] == len((run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines())


def test_run_form_and_frequency_flags_reach_the_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--offline-fixture", "--scenario", "test-scenario", "--form", "noul", "--freq", "monthly"]) == 0
    run_dir = next(p for p in (tmp_path / "results").iterdir() if (p / "run.json").is_file())
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["decision"]["question_form"] == "noul"
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["calibration"]["signals"]["noul"]["n"] > 0
    assert "choice" not in metrics["calibration"]["signals"]


# --------------------------------------------------------------------------- #
# calibrate / verify
# --------------------------------------------------------------------------- #


def test_calibrate_prints_the_table_and_json(dcn_run_dir, capsys):
    assert main(["calibrate", str(dcn_run_dir)]) == 0
    out = capsys.readouterr().out
    for label in ("signal", "noul", "choice", "base rate", "brier", "skill", "log loss", "auc", "ece"):
        assert label in out
    assert "miscalibration gap" in out

    assert main(["calibrate", str(dcn_run_dir), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision_model"]
    assert payload["calibration"]["signals"]["noul"]["n"] > 0
    assert payload["run_id"]


def test_calibrate_resolves_a_run_id_prefix(tmp_path, monkeypatch, dcn_run_dir, capsys):
    # dcn_run_dir already lives under tmp_path/results, which is results_dir() here.
    monkeypatch.chdir(tmp_path)
    assert main(["calibrate", dcn_run_dir.name[:12]]) == 0
    assert dcn_run_dir.name in capsys.readouterr().out


def test_calibrate_unknown_run_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["calibrate", "does-not-exist"]) == 1
    assert "no run matches" in capsys.readouterr().err


def test_verify_accepts_a_clean_run_and_rejects_a_tampered_metric(dcn_run_dir, tmp_path, capsys):
    assert main(["verify", str(dcn_run_dir)]) == 0
    assert "PASS" in capsys.readouterr().out

    tampered = tmp_path / "tampered"
    shutil.copytree(dcn_run_dir, tampered)
    metrics = json.loads((tampered / "metrics.json").read_text(encoding="utf-8"))
    metrics["calibration"]["signals"]["noul"]["brier"] = 0.0001
    (tampered / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    assert main(["verify", str(tampered)]) == 1
    assert "FAIL" in capsys.readouterr().err


def test_local_verify_rebuilds_calibration_from_decisions(dcn_run_dir, tmp_path, capsys):
    """The reduced replay used when beatspy.dcn.verify is unavailable."""
    assert _local_verify(dcn_run_dir) == 0
    assert "PASS" in capsys.readouterr().out

    tampered = tmp_path / "tampered-local"
    shutil.copytree(dcn_run_dir, tampered)
    metrics = json.loads((tampered / "metrics.json").read_text(encoding="utf-8"))
    metrics["calibration"]["coverage"] = 0.5
    (tampered / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    assert _local_verify(tampered) == 1
    assert "coverage" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# demo / compare / report
# --------------------------------------------------------------------------- #


def test_demo_writes_synthetic_runs_and_compare_hides_them(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["demo", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out

    runs = [p for p in (tmp_path / "results").iterdir() if (p / "run.json").is_file()]
    assert len(runs) == 3
    for run_dir in runs:
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert meta["synthetic"] is True
        assert meta["benchmark"] == "dcn"
        assert meta["schema_version"] == 3
        assert meta["decision"]["decision_model"] in DECISION_MODELS
        assert metrics["calibration"]["signals"]["noul"]["brier"] is not None

    assert main(["compare"]) == 1
    assert "No runs found" in capsys.readouterr().out

    assert main(["compare", "--include-synthetic"]) == 0
    table = capsys.readouterr().out
    assert "excess" in table
    for model in ("gpt-6-luna", "jev-latest", "clef-flash"):
        assert model in table

    assert main(["compare", "--include-synthetic", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {row["decision_model"] for row in rows} == {"gpt-6-luna", "jev-latest", "clef-flash"}
    assert all(row["provider"] for row in rows)


def test_compare_reports_latest_run_per_model(tmp_path, monkeypatch, capsys):
    from beatspy.cli import _write_demo_runs

    _write_demo_runs(seed=3, out_root=tmp_path / "results")
    # A second, newer run for the same model must replace the first, not duplicate it.
    first = next(p for p in (tmp_path / "results").iterdir() if (p / "run.json").is_file())
    meta = json.loads((first / "run.json").read_text(encoding="utf-8"))
    meta["run_id"] = meta["run_id"] + "-newer"
    meta["created_utc"] = "2999-01-01T00:00:00+00:00"
    newer = tmp_path / "results" / meta["run_id"]
    shutil.copytree(first, newer)
    (newer / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["compare", "--include-synthetic"]) == 0
    table = capsys.readouterr().out
    model = meta["decision"]["decision_model"]
    assert table.count(model) == 1


def test_compare_skips_unreadable_runs_with_a_warning(tmp_path, monkeypatch, capsys):
    from beatspy.cli import _write_demo_runs

    _write_demo_runs(seed=4, out_root=tmp_path / "results")
    broken = tmp_path / "results" / "20260101-000000_broken_clef-flash_deadbeef"
    broken.mkdir(parents=True)
    (broken / "run.json").write_text(json.dumps({"run_id": "broken"}), encoding="utf-8")
    (broken / "metrics.json").write_text("{not json", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["compare", "--include-synthetic"]) == 0
    captured = capsys.readouterr()
    assert "skipping unreadable run" in captured.err
    assert "clef-flash" in captured.out


def test_report_renders_the_dashboard(tmp_path, monkeypatch, capsys):
    pytest.importorskip("beatspy.reporting.catalog")
    from beatspy.cli import _write_demo_runs

    _write_demo_runs(seed=5, out_root=tmp_path / "results")
    monkeypatch.chdir(tmp_path)
    assert main(["report"]) == 0
    out = capsys.readouterr().out
    assert "dashboard" in out.lower()
    assert (tmp_path / "results" / "dashboard" / "index.html").is_file()


# --------------------------------------------------------------------------- #
# parser surface
# --------------------------------------------------------------------------- #


def test_parser_keeps_the_hand_rolled_style():
    parser = build_parser()
    args = parser.parse_args(["run", "--model", "clef", "--scenario", "2022-bear", "--form", "rank"])
    assert args.model == "clef"
    assert args.scenario == "2022-bear"
    assert args.form == "rank"
    assert args.offline_fixture is False
    assert parser.parse_args(["run"]).form == "twin"
    assert parser.parse_args(["run", "--offline-fixture"]).offline_fixture is True
    assert parser.parse_args(["run", "--max-decisions", "6"]).max_decisions == 6
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--max-decisions", "0"])


def test_removed_agent_benchmark_surface_is_gone(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "submit" not in out
    assert "validate" not in out
    assert "scorecard" not in out
    assert "windows" not in out

    for command in ("submit", "validate", "scorecard", "windows"):
        with pytest.raises(SystemExit):
            main([command, "--help"])


def test_no_command_shows_help(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "calibrate" in out
    assert "demo" in out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    from beatspy import __version__

    assert __version__ in capsys.readouterr().out