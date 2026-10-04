"""Interactive-command tests (setup, doctor) with mocked probes, plus reporting robustness."""

from __future__ import annotations

import json

import pytest

import beatspy.cli as cli
from beatspy.reporting.dashboard import collect_runs, render_dashboard


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("BEATSPY_HOME", str(tmp_path))
    return tmp_path


def async_probes(results):
    async def _probe(provider, model):
        return results

    return _probe


def run_setup(monkeypatch, answers: list[str], probes: dict | None = None) -> int:
    answers = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    monkeypatch.setattr(cli, "_secret_input", lambda prompt="": next(answers))
    monkeypatch.setattr(
        cli,
        "probe_capabilities",
        async_probes(probes or {"chat": (True, "pong"), "tools": (True, "tool call observed")}),
    )
    return cli.main(["setup"])


def test_setup_writes_config_and_secrets(home, monkeypatch):
    code = run_setup(
        monkeypatch,
        ["2", "http://localhost:8000/v1", "qwen2.5:7b", "sk-abc", "", "finnhub-key", "", "naive"],
    )
    assert code == 0
    config = (home / "config.toml").read_text()
    assert 'base_url = "http://localhost:8000/v1"' in config
    assert 'model = "qwen2.5:7b"' in config
    assert 'forecast_provider = "naive"' in config
    secrets = (home / "secrets.env").read_text()
    assert "BEATSPY_API_KEY=sk-abc" in secrets
    assert "BEATSPY_FINNHUB_API_KEY=finnhub-key" in secrets


def test_setup_blank_keys_keep_previous_secrets(home, monkeypatch):
    (home / "secrets.env").write_text("# BeatSPY secrets. Do not commit or share this file.\nBEATSPY_API_KEY=old-key\n")
    run_setup(monkeypatch, ["1", "", "llama3.1:8b", "", "", "", "", ""])
    secrets = (home / "secrets.env").read_text()
    assert "BEATSPY_API_KEY=old-key" in secrets  # blank prompt must not wipe the stored key


def test_setup_reports_failed_probe(home, monkeypatch):
    code = run_setup(
        monkeypatch,
        ["4", "", "gpt-4o-mini", "sk", "", "", "", ""],
        probes={"chat": (True, "pong"), "tools": (False, "no tool call observed")},
    )
    assert code == 1


def test_doctor_passes_with_mocked_probes(home, monkeypatch, capsys, tmp_path, scenario):
    from beatspy.data.freeze import freeze_snapshot

    monkeypatch.setattr(
        "beatspy.data.freeze.download_ohlc",
        lambda *a, **k: __import__("sys").modules["conftest"].make_prices(scenario.universe),
    )
    monkeypatch.chdir(tmp_path)
    freeze_snapshot(scenario, root=tmp_path / ".beatspy-data")
    monkeypatch.setattr(cli, "probe_capabilities", async_probes({"chat": (True, "pong"), "tools": (True, "ok")}))
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert out.count("PASS") >= 5


def test_doctor_fails_when_tool_calling_missing(home, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "probe_capabilities", async_probes({"chat": (True, "pong"), "tools": (False, "model never called a tool")})
    )
    assert cli.main(["doctor"]) == 1
    assert "FAIL" in capsys.readouterr().out


def test_collect_runs_skips_corrupt_dirs(tmp_path, capsys):
    good = tmp_path / "run-good"
    good.mkdir()
    (good / "run.json").write_text(
        json.dumps({"run_id": "run-good", "model": {"model": "m"}, "scenario": {"name": "s"}, "synthetic": False})
    )
    (good / "metrics.json").write_text(json.dumps({"total_return": 0.1, "excess_return_vs_spy": 0.1}))
    bad = tmp_path / "run-bad"
    bad.mkdir()
    (bad / "run.json").write_text("{not json")

    runs = collect_runs(tmp_path)
    assert [r["run_id"] for r in runs] == ["run-good"]

    corrupt_only = tmp_path / "corrupt-only"
    corrupt_only.mkdir()
    (corrupt_only / "run-bad").mkdir()
    (corrupt_only / "run-bad" / "run.json").write_text("{not json")
    with pytest.raises(RuntimeError, match="No runs found"):
        render_dashboard(corrupt_only, out_path=corrupt_only / "dash.html")
