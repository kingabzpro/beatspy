"""CLI tests for the commands that need no network or model."""

from __future__ import annotations

from beatspy.cli import main


def test_scenarios_command_lists_builtins(capsys):
    assert main(["scenarios"]) == 0
    out = capsys.readouterr().out
    assert "2022-bear" in out
    assert "2020-covid" in out
    assert "2023-recovery" in out


def test_demo_compare_and_report(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["demo", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out

    results = tmp_path / "results"
    run_dirs = [p for p in results.iterdir() if (p / "run.json").exists()]
    assert len(run_dirs) == 3
    assert (results / "dashboard/index.html").exists()

    # synthetic runs are excluded from the default leaderboard
    assert main(["compare"]) == 1
    assert main(["compare", "--include-synthetic"]) == 0
    table = capsys.readouterr().out
    assert "demo-alpha" in table
    assert "excess" in table

    assert main(["compare", "--include-synthetic", "--json"]) == 0
    assert "demo-beta" in capsys.readouterr().out

    assert main(["report"]) == 0
    assert "dashboard" in capsys.readouterr().out

    html = (results / "dashboard/index.html").read_text(encoding="utf-8")
    assert "Synthetic demo" in (results / "dashboard/app.js").read_text(encoding="utf-8")
    assert "Real model calls. Five trading agents. Frozen market data." in html


def test_report_specific_run(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    main(["demo", "--seed", "2"])
    results = tmp_path / "results"
    run_id = sorted(p.name for p in results.iterdir() if (p / "run.json").exists())[0]
    assert main(["report", "--run", run_id]) == 0
    assert main(["report", "--run", "does-not-exist"]) == 1


def test_compare_empty_results(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["compare"]) == 1
    assert "No runs found" in capsys.readouterr().out


def test_compare_one_window_latest_model_names(monkeypatch, capsys):
    from beatspy.reporting import dashboard

    def row(model, end, created, value):
        return {
            "model": model,
            "scenario": "recent",
            "start": "2026-07-05",
            "end": end,
            "created_utc": created,
            "synthetic": False,
            "metrics": {"total_return": value, "excess_return_vs_spy": value},
        }

    monkeypatch.setattr(
        dashboard,
        "collect_runs",
        lambda _: [
            row("vendor/model-a", "2026-10-02", "1", 0.1),
            row("vendor/model-a", "2026-10-02", "2", 0.2),
            row("vendor/model-b", "2026-10-02", "2", 0.15),
            row("vendor/old-model", "2025-12-31", "3", 0.9),
        ],
    )
    assert main(["compare"]) == 0
    table = capsys.readouterr().out
    assert table.count("model-a") == table.count("model-b") == 1
    assert "20.0%" in table and "15.0%" in table
    assert "vendor/" not in table and "old-model" in table and "group" not in table


def test_version_flag(capsys):
    with __import__("pytest").raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    from beatspy import __version__

    assert __version__ in capsys.readouterr().out


def test_no_command_shows_help(capsys):
    assert main([]) == 0
    assert "submit" in capsys.readouterr().out


def test_run_defaults_and_repeated_arguments():
    from beatspy.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["run", "--model", "a", "--model", "b", "--scenario", "2025-recent"])
    assert args.model == ["a", "b"] and args.scenario == ["2025-recent"]
    assert args.jobs == 2 and args.concurrency == 6
    assert parser.parse_args(["run", "--max-decisions", "60"]).max_decisions == 60
    assert not parser.parse_args(["run"]).web_research
    assert parser.parse_args(["run", "--web-research"]).web_research
    with __import__("pytest").raises(SystemExit):
        parser.parse_args(["run", "--jobs", "0"])


def test_web_research_requires_a_provider_key(monkeypatch, capsys):
    monkeypatch.setattr("beatspy.cli.provider_api_key", lambda *a: None)
    assert main(["run", "--web-research"]) == 1
    assert "OLOSTEP_API_KEY" in capsys.readouterr().err


def test_batch_failure_returns_nonzero_and_reports_completed_run(tmp_path, monkeypatch, capsys):
    import json

    from beatspy.config import Settings

    complete = tmp_path / "complete"
    complete.mkdir()
    (complete / "metrics.json").write_text(
        json.dumps(
            {
                "total_return": 0,
                "spy_total_return": 0,
                "excess_return_vs_spy": 0,
                "requests": 5,
                "input_tokens": 100,
                "output_tokens": 50,
            }
        )
    )

    async def batch(*args, **kwargs):
        return [complete, RuntimeError("provider failed")]

    monkeypatch.setattr("beatspy.cli.load_settings", Settings)
    monkeypatch.setattr("beatspy.bench.batch.run_batch", batch)
    assert main(["run"]) == 1
    output = capsys.readouterr()
    assert "Run complete" in output.out and "provider failed" in output.err
    assert (complete / "metrics.json").exists()
