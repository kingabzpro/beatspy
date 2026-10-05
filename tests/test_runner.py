"""End-to-end runner tests: full benchmark loop with scripted agents and no network."""

from __future__ import annotations

import asyncio
import json

import pandas as pd

from beatspy.bench.runner import run_benchmark
from conftest import FakeExecutor, make_prices


def test_full_run_produces_all_artifacts(tmp_path, monkeypatch, scenario, settings):
    monkeypatch.setattr("beatspy.data.freeze.download_ohlc", lambda *a, **k: make_prices(scenario.universe))
    run_dir = asyncio.run(
        run_benchmark(
            scenario,
            settings,
            executor=FakeExecutor(),
            out_root=tmp_path,
            data_root=tmp_path,
        )
    )

    for name in (
        "run.json",
        "metrics.json",
        "equity_curve.csv",
        "trades.csv",
        "decisions.jsonl",
        "events.jsonl",
        "snapshot/prices.csv",
        "snapshot/MANIFEST.json",
    ):
        assert (run_dir / name).exists(), f"missing artifact {name}"

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["decisions"] == 3  # monthly decisions Jan-Mar 2022
    assert metrics["requests"] == 15  # 3 decisions x 5 agents
    assert metrics["invalid_outputs"] == 0
    assert set(metrics["baselines"]) == {"equal_weight", "sixty_forty", "momentum_12_1"}

    run_meta = json.loads((run_dir / "run.json").read_text())
    assert run_meta["synthetic"] is False
    assert run_meta["data"]["snapshot_id"]
    assert run_meta["scenario"]["name"] == scenario.name

    decisions = [json.loads(line) for line in (run_dir / "decisions.jsonl").read_text().splitlines() if line.strip()]
    assert len(decisions) == 3
    assert all("usage" in d and d["usage"] for d in decisions)

    # sanity: finite, plausible return for a defensive allocation in a falling market
    assert -1.0 < metrics["total_return"] < 10.0


def test_run_is_deterministic_for_scripted_agents(tmp_path, monkeypatch, scenario, settings):
    monkeypatch.setattr("beatspy.data.freeze.download_ohlc", lambda *a, **k: make_prices(scenario.universe))

    def run_once():
        return asyncio.run(
            run_benchmark(scenario, settings, executor=FakeExecutor(), out_root=tmp_path, data_root=tmp_path)
        )

    first = run_once()
    second = run_once()
    assert (first / "equity_curve.csv").read_bytes() == (second / "equity_curve.csv").read_bytes()
    assert (first / "trades.csv").read_bytes() == (second / "trades.csv").read_bytes()
    assert (first / "metrics.json").read_text() == (second / "metrics.json").read_text()


def test_invalid_decisions_hold_instead_of_liquidating(tmp_path, monkeypatch, scenario, settings):
    # regression: an unparseable PM decision used to sell everything ("hold" contract)
    monkeypatch.setattr("beatspy.data.freeze.download_ohlc", lambda *a, **k: make_prices(scenario.universe))
    executor = FakeExecutor()
    executor.outputs["portfolio_manager"] = "the model emitted no JSON at all"
    run_dir = asyncio.run(run_benchmark(scenario, settings, executor=executor, out_root=tmp_path, data_root=tmp_path))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["invalid_outputs"] == 3
    assert metrics["trades"] == 0  # never invested, never liquidated
    assert metrics["total_return"] == 0.0  # all cash throughout


def test_nested_fragments_never_liquidate_existing_holdings(tmp_path, monkeypatch, scenario, settings):
    from conftest import PM_JSON

    monkeypatch.setattr("beatspy.data.freeze.download_ohlc", lambda *a, **k: make_prices(scenario.universe))
    executor = FakeExecutor()
    malformed = '{"allocations": [{"ticker": "SPY", "weight": 0.3}], "rationale": "unfinished'
    outputs = iter([PM_JSON, malformed, malformed])
    executor.outputs["portfolio_manager"] = lambda _: next(outputs)
    run_dir = asyncio.run(run_benchmark(scenario, settings, executor=executor, out_root=tmp_path, data_root=tmp_path))
    metrics = json.loads((run_dir / "metrics.json").read_text())
    decisions = [json.loads(line) for line in (run_dir / "decisions.jsonl").read_text().splitlines()]
    holdings = [row["market_brief"]["current_portfolio_weights"] for row in decisions[1:]]
    assert metrics["invalid_outputs"] == 2
    assert all(weights.get("MSFT", 0) > 0 and weights.get("TLT", 0) > 0 for weights in holdings)
    import csv

    with (run_dir / "trades.csv").open(newline="") as stream:
        sells = [float(row["notional"]) for row in csv.DictReader(stream) if row["action"] == "sell"]
    assert all(notional < scenario.initial_capital * 0.1 for notional in sells)


def test_report_contains_provenance_and_leaderboard(tmp_path, monkeypatch, scenario, settings):
    monkeypatch.setattr("beatspy.data.freeze.download_ohlc", lambda *a, **k: make_prices(scenario.universe))
    run_dir = asyncio.run(
        run_benchmark(scenario, settings, executor=FakeExecutor(), out_root=tmp_path, data_root=tmp_path)
    )
    from beatspy.reporting.dashboard import render_dashboard

    path = render_dashboard(tmp_path, run_id=run_dir.name)
    html = path.read_text(encoding="utf-8")
    assert "Real model calls. Five trading agents. Frozen market data." in html
    assert "Leaderboard" in html
    assert "scenario.name" not in html  # no unrendered placeholders


def test_spy_attempt_cannot_create_trades_but_comparison_is_preserved(tmp_path, monkeypatch, scenario, settings):
    from beatspy.bench.validation import validate_run

    monkeypatch.setattr("beatspy.data.freeze.download_ohlc", lambda *a, **k: make_prices(scenario.universe))
    output = json.dumps(
        {"allocations": [{"ticker": "SPY", "weight": 0.3}, {"ticker": "MSFT", "weight": 0.3}], "cash_weight": 0.4}
    )
    directory = asyncio.run(
        run_benchmark(
            scenario,
            settings,
            executor=FakeExecutor({"portfolio_manager": output}),
            out_root=tmp_path / "results",
            data_root=tmp_path / "cache",
        )
    )
    meta = validate_run(directory)
    assert meta["protocol_version"] == 4
    trades = pd.read_csv(directory / "trades.csv")
    assert set(trades.ticker) == {"MSFT"}
    equity = pd.read_csv(directory / "equity_curve.csv")
    assert "spy_buy_hold" in equity.columns and equity.spy_buy_hold.nunique() > 1
    rows = [json.loads(line) for line in (directory / "decisions.jsonl").read_text().splitlines()]
    assert all("SPY" not in row["validated"]["weights"] for row in rows)
    assert all(any("SPY" in violation for violation in row["validated"]["violations"]) for row in rows)
