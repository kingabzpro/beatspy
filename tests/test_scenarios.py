"""Bundled scenario integrity and end-to-end runs over every bundled scenario."""

from __future__ import annotations

from datetime import date

import pytest

from beatspy.bench.runner import run_benchmark
from beatspy.scenarios import available_names, load_events, load_scenario
from conftest import FakeExecutor, make_prices

BUNDLED = ["2020-covid", "2022-bear", "2023-recovery"]
SIX_MONTH = ["2026-6m-buyhold", "2026-6m-monthly", "2026-6m-weekly"]
SIX_MONTH_WINDOW = ("2026-04-06", "2026-10-02")


def test_all_bundled_scenarios_are_listed():
    for name in BUNDLED:
        assert name in available_names()


@pytest.mark.parametrize("name", SIX_MONTH)
def test_six_month_family_shares_one_window_and_universe(name):
    """The three versions must differ only by cadence, so results are comparable."""
    scenario = load_scenario(name)
    assert (scenario.start, scenario.end) == SIX_MONTH_WINDOW
    assert scenario.protocol_version == 9
    assert scenario.benchmark == "SPY"
    # SPY stays comparison-only in every version, so no model can buy the index.
    assert scenario.benchmark not in scenario.tradable
    # At least 20 individual stocks are always available to choose from.
    assert len(scenario.universe_groups["stocks"]) >= 20
    assert {
        "GLD",
        "USO",
        "TLT",
    } <= set(scenario.universe_groups["hedges"])
    assert scenario.allow_finnhub is False and scenario.allow_web_search is False


@pytest.mark.parametrize("name", SIX_MONTH)
def test_six_month_universe_matches_across_versions(name):
    """A shared price set keeps SPY and every baseline line reconcilable."""
    baseline = load_scenario("2026-6m-weekly")
    assert set(load_scenario(name).universe) == set(baseline.universe)


def test_buyhold_version_originates_its_own_portfolio():
    scenario = load_scenario("2026-6m-buyhold")
    assert scenario.first_only is True
    assert scenario.reference_kind == "none"
    assert scenario.perf_feedback is False
    # 22 funded names must be feasible: the per-name cap has to leave room for them.
    assert scenario.require_min_names >= 20
    assert scenario.require_min_names * scenario.max_position_weight >= 0.8


def test_monthly_and_weekly_versions_use_their_own_cadence():
    monthly = load_scenario("2026-6m-monthly")
    weekly = load_scenario("2026-6m-weekly")
    assert monthly.frequency == "monthly" and monthly.first_only is False
    assert weekly.frequency == "weekly"
    assert weekly.reference_kind == "core"
    # Hedge assets are scored on their own so an oil/gold sleeve can be judged.
    assert {"GLD", "USO"} <= set(weekly.score_assets)


@pytest.mark.parametrize("name", BUNDLED)
def test_bundled_scenario_and_events_are_consistent(name):
    scenario = load_scenario(name)
    assert scenario.start < scenario.end
    assert scenario.benchmark in scenario.universe
    assert 0 < scenario.max_position_weight <= 1
    assert scenario.frequency in ("weekly", "monthly")
    for event in load_events(name):
        parsed = date.fromisoformat(event.date)  # every event date must be ISO
        assert date.fromisoformat(scenario.start) - __import__("datetime").timedelta(days=366) <= parsed
        assert parsed <= date.fromisoformat(scenario.end)


@pytest.mark.parametrize(
    ("name", "window", "expected_decisions"),
    [
        ("2022-bear", ("2022-01-03", "2022-03-31"), 3),  # monthly
        ("2020-covid", ("2020-02-03", "2020-02-28"), 4),  # skip cutoff-day orders that cannot fill
        ("2023-recovery", ("2023-01-03", "2023-03-31"), 3),  # monthly, 2023 universe with META
        ("2025-recent", ("2025-10-03", "2025-12-31"), 3),
        ("2026-recent", ("2026-07-05", "2026-10-02"), 4),
        ("2026-6m-buyhold", ("2026-04-06", "2026-10-02"), 1),
        ("2026-6m-monthly", ("2026-04-06", "2026-10-02"), 7),
        # 24 Fridays fall in the window; 23 can fill, plus the opening session.
        ("2026-6m-weekly", ("2026-04-06", "2026-10-02"), 24),
    ],
)
def test_every_bundled_scenario_runs_end_to_end(tmp_path, monkeypatch, settings, name, window, expected_decisions):
    scenario = load_scenario(name)
    monkeypatch.setattr(
        "beatspy.data.freeze.download_ohlc",
        lambda *a, **k: make_prices(scenario.universe, start="2019-09-03", end="2026-10-02", seed=11),
    )
    run_dir = __import__("asyncio").run(
        run_benchmark(
            scenario,
            settings,
            start=window[0],
            end=window[1],
            executor=FakeExecutor(),
            out_root=tmp_path,
            data_root=tmp_path,
        )
    )
    metrics = __import__("json").loads((run_dir / "metrics.json").read_text())
    assert metrics["decisions"] == expected_decisions
    # A single-call scenario makes one request per decision; the five-agent
    # pipeline makes one per role.
    per_decision = 1 if scenario.pipeline == "single" else 5
    assert metrics["requests"] == expected_decisions * per_decision
    assert metrics["invalid_outputs"] == 0
    # baselines must be present for every scenario
    assert {"equal_weight", "sixty_forty", "momentum_12_1"} <= set(metrics["baselines"])
    # momentum needs ~252 trading days of lookback, which the long price window provides
    assert metrics["baselines"]["momentum_12_1"] != 0.0 or metrics["total_return"] != 0.0


def test_weekly_version_scores_its_hedge_assets(tmp_path, monkeypatch, settings):
    """A hedge sleeve's own market return must be replayable, not asserted."""
    scenario = load_scenario("2026-6m-weekly")
    monkeypatch.setattr(
        "beatspy.data.freeze.download_ohlc",
        lambda *a, **k: make_prices(scenario.universe, start="2025-03-01", end="2026-10-02", seed=13),
    )
    run_dir = __import__("asyncio").run(
        run_benchmark(
            scenario,
            settings,
            start="2026-04-06",
            end="2026-10-02",
            executor=FakeExecutor(),
            out_root=tmp_path,
            data_root=tmp_path,
        )
    )
    metrics = __import__("json").loads((run_dir / "metrics.json").read_text())
    for ticker in scenario.score_assets:
        assert ticker in metrics["baselines"], f"{ticker} was not scored"
    equity = (run_dir / "equity_curve.csv").read_text().splitlines()[0].split(",")
    assert set(scenario.score_assets) <= set(equity)
