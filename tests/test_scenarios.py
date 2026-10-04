"""Bundled scenario integrity and end-to-end runs over every bundled scenario."""

from __future__ import annotations

from datetime import date

import pytest

from beatspy.bench.runner import run_benchmark
from beatspy.scenarios import available_names, load_events, load_scenario
from conftest import FakeExecutor, make_prices

BUNDLED = ["2020-covid", "2022-bear", "2023-recovery"]


def test_all_bundled_scenarios_are_listed():
    for name in BUNDLED:
        assert name in available_names()


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
    assert metrics["requests"] == expected_decisions * 5
    assert metrics["invalid_outputs"] == 0
    # baselines must be present for every scenario
    assert set(metrics["baselines"]) == {"equal_weight", "sixty_forty", "momentum_12_1"}
    # momentum needs ~252 trading days of lookback, which the long price window provides
    assert metrics["baselines"]["momentum_12_1"] != 0.0 or metrics["total_return"] != 0.0
