"""Bundled scenario integrity, plus an end-to-end DCN run over one of them.

A bundled scenario is a frozen price window plus a universe and cost model. This
checks that every bundled file is internally consistent and that the runner can
actually complete a run against one without a live price provider.
"""

from __future__ import annotations

import json

import pytest

from beatspy.scenarios import DEFAULT_SCENARIO, available_names, load_scenario, make_scenario

BUNDLED = [
    "2020-covid",
    "2022-bear",
    "2023-recovery",
    "2025-recent",
    "2026-recent",
    "2026-6m",
    "2026-ytd",
]


def test_all_bundled_scenarios_are_listed():
    listed = available_names()
    for name in BUNDLED:
        assert name in listed, f"{name} is missing from available scenarios"


@pytest.mark.parametrize("name", BUNDLED)
def test_bundled_scenarios_are_internally_consistent(name):
    from beatspy.data.calendar import resolve_scenario

    # Bundled scenarios may be rolling windows, so resolve dates like a real run.
    scenario = resolve_scenario(load_scenario(name))
    assert scenario.name == name
    assert scenario.title
    assert scenario.start < scenario.end
    assert scenario.benchmark in scenario.universe
    assert scenario.tradable, "a scenario needs an investable universe"
    assert len(scenario.universe) == len(set(scenario.universe))
    assert 0 < scenario.max_position_weight <= 1
    assert scenario.frequency in ("weekly", "monthly")
    assert 2 <= scenario.max_candidates <= 64


def test_default_scenario_exists():
    assert DEFAULT_SCENARIO in available_names()


def test_unknown_scenario_names_its_alternatives():
    with pytest.raises(FileNotFoundError) as excinfo:
        load_scenario("does-not-exist")
    assert "2026-ytd" in str(excinfo.value)


def test_latest_six_month_scenario_is_weekly_over_25_assets():
    scenario = load_scenario("2026-6m")
    assert scenario.recent_months == 6
    assert scenario.frequency == "weekly"
    assert len(scenario.tradable) == 25
    assert len(set(scenario.tradable)) == 25
    assert scenario.benchmark == "SPY"
    assert scenario.benchmark not in scenario.tradable, "SPY is comparison-only, never investable"
    # A weekly cadence caps decisions by default rather than running hundreds.
    assert scenario.max_decisions and scenario.max_decisions <= 30


def test_rolling_window_resolves_to_the_latest_completed_session():
    from datetime import UTC, datetime

    from beatspy.data.calendar import completed_session, resolve_scenario, subtract_months

    scenario = load_scenario("2026-6m")
    resolved = resolve_scenario(scenario, now=datetime(2026, 10, 8, 20, 0, tzinfo=UTC))
    assert resolved.end == completed_session(now=datetime(2026, 10, 8, 20, 0, tzinfo=UTC)).isoformat()
    assert resolved.start == subtract_months(completed_session(
        now=datetime(2026, 10, 8, 20, 0, tzinfo=UTC)), 6).isoformat()
    assert resolved.start < resolved.end
    assert resolved.through_latest is False


def test_subtract_months_clamps_short_months():
    from datetime import date

    from beatspy.data.calendar import subtract_months

    assert subtract_months(date(2026, 10, 8), 6) == date(2026, 4, 8)
    assert subtract_months(date(2026, 8, 31), 6) == date(2026, 2, 28)  # not February 31
    assert subtract_months(date(2026, 3, 31), 1) == date(2026, 2, 28)
    assert subtract_months(date(2026, 1, 15), 1) == date(2025, 12, 15)  # crosses the year


def test_make_scenario_applies_overrides():
    scenario = make_scenario(name="custom", tradable=["AAPL"], benchmark="AAPL", max_candidates=4)
    assert scenario.name == "custom"
    assert scenario.tradable == ["AAPL"]
    assert scenario.max_candidates == 4


def test_bundled_scenario_runs_end_to_end_offline(tmp_path, scenario, frozen_data, offline_client):
    """A bundled scenario must complete a DCN run with no provider and no network."""
    from conftest import run_offline

    active = load_scenario("2022-bear")
    run_dir = run_offline(
        tmp_path,
        active,
        frozen_data,
        offline_client,
        question_form="twin",
        scenario_overrides={"start": "2022-01-03", "end": "2022-03-31", "max_decisions": 3},
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["benchmark"] == "dcn"
    assert meta["schema_version"] == 3
    assert metrics["decisions"] == 3
    assert metrics["invalid_outputs"] == 0
    assert set(metrics["baselines"]) == {"equal_weight", "sixty_forty", "momentum_12_1"}
    assert metrics["calibration"]["signals"]["noul"]["n"] > 0