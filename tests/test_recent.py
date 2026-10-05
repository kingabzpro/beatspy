"""Recent-session boundaries, immutable refreshes, and strict data coverage."""

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from beatspy.data import freeze
from beatspy.data.calendar import completed_session, resolve_scenario
from beatspy.scenarios import load_scenario
from conftest import make_prices


@pytest.mark.parametrize(
    ("instant", "expected"),
    [
        ("2026-10-04T12:00:00+00:00", "2026-10-02"),  # Sunday
        ("2026-10-02T19:00:00+00:00", "2026-10-01"),  # before close
        ("2026-10-02T20:16:00+00:00", "2026-10-02"),
        ("2026-07-03T22:00:00+00:00", "2026-07-02"),  # observed Independence Day
        ("2025-11-28T18:16:00+00:00", "2025-11-28"),  # early close
        ("2025-11-28T18:05:00+00:00", "2025-11-26"),
    ],
)
def test_completed_session(instant, expected):
    assert completed_session(now=datetime.fromisoformat(instant)) == date.fromisoformat(expected)


def test_recent_windows_are_90_days_and_year_bounded():
    now = datetime(2026, 10, 4, tzinfo=UTC)
    for year, expected in [(2025, "2025-12-31"), (2026, "2026-10-02")]:
        scenario = resolve_scenario(load_scenario(f"{year}-recent"), now)
        assert scenario.end == expected
        assert date.fromisoformat(scenario.end) - date.fromisoformat(scenario.start) == timedelta(days=89)
    early = resolve_scenario(load_scenario("2026-recent"), datetime(2026, 1, 5, 22, tzinfo=UTC))
    assert early.start == "2026-01-01"
    assert early.end == "2026-01-05"
    with pytest.raises(ValueError, match="no completed"):
        completed_session(2027, now)


def test_fixed_scenario_stays_fixed():
    original = load_scenario("2022-bear")
    assert resolve_scenario(original) == original


def test_refresh_preserves_previous_snapshot(tmp_path, scenario, monkeypatch):
    prices = make_prices(scenario.universe)
    monkeypatch.setattr(freeze, "download_ohlc", lambda *a: prices)
    _, first = freeze.load_snapshot(scenario, tmp_path)
    original = (tmp_path / "snapshots" / first["snapshot_id"] / "prices.csv").read_bytes()
    prices = prices.copy()
    prices.loc[prices.ticker == "AAPL", "close"] *= 1.01
    _, second = freeze.load_snapshot(scenario, tmp_path, refresh=True)
    assert first["snapshot_id"] != second["snapshot_id"]
    assert (tmp_path / "snapshots" / first["snapshot_id"] / "prices.csv").read_bytes() == original


def test_new_cutoff_fetches_once_and_failures_do_not_replace_pointer(tmp_path, scenario, monkeypatch):
    calls = []
    monkeypatch.setattr(freeze, "download_ohlc", lambda *a: calls.append(a) or make_prices(scenario.universe))
    freeze.load_snapshot(scenario, tmp_path)
    freeze.load_snapshot(scenario, tmp_path)
    assert len(calls) == 1
    updated = scenario.model_copy(update={"end": "2022-04-01"})
    freeze.load_snapshot(updated, tmp_path)
    assert len(calls) == 2
    old = (tmp_path / scenario.name / "latest.json").read_bytes()

    def fail(*args):
        raise RuntimeError("download failed")

    monkeypatch.setattr(freeze, "download_ohlc", fail)
    with pytest.raises(RuntimeError, match="download failed"):
        freeze.load_snapshot(updated, tmp_path, refresh=True)
    assert (tmp_path / scenario.name / "latest.json").read_bytes() == old


@pytest.mark.parametrize("problem", ["missing", "gap", "all-ticker-gap", "nan", "negative", "duplicate"])
def test_bad_prices_rejected(tmp_path, scenario, monkeypatch, problem):
    prices = make_prices(scenario.universe)
    if problem == "missing":
        prices = prices[prices.ticker != "AAPL"]
    elif problem == "gap":
        prices = prices[~((prices.ticker == "AAPL") & (prices.date == "2022-02-01"))]
    elif problem == "all-ticker-gap":
        prices = prices[prices.date != "2022-02-01"]
    elif problem in ("nan", "negative"):
        index = prices.index[prices.date == "2022-01-03"][0]
        prices.loc[index, "close"] = float("nan") if problem == "nan" else -1
    else:
        prices = __import__("pandas").concat([prices, prices[prices.date == "2022-01-03"].iloc[[0]]])
    monkeypatch.setattr(freeze, "download_ohlc", lambda *a: prices)
    with pytest.raises(ValueError):
        freeze.load_snapshot(scenario, tmp_path)


def test_snapshot_id_excludes_fetch_time(tmp_path, scenario, monkeypatch):
    monkeypatch.setattr(freeze, "download_ohlc", lambda *a: make_prices(scenario.universe))
    first = freeze.freeze_snapshot(scenario, tmp_path)
    second = {**first, "fetched_at_utc": "2030-01-01T00:00:00Z"}
    assert freeze.snapshot_id(first) == freeze.snapshot_id(second)
    assert json.loads((tmp_path / scenario.name / "latest.json").read_text())["snapshot_id"] == first["snapshot_id"]


def test_ytd_resolves_latest_session_and_covers_sectors():
    scenario = resolve_scenario(load_scenario("2026-ytd"), datetime(2026, 10, 4, tzinfo=UTC))
    assert scenario.start == "2026-01-01" and scenario.end == "2026-10-02"
    assert len(scenario.tradable) == 25
    assert "SPY" not in scenario.tradable and {"TLT", "GLD"} <= set(scenario.tradable)
    assert {"META", "WMT", "V", "LLY", "GE", "SO", "LIN"} <= set(scenario.tradable)
    assert not scenario.allow_finnhub and not scenario.allow_web_search
    assert scenario.pipeline == "single"
    assert scenario.max_decisions == 12
    assert scenario.frequency == "weekly"
    assert not scenario.through_latest
    assert {"AMT", "LIN", "SO", "JNJ", "CAT", "PG", "JPM", "XOM", "GOOGL", "AMZN", "MSFT"} <= set(scenario.tradable)


def test_ytd_schedule_only_scores_2026_and_respects_limit():
    from beatspy.data.service import DataService

    scenario = resolve_scenario(load_scenario("2026-ytd"), datetime(2026, 10, 4, tzinfo=UTC))
    data = DataService(make_prices(["AAPL"], start="2025-01-01", end=scenario.end))
    days = data.decision_dates(
        date.fromisoformat(scenario.start),
        date.fromisoformat(scenario.end),
        scenario.frequency,
        max_decisions=scenario.max_decisions,
    )
    assert 1 < len(days) == 12
    assert {day.year for day in days} == {2026}
    assert days[0] == date(2026, 1, 2)
    assert days[-1] == date(2026, 9, 25)
    assert all(data.next_trading_day(day) <= date.fromisoformat(scenario.end) for day in days)
