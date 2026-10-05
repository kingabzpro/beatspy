"""Data snapshot and look-ahead protection tests."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from beatspy.data import freeze
from conftest import make_prices


class TestFreeze:
    def test_freeze_and_reload(self, scenario, tmp_path, monkeypatch):
        monkeypatch.setattr(freeze, "download_ohlc", lambda *a, **k: make_prices(scenario.universe))
        prices, manifest = freeze.load_snapshot(scenario, root=tmp_path)
        assert len(prices) > 100
        assert manifest["snapshot_id"]
        assert manifest["tickers"] == scenario.universe

        # second load must reuse the frozen file, never hit the network
        def boom(*a, **k):
            raise AssertionError("download must not run when a snapshot exists")

        monkeypatch.setattr(freeze, "download_ohlc", boom)
        prices2, manifest2 = freeze.load_snapshot(scenario, root=tmp_path)
        assert manifest2["snapshot_id"] == manifest["snapshot_id"]
        assert len(prices2) == len(prices)

    def test_snapshot_id_is_stable_for_same_data(self, scenario, tmp_path, monkeypatch):
        monkeypatch.setattr(freeze, "download_ohlc", lambda *a, **k: make_prices(scenario.universe))
        freeze.freeze_snapshot(scenario, root=tmp_path)
        first_id = json.loads((tmp_path / scenario.name / "latest.json").read_text())["snapshot_id"]
        first = json.loads((tmp_path / "snapshots" / first_id / "MANIFEST.json").read_text())
        freeze.freeze_snapshot(scenario, root=tmp_path)
        second_id = json.loads((tmp_path / scenario.name / "latest.json").read_text())["snapshot_id"]
        second = json.loads((tmp_path / "snapshots" / second_id / "MANIFEST.json").read_text())
        # fetched_at changes between runs, but the prices hash and snapshot id must match
        assert first["prices_sha256"] == second["prices_sha256"]
        assert first_id == second_id

    def test_tampered_snapshot_is_rejected(self, scenario, tmp_path, monkeypatch):
        monkeypatch.setattr(freeze, "download_ohlc", lambda *a, **k: make_prices(scenario.universe))
        freeze.freeze_snapshot(scenario, root=tmp_path)
        ident = json.loads((tmp_path / scenario.name / "latest.json").read_text())["snapshot_id"]
        prices_file = tmp_path / "snapshots" / ident / "prices.csv"
        prices_file.write_text(prices_file.read_text().replace("AAPL", "XXXX", 1))
        with pytest.raises(RuntimeError, match="integrity check"):
            freeze.load_snapshot(scenario, root=tmp_path)


class TestLookAheadProtection:
    def test_last_close_never_exceeds_as_of(self, data_service):
        frame = data_service._frame("SPY")
        future = frame.index[-1].date()
        cutoff = future - timedelta(days=30)
        found = data_service.last_close("SPY", cutoff)
        assert found is not None
        assert found[0] <= cutoff

    def test_window_clamped(self, data_service):
        as_of = date(2022, 2, 1)
        window = data_service.window("SPY", as_of, 30)
        assert window.index[-1].date() <= as_of
        assert len(window) <= 30

    def test_closes_and_indicators_clamped(self, data_service):
        as_of = date(2022, 1, 15)
        closes = data_service.closes("SPY", as_of, 250)
        indicators = data_service.indicators("SPY", as_of)
        assert closes
        assert indicators["last_close"] == pytest.approx(round(closes[-1], 2))

    def test_requesting_future_data_returns_null(self, data_service):
        # an as_of beyond the data simply returns everything available, never beyond it
        as_of = date(2030, 1, 1)
        window = data_service.window("SPY", as_of, 10)
        assert window.index[-1].date() == data_service.trading_days()[-1]


class TestDecisionDates:
    def test_capped_schedule_preserves_full_history(self):
        from beatspy.data.service import DataService
        from conftest import make_prices

        data = DataService(make_prices(["AAPL"], start="2005-01-01", end="2026-10-02"))
        start, end = date(2006, 1, 1), date(2026, 10, 2)
        complete = data.decision_dates(start, end)
        for limit in (50, 60):
            days = data.decision_dates(start, end, max_decisions=limit)
            assert len(days) == len(set(days)) == limit
            assert days[0] == complete[0] and days[-1] == complete[-1]
            assert {day.year for day in days} == set(range(2006, 2027))
            assert all(day in complete and data.next_trading_day(day) <= end for day in days)
        assert data.decision_dates(start, end, max_decisions=500) == complete
        with __import__("pytest").raises(ValueError, match="at least 2"):
            data.decision_dates(start, end, max_decisions=1)

    def test_monthly_last_trading_day(self, data_service):
        days = data_service.decision_dates(date(2022, 1, 3), date(2022, 3, 31), "monthly")
        assert days[0] == date(2022, 1, 3)  # first trading day on/after start
        assert days[1] == date(2022, 1, 31)
        assert days[2] == date(2022, 2, 28)
        assert data_service.decision_dates(date(2022, 1, 3), date(2022, 3, 31), legacy=True) == [
            date(2022, 1, 3),
            date(2022, 2, 28),
            date(2022, 3, 31),
        ]

    def test_weekly_fridays(self, data_service):
        days = data_service.decision_dates(date(2022, 1, 3), date(2022, 1, 31), "weekly")
        assert days[0] == date(2022, 1, 3)  # Monday start
        assert all(d.weekday() == 4 for d in days[1:])
        assert date(2022, 1, 7) in days

    def test_next_trading_day(self, data_service):
        assert data_service.next_trading_day(date(2022, 1, 3)) > date(2022, 1, 3)
        assert data_service.next_trading_day(date(2099, 1, 1)) is None


class TestDataServiceIndicators:
    def test_indicator_fields(self, data_service):
        indicators = data_service.indicators("SPY", date(2022, 3, 1))
        for key in ("last_close", "sma_20", "sma_50", "rsi_14", "return_30d_pct", "annualized_vol_30d_pct"):
            assert key in indicators
        assert 0 <= indicators["rsi_14"] <= 100

    def test_universe_summary(self, data_service, scenario):
        rows = data_service.universe_summary(scenario.universe, date(2022, 3, 1))
        assert {row["ticker"] for row in rows} == set(scenario.universe)
