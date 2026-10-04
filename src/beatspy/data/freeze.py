"""Immutable content-addressed market snapshots; legacy files stay untouched."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from ..artifacts import atomic_json, sha256
from ..schemas import Scenario
from .calendar import completed_session, resolve_scenario

log = logging.getLogger(__name__)
PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def scenario_data_dir(scenario: Scenario, root: Path | None = None) -> Path:
    return (root or Path.cwd() / ".beatspy-data") / scenario.name


def download_ohlc(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers=list(tickers), start=start, end=end, auto_adjust=True, progress=False, threads=False, group_by="ticker"
    )
    rows = []
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                continue
            frame = raw[ticker]
        else:
            frame = raw
        for idx, row in frame.dropna(subset=["Open", "Close"]).iterrows():
            rows.append(
                (
                    idx.date().isoformat(),
                    ticker.upper(),
                    *[float(row[c]) for c in ("Open", "High", "Low", "Close", "Volume")],
                )
            )
    return pd.DataFrame(rows, columns=PRICE_COLUMNS)


def snapshot_id(manifest: dict) -> str:
    keys = ("prices_sha256", "tickers", "start", "end", "source")
    payload = {key: manifest[key] for key in keys}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_prices(prices: pd.DataFrame, scenario: Scenario) -> None:
    if prices.empty or list(prices.columns) != PRICE_COLUMNS:
        raise ValueError("missing prices or invalid price columns")
    if prices.date.isna().any() or prices.duplicated(["date", "ticker"]).any():
        raise ValueError("invalid date or duplicate ticker/date in prices")
    numbers = prices[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(numbers).all() or (numbers[:, :4] <= 0).any() or (numbers[:, 4] < 0).any():
        raise ValueError("prices must be finite and positive; volume must be nonnegative")
    expected = set(scenario.universe)
    if set(prices.ticker) != expected:
        raise ValueError(f"missing or unexpected tickers: expected {sorted(expected)}")
    trading = prices[prices.ticker == scenario.benchmark.upper()]
    trading = set(
        trading.loc[
            (trading.date >= pd.Timestamp(scenario.start)) & (trading.date <= pd.Timestamp(scenario.end)), "date"
        ]
    )
    if not trading:
        raise ValueError("no benchmark sessions in evaluation window")
    calendar = xcals.get_calendar("XNYS", start=scenario.start, end=scenario.end)
    sessions = calendar.sessions
    sessions = sessions[(sessions >= pd.Timestamp(scenario.start)) & (sessions <= pd.Timestamp(scenario.end))]
    if trading != set(sessions):
        raise ValueError("missing or unexpected benchmark sessions in evaluation window")
    for ticker in expected:
        available = set(prices.loc[prices.ticker == ticker, "date"])
        if not trading <= available:
            raise ValueError(f"missing required ticker coverage: {ticker}")


def read_snapshot(directory: Path, scenario: Scenario) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))
    if sha256(directory / "prices.csv") != manifest["prices_sha256"]:
        raise RuntimeError("frozen snapshot failed its integrity check (prices.csv sha256 mismatch)")
    if manifest.get("schema_version") == 2 and snapshot_id(manifest) != manifest["snapshot_id"]:
        raise ValueError("snapshot manifest hash mismatch")
    prices = pd.read_csv(directory / "prices.csv", parse_dates=["date"])
    validate_prices(prices, scenario)
    if len(prices) != manifest["rows"]:
        raise ValueError("snapshot row count mismatch")
    if (manifest["start"], manifest["end"], manifest["tickers"]) != (
        prices.date.min().date().isoformat(),
        prices.date.max().date().isoformat(),
        scenario.universe,
    ):
        raise ValueError("snapshot coverage metadata mismatch")
    return prices, manifest


def freeze_snapshot(scenario: Scenario, root: Path | None = None) -> dict:
    scenario = resolve_scenario(scenario)
    sdir = scenario_data_dir(scenario, root)
    sdir.mkdir(parents=True, exist_ok=True)
    start = (datetime.fromisoformat(scenario.start) - timedelta(days=scenario.lookback_days)).date().isoformat()
    cutoff = min(datetime.fromisoformat(scenario.end).date(), completed_session())
    if cutoff.isoformat() < scenario.end:
        raise ValueError("requested end includes an incomplete or future session")
    log.info("Downloading %s through %s", scenario.universe, cutoff)
    prices = download_ohlc(scenario.universe, start, (cutoff + timedelta(days=1)).isoformat())
    prices = prices.assign(date=pd.to_datetime(prices.date))
    prices = prices.loc[(prices.date >= pd.Timestamp(start)) & (prices.date <= pd.Timestamp(cutoff))]
    prices = prices.sort_values(["date", "ticker"]).reset_index(drop=True)
    validate_prices(prices, scenario)
    latest = prices.loc[prices.ticker == scenario.benchmark.upper(), "date"].max().date()
    if scenario.year is not None and latest != cutoff:
        raise ValueError(f"price provider is stale: expected {cutoff}, received {latest}")
    snapshots = sdir.parent / "snapshots"
    snapshots.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="snapshot-", dir=snapshots))
    try:
        prices.to_csv(staging / "prices.csv", index=False, date_format="%Y-%m-%d")
        manifest = {
            "schema_version": 2,
            "scenario": scenario.name,
            "tickers": scenario.universe,
            "start": prices.date.min().date().isoformat(),
            "end": latest.isoformat(),
            "requested_start": scenario.start,
            "requested_end": scenario.end,
            "lookback_days": scenario.lookback_days,
            "rows": len(prices),
            "source": "Yahoo Finance via yfinance (adjusted OHLCV)",
            "prices_sha256": sha256(staging / "prices.csv"),
            "fetched_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        manifest["snapshot_id"] = snapshot_id(manifest)
        atomic_json(staging / "MANIFEST.json", manifest)
        target = snapshots / manifest["snapshot_id"]
        try:
            staging.rename(target)
        except OSError:
            if not target.exists():
                raise
            read_snapshot(target, scenario)
        atomic_json(
            sdir / "latest.json",
            {
                "snapshot_id": manifest["snapshot_id"],
                "requested_start": scenario.start,
                "requested_end": scenario.end,
                "lookback_days": scenario.lookback_days,
                "tickers": scenario.universe,
            },
        )
        return json.loads((target / "MANIFEST.json").read_text(encoding="utf-8"))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def load_snapshot(scenario: Scenario, root: Path | None = None, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    scenario = resolve_scenario(scenario)
    sdir = scenario_data_dir(scenario, root)
    with _locks_guard:
        lock = _locks.setdefault(str(sdir.resolve()), threading.Lock())
    with lock:
        pointer_path = sdir / "latest.json"
        pointer = json.loads(pointer_path.read_text()) if pointer_path.exists() else {}
        matching = all(
            pointer.get(k) == value
            for k, value in {
                "requested_start": scenario.start,
                "requested_end": scenario.end,
                "lookback_days": scenario.lookback_days,
                "tickers": scenario.universe,
            }.items()
        )
        if refresh or not matching:
            pointer = freeze_snapshot(scenario, root)
        ident = pointer["snapshot_id"]
        if not re.fullmatch(r"[a-f0-9]{64}", ident):
            raise ValueError("invalid snapshot id")
        directory = sdir.parent / "snapshots" / ident
        prices, manifest = read_snapshot(directory, scenario)
        return prices, {**manifest, "path": str(directory.resolve())}
