"""In-memory access to frozen market data.

Every query is clamped to an as-of date supplied by the benchmark harness, never
by the model. This is the single enforcement point for look-ahead protection on
price data.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class DataService:
    def __init__(self, prices: pd.DataFrame, benchmark: str = "SPY"):
        self.benchmark = benchmark.upper()
        self._frames: dict[str, pd.DataFrame] = {}
        for ticker, sub in prices.groupby("ticker"):
            frame = sub.set_index("date").sort_index()
            self._frames[str(ticker).upper()] = frame[["open", "high", "low", "close", "volume"]]
        cal_source = self._frames.get(self.benchmark)
        if cal_source is None:
            cal_source = next(iter(self._frames.values()))
        self.calendar: list[date] = [ts.date() for ts in cal_source.index]

    # ------------------------------------------------------------------ basics

    def trading_days(self) -> list[date]:
        return list(self.calendar)

    def has_ticker(self, ticker: str) -> bool:
        return ticker.upper() in self._frames

    def _frame(self, ticker: str) -> pd.DataFrame | None:
        return self._frames.get(ticker.upper())

    def _up_to(self, ticker: str, as_of: date) -> pd.DataFrame | None:
        frame = self._frame(ticker)
        if frame is None:
            return None
        return frame.loc[frame.index <= pd.Timestamp(as_of)]

    def last_close(self, ticker: str, as_of: date) -> tuple[date, float] | None:
        sub = self._up_to(ticker, as_of)
        if sub is None or sub.empty:
            return None
        ts = sub.index[-1]
        return ts.date(), float(sub["close"].iloc[-1])

    def close_on(self, ticker: str, day: date) -> float | None:
        sub = self._up_to(ticker, day)
        if sub is None or sub.empty or sub.index[-1].date() != day:
            return None
        return float(sub["close"].iloc[-1])

    def open_on(self, ticker: str, day: date) -> float | None:
        frame = self._frame(ticker)
        if frame is None:
            return None
        try:
            return float(frame.loc[pd.Timestamp(day), "open"])
        except KeyError:
            return None

    def next_trading_day(self, day: date) -> date | None:
        later = [d for d in self.calendar if d > day]
        return later[0] if later else None

    def close_on_offset(self, ticker: str, day: date, offset: int) -> float | None:
        """Close `offset` calendar trading days after `day`, for realized outcomes.

        This is the only accessor that deliberately looks forward: it exists so a
        decision recorded at `day` can be labeled with what actually happened.
        It must never be reachable from state construction, which is why state
        building goes through `window`/`trailing_return_pct` instead.
        """
        if offset <= 0:
            return None
        later = [d for d in self.calendar if d > day]
        if len(later) < offset:
            return None
        return self.close_on(ticker, later[offset - 1])

    def decision_dates(
        self, start: date, end: date, freq: str = "monthly", *, legacy=False, max_decisions: int | None = None
    ) -> list[date]:
        """First trading day on/after start, then weekly Fridays or month-ends."""
        days = [d for d in self.calendar if start <= d <= end]
        if not days:
            return []
        out = [days[0]]
        if freq == "weekly":
            out += [d for d in days[1:] if d.weekday() == 4 and d > out[-1]]
        else:  # monthly: include the first month's last trading day
            month_ends: dict[tuple[int, int], date] = {}
            for d in days:
                month_ends[(d.year, d.month)] = d
            first_month = (days[0].year, days[0].month)
            out += [d for key, d in month_ends.items() if (not legacy or key != first_month) and d != days[0]]
        if not legacy:
            out = [d for d in out if (following := self.next_trading_day(d)) is not None and following <= end]
        if max_decisions is not None:
            if type(max_decisions) is not int or max_decisions < 2:
                raise ValueError("max_decisions must be an integer of at least 2")
            if len(out) > max_decisions:
                # Retain both ends of the schedule and spread decisions over the full history.
                out = [out[i * (len(out) - 1) // (max_decisions - 1)] for i in range(max_decisions)]
        return out

    # ------------------------------------------------------------- analytics

    def window(self, ticker: str, as_of: date, days: int) -> pd.DataFrame | None:
        sub = self._up_to(ticker, as_of)
        if sub is None or sub.empty:
            return None
        return sub.tail(days)

    def indicators(self, ticker: str, as_of: date) -> dict | None:
        sub = self._up_to(ticker, as_of)
        if sub is None or len(sub) < 20:
            return None
        close = sub["close"]
        ret = close.pct_change().dropna()
        last = float(close.iloc[-1])

        def pct_since(n: int) -> float | None:
            if len(close) <= n:
                return None
            return round((last / float(close.iloc[-n - 1]) - 1.0) * 100.0, 2)

        rsi = None
        if len(ret) >= 14:
            gains = ret.clip(lower=0).rolling(14).mean()
            losses = ret.clip(upper=0).abs().rolling(14).mean()
            denom = losses.iloc[-1]
            rsi = 100.0 if denom == 0 else round(float(100 - 100 / (1 + gains.iloc[-1] / denom)), 1)

        vol_30d = round(float(ret.tail(30).std() * np.sqrt(252) * 100.0), 2) if len(ret) >= 15 else None
        high_252 = float(close.tail(252).max())
        vol_avg = float(sub["volume"].tail(20).mean()) if "volume" in sub else 0.0
        return {
            "last_close": round(last, 2),
            "sma_20": round(float(close.tail(20).mean()), 2),
            "sma_50": round(float(close.tail(50).mean()), 2) if len(close) >= 50 else None,
            "sma_200": round(float(close.tail(200).mean()), 2) if len(close) >= 200 else None,
            "rsi_14": rsi,
            "return_30d_pct": pct_since(21),
            "return_90d_pct": pct_since(63),
            "momentum_12_1_pct": (
                round((float(close.iloc[-22]) / float(close.iloc[-253]) - 1.0) * 100, 2) if len(close) >= 253 else None
            ),
            "annualized_vol_30d_pct": vol_30d,
            "off_52w_high_pct": round((last / high_252 - 1.0) * 100.0, 2) if high_252 else None,
            "avg_volume_20d": int(vol_avg) if vol_avg else None,
        }

    def universe_summary(self, tickers: list[str], as_of: date) -> list[dict]:
        rows = []
        for ticker in tickers:
            sub = self._up_to(ticker, as_of)
            if sub is None or sub.empty:
                continue
            close = sub["close"]
            last = float(close.iloc[-1])
            r30 = (last / float(close.iloc[-22]) - 1) * 100 if len(close) > 22 else None
            r90 = (last / float(close.iloc[-64]) - 1) * 100 if len(close) > 64 else None
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "last_close": round(last, 2),
                    "as_of": sub.index[-1].date().isoformat(),
                    "return_30d_pct": round(r30, 2) if r30 is not None else None,
                    "return_90d_pct": round(r90, 2) if r90 is not None else None,
                    **(self.indicators(ticker, as_of) or {}),
                }
            )
        return rows

    def decision_period(self, day: date, decisions: list[date], cutoff: date) -> dict:
        following = next((d for d in decisions if d > day), None)
        end = following or cutoff
        return {
            "horizon_trading_days": sum(day < d <= end for d in self.calendar),
            "next_decision_date": following.isoformat() if following else None,
            "evaluation_cutoff": cutoff.isoformat(),
        }

    def closes(self, ticker: str, as_of: date, days: int) -> list[float]:
        sub = self._up_to(ticker, as_of)
        if sub is None or sub.empty:
            return []
        return [float(v) for v in sub["close"].tail(days)]

    def trailing_return_pct(self, ticker: str, as_of: date, days: int) -> float | None:
        closes = self.closes(ticker, as_of, days + 1)
        if len(closes) < 2:
            return None
        return round((closes[-1] / closes[0] - 1.0) * 100.0, 2)
