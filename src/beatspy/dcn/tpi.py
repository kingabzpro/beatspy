"""The TPI-RS deterministic baseline.

This is not an LLM decision model. It computes the TPI-RS trend/breakout score
from the frozen prices the harness already holds, then encodes rule agreement as
a probability. The Rust implementation is the reference; `tests/test_dcn_tpi.py`
pins this port to the Rust indicator values and composite score on a fixed
series.

The scoring profile is the short horizon from TPI-RS `config/default.toml`,
which matches the benchmark's five-trading-day questions. The client reads bars
through `DataService.window`, so no bar dated after the decision can enter an
answer. A decision before the 220-bar warmup is reported as a missing answer,
never a fabricated neutral.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ..schemas import ChoiceAnswer, NoulAnswer, Question, ScoreAnswer
from .clients import DcnResult

# Short-horizon profile, mirrored from TPI-RS config/default.toml.
MIN_BARS = 220
RSI_MIN = 50.0
RSI_MAX = 78.0
BUY_THRESHOLD = 62.0
MIN_ATR_PCT = 0.8
MAX_ATR_PCT = 7.0
VOLUME_RATIO_MIN = 1.0
# Ten score points per logit. Fixed, not fitted: this is a monotone encoding of
# rule agreement, not a calibrated forecast.
LOGIT_POINTS = 10.0


@dataclass(frozen=True)
class TpiSignal:
    score: float
    buy: bool


@dataclass(frozen=True)
class _Features:
    close: float
    ema20: float
    ema50: float
    ema200: float
    ema50_slope_pct: float
    rsi14: float
    rsi14_prev3: float
    atr14: float
    atr_pct: float
    adx14: float
    plus_di: float
    minus_di: float
    don20_high: float
    don40_high: float
    don40_low: float
    macd_hist: float
    volume_ratio: float
    dist_ema20_atr: float


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss <= np.finfo(float).eps:
        return 100.0
    if avg_gain <= np.finfo(float).eps:
        return 0.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _wilder(values: pd.Series, period: int) -> pd.Series:
    out = np.full(len(values), np.nan)
    if len(values) < period:
        return pd.Series(out, index=values.index)
    arr = values.to_numpy(dtype=float)
    avg = float(arr[:period].mean())
    out[period - 1] = avg
    for i in range(period, len(arr)):
        avg = (avg * (period - 1) + arr[i]) / period
        out[i] = avg
    return pd.Series(out, index=values.index)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    out = np.full(len(close), np.nan)
    if len(close) <= period:
        return pd.Series(out, index=close.index)
    values = close.to_numpy(dtype=float)
    gains = np.zeros(len(values))
    losses = np.zeros(len(values))
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        if change >= 0.0:
            gains[i] = change
        else:
            losses[i] = -change
    avg_gain = float(gains[1 : period + 1].mean())
    avg_loss = float(losses[1 : period + 1].mean())
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return pd.Series(out, index=close.index)


def _true_range(frame: pd.DataFrame) -> np.ndarray:
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    tr = high - low
    if len(tr) > 1:
        prev = close[:-1]
        tr[1:] = np.maximum.reduce(
            [tr[1:], np.abs(high[1:] - prev), np.abs(low[1:] - prev)]
        )
    return np.maximum(tr, 0.0)


def _adx(frame: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    n = len(frame)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0.0:
            plus_dm[i] = up
        if down > up and down > 0.0:
            minus_dm[i] = down

    tr_s = _wilder(pd.Series(_true_range(frame), index=frame.index), period).to_numpy()
    plus_s = _wilder(pd.Series(plus_dm), period).to_numpy()
    minus_s = _wilder(pd.Series(minus_dm), period).to_numpy()

    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        if not math.isnan(tr_s[i]) and tr_s[i] > 0.0:
            pdi = 100.0 * plus_s[i] / tr_s[i]
            mdi = 100.0 * minus_s[i] / tr_s[i]
            plus_di[i] = pdi
            minus_di[i] = mdi
            denom = pdi + mdi
            if denom > 0.0:
                dx[i] = 100.0 * abs(pdi - mdi) / denom
                valid[i] = True

    adx = np.full(n, np.nan)
    collected = [(i, dx[i]) for i in range(n) if valid[i]]
    if len(collected) >= period:
        first_avg = sum(value for _, value in collected[:period]) / period
        adx[collected[period - 1][0]] = first_avg
        prev = first_avg
        for index, value in collected[period:]:
            prev = (prev * (period - 1) + value) / period
            adx[index] = prev
    return (
        pd.Series(adx, index=frame.index),
        pd.Series(plus_di, index=frame.index),
        pd.Series(minus_di, index=frame.index),
    )


def _macd_hist(close: pd.Series) -> pd.Series:
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd - signal_line


def _features(frame: pd.DataFrame | None) -> _Features | None:
    """The last-bar indicators, or None before the 220-bar warmup."""
    if frame is None or len(frame) < MIN_BARS:
        return None

    close = frame["close"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    rsi = _rsi(close)
    atr = _wilder(pd.Series(_true_range(frame), index=frame.index), 14)
    adx, plus_di, minus_di = _adx(frame)
    macd_hist = _macd_hist(close)
    don20_high = frame["high"].rolling(20).max().shift(1)
    don40_high = frame["high"].rolling(40).max().shift(1)
    don40_low = frame["low"].rolling(40).min().shift(1)
    volume_ratio = frame["volume"] / frame["volume"].rolling(20).mean()

    last = -1
    price = float(close.iloc[last])
    ema50_now = float(ema50.iloc[last])
    ema200_now = float(ema200.iloc[last])
    rsi_now = float(rsi.iloc[last])
    atr_now = float(atr.iloc[last])
    don20_now = float(don20_high.iloc[last])
    if any(
        math.isnan(value)
        for value in (price, ema50_now, ema200_now, rsi_now, atr_now, don20_now)
    ):
        return None

    def optional(series: pd.Series) -> float:
        value = float(series.iloc[last])
        return 0.0 if math.isnan(value) else value

    ema20_now = float(ema20.iloc[last])
    atr_pct_now = atr_now / price * 100.0 if price > 0.0 else 0.0
    dist_now = (
        (price - ema20_now) / atr_now
        if not math.isnan(ema20_now) and atr_now > 0.0
        else 0.0
    )
    ema50_prev = float(ema50.iloc[-11]) if len(frame) >= 11 else 0.0
    slope = (
        (ema50_now - ema50_prev) / ema50_prev * 100.0
        if abs(ema50_prev) > np.finfo(float).eps
        else 0.0
    )
    rsi_prevg = float(rsi.iloc[-4]) if len(frame) >= 4 else math.nan

    return _Features(
        close=price,
        ema20=0.0 if math.isnan(ema20_now) else ema20_now,
        ema50=ema50_now,
        ema200=ema200_now,
        ema50_slope_pct=slope,
        rsi14=rsi_now,
        rsi14_prev3=rsi_prevg,
        atr14=atr_now,
        atr_pct=atr_pct_now,
        adx14=optional(adx),
        plus_di=optional(plus_di),
        minus_di=optional(minus_di),
        don20_high=don20_now,
        don40_high=optional(don40_high),
        don40_low=optional(don40_low),
        macd_hist=optional(macd_hist),
        volume_ratio=optional(volume_ratio),
        dist_ema20_atr=dist_now,
    )


def signal(frame: pd.DataFrame | None) -> TpiSignal | None:
    """The short-horizon TPI score at the last bar, or None before the warmup."""
    features = _features(frame)
    if features is None:
        return None

    trend = 0.0
    if features.close > features.ema50:
        trend += 15.0
    if features.ema50 > features.ema200:
        trend += 10.0
    if features.close > features.ema200:
        trend += 5.0
    if features.ema50_slope_pct > 0.0:
        trend += 5.0

    breakout = 25.0 if features.close > features.don20_high else 0.0

    momentum = 0.0
    if RSI_MIN <= features.rsi14 <= RSI_MAX:
        momentum += 10.0
    if not math.isnan(features.rsi14_prev3) and features.rsi14 > features.rsi14_prev3:
        momentum += 4.0
    if features.macd_hist > 0.0:
        momentum += 6.0

    confirmation = 0.0
    if features.adx14 >= 20.0:
        confirmation += 4.0
    if features.plus_di > features.minus_di:
        confirmation += 3.0
    if features.volume_ratio >= VOLUME_RATIO_MIN:
        confirmation += 3.0

    volatility = 0.0
    if MIN_ATR_PCT <= features.atr_pct <= MAX_ATR_PCT:
        volatility += 5.0
    if -1.0 <= features.dist_ema20_atr <= 3.0:
        volatility += 5.0

    score = trend + breakout + momentum + confirmation + volatility
    if features.rsi14 > 80.0:
        score -= 12.0
    if features.atr_pct > MAX_ATR_PCT:
        score -= 10.0
    if features.dist_ema20_atr > 3.0:
        score -= 5.0
    if features.close < features.ema200:
        score -= 10.0
    score = min(max(score, 0.0), 100.0)

    buy = (
        features.close > features.ema200
        and features.ema50 > features.ema200
        and features.close > features.don20_high
        and RSI_MIN <= features.rsi14 <= RSI_MAX
        and MIN_ATR_PCT <= features.atr_pct <= MAX_ATR_PCT
        and score >= BUY_THRESHOLD
    )
    return TpiSignal(score=score, buy=buy)


def probability(score: float, *, buy: bool) -> float:
    """Map rule agreement to a probability, with the BUY gate as the midpoint.

    A name the strategy would not buy stays below 0.5 even if its raw score is
    high but a gate failed. The encoding is monotone in the score on both sides.
    """
    logit = (score - BUY_THRESHOLD) / LOGIT_POINTS
    scaled = 1.0 / (1.0 + math.exp(-logit))
    return 0.5 + 0.5 * scaled if buy else 0.5 * scaled


def _ticker(question_id: str, question: Question) -> str:
    instructions = question.instructions
    if isinstance(instructions, dict) and instructions.get("ticker"):
        return str(instructions["ticker"]).upper()
    return question_id.rsplit("_", 1)[-1].upper()


class TpiClient:
    """A local `DcnClient` backed by the frozen `DataService`, no network."""

    provider = "local"
    model = "tpi-rs"
    endpoint = "local://tpi-rs"

    def __init__(self, data):
        self.data = data

    async def decide(self, state: dict, questions: dict[str, Question]) -> DcnResult:
        as_of = date.fromisoformat(str(state["as_of"]))
        signals: dict[str, TpiSignal | None] = {}
        answers: dict[str, object] = {}
        missing: list[str] = []

        for question_id, question in questions.items():
            ticker = _ticker(question_id, question)
            if ticker not in signals:
                frame = self.data.window(ticker, as_of, 100_000) if self.data is not None else None
                signals[ticker] = signal(frame)
            found = signals[ticker]
            if found is None:
                # Warmup or missing data is missing coverage, not a coin flip.
                missing.append(question_id)
                continue
            p = probability(found.score, buy=found.buy)

            if question.type == "noul":
                answers[question_id] = NoulAnswer(noul=p)
            elif question.type == "choice":
                answers[question_id] = ChoiceAnswer(
                    choice="outperform" if p >= 0.5 else "underperform",
                    probabilities={"outperform": p, "underperform": 1.0 - p},
                )
            else:
                levels = question.criteria or []
                answers[question_id] = ScoreAnswer(
                    score=p * (len(levels) - 1),
                    legend={str(i): str(level) for i, level in enumerate(levels)},
                )

        return DcnResult(
            answers=answers,
            missing_ids=missing,
            input_tokens=0,
            output_tokens=0,
            latency_s=0.0,
            attempts=1,
            requests=0,
        )

    async def close(self) -> None:
        return None
