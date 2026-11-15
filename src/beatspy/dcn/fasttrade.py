"""FastTrade's model signal, adapted to BeatSPY's daily bars.

FastTrade normally runs on hourly candles. BeatSPY's frozen snapshots are daily,
so this applies the same feature list to daily bars: EMA 9/21/50, RSI-14, MACD,
ATR-14, VWAP (the last bar's typical price), the volume ratio and 1/3/5-bar
returns. The model is asked FastTrade's own BUY/HOLD/SELL choice question
through TypeSafe Jev, then the three probabilities become one relative view:

    P(beat SPY) = P(BUY) + 0.5 * P(HOLD)

That is the expected value of a three-level score (BUY 1, HOLD 0.5, SELL 0).
It is a fixed mapping, not a fitted calibration.
"""

from __future__ import annotations

import asyncio
import math
import time
from datetime import date

import pandas as pd

from ..schemas import ChoiceAnswer, NoulAnswer, Question, ScoreAnswer
from .clients import DcnClient, DcnResult

MIN_BARS = 50
TIMEFRAME = "1d"
ACTION_QUESTION = "action"
ACTION_INSTRUCTIONS = "Which long-only action is supported by this market snapshot?"
ACTION_CRITERIA = {
    "BUY": "Evidence favors a long entry",
    "HOLD": "No clear trade opportunity",
    "SELL": "Evidence favors reducing or exiting an existing long position",
}
PROVIDER = "fasttrade"
MODEL = "fasttrade-jev"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MAX_CONCURRENCY = 5


def fasttrade_features(frame: pd.DataFrame | None) -> dict | None:
    """FastTrade's feature snapshot at the last bar, or None before the warmup."""
    if frame is None or len(frame) < MIN_BARS:
        return None

    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    volume = frame["volume"].astype(float)

    def ema(period: int) -> pd.Series:
        return close.ewm(span=period, adjust=False, min_periods=period).mean()

    change = close.diff()
    gain = float(
        change.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().iloc[-1]
    )
    loss = float(
        (-change.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().iloc[-1]
    )
    if math.isnan(gain) or math.isnan(loss):
        return None
    rsi = 50.0 if gain == loss == 0 else (100.0 if loss == 0 else 100.0 - 100.0 / (1.0 + gain / loss))

    true_range = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    typical = (high + low + close) / 3.0
    last_volume = float(volume.iloc[-1])
    average_volume = float(volume.tail(20).mean())
    if last_volume <= 0.0 or average_volume <= 0.0:
        return None
    vwap = float((typical.iloc[-1] * last_volume) / last_volume)
    if vwap <= 0.0:
        return None

    macd_series = ema(12) - ema(26)
    macd = float(macd_series.iloc[-1])
    macd_signal = float(macd_series.ewm(span=9, adjust=False, min_periods=9).mean().iloc[-1])
    atr = float(true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().iloc[-1])
    volume_ratio = last_volume / average_volume
    returns = {
        f"return_{n}": float(close.iloc[-1] / close.iloc[-1 - n] - 1.0) for n in (1, 3, 5)
    }

    features = {
        "price": float(close.iloc[-1]),
        "ema_9": float(ema(9).iloc[-1]),
        "ema_21": float(ema(21).iloc[-1]),
        "ema_50": float(ema(50).iloc[-1]),
        "rsi_14": rsi,
        "macd": macd,
        "macd_signal": macd_signal,
        "vwap": vwap,
        "vwap_delta": float(close.iloc[-1] / vwap - 1.0),
        "atr_14": atr,
        "volume_ratio": volume_ratio,
        **returns,
        "spread": None,
        "liquidity": None,
    }
    for key in (
        "price",
        "ema_9",
        "ema_21",
        "ema_50",
        "rsi_14",
        "macd",
        "macd_signal",
        "vwap",
        "vwap_delta",
        "atr_14",
        "volume_ratio",
        *returns,
    ):
        if not math.isfinite(float(features[key])):
            return None
    return features


def fasttrade_state(frame: pd.DataFrame | None, ticker: str, as_of: date) -> dict | None:
    """The exact state shape FastTrade sends to its decision model."""
    features = fasttrade_features(frame)
    if features is None:
        return None
    return {
        "symbol": ticker.upper(),
        "timeframe": TIMEFRAME,
        "candle_timestamp": pd.Timestamp(frame.index[-1]).isoformat(),
        "evaluation_time": as_of.isoformat(),
        "features": features,
        "ticker_details": {"exchangeTimezoneName": "America/New_York"},
        "headlines": [],
    }


def beat_probability(probabilities: dict) -> float | None:
    """P(beat SPY) from FastTrade's three action probabilities."""
    if set(probabilities) != {"BUY", "HOLD", "SELL"}:
        return None
    try:
        values = {key: float(value) for key, value in probabilities.items()}
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values.values()):
        return None
    return min(max(values["BUY"] + 0.5 * values["HOLD"], 0.0), 1.0)


def _ticker(question_id: str, question: Question) -> str:
    instructions = question.instructions
    if isinstance(instructions, dict) and instructions.get("ticker"):
        return str(instructions["ticker"]).upper()
    return question_id.rsplit("_", 1)[-1].upper()


class FastTradeClient:
    """FastTrade's per-symbol model, run against the frozen `DataService`."""

    provider = PROVIDER
    model = MODEL
    endpoint = ENDPOINT
    rule = "action"

    def __init__(self, data, provider: DcnClient, *, concurrency: int = MAX_CONCURRENCY):
        self.data = data
        self.inner = provider
        self._semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _decide_noul(self, state: dict, questions: dict[str, Question]) -> DcnResult:
        """Ask the benchmark's own typed questions, the same rule as every board model."""
        as_of = date.fromisoformat(str(state["as_of"]))
        by_ticker: dict[str, dict[str, Question]] = {}
        for question_id, question in questions.items():
            by_ticker.setdefault(_ticker(question_id, question), {})[question_id] = question

        async def one(ticker: str, ticker_questions: dict[str, Question]):
            async with self._semaphore:
                frame = self.data.window(ticker, as_of, 100_000) if self.data is not None else None
                ft_state = fasttrade_state(frame, ticker, as_of) if frame is not None else None
                if ft_state is None:
                    return ticker, None
                try:
                    result = await self.inner.decide(ft_state, ticker_questions)
                except Exception:
                    return ticker, None
                return ticker, result

        started = time.perf_counter()
        outcomes = await asyncio.gather(*(one(ticker, qs) for ticker, qs in by_ticker.items()))
        elapsed = time.perf_counter() - started

        answers: dict[str, object] = {}
        missing: list[str] = []
        input_tokens = output_tokens = requests = 0
        attempts = 1
        for ticker, result in outcomes:
            if result is None:
                missing.extend(by_ticker[ticker])
                continue
            answers.update(result.answers)
            missing.extend(result.missing_ids)
            input_tokens += int(result.input_tokens)
            output_tokens += int(result.output_tokens)
            requests += int(result.requests)
            attempts = max(attempts, int(result.attempts))
        return DcnResult(
            answers=answers,
            missing_ids=missing,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=elapsed,
            attempts=attempts,
            requests=requests,
        )

    async def decide(self, state: dict, questions: dict[str, Question]) -> DcnResult:
        if getattr(self, "rule", "action") == "noul":
            return await self._decide_noul(state, questions)
        as_of = date.fromisoformat(str(state["as_of"]))
        by_ticker: dict[str, list[tuple[str, Question]]] = {}
        for question_id, question in questions.items():
            by_ticker.setdefault(_ticker(question_id, question), []).append((question_id, question))

        action = Question(
            type="choice",
            instructions=ACTION_INSTRUCTIONS,
            criteria=ACTION_CRITERIA,
        )

        async def one(ticker: str):
            async with self._semaphore:
                frame = self.data.window(ticker, as_of, 100_000) if self.data is not None else None
                fast_state = fasttrade_state(frame, ticker, as_of) if frame is not None else None
                if fast_state is None:
                    return ticker, None, None
                try:
                    result = await self.inner.decide(fast_state, {ACTION_QUESTION: action})
                except Exception:
                    return ticker, None, None
                answer = result.answers.get(ACTION_QUESTION)
                if not isinstance(answer, ChoiceAnswer):
                    return ticker, None, result
                p = beat_probability(answer.probabilities)
                if p is None:
                    return ticker, None, result
                return ticker, p, result

        started = time.perf_counter()
        results = await asyncio.gather(*(one(ticker) for ticker in by_ticker))
        elapsed = time.perf_counter() - started

        answers: dict[str, object] = {}
        missing: list[str] = []
        input_tokens = 0
        requests = 0
        attempts = 1
        for ticker, p, result in results:
            if result is not None:
                input_tokens += int(result.input_tokens)
                requests += int(result.requests)
                attempts = max(attempts, int(result.attempts))
            if p is None:
                missing.extend(question_id for question_id, _ in by_ticker[ticker])
                continue
            for question_id, question in by_ticker[ticker]:
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
            input_tokens=input_tokens,
            output_tokens=0,
            latency_s=elapsed,
            attempts=attempts,
            requests=requests,
        )

    async def close(self) -> None:
        await self.inner.close()


class FastTradeSubqClient(FastTradeClient):
    """FastTrade's daily feature state, asked the benchmark's typed questions by SubQ Snap 1.1."""

    provider = "fasttrade-subq"
    model = "fasttrade-subq"
    endpoint = "https://api.subq.ai/v1/systemone"
    rule = "noul"
