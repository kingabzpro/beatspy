"""BetterTradingAgents' Fast profile as a deterministic replay.

Ported from BetterTradingAgents (MIT):
  app/tools/indicators.py
  app/agents/technical.py
  app/agents/forecast.py
  app/agents/manager.py

The Fast profile runs three agents: technical, forecast, manager. Without an
LLM, the app substitutes deterministic rules for each one. This module ports
that replay to BeatSPY's daily bars. It is not the live agent pipeline: no
model is called, and the label says so.

The manager's ``probability_beat_spy`` is direction-specific in the app (for a
BUY it means P(outperform), for a SELL it means P(underperform)), so the
BeatSPY probability is ``p`` for BUY, ``1 - p`` for SELL, and 0.5 for HOLD.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp
from statistics import mean, pstdev

import numpy as np
import pandas as pd

from ..schemas import ChoiceAnswer, NoulAnswer, Question, ScoreAnswer
from .clients import DcnResult

MIN_BARS = 50
HORIZON_DAYS = 5
PROVIDER = "bta"
MODEL = "bettertradingagents"
ENDPOINT = "local://bettertradingagents"
RIDGE_LAGS = 5
RIDGE_ALPHA = 1.0


# --------------------------------------------------------------------------- #
# Indicators, ported verbatim from app/tools/indicators.py
# --------------------------------------------------------------------------- #


def sma(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return mean(values[-window:])


def ema_series(values: list[float], window: int) -> list[float] | None:
    """Exponential moving average over the whole series (seeded with the SMA)."""
    if len(values) < window:
        return None
    seed = mean(values[:window])
    alpha = 2.0 / (window + 1)
    out = [seed]
    for value in values[window:]:
        out.append(alpha * value + (1 - alpha) * out[-1])
    return out


def rsi(closes: list[float], period: int = 14) -> float | None:
    """Simple-average RSI, matching the app (not Wilder smoothing)."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for prev, cur in zip(closes[-(period + 1) :], closes[-period:], strict=False):
        change = cur - prev
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = mean(gains)
    avg_loss = mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict | None:
    fast_ema = ema_series(closes, fast)
    slow_ema = ema_series(closes, slow)
    if fast_ema is None or slow_ema is None:
        return None
    offset = len(fast_ema) - len(slow_ema)
    macd_line = [f - s for f, s in zip(fast_ema[offset:], slow_ema, strict=True)]
    signal_ema = ema_series(macd_line, signal)
    if signal_ema is None:
        return None
    return {
        "macd": round(macd_line[-1], 3),
        "macd_signal": round(signal_ema[-1], 3),
        "macd_histogram": round(macd_line[-1] - signal_ema[-1], 3),
    }


def bollinger(closes: list[float], window: int = 20, num_std: float = 2.0) -> dict | None:
    if len(closes) < window:
        return None
    mid = mean(closes[-window:])
    dev = pstdev(closes[-window:])
    upper, lower = mid + num_std * dev, mid - num_std * dev
    price = closes[-1]
    percent_b = (price - lower) / (upper - lower) if upper > lower else None
    return {
        "bollinger_percent_b": round(percent_b, 3) if percent_b is not None else None,
        "bollinger_width_pct": round((upper - lower) / mid * 100, 2) if mid else None,
    }


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float | None:
    if not highs or len(highs) != len(lows) or len(highs) != len(closes):
        return None
    if len(closes) < period + 1:
        return None
    ranges = []
    for prev_close, high, low in zip(closes[-(period + 1) :], highs[-period:], lows[-period:], strict=False):
        ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return mean(ranges)


def pct_change(closes: list[float], lookback_days: int) -> float | None:
    if len(closes) < lookback_days + 1 or closes[-lookback_days - 1] == 0:
        return None
    return (closes[-1] / closes[-lookback_days - 1] - 1.0) * 100.0


def annualized_volatility(closes: list[float], window: int = 20) -> float | None:
    if len(closes) < window + 1:
        return None
    returns = [cur / prev - 1.0 for prev, cur in zip(closes[-(window + 1) :], closes[-window:], strict=False)]
    return pstdev(returns) * (252**0.5) * 100.0


def volume_stats(volumes: list[float], window: int = 20) -> dict:
    if not volumes or volumes[-1] in (None, 0):
        return {}
    recent = [value for value in volumes[-window:] if value is not None]
    if not recent:
        return {}
    avg = mean(recent)
    if avg == 0:
        return {}
    return {
        "avg_volume_20d": round(avg),
        "relative_volume": round(volumes[-1] / avg, 2),
    }


def _ridge_fit(x: np.ndarray, y: np.ndarray):
    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    x_std = np.where(x_std > 0, x_std, 1.0)
    y_mean = float(y.mean())
    xs = (x - x_mean) / x_std
    beta = np.linalg.solve(
        xs.T @ xs + RIDGE_ALPHA * np.eye(xs.shape[1]), xs.T @ (y - y_mean)
    )

    def predict(rows: np.ndarray) -> np.ndarray:
        return ((rows - x_mean) / x_std) @ beta + y_mean

    return predict


def historical_forecast(
    closes: list[float], horizon: int = 5, window: int = 120
) -> dict | None:
    if horizon < 1 or len(closes) < 30:
        return None
    sample = closes[-window:]
    if any(price <= 0 for price in sample):
        return None

    logs = np.log(np.asarray(sample, dtype=float))
    rets = np.diff(logs)
    m = len(sample)
    if m - RIDGE_LAGS - horizon < 20:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(rets, RIDGE_LAGS)

    def targets(step: int, rows: int) -> np.ndarray:
        return (logs[step:] - logs[:-step])[RIDGE_LAGS : RIDGE_LAGS + rows]

    x_last = windows[-1]
    path = []
    for step in range(1, horizon + 1):
        rows = m - step - RIDGE_LAGS
        predict = _ridge_fit(windows[:rows], targets(step, rows))
        path.append(round(sample[-1] * exp(float(predict(x_last))), 2))

    rows = m - horizon - RIDGE_LAGS
    y = targets(horizon, rows)
    if rows >= 50:
        split = rows - 30
        predicted = _ridge_fit(windows[:split], y[:split])(windows[split:rows])
        actual = y[split:]
    else:
        predicted = _ridge_fit(windows[:rows], y)(windows[:rows])
        actual = y
    resid = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - actual.mean()) ** 2))
    r_squared = 1.0 if (total == 0 and resid == 0) else (0.0 if total == 0 else 1 - resid / total)
    r_squared = max(-1.0, min(1.0, r_squared))

    return {
        "forecast_price_5d": path[-1],
        "forecast_change_5d_pct": round((path[-1] / closes[-1] - 1) * 100, 2),
        "forecast_trend_r2": round(r_squared, 3),
        "forecast_window_days": len(sample),
        "forecast_horizon_days": horizon,
        "forecast_path": path,
        "forecast_method": "ridge_lag",
    }


def _safe_round(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if value is not None else None


def compute_indicators(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
) -> dict:
    if len(closes) < 2:
        return {"error": "not enough price history"}

    price = closes[-1]
    sma20 = sma(closes, 20)
    sma50 = sma(closes, 50)
    momentum = pct_change(closes, 10)
    rsi_now = rsi(closes)
    vol_now = annualized_volatility(closes)

    out = {
        "price": round(price, 2),
        "sma_20": round(sma20, 2) if sma20 else None,
        "sma_50": round(sma50, 2) if sma50 else None,
        "rsi_14": round(rsi_now, 1) if rsi_now is not None else None,
        "momentum_10d_pct": round(momentum, 2) if momentum is not None else None,
        "volatility_annualized_pct": round(vol_now, 2) if vol_now is not None else None,
        "change_1d_pct": _safe_round(pct_change(closes, 1)),
        "change_5d_pct": _safe_round(pct_change(closes, 5)),
        "change_21d_pct": _safe_round(pct_change(closes, 21)),
        "change_63d_pct": _safe_round(pct_change(closes, 63)),
        "price_above_sma20": (price > sma20) if sma20 else None,
        "sma20_above_sma50": (sma20 > sma50) if (sma20 and sma50) else None,
        "history_days": len(closes),
    }

    macd_data = macd(closes)
    if macd_data:
        out.update(macd_data)
    boll_data = bollinger(closes)
    if boll_data:
        out.update(boll_data)
    if highs and lows:
        atr_value = atr(highs, lows, closes)
        if atr_value is not None:
            out["atr_14"] = round(atr_value, 2)
            out["atr_pct_of_price"] = round(atr_value / price * 100, 2) if price else None
    if volumes:
        out.update(volume_stats(volumes))
    forecast = historical_forecast(closes)
    if forecast:
        out.update(forecast)
    return out


# --------------------------------------------------------------------------- #
# Fast profile mock agents, ported from app/agents
# --------------------------------------------------------------------------- #


def technical_mock(ticker: str, payload: dict) -> dict:
    """Deterministic rule-based fallback used when no LLM is configured."""
    score = 0.0
    if payload.get("price_above_sma20"):
        score += 1
    if payload.get("sma20_above_sma50"):
        score += 1
    if (payload.get("momentum_10d_pct") or 0) > 0:
        score += 1
    if (payload.get("change_21d_pct") or 0) > 0:
        score += 1
    if (payload.get("macd_histogram") or 0) > 0:
        score += 0.5
    forecast_method = str(payload.get("forecast_method") or "")
    r2 = payload.get("forecast_trend_r2") or 0
    if forecast_method.startswith(("timegpt-", "bako-")):
        forecast_usable = True
    elif forecast_method == "ridge_lag":
        forecast_usable = r2 > 0
    else:
        forecast_usable = r2 >= 0.25
    if forecast_usable:
        score += 0.5 if (payload.get("forecast_change_5d_pct") or 0) > 0 else -0.5
    rsi_value = payload.get("rsi_14")
    if rsi_value is not None and rsi_value > 70:
        score -= 0.5
    if rsi_value is not None and rsi_value < 30:
        score += 0.5
    signal = "bullish" if score >= 2.5 else "bearish" if score <= 0.5 else "neutral"
    confidence = min(0.55 + 0.08 * abs(score - 1.5), 0.85)
    return {
        "ticker": ticker,
        "signal": signal,
        "confidence": round(confidence, 2),
        "summary": (
            f"[mock] Price {payload.get('price')} vs SMA20 {payload.get('sma_20')} / "
            f"SMA50 {payload.get('sma_50')}, RSI {rsi_value}, 10d momentum "
            f"{payload.get('momentum_10d_pct')}%, 5d forecast "
            f"{payload.get('forecast_change_5d_pct')}% (trend R2 "
            f"{payload.get('forecast_trend_r2')}, {payload.get('forecast_method')}) "
            f"-> {signal}."
        ),
    }


def forecast_payload(indicators: dict, history_days: int, horizon: int = HORIZON_DAYS) -> dict:
    local = {
        key: indicators.get(key)
        for key in (
            "forecast_price_5d",
            "forecast_change_5d_pct",
            "forecast_trend_r2",
            "forecast_window_days",
            "forecast_horizon_days",
            "forecast_path",
            "forecast_method",
        )
    }
    vol = indicators.get("volatility_annualized_pct")
    noise = vol / (252**0.5) * (horizon**0.5) if vol is not None else None
    return {
        "price": indicators.get("price"),
        "history_days": history_days,
        "hosted_forecast": None,
        "local_trend_forecast": local,
        "noise_band_pct": round(noise, 2) if noise is not None else None,
        "horizon_days": horizon,
        "trend_context": {
            key: indicators.get(key)
            for key in (
                "sma_20",
                "sma_50",
                "change_5d_pct",
                "change_21d_pct",
                "rsi_14",
                "volatility_annualized_pct",
                "atr_pct_of_price",
            )
        },
        "user_context": "",
    }


def forecast_mock(ticker: str, payload: dict) -> dict:
    """Deterministic rule-based fallback used when no LLM is configured."""
    hosted = payload.get("hosted_forecast") or {}
    local = payload.get("local_trend_forecast") or {}
    primary = hosted or local
    method = primary.get("forecast_method", "none")
    change = primary.get("forecast_change_5d_pct")
    r2 = primary.get("forecast_trend_r2")
    vol = (payload.get("trend_context") or {}).get("volatility_annualized_pct")
    horizon = int(payload.get("horizon_days") or HORIZON_DAYS)
    noise = round(vol / (252**0.5) * (horizon**0.5), 2) if vol else None

    if method == "none":
        signal, confidence = "neutral", 0.4
        detail = "unusable projection (none)"
    elif method == "ridge_lag" and (r2 is None or r2 <= 0):
        signal, confidence = "neutral", 0.4
        detail = f"unusable projection ({method}, out-of-sample R2 {r2})"
    elif method == "log_linear_trend" and (r2 is None or r2 < 0.25):
        signal, confidence = "neutral", 0.4
        detail = f"unusable projection ({method}, trend R2 {r2})"
    elif not change:
        signal, confidence = "neutral", 0.45
        detail = "flat projection"
    elif noise and abs(change) < noise:
        signal, confidence = "neutral", 0.45
        detail = f"{change:+.2f}% sits inside the ~{noise:.1f}% {horizon}-day noise band"
    else:
        signal = "bullish" if change > 0 else "bearish"
        ratio = abs(change) / noise if noise else 1.0
        confidence = round(min(0.5 + 0.15 * ratio + (0.05 if hosted else 0.0), 0.8), 2)
        detail = f"{change:+.2f}% clears the ~{noise:.1f}% {horizon}-day noise band"
    return {
        "ticker": ticker,
        "signal": signal,
        "confidence": confidence,
        "summary": f"[mock] {horizon}-day forecast {detail} via {method} -> {signal}.",
    }


def manager_mock(ticker: str, payload: dict) -> dict:
    """Fast depth skips the debate, so the net comes from the research signals."""
    bull = payload.get("bull")
    bear = payload.get("bear")
    if isinstance(bull, dict) and bull and isinstance(bear, dict) and bear:
        bull_score = bull.get("confidence", 0.5)
        bear_score = bear.get("confidence", 0.5)
        net = bull_score - bear_score
        basis = f"Bull {bull_score:.2f} vs bear {bear_score:.2f}"
    else:
        bull_score = bear_score = 0.5
        net = 0.0
        for key in ("market", "technical", "fundamental", "news", "sentiment", "forecast"):
            entry = payload.get(key)
            if not isinstance(entry, dict):
                continue
            signal = str(entry.get("signal") or "").lower()
            if signal in ("bullish", "positive"):
                net += 1
            elif signal in ("bearish", "negative"):
                net -= 1
        net = round(net / 2, 2)
        basis = "Research net (no debate)"
    decision = "BUY" if net >= 0.15 else "SELL" if net <= -0.15 else "HOLD"
    probability = (
        None
        if decision == "HOLD"
        else round(min(0.95, max(0.05, 0.5 + abs(net))), 2)
    )
    return {
        "ticker": ticker,
        "decision": decision,
        "confidence": round(min(0.95, 0.5 + abs(net)), 2),
        "probability_beat_spy": probability,
        "summary": f"[mock] {basis} -> {decision}.",
        "bull_case": "[mock] See bull researcher summary.",
        "bear_case": "[mock] See bear researcher summary.",
        "would_upgrade_if": (
            "[mock] The bear case weakens below "
            f"{max(0.0, bear_score - 0.2):.2f} or fresh research flips the net score."
        ),
        "would_downgrade_if": (
            "[mock] The bull case weakens below "
            f"{max(0.0, bull_score - 0.2):.2f} or fresh research flips the net score."
        ),
    }


@dataclass(frozen=True)
class FastDecision:
    decision: str
    probability: float
    technical: dict
    forecast: dict
    manager: dict


def beat_probability(decision: str, raw: float | None) -> float:
    """Map the manager's direction-specific probability to P(beat SPY)."""
    if decision == "BUY":
        return raw if raw is not None else 0.5
    if decision == "SELL":
        return 1.0 - raw if raw is not None else 0.5
    return 0.5


def decision_for(frame: pd.DataFrame | None, ticker: str = "TEST") -> FastDecision | None:
    """The Fast profile's deterministic replay at the last bar."""
    if frame is None or len(frame) < MIN_BARS:
        return None
    closes = [float(value) for value in frame["close"]]
    highs = [float(value) for value in frame["high"]]
    lows = [float(value) for value in frame["low"]]
    volumes = [float(value) for value in frame["volume"]]
    indicators = compute_indicators(closes, highs, lows, volumes)
    if "error" in indicators:
        return None

    technical = technical_mock(ticker, indicators)
    forecast = forecast_mock(ticker, forecast_payload(indicators, len(closes)))
    manager = manager_mock(
        ticker, {"ticker": ticker, "technical": technical, "forecast": forecast}
    )
    decision = manager["decision"]
    probability = beat_probability(decision, manager.get("probability_beat_spy"))
    return FastDecision(
        decision=decision,
        probability=min(max(float(probability), 0.0), 1.0),
        technical=technical,
        forecast=forecast,
        manager=manager,
    )


def _ticker(question_id: str, question: Question) -> str:
    instructions = question.instructions
    if isinstance(instructions, dict) and instructions.get("ticker"):
        return str(instructions["ticker"]).upper()
    return question_id.rsplit("_", 1)[-1].upper()


class BetterTradingClient:
    """The Fast profile's rule replay, run against the frozen `DataService`."""

    provider = PROVIDER
    model = MODEL
    endpoint = ENDPOINT

    def __init__(self, data):
        self.data = data

    async def decide(self, state: dict, questions: dict[str, Question]) -> DcnResult:
        as_of = date.fromisoformat(str(state["as_of"]))
        by_ticker: dict[str, list[tuple[str, Question]]] = {}
        for question_id, question in questions.items():
            by_ticker.setdefault(_ticker(question_id, question), []).append((question_id, question))

        answers: dict[str, object] = {}
        missing: list[str] = []
        for ticker, ticker_questions in by_ticker.items():
            frame = self.data.window(ticker, as_of, 100_000) if self.data is not None else None
            result = decision_for(frame, ticker)
            if result is None:
                missing.extend(question_id for question_id, _ in ticker_questions)
                continue
            p = result.probability
            for question_id, question in ticker_questions:
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
