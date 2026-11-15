"""PaperRL's five-factor signal on BeatSPY's daily bars, with the PPO overlay.

Ported from PaperRL (MIT), ``crates/paperrl-trade/src/lib.rs::recommend``. The
PPO target weights in ``paperrl_ppo.json`` were dumped from PaperRL's own
``PpoStrategy`` over the frozen snapshots: newest per-symbol checkpoint, seed
42, position and drawdown at zero, last 120 bars, exactly as the app's
``latest_model_weight`` does. The overlay adds +0.35 at a target weight of at
least 60% and subtracts 0.35 at 20% or less.

The checkpoints were trained over a range that overlaps these windows, so the
PPO part is not strictly out-of-sample. The app also only has checkpoints for
some symbols; the rest use the five-factor vote alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ..schemas import ChoiceAnswer, NoulAnswer, Question, ScoreAnswer
from .clients import DcnResult

MIN_BARS = 60
PROVIDER = "paperrl"
MODEL = "paperrl"
ENDPOINT = "local://paperrl"
PPO_SEED = 42
PPO_TABLE = Path(__file__).with_name("paperrl_ppo.json")


def ema(closes: list[float], n: int) -> list[float]:
    """EMA seeded with the SMA of the first n closes."""
    if n == 0 or len(closes) < n:
        return []
    out = [sum(closes[:n]) / n]
    k = 2.0 / (n + 1.0)
    for close in closes[n:]:
        out.append(out[-1] + k * (close - out[-1]))
    return out


def rsi(closes: list[float], n: int = 14) -> float:
    """Wilder's RSI; neutral 50 when there is not enough history."""
    if n == 0 or len(closes) < n + 1:
        return 50.0
    avg_gain = 0.0
    avg_loss = 0.0
    for prev, cur in zip(closes[:n], closes[1 : n + 1], strict=True):
        delta = cur - prev
        if delta >= 0.0:
            avg_gain += delta
        else:
            avg_loss -= delta
    avg_gain /= n
    avg_loss /= n
    for prev, cur in zip(closes[n:-1], closes[n + 1 :], strict=True):
        delta = cur - prev
        gain, loss = (delta, 0.0) if delta >= 0.0 else (0.0, -delta)
        avg_gain = (avg_gain * (n - 1.0) + gain) / n
        avg_loss = (avg_loss * (n - 1.0) + loss) / n
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd_hist(closes: list[float]) -> float:
    fast = ema(closes, 12)
    slow = ema(closes, 26)
    if not fast or not slow:
        return 0.0
    size = min(len(fast), len(slow))
    macd_line = [f - s for f, s in zip(fast[-size:], slow[-size:], strict=True)]
    signal = ema(macd_line, 9)
    if not signal:
        return 0.0
    return macd_line[-1] - signal[-1]


def load_ppo_table() -> dict:
    """The PPO target weights per window, keyed by ticker and date."""
    if not PPO_TABLE.is_file():
        return {}
    data = json.loads(PPO_TABLE.read_text(encoding="utf-8"))
    return data.get("windows") or {}


@dataclass(frozen=True)
class PaperRLSignal:
    action: str
    score: float
    confidence: float
    probability: float
    indicators: dict
    reasons: list[str]
    ppo_weight: float | None = None


def recommend(
    frame: pd.DataFrame | None,
    ticker: str = "TEST",
    ppo_weight: float | None = None,
) -> PaperRLSignal | None:
    """The five-factor verdict plus the PPO overlay at the last bar."""
    if frame is None or len(frame) < MIN_BARS:
        return None
    closes = [float(value) for value in frame["close"]]
    price = closes[-1]
    change_pct = (closes[-1] / closes[-2] - 1.0) * 100.0
    ema20 = ema(closes, 20)[-1]
    ema50 = ema(closes, 50)[-1]
    rsi14 = rsi(closes, 14)
    hist = macd_hist(closes)
    ret20 = closes[-1] / closes[-21] - 1.0
    ret5 = closes[-1] / closes[-6] - 1.0

    indicators = {
        "price": price,
        "change_pct": change_pct,
        "ema20": ema20,
        "ema50": ema50,
        "rsi14": rsi14,
        "macd_hist": hist,
        "ret5": ret5,
        "ret20": ret20,
    }

    score = 0.0
    reasons: list[str] = []
    trend = max(-1.0, min(1.0, ((price / ema50 - 1.0) / 0.02)))
    score += 0.25 * trend
    if trend > 0.1:
        reasons.append(f"Price {price:.2f} is above the 50-day EMA ({ema50:.2f})")
    elif trend < -0.1:
        reasons.append(f"Price {price:.2f} is below the 50-day EMA ({ema50:.2f})")

    cross = max(-1.0, min(1.0, ((ema20 / ema50 - 1.0) / 0.01)))
    score += 0.20 * cross
    if cross > 0.1:
        reasons.append("EMA20 is above EMA50 (uptrend)")
    elif cross < -0.1:
        reasons.append("EMA20 is below EMA50 (downtrend)")

    momentum = max(-1.0, min(1.0, ret20 / 0.10))
    score += 0.25 * momentum
    if abs(ret20) > 0.02:
        reasons.append(f"20-day return is {ret20 * 100.0:+.1f}%")

    macd = max(-1.0, min(1.0, hist / (price * 0.01)))
    score += 0.15 * macd
    if hist > 0.0:
        reasons.append("MACD histogram is positive")
    elif hist < 0.0:
        reasons.append("MACD histogram is negative")

    if rsi14 >= 70.0:
        rsi_factor = -min((rsi14 - 70.0) / 30.0, 1.0)
    elif rsi14 <= 30.0:
        rsi_factor = min((30.0 - rsi14) / 30.0, 1.0)
    else:
        rsi_factor = 0.0
    score += 0.15 * rsi_factor
    if rsi14 >= 70.0:
        reasons.append(f"RSI is {rsi14:.0f} (overbought)")
    elif rsi14 <= 30.0:
        reasons.append(f"RSI is {rsi14:.0f} (oversold)")

    resolved_ppo = None if ppo_weight is None else float(ppo_weight)
    if resolved_ppo is not None:
        if resolved_ppo >= 0.6:
            score += 0.35
            reasons.append(
                f"PPO model (seed {PPO_SEED}) targets {resolved_ppo * 100.0:.0f}% of equity"
            )
        elif resolved_ppo <= 0.2:
            score -= 0.35
            reasons.append(
                f"PPO model (seed {PPO_SEED}) stays near cash ({resolved_ppo * 100.0:.0f}% weight)"
            )
        else:
            reasons.append(
                f"PPO model (seed {PPO_SEED}) is neutral ({resolved_ppo * 100.0:.0f}% weight)"
            )

    score = max(-1.0, min(1.0, score))
    action = "Buy" if score >= 0.3 else "Sell" if score <= -0.3 else "Hold"
    confidence = 0.5 + abs(score) / 2.0
    return PaperRLSignal(
        action=action,
        score=score,
        confidence=confidence,
        probability=0.5 + score / 2.0,
        indicators=indicators,
        reasons=reasons,
        ppo_weight=resolved_ppo,
    )


def _ticker(question_id: str, question: Question) -> str:
    instructions = question.instructions
    if isinstance(instructions, dict) and instructions.get("ticker"):
        return str(instructions["ticker"]).upper()
    return question_id.rsplit("_", 1)[-1].upper()


class PaperRLClient:
    """The five-factor rule plus the PPO overlay, against the frozen data."""

    provider = PROVIDER
    model = MODEL
    endpoint = ENDPOINT

    def __init__(self, data, ppo_table: dict | None = None):
        self.data = data
        table = load_ppo_table() if ppo_table is None else ppo_table
        # Flatten the window sections by date. The windows touch at their
        # boundaries, and a decision on a boundary date belongs to the newer
        # section, so a date lookup is less ambiguous than a range lookup.
        self._weights: dict[str, dict[str, float]] = {}
        for section in table.values():
            for ticker, dates in (section.get("weights") or {}).items():
                self._weights.setdefault(ticker, {}).update(dates)

    async def decide(self, state: dict, questions: dict[str, Question]) -> DcnResult:
        as_of = date.fromisoformat(str(state["as_of"]))
        by_ticker: dict[str, list[tuple[str, Question]]] = {}
        for question_id, question in questions.items():
            by_ticker.setdefault(_ticker(question_id, question), []).append((question_id, question))

        answers: dict[str, object] = {}
        missing: list[str] = []
        for ticker, ticker_questions in by_ticker.items():
            frame = self.data.window(ticker, as_of, 100_000) if self.data is not None else None
            ppo_weight = None
            if frame is not None and len(frame) > 0:
                last_date = frame.index[-1].date().isoformat()
                ppo_weight = (self._weights.get(ticker) or {}).get(last_date)
            signal = recommend(frame, ticker, ppo_weight)
            if signal is None:
                missing.extend(question_id for question_id, _ in ticker_questions)
                continue
            probability = signal.probability
            for question_id, question in ticker_questions:
                if question.type == "noul":
                    answers[question_id] = NoulAnswer(noul=probability)
                elif question.type == "choice":
                    answers[question_id] = ChoiceAnswer(
                        choice="outperform" if probability >= 0.5 else "underperform",
                        probabilities={
                            "outperform": probability,
                            "underperform": 1.0 - probability,
                        },
                    )
                else:
                    levels = question.criteria or []
                    answers[question_id] = ScoreAnswer(
                        score=probability * (len(levels) - 1),
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
