"""DeepTrade: candles + statistics + TPI, one structured LLM call per ticker.

The dossier is built point-in-time from the frozen `DataService`: candle-derived
returns, volatility, 52-week distance, the TPI short-horizon score and its
indicator block, the universe base rate, and the ticker's cross-sectional
percentile. One OpenAI SDK call per ticker returns BUY, HOLD or SELL plus a
direction probability. The app's configured DeepSeek endpoint returns the
structured JSON in ``reasoning_content`` rather than ``content``, so this
validates that field when the SDK's ``parsed`` is None.

Live runs need the OpenAI SDK; the rest of the benchmark does not:

    uv run --with openai beatspy run --model deeptrade ...
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import warnings
from dataclasses import asdict
from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from ..schemas import ChoiceAnswer, NoulAnswer, Question, ScoreAnswer
from .clients import DcnError, DcnResult
from .tpi import MIN_BARS, _features, signal

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

PROVIDER = "deeptrade"
MODEL = "deeptrade"
DEFAULT_ENDPOINT = "https://api.openai.com/v1"
HORIZON = 5

SYSTEM = """You are DeepTrade, a disciplined systematic trading analyst.

You receive one point-in-time dossier built from historical daily candles,
statistical features and the TPI trend/breakout score. Decide whether the
ticker will beat SPY over the next horizon.

- BUY means the ticker will outperform SPY.
- SELL means it will underperform SPY.
- HOLD means there is no clear edge.

Use only the dossier. Treat missing values as unavailable, never invent
facts. BUY and SELL need a probability for the chosen direction; HOLD is
neutral."""

USER = """Dossier:
{dossier}

Return a JSON object with:
- ticker: the ticker
- decision: BUY, HOLD or SELL
- confidence: 0.0 to 1.0, the strength of the evidence
- probability_beat_spy: P(outperform) for BUY, P(underperform) for SELL, 0.5 for HOLD
- reason: one short sentence naming the decisive evidence

Decide for the next {horizon} trading days against SPY."""


class Decision(BaseModel):
    """The structured answer DeepTrade asks every model for."""

    ticker: str = Field(description="The ticker the decision is about")
    decision: Literal["BUY", "HOLD", "SELL"]
    confidence: float = Field(ge=0.0, le=1.0, description="Evidence strength, 0 to 1")
    probability_beat_spy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="P(outperform) for BUY, P(underperform) for SELL, 0.5 for HOLD",
    )
    reason: str = Field(default="", description="One short sentence with the decisive evidence")


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def _pct_change(close: pd.Series, n: int) -> float | None:
    if len(close) <= n:
        return None
    return _round((close.iloc[-1] / close.iloc[-1 - n] - 1.0) * 100.0)


def _json_text(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the model response")
    return text[start:]


def _normalise(payload: dict, ticker: str) -> Decision:
    body = dict(payload)
    body["ticker"] = str(body.get("ticker") or ticker).upper()
    body["decision"] = str(body.get("decision", "")).strip().upper()
    decision = Decision.model_validate(body)
    if decision.probability_beat_spy is None and decision.decision != "HOLD":
        decision.probability_beat_spy = 0.5
    return decision


def beat_probability(decision: Decision) -> float:
    """Map the direction-specific probability to P(beat SPY)."""
    if decision.decision == "BUY":
        return min(max(decision.probability_beat_spy or 0.5, 0.0), 1.0)
    if decision.decision == "SELL":
        return min(max(1.0 - (decision.probability_beat_spy or 0.5), 0.0), 1.0)
    return 0.5


def _percentile(value: float, values: list[float]) -> float:
    if len(values) <= 1:
        return 50.0
    rank = sum(other <= value for other in values)
    return _round(100.0 * (rank - 1) / (len(values) - 1), 1)


def base_rate(data, tickers: list[str], as_of: date) -> float | None:
    """Share of the requested universe that beat SPY over the prior 63 sessions."""
    spy = data.window("SPY", as_of, 100_000)
    if spy is None or len(spy) <= 63:
        return None
    spy_return = spy["close"].iloc[-1] / spy["close"].iloc[-64] - 1.0
    beat = total = 0
    for ticker in tickers:
        frame = data.window(ticker, as_of, 100_000)
        if frame is None or len(frame) <= 63:
            continue
        total += 1
        if frame["close"].iloc[-1] / frame["close"].iloc[-64] - 1.0 > spy_return:
            beat += 1
    return _round(100.0 * beat / total, 1) if total else None


def dossier(
    frame: pd.DataFrame,
    ticker: str,
    as_of: date,
    *,
    base: float | None,
    cross_section: dict,
    horizon: int = HORIZON,
) -> dict | None:
    close = frame["close"].astype(float)
    features = _features(frame)
    tpi = signal(frame)
    if features is None or tpi is None:
        return None
    doc = asdict(features)
    return {
        "ticker": ticker,
        "requested_date": as_of.isoformat(),
        "last_bar": frame.index[-1].date().isoformat(),
        "horizon_days": horizon,
        "benchmark": "SPY",
        "returns_pct": {
            "1d": _pct_change(close, 1),
            "5d": _pct_change(close, 5),
            "20d": _pct_change(close, 20),
            "60d": _pct_change(close, 60),
        },
        "volatility_20d_pct": _round(close.pct_change().tail(20).std() * math.sqrt(252) * 100.0),
        "off_52w_high_pct": _round((close.iloc[-1] / close.tail(252).max() - 1.0) * 100.0),
        "tpi": {
            "score": _round(tpi.score),
            "buy": bool(tpi.buy),
            "ema20": _round(doc["ema20"]),
            "ema50": _round(doc["ema50"]),
            "ema200": _round(doc["ema200"]),
            "ema50_slope_pct": _round(doc["ema50_slope_pct"]),
            "rsi14": _round(doc["rsi14"]),
            "atr14": _round(doc["atr14"]),
            "atr_pct": _round(doc["atr_pct"]),
            "adx14": _round(doc["adx14"]),
            "plus_di": _round(doc["plus_di"]),
            "minus_di": _round(doc["minus_di"]),
            "macd_hist": _round(doc["macd_hist"]),
            "don20_high": _round(doc["don20_high"]),
            "don40_high": _round(doc["don40_high"]),
            "don40_low": _round(doc["don40_low"]),
            "volume_ratio": _round(doc["volume_ratio"]),
            "dist_ema20_atr": _round(doc["dist_ema20_atr"]),
        },
        "base_rate_pct": base,
        "cross_section": cross_section,
    }


def _ticker(question_id: str, question: Question) -> str:
    instructions = question.instructions
    if isinstance(instructions, dict) and instructions.get("ticker"):
        return str(instructions["ticker"]).upper()
    return question_id.rsplit("_", 1)[-1].upper()


class DeepTradeClient:
    """One LLM decision per ticker over the point-in-time DeepTrade dossier."""

    provider = PROVIDER
    model = MODEL
    endpoint = DEFAULT_ENDPOINT

    def __init__(self, data, *, endpoint: str | None = None, concurrency: int = 12):
        self.data = data
        self.endpoint = os.environ.get("LLM_BASE_URL") or endpoint or DEFAULT_ENDPOINT
        self.model = os.environ.get("LLM_MODEL") or MODEL
        self._api_key = os.environ.get("LLM_API_KEY")
        if not self._api_key:
            raise DcnError(
                "DeepTrade needs LLM_API_KEY (and usually LLM_BASE_URL and LLM_MODEL)."
            )
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._client = None

    def _openai(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise DcnError(
                    "DeepTrade live mode needs the OpenAI SDK; run it with "
                    "`uv run --with openai beatspy run ...`."
                ) from None
            self._client = AsyncOpenAI(
                base_url=self.endpoint, api_key=self._api_key, timeout=90.0
            )
        return self._client

    async def _one(self, client, doc: dict, horizon: int):
        prompt = USER.format(
            dossier=json.dumps(doc, indent=2, sort_keys=True, default=str),
            horizon=horizon,
        )
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]
        async with self._semaphore:
            started = time.perf_counter()
            try:
                completion = await client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=Decision,
                    temperature=0.0,
                    max_tokens=600,
                )
                message = completion.choices[0].message
                decision = message.parsed
                if decision is None:
                    raw = (
                        message.content
                        or getattr(message, "reasoning_content", None)
                        or (getattr(message, "model_extra", None) or {}).get("reasoning_content")
                        or ""
                    )
                    decision = _normalise(json.loads(_json_text(raw)), doc["ticker"])
                return decision, completion.usage, time.perf_counter() - started
            except Exception:
                # Some compatible providers reject the strict schema. Fall
                # back to json_object and validate the answer ourselves.
                completion = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=600,
                )
                message = completion.choices[0].message
                raw = (
                    message.content
                    or getattr(message, "reasoning_content", None)
                    or (getattr(message, "model_extra", None) or {}).get("reasoning_content")
                    or ""
                )
                decision = _normalise(json.loads(_json_text(raw)), doc["ticker"])
                return decision, completion.usage, time.perf_counter() - started

    async def decide(self, state: dict, questions: dict[str, Question]) -> DcnResult:
        as_of = date.fromisoformat(str(state["as_of"]))
        horizon = int(state.get("horizon_trading_days") or HORIZON)
        by_ticker: dict[str, list[tuple[str, Question]]] = {}
        for question_id, question in questions.items():
            by_ticker.setdefault(_ticker(question_id, question), []).append((question_id, question))
        tickers = list(by_ticker)

        frames: dict[str, pd.DataFrame] = {}
        scored: dict[str, dict] = {}
        for ticker in tickers:
            frame = self.data.window(ticker, as_of, 100_000) if self.data is not None else None
            if frame is None or len(frame) < MIN_BARS:
                continue
            tpi = signal(frame)
            if tpi is None:
                continue
            frames[ticker] = frame
            scored[ticker] = {"score": tpi.score, "ret20": _pct_change(frame["close"], 20)}

        base = base_rate(self.data, tickers, as_of) if self.data is not None else None
        scores = [info["score"] for info in scored.values()]
        returns = [info["ret20"] for info in scored.values() if info["ret20"] is not None]
        docs = []
        for ticker, frame in frames.items():
            ret20 = scored[ticker]["ret20"]
            cross = {
                "tpi_percentile": _percentile(scored[ticker]["score"], scores),
                "return_20d_percentile": _percentile(ret20, returns) if ret20 is not None else None,
                "tickers_in_slice": len(frames),
            }
            doc = dossier(frame, ticker, as_of, base=base, cross_section=cross, horizon=horizon)
            if doc is not None:
                docs.append(doc)

        if not docs:
            return DcnResult(
                answers={},
                missing_ids=list(questions),
                input_tokens=0,
                output_tokens=0,
                latency_s=0.0,
                attempts=1,
                requests=0,
            )

        client = self._openai()
        started = time.perf_counter()
        outcomes = await asyncio.gather(
            *(self._one(client, doc, horizon) for doc in docs), return_exceptions=True
        )
        elapsed = time.perf_counter() - started

        by_doc = {doc["ticker"]: outcome for doc, outcome in zip(docs, outcomes, strict=True)}
        answers: dict[str, object] = {}
        missing: list[str] = []
        input_tokens = output_tokens = 0
        for ticker, ticker_questions in by_ticker.items():
            outcome = by_doc.get(ticker)
            if outcome is None or isinstance(outcome, Exception):
                missing.extend(question_id for question_id, _ in ticker_questions)
                continue
            decision, usage, _ = outcome
            probability = beat_probability(decision)
            input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
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
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=elapsed,
            attempts=1,
            requests=len(docs),
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
