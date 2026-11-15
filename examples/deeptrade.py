"""DeepTrade: candles + statistics + TPI, one structured LLM call per ticker.

DeepTrade builds a point-in-time dossier from a frozen BeatSPY snapshot:
daily candles, statistical features, the TPI short-horizon score and its
indicator block, the universe base rate, and the ticker's cross-sectional
rank. One chat call per ticker asks the model for BUY, HOLD or SELL and a
probability that the ticker beats SPY over the next five trading days.

Structured output uses the OpenAI SDK's Pydantic parse. The configured
DeepSeek endpoint returns the JSON in ``reasoning_content`` rather than
``content``, so the script validates that field when ``parsed`` is None.

Live mode needs the OpenAI SDK; dry-run does not:

    uv run --with openai python examples/deeptrade.py \
        --snapshot dashboard/data/snapshots/f03a0b37... --date 2026-03-06 \
        --tickers AAPL,MSFT,NVDA

    uv run python examples/deeptrade.py --snapshot ... --date ... --dry-run

Environment: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY. Flags override them.
Only bars dated on or before --date are read.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
import warnings
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from beatspy.dcn.tpi import MIN_BARS, _features, signal

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

SYSTEM = """You are DeepTrade, a disciplined systematic trading analyst.

You receive one point-in-time dossier built from historical daily candles,
statistical features and the TPI trend/breakout score. Decide whether the
ticker will beat SPY over the next 5 trading days.

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


def _ticker_frame(prices: pd.DataFrame, ticker: str, as_of: pd.Timestamp) -> pd.DataFrame | None:
    rows = prices[prices["ticker"] == ticker]
    if rows.empty:
        return None
    frame = rows.set_index("date").sort_index()
    frame = frame[frame.index <= as_of]
    if len(frame) < MIN_BARS:
        return None
    return frame


def _base_rate(prices: pd.DataFrame, as_of: pd.Timestamp) -> float | None:
    """Share of the snapshot universe that beat SPY over the prior 63 sessions."""
    spy = prices[prices["ticker"] == "SPY"].set_index("date").sort_index()
    spy = spy[spy.index <= as_of]
    if len(spy) <= 63:
        return None
    spy_return = spy["close"].iloc[-1] / spy["close"].iloc[-64] - 1.0
    beat = total = 0
    for ticker in sorted(set(prices["ticker"]) - {"SPY"}):
        rows = prices[prices["ticker"] == ticker].set_index("date").sort_index()
        rows = rows[rows.index <= as_of]
        if len(rows) <= 63:
            continue
        total += 1
        if rows["close"].iloc[-1] / rows["close"].iloc[-64] - 1.0 > spy_return:
            beat += 1
    return _round(100.0 * beat / total, 1) if total else None


def _percentile(value: float, values: list[float]) -> float:
    if len(values) <= 1:
        return 50.0
    rank = sum(other <= value for other in values)
    return _round(100.0 * (rank - 1) / (len(values) - 1), 1)


def _dossier(prices: pd.DataFrame, ticker: str, as_of: pd.Timestamp) -> dict | None:
    frame = _ticker_frame(prices, ticker, as_of)
    if frame is None:
        return None
    close = frame["close"].astype(float)
    features = _features(frame)
    if features is None:
        return None
    tpi = signal(frame)
    if tpi is None:
        return None
    doc = asdict(features)
    return {
        "ticker": ticker,
        "requested_date": as_of.date().isoformat(),
        "last_bar": frame.index[-1].date().isoformat(),
        "horizon_days": 5,
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
    }


def _add_context(dossiers: dict[str, dict], base_rate: float | None) -> None:
    scores = [doc["tpi"]["score"] for doc in dossiers.values()]
    returns = [doc["returns_pct"]["20d"] for doc in dossiers.values() if doc["returns_pct"]["20d"] is not None]
    for doc in dossiers.values():
        ret20 = doc["returns_pct"]["20d"]
        doc["base_rate_pct"] = base_rate
        doc["cross_section"] = {
            "tpi_percentile": _percentile(doc["tpi"]["score"], scores),
            "return_20d_percentile": _percentile(ret20, returns) if ret20 is not None else None,
            "tickers_in_slice": len(dossiers),
        }


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


def _beat_probability(decision: Decision) -> float:
    if decision.decision == "BUY":
        return min(max(decision.probability_beat_spy or 0.5, 0.0), 1.0)
    if decision.decision == "SELL":
        return min(max(1.0 - (decision.probability_beat_spy or 0.5), 0.0), 1.0)
    return 0.5


async def _live_one(client, semaphore, model: str, doc: dict, horizon: int):
    ticker = doc["ticker"]
    prompt = USER.format(
        dossier=json.dumps(doc, indent=2, sort_keys=True, default=str),
        horizon=horizon,
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    async with semaphore:
        started = time.perf_counter()
        try:
            completion = await client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=Decision,
                temperature=0.0,
                max_tokens=600,
            )
            message = completion.choices[0].message
            parsed = message.parsed
            if parsed is None:
                raw = (
                    message.content
                    or getattr(message, "reasoning_content", None)
                    or (getattr(message, "model_extra", None) or {}).get("reasoning_content")
                    or ""
                )
                parsed = _normalise(json.loads(_json_text(raw)), ticker)
            usage = completion.usage
            return parsed, usage, time.perf_counter() - started, "parse"
        except Exception:
            # Some OpenAI-compatible providers reject the strict schema. Fall
            # back to json_object and validate the answer ourselves.
            completion = await client.chat.completions.create(
                model=model,
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
            parsed = _normalise(json.loads(_json_text(raw)), ticker)
            usage = completion.usage
            return parsed, usage, time.perf_counter() - started, "json_object"


async def _live_all(dossiers: list[dict], args) -> list[dict]:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise SystemExit(
            "DeepTrade live mode needs the OpenAI SDK. Run it with: "
            "uv run --with openai python examples/deeptrade.py ..."
        ) from None

    base_url = args.base_url or os.environ.get("LLM_BASE_URL")
    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    model = args.model or os.environ.get("LLM_MODEL")
    if not (base_url and api_key and model):
        raise SystemExit("DeepTrade live mode needs LLM_BASE_URL, LLM_API_KEY and LLM_MODEL (or flags).")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=90.0)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    try:
        outcomes = await asyncio.gather(
            *(_live_one(client, semaphore, model, doc, args.horizon) for doc in dossiers),
            return_exceptions=True,
        )
    finally:
        await client.close()

    rows = []
    for doc, outcome in zip(dossiers, outcomes, strict=True):
        if isinstance(outcome, Exception):
            rows.append(
                {
                    "date": doc["requested_date"],
                    "ticker": doc["ticker"],
                    "decision": "ERROR",
                    "confidence": None,
                    "model_probability": None,
                    "probability_beat_spy": None,
                    "reason": f"{type(outcome).__name__}: {outcome}",
                    "mode": "live",
                }
            )
            continue
        decision, usage, latency, path = outcome
        rows.append(
            {
                "date": doc["requested_date"],
                "ticker": doc["ticker"],
                "decision": decision.decision,
                "confidence": _round(decision.confidence),
                "model_probability": _round(decision.probability_beat_spy),
                "probability_beat_spy": _round(_beat_probability(decision)),
                "reason": decision.reason,
                "mode": f"live/{path}",
                "latency_s": _round(latency, 2),
                "input_tokens": getattr(usage, "prompt_tokens", 0),
                "output_tokens": getattr(usage, "completion_tokens", 0),
            }
        )
    return rows


def _dry_one(doc: dict) -> dict:
    score = doc["tpi"]["score"]
    if doc["tpi"]["buy"]:
        decision = "BUY"
    elif score <= 40:
        decision = "SELL"
    else:
        decision = "HOLD"
    probability = 0.5 + (score - 50.0) / 100.0
    return {
        "date": doc["requested_date"],
        "ticker": doc["ticker"],
        "decision": decision,
        "confidence": _round(0.5 + abs(score - 50.0) / 100.0),
        "model_probability": _round(probability),
        "probability_beat_spy": _round(probability),
        "reason": f"[dry-run] TPI score {score:.1f}, buy gate {doc['tpi']['buy']}",
        "mode": "dry-run",
    }


def _print_rows(rows: list[dict]) -> None:
    print(f"{'ticker':8s} {'decision':9s} {'p(beat)':>8s} {'conf':>6s}  reason")
    for row in rows:
        probability = row["probability_beat_spy"]
        confidence = row["confidence"]
        print(
            f"{row['ticker']:8s} {row['decision']:9s} "
            f"{probability if probability is not None else 'n/a':>8} "
            f"{confidence if confidence is not None else 'n/a':>6}  {str(row['reason'])[:90]}"
        )


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "date", "ticker", "decision", "confidence", "model_probability",
        "probability_beat_spy", "reason", "mode", "latency_s", "input_tokens", "output_tokens",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepTrade: candles + stats + TPI, one LLM call per ticker.")
    parser.add_argument("--snapshot", required=True, help="frozen snapshot directory with prices.csv")
    parser.add_argument("--date", required=True, help="decision date, YYYY-MM-DD")
    parser.add_argument("--tickers", help="comma-separated tickers (default: every snapshot ticker except SPY)")
    parser.add_argument("--horizon", type=int, default=5, help="decision horizon in trading days (default 5)")
    parser.add_argument("--out", help="write the decisions to this CSV")
    parser.add_argument("--dry-run", action="store_true", help="use the TPI rule instead of the LLM")
    parser.add_argument("--concurrency", type=int, default=12, help="live calls in flight (default 12)")
    parser.add_argument("--model", help="override LLM_MODEL")
    parser.add_argument("--base-url", help="override LLM_BASE_URL")
    parser.add_argument("--api-key", help="override LLM_API_KEY")
    args = parser.parse_args()

    as_of = pd.Timestamp(date.fromisoformat(args.date))
    prices_path = Path(args.snapshot) / "prices.csv"
    if not prices_path.is_file():
        raise SystemExit(f"missing snapshot prices: {prices_path}")
    prices = pd.read_csv(prices_path, parse_dates=["date"])
    available = sorted(set(prices["ticker"]) - {"SPY"})
    tickers = (
        [ticker.strip().upper() for ticker in args.tickers.split(",") if ticker.strip()]
        if args.tickers
        else available
    )
    unknown = [ticker for ticker in tickers if ticker not in available]
    if unknown:
        raise SystemExit(f"tickers not in the snapshot: {', '.join(unknown)}")

    dossiers = {}
    skipped = []
    for ticker in tickers:
        doc = _dossier(prices, ticker, as_of)
        if doc is None:
            skipped.append(ticker)
        else:
            dossiers[ticker] = doc
    if not dossiers:
        raise SystemExit("no ticker has enough history for DeepTrade")
    _add_context(dossiers, _base_rate(prices, as_of))

    docs = [dossiers[ticker] for ticker in tickers if ticker in dossiers]
    rows = (
        [_dry_one(doc) for doc in docs]
        if args.dry_run
        else asyncio.run(_live_all(docs, args))
    )

    _print_rows(rows)
    print(f"\n{len(rows)} decisions, {len(skipped)} skipped (under {MIN_BARS} bars): {', '.join(skipped) or 'none'}")
    if args.out:
        _write_csv(rows, Path(args.out))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
