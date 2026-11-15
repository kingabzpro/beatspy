"""TPI variant study: mapping and horizon, evaluated on the three scored windows.

Research script, not part of the benchmark. Each run is deterministic and uses
the published frozen snapshots, exactly like the TPI baseline runs.
"""

from __future__ import annotations

import asyncio
import json
import math
from datetime import date
from pathlib import Path

from beatspy.config import DecisionConfig, Settings
from beatspy.data.freeze import read_snapshot
from beatspy.data.service import DataService
from beatspy.dcn import tpi
from beatspy.dcn.clients import DcnResult
from beatspy.dcn.runner import run_dcn
from beatspy.scenarios import load_scenario
from beatspy.schemas import ChoiceAnswer, NoulAnswer, Question, Scenario, ScoreAnswer

WINDOWS = {
    "2025-10-6m": {
        "snapshot": "dashboard/data/snapshots/01a0b732bcd0226fdf9d7a66f4cc52b381f6140f930e3ced4b87567e049f0128",
        "start": "2025-04-09",
        "end": "2025-10-09",
    },
    "2026-04-6m": {
        "snapshot": "dashboard/data/snapshots/f03a0b374ba481e50962b58f00062aae041eb3bf10f81319903eab863200511d",
        "start": "2025-10-09",
        "end": "2026-04-09",
    },
    "2026-6m": {
        "snapshot": "dashboard/data/snapshots/fd1dd42655bfba041c21594eb22ffc3eb684557709a36f629a149a97942fd45d",
        "start": "2026-04-09",
        "end": "2026-10-09",
    },
}
# (horizon profile, score-to-probability mapping)
VARIANTS = (
    ("short", "gate"),
    ("short", "score5"),
    ("short", "score10"),
    ("short", "score20"),
    ("short", "score_shrink"),
    ("short", "gate_shrink"),
    ("short", "rank"),
    ("short", "rank_shrink"),
    ("short", "equal"),
    ("medium", "gate"),
    ("medium", "score10"),
)
SCALES = {"score5": 5.0, "score10": 10.0, "score20": 20.0}


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _gate_probability(score: float, buy: bool) -> float:
    """The retired gate-aware mapping, kept so the study stays reproducible."""
    logit = (score - tpi.BUY_THRESHOLD) / tpi.LOGIT_POINTS
    scaled = _sigmoid(logit)
    return 0.5 + 0.5 * scaled if buy else 0.5 * scaled


def _probability(mapping: str, score: float, buy: bool, percentile: float) -> float:
    if mapping == "gate":
        return _gate_probability(score, buy)
    if mapping in SCALES:
        return _sigmoid((score - 50.0) / SCALES[mapping])
    if mapping == "score_shrink":
        return 0.25 + 0.5 * _sigmoid((score - 50.0) / 10.0)
    if mapping == "gate_shrink":
        return 0.5 + 0.5 * (_gate_probability(score, buy) - 0.5)
    if mapping == "rank":
        return 0.5 + 0.5 * (2.0 * percentile - 1.0)
    if mapping == "rank_shrink":
        return 0.5 + 0.25 * (2.0 * percentile - 1.0)
    if mapping == "equal":
        return 0.5
    raise ValueError(mapping)


def _medium_signal(frame):
    """Medium-horizon TPI score: 20-day breakout plus 40-day channel confirmation."""
    features = tpi._features(frame)
    if features is None:
        return None
    f = features
    trend = 0.0
    if f.close > f.ema50:
        trend += 15.0
    if f.ema50 > f.ema200:
        trend += 10.0
    if f.close > f.ema200:
        trend += 5.0
    if f.ema50_slope_pct > 0.0:
        trend += 5.0

    don20_break = f.close > f.don20_high
    don40_mid = (f.don40_high + f.don40_low) / 2.0
    breakout = 0.0
    if don20_break:
        breakout += 15.0
    if f.close > don40_mid:
        breakout += 10.0

    momentum = 0.0
    if 50.0 <= f.rsi14 <= 75.0:
        momentum += 10.0
    if not math.isnan(f.rsi14_prev3) and f.rsi14 > f.rsi14_prev3:
        momentum += 4.0
    if f.macd_hist > 0.0:
        momentum += 6.0

    confirmation = 0.0
    if f.adx14 >= 20.0:
        confirmation += 4.0
    if f.plus_di > f.minus_di:
        confirmation += 3.0
    if f.volume_ratio >= 1.0:
        confirmation += 3.0

    volatility = 0.0
    if 0.6 <= f.atr_pct <= 8.0:
        volatility += 5.0
    if -1.0 <= f.dist_ema20_atr <= 3.0:
        volatility += 5.0

    score = trend + breakout + momentum + confirmation + volatility
    if f.rsi14 > 80.0:
        score -= 12.0
    if f.atr_pct > 8.0:
        score -= 10.0
    if f.dist_ema20_atr > 3.0:
        score -= 5.0
    if f.close < f.ema200:
        score -= 10.0
    score = min(max(score, 0.0), 100.0)
    buy = (
        f.close > f.ema200
        and f.ema50 > f.ema200
        and don20_break
        and f.close > don40_mid
        and 50.0 <= f.rsi14 <= 75.0
        and 0.6 <= f.atr_pct <= 8.0
        and score >= 68.0
    )
    return tpi.TpiSignal(score=score, buy=buy)


def _signal_for(frame, profile: str):
    if profile == "short":
        return tpi.signal(frame)
    return _medium_signal(frame)


def _ticker(question_id: str, question: Question) -> str:
    instructions = question.instructions
    if isinstance(instructions, dict) and instructions.get("ticker"):
        return str(instructions["ticker"]).upper()
    return question_id.rsplit("_", 1)[-1].upper()


class TpiVariantClient:
    provider = "tpi-rs"
    model = "tpi-rs"
    endpoint = "local://tpi-rs"

    def __init__(self, data, profile: str, mapping: str):
        self.data = data
        self.profile = profile
        self.mapping = mapping

    async def decide(self, state: dict, questions: dict[str, Question]) -> DcnResult:
        as_of = date.fromisoformat(str(state["as_of"]))
        by_ticker: dict[str, list[tuple[str, Question]]] = {}
        for question_id, question in questions.items():
            by_ticker.setdefault(_ticker(question_id, question), []).append((question_id, question))

        signals: dict[str, object] = {}
        for ticker in by_ticker:
            frame = self.data.window(ticker, as_of, 100_000) if self.data is not None else None
            found = _signal_for(frame, self.profile) if frame is not None else None
            if found is not None:
                signals[ticker] = found
        ranked = sorted(signals, key=lambda ticker: (-signals[ticker].score, ticker))
        rank = {ticker: index + 1 for index, ticker in enumerate(ranked)}
        count = len(ranked)

        answers: dict[str, object] = {}
        missing: list[str] = []
        for ticker, ticker_questions in by_ticker.items():
            found = signals.get(ticker)
            if found is None:
                missing.extend(question_id for question_id, _ in ticker_questions)
                continue
            percentile = (count - rank[ticker]) / (count - 1) if count > 1 else 0.5
            probability = _probability(self.mapping, found.score, found.buy, percentile)
            probability = min(max(probability, 0.0), 1.0)
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


def _scenario(name: str, start: str, end: str) -> Scenario:
    base = load_scenario(name)
    payload = {**base.model_dump(), "start": start, "end": end}
    if name == "2026-6m":
        payload["recent_months"] = None
    return Scenario.model_validate(payload)


async def _run(scenario: Scenario, snapshot: Path, profile: str, mapping: str) -> dict:
    prices, manifest = read_snapshot(snapshot, scenario)
    data = DataService(prices, benchmark=scenario.benchmark)
    manifest = {**manifest, "path": str(snapshot.resolve())}
    client = TpiVariantClient(data, profile, mapping)
    settings = Settings(decision=DecisionConfig(decision_model="tpi-rs"))
    out = await run_dcn(
        scenario,
        settings,
        prepared_data=(data, manifest),
        out_root=Path("results/tpi_variants"),
        label=f"{profile}:{mapping}:{scenario.name}",
        client=client,
    )
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    cal = (metrics.get("calibration") or {}).get("signals", {}).get("noul", {})
    return {
        "profile": profile,
        "mapping": mapping,
        "window": scenario.name,
        "return": metrics["total_return"],
        "spy": metrics["spy_total_return"],
        "excess": metrics["excess_return_vs_spy"],
        "equal_weight": (metrics.get("baselines") or {}).get("equal_weight"),
        "brier": cal.get("brier"),
        "skill": cal.get("brier_skill_score"),
        "auc": cal.get("auc"),
    }


async def main() -> None:
    rows = []
    for profile, mapping in VARIANTS:
        for name, info in WINDOWS.items():
            scenario = _scenario(name, info["start"], info["end"])
            row = await _run(scenario, Path(info["snapshot"]), profile, mapping)
            rows.append(row)
            print(
                f"{profile:6s} {mapping:13s} {name:12s} ret {row['return']:+7.2%} "
                f"excess {row['excess']:+7.2%} vsEW {row['return'] - row['equal_weight']:+7.2%} "
                f"brier {row['brier']:.4f} skill {row['skill']:+.4f} auc {row['auc']:.4f}",
                flush=True,
            )
    print("\nSUMMARY")
    for profile, mapping in VARIANTS:
        subset = [row for row in rows if row["profile"] == profile and row["mapping"] == mapping]
        average = sum(row["excess"] for row in subset) / len(subset)
        vs_equal = sum(row["return"] - row["equal_weight"] for row in subset) / len(subset)
        skill = sum(row["skill"] for row in subset) / len(subset)
        print(
            f"{profile:6s} {mapping:13s} avg excess {average:+.2%} "
            f"avg vs equal {vs_equal:+.2%} avg Brier skill {skill:+.4f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
