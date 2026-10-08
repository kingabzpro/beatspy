"""Calibration and discrimination for decision-model probabilities.

The decision models return a probability per question, so the natural question
is not "did the portfolio go up" but "were the probabilities right". Everything
here is standard and deterministic:

- **Brier score** — mean squared error of the probability. Lower is better.
- **Brier skill score** — 1 - brier/brier_base, where the base model always
  predicts the sample rate. 0 means "no better than a constant"; negative is
  worse than a constant.
- **Log loss** — clipped at 1e-15 so a confident miss is expensive but finite.
- **AUC** — rank-based discrimination via the Mann-Whitney U statistic, so large
  numbers of tied probabilities are handled correctly instead of inflating AUC.
- **Reliability bins** — 10 equal-width bins with observed rates and counts, plus
  the count-weighted expected and maximum calibration error.

`miscalibration_gap` compares the noul and choice forms of the same question,
which is the specific failure mode reported for these models: the choice form
piles probability on its favourite and moves when options are reordered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..schemas import CalibrationReport, ReliabilityBin, SignalStats

CLIP = 1e-15
BIN_COUNT = 10


@dataclass
class Observation:
    """One scored prediction: a probability and the realized binary outcome."""

    ticker: str
    # noul probability that the ticker beats the benchmark
    noul: float | None = None
    # the same question asked as a choice
    choice: float | None = None
    # the five-level score form, normalized to 0-1
    rank: float | None = None
    # realized: did the ticker beat the benchmark over the horizon?
    outcome: int | None = None
    # excess return actually realized, for diagnostics
    excess_return: float | None = None

    @property
    def scored(self) -> bool:
        return self.outcome is not None


def _clip(value: float) -> float:
    return min(max(float(value), CLIP), 1.0 - CLIP)


def brier(pairs: list[tuple[float, int]]) -> float:
    return sum((probability - outcome) ** 2 for probability, outcome in pairs) / len(pairs)


def log_loss(pairs: list[tuple[float, int]]) -> float:
    total = 0.0
    for probability, outcome in pairs:
        p = _clip(probability)
        total += -(math.log(p) if outcome else math.log(1.0 - p))
    return total / len(pairs)


def auc(pairs: list[tuple[float, int]]) -> float | None:
    """Rank-based AUC with ties counted as half, via the Mann-Whitney U statistic."""
    positives = [(p, o) for p, o in pairs if o == 1]
    negatives = [(p, o) for p, o in pairs if o == 0]
    if not positives or not negatives:
        return None  # a single class has no discrimination to measure
    wins = 0.0
    for p_pos, _ in positives:
        for p_neg, _ in negatives:
            if p_pos > p_neg:
                wins += 1.0
            elif p_pos == p_neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def reliability(pairs: list[tuple[float, int]]) -> tuple[list[ReliabilityBin], float, float]:
    """Return (bins, expected calibration error, maximum calibration error)."""
    bins: list[ReliabilityBin] = []
    ece = 0.0
    mce = 0.0
    total = len(pairs)
    for index in range(BIN_COUNT):
        lower = index / BIN_COUNT
        upper = (index + 1) / BIN_COUNT
        in_bin = [
            (p, o)
            for p, o in pairs
            if (lower <= p < upper) or (index == BIN_COUNT - 1 and p == 1.0)
        ]
        if not in_bin:
            bins.append(ReliabilityBin(lower=lower, upper=upper, count=0, mean_predicted=0.0, observed_rate=None))
            continue
        mean_predicted = sum(p for p, _ in in_bin) / len(in_bin)
        observed = sum(o for _, o in in_bin) / len(in_bin)
        gap = abs(mean_predicted - observed)
        ece += gap * len(in_bin) / total
        mce = max(mce, gap)
        bins.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=len(in_bin),
                mean_predicted=round(mean_predicted, 6),
                observed_rate=round(observed, 6),
            )
        )
    return bins, round(ece, 6), round(mce, 6)


def signal_stats(signal: str, pairs: list[tuple[float, int]]) -> SignalStats:
    """Full calibration/discrimination summary for one question form."""
    if not pairs:
        return SignalStats(signal=signal, n=0)
    base_rate = sum(outcome for _, outcome in pairs) / len(pairs)
    brier_score = brier(pairs)
    brier_base = brier([(base_rate, outcome) for _, outcome in pairs])
    bins, ece, mce = reliability(pairs)
    skill = None if brier_base == 0 else 1.0 - brier_score / brier_base
    return SignalStats(
        signal=signal,
        n=len(pairs),
        base_rate=round(base_rate, 6),
        brier=round(brier_score, 6),
        brier_skill_score=None if skill is None else round(skill, 6),
        log_loss=round(log_loss(pairs), 6),
        auc=None if (value := auc(pairs)) is None else round(value, 6),
        accuracy=round(sum(1 for p, o in pairs if (p >= 0.5) == bool(o)) / len(pairs), 6),
        mean_predicted=round(sum(p for p, _ in pairs) / len(pairs), 6),
        expected_calibration_error=ece,
        max_calibration_error=mce,
        bins=bins,
    )


def build_report(
    observations: list[Observation],
    *,
    coverage: float,
    horizons: int,
    note: str = "",
) -> CalibrationReport:
    """Aggregate every observation in a run into the reported calibration block."""
    scored = [obs for obs in observations if obs.scored]
    signals: dict[str, SignalStats] = {}
    for name, attr in (("noul", "noul"), ("choice", "choice"), ("rank", "rank")):
        pairs = [
            (getattr(obs, attr), int(obs.outcome))
            for obs in scored
            if getattr(obs, attr) is not None and obs.outcome is not None
        ]
        if pairs:
            signals[name] = signal_stats(name, pairs)

    # Paired form comparison: same ticker, same horizon, one call apart.
    gaps = [
        abs(obs.noul - obs.choice)
        for obs in scored
        if obs.noul is not None and obs.choice is not None
    ]
    miscalibration_gap = round(sum(gaps) / len(gaps), 6) if gaps else None

    return CalibrationReport(
        horizons=horizons,
        observations=len(scored),
        coverage=round(coverage, 4),
        signals=signals,
        miscalibration_gap=miscalibration_gap,
        note=note,
    )


# Score levels are ordered worst-to-best, so level index / (levels - 1) is a
# monotone 0-1 attractiveness that can be compared with a noul probability.
def score_to_probability(score: float, level_count: int) -> float:
    if level_count <= 1:
        return 0.5
    return min(max(score / (level_count - 1), 0.0), 1.0)


def choice_probability(probabilities: dict[str, float], positive: str = "outperform") -> float | None:
    """Pull the outperform probability out of a choice answer's distribution."""
    if not probabilities:
        return None
    total = sum(probabilities.values())
    if total <= 0:
        return None
    return min(max(probabilities.get(positive, 0.0) / total, 0.0), 1.0)