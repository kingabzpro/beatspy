"""Uncertainty for a benchmark whose samples are clustered by decision date.

A 25-week run with 25 assets per week looks like 625 observations, but the assets
inside one week move together and share the same market regime, so the effective
sample is closer to the number of decision dates. Two consequences, both handled
here:

- **Portfolio returns**: the interval comes from a block bootstrap that resamples
  whole decision windows with replacement (a stationary block bootstrap over the
  decision schedule). Resampling individual days would understate the interval.
- **Calibration statistics**: the p-value comes from a permutation test that
  shuffles outcomes *within* each decision date, holding the date-level outcome
  counts fixed. The null is "the model cannot tell the assets apart within a
  week, but the week itself may be up or down", which is the honest null for a
  benchmark-relative label.

Everything is seeded from a module constant, so replay reproduces every interval
and p-value exactly. Draw counts are fixed for the same reason: a confidence
interval that changes between runs is not auditable.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

BOOTSTRAP_DRAWS = 1000
PERMUTATION_DRAWS = 1000
SEED = 20261011
LEVEL = 0.95


def _percentile(sorted_values: list[float], quantile: float) -> float:
    """Linear-interpolation percentile; `sorted_values` must be sorted."""
    if not sorted_values:
        raise ValueError("empty sample")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def block_bootstrap_returns(
    blocks: Sequence[Sequence[float]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    level: float = LEVEL,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile interval for a compounded return over resampled decision windows.

    `blocks` is one list of daily returns per decision window. Each draw resamples
    as many windows as the run has, with replacement, and compounds the returns
    inside them.
    """
    blocks = [list(block) for block in blocks if block]
    if not blocks:
        raise ValueError("no return blocks to resample")
    rng = random.Random(seed)
    count = len(blocks)
    totals: list[float] = []
    for _ in range(draws):
        product = 1.0
        for _ in range(count):
            for value in blocks[rng.randrange(count)]:
                product *= 1.0 + value
        totals.append(product - 1.0)
    totals.sort()
    tail = (1.0 - level) / 2.0
    return _percentile(totals, tail), _percentile(totals, 1.0 - tail)


def block_bootstrap_excess(
    portfolio_blocks: Sequence[Sequence[float]],
    spy_blocks: Sequence[Sequence[float]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    level: float = LEVEL,
    seed: int = SEED,
) -> tuple[float, float]:
    """The same bootstrap, resampling portfolio and benchmark windows together.

    Both legs use the same resampled window per draw, so the interval is for the
    paired excess return, not for two independent returns.
    """
    portfolio_blocks = [list(block) for block in portfolio_blocks if block]
    spy_blocks = [list(block) for block in spy_blocks if block]
    if not portfolio_blocks or not spy_blocks:
        raise ValueError("no return blocks to resample")
    rng = random.Random(seed)
    count = min(len(portfolio_blocks), len(spy_blocks))
    totals: list[float] = []
    for _ in range(draws):
        portfolio_product = 1.0
        spy_product = 1.0
        for _ in range(count):
            index = rng.randrange(count)
            for value in portfolio_blocks[index]:
                portfolio_product *= 1.0 + value
            for value in spy_blocks[index]:
                spy_product *= 1.0 + value
        totals.append((portfolio_product - 1.0) - (spy_product - 1.0))
    totals.sort()
    tail = (1.0 - level) / 2.0
    return _percentile(totals, tail), _percentile(totals, 1.0 - tail)


def permutation_p_value(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    clusters: Sequence[str],
    statistic: Callable[[Sequence[float], Sequence[int]], float],
    *,
    draws: int = PERMUTATION_DRAWS,
    seed: int = SEED,
) -> float | None:
    """Two-sided permutation p-value with outcomes shuffled within clusters.

    The observed statistic is compared against its null distribution under
    within-date shuffles. Returns None when a test is impossible (fewer than two
    clusters, or a single outcome class), because a p-value there would be
    meaningless rather than merely wide.
    """
    if len(probabilities) != len(outcomes) or len(outcomes) != len(clusters):
        raise ValueError("probabilities, outcomes, and clusters must be aligned")
    if len(set(clusters)) < 2 or len(set(outcomes)) < 2:
        return None
    observed = statistic(list(probabilities), list(outcomes))
    positions: dict[str, list[int]] = {}
    for index, cluster in enumerate(clusters):
        positions.setdefault(str(cluster), []).append(index)
    rng = random.Random(seed)
    shuffled = list(outcomes)
    hits = 0
    for _ in range(draws):
        for indices in positions.values():
            values = [outcomes[index] for index in indices]
            rng.shuffle(values)
            for index, value in zip(indices, values, strict=True):
                shuffled[index] = value
        null = statistic(list(probabilities), shuffled)
        if abs(null) >= abs(observed):
            hits += 1
    return (hits + 1) / (draws + 1)


def brier_skill_statistic(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Brier skill score against a constant at the sample rate."""
    count = len(outcomes)
    base_rate = sum(outcomes) / count
    brier = sum((p - o) ** 2 for p, o in zip(probabilities, outcomes, strict=True)) / count
    base = sum((base_rate - o) ** 2 for o in outcomes) / count
    return 0.0 if base == 0 else 1.0 - brier / base


def auc_statistic(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Tie-aware AUC via average ranks, fast enough for a permutation loop.

    Agrees with `dcn.calibration.auc` (Mann-Whitney with ties counted half) but
    runs in O(n log n) instead of O(n_pos * n_neg).
    """
    count = len(probabilities)
    positives = sum(outcomes)
    negatives = count - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = sorted(range(count), key=probabilities.__getitem__)
    rank_sum = 0.0
    index = 0
    while index < count:
        end = index
        while end + 1 < count and probabilities[order[end + 1]] == probabilities[order[index]]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            if outcomes[order[position]]:
                rank_sum += average_rank
        index = end + 1
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


__all__ = [
    "BOOTSTRAP_DRAWS",
    "LEVEL",
    "PERMUTATION_DRAWS",
    "SEED",
    "auc_statistic",
    "block_bootstrap_excess",
    "block_bootstrap_returns",
    "brier_skill_statistic",
    "permutation_p_value",
]
