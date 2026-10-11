"""How much can one window tell you? A simulation of the benchmark's power.

Run this to reproduce the table in the README's uncertainty section:

    uv run python examples/power_analysis.py

Two questions, both answered with the same machinery the benchmark scores runs
with (`beatspy.dcn.statistics`):

1. **Returns.** For a book with a given daily excess drift and volatility, how
   large must the true annual excess be before the 95% block-bootstrap interval
   excludes zero in 80% of simulations? The unit of resampling is one decision
   window, so the answer scales with the number of decisions, not the number of
   asset-days.
2. **Calibration.** For a model whose probabilities spread with a given standard
   deviation and whose true win rate moves with its probability, how large must
   that true relationship be before the within-date permutation test on Brier
   skill rejects "no relationship" at 5% in 80% of simulations?

The numbers are simulations under stated assumptions, not analytic truths. The
point is the order of magnitude: it tells you whether a run has any chance of
distinguishing a real edge from noise before you spend a cent on it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from beatspy.dcn.statistics import (
    auc_statistic,
    block_bootstrap_returns,
    brier_skill_statistic,
    permutation_p_value,
)

# Simulation size. Kept modest so the script finishes in a couple of minutes;
# raising reps narrows the Monte Carlo error on each cell, it does not change
# the conclusions.
RETURN_REPS = 200
CALIBRATION_REPS = 100
PERMUTATION_DRAWS = 150
BOOTSTRAP_DRAWS = 300
SEED = 7

# A 25-name long-only book: 1% daily volatility is a realistic order of magnitude.
DAILY_VOLATILITY = 0.010
TRADING_DAYS = 252
DAYS_PER_WINDOW = 5
ASSETS_PER_WINDOW = 25


@dataclass
class ReturnPower:
    windows: int
    daily_volatility: float
    annual_excess: float
    power: float


def _simulate_return_power(
    windows: int,
    annual_excess: float,
    *,
    reps: int = RETURN_REPS,
    seed: int = SEED,
) -> float:
    """Share of simulations whose 95% interval excludes zero excess."""
    rng = random.Random(seed)
    daily_drift = annual_excess / TRADING_DAYS
    hits = 0
    for _ in range(reps):
        blocks = [
            [rng.gauss(daily_drift, DAILY_VOLATILITY) for _ in range(DAYS_PER_WINDOW)]
            for _ in range(windows)
        ]
        lower, upper = block_bootstrap_returns(blocks, draws=BOOTSTRAP_DRAWS, seed=seed + 1)
        if lower > 0.0 or upper < 0.0:
            hits += 1
    return hits / reps


def _simulate_calibration(
    windows: int,
    true_signal: float,
    *,
    probability_spread: float = 0.047,
    reps: int = CALIBRATION_REPS,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """(power of the BSS test, power of the AUC test, mean realized AUC) for one cell.

    The model's probability is 0.5 + spread * z, and the true win rate is
    0.5 + true_signal * (probability - 0.5), capped to [0, 1]. `true_signal = 0`
    is a perfectly calibrated but uninformative model: neither test may reject
    it. `true_signal = 1` is a perfectly calibrated and informative one.

    Both p-values are simulated because they behave very differently: at narrow
    spreads the Brier score is dominated by its calibration term and barely
    moves, so its test needs a much stronger ranking than the AUC test to fire.
    """
    rng = random.Random(seed)
    bss_rejections = 0
    auc_rejections = 0
    aucs: list[float] = []
    for _ in range(reps):
        probabilities = []
        outcomes = []
        clusters = []
        for window in range(windows):
            for _ in range(ASSETS_PER_WINDOW):
                z = rng.gauss(0.0, 1.0)
                probability = min(max(0.5 + probability_spread * z, 0.01), 0.99)
                true_rate = min(max(0.5 + true_signal * (probability - 0.5), 0.0), 1.0)
                probabilities.append(round(probability, 4))
                outcomes.append(int(rng.random() < true_rate))
                clusters.append(f"w{window}")
        aucs.append(auc_statistic(probabilities, outcomes))
        bss_p = permutation_p_value(
            probabilities, outcomes, clusters, brier_skill_statistic, draws=PERMUTATION_DRAWS, seed=seed + 1
        )
        auc_p = permutation_p_value(
            probabilities, outcomes, clusters, auc_statistic, draws=PERMUTATION_DRAWS, seed=seed + 1
        )
        bss_rejections += int(bss_p is not None and bss_p < 0.05)
        auc_rejections += int(auc_p is not None and auc_p < 0.05)
    return bss_rejections / reps, auc_rejections / reps, sum(aucs) / len(aucs)


def main() -> None:
    print("Returns: power to exclude a zero excess from the 95% interval")
    print(f"  assumptions: {DAYS_PER_WINDOW} trading days per window, "
          f"{DAILY_VOLATILITY:.1%} daily volatility, {RETURN_REPS} simulations per cell")
    print(f"  {'windows':>8}  {'annual excess':>14}  {'power':>6}")
    for windows in (25, 50, 100, 250):
        for annual_excess in (0.02, 0.05, 0.10, 0.20):
            power = _simulate_return_power(windows, annual_excess)
            flag = "  <- 80%" if power >= 0.8 else ""
            print(f"  {windows:>8}  {annual_excess:>13.0%}  {power:>6.0%}{flag}")

    print()
    print("Calibration: power to reject 'no relationship' at 5%")
    print(f"  assumptions: probability spread sd {0.047}, {ASSETS_PER_WINDOW} assets per window, "
          f"{PERMUTATION_DRAWS} permutation draws, {CALIBRATION_REPS} simulations per cell")
    print(f"  {'windows':>8}  {'true signal':>12}  {'mean AUC':>9}  {'power BSS':>10}  {'power AUC':>10}")
    for windows in (25, 100):
        for true_signal in (0.0, 1.0, 2.0, 4.0):
            bss_power, auc_power, mean_auc = _simulate_calibration(windows, true_signal)
            flag = "  <- null: neither test may reject" if true_signal == 0.0 else ""
            print(
                f"  {windows:>8}  {true_signal:>12.1f}  {mean_auc:>9.3f}"
                f"  {bss_power:>10.0%}  {auc_power:>10.0%}{flag}"
            )


if __name__ == "__main__":
    main()
