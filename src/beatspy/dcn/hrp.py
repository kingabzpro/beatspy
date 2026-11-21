"""Hierarchical risk parity on point-in-time adjusted closing prices."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..data.service import DataService
from ..schemas import Scenario, ValidatedDecision, validate_weights

MODEL = "hrp"
FORM = "allocation"
LOOKBACK = 126
MIN_HISTORY = 63


def hrp_weights(returns: pd.DataFrame) -> dict[str, float]:
    """Cluster correlation distances, then split capital by cluster variance."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    returns = returns.sort_index(axis=1)
    covariance = returns.cov().to_numpy()
    if not len(covariance) or not np.isfinite(covariance).all() or np.any(np.diag(covariance) <= 0):
        raise ValueError("HRP requires finite returns with positive variance for every asset")
    if len(covariance) == 1:
        return {returns.columns[0]: 1.0}

    vol = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(vol, vol)
    distance = np.sqrt(np.clip((1.0 - correlation) / 2.0, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="single"))
    weights = np.ones(len(order))

    def cluster_variance(indices: np.ndarray) -> float:
        sub = covariance[np.ix_(indices, indices)]
        inverse = 1.0 / np.diag(sub)
        relative = inverse / inverse.sum()
        return max(float(relative @ sub @ relative), 0.0)

    def bisect(indices: np.ndarray) -> None:
        if len(indices) < 2:
            return
        middle = len(indices) // 2
        left, right = indices[:middle], indices[middle:]
        left_var, right_var = cluster_variance(left), cluster_variance(right)
        total = left_var + right_var
        share = right_var / total if total > 0 else 0.5
        weights[left] *= share
        weights[right] *= 1.0 - share
        bisect(left)
        bisect(right)

    bisect(order)
    return dict(zip(returns.columns, weights.tolist(), strict=True))


def allocation(data: DataService, scenario: Scenario, day: date) -> ValidatedDecision:
    """Use at most 126 aligned close-to-close returns ending on the decision day."""
    benchmark = data.window(scenario.benchmark, day, LOOKBACK + 1)
    if benchmark is None or benchmark.index[-1].date() != day:
        raise ValueError(f"HRP is missing the benchmark close on {day}")
    prices = {}
    for ticker in sorted(set(scenario.tradable) - {scenario.benchmark}):
        frame = data.window(ticker, day, LOOKBACK + 1)
        if frame is not None and frame.index[-1].date() == day:
            prices[ticker] = frame["close"]
    closes = pd.DataFrame(prices, index=benchmark.index)
    returns = closes.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    returns = returns.loc[:, returns.notna().sum() >= MIN_HISTORY].dropna()
    if returns.empty or len(returns) < MIN_HISTORY:
        raise ValueError(f"HRP has fewer than {MIN_HISTORY} aligned daily returns on {day}")
    returns = returns.loc[:, returns.var() > 0]
    weights = {ticker: value for ticker, value in hrp_weights(returns).items() if value > 0}
    return validate_weights(weights, None, scenario.tradable, scenario.max_position_weight)
