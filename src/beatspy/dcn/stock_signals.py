"""Walk-forward models and rolling beta-adjusted signals for stock selection."""

from __future__ import annotations

from bisect import bisect_left
from datetime import date

import numpy as np
import pandas as pd

from ..data.service import DataService
from .clients import DcnError
from .lightgbm import MIN_TRAINING_ROWS, LightGBMClient

SEED = 42
BETA_SESSIONS = 126
SIGNAL_COLUMNS = ["residual_momentum", "residual_reversal"]


def residual_signals(stock: np.ndarray, market: np.ndarray) -> tuple[float, float]:
    centered = market - market.mean()
    beta = np.dot(stock, centered) / max(np.dot(centered, centered), 1e-12)
    residual = stock - beta * market
    scale = max(residual.std(), 1e-6)
    return float(residual[-65:-5].mean() / scale), float(-residual[-5:].mean() / scale)


def rank_targets(returns: np.ndarray, origins: pd.Index) -> tuple[np.ndarray, list[int]]:
    groups = pd.Series(returns).groupby(origins, sort=False)
    buckets = np.ceil(groups.rank(pct=True).to_numpy() * 5).astype(int) - 1
    return buckets, groups.size().tolist()


def fit_scores(name: str, training: pd.DataFrame, returns: np.ndarray):
    if name in ("residual-momentum", "residual-reversal"):
        column = name.replace("-", "_")
        return lambda rows: rows[column].to_numpy()
    if name == "lambdamart":
        from lightgbm import LGBMRanker

        target, groups = rank_targets(returns, training.index.get_level_values("origin"))
        ranker = LGBMRanker(
            n_estimators=80, num_leaves=7, learning_rate=0.05, min_child_samples=30, reg_lambda=1.0,
            lambdarank_truncation_level=10, deterministic=True, force_col_wise=True,
            n_jobs=2, random_state=SEED, verbosity=-1,
        )
        ranker.fit(training, target, group=groups)
        return ranker.predict

    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = [column for column in training.columns if column != "sector"]
    preprocess = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("sector", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sector"]),
    ])
    if name == "elastic-net":
        regressor = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=2000)
    else:
        regressor = MLPRegressor(hidden_layer_sizes=(8,), solver="lbfgs", alpha=4.0, max_iter=200, random_state=SEED)
    model = TransformedTargetRegressor(regressor=make_pipeline(preprocess, regressor), transformer=StandardScaler())
    model.fit(training, returns)
    return model.predict


def score_probabilities(calibration: np.ndarray, outcomes: np.ndarray, current: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize
    from scipy.special import expit, logit

    center, scale = calibration.mean(), max(calibration.std(), 1e-8)
    z = np.clip((calibration - center) / scale, -8.0, 8.0)
    current_z = (current - center) / scale
    prior = (outcomes.sum() + 1) / (len(outcomes) + 2)

    def loss(params):
        slope, intercept = params
        logits = slope * z + intercept
        return np.mean(np.logaddexp(0, logits) - outcomes * logits) + 0.01 * slope**2

    result = minimize(loss, [0.1, logit(prior)], bounds=[(0.05, 5.0), (-12.0, 12.0)], method="L-BFGS-B")
    if not result.success:
        raise DcnError(f"Stock-signal calibration failed: {result.message}")
    slope, intercept = result.x
    return expit(np.clip(slope * current_z + intercept, -30.0, 30.0))


class StockSignalClient(LightGBMClient):
    def __init__(self, data: DataService, model: str):
        super().__init__(data)
        self.model = model
        self.endpoint = f"local://{model}"
        self._residuals: dict[tuple[date, str], tuple[float, float]] = {}
        self._market_closes: dict[date, pd.Series] = {}

    def _signals(self, day: date, ticker: str) -> tuple[float, float]:
        key = (day, ticker)
        if key not in self._residuals:
            if day not in self._market_closes:
                market = self.data.window(self.data.benchmark, day, BETA_SESSIONS + 1)
                self._market_closes[day] = market["close"] if market is not None else pd.Series(dtype=float)
            market_close = self._market_closes[day]
            stock = self.data.window(ticker, day, BETA_SESSIONS + 1)
            signals = (float("nan"), float("nan"))
            if stock is not None and len(market_close) >= 66:
                aligned = stock["close"].reindex(market_close.index)
                if (aligned > 0).all() and (market_close > 0).all():
                    signals = residual_signals(np.diff(np.log(aligned)), np.diff(np.log(market_close)))
            self._residuals[key] = signals
        return self._residuals[key]

    def decision_probabilities(self, day: date, horizon: int, tickers: list[str], inputs: pd.DataFrame):
        training, excess = self.training_returns(day, horizon, tickers)
        training[SIGNAL_COLUMNS] = [self._signals(origin, ticker) for origin, ticker in training.index]
        current = inputs.copy()
        current[SIGNAL_COLUMNS] = [self._signals(day, ticker) for ticker in current.index]
        valid = training[SIGNAL_COLUMNS].notna().all(axis=1)
        training = training.loc[valid]
        excess = np.asarray(excess)[valid]
        origins = training.index.get_level_values("origin")
        unique_origins = origins.unique()
        if len(unique_origins) < 8 or current[SIGNAL_COLUMNS].isna().any().any():
            raise DcnError(f"Insufficient price history for {self.model} on {day}")

        calibration_start = unique_origins[len(unique_origins) * 3 // 4]
        calendar = self.data.calendar
        fit_end = bisect_left(calendar, calibration_start) - horizon
        fit = origins <= calendar[fit_end]
        calibrate = origins >= calibration_start
        if fit.sum() < MIN_TRAINING_ROWS:
            raise DcnError(f"Insufficient settled {self.model} training rows before {calibration_start}")
        try:
            predict = fit_scores(self.model, training.loc[fit], excess[fit])
            return score_probabilities(predict(training.loc[calibrate]), excess[calibrate] > 0, predict(current))
        except ImportError as exc:
            raise DcnError("Stock-signal baselines require scikit-learn, SciPy, and LightGBM for LambdaMART.") from exc
