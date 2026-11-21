"""Walk-forward gradient boosting for BeatSPY's relative-return labels."""

from __future__ import annotations

import time
from bisect import bisect_right
from datetime import date

import pandas as pd

from ..data.service import DataService
from ..schemas import ChoiceAnswer, NoulAnswer, Question
from .clients import DcnError, DcnResult
from .state import SECTORS, _ticker_row, benchmark_row

FEATURES = (
    "last_close", "last_5d_pct", "return_30d_pct", "return_90d_pct", "return_90d_vs_benchmark_pct",
    "rsi_14", "annualized_vol_30d_pct", "off_52w_high_pct", "volume_ratio_10_50", "sector",
    "benchmark_return_30d_pct", "benchmark_return_90d_pct",
)
SECTOR_DTYPE = pd.CategoricalDtype(sorted(set(SECTORS.values())))
WARMUP_SESSIONS = 20
MIN_TRAINING_ROWS = 50


def _feature_frame(rows: list[dict], index: pd.Index) -> pd.DataFrame:
    frame = pd.DataFrame(rows, index=index, columns=FEATURES)
    numeric = [feature for feature in FEATURES if feature != "sector"]
    frame[numeric] = frame[numeric].astype(float)
    frame["sector"] = frame["sector"].astype(SECTOR_DTYPE)
    return frame


class LightGBMClient:
    provider = "local"
    model = "lightgbm"
    endpoint = "local://lightgbm"

    def __init__(self, data: DataService):
        self.data = data
        self._features: dict[tuple[date, str], dict | None] = {}
        self._benchmarks: dict[date, dict] = {}

    def _feature_row(self, ticker: str, day: date) -> dict | None:
        key = (day, ticker)
        if key not in self._features:
            if day not in self._benchmarks:
                self._benchmarks[day] = benchmark_row(self.data, self.data.benchmark, day)
            benchmark = self._benchmarks[day]
            row = _ticker_row(self.data, ticker, day, benchmark["return_90d_pct"])
            self._features[key] = None if row is None else {
                **row,
                "benchmark_return_30d_pct": benchmark["return_30d_pct"],
                "benchmark_return_90d_pct": benchmark["return_90d_pct"],
            }
        return self._features[key]

    def training_data(self, as_of: date, horizon: int, tickers: list[str]) -> tuple[pd.DataFrame, list[int]]:
        """Use Friday origins whose horizon-end close is observable by `as_of`."""
        if horizon < 1:
            raise ValueError("Forecast horizon must be positive")
        dates = self.data.calendar
        cutoff = bisect_right(dates, as_of)
        inputs, labels, index = [], [], []
        for position in range(WARMUP_SESSIONS - 1, cutoff - horizon):
            origin = dates[position]
            if origin.weekday() != 4:
                continue
            settled = dates[position + horizon]
            before = self.data.close_on(self.data.benchmark, origin)
            after = self.data.close_on(self.data.benchmark, settled)
            if not before or after is None:
                continue
            benchmark_return = after / before
            for ticker in tickers:
                start = self.data.close_on(ticker, origin)
                end = self.data.close_on(ticker, settled)
                if not start or end is None:
                    continue
                row = self._feature_row(ticker, origin)
                if row is None:
                    continue
                index.append((origin, ticker))
                inputs.append(row)
                labels.append(int(end / start > benchmark_return))
        return _feature_frame(inputs, pd.MultiIndex.from_tuples(index, names=["origin", "ticker"])), labels

    def _predict_probabilities(self, training: pd.DataFrame, labels: list[int], inputs: pd.DataFrame):
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise DcnError("LightGBM requires `pip install lightgbm==4.6.0`.") from exc

        model = LGBMClassifier(
            n_estimators=80, num_leaves=7, learning_rate=0.05, min_child_samples=40,
            reg_lambda=1.0, random_state=42, deterministic=True, force_col_wise=True, n_jobs=2, verbosity=-1,
        )
        model.fit(training, labels)
        return model.predict_proba(inputs)[:, 1]

    async def decide(self, state: dict, questions: dict[str, Question]) -> DcnResult:
        started = time.perf_counter()
        day = date.fromisoformat(state["as_of"])
        horizon = state["horizon_trading_days"]
        tickers = sorted({question.instructions["ticker"] for question in questions.values()})
        training, labels = self.training_data(day, horizon, tickers)
        if len(training) < MIN_TRAINING_ROWS or len(set(labels)) < 2:
            raise DcnError(f"Insufficient settled {self.model} training labels on {day} ({len(training)} rows).")
        current = {}
        for ticker in tickers:
            row = self._feature_row(ticker, day)
            if row is not None and self.data.close_on(ticker, day) is not None:
                current[ticker] = row
        if not current:
            return DcnResult(missing_ids=list(questions), requests=0)

        inputs = _feature_frame(list(current.values()), pd.Index(current))
        probabilities = dict(zip(current, self._predict_probabilities(training, labels, inputs), strict=True))
        answers: dict[str, NoulAnswer | ChoiceAnswer] = {}
        for question_id, question in questions.items():
            probability = probabilities.get(question.instructions["ticker"])
            if probability is None:
                continue
            if question.type == "noul":
                answers[question_id] = NoulAnswer(noul=probability)
            elif question.type == "choice":
                answers[question_id] = ChoiceAnswer(
                    choice="outperform" if probability > 0.5 else "underperform",
                    probabilities={"outperform": probability, "underperform": 1.0 - probability},
                )
        return DcnResult(
            answers=answers, missing_ids=[key for key in questions if key not in answers],
            latency_s=time.perf_counter() - started,
        )

    async def close(self) -> None:
        pass
