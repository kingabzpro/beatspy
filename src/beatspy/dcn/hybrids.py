"""Offline hybrids built on the existing frozen-price benchmark models."""

from __future__ import annotations

import os
from bisect import bisect_left
from datetime import date

import numpy as np
import pandas as pd

from ..data.service import DataService
from ..schemas import NoulAnswer, Scenario, ValidatedDecision, validate_weights
from .clients import DcnError, DcnResult
from .foundation_forecast import Chronos2Client
from .hrp import LOOKBACK, MIN_HISTORY, hrp_weights
from .lightgbm import MIN_TRAINING_ROWS, LightGBMClient
from .stock_jepa import CONTEXT, HORIZON, market_panel, train_predictor
from .stock_signals import StockSignalClient
from .timestral import CONTEXT_SESSIONS, MIN_OBSERVATIONS

RANK_COUNT = 5


def ranked_allocation(probabilities: dict[str, float], data: DataService, day: date, *, risk_parity: bool):
    picks = sorted(probabilities, key=lambda ticker: (-probabilities[ticker], ticker))[:RANK_COUNT]
    if not picks:
        return {}
    if not risk_parity:
        return dict.fromkeys(picks, 1.0 / len(picks))
    frames = {ticker: data.window(ticker, day, LOOKBACK + 1) for ticker in picks}
    if any(frame is None or frame.index[-1].date() != day for frame in frames.values()):
        raise DcnError(f"LambdaMART + HRP is missing a selected stock's close on {day}")
    closes = pd.concat({ticker: frame["close"] for ticker, frame in frames.items()}, axis=1, join="inner")
    returns = closes.pct_change(fill_method=None).dropna()
    if len(returns) < MIN_HISTORY:
        raise DcnError(f"LambdaMART + HRP needs {MIN_HISTORY} aligned returns on {day}")
    return hrp_weights(returns)


class RankPortfolioClient(StockSignalClient):
    def __init__(self, data: DataService, *, model: str):
        super().__init__(data, model, predictor="lambdamart")
        method = "HRP" if model.endswith("-hrp") else "equal weight"
        self.portfolio_method = f"LambdaMART top {RANK_COUNT} + {method}"

    def target_portfolio(self, day: date, scenario: Scenario, probabilities, target) -> ValidatedDecision:
        weights = ranked_allocation(probabilities, self.data, day, risk_parity=self.model.endswith("-hrp"))
        return validate_weights(weights, None, scenario.tradable, scenario.max_position_weight)


class MomentumFilterClient(LightGBMClient):
    model = "lightgbm-momentum"
    endpoint = f"local://{model}"
    portfolio_method = "LightGBM weights with positive residual momentum"

    def __init__(self, data: DataService):
        super().__init__(data)
        self.signals = StockSignalClient(data, "residual-momentum")

    def target_portfolio(self, day: date, scenario: Scenario, probabilities, target) -> ValidatedDecision:
        eligible = {ticker: weight for ticker, weight in target.weights.items()
                    if self.signals._signals(day, ticker)[0] > 0}
        total = sum(eligible.values())
        weights = {ticker: weight / total for ticker, weight in eligible.items()}
        return validate_weights(weights, None, scenario.tradable, scenario.max_position_weight)


def high_volatility(data: DataService, day: date) -> bool:
    frame = data.window(data.benchmark, day, 147)
    if frame is None or len(frame) < 63:
        raise DcnError(f"Momentum + Reversal has insufficient market history for {day}")
    rolling = frame["close"].pct_change(fill_method=None).rolling(21).std().dropna()
    return bool(rolling.iloc[-1] > rolling.iloc[:-1].median())


class MomentumReversalClient:
    provider = "local"

    def __init__(self, data: DataService, model: str):
        self.data = data
        self.model = model
        self.endpoint = f"local://{model}"
        self.momentum = StockSignalClient(data, "residual-momentum")
        self.reversal = StockSignalClient(data, "residual-reversal")

    async def decide(self, state, questions) -> DcnResult:
        if self.model == "momentum-reversal":
            day = date.fromisoformat(state["as_of"])
            client = self.reversal if high_volatility(self.data, day) else self.momentum
            return await client.decide(state, questions)
        momentum = await self.momentum.decide(state, questions)
        reversal = await self.reversal.decide(state, questions)
        answers = {key: NoulAnswer(noul=(momentum.answers[key].noul + reversal.answers[key].noul) / 2)
                   for key in questions if key in momentum.answers and key in reversal.answers}
        return DcnResult(answers=answers, missing_ids=[key for key in questions if key not in answers],
                         latency_s=momentum.latency_s + reversal.latency_s)

    async def close(self) -> None:
        pass


class ChronosLightGBMClient(LightGBMClient):
    model = "chronos-lightgbm"
    endpoint = f"local://{model}"

    def __init__(self, data: DataService, forecaster):
        super().__init__(data)
        self.chronos = Chronos2Client(data, forecaster=forecaster)
        self._forecasts = {}

    def forecast(self, day: date, horizon: int, tickers: tuple[str, ...]) -> dict[str, tuple[float, float]]:
        key = (day, horizon, tickers)
        if key in self._forecasts:
            return self._forecasts[key]
        spy = self.data.window(self.data.benchmark, day, CONTEXT_SESSIONS)["close"]
        contexts = {}
        for ticker in tickers:
            frame = self.data.window(ticker, day, CONTEXT_SESSIONS)
            if frame is not None:
                close = frame["close"].reindex(spy.index)
                relative = (100 * np.log(close.where(close > 0) / spy)).to_numpy(dtype=np.float32)
                if np.isfinite(relative[-1]) and np.isfinite(relative).sum() >= MIN_OBSERVATIONS:
                    contexts[ticker] = relative
        features = {}
        if contexts:
            values = np.stack(list(contexts.values()))
            quantiles, _ = self.chronos.terminal_quantiles(values, horizon)
            for ticker, last, knots in zip(contexts, values[:, -1], quantiles, strict=True):
                if np.isfinite(knots).all():
                    features[ticker] = (float(knots[4] - last), float(knots[-1] - knots[0]))
        self._forecasts[key] = features
        return features

    def decision_probabilities(self, day, horizon, tickers, inputs):
        training, labels = self.training_data(day, horizon, tickers)
        if len(training) < MIN_TRAINING_ROWS or len(set(labels)) < 2:
            raise DcnError(f"Insufficient settled {self.model} training labels on {day} ({len(training)} rows).")
        inputs = inputs.copy()
        columns = ["chronos_median_excess", "chronos_interval_width"]
        missing = (np.nan, np.nan)
        universe = tuple(tickers)
        training[columns] = [self.forecast(origin, horizon, universe).get(ticker, missing)
                             for origin, ticker in training.index]
        current = self.forecast(day, horizon, universe)
        if any(ticker not in current for ticker in inputs.index):
            raise DcnError(f"Chronos-2 forecast coverage is incomplete for {day}")
        inputs[columns] = [current[ticker] for ticker in inputs.index]
        return self._predict_probabilities(training, labels, inputs)


class JepaElasticNetClient(StockSignalClient):
    def __init__(self, data: DataService, device: str):
        super().__init__(data, "jepa-elastic-net", predictor="elastic-net")
        self.device = device
        self.predict = None
        self._jepa = {}

    def extra_features(self, training, current, day, horizon, tickers):
        if not 1 <= horizon <= HORIZON:
            raise DcnError("STOCK-JEPA + Elastic Net requires a horizon of 1 to 21 trading sessions")
        if self.predict is None:
            prior_count = bisect_left(self.data.calendar, day)
            # Train the encoder on the early pre-window segment and freeze it
            # before generating any scores for the later Elastic Net labels.
            self.fit_through = self.data.calendar[2 * prior_count // 3 - 1]
            panel = market_panel(self.data, tickers, self.fit_through, limit=10_000)
            self.predict = train_predictor(panel, self.device)

        def scores(origin):
            if origin <= self.fit_through:
                return {}
            key = (origin, tuple(tickers))
            if key not in self._jepa:
                panel = market_panel(self.data, tickers, origin, limit=CONTEXT + 1)
                valid = np.isfinite(panel).all(axis=(1, 2))
                self._jepa[key] = dict(zip(np.asarray(tickers)[valid], self.predict(panel[valid]), strict=True))
            return self._jepa[key]

        missing = np.full(HORIZON, np.nan)
        training["stock_jepa"] = [scores(origin).get(ticker, missing)[horizon - 1] for origin, ticker in training.index]
        current["stock_jepa"] = [scores(day).get(ticker, missing)[horizon - 1] for ticker in current.index]
        return ["stock_jepa"]


def load_chronos_checkpoint(path, device: str):
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(str(path), local_files_only=True, device_map=device)


def make_hybrid_client(data: DataService, model: str, *, forecaster=None, device: str = "cpu"):
    if model in ("lambdamart-hrp", "lambdamart-equal"):
        client = RankPortfolioClient(data, model=model)
    elif model == "chronos-lightgbm":
        if forecaster is None:
            path = os.environ.get("BEATSPY_CHRONOS_CHECKPOINT")
            if not path:
                raise DcnError("Set BEATSPY_CHRONOS_CHECKPOINT to a local Chronos-2 checkpoint.")
            forecaster = load_chronos_checkpoint(path, device)
        client = ChronosLightGBMClient(data, forecaster)
    elif model in ("momentum-reversal", "momentum-reversal-blend"):
        client = MomentumReversalClient(data, model)
    elif model == "jepa-elastic-net":
        client = JepaElasticNetClient(data, device)
    elif model == "lightgbm-momentum":
        client = MomentumFilterClient(data)
    else:
        raise ValueError(f"Unknown hybrid model {model}")
    return client
