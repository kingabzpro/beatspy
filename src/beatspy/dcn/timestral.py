"""Forecast relative closing prices and answer BeatSPY's outperformance questions."""

from __future__ import annotations

import time
from datetime import date

import numpy as np

from ..data.service import DataService
from ..schemas import ChoiceAnswer, NoulAnswer, Question
from .clients import DcnError, DcnResult

CONTEXT_SESSIONS = 256
MIN_OBSERVATIONS = 64


class TimestralClient:
    provider = "timestral"
    model = "timestral-209m"
    checkpoint = "mistralai/Timestral-209M"
    revision = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"
    endpoint = f"local://{model}"

    def __init__(self, data: DataService, *, forecaster=None):
        self.data = data
        self.forecaster = self.load_forecaster() if forecaster is None else forecaster

    @classmethod
    def load_forecaster(cls):
        try:
            import torch
            from timestral import TimestralForecaster
        except ImportError as exc:
            raise DcnError(
                "Timestral requires the Mistral inference package and PyTorch. See examples/kaggle_timestral.py."
            ) from exc
        if not torch.cuda.is_available():
            raise DcnError("Timestral-209M requires a CUDA GPU in BeatSPY. Enable a GPU accelerator on Kaggle.")
        return TimestralForecaster.from_pretrained(cls.checkpoint, device="cuda", revision=cls.revision)

    async def decide(self, state: dict, questions: dict[str, Question]) -> DcnResult:
        started = time.perf_counter()
        as_of = date.fromisoformat(state["as_of"])
        horizon = state["horizon_trading_days"]
        benchmark = self.data.benchmark
        market = self.data.window(benchmark, as_of, CONTEXT_SESSIONS)
        if horizon <= 0 or market is None or market.index[-1].date() != as_of:
            return DcnResult(missing_ids=list(questions), requests=0)

        market_close = market["close"].where(market["close"] > 0)
        tickers = {
            question.instructions["ticker"]
            for question in questions.values()
            if question.type in ("noul", "choice")
        }
        contexts = {}
        for ticker in sorted(tickers):
            frame = self.data.window(ticker, as_of, CONTEXT_SESSIONS)
            if frame is None:
                continue
            close = frame["close"].reindex(market.index)
            close = close.where(close > 0)
            relative = (100.0 * (np.log(close) - np.log(market_close))).to_numpy(dtype=np.float32)
            valid = np.isfinite(relative)
            if valid[-1] and valid.sum() >= MIN_OBSERVATIONS:
                contexts[ticker] = np.where(valid, relative, np.nan)

        if not contexts:
            return DcnResult(missing_ids=list(questions), requests=0)

        context = np.stack(list(contexts.values()))
        forecast = self.forecaster.forecast(context, prediction_length=horizon)
        quantiles = forecast.quantiles[:, 0, -1, :].tolist()
        levels = forecast.quantile_levels
        probabilities = {}
        for ticker, threshold, knots in zip(contexts, context[:, -1], quantiles, strict=True):
            if np.isfinite(knots).all():
                # A rising asset/SPY ratio is exactly a higher close-to-close return.
                # Interpolate the CDF, using the outer emitted quantiles as tail bounds.
                probabilities[ticker] = 1.0 - float(np.interp(threshold, knots, levels))

        answers: dict[str, NoulAnswer | ChoiceAnswer] = {}
        for question_id, question in questions.items():
            if question.type not in ("noul", "choice"):
                continue
            probability = probabilities.get(question.instructions["ticker"])
            if probability is None:
                continue
            if question.type == "noul":
                answers[question_id] = NoulAnswer(noul=probability)
            else:
                answers[question_id] = ChoiceAnswer(
                    choice="outperform" if probability > 0.5 else "underperform",
                    probabilities={"outperform": probability, "underperform": 1.0 - probability},
                )

        return DcnResult(
            answers=answers,
            missing_ids=[question_id for question_id in questions if question_id not in answers],
            latency_s=time.perf_counter() - started,
        )

    async def close(self) -> None:
        pass
