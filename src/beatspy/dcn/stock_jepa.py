"""A compact, independently trained STOCK-JEPA adaptation for BeatSPY."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..data.service import DataService
from ..schemas import NoulAnswer, Question
from .clients import DcnError, DcnResult

MODEL = "stock-jepa"
CONTEXT = 32
HORIZON = 21
SEED = 42


def market_panel(data: DataService, tickers: list[str], through: date, *, limit: int) -> np.ndarray:
    """Return stock log returns, SPY log returns, and log volume changes through a cutoff."""
    benchmark = data.window(data.benchmark, through, limit)
    if benchmark is None or len(benchmark) < 2:
        raise DcnError(f"Stock-JEPA has no market history through {through}")
    spy = np.log(benchmark["close"].to_numpy(dtype=float))
    matrix = []
    for ticker in tickers:
        frame = data.window(ticker, through, limit)
        aligned = (frame if frame is not None else pd.DataFrame(columns=["close", "volume"]))
        aligned = aligned.reindex(benchmark.index)
        close = np.log(aligned["close"].to_numpy(dtype=float))
        volume = np.log1p(aligned["volume"].to_numpy(dtype=float))
        matrix.append(np.diff(np.column_stack((close, spy, volume)), axis=0))
    return np.asarray(matrix, dtype=np.float32)


def _samples(panel: np.ndarray, *, first: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Each example's context ends before its first target return."""
    contexts, futures = [], []
    for boundary in range(first, stop):
        context = panel[:, boundary - CONTEXT:boundary]
        future = panel[:, boundary:boundary + HORIZON]
        valid = np.isfinite(context).all(axis=(1, 2)) & np.isfinite(future).all(axis=(1, 2))
        contexts.append(context[valid])
        futures.append(future[valid])
    if not contexts or not any(len(context) for context in contexts):
        raise DcnError("Stock-JEPA has insufficient aligned history for both training phases")
    return np.concatenate(contexts), np.concatenate(futures)


def train_predictor(panel: np.ndarray, device: str) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a historical ridge prior, learn latent revisions, and freeze a probability readout."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise DcnError("Stock-JEPA requires PyTorch") from exc

    torch.manual_seed(SEED)

    def inputs(values):
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    split = panel.shape[1] // 2
    prior_x, prior_y = map(inputs, _samples(panel, first=CONTEXT, stop=split - HORIZON + 1))
    context, future = map(inputs, _samples(panel, first=split, stop=panel.shape[1] - HORIZON + 1))
    center = prior_x.mean(dim=(0, 1))
    scale = prior_x.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)

    def normalize(values):
        return ((values - center) / scale).clamp(-8.0, 8.0)

    def descriptors(values):
        return torch.cat((values[:, -5:].mean(1), values[:, -21:].mean(1), values.std(1, unbiased=False)), dim=1)

    def targets(values):
        statistics = []
        for horizon in (5, 21):
            stock = values[:, :horizon, 0]
            spy = values[:, :horizon, 1]
            path = torch.nn.functional.pad(stock.cumsum(1), (1, 0))
            drawdown = (path - path.cummax(1).values).amin(1)
            statistics.extend([stock.sum(1), stock.std(1, unbiased=False), drawdown, (stock - spy).sum(1)])
        return torch.stack(statistics, dim=1)

    prior_target = targets(prior_y)
    prior_target = (prior_target - prior_target.mean(0)) / prior_target.std(0, unbiased=False).clamp_min(1e-6)
    design = torch.nn.functional.pad(descriptors(normalize(prior_x)), (0, 1), value=1.0)
    penalty = 10.0 * torch.eye(design.shape[1], device=device)
    penalty[-1, -1] = 0.0
    coefficients = torch.linalg.solve(design.T @ design + penalty, design.T @ prior_target)

    def prior(values):
        return torch.nn.functional.pad(descriptors(values), (0, 1), value=1.0) @ coefficients

    width = 24
    encoder = nn.GRU(3, width, batch_first=True).to(device)
    target_encoder = copy.deepcopy(encoder).requires_grad_(False)
    anchor = nn.Sequential(nn.Linear(8, width), nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width)).to(device)
    target_anchor = copy.deepcopy(anchor).requires_grad_(False)
    revision = nn.Sequential(nn.Linear(width * 2, width * 2), nn.GELU(), nn.Linear(width * 2, width)).to(device)
    optimizer = torch.optim.AdamW([*encoder.parameters(), *anchor.parameters(), *revision.parameters()], lr=0.002)
    market_context = normalize(context)
    market_future = normalize(future)
    priors = prior(market_context)

    def encode(model, values):
        return model(values)[1][-1]

    def batches():
        return torch.randperm(len(context), device=device).split(256)

    for _ in range(6):
        for batch in batches():
            features = market_context[batch]
            with torch.no_grad():
                target = encode(target_encoder, market_future[batch])
                fixed_anchor = target_anchor(priors[batch])
                residual_target = target - fixed_anchor
                keep = torch.rand((*features.shape[:2], 1), device=device) >= 0.15
            predicted = revision(torch.cat((encode(encoder, features * keep), fixed_anchor), dim=1))
            loss = nn.functional.mse_loss(anchor(priors[batch]), target) + nn.functional.mse_loss(
                predicted, residual_target
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for target_model, online, blend in ((target_encoder, encoder, 0.02), (target_anchor, anchor, 0.05)):
                    for fixed, updated in zip(target_model.parameters(), online.parameters(), strict=True):
                        fixed.lerp_(updated, blend)

    def embedding(values, statistics):
        fixed_anchor = target_anchor(statistics)
        update = revision(torch.cat((encode(encoder, values), fixed_anchor), dim=1))
        return torch.cat((statistics, fixed_anchor, update), dim=1)

    with torch.no_grad():
        frozen = embedding(market_context, priors)
        labels = ((future[:, :, 0] - future[:, :, 1]).cumsum(1) > 0).float()
    readout = nn.Sequential(nn.Linear(frozen.shape[1], width), nn.GELU(), nn.Linear(width, HORIZON)).to(device)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=0.002)
    for _ in range(12):
        for batch in batches():
            loss = nn.functional.binary_cross_entropy_with_logits(readout(frozen[batch]), labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(values: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            normalized = normalize(inputs(values))
            return readout(embedding(normalized, prior(normalized))).sigmoid().cpu().numpy()

    return predict


class StockJepaClient:
    provider = "local"
    model = MODEL
    endpoint = "local://stock-jepa"

    def __init__(self, data: DataService, *, device: str = "cpu"):
        self.data = data
        self.device = device
        self.predict = None

    async def decide(self, state: dict, questions: dict[str, Question]) -> DcnResult:
        if not questions:
            return DcnResult(requests=0)
        day = date.fromisoformat(state["as_of"])
        horizon = state["horizon_trading_days"]
        if not 1 <= horizon <= HORIZON or any(question.type != "noul" for question in questions.values()):
            raise DcnError("Stock-JEPA supports binary questions with 1–21 trading-session horizons")
        tickers = [question.instructions["ticker"] for question in questions.values()]
        started = time.perf_counter()
        if self.predict is None:
            panel = market_panel(self.data, tickers, day - timedelta(days=1), limit=10_000)
            self.predict = train_predictor(panel, self.device)
        context = market_panel(self.data, tickers, day, limit=CONTEXT + 1)
        valid = np.isfinite(context).all(axis=(1, 2))
        if context.shape[1] != CONTEXT or not valid.any():
            return DcnResult(missing_ids=list(questions), requests=0)
        probabilities = self.predict(context[valid])[:, horizon - 1]
        answers = {
            key: NoulAnswer(noul=float(probability))
            for key, probability in zip(np.asarray(list(questions))[valid], probabilities, strict=True)
        }
        return DcnResult(answers=answers, missing_ids=[key for key in questions if key not in answers],
                         latency_s=time.perf_counter() - started)

    async def close(self) -> None:
        pass
