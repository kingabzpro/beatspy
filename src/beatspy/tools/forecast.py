"""Forecast providers: deterministic statistical baselines plus optional extras.

- baseline: log-drift with a p10-p90 (80%) band from trailing volatility (default, no deps)
- naive: flat forecast (control: does forecasting help at all?)
- chronos: local Amazon Chronos (requires `uv sync --extra chronos`)
- timegpt: Nixtla TimeGPT API (requires `uv sync --extra timegpt` + API key)
"""

from __future__ import annotations

import logging
import math
import statistics
import threading
from dataclasses import dataclass

log = logging.getLogger(__name__)

Z90 = 1.2816


@dataclass
class ForecastResult:
    ticker: str
    horizon_days: int
    expected_return_pct: float
    direction: str
    p10_pct: float
    p50_pct: float
    p90_pct: float
    method: str

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "horizon_days": self.horizon_days,
            "expected_return_pct": self.expected_return_pct,
            "direction": self.direction,
            "p10_pct": self.p10_pct,
            "p50_pct": self.p50_pct,
            "p90_pct": self.p90_pct,
            "method": self.method,
        }


def _direction(expected_pct: float) -> str:
    if expected_pct > 0.5:
        return "up"
    if expected_pct < -0.5:
        return "down"
    return "flat"


def baseline_forecast(ticker: str, closes: list[float], horizon: int, method: str = "drift") -> ForecastResult | None:
    """Log-drift (or flat) mean with a normal-approximation p10-p90 (80%) band. Deterministic."""
    if len(closes) < 20 or horizon <= 0:
        return None
    last = closes[-1]
    log_rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    sigma = statistics.stdev(log_rets) * math.sqrt(horizon)
    mu = 0.0 if method == "naive" else (sum(log_rets) / len(log_rets)) * horizon

    def pct(value: float) -> float:
        return round((value / last - 1.0) * 100.0, 2)

    expected = pct(last * math.exp(mu))
    return ForecastResult(
        ticker=ticker,
        horizon_days=horizon,
        expected_return_pct=expected,
        direction=_direction(expected),
        p10_pct=pct(last * math.exp(mu - Z90 * sigma)),
        p50_pct=expected,
        p90_pct=round(pct(last * math.exp(mu + Z90 * sigma)), 2),
        method=method,
    )


_chronos_pipeline = None
_chronos_lock = threading.Lock()


def chronos_forecast(ticker: str, closes: list[float], horizon: int, seed: int = 42) -> ForecastResult | None:
    with _chronos_lock:
        return _chronos_forecast(ticker, closes, horizon, seed)


def _chronos_forecast(ticker: str, closes: list[float], horizon: int, seed: int) -> ForecastResult | None:
    global _chronos_pipeline
    if len(closes) < 20 or horizon <= 0:
        return None
    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("forecast_provider=chronos requires the optional extra: uv sync --extra chronos") from exc
    if _chronos_pipeline is None:
        _chronos_pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small")
    torch.manual_seed(seed)
    context = torch.tensor([closes], dtype=torch.float32)
    samples = _chronos_pipeline.predict(context=context, prediction_length=horizon, num_samples=50)
    # Chronos returns (batch, sample paths, forecast steps).
    final = samples[0, :, -1].numpy()
    last = closes[-1]

    def pct(value: float) -> float:
        return round((float(value) / last - 1.0) * 100.0, 2)

    # samples are unsorted draws: positional indexing is not a quantile
    import numpy as np

    q10, q50, q90 = (float(v) for v in np.quantile(final, [0.1, 0.5, 0.9]))
    p10, p50, p90 = pct(q10), pct(q50), pct(q90)
    return ForecastResult(
        ticker=ticker,
        horizon_days=horizon,
        expected_return_pct=p50,
        direction=_direction(p50),
        p10_pct=p10,
        p50_pct=p50,
        p90_pct=p90,
        method="chronos-t5-small",
    )


def timegpt_forecast(ticker: str, closes: list[float], horizon: int, api_key: str | None) -> ForecastResult | None:
    if len(closes) < 20 or horizon <= 0:
        return None
    if not api_key:
        raise RuntimeError("forecast_provider=timegpt requires BEATSPY_TIMEGPT_API_KEY")
    try:
        from nixtla import NixtlaClient
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("forecast_provider=timegpt requires the optional extra: uv sync --extra timegpt") from exc
    import pandas as pd

    client = NixtlaClient(api_key=api_key)
    # NixtlaClient.forecast expects the standard ds/y column layout
    frame = pd.DataFrame({"ds": pd.date_range("2000-01-01", periods=len(closes), freq="D"), "y": closes})
    forecast = client.forecast(df=frame, h=horizon, freq="D", model="timegpt-1")
    last = closes[-1]
    mid = float(forecast["TimeGPT"].iloc[-1])
    log_rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    sigma = statistics.stdev(log_rets) * math.sqrt(horizon)

    def pct(value: float) -> float:
        return round((value / last - 1.0) * 100.0, 2)

    expected = pct(mid)
    return ForecastResult(
        ticker=ticker,
        horizon_days=horizon,
        expected_return_pct=expected,
        direction=_direction(expected),
        p10_pct=round(pct(last * math.exp(math.log(mid / last) - Z90 * sigma)), 2),
        p50_pct=expected,
        p90_pct=round(pct(last * math.exp(math.log(mid / last) + Z90 * sigma)), 2),
        method="timegpt-1",
    )


def get_forecast_fn(provider: str, timegpt_api_key: str | None = None):
    """Resolve a forecast provider name to forecast(ticker, closes, horizon) -> dict|None."""
    name = (provider or "baseline").lower()
    if name in ("baseline", "drift"):
        return lambda ticker, closes, horizon: baseline_forecast(ticker, closes, horizon, "drift")
    if name == "naive":
        return lambda ticker, closes, horizon: baseline_forecast(ticker, closes, horizon, "naive")
    if name == "chronos":
        return chronos_forecast
    if name == "timegpt":
        return lambda ticker, closes, horizon: timegpt_forecast(ticker, closes, horizon, timegpt_api_key)
    raise ValueError(f"unknown forecast provider {provider!r}; use baseline, naive, chronos, or timegpt")
