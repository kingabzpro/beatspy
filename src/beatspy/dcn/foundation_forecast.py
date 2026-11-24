"""GPU adapters for pretrained quantile forecasters."""

from __future__ import annotations

import numpy as np

from .timestral import RelativeForecastClient

QUANTILE_LEVELS = tuple(level / 10 for level in range(1, 10))


class Chronos2Client(RelativeForecastClient):
    provider = "local"
    model = "chronos-2"
    endpoint = f"local://{model}"
    checkpoint = "amazon/chronos-2"
    revision = "29ec3766d36d6f73f0696f85560a422f50e8498c"
    package = "chronos-forecasting"

    @classmethod
    def load_forecaster(cls):
        from chronos import Chronos2Pipeline

        return Chronos2Pipeline.from_pretrained(cls.checkpoint, revision=cls.revision, device_map="cuda")

    def terminal_quantiles(self, context: np.ndarray, horizon: int):
        quantiles, _ = self.forecaster.predict_quantiles(
            list(context), prediction_length=horizon, quantile_levels=list(QUANTILE_LEVELS),
        )
        return np.sort(np.stack([np.asarray(item[0, -1]) for item in quantiles])), QUANTILE_LEVELS


class TimesFM3Client(RelativeForecastClient):
    provider = "local"
    model = "timesfm-3.0"
    endpoint = f"local://{model}"
    checkpoint = "google/timesfm-3.0-pytorch"
    revision = "43046b85ec22d584a13f8098c2ed39c889e129c2"
    package = "timesfm"

    @classmethod
    def load_forecaster(cls):
        from huggingface_hub import snapshot_download
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        checkpoint = snapshot_download(
            cls.checkpoint, revision=cls.revision, allow_patterns=["config.json", "model.safetensors", "LICENSE"],
        )
        return TimesFM3Evaluator(ModelConfig(checkpoint_path=checkpoint, per_core_batch_size=32, device="cuda"))

    def terminal_quantiles(self, context: np.ndarray, horizon: int):
        predictions = self.forecaster.predict_batch(
            list(context), horizon=horizon, return_quantiles=True, use_symmetric_averaging=False,
        )
        return np.sort(np.stack([item.quantiles[-1] for item in predictions])), QUANTILE_LEVELS


class T0GammaClient(RelativeForecastClient):
    provider = "local"
    model = "t0-gamma"
    endpoint = f"local://{model}"
    checkpoint = "theforecastingcompany/t0-gamma"
    revision = "fef67abad3eed4f227c7b536a6e8ee4e9ab1080d"
    package = "tfc-t0"

    @classmethod
    def load_forecaster(cls):
        from t0 import T0Forecaster

        return T0Forecaster.from_pretrained(cls.checkpoint, revision=cls.revision).to("cuda").eval()

    def terminal_quantiles(self, context: np.ndarray, horizon: int):
        prediction = self.forecaster.predict(context, horizon=horizon, quantile_levels=list(QUANTILE_LEVELS))
        return np.sort(np.asarray(prediction.quantiles[:, -1].cpu())), QUANTILE_LEVELS


FORECAST_CLIENTS = {client.model: client for client in (Chronos2Client, TimesFM3Client, T0GammaClient)}
