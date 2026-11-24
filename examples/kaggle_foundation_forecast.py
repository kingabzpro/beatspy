"""Run a pretrained forecast model over all three frozen windows on Kaggle GPU."""

from __future__ import annotations

import argparse
import asyncio
from importlib.metadata import version
from pathlib import Path

import torch
from published_windows import benchmark

from beatspy.config import DecisionConfig, Settings
from beatspy.dcn.foundation_forecast import FORECAST_CLIENTS, QUANTILE_LEVELS
from beatspy.dcn.timestral import CONTEXT_SESSIONS


async def run(model: str, output: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator for this benchmark")
    client = FORECAST_CLIENTS[model]
    print(f"Running {model} on {torch.cuda.get_device_name()}", flush=True)
    forecaster = client.load_forecaster()
    settings = Settings(decision=DecisionConfig(decision_model=model))
    runtime = {
        "model": client.checkpoint,
        "checkpoint_revision": client.revision,
        "package": client.package,
        "package_version": version(client.package),
        "context_sessions": CONTEXT_SESSIONS,
        "quantile_levels": QUANTILE_LEVELS,
        "gpu": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
    }
    await benchmark(output, settings, lambda data: client(data, forecaster=forecaster), runtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=FORECAST_CLIENTS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.model, args.output))
