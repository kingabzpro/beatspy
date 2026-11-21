"""Run and replay all three published BeatSPY windows on a Kaggle GPU."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import torch
from published_windows import benchmark as run_published_windows

from beatspy.config import DecisionConfig, Settings
from beatspy.dcn.timestral import TimestralClient


async def benchmark(output: Path) -> None:
    settings = Settings(decision=DecisionConfig(decision_model=TimestralClient.model))
    forecaster = TimestralClient.load_forecaster()
    if forecaster.device.type != "cuda":
        raise RuntimeError(f"Expected a CUDA GPU for Timestral, got {forecaster.device}")
    gpu = torch.cuda.get_device_name(forecaster.device)
    print(f"Running {TimestralClient.model} on {gpu}", flush=True)
    runtime = {
        "model": TimestralClient.checkpoint,
        "checkpoint_revision": TimestralClient.revision,
        "gpu": gpu,
        "torch_version": str(torch.__version__),
    }
    await run_published_windows(output, settings, lambda data: TimestralClient(data, forecaster=forecaster), runtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/timestral-results"))
    asyncio.run(benchmark(parser.parse_args().output))
