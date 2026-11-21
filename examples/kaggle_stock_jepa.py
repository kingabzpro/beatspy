"""Run BeatSPY's STOCK-JEPA adaptation on the three frozen windows."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import torch
from published_windows import benchmark

from beatspy.config import DecisionConfig, Settings
from beatspy.data.service import DataService
from beatspy.dcn.stock_jepa import CONTEXT, MODEL, SEED, StockJepaClient


def client(data: DataService) -> StockJepaClient:
    return StockJepaClient(data, device="cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/stock-jepa-results"))
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator for this benchmark")
    settings = Settings(decision=DecisionConfig(decision_model=MODEL))
    runtime = {
        "model": MODEL, "adaptation": "BeatSPY", "seed": SEED, "context_sessions": CONTEXT,
        "gpu": torch.cuda.get_device_name(), "torch_version": str(torch.__version__),
    }
    print(f"Running {MODEL} on {runtime['gpu']}", flush=True)
    asyncio.run(benchmark(parser.parse_args().output, settings, client, runtime))
