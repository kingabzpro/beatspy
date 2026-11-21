"""Run the walk-forward LightGBM baseline on Kaggle."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import lightgbm
from published_windows import benchmark

from beatspy.config import DecisionConfig, Settings
from beatspy.dcn.clients import make_client

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/lightgbm-results"))
    settings = Settings(decision=DecisionConfig(decision_model="lightgbm"))
    runtime = {"model": "LightGBM", "lightgbm_version": lightgbm.__version__, "device": "CPU"}
    asyncio.run(benchmark(parser.parse_args().output, settings, lambda data: make_client(settings, data=data), runtime))
