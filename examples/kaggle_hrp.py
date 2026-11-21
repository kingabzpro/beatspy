"""Run and replay Hierarchical Risk Parity on BeatSPY's published windows."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import scipy
from published_windows import benchmark

from beatspy.config import DecisionConfig, Settings
from beatspy.dcn.hrp import LOOKBACK, MODEL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/hrp-results"))
    settings = Settings(decision=DecisionConfig(decision_model=MODEL))
    runtime = {"model": MODEL, "lookback_sessions": LOOKBACK, "scipy_version": scipy.__version__, "device": "CPU"}
    asyncio.run(benchmark(parser.parse_args().output, settings, None, runtime))
