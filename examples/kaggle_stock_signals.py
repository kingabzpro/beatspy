"""Benchmark five equity signals on the three published BeatSPY windows."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import lightgbm
import pandas as pd
import sklearn
from published_windows import benchmark
from threadpoolctl import threadpool_limits

from beatspy.artifacts import atomic_json
from beatspy.config import STOCK_SIGNAL_MODELS, DecisionConfig, Settings
from beatspy.dcn.runner import read_run
from beatspy.dcn.stock_signals import SEED
from beatspy.dcn.verify import read_json


def mean_rank_ic(run: Path) -> float | None:
    _, _, decisions = read_run(run)
    correlations = []
    for record in decisions:
        observed = pd.DataFrame(record["observed"])
        if observed["noul"].nunique() > 1 and observed["excess_return"].nunique() > 1:
            correlations.append(observed["noul"].corr(observed["excess_return"], method="spearman"))
    return round(sum(correlations) / len(correlations), 6) if correlations else None


async def run(output: Path) -> None:
    results = []
    with threadpool_limits(limits=2):
        for model in STOCK_SIGNAL_MODELS:
            settings = Settings(decision=DecisionConfig(decision_model=model))
            runtime = {
                "model": model, "device": "CPU", "seed": SEED,
                "lightgbm_version": lightgbm.__version__, "scikit_learn_version": sklearn.__version__,
            }
            print(f"Running {model} on CPU", flush=True)
            model_output = output / model
            await benchmark(model_output, settings, None, runtime)
            summary_file = model_output / "summary.json"
            summary = read_json(summary_file)
            for window in summary["windows"]:
                window["rank_ic"] = mean_rank_ic(model_output / window["run_id"])
            atomic_json(summary_file, summary)
            results.append(summary)
    atomic_json(output / "summary.json", {"models": results})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args().output))
