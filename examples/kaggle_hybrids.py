"""Run the five offline hybrids and paired controls in one Kaggle session."""

from __future__ import annotations

import argparse
import asyncio
import csv
from functools import partial
from importlib.metadata import version
from pathlib import Path

import torch
from published_windows import REPOSITORY, benchmark
from threadpoolctl import threadpool_limits

from beatspy.artifacts import atomic_json
from beatspy.config import HYBRID_CONTROLS, HYBRID_MODELS, DecisionConfig, Settings
from beatspy.dcn.foundation_forecast import Chronos2Client
from beatspy.dcn.hybrids import load_chronos_checkpoint, make_hybrid_client
from beatspy.dcn.runner import read_run
from beatspy.dcn.verify import read_json

COMPARATORS = {
    "lambdamart-hrp": ("lambdamart-equal",),
    "chronos-lightgbm": ("lightgbm",),
    "momentum-reversal": ("residual-momentum", "residual-reversal", "momentum-reversal-blend"),
    "jepa-elastic-net": ("elastic-net",),
    "lightgbm-momentum": ("lightgbm",),
}


def check_same_picks(output: Path, results: dict) -> None:
    rank = results["lambdamart-hrp"]["windows"]
    equal = results["lambdamart-equal"]["windows"]
    for hrp, control in zip(rank, equal, strict=True):
        _, _, hrp_decisions = read_run(output / "lambdamart-hrp" / hrp["run_id"])
        _, _, equal_decisions = read_run(output / "lambdamart-equal" / control["run_id"])
        for decision, matched in zip(hrp_decisions, equal_decisions, strict=True):
            picks = set(decision["validated"]["weights"])
            if decision["date"] != matched["date"] or picks != set(matched["validated"]["weights"]):
                raise RuntimeError("The LambdaMART sizing comparison used different stock picks")


def write_comparison(output: Path, results: dict) -> None:
    catalog = read_json(REPOSITORY / "dashboard/data/index.json")["runs"]
    tested = {(model, row["window"]): row for model, data in results.items() for row in data["windows"]}
    fields = ("model", "window", "comparator", "strategy_return", "control_return", "spy_return",
              "strategy_vs_spy", "incremental_return", "strategy_drawdown", "control_drawdown",
              "strategy_turnover", "control_turnover")
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for model, controls in COMPARATORS.items():
            for row in results[model]["windows"]:
                for comparator in controls:
                    if control := tested.get((comparator, row["window"])):
                        baseline = control
                    else:
                        matches = [item for item in catalog if item["model"] == comparator
                                   and item["scenario"] == row["window"] and item["snapshot_id"] == row["snapshot_id"]
                                   and item["decisions"] == row["decisions"] and not item["synthetic"]]
                        baseline = max(matches, key=lambda item: item["created_utc"])["metrics"]
                    writer.writerow((model, row["window"], comparator, row["total_return"], baseline["total_return"],
                                     row["spy_total_return"], row["excess_return_vs_spy"],
                                     round(row["total_return"] - baseline["total_return"], 6),
                                     row["max_drawdown"], baseline["max_drawdown"],
                                     row["avg_turnover"], baseline["avg_turnover"]))


async def run(output: Path, checkpoint: Path, source_sha256: str, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU for the Chronos-2 and STOCK-JEPA hybrids")
    torch.set_num_threads(4 if device == "cpu" else 2)
    forecaster = load_chronos_checkpoint(checkpoint, device)
    runtime = {
        "source_sha256": source_sha256, "device": device, "torch": str(torch.__version__),
        **({"gpu": torch.cuda.get_device_name()} if device == "cuda" else {}),
        "chronos_revision": Chronos2Client.revision, "chronos_version": version("chronos-forecasting"),
        "lightgbm_version": version("lightgbm"), "scikit_learn_version": version("scikit-learn"),
    }
    results = {}
    with threadpool_limits(limits=2):
        for model in (HYBRID_MODELS | HYBRID_CONTROLS):
            settings = Settings(decision=DecisionConfig(decision_model=model))
            print(f"Running {settings.decision.label}", flush=True)
            folder = output / model
            factory = partial(make_hybrid_client, model=model, forecaster=forecaster, device=device)
            await benchmark(folder, settings, factory, {"model": model, "runtime": runtime})
            results[model] = read_json(folder / "summary.json")
    check_same_picks(output, results)
    write_comparison(output, results)
    atomic_json(output / "summary.json", {"models": results, "runtime": runtime})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    asyncio.run(run(args.output, args.checkpoint, args.source_sha256, args.device))
