"""Run logistic regression and its fixed LightGBM blend on the published windows."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import lightgbm
import sklearn
from published_windows import benchmark

from beatspy.config import DecisionConfig, Settings
from beatspy.data.service import DataService
from beatspy.dcn.clients import DcnClient, make_client


async def score_model(model: str, output: Path) -> None:
    settings = Settings(decision=DecisionConfig(decision_model=model))

    def client(data: DataService) -> DcnClient:
        return make_client(settings, data=data)

    runtime = {
        "model": model, "scikit_learn_version": sklearn.__version__,
        "lightgbm_version": lightgbm.__version__, "device": "CPU",
    }
    await benchmark(output, settings, client, runtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=("logistic-regression", "logistic-lightgbm"))
    args = parser.parse_args()
    asyncio.run(score_model(args.model, args.output))
