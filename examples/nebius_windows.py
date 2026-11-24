"""Benchmark Nebius structured-output models on the three published windows."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from published_windows import benchmark

from beatspy.artifacts import atomic_json
from beatspy.config import NEBIUS_MODELS, DecisionConfig, Settings
from beatspy.dcn.verify import read_json


async def run(output: Path, models: list[str]) -> int:
    results = []
    errors = {}
    for name in models:
        settings = Settings(decision=DecisionConfig(decision_model=name))
        model_output = output / name
        print(f"Running {settings.decision.label} on Nebius", flush=True)
        try:
            await benchmark(model_output, settings, None, {"model": name, "provider": "nebius"})
        except Exception as exc:
            errors[name] = str(exc)
            print(f"{name} failed: {exc}", flush=True)
        else:
            results.append(read_json(model_output / "summary.json"))
        atomic_json(output / "summary.json", {"models": results, "errors": errors})
    return int(bool(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(NEBIUS_MODELS), action="append")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.output, args.model or list(NEBIUS_MODELS))))
