"""Run the direct structured-output LLMs on the published BeatSPY windows."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from published_windows import benchmark

from beatspy.artifacts import atomic_json
from beatspy.config import DIRECT_LLM_MODELS, DecisionConfig, Settings
from beatspy.dcn.verify import read_json


async def run(output: Path, models: list[str]) -> int:
    async def bench(name: str) -> dict:
        settings = Settings(decision=DecisionConfig(decision_model=name))
        model_output = output / name
        print(f"Running {settings.decision.label}", flush=True)
        await benchmark(model_output, settings, None, {"model": name, "provider": settings.decision.provider})
        return read_json(model_output / "summary.json")

    output.mkdir(parents=True, exist_ok=True)
    results = await asyncio.gather(*(bench(name) for name in models), return_exceptions=True)
    errors = {name: str(result) for name, result in zip(models, results, strict=True) if isinstance(result, Exception)}
    atomic_json(output / "summary.json", {
        "models": [result for result in results if isinstance(result, dict)],
        "errors": errors,
    })
    for name, error in errors.items():
        print(f"{name} failed: {error}", flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=DIRECT_LLM_MODELS, action="append")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.output, list(dict.fromkeys(args.model or DIRECT_LLM_MODELS)))))
