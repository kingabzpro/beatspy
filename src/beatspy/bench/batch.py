"""Bounded parallel independent runs, with one snapshot preparation per scenario."""

import asyncio
from datetime import UTC, datetime

import httpx

from ..data.calendar import resolve_scenario
from .runner import ensure_data, run_benchmark


async def run_batch(scenarios, settings, models=None, *, jobs=2, concurrency=6, **kwargs):
    if jobs < 1 or concurrency < 1:
        raise ValueError("jobs and concurrency must be positive")
    now = datetime.now(UTC)
    run_limit = asyncio.Semaphore(jobs)
    agent_limit = asyncio.Semaphore(concurrency)
    external_limit = asyncio.Semaphore(concurrency)
    prepared = {}

    async def prepare(scenario):
        async with external_limit:
            work = asyncio.create_task(
                asyncio.to_thread(ensure_data, scenario, kwargs.get("data_root"), kwargs.get("refresh_data", False))
            )
            try:
                return await asyncio.shield(work)
            except asyncio.CancelledError:
                await asyncio.gather(work, return_exceptions=True)
                raise

    async def one(original, model, client):
        async with run_limit:
            try:
                scenario = resolve_scenario(original, now)
                scenario = scenario.model_copy(
                    update={
                        "start": kwargs.get("start") or scenario.start,
                        "end": kwargs.get("end") or scenario.end,
                        "frequency": kwargs.get("freq") or scenario.frequency,
                    }
                )
                key = scenario.model_dump_json()
                if key not in prepared:
                    prepared[key] = asyncio.create_task(prepare(scenario))
                data = await asyncio.shield(prepared[key])
                model_settings = settings.model_copy(deep=True)
                model_settings.model.model = model
                path = await run_benchmark(
                    scenario,
                    model_settings,
                    prepared_data=data,
                    agent_limit=agent_limit,
                    external_limit=external_limit,
                    http_client=client,
                    **kwargs,
                )
                return path
            except Exception as exc:
                return RuntimeError(f"{original.name} / {model}: {exc}")

    with httpx.Client() as client:
        tasks = [asyncio.create_task(one(s, m, client)) for s in scenarios for m in (models or [settings.model.model])]
        try:
            return await asyncio.gather(*tasks)
        finally:
            for task in [*tasks, *prepared.values()]:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, *prepared.values(), return_exceptions=True)
