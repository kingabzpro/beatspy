"""Parallel execution overlaps without crossing dependency or budget boundaries."""

import asyncio
import json
from contextlib import suppress
from dataclasses import replace
from datetime import date

from beatspy.bench.batch import run_batch
from beatspy.tools.market import _external
from conftest import FakeExecutor, make_prices, make_tctx


def test_shared_limits_stage_order_and_single_snapshot(tmp_path, scenario, settings, monkeypatch):
    downloads = []
    monkeypatch.setattr(
        "beatspy.data.freeze.download_ohlc", lambda *a: downloads.append(a) or make_prices(scenario.universe)
    )

    class TrackingExecutor(FakeExecutor):
        active = 0
        peak = 0
        finished = {}

        async def run(self, agent, input_text, tctx):
            key = (agent.model, tctx.as_of)
            done = self.finished.setdefault(key, set())
            if agent.name == "critic":
                assert {"research", "analyst", "forecaster"} <= done
            if agent.name == "portfolio_manager":
                assert "critic" in done
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0.005)
            result = await super().run(agent, input_text, tctx)
            self.active -= 1
            done.add(agent.name)
            return result

    executor = TrackingExecutor()
    paths = asyncio.run(
        run_batch(
            [scenario],
            settings,
            ["first", "second"],
            jobs=2,
            concurrency=4,
            executor=executor,
            data_root=tmp_path / "cache",
            out_root=tmp_path / "results",
        )
    )
    assert len(downloads) == 1
    assert 2 <= executor.peak <= 4
    assert all(not isinstance(path, Exception) for path in paths)
    assert paths[0] != paths[1]
    assert json.loads((paths[0] / "metrics.json").read_text()) == json.loads((paths[1] / "metrics.json").read_text())
    assert settings.model.model == "gpt-4o-mini"


def test_failure_keeps_completed_runs(tmp_path, scenario, settings, monkeypatch):
    monkeypatch.setattr("beatspy.data.freeze.download_ohlc", lambda *a: make_prices(scenario.universe))

    class FailingExecutor(FakeExecutor):
        async def run(self, agent, input_text, tctx):
            if agent.model == "broken":
                raise RuntimeError("fatal test failure")
            return await super().run(agent, input_text, tctx)

    outcomes = asyncio.run(
        run_batch(
            [scenario],
            settings,
            ["good", "broken"],
            executor=FailingExecutor(),
            out_root=tmp_path / "results",
            data_root=tmp_path / "cache",
        )
    )
    assert (outcomes[0] / "metrics.json").exists()
    assert isinstance(outcomes[1], Exception)
    assert "broken" in str(outcomes[1])


def test_external_work_is_bounded_off_event_loop_and_budgets_are_isolated(data_service, scenario):
    import threading
    import time

    active = 0
    peak = 0
    lock = threading.Lock()

    def work():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return "ok"

    async def exercise():
        base = make_tctx(data_service, scenario, date(2022, 2, 1), budget=2)
        base.external_limit = asyncio.Semaphore(2)
        contexts = [replace(base, agent=str(i)) for i in range(6)]
        for ctx in contexts:
            assert ctx.spend(ctx.agent, "external") is None
        results = await asyncio.gather(*(_external(ctx, work) for ctx in contexts))
        assert results == ["ok"] * 6
        assert base.used == {str(i): 1 for i in range(6)}

    asyncio.run(exercise())
    assert peak == 2


def test_cancellation_drains_batch_tasks(tmp_path, scenario, settings, monkeypatch):
    monkeypatch.setattr("beatspy.data.freeze.download_ohlc", lambda *a: make_prices(scenario.universe))

    async def exercise():
        entered = asyncio.Event()

        class WaitingExecutor(FakeExecutor):
            async def run(self, agent, input_text, tctx):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(
            run_batch(
                [scenario],
                settings,
                executor=WaitingExecutor(),
                out_root=tmp_path / "results",
                data_root=tmp_path / "cache",
            )
        )
        await entered.wait()
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        assert not [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]

    asyncio.run(exercise())


def test_external_cancellation_keeps_slot_until_worker_finishes(data_service, scenario):
    import threading

    started, finish = threading.Event(), threading.Event()

    def work():
        started.set()
        assert finish.wait(2)

    async def exercise():
        ctx = make_tctx(data_service, scenario)
        ctx.external_limit = asyncio.Semaphore(1)
        task = asyncio.create_task(_external(ctx, work))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert ctx.external_limit.locked() and not task.done()
        finish.set()
        with suppress(asyncio.CancelledError):
            await task
        assert not ctx.external_limit.locked()

    asyncio.run(exercise())


def test_snapshot_cache_race_downloads_once(tmp_path, scenario, monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    from beatspy.data.freeze import load_snapshot

    downloads = []

    def download(*args):
        downloads.append(args)
        time.sleep(0.02)
        return make_prices(scenario.universe)

    monkeypatch.setattr("beatspy.data.freeze.download_ohlc", download)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: load_snapshot(scenario, tmp_path), range(4)))
    assert len(downloads) == 1
    assert len({manifest["snapshot_id"] for _, manifest in results}) == 1
    assert not list((tmp_path / "snapshots").glob("snapshot-*"))
