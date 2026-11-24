"""Run and replay the three published BeatSPY windows on their exact snapshots."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

from tpi_variants import WINDOWS

from beatspy.artifacts import atomic_json
from beatspy.config import Settings
from beatspy.data.freeze import read_snapshot
from beatspy.data.service import DataService
from beatspy.dcn.clients import DcnClient
from beatspy.dcn.runner import read_run, run_dcn
from beatspy.dcn.verify import read_json, replay_run
from beatspy.reporting.catalog import comparison_group, decision_actions, summary
from beatspy.scenarios import load_scenario

REPOSITORY = Path(__file__).resolve().parents[1]


def progress(index: int, count: int, day: date) -> None:
    print(f"  decision {index}/{count} {day}", flush=True)


async def benchmark(
    output: Path,
    settings: Settings,
    client_factory: Callable[[DataService], DcnClient] | None,
    runtime: dict,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    published = read_json(REPOSITORY / "dashboard/data/index.json")["runs"]
    board_groups = {row["scenario"]: row["group"] for row in published if row["model"] == "minara-kairos"}
    results = []
    catalog_rows = []
    for name, window in WINDOWS.items():
        start, end = window["start"], window["end"]
        scenario = load_scenario(name).model_copy(update={"start": start, "end": end})
        snapshot = REPOSITORY / window["snapshot"]
        prices, manifest = read_snapshot(snapshot, scenario)
        data = DataService(prices, benchmark=scenario.benchmark)
        manifest = {**manifest, "path": str(snapshot)}
        client = client_factory(data) if client_factory else None

        print(f"{name} ({start} through {end})", flush=True)
        run_dir = await run_dcn(
            scenario,
            settings,
            start=start,
            end=end,
            prepared_data=(data, manifest),
            out_root=output,
            progress=progress,
            client=client,
            max_consecutive_failures=1,
        )
        meta, metrics, decisions = read_run(run_dir)
        allocation_only = meta["decision"]["question_form"] == "allocation"
        if any(
            line["coverage"] != (0.0 if allocation_only else 1.0) or line["error"]
            or line["validated"]["invalid"] or len(line["observed"]) != len(line["questions"])
            for line in decisions
        ):
            raise RuntimeError(f"{name} has incomplete forecast or outcome coverage")
        group = comparison_group(meta)
        if not allocation_only and not meta["decision"].get("portfolio_method") and group != board_groups[name]:
            raise RuntimeError(f"{name} comparison group {group} differs from the published {board_groups[name]}")
        await replay_run(run_dir)
        row = summary(meta, metrics, f"runs/{run_dir.name}")
        row["snapshot_dir"] = f"snapshots/{meta['data']['snapshot_id']}"
        row["actions"] = None if allocation_only else decision_actions(run_dir)
        catalog_rows.append(row)
        calibration = (metrics["calibration"] or {}).get("signals", {}).get("noul", {})
        results.append({
            "window": name,
            "start": scenario.start,
            "end": scenario.end,
            "run_id": meta["run_id"],
            "snapshot_id": meta["data"]["snapshot_id"],
            "comparison_group": group,
            "decisions": metrics["decisions"],
            "answer_coverage": metrics["answer_coverage"],
            "total_return": metrics["total_return"],
            "spy_total_return": metrics["spy_total_return"],
            "excess_return_vs_spy": metrics["excess_return_vs_spy"],
            "equal_weight_return": metrics["baselines"]["equal_weight"],
            "max_drawdown": metrics["max_drawdown"],
            "avg_turnover": metrics["avg_turnover"],
            "brier": calibration.get("brier"),
            "brier_skill_score": calibration.get("brier_skill_score"),
            "auc": calibration.get("auc"),
        })
        print(f"{name} replay verified, SPY gap {metrics['excess_return_vs_spy']:+.2%}", flush=True)

    atomic_json(output / "catalog_rows.json", catalog_rows)
    atomic_json(output / "summary.json", {**runtime, "windows": results})
