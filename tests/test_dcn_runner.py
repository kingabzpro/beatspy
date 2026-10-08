"""The run loop end to end, offline: artifacts, reproducibility, and failure modes."""

from __future__ import annotations

import json

import pytest

from beatspy.dcn.clients import DcnResult
from beatspy.dcn.runner import calibration_from_decisions, read_run, summarize_decisions
from conftest import OfflineDcnClient, run_offline


def test_a_run_writes_the_complete_artifact_set(dcn_run_dir):
    from beatspy.artifacts import RUN_FILES

    for name in RUN_FILES:
        assert (dcn_run_dir / name).is_file(), f"missing {name}"
    assert (dcn_run_dir / "run.json").is_file()
    assert (dcn_run_dir / "snapshot" / "prices.csv").is_file()
    assert (dcn_run_dir / "snapshot" / "MANIFEST.json").is_file()


def test_every_question_form_is_scored_per_form(tmp_path, scenario, frozen_data):
    """One run per form, and only the forms actually asked appear in the report."""
    twin = run_offline(tmp_path, scenario, frozen_data, OfflineDcnClient(), question_form="twin")
    rank = run_offline(tmp_path, scenario, frozen_data, OfflineDcnClient(), question_form="rank")
    everything = run_offline(tmp_path, scenario, frozen_data, OfflineDcnClient(), question_form="all")

    _, twin_metrics, _ = read_run(twin)
    _, rank_metrics, _ = read_run(rank)
    _, all_metrics, _ = read_run(everything)

    assert set(twin_metrics["calibration"]["signals"]) == {"noul", "choice"}
    assert set(rank_metrics["calibration"]["signals"]) == {"rank"}
    assert set(all_metrics["calibration"]["signals"]) == {"noul", "choice", "rank"}
    # A form that was never asked cannot produce a gap.
    assert rank_metrics["calibration"]["miscalibration_gap"] is None
    assert twin_metrics["calibration"]["miscalibration_gap"] is not None


def test_run_metadata_identifies_the_decision_model_and_policy(dcn_run_dir):
    meta, _, _ = read_run(dcn_run_dir)
    assert meta["benchmark"] == "dcn"
    assert meta["schema_version"] == 3
    assert meta["dcn_protocol"] >= 1
    assert meta["synthetic"] is False
    decision = meta["decision"]
    assert decision["decision_model"]
    assert decision["provider"] in ("openai_decisions", "typesafe", "cloudflare")
    assert decision["question_form"] == "twin"
    assert decision["policy"] == {"min_probability": 0.5, "top_n": 3, "min_names": 2}
    assert decision["pricing"]["cost_per_m_input"] is not None
    # The snapshot the run was scored against is recorded for replay.
    assert meta["data"]["prices_sha256"]
    assert meta["artifact_hashes"]["metrics.json"]


def test_metrics_carry_both_halves_of_the_benchmark(dcn_run_dir):
    _, metrics, lines = read_run(dcn_run_dir)
    # Portfolio half.
    for field in (
        "total_return",
        "cagr",
        "sharpe",
        "max_drawdown",
        "spy_total_return",
        "excess_return_vs_spy",
        "outperformance_hit_rate",
        "avg_turnover",
        "trades",
    ):
        assert field in metrics, f"missing portfolio metric {field}"
    # Calibration half.
    calibration = metrics["calibration"]
    assert calibration["horizons"] == metrics["decisions"] == len(lines)
    assert calibration["observations"] > 0
    assert set(calibration["signals"]) == {"noul", "choice"}
    for stats in calibration["signals"].values():
        assert stats["n"] > 0
        assert 0.0 <= stats["brier"] <= 1.0
        assert len(stats["bins"]) == 10
    # The four deterministic references are scored on the same frozen prices.
    assert set(metrics["baselines"]) == {"equal_weight", "sixty_forty", "momentum_12_1"}


def test_cost_is_input_only_for_decision_models(dcn_run_dir):
    _, metrics, _ = read_run(dcn_run_dir)
    assert metrics["output_tokens"] == 0  # decision models do not generate tokens
    assert metrics["estimated_cost_usd"] is not None
    # 500 input tokens per call at a non-zero input rate.
    assert metrics["estimated_cost_usd"] > 0


def test_equity_curve_scores_the_portfolio_against_every_baseline(dcn_run_dir):
    import pandas as pd

    equity = pd.read_csv(dcn_run_dir / "equity_curve.csv")
    assert set(equity.columns) == {
        "date",
        "portfolio",
        "spy_buy_hold",
        "equal_weight",
        "sixty_forty",
        "momentum_12_1",
    }
    assert len(equity) > 10
    assert (equity["portfolio"] > 0).all()


def test_each_decision_records_its_questions_answers_and_state_hash(dcn_run_dir):
    _, _, lines = read_run(dcn_run_dir)
    for line in lines:
        assert line["state_sha256"]
        assert line["state_chars"] > 0
        assert line.get("call", {}).get("question_ids")
        assert set(line["questions"])  # the exact typed questions asked
        assert line["tickers"]
        assert line["probabilities"]
        assert line["validated"]["weights"] is not None
        # Every ticker asked about is scored against what actually happened.
        assert {row["ticker"] for row in line["observed"]} <= {t.upper() for t in line["tickers"]}


def test_the_same_scripted_model_produces_an_identical_run(tmp_path, scenario, frozen_data):
    """Two runs of the same scripted model must agree exactly."""
    first = run_offline(tmp_path, scenario, frozen_data, OfflineDcnClient())
    second = run_offline(tmp_path, scenario, frozen_data, OfflineDcnClient())
    assert (first / "metrics.json").read_text(encoding="utf-8") == (second / "metrics.json").read_text(
        encoding="utf-8"
    )
    assert (first / "decisions.jsonl").read_text(encoding="utf-8") == (second / "decisions.jsonl").read_text(
        encoding="utf-8"
    )
    assert (first / "equity_curve.csv").read_text(encoding="utf-8") == (second / "equity_curve.csv").read_text(
        encoding="utf-8"
    )


def test_every_probability_is_scored_against_a_relative_outcome(dcn_run_dir):
    """Labels are benchmark-relative, so a rising market cannot mark everything right."""
    _, metrics, lines = read_run(dcn_run_dir)
    outcomes = [row["outcome"] for line in lines for row in line["observed"]]
    assert set(outcomes) <= {0, 1}
    assert 0 < sum(outcomes) < len(outcomes), "outcomes should not be all the same"
    excess = [row["excess_return"] for line in lines for row in line["observed"]]
    # The sign of the excess return must agree with the recorded label.
    for row in (row for line in lines for row in line["observed"]):
        assert row["outcome"] == int(row["excess_return"] > 0)
    assert any(value > 0 for value in excess) and any(value < 0 for value in excess)


class ExplodingClient:
    provider = "offline"
    model = "exploding"
    endpoint = "https://offline.invalid"

    async def decide(self, state, questions):
        raise RuntimeError("boom")

    async def close(self):
        return None


def test_a_completely_failed_provider_holds_the_portfolio_and_records_it(tmp_path, scenario, frozen_data):
    run_dir = run_offline(tmp_path, scenario, frozen_data, ExplodingClient(), question_form="noul")
    _, metrics, lines = read_run(run_dir)
    assert metrics["invalid_outputs"] == len(lines)
    assert metrics["answer_coverage"] == 0.0
    assert metrics["trades"] == 0  # never liquidated into a crash
    for line in lines:
        assert line["error"]
        assert line["validated"]["invalid"] is True
        assert line["validated"]["cash"] == 1.0
        assert line["observed"] == []  # nothing was asked, so nothing is scored
    assert metrics["calibration"]["observations"] == 0


def test_calibration_can_be_rebuilt_from_the_decisions_alone(dcn_run_dir):
    """Replay must be possible without the network and without re-deriving labels."""
    _, metrics, lines = read_run(dcn_run_dir)
    rebuilt = calibration_from_decisions(lines)
    stored = metrics["calibration"]
    assert rebuilt.observations == stored["observations"]
    assert rebuilt.coverage == pytest.approx(stored["coverage"])
    assert rebuilt.miscalibration_gap == pytest.approx(stored["miscalibration_gap"])
    for signal, stats in stored["signals"].items():
        assert rebuilt.signals[signal].brier == pytest.approx(stats["brier"])


def test_summarize_decisions_reports_latency_and_failures(dcn_run_dir):
    _, _, lines = read_run(dcn_run_dir)
    summary = summarize_decisions(lines)
    assert summary["coverage"] == pytest.approx(1.0)
    assert summary["mean_latency_s"] > 0
    assert summary["failed_decisions"] == []


def test_run_directory_name_identifies_model_and_scenario(dcn_run_dir):
    assert scenario_slug(dcn_run_dir.name)
    assert dcn_run_dir.name.count("_") >= 3


def scenario_slug(name: str) -> bool:
    return name.split("_")[1] != ""


def test_events_file_is_a_compact_decision_journal(dcn_run_dir):
    rows = [
        json.loads(line)
        for line in (dcn_run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    for row in rows:
        assert row["type"] == "decision"
        assert "weights" in row and "coverage" in row
        assert row["cash"] >= 0


class PartialClient:
    """Answers the noul questions but not the choice ones."""

    provider = "offline"
    model = "partial"
    endpoint = "https://offline.invalid"

    async def decide(self, state, questions):
        answers = {
            question_id: OfflineDcnClient.probability_for(question_id)
            for question_id in questions
            if question_id.startswith("noul_")
        }
        from beatspy.schemas import NoulAnswer

        return DcnResult(
            answers={qid: NoulAnswer(noul=round(value, 4)) for qid, value in answers.items()},
            missing_ids=[qid for qid in questions if not qid.startswith("noul_")],
            input_tokens=500,
        )

    async def close(self):
        return None


def test_partial_coverage_is_recorded_without_inventing_an_answer(tmp_path, scenario, frozen_data):
    run_dir = run_offline(tmp_path, scenario, frozen_data, PartialClient(), question_form="twin")
    _, metrics, lines = read_run(run_dir)
    assert 0.0 < metrics["answer_coverage"] < 1.0
    assert "choice" not in metrics["calibration"]["signals"]
    assert set(metrics["calibration"]["signals"]) == {"noul"}
    for line in lines:
        assert line["missing_ids"]  # the unanswered choice questions are named
        assert line["call"]["missing_ids"] == line["missing_ids"]