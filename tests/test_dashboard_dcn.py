"""Dashboard and catalog tests for the DCN (decision-model) benchmark.

Every test here runs offline: the run directory comes from the ``dcn_run_dir``
fixture, which executes the real runner against the scripted client in
``conftest``. Nothing in this module reaches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from beatspy.artifacts import atomic_json
from beatspy.reporting.catalog import (
    PRIMARY_SIGNAL,
    SAFE_ID,
    build_catalog,
    comparison_group,
    read_json,
    sanitized,
    summary,
    validate_run_meta,
)
from beatspy.reporting.dashboard import collect_runs, render_dashboard, static_dir

INDEX_ROW_KEYS = {
    "run_id",
    "dir",
    "model",
    "model_label",
    "provider",
    "question_form",
    "policy",
    "pricing",
    "scenario",
    "start",
    "end",
    "year",
    "data_cutoff",
    "snapshot_id",
    "created_utc",
    "synthetic",
    "trust",
    "label",
    "group",
    "decisions",
    "invalid_outputs",
    "answer_coverage",
    "mean_latency_s",
    "estimated_cost_usd",
    "brier",
    "brier_skill_score",
    "auc",
    "ece",
    "max_calibration_error",
    "calibration_gap",
    "metrics",
}

CALIBRATION_ROW_KEYS = (
    "brier",
    "brier_skill_score",
    "auc",
    "ece",
    "max_calibration_error",
    "calibration_gap",
    "answer_coverage",
    "mean_latency_s",
    "decisions",
    "invalid_outputs",
)


@pytest.fixture
def results_root(dcn_run_dir: Path) -> Path:
    """The directory that holds run directories, as ``results_dir()`` would."""
    return dcn_run_dir.parent


def test_fixture_run_records_a_complete_dcn_run(dcn_run_dir: Path):
    meta = read_json(dcn_run_dir / "run.json")
    assert meta["schema_version"] == 3
    assert meta["benchmark"] == "dcn"
    assert meta["dcn_protocol"] >= 1
    assert meta["run_id"] == dcn_run_dir.name
    assert meta["synthetic"] is False
    assert set(meta["decision"]) >= {"decision_model", "provider", "question_form", "pricing", "policy"}
    assert meta["decision"]["question_form"] == "twin"
    metrics = read_json(dcn_run_dir / "metrics.json")
    assert metrics["calibration"]["signals"]["noul"]["bins"]
    assert metrics["calibration"]["miscalibration_gap"] is not None


def test_summary_flattens_a_dcn_run(dcn_run_dir: Path):
    meta = read_json(dcn_run_dir / "run.json")
    metrics = read_json(dcn_run_dir / "metrics.json")
    row = summary(meta, metrics, dcn_run_dir.name, "replayed")
    assert set(row) == INDEX_ROW_KEYS | {"noul_brier", "noul_ece", "choice_brier", "choice_ece"}
    assert row["model"] == meta["decision"]["decision_model"]
    assert row["provider"] == meta["decision"]["provider"]
    assert row["question_form"] == "twin"
    assert row["dir"] == dcn_run_dir.name
    assert row["trust"] == "replayed"
    assert row["scenario"] == meta["scenario"]["name"]
    assert row["start"] == meta["requested"]["start"]
    assert row["end"] == meta["requested"]["end"]
    assert row["year"] == row["end"][:4]
    assert row["metrics"] == metrics
    noul = metrics["calibration"]["signals"][PRIMARY_SIGNAL]
    assert row["brier"] == noul["brier"]
    assert row["brier_skill_score"] == noul["brier_skill_score"]
    assert row["auc"] == noul["auc"]
    assert row["ece"] == noul["expected_calibration_error"]
    assert row["calibration_gap"] == metrics["calibration"]["miscalibration_gap"]
    assert row["answer_coverage"] == metrics["answer_coverage"]
    assert row["mean_latency_s"] == metrics["mean_latency_s"]
    assert row["decisions"] == metrics["decisions"]
    assert row["invalid_outputs"] == metrics["invalid_outputs"]
    assert row["pricing"]["cost_per_m_output"] == 0.0


def test_summary_tolerates_a_run_without_calibration(dcn_run_dir: Path):
    meta = read_json(dcn_run_dir / "run.json")
    row = summary(meta, {}, dcn_run_dir.name, "local")
    for key in CALIBRATION_ROW_KEYS:
        assert row.get(key) is None, key
    assert row["model"] == meta["decision"]["decision_model"]


def test_comparison_group_pairs_models_and_separates_question_forms(dcn_run_dir: Path):
    meta = read_json(dcn_run_dir / "run.json")
    other = json.loads(json.dumps(meta))
    other["run_id"] = "20260101-000000_test-scenario_clef_0000000000"
    other["decision"]["decision_model"] = "clef"
    other["decision"]["question_form"] = "noul"
    assert comparison_group(meta) == comparison_group({**other, "decision": {**other["decision"], "question_form": "twin"}})
    assert comparison_group(meta) != comparison_group(other)
    changed_policy = json.loads(json.dumps(meta))
    changed_policy["decision"]["policy"]["top_n"] = 5
    assert comparison_group(meta) != comparison_group(changed_policy)
    # Titles are cosmetic: a renamed scenario is still the same comparison group.
    retitled = json.loads(json.dumps(meta))
    retitled["scenario"]["title"] = "Renamed"
    assert comparison_group(meta) == comparison_group(retitled)


def test_validate_run_meta_rejects_non_dcn_and_renamed_runs(dcn_run_dir: Path, tmp_path: Path):
    assert validate_run_meta(dcn_run_dir)["run_id"] == dcn_run_dir.name
    assert SAFE_ID.fullmatch(dcn_run_dir.name)
    legacy = tmp_path / "20261005-144148_2026-ytd_deepseek"
    legacy.mkdir()
    atomic_json(legacy / "run.json", {"run_id": legacy.name, "schema_version": 2, "benchmark": "agents"})
    atomic_json(legacy / "metrics.json", {})
    with pytest.raises(ValueError, match="dcn"):
        validate_run_meta(legacy)
    meta = read_json(dcn_run_dir / "run.json")
    meta["schema_version"] = 2
    atomic_json(legacy / "run.json", meta)
    with pytest.raises(ValueError, match="schema_version 3"):
        validate_run_meta(legacy)


def test_build_catalog_index_carries_the_calibration_fields(dcn_run_dir: Path, tmp_path: Path):
    data_root = tmp_path / "dashboard"
    (data_root / "runs").mkdir(parents=True)
    import shutil

    shutil.copytree(dcn_run_dir, data_root / "runs" / dcn_run_dir.name)
    runs = build_catalog(data_root)
    assert [row["dir"] for row in runs] == [f"runs/{dcn_run_dir.name}"]
    catalog = read_json(data_root / "index.json")
    assert catalog["schema_version"] == 1
    assert catalog["default_trust"] == "all"
    assert "selected" not in catalog
    row = catalog["runs"][0]
    assert set(row) == INDEX_ROW_KEYS | {"noul_brier", "noul_ece", "choice_brier", "choice_ece"}
    assert row["trust"] == "replayed"
    for key in CALIBRATION_ROW_KEYS:
        assert row[key] is not None, key
    assert 0.0 <= row["brier"] <= 1.0
    assert 0.0 <= row["ece"] <= 1.0
    assert row["calibration_gap"] and row["calibration_gap"] > 0
    assert row["metrics"]["calibration"]["signals"]["choice"]["bins"]


def test_build_catalog_rejects_symlinked_roots(tmp_path: Path):
    data_root = tmp_path / "dashboard"
    data_root.mkdir()
    try:
        (data_root / "runs").symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("this host does not permit creating symlinks")
    with pytest.raises(ValueError, match="symlink"):
        build_catalog(data_root)


def test_sanitized_redacts_secrets_and_drops_credential_keys():
    secrets = {"hunter2-secret-value"}
    payload = {"api_key": "hunter2-secret-value", "Authorization": "Bearer hunter2-secret-value",
               "nested": [{"token": "hunter2-secret-value", "endpoint": "https://example.invalid/v1"}],
               "note": "leaked hunter2-secret-value here", "try": "short"}
    cleaned = sanitized(payload, secrets)
    assert "api_key" not in cleaned and "Authorization" not in cleaned
    # A value that is a known secret is redacted wherever it appears, even under a harmless key.
    assert cleaned["nested"][0]["token"] == "[REDACTED]"
    assert cleaned["nested"][0]["endpoint"] == "https://example.invalid/v1"
    assert cleaned["note"] == "leaked [REDACTED] here"
    assert sanitized({"k": "abc"}, {"abc"}) == {"k": "abc"}


def test_collect_runs_reads_dcn_runs_and_skips_others(results_root: Path):
    rows = collect_runs(results_root)
    assert [row["run_id"] for row in rows] == [directory.name for directory in sorted(results_root.iterdir())]
    row = rows[0]
    assert row["dir"] == row["run_id"]
    assert row["trust"] == "local"
    assert row["model"] != "?"
    assert row["metrics"]["calibration"]["signals"]
    assert collect_runs(results_root / "missing") == []


def test_collect_runs_skips_a_non_dcn_directory(results_root: Path, tmp_path: Path):
    impostor = results_root / "20260101-000000_legacy_run_0000000000"
    impostor.mkdir()
    atomic_json(impostor / "run.json", {"run_id": impostor.name, "schema_version": 2, "benchmark": "agents",
                                        "model": {"model": "gpt-x"}})
    atomic_json(impostor / "metrics.json", {"total_return": 0.1})
    assert all(row["run_id"] != impostor.name for row in collect_runs(results_root))


def test_render_dashboard_writes_the_dcn_payload(results_root: Path, dcn_run_dir: Path, tmp_path: Path):
    out = tmp_path / "site" / "index.html"
    written = render_dashboard(results_root, out_path=out)
    assert written == out
    assert written.is_file()
    for name in ("styles.css", "app.js", "favicon.ico", "site.webmanifest", "robots.txt", "sitemap.xml"):
        assert (out.parent / name).is_file(), name
    assert (out.parent / "assets" / "favicon.svg").is_file()
    catalog = read_json(out.parent / "data" / "index.json")
    assert catalog["schema_version"] == 1
    assert catalog["default_trust"] == "all"
    assert catalog["selected"] is None
    assert len(catalog["runs"]) == 1
    row = catalog["runs"][0]
    assert row["dir"] == f"runs/{dcn_run_dir.name}"
    assert "meta" not in row and "endpoint" not in row
    for key in CALIBRATION_ROW_KEYS:
        assert row[key] is not None, key
    copied = out.parent / "data" / row["dir"]
    assert (copied / "run.json").is_file()
    assert (copied / "decisions.jsonl").is_file()
    assert (copied / "equity_curve.csv").is_file()
    assert read_json(copied / "metrics.json") == row["metrics"]
    # The snapshot travels with the report so the run stays replayable offline.
    assert row["snapshot_dir"] == f"snapshots/{row['snapshot_id']}"
    snapshot = out.parent / "data" / row["snapshot_dir"]
    assert (snapshot / "prices.csv").is_file() and (snapshot / "MANIFEST.json").is_file()
    equity = (out.parent / "data" / row["dir"] / "equity_curve.csv").read_text(encoding="utf-8")
    assert "spy_buy_hold" in equity and "equal_weight" in equity and "sixty_forty" in equity
    assert "momentum_12_1" in equity


def test_render_dashboard_selects_a_run_and_rejects_unknown_ids(results_root: Path, dcn_run_dir: Path, tmp_path: Path):
    out = tmp_path / "site" / "index.html"
    render_dashboard(results_root, run_id=dcn_run_dir.name, out_path=out)
    catalog = read_json(out.parent / "data" / "index.json")
    assert catalog["selected"] == dcn_run_dir.name
    with pytest.raises(ValueError, match="unknown run"):
        render_dashboard(results_root, run_id="not-a-run", out_path=out)


def test_render_dashboard_cannot_overwrite_the_public_site(results_root: Path):
    with pytest.raises(ValueError, match="public dashboard"):
        render_dashboard(results_root, out_path=static_dir() / "index.html")


def test_render_dashboard_requires_at_least_one_run(tmp_path: Path):
    with pytest.raises(RuntimeError, match="No runs found"):
        render_dashboard(tmp_path / "empty")


def test_committed_public_catalog_matches_the_dcn_shape():
    """Whatever is published under dashboard/data must validate against the DCN shape."""
    data_root = static_dir() / "data"
    index = data_root / "index.json"
    if not index.is_file():
        pytest.skip("this checkout publishes no catalog yet")
    catalog = read_json(index)
    assert catalog["schema_version"] == 1
    assert catalog["default_trust"] == "all"
    assert isinstance(catalog["runs"], list)
    assert "selected" not in catalog or catalog["selected"] is None
    for row in catalog["runs"]:
        run_dir = data_root / "runs" / row["run_id"]
        assert row["dir"] == f"runs/{row['run_id']}"
        assert set(row) >= INDEX_ROW_KEYS
        meta = validate_run_meta(run_dir)
        assert row["model"] == meta["decision"]["decision_model"]
        assert row["provider"] == meta["decision"]["provider"]
        assert row["question_form"] == meta["decision"]["question_form"]
        metrics = read_json(run_dir / "metrics.json")
        noul = metrics["calibration"]["signals"][PRIMARY_SIGNAL]
        assert row["brier"] == noul["brier"]
        assert row["ece"] == noul["expected_calibration_error"]
        assert row["calibration_gap"] == metrics["calibration"]["miscalibration_gap"]
        assert row["answer_coverage"] == metrics["answer_coverage"]
        assert row["metrics"] == metrics


def test_public_site_assets_have_no_agent_wording():
    html = (static_dir() / "index.html").read_text(encoding="utf-8")
    js = (static_dir() / "app.js").read_text(encoding="utf-8")
    css = (static_dir() / "styles.css").read_text(encoding="utf-8")
    banned = ("agent", "pipeline", "critic", "reasoning effort", "reasoning_effort", "tool call", "tool_call",
              "telemetry", "five trading", "max turns", "max_turns", "web search", "web_search", "read_file",
              "shell command")
    for name, text in (("index.html", html), ("app.js", js), ("styles.css", css)):
        lowered = text.lower()
        for word in banned:
            assert word not in lowered, f"{name} mentions {word!r}"
    # The DCN payload is what the site renders.
    for token in ("reliability", "brier", "decision model", "calibration", "provider"):
        assert token in html.lower(), token


def test_public_site_renders_data_as_text_and_shows_the_reliability_section():
    html = (static_dir() / "index.html").read_text(encoding="utf-8")
    js = (static_dir() / "app.js").read_text(encoding="utf-8")
    assert not any(bad in js for bad in ("innerHTML", "outerHTML", "document.write", "eval("))
    assert "textContent" in js
    for anchor in ("leaderboard", "calibration", "reliability", "reliability-bins", "gap-callout", "pricing-table",
                   "model-costs", "run-cost", "performance", "drawdown", "allocations", "status", "run-count"):
        assert f'id="{anchor}"' in html, anchor
    assert 'aria-live="polite"' in html
    assert "styles.css" in html and "app.js" in html
    assert "Reliability" in html


def test_pricing_table_covers_the_four_decision_models():
    js = (static_dir() / "app.js").read_text(encoding="utf-8")
    for model, rate, url in (
        ("gpt-6-luna", "0.10", "https://community.openai.com/t/decisions-api-is-now-available-in-public-beta/1403877"),
        ("jev-latest", "0.042", "https://docs.typesafe.ai/models"),
        ("clef", "0.24", "https://developers.cloudflare.com/workers-ai/models/clef/index.md"),
        ("clef-flash", "0.09", "https://developers.cloudflare.com/workers-ai/models/clef/index.md"),
    ):
        assert f'"{model}"' in js, model
        assert rate in js, model
        assert url in js, model