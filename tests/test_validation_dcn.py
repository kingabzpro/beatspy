"""Offline replay verification for DCN runs.

Every case here is derived from one real run of the actual runner over a frozen
synthetic snapshot (the ``dcn_run_dir`` fixture), so the clean case exercises the
same artifact layout a live run produces and the negative cases are honest
tampering rather than hand-built JSON.

No test touches the network: the fixtures build their own snapshot and the
replay path never leaves the run directory.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest

from beatspy.artifacts import sha256
from beatspy.dcn import DCN_PROTOCOL
from beatspy.dcn.verify import (
    check_numbers,
    lines,
    read_json,
    snapshot_directory,
    validate_run,
)
from conftest import OfflineDcnClient, run_offline

# --------------------------------------------------------------------------- #
# Mutation helpers: edit an artifact, then repair run.json's recorded hash so the
# test reaches the check it is actually about.
# --------------------------------------------------------------------------- #


def _read_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _write_meta(run_dir: Path, meta: dict) -> None:
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def edit_meta(run_dir: Path, mutate) -> dict:
    meta = _read_meta(run_dir)
    mutate(meta)
    _write_meta(run_dir, meta)
    return meta


def rehash(run_dir: Path, *names: str) -> None:
    meta = _read_meta(run_dir)
    for name in names:
        meta["artifact_hashes"][name] = sha256(run_dir / name)
    _write_meta(run_dir, meta)


def edit_decisions(run_dir: Path, mutate) -> None:
    path = run_dir / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mutate(rows)
    path.write_text("".join(json.dumps(row, default=str) + "\n" for row in rows), encoding="utf-8")
    rehash(run_dir, "decisions.jsonl")


def _make_link(source: Path, target: Path) -> Path:
    """Create a symlink, or a Windows junction where symlinks need a privilege."""
    try:
        os.symlink(source, target, target_is_directory=True)
        return target
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(target), str(source)], capture_output=True, text=True)
        if result.returncode == 0:
            return target
    pytest.skip("this platform cannot create a symlink or a junction")


@pytest.fixture
def clone(tmp_path: Path, dcn_run_dir: Path):
    """A private, mutable copy of the clean fixture run."""

    def make() -> Path:
        target = tmp_path / "clone" / dcn_run_dir.name
        shutil.copytree(dcn_run_dir, target)
        return target

    return make


# --------------------------------------------------------------------------- #
# The clean case
# --------------------------------------------------------------------------- #


def test_clean_run_replays(dcn_run_dir: Path):
    meta = validate_run(dcn_run_dir)
    assert meta["run_id"] == dcn_run_dir.name
    assert meta["benchmark"] == "dcn"
    assert meta["dcn_protocol"] == DCN_PROTOCOL
    assert meta["synthetic"] is False


def _digests(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256(path) for path in sorted(root.rglob("*")) if path.is_file()}


def test_clean_run_replays_twice(dcn_run_dir: Path):
    """Replay is deterministic and leaves the run directory untouched."""
    before = _digests(dcn_run_dir)
    assert validate_run(dcn_run_dir)["run_id"] == dcn_run_dir.name
    assert validate_run(dcn_run_dir)["run_id"] == dcn_run_dir.name
    assert _digests(dcn_run_dir) == before


def test_replay_needs_no_network(monkeypatch, dcn_run_dir: Path):
    """Outbound connections are denied; the loopback self-pipe asyncio needs is not."""
    import errno

    blocked: list[object] = []

    def guarded(self, address, *args, **kwargs):
        if isinstance(address, tuple) and address and str(address[0]) not in {"127.0.0.1", "::1", "localhost"}:
            blocked.append(address)
            raise OSError(errno.ENETUNREACH, "network access is not allowed during replay")
        return _real_connect(self, address, *args, **kwargs)

    _real_connect = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect", guarded)
    assert validate_run(dcn_run_dir)["benchmark"] == "dcn"
    assert blocked == []


def test_provider_failure_run_still_validates_and_replays_as_a_hold(tmp_path: Path, scenario, frozen_data):
    """A decision whose provider call failed validates end to end, replayed as "hold".

    The recorded line carries an error, ``invalid: true``, no call, and an empty
    weight map. Replay must treat that as "keep the current book": reading the
    empty map as a target would liquidate, which shows up as a trades/equity
    mismatch rather than a clean pass.
    """

    class FlakyClient(OfflineDcnClient):
        async def decide(self, state, questions):
            if state["as_of"] == "2022-01-31":
                raise RuntimeError("provider exploded")
            return await super().decide(state, questions)

    run = run_offline(tmp_path, scenario, frozen_data, FlakyClient())
    rows = [json.loads(line) for line in (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    failed = next(row for row in rows if row["date"] == "2022-01-31")
    assert failed["error"] and failed["validated"]["invalid"] is True
    assert failed["validated"]["weights"] == {}
    assert failed["call"] is None

    assert validate_run(run)["run_id"] == run.name


# --------------------------------------------------------------------------- #
# Result tampering
# --------------------------------------------------------------------------- #


def test_tampered_metrics_is_rejected(clone):
    run = clone()
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    metrics["total_return"] += 0.5
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rehash(run, "metrics.json")

    with pytest.raises(ValueError, match=r"metrics replay mismatch: total_return"):
        validate_run(run)


def test_reweighted_decision_is_rejected(clone):
    """Changing a recorded weight changes the portfolio, so the metrics disagree."""

    def mutate(rows):
        rows[0]["validated"]["weights"] = {"AAPL": 0.5, "MSFT": 0.5}
        rows[0]["validated"]["cash"] = 0.0

    run = clone()
    edit_decisions(run, mutate)

    with pytest.raises(ValueError, match="metrics replay mismatch"):
        validate_run(run)


def test_negative_recorded_weight_is_rejected(clone):
    def mutate(rows):
        rows[0]["validated"]["weights"] = {"AAPL": -0.5}

    run = clone()
    edit_decisions(run, mutate)

    with pytest.raises(ValueError, match="negative recorded weight for 'AAPL'"):
        validate_run(run)


def test_tampered_equity_curve_is_rejected(clone):
    run = clone()
    path = run / "equity_curve.csv"
    rows = path.read_text(encoding="utf-8").splitlines()
    columns = rows[-1].split(",")
    columns[1] = str(float(columns[1]) + 1000.0)
    rows[-1] = ",".join(columns)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    rehash(run, "equity_curve.csv")

    with pytest.raises(ValueError, match="equity curve replay mismatch"):
        validate_run(run)


def test_tampered_trades_are_rejected(clone):
    run = clone()
    path = run / "trades.csv"
    rows = path.read_text(encoding="utf-8").splitlines()
    columns = rows[1].split(",")
    columns[4] = str(float(columns[4]) * 2.0)
    rows[1] = ",".join(columns)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    rehash(run, "trades.csv")

    with pytest.raises(ValueError, match="trades replay mismatch"):
        validate_run(run)


# --------------------------------------------------------------------------- #
# Run identity and provenance
# --------------------------------------------------------------------------- #


def test_synthetic_run_is_rejected(clone):
    run = clone()
    edit_meta(run, lambda meta: meta.__setitem__("synthetic", True))

    with pytest.raises(ValueError, match="synthetic runs cannot be verified"):
        validate_run(run)


def test_non_dcn_run_is_rejected(clone):
    run = clone()
    edit_meta(run, lambda meta: meta.__setitem__("benchmark", "agents"))

    with pytest.raises(ValueError, match="not a DCN run"):
        validate_run(run)


def test_missing_benchmark_is_rejected(clone):
    run = clone()
    edit_meta(run, lambda meta: meta.pop("benchmark"))

    with pytest.raises(ValueError, match="missing its 'benchmark' field"):
        validate_run(run)


def test_wrong_protocol_is_rejected(clone):
    run = clone()
    edit_meta(run, lambda meta: meta.__setitem__("dcn_protocol", DCN_PROTOCOL + 1))

    with pytest.raises(ValueError, match="DCN protocol"):
        validate_run(run)


def test_unexpected_schema_version_is_rejected(clone):
    run = clone()
    edit_meta(run, lambda meta: meta.__setitem__("schema_version", 2))

    with pytest.raises(ValueError, match="unsupported run schema version"):
        validate_run(run)


def test_mismatched_run_id_is_rejected(clone):
    run = clone()
    moved = run.parent / "renamed-run"
    run.rename(moved)

    with pytest.raises(ValueError, match="invalid or mismatched run id"):
        validate_run(moved)


def test_self_assigned_trust_is_rejected(clone):
    run = clone()
    edit_meta(run, lambda meta: meta.__setitem__("trust", "verified"))

    with pytest.raises(ValueError, match="cannot assign its own trust"):
        validate_run(run)


def test_symlinked_run_is_rejected(tmp_path: Path, dcn_run_dir: Path):
    link = _make_link(dcn_run_dir, tmp_path / "linked-run")
    assert link.is_symlink() or os.path.isdir(link)  # junction fallback

    with pytest.raises(ValueError, match="symlinks are not allowed"):
        validate_run(link)


def test_symlinked_run_parent_is_rejected(tmp_path: Path, dcn_run_dir: Path):
    parent = _make_link(dcn_run_dir.parent, tmp_path / "linked-results")
    run = parent / dcn_run_dir.name

    with pytest.raises(ValueError, match="symlinks are not allowed"):
        validate_run(run)


# --------------------------------------------------------------------------- #
# Scenario window
# --------------------------------------------------------------------------- #


def test_future_scenario_is_rejected(clone):
    run = clone()
    edit_meta(run, lambda meta: meta["scenario"].__setitem__("end", "2099-12-31"))

    with pytest.raises(ValueError, match="after the last completed session"):
        validate_run(run)


def test_requested_window_must_match_the_scenario(clone):
    run = clone()
    edit_meta(run, lambda meta: meta["requested"].__setitem__("frequency", "weekly"))

    with pytest.raises(ValueError, match="requested window does not match the scenario"):
        validate_run(run)


# --------------------------------------------------------------------------- #
# Artifact integrity
# --------------------------------------------------------------------------- #


def test_missing_artifact_is_rejected(clone):
    run = clone()
    (run / "trades.csv").unlink()

    with pytest.raises(ValueError, match="missing, oversized, or unsafe artifact: trades.csv"):
        validate_run(run)


def test_artifact_hash_mismatch_is_rejected(clone):
    run = clone()
    path = run / "decisions.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch: decisions.jsonl"):
        validate_run(run)


def test_incomplete_artifact_hash_set_is_rejected(clone):
    run = clone()
    edit_meta(run, lambda meta: meta["artifact_hashes"].pop("trades.csv"))

    with pytest.raises(ValueError, match="artifact hash set"):
        validate_run(run)


def test_unexpected_artifact_hash_key_is_rejected(clone):
    run = clone()
    edit_meta(run, lambda meta: meta["artifact_hashes"].__setitem__("extra.json", "0" * 64))

    with pytest.raises(ValueError, match="artifact hash set"):
        validate_run(run)


def test_tampered_snapshot_prices_are_rejected(clone):
    run = clone()
    path = run / "snapshot" / "prices.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch: snapshot/prices.csv"):
        validate_run(run)


def test_recorded_price_hash_must_match_the_snapshot(clone):
    run = clone()
    edit_meta(run, lambda meta: meta["data"].__setitem__("prices_sha256", "0" * 64))

    with pytest.raises(ValueError, match="data.prices_sha256"):
        validate_run(run)


def test_recorded_snapshot_id_must_match_the_snapshot(clone):
    run = clone()
    edit_meta(run, lambda meta: meta["data"].__setitem__("snapshot_id", "1" * 64))

    with pytest.raises(ValueError, match="data.snapshot_id"):
        validate_run(run)


def test_recorded_row_count_must_match_the_snapshot(clone):
    run = clone()
    edit_meta(run, lambda meta: meta["data"].__setitem__("rows", 1))

    with pytest.raises(ValueError, match="data.rows"):
        validate_run(run)


# --------------------------------------------------------------------------- #
# Schedule
# --------------------------------------------------------------------------- #


def test_decision_dates_must_match_the_schedule(clone):
    run = clone()
    edit_decisions(run, lambda rows: rows.pop())

    with pytest.raises(ValueError, match="recorded decision dates do not match the scenario schedule"):
        validate_run(run)


def test_event_outside_the_schedule_is_rejected(clone):
    run = clone()
    path = run / "events.jsonl"
    path.write_text(
        path.read_text(encoding="utf-8") + json.dumps({"type": "decision", "date": "2022-03-01"}) + "\n",
        encoding="utf-8",
    )
    rehash(run, "events.jsonl")

    with pytest.raises(ValueError, match="outside the decision schedule"):
        validate_run(run)


# --------------------------------------------------------------------------- #
# The shared safety helpers keep their behaviour
# --------------------------------------------------------------------------- #


def test_read_json_rejects_missing_and_oversized_files(tmp_path: Path):
    with pytest.raises(ValueError, match="missing, oversized, or unsafe JSON artifact"):
        read_json(tmp_path / "absent.json")

    huge = tmp_path / "huge.json"
    huge.write_bytes(b" " * 25_000_001)
    with pytest.raises(ValueError, match="oversized"):
        read_json(huge)


def test_read_json_rejects_non_finite_numbers(tmp_path: Path):
    for payload in ('{"value": NaN}', '{"value": Infinity}', '{"value": -Infinity}'):
        path = tmp_path / "number.json"
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON number"):
            read_json(path)


def test_read_json_accepts_a_plain_document(tmp_path: Path):
    path = tmp_path / "ok.json"
    path.write_text('{"value": 1.5}', encoding="utf-8")
    assert read_json(path) == {"value": 1.5}


def test_check_numbers_walks_the_whole_document():
    check_numbers({"a": [1.0, {"b": 2}], "c": None})
    with pytest.raises(ValueError, match="non-finite number"):
        check_numbers({"a": [1.0, {"b": float("inf")}]})


def test_lines_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert lines(path) == [{"a": 1}, {"a": 2}]


def test_snapshot_directory_resolves_the_local_copy(dcn_run_dir: Path):
    meta = _read_meta(dcn_run_dir)
    assert snapshot_directory(dcn_run_dir, meta) == dcn_run_dir / "snapshot"

    with pytest.raises(ValueError, match="invalid snapshot id"):
        snapshot_directory(dcn_run_dir, {"data": {"snapshot_id": "../../etc"}})
    with pytest.raises(ValueError, match="invalid snapshot id"):
        snapshot_directory(dcn_run_dir, {"data": {}})
