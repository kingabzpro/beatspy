"""Shared test fixtures. All tests are hermetic: no network, no real models.

The DCN benchmark never needs a live provider in tests. `OfflineDcnClient` stands
in for one: it answers every typed question deterministically, so a whole run can
be executed, scored, replayed, and rendered offline.

`frozen_data` writes a real content-addressed snapshot (valid `prices_sha256`,
row count and coverage) so a run can copy it into its artifact directory exactly
as a live run does — without touching the network.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from beatspy.artifacts import sha256
from beatspy.config import Settings
from beatspy.data.freeze import PRICE_COLUMNS, snapshot_id
from beatspy.data.service import DataService
from beatspy.dcn import DCN_PROTOCOL
from beatspy.dcn.clients import DcnResult
from beatspy.schemas import ChoiceAnswer, NoulAnswer, Question, Scenario, ScoreAnswer

TICKERS = ["AAPL", "MSFT", "TLT"]


def make_scenario(**overrides) -> Scenario:
    """A small test scenario; override any field by keyword."""
    payload = {
        "name": "test-scenario",
        "title": "Test Scenario",
        "description": "synthetic",
        "start": "2022-01-03",
        "end": "2022-03-31",
        "frequency": "monthly",
        "benchmark": "SPY",
        "tradable": list(TICKERS),
        "initial_capital": 100_000.0,
    }
    payload.update(overrides)
    return Scenario(**payload)


def make_prices(
    tickers: list[str] | None = None,
    start: str = "2021-09-01",
    end: str = "2022-07-05",
    seed: int = 7,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """Deterministic synthetic OHLCV covering the scenario window plus lookback."""
    tradable = [t for t in (tickers or TICKERS) if t != benchmark]
    names = [benchmark, *tradable]
    rng = np.random.default_rng(seed)
    calendar = xcals.get_calendar("XNYS", start=start, end=end)
    sessions = calendar.sessions
    sessions = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]
    rows = []
    for ticker in names:
        n = len(sessions)
        drift = -0.002 if ticker == benchmark else float(rng.normal(0.0, 0.001))
        close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
        open_ = close * (1.0 + rng.normal(0.0, 0.003, n))
        for i, day in enumerate(sessions):
            rows.append(
                (
                    day.date().isoformat(),
                    ticker,
                    float(open_[i]),
                    float(close[i]) * 1.01,
                    float(close[i]) * 0.99,
                    float(close[i]),
                    1_000_000.0,
                )
            )
    return pd.DataFrame(rows, columns=PRICE_COLUMNS).assign(date=lambda df: pd.to_datetime(df["date"]))


@pytest.fixture
def scenario() -> Scenario:
    """A dated test scenario. Bundled scenarios can carry `through_latest` and no
    explicit end, so resolve the dates exactly as a real run does."""
    from beatspy.data.calendar import resolve_scenario

    return resolve_scenario(make_scenario())


@pytest.fixture
def data_service(scenario) -> DataService:
    return DataService(make_prices(scenario.universe), benchmark=scenario.benchmark)


@pytest.fixture
def settings() -> Settings:
    return Settings()


class OfflineDcnClient:
    """A deterministic decision model. Answers every question form, no network.

    The probability comes from the question id, so a fixture run is reproducible
    byte-for-byte while still varying across tickers. The choice answer it returns
    is deliberately sharper than its noul answer, mirroring the reported
    overconfidence of the choice form that this benchmark exists to measure.
    """

    provider = "offline"
    model = "offline-scripted"
    endpoint = "https://offline.invalid/v1/systemone"

    def __init__(self, *, fixed: float | None = None, fail_ids: set[str] | None = None):
        self.fixed = fixed
        self.fail_ids = fail_ids or set()
        self.calls: list[str] = []
        self.states: list[dict] = []

    @staticmethod
    def probability_for(question_id: str) -> float:
        digest = hashlib.sha256(question_id.encode("utf-8")).digest()
        return 0.05 + 0.90 * (digest[0] / 255.0)

    async def decide(self, state: dict, questions: dict[str, Question]) -> DcnResult:
        self.calls.append(state["as_of"])
        self.states.append(state)
        answers: dict[str, object] = {}
        for question_id, question in questions.items():
            if question_id in self.fail_ids:
                continue
            probability = self.fixed if self.fixed is not None else self.probability_for(question_id)
            if question.type == "noul":
                answers[question_id] = NoulAnswer(noul=round(probability, 4))
            elif question.type == "choice":
                sharp = min(0.99, max(0.01, (probability - 0.5) * 1.8 + 0.5))
                answers[question_id] = ChoiceAnswer(
                    choice="outperform" if sharp >= 0.5 else "underperform",
                    probabilities={"outperform": round(sharp, 4), "underperform": round(1 - sharp, 4)},
                    confidence=0.8,
                )
            else:
                levels = len(question.criteria or [])
                answers[question_id] = ScoreAnswer(
                    score=round(min(probability, 0.999) * max(levels - 1, 1), 4),
                    legend={str(i): str(level) for i, level in enumerate(question.criteria or [])},
                    probabilities={},
                )
        return DcnResult(answers=answers, input_tokens=500, output_tokens=0, latency_s=0.01, attempts=1)

    async def close(self) -> None:
        return None


@pytest.fixture
def offline_client() -> OfflineDcnClient:
    return OfflineDcnClient()


@pytest.fixture
def frozen_data(tmp_path: Path, scenario, data_service):
    """Write a real, valid frozen snapshot and return (DataService, manifest)."""
    prices = make_prices(scenario.universe)
    directory = tmp_path / "snapshots" / "fixture"
    directory.mkdir(parents=True, exist_ok=True)
    prices.to_csv(directory / "prices.csv", index=False, date_format="%Y-%m-%d")
    manifest = {
        "schema_version": 2,
        "scenario": scenario.name,
        "tickers": scenario.universe,
        "start": prices.date.min().date().isoformat(),
        "end": prices.date.max().date().isoformat(),
        "requested_start": scenario.start,
        "requested_end": scenario.end,
        "lookback_days": scenario.lookback_days,
        "rows": len(prices),
        "source": "synthetic fixture",
        "prices_sha256": sha256(directory / "prices.csv"),
        "fetched_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest["snapshot_id"] = snapshot_id(manifest)
    (directory / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return data_service, {**manifest, "path": str(directory.resolve())}


def run_offline(
    tmp_path: Path,
    scenario,
    frozen,
    client,
    *,
    question_form: str = "per_asset",
    settings: Settings | None = None,
    label: str | None = None,
    scenario_overrides: dict | None = None,
):
    """Execute a complete DCN run offline through the real runner."""
    import asyncio

    from beatspy.dcn.runner import run_dcn

    # Overrides are applied to the raw scenario and resolved inside run_dcn, so a
    # test that narrows the window really narrows it.
    active = scenario.model_copy(update=scenario_overrides or {})
    return asyncio.run(
        run_dcn(
            active,
            settings or Settings(),
            out_root=tmp_path / "results",
            prepared_data=frozen,
            client=client,
            question_form=question_form,
            label=label,
        )
    )


@pytest.fixture
def dcn_run_dir(tmp_path: Path, scenario, frozen_data, offline_client) -> Path:
    """A complete run in the `twin` form, which scores both noul and choice.

    Kept on twin because report/catalog tests assert that a two-signal run
    renders correctly. Use `per_asset_run_dir` for the default form.
    """
    return run_offline(tmp_path, scenario, frozen_data, offline_client, question_form="twin")


@pytest.fixture
def per_asset_run_dir(tmp_path: Path, scenario, frozen_data, offline_client) -> Path:
    """A complete run in the default `per_asset` form: one question per asset."""
    return run_offline(tmp_path, scenario, frozen_data, offline_client, question_form="per_asset")


__all__ = [
    "DCN_PROTOCOL",
    "OfflineDcnClient",
    "TICKERS",
    "date",
    "make_prices",
    "make_scenario",
    "run_offline",
]