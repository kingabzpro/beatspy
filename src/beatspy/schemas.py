"""Pydantic schemas for agent artifacts, decisions, scenarios, and metrics.

Agent outputs are requested as JSON in the prompt and parsed leniently
(`extract_json`), then validated here. Parse failures are a measured benchmark
signal (invalid outputs), never a crash.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# --------------------------------------------------------------------------- #
# Agent artifacts
# --------------------------------------------------------------------------- #


class KeyEvent(BaseModel):
    date: str = ""
    headline: str = ""
    relevance: str = ""


class ResearchReport(BaseModel):
    summary: str = ""
    key_events: list[KeyEvent] = Field(default_factory=list)
    sentiment: dict[str, float] = Field(default_factory=dict)
    risks: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class TickerTrend(BaseModel):
    trend: str = "sideways"
    momentum_30d_pct: float = 0.0
    volatility_regime: str = "normal"
    notes: str = ""


class MarketAnalysis(BaseModel):
    trend: dict[str, TickerTrend] = Field(default_factory=dict)
    opportunities: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class ForecastPoint(BaseModel):
    ticker: str = ""
    direction: str = "flat"
    expected_return_pct: float = 0.0
    p10_pct: float | None = None
    p90_pct: float | None = None
    confidence: float = 0.5


class ForecastReport(BaseModel):
    forecasts: list[ForecastPoint] = Field(default_factory=list)
    method_notes: str = ""


class CriticIssue(BaseModel):
    agent: str = ""
    claim: str = ""
    problem: str = ""
    severity: str = "medium"


class CritiqueReport(BaseModel):
    issues: list[CriticIssue] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    overall_assessment: str = ""
    adjusted_confidence: float = 0.5


class Allocation(BaseModel):
    ticker: str
    weight: float


class PortfolioDecision(BaseModel):
    allocations: list[Allocation] = Field(default_factory=list)
    cash_weight: float | None = None
    expected_direction: str = "flat"
    expected_return_pct: float | None = None
    rationale: str = ""


ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "research": ResearchReport,
    "analyst": MarketAnalysis,
    "forecaster": ForecastReport,
    "critic": CritiqueReport,
}


# --------------------------------------------------------------------------- #
# Lenient parsing
# --------------------------------------------------------------------------- #


def extract_json(text: str, *, legacy=False) -> Any | None:
    """Extract the first JSON object from model output, tolerating prose and fences."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch == "{":
            try:
                return decoder.raw_decode(cleaned, idx)[0]
            except json.JSONDecodeError:
                if not legacy:
                    return None  # Never reinterpret a nested fragment as a complete report.
                continue
    return None


def parse_artifact(role: str, text: str, *, legacy=False) -> tuple[BaseModel | None, str | None]:
    """Parse an agent output into its schema. Returns (model, error)."""
    model_cls = ARTIFACT_MODELS.get(role)
    if model_cls is None:
        return None, f"unknown artifact role: {role}"
    obj = extract_json(text, legacy=legacy)
    if obj is None:
        return None, "no JSON object found in output"
    required = {"research": "summary", "analyst": "trend", "forecaster": "forecasts", "critic": "overall_assessment"}
    if not legacy and (not isinstance(obj, dict) or required[role] not in obj):
        return None, "schema validation failed: missing report fields"
    try:
        return model_cls.model_validate(obj), None
    except ValidationError as exc:
        return None, f"schema validation failed: {exc.errors()[0].get('msg', 'invalid')}"


# --------------------------------------------------------------------------- #
# Decision validation (deterministic, enforced by code not by trust)
# --------------------------------------------------------------------------- #


@dataclass
class ValidatedDecision:
    weights: dict[str, float] = field(default_factory=dict)
    cash: float = 1.0
    violations: list[str] = field(default_factory=list)
    invalid: bool = False
    raw: dict | None = None


def validate_decision(
    obj: Any, tradable: list[str], max_weight: float, *, legacy=False, min_names: int = 1
) -> ValidatedDecision:
    """Clamp a raw portfolio decision to tradable tickers and risk limits.

    Deterministic: unknown tickers are dropped, negative weights dropped,
    per-ticker weight clamped to max_weight, and exposure capped at 1.0 with
    the remainder in cash. Every correction is recorded as a violation.

    ``min_names`` is a diversification floor: a decision funding fewer distinct
    names is measured as a violation rather than rejected, keeping the harness
    rule ("measure, never crash") consistent with every other risk breach.
    """
    if not isinstance(obj, dict):
        return ValidatedDecision(invalid=True, violations=["decision was not a JSON object"])
    if not legacy and "allocations" not in obj:
        return ValidatedDecision(invalid=True, violations=["decision missing allocations"])
    try:
        pm = PortfolioDecision.model_validate(obj)
    except ValidationError:
        return ValidatedDecision(invalid=True, violations=["decision did not match the schema"])

    violations: list[str] = []
    weights: dict[str, float] = {}
    for alloc in pm.allocations:
        ticker = (alloc.ticker or "").upper().strip()
        weight = float(alloc.weight) if math.isfinite(alloc.weight) else 0.0
        if ticker not in tradable:
            violations.append(f"dropped unknown ticker {ticker!r}")
            continue
        if weight <= 0:
            violations.append(f"dropped non-positive weight for {ticker}")
            continue
        weights[ticker] = weights.get(ticker, 0.0) + weight

    cash = pm.cash_weight
    if cash is None:
        cash = max(0.0, 1.0 - sum(weights.values()))
    cash = min(max(float(cash), 0.0), 1.0)

    total = sum(weights.values())
    if total > 1e-9 and total + cash > 1.0 + 1e-9:
        scale = (1.0 - cash) / total
        weights = {t: w * scale for t, w in weights.items()}
        violations.append(f"scaled exposure from {total:.1%} to {1.0 - cash:.1%}")

    for ticker in list(weights):
        if weights[ticker] > max_weight + 1e-9:
            violations.append(f"clamped {ticker} from {weights[ticker]:.1%} to {max_weight:.1%}")
            weights[ticker] = max_weight

    funded = [ticker for ticker, weight in weights.items() if weight > 1e-9]
    if min_names > 1 and len(funded) < min_names:
        violations.append(f"only {len(funded)} funded names; diversification floor is {min_names}")

    final_cash = max(0.0, 1.0 - sum(weights.values()))
    return ValidatedDecision(weights=weights, cash=final_cash, violations=violations, raw=obj)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


class RunMetrics(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    spy_total_return: float
    excess_return_vs_spy: float
    outperformance_hit_rate: float
    directional_accuracy: float
    decisions: int
    avg_turnover: float
    trades: int
    invalid_outputs: int
    risk_violations: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    requests: int
    estimated_cost_usd: float | None = None
    baselines: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #


class EventItem(BaseModel):
    date: str
    headline: str
    detail: str = ""
    source: str = ""


class Scenario(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    title: str
    description: str = ""
    start: str = ""
    end: str = ""
    through_latest: bool = False
    year: int | None = Field(default=None, ge=2000, le=2100)
    window_days: int = Field(default=90, ge=2, le=366)
    frequency: str = "monthly"  # weekly | monthly
    first_only: bool = False  # single decision at the start, then hold
    pipeline: Literal["team", "single"] = "team"
    # Explicit protocol. None keeps the legacy derivation (single -> 8, team -> 5),
    # so archived runs replay unchanged; 9 opts into the configurable family below.
    protocol_version: int | None = Field(default=None, ge=2, le=9)
    # Which reference allocation the portfolio manager receives.
    reference_kind: Literal["momentum", "core", "none"] = "momentum"
    perf_feedback: bool = True
    # Labelled sleeves (e.g. {"stocks": [...], "hedges": [...]}) used for prompt
    # guidance and validation. Every listed ticker must belong to the universe.
    universe_groups: dict[str, list[str]] = Field(default_factory=dict)
    # Minimum distinct funded names a fully invested portfolio must spread across.
    require_min_names: int = Field(default=1, ge=1, le=100)
    # Non-tradable buy-and-hold assets scored alongside the benchmark.
    score_assets: list[str] = Field(default_factory=list)
    max_decisions: int | None = Field(default=None, ge=2, le=10_000, strict=True)
    benchmark: str = "SPY"
    tradable: list[str] = Field(default_factory=list)
    cash: bool = True
    initial_capital: float = 100_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    max_position_weight: float = 0.35
    allow_web_search: bool = False
    allow_finnhub: bool = False
    forecast_provider: str | None = None
    lookback_days: int = 90

    @model_validator(mode="after")
    def check_window(self):
        from datetime import date

        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", self.name):
            raise ValueError("invalid scenario name")
        if self.frequency not in ("weekly", "monthly"):
            raise ValueError("frequency must be weekly or monthly")
        if self.through_latest:
            if not self.start or self.end or self.year is not None:
                raise ValueError("through_latest requires start only")
            date.fromisoformat(self.start)
        elif bool(self.start) != bool(self.end):
            raise ValueError("provide both start and end")
        if not self.through_latest and (not self.start or not self.end):
            if self.year is None:
                raise ValueError("scenario needs start/end or year")
        elif self.end and date.fromisoformat(self.start) > date.fromisoformat(self.end):
            raise ValueError("start must not follow end")
        if not self.tradable or len(self.universe) != len(set(self.universe)):
            raise ValueError("scenario needs a ticker universe")
        if self.initial_capital <= 0 or self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("invalid capital or costs")
        if not all(math.isfinite(v) for v in (self.initial_capital, self.fee_bps, self.slippage_bps)):
            raise ValueError("capital and costs must be finite")
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight must be in (0, 1]")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", self.benchmark):
            raise ValueError("invalid benchmark ticker")
        known = set(self.universe)
        for group, tickers in self.universe_groups.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", group):
                raise ValueError(f"invalid universe group name {group!r}")
            unknown = sorted({t.upper() for t in tickers} - known)
            if unknown:
                raise ValueError(f"universe group {group!r} lists tickers outside the universe: {unknown}")
        unknown_assets = sorted({t.upper() for t in self.score_assets} - known)
        if unknown_assets:
            raise ValueError(f"score_assets lists tickers outside the universe: {unknown_assets}")
        if "stocks" in self.universe_groups and len(self.universe_groups["stocks"]) < self.require_min_names:
            raise ValueError("require_min_names exceeds the number of listed stocks")
        if self.require_min_names > len(self.universe):
            raise ValueError("require_min_names exceeds the tradable universe")
        return self

    @property
    def universe(self) -> list[str]:
        tickers = list(dict.fromkeys([t.upper() for t in self.tradable]))
        benchmark = self.benchmark.upper()
        if benchmark not in tickers:
            tickers.append(benchmark)
        return tickers

    @property
    def total_fee_rate(self) -> float:
        return (self.fee_bps + self.slippage_bps) / 10_000.0
