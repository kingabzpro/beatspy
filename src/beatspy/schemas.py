"""Pydantic schemas for DCN questions, answers, decisions, scenarios, and metrics.

A decision model (DCN) does not generate text. It reads a `state` plus a map of
typed `questions` and returns one `answer` per question: a probability for a
noul, a distribution over options for a choice, a weighted level for a score.

The three primitives and their response fields are shared by every provider this
benchmark covers (OpenAI Decisions, TypeSafe Jev, Cloudflare Clef), so one
schema set serves all of them. A malformed answer is a measured benchmark
signal, never a crash.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# --------------------------------------------------------------------------- #
# Typed questions (the request side of the System One contract)
# --------------------------------------------------------------------------- #

QuestionType = Literal["noul", "choice", "score"]


class Question(BaseModel):
    """One typed question. `criteria` is a map for choice, a list for score."""

    type: QuestionType
    instructions: str | dict | list
    criteria: dict | list | None = None

    @model_validator(mode="after")
    def check_criteria(self):
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not self.criteria:
                raise ValueError("a choice question needs a non-empty criteria map of option -> description")
            if len(self.criteria) > 255:
                raise ValueError("a choice question accepts at most 255 options")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
                raise ValueError("a score question needs 2-10 ordered levels")
        elif self.criteria is not None:
            raise ValueError("a noul question takes no criteria")
        return self


# --------------------------------------------------------------------------- #
# Typed answers (the response side)
# --------------------------------------------------------------------------- #


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float = Field(ge=0.0, le=1.0)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float | None = None

    @model_validator(mode="after")
    def check_distribution(self):
        if not self.probabilities:
            raise ValueError("choice answer needs probabilities")
        if self.choice not in self.probabilities:
            raise ValueError("the chosen option is missing from probabilities")
        total = sum(self.probabilities.values())
        if not 0.98 <= total <= 1.02:
            raise ValueError(f"choice probabilities must sum to 1 (got {total:.4f})")
        return self


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str] = Field(default_factory=dict)
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer


class AnswerSet(BaseModel):
    """Answers returned for one request, keyed by the question ids we sent."""

    answers: dict[str, Answer] = Field(default_factory=dict)

    def noul(self, question_id: str) -> float | None:
        answer = self.answers.get(question_id)
        return answer.noul if isinstance(answer, NoulAnswer) else None


class CallRecord(BaseModel):
    """One DCN request/response, kept whole so a run replays without the network."""

    provider: str
    model: str
    endpoint: str
    state_sha256: str
    state_chars: int
    question_ids: list[str]
    answers: dict[str, Answer] = Field(default_factory=dict)
    missing_ids: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    latency_s: float = 0.0
    attempts: int = 1
    error: str | None = None
    raw: dict | None = None

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "state_sha256": self.state_sha256,
            "state_chars": self.state_chars,
            "question_ids": self.question_ids,
            "answers": {key: value.model_dump() for key, value in self.answers.items()},
            "missing_ids": self.missing_ids,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "requests": self.requests,
            "latency_s": round(self.latency_s, 3),
            "attempts": self.attempts,
            "error": self.error,
        }


def parse_answers(payload: dict | None, question_ids: list[str]) -> tuple[dict[str, Answer], list[str]]:
    """Validate a provider's `answers` map. Returns (answers, missing ids).

    An answer that fails validation is treated as missing rather than fatal: the
    benchmark measures coverage, so one malformed answer must not discard the
    rest of the response.
    """
    raw = (payload or {}).get("answers")
    if not isinstance(raw, dict):
        return {}, list(question_ids)
    models: dict[str, type[BaseModel]] = {"noul": NoulAnswer, "choice": ChoiceAnswer, "score": ScoreAnswer}
    answers: dict[str, Answer] = {}
    for question_id in question_ids:
        candidate = raw.get(question_id)
        if not isinstance(candidate, dict):
            continue
        model_cls = models.get(str(candidate.get("type", "")).strip().lower())
        if model_cls is None:
            continue
        try:
            answers[question_id] = model_cls.model_validate(candidate)
        except ValidationError:
            continue
    missing = [question_id for question_id in question_ids if question_id not in answers]
    return answers, missing


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


class ReliabilityBin(BaseModel):
    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float | None = None


class SignalStats(BaseModel):
    """Calibration and discrimination for one question form (noul/choice/score)."""

    signal: str
    n: int
    base_rate: float | None = None
    brier: float | None = None
    brier_skill_score: float | None = None
    log_loss: float | None = None
    auc: float | None = None
    accuracy: float | None = None
    mean_predicted: float | None = None
    expected_calibration_error: float | None = None
    max_calibration_error: float | None = None
    bins: list[ReliabilityBin] = Field(default_factory=list)


class CalibrationReport(BaseModel):
    """Everything the calibration side of the benchmark reports."""

    horizons: int = 0
    observations: int = 0
    coverage: float = 0.0
    signals: dict[str, SignalStats] = Field(default_factory=dict)
    # Paired comparison of the noul and choice forms: the claim this benchmark tests.
    miscalibration_gap: float | None = None
    note: str = ""


# --------------------------------------------------------------------------- #
# Decision validation (deterministic, enforced by code not by trust)
# --------------------------------------------------------------------------- #


@dataclass
class ValidatedDecision:
    weights: dict[str, float] = field(default_factory=dict)
    cash: float = 1.0
    violations: list[str] = field(default_factory=list)
    invalid: bool = False


def validate_weights(
    weights: dict[str, float],
    cash: float | None,
    tradable: list[str],
    max_weight: float,
) -> ValidatedDecision:
    """Clamp a weight map to tradable tickers and risk limits.

    Deterministic: unknown or non-positive tickers are dropped, each weight is
    clamped to max_weight, and exposure is capped at 1.0 with the remainder in
    cash. Every correction is recorded as a violation rather than silently
    applied.
    """
    violations: list[str] = []
    cleaned: dict[str, float] = {}
    for ticker, weight in weights.items():
        name = str(ticker).upper().strip()
        value = float(weight) if isinstance(weight, (int, float)) and math.isfinite(weight) else 0.0
        if name not in tradable:
            violations.append(f"dropped unknown ticker {name!r}")
            continue
        if value <= 0:
            violations.append(f"dropped non-positive weight for {name}")
            continue
        cleaned[name] = cleaned.get(name, 0.0) + value

    if cash is None:
        cash = max(0.0, 1.0 - sum(cleaned.values()))
    cash = min(max(float(cash), 0.0), 1.0)

    total = sum(cleaned.values())
    if total > 1e-9 and total + cash > 1.0 + 1e-9:
        scale = (1.0 - cash) / total
        cleaned = {ticker: value * scale for ticker, value in cleaned.items()}
        violations.append(f"scaled exposure from {total:.1%} to {1.0 - cash:.1%}")

    for ticker in list(cleaned):
        if cleaned[ticker] > max_weight + 1e-9:
            violations.append(f"clamped {ticker} from {cleaned[ticker]:.1%} to {max_weight:.1%}")
            cleaned[ticker] = max_weight

    return ValidatedDecision(
        weights=cleaned,
        cash=max(0.0, 1.0 - sum(cleaned.values())),
        violations=violations,
    )


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
    decisions: int
    avg_turnover: float
    trades: int
    invalid_outputs: int
    risk_violations: int
    input_tokens: int
    output_tokens: int
    requests: int
    # DCN-specific: the decision-model side of the benchmark.
    calibration: CalibrationReport | None = None
    answer_coverage: float = 0.0
    mean_latency_s: float = 0.0
    estimated_cost_usd: float | None = None
    baselines: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #


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
    # A rolling window that always ends at the latest completed session: `start`
    # is computed as this many months back. This is what keeps a "latest 6 months"
    # scenario from silently aging into an old period.
    recent_months: int | None = Field(default=None, ge=1, le=60)
    frequency: str = "monthly"  # weekly | monthly
    max_decisions: int | None = Field(default=None, ge=2, le=10_000, strict=True)
    benchmark: str = "SPY"
    tradable: list[str] = Field(default_factory=list)
    cash: bool = True
    initial_capital: float = 100_000.0
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    max_position_weight: float = 0.35
    lookback_days: int = 90
    # How many candidates a single decision asks about. The whole universe stays
    # in the state for context; this bounds the number of typed questions per
    # request so cost and provider question limits stay predictable.
    max_candidates: int = Field(default=10, ge=2, le=64)

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
        elif self.recent_months is not None:
            # A rolling window resolves both dates, so only a fixed window may
            # declare them. Reject a half-specified window either way.
            if bool(self.start) != bool(self.end):
                raise ValueError("provide both start and end, or neither for a rolling window")
        elif bool(self.start) != bool(self.end):
            raise ValueError("provide both start and end")
        if not self.through_latest and not self.start and not self.end:
            # A rolling window resolves its own dates; otherwise a scenario needs
            # an explicit window or a year to anchor one.
            if self.year is None and self.recent_months is None:
                raise ValueError("scenario needs start/end, a year, or recent_months")
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