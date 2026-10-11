"""One decision: build the state, ask the typed questions, size the portfolio.

Two design choices keep this benchmark honest:

- **The sizing policy is fixed and deterministic.** Only noul probabilities enter
  the portfolio, and the rule is recorded in `run.json` before any result is
  seen. The default (`graded`) holds the whole universe: half the book is an even
  spread and half follows the probability above the neutral 0.5. The legacy
  `top_n` cut, which held only the strongest names above `min_probability`, is
  kept so runs recorded under it stay reproducible. Nothing here is tuned after
  seeing a result.
- **Missing answers cannot become trades.** An unanswered ticker is backfilled at
  the neutral 0.5, so a provider failure is recorded as missing coverage rather
  than expressed as a directional bet.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date

from ..schemas import (
    Answer,
    CallRecord,
    ChoiceAnswer,
    NoulAnswer,
    Question,
    ScoreAnswer,
    ValidatedDecision,
    validate_weights,
)
from . import calibration as cal
from .clients import DcnClient, build_call_record
from .questions import QuestionSet, build_questions
from .state import build_state, enforce_budget, render_state

NEUTRAL = 0.5


@dataclass
class DcnDecision:
    """Everything one decision produced, in a form that replays without a network."""

    date: date
    question_form: str
    state: dict
    state_text: str
    questions: dict[str, Question] = field(default_factory=dict)
    question_ids: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    # `probabilities` always covers every ticker and is what sizes the portfolio;
    # unanswered tickers sit at the neutral 0.5 there. The `answered_*` maps hold
    # ONLY what the model actually returned, because calibration must never score
    # a number this benchmark invented.
    probabilities: dict[str, float] = field(default_factory=dict)
    choice_probabilities: dict[str, float] = field(default_factory=dict)
    rank_probabilities: dict[str, float] = field(default_factory=dict)
    answered_noul: dict[str, float] = field(default_factory=dict)
    answered_choice: dict[str, float] = field(default_factory=dict)
    answered_rank: dict[str, float] = field(default_factory=dict)
    answers: dict[str, Answer] = field(default_factory=dict)
    missing_ids: list[str] = field(default_factory=list)
    backfilled: list[str] = field(default_factory=list)
    call: CallRecord | None = None
    error: str | None = None
    validated: ValidatedDecision = field(default_factory=ValidatedDecision)
    observed: list[cal.Observation] = field(default_factory=list)
    horizon_trading_days: int = 0
    # What the token-budget enforcement dropped from the state, if anything.
    state_trimmed: list[str] = field(default_factory=list)

    @property
    def state_sha256(self) -> str:
        return hashlib.sha256(self.state_text.encode("utf-8")).hexdigest()

    @property
    def coverage(self) -> float:
        """Share of requested questions the provider actually answered."""
        if not self.question_ids:
            return 1.0
        return (len(self.question_ids) - len(self.missing_ids)) / len(self.question_ids)


def executable_tickers(scenario, policy) -> list[str]:
    """The tickers the accounting engine may hold under one policy.

    Every policy may hold the tradable universe. `core_satellite` additionally
    holds the benchmark itself as the index core, which is what makes the
    portfolio track the benchmark instead of carrying the drift of an
    equal-weighted book.
    """
    tickers = [ticker.upper() for ticker in scenario.tradable]
    benchmark = scenario.benchmark.upper()
    if policy.selection == "core_satellite" and benchmark not in tickers:
        tickers.append(benchmark)
    return tickers


def _cross_sectional_tilts(
    probabilities: dict[str, float],
    answered: dict[str, float] | None,
    universe: list[str],
) -> tuple[dict[str, float], float]:
    """Tilts about the decision's own mean, and that mean.

    Only values the provider actually returned are considered, so a backfilled
    neutral cannot win the conviction share. Invariant to any positive affine
    transform of the provider's scale: the center moves with the numbers.
    """
    source = probabilities if answered is None else answered
    values = {
        ticker: float(value)
        for ticker, value in source.items()
        if ticker in universe and value is not None
    }
    center = sum(values.values()) / len(values) if values else 0.0
    tilts = {ticker: max(0.0, value - center) for ticker, value in values.items()}
    return tilts, center


def derive_weights(
    probabilities: dict[str, float],
    policy,
    scenario,
    *,
    benchmark: str | None = None,
    answered: dict[str, float] | None = None,
) -> ValidatedDecision:
    """Turn probabilities into a portfolio.

    Four fixed policies, selected by `policy.selection`:

    - **core_satellite** (default): `policy.core_weight` of the book holds the
      benchmark itself, and the remaining sleeve is allocated across the
      universe by the cross-sectional rule below. The portfolio therefore tracks
      the index by construction, and the excess over SPY measures the model's
      active sleeve rather than the cap-weighting drift of an equal-weighted
      book. The core is exempt from the per-name position limit.
    - **graded_cs**: the cross-sectional graded book. Half the book is an even
      spread over the universe; the other half follows each probability's
      distance above the decision's own mean. Invariant to a provider's
      probability scale.
    - **graded**: the original, conviction measured above the fixed 0.5 prior.
      Kept so runs recorded under it stay reproducible.
    - **top_n**: the legacy cut, kept for the same reason.

    Under every graded policy nothing is chosen and nothing is dropped, so a
    weak forecast is a mild tilt rather than a binary switch. If no asset has any
    tilt, the book holds the even spread rather than going to cash, so the
    portfolio is always comparable. Names the provider did not answer carry no
    tilt: an unanswered question must never become a directional bet.
    """
    benchmark_ticker = (benchmark or scenario.benchmark).upper()
    tradable = [ticker for ticker in scenario.tradable if ticker.upper() != benchmark_ticker]

    if policy.selection == "core_satellite":
        core = min(max(float(policy.core_weight), 0.0), 1.0)
        sleeve = 1.0 - core
        weights: dict[str, float] = {}
        if core > 0:
            weights[benchmark_ticker] = core
        if sleeve > 0 and tradable:
            tilts, _ = _cross_sectional_tilts(probabilities, answered, tradable)
            total = sum(tilts.values())
            uniform = sleeve / len(tradable)
            if total <= 1e-12:
                for ticker in tradable:
                    weights[ticker] = weights.get(ticker, 0.0) + uniform
            else:
                base = sleeve * (1.0 - policy.base_weight)
                for ticker in tradable:
                    weights[ticker] = weights.get(ticker, 0.0) + base / len(tradable)
                for ticker, tilt in tilts.items():
                    weights[ticker] = weights.get(ticker, 0.0) + sleeve * policy.base_weight * (tilt / total)
        if not weights:
            return ValidatedDecision(weights={}, cash=1.0)
        return validate_weights(
            weights,
            None,
            executable_tickers(scenario, policy),
            scenario.max_position_weight,
            exempt={benchmark_ticker},
        )

    if not tradable:
        return ValidatedDecision(weights={}, cash=1.0)

    if policy.selection == "top_n":
        eligible = sorted(
            (
                (ticker, probability)
                for ticker, probability in probabilities.items()
                if probability > policy.min_probability
            ),
            key=lambda item: (-item[1], item[0]),
        )[: policy.top_n]
        if len(eligible) < policy.min_names:
            return ValidatedDecision(weights={}, cash=1.0)
        weight = 1.0 / len(eligible)
        return validate_weights(
            {ticker: weight for ticker, _ in eligible},
            None,
            scenario.tradable,
            scenario.max_position_weight,
        )

    uniform = 1.0 / len(tradable)
    base = 1.0 - policy.base_weight

    if policy.selection == "graded_cs":
        tilts, _ = _cross_sectional_tilts(probabilities, answered, tradable)
        total = sum(tilts.values())
        if total <= 1e-12:
            # Nothing to distinguish, or a degenerate cross-section: the even
            # spread is the whole book.
            weights = {ticker: uniform for ticker in tradable}
        else:
            weights = {ticker: base * uniform for ticker in tradable}
            for ticker, tilt in tilts.items():
                weights[ticker] += policy.base_weight * (tilt / total)
        return validate_weights(weights, None, scenario.tradable, scenario.max_position_weight)

    # Graded. Half the book is a uniform spread across every asset; the other half
    # is allocated in proportion to the tilt above the prior. Blending matters: a
    # pure tilt allocation gives a single confident asset 100% of the book, which
    # is the concentration this rewrite exists to remove.
    tilts = {
        ticker: max(0.0, float(probabilities.get(ticker, policy.min_probability)) - policy.min_probability)
        for ticker in tradable
    }
    total = sum(tilts.values())
    if total <= 1e-12:
        # No view anywhere: the even spread is the whole book, so the run still
        # participates rather than sitting in cash.
        weights = {ticker: uniform for ticker in tradable}
    else:
        weights = {ticker: base * uniform + policy.base_weight * (tilt / total) for ticker, tilt in tilts.items()}
    return validate_weights(weights, None, scenario.tradable, scenario.max_position_weight)


def label_observations(data, scenario, decision: DcnDecision) -> list[cal.Observation]:
    """Score one decision's probabilities against what actually happened.

    The realized label is a *relative* outcome (did the ticker beat the
    benchmark over the same horizon), so a rising tide cannot make every
    prediction correct. Both legs are measured close-to-close strictly after the
    decision date, on frozen prices only.
    """
    horizon = decision.horizon_trading_days
    if horizon <= 0:
        return []
    benchmark = scenario.benchmark.upper()
    before = data.close_on(benchmark, decision.date)
    after = data.close_on_offset(benchmark, decision.date, horizon)
    observations: list[cal.Observation] = []
    for ticker in decision.tickers:
        name = ticker.upper()
        # Only what the model actually returned is scored. A ticker that was
        # asked about but never answered, or one that was never asked about,
        # contributes no observation at all rather than a fabricated neutral.
        noul = decision.answered_noul.get(name)
        choice = decision.answered_choice.get(name)
        rank = decision.answered_rank.get(name)
        if noul is None and choice is None and rank is None:
            continue
        own_before = data.close_on(name, decision.date)
        own_after = data.close_on_offset(name, decision.date, horizon)
        if None in (before, after, own_before, own_after) or not before or not own_before:
            continue
        excess = (own_after / own_before - 1.0) - (after / before - 1.0)
        observations.append(
            cal.Observation(
                ticker=name,
                noul=noul,
                choice=choice,
                rank=rank,
                outcome=int(excess > 0),
                excess_return=round(excess, 6),
            )
        )
    return observations


class DcnPipeline:
    """Asks one provider for one decision's probabilities, then sizes the book."""

    def __init__(self, settings, scenario, client: DcnClient, *, question_form: str = "per_asset"):
        from ..data.calendar import resolve_scenario

        self.settings = settings
        # A bundled scenario may still be a rolling window (`through_latest`) with
        # no explicit end; resolving here means every entry point gets real dates.
        self.scenario = resolve_scenario(scenario)
        self.client = client
        self.question_form = question_form

    async def decide(
        self,
        data,
        as_of: date,
        *,
        holdings: dict[str, float],
        cash: float,
        horizon_trading_days: int,
        next_decision_date: str | None,
        recent: list[dict] | None = None,
    ) -> DcnDecision:
        tickers = self.candidates(data, as_of, holdings)
        question_set = build_questions(
            self.question_form, tickers, self.scenario.benchmark, horizon_trading_days
        )
        state = build_state(
            data,
            self.scenario,
            as_of,
            holdings=holdings,
            cash=cash,
            question_tickers=tickers,
            horizon_trading_days=horizon_trading_days,
            next_decision_date=next_decision_date,
            recent=recent,
        )
        # Workers AI truncates past roughly 2K tokens, so the budget is enforced
        # here, before the state is sent, rather than trusted to hold by itself.
        state, state_trimmed = enforce_budget(state, self.settings.decision.max_state_tokens)
        state_text = render_state(state)
        decision = DcnDecision(
            date=as_of,
            question_form=self.question_form,
            state=state,
            state_text=state_text,
            questions=question_set.questions,
            question_ids=question_set.ids,
            tickers=list(tickers),
            horizon_trading_days=horizon_trading_days,
            state_trimmed=state_trimmed,
        )
        try:
            result = await self.client.decide(state, question_set.questions)
        except Exception as exc:  # provider failure is a measured signal, not a crash
            decision.error = f"{type(exc).__name__}: {exc}"
            decision.missing_ids = list(question_set.ids)
            decision.validated = ValidatedDecision(weights={}, cash=1.0, invalid=True)
            return decision

        decision.answers = result.answers
        decision.missing_ids = list(result.missing_ids)
        decision.call = build_call_record(result, state_text, question_set.ids, self.client)
        self._extract(decision, question_set)
        decision.validated = derive_weights(
            decision.probabilities,
            self.settings.decision,
            self.scenario,
            answered=decision.answered_noul,
        )
        return decision

    def candidates(self, data, as_of: date, holdings: dict[str, float]) -> list[str]:
        """Which assets this decision asks about.

        `per_asset` asks about **every investable asset**, one question each, so a
        single asset's answer stands on its own and the portfolio covers the whole
        universe. Every other form keeps the bounded momentum-ranked candidate set.
        """
        from .state import candidates

        if self.question_form == "per_asset":
            return [ticker.upper() for ticker in self.scenario.tradable]
        return candidates(
            data,
            self.scenario,
            as_of,
            extra=list(holdings),
            max_candidates=self.scenario.max_candidates,
        )

    def _extract(self, decision: DcnDecision, question_set: QuestionSet) -> None:
        """Read probabilities out of the answers, backfilling misses at neutral.

        A missing answer becomes the neutral 0.5 rather than a direction, so a
        provider failure is recorded as missing coverage and cannot turn into a
        trade.
        """
        for question_id, ticker in question_set.noul_tickers.items():
            answer = decision.answers.get(question_id)
            if isinstance(answer, NoulAnswer):
                decision.probabilities[ticker] = float(answer.noul)
                decision.answered_noul[ticker] = float(answer.noul)
            else:
                decision.probabilities[ticker] = NEUTRAL
                decision.backfilled.append(ticker)
        for question_id, ticker in question_set.choice_tickers.items():
            answer = decision.answers.get(question_id)
            probability = cal.choice_probability(answer.probabilities) if isinstance(answer, ChoiceAnswer) else None
            if probability is None:
                decision.choice_probabilities[ticker] = NEUTRAL
                decision.backfilled.append(ticker)
            else:
                decision.choice_probabilities[ticker] = probability
                decision.answered_choice[ticker] = probability
        for question_id, ticker in question_set.score_tickers.items():
            answer = decision.answers.get(question_id)
            if isinstance(answer, ScoreAnswer):
                levels = question_set.questions[question_id].criteria or []
                probability = cal.score_to_probability(float(answer.score), len(levels))
                decision.rank_probabilities[ticker] = probability
                decision.answered_rank[ticker] = probability
            else:
                decision.rank_probabilities[ticker] = NEUTRAL
                decision.backfilled.append(ticker)