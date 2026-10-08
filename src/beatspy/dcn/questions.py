"""Typed questions for the decision models.

One question per candidate ticker, phrased identically across providers. Three
answer forms are available, and the benchmark measures them against each other
because the reported failure mode of these models is form-dependent:

- `noul`  — "will this beat the benchmark?" as a yes/no probability. The primary
  signal, and the only one that can size a portfolio.
- `choice` — the same question as an option pick with a distribution. Community
  replication found this form badly miscalibrated (a 70/30 event answered as
  heads 98% of the time, with probabilities that moved when the options were
  reordered), so it is measured, not trusted.
- `rank`  — a five-level score per ticker, normalized into a probability so it is
  comparable with the other two.
- `twin`  — asks the noul and choice forms for the same ticker in one call, which
  is the cheapest way to quantify that form gap inside one provider.
- `all`   — asks all three forms for every ticker, so one run produces the full
  calibration picture (noul, choice and rank) for the same decisions.
- `per_asset` — ONE question per investable asset, every asset, every decision.
  This is the benchmark's default: 25 assets means 25 independent decisions each
  week. One question per stock is what makes a single asset's answer mean
  something on its own, instead of being read only through a top-N cut.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import Question

FORMS = ("noul", "choice", "rank", "twin", "all", "per_asset")

# Shared level definitions, so `rank` means the same thing for every provider.
RANK_LEVELS = [
    "Clearly worse than the benchmark over the horizon",
    "Somewhat worse than the benchmark",
    "About the same as the benchmark",
    "Somewhat better than the benchmark",
    "Clearly better than the benchmark over the horizon",
]

CHOICE_OPTIONS = {
    "outperform": "Total return above the benchmark over the horizon",
    "underperform": "Total return at or below the benchmark over the horizon",
}


@dataclass
class QuestionSet:
    """The questions for one decision, plus the mapping back to tickers."""

    form: str
    questions: dict[str, Question] = field(default_factory=dict)
    # question id -> ticker (only the ids that carry a tradeable probability)
    noul_tickers: dict[str, str] = field(default_factory=dict)
    choice_tickers: dict[str, str] = field(default_factory=dict)
    score_tickers: dict[str, str] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    @property
    def ids(self) -> list[str]:
        return list(self.questions)

    def ticker_for(self, question_id: str) -> str | None:
        return self.noul_tickers.get(question_id) or self.choice_tickers.get(question_id) or self.score_tickers.get(
            question_id
        )


def _horizon(horizon_trading_days: int) -> str:
    return f"the next {horizon_trading_days} trading days"


def _outperform_instructions(ticker: str, benchmark: str, horizon: str) -> str:
    return {
        "question": (
            f"Will `{ticker}` deliver a higher total return than `{benchmark}` over {horizon}, "
            f"judged on closing prices?"
        ),
        "ticker": ticker,
        "benchmark": benchmark,
        "horizon": horizon,
        "state_fields": "market",
    }


def build_questions(
    form: str,
    tickers: list[str],
    benchmark: str,
    horizon_trading_days: int,
) -> QuestionSet:
    """Build the typed questions for one decision."""
    if form not in FORMS:
        raise ValueError(f"unknown question form {form!r}; use one of {FORMS}")
    horizon = _horizon(horizon_trading_days)
    bench = benchmark.upper()
    qs = QuestionSet(form=form)

    for ticker in tickers:
        name = ticker.upper()
        if form in ("noul", "twin", "all", "per_asset"):
            question_id = f"noul_{name}"
            qs.questions[question_id] = Question(
                type="noul",
                instructions=_outperform_instructions(name, bench, horizon),
            )
            qs.noul_tickers[question_id] = name
        if form in ("choice", "twin", "all"):
            question_id = f"choice_{name}"
            qs.questions[question_id] = Question(
                type="choice",
                instructions=_outperform_instructions(name, bench, horizon),
                criteria=dict(CHOICE_OPTIONS),
            )
            qs.choice_tickers[question_id] = name
        if form in ("rank", "all"):
            question_id = f"rank_{name}"
            qs.questions[question_id] = Question(
                type="score",
                instructions={
                    "question": (
                        f"How will `{name}` perform relative to `{bench}` over {horizon}, "
                        f"judged on closing prices?"
                    ),
                    "ticker": name,
                    "benchmark": bench,
                    "horizon": horizon,
                    "state_fields": "market",
                },
                criteria=list(RANK_LEVELS),
            )
            qs.score_tickers[question_id] = name
        qs.order.append(name)

    return qs