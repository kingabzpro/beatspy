# BeatSPY

**Decision models, not chat models.** Four decision models read the same frozen
point-in-time market state, answer the same typed questions, and return
probabilities. This benchmark scores those probabilities twice: were they
*calibrated*, and did a portfolio built only from them beat SPY?

[Open dashboard →](https://beatspy.vercel.app) · [Architecture](ARCHITECTURE.md) · [Contributing](CONTRIBUTING.md)

There is no agent relay, no tool calling, and no prompt engineering here. A
decision model returns typed answers, so the input is data and the output is a
number. That makes the whole thing auditable: every probability the model returns
is recorded and scored against what actually happened next.

## The models

| Model | Provider | Endpoint | Input price |
| --- | --- | --- | --- |
| `gpt-6-luna` | OpenAI [Decisions API](https://developers.openai.com/api/docs/guides/decisions) (public beta) | `api.openai.com/v1/decisions` | $0.10 / 1M |
| `jev-latest` | [TypeSafe Jev](https://docs.typesafe.ai/api-reference) | `api.typesafe.ai/v1/systemone` | $0.042 / 1M |
| `clef` | Cloudflare [Clef](https://developers.cloudflare.com/workers-ai/models/clef/) (27B) | Workers AI | $0.24 / 1M |
| `clef-flash` | Cloudflare Clef-flash (9B) | Workers AI | $0.09 / 1M |

All four bill **input tokens only** — output is free because there is no output
text, and none of them offers prompt caching. Pricing verified 2026-10-08.

Because a run asks one question per asset and one model call answers them all, a
complete 25-week benchmark of all four models costs about **$0.07**.

## Run it

Install [Python 3.11+](https://www.python.org/) and [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/kingabzpro/beatspy.git
cd beatspy
uv sync --extra dev
uv run beatspy models                # the decision models, pricing, and which keys resolve
uv run beatspy setup                 # pick a model and store its key
uv run beatspy run --model clef-flash
uv run beatspy report --open
```

Run several models at once. They share only the read-only snapshot, so they run
concurrently:

```bash
uv run beatspy run --model clef-flash clef jev-latest gpt-6-luna --scenario 2026-6m
```

Keys come from the environment, `.env`, or `~/.beatspy/secrets.env`, in that
order of precedence for an exported variable, the deliberate store, and then the
project file. `beatspy doctor --probe` makes one live call per model and reports
the answer it returned.

`beatspy run` fetches frozen adjusted prices, walks the decision schedule, asks one
provider for one decision's probabilities at a time, executes at the next
session's open, and writes a scored run under `results/`.

Useful flags:

```bash
uv run beatspy run --model jev-latest --scenario 2026-ytd --freq weekly
uv run beatspy run --model gpt-6-luna --max-decisions 12
uv run beatspy run --model clef --form all      # ask every question form
uv run beatspy calibrate <run-id>               # calibration table for one run
uv run beatspy verify results/<run-id>          # offline replay of a finished run
uv run beatspy compare                          # latest run per decision model
```

`beatspy demo` generates synthetic runs so you can preview the dashboard with no
keys and no network.

## What it asks

`--form per_asset` (the default) asks **one question per investable asset, for
every asset, every decision** — 25 assets means 25 independent predictions each
week, and each asset's answer stands on its own. `--form twin` or `all` add the
choice and rank forms for the same ticker in the same request, which is how the
form itself gets measured.

This matters, because the reported failure mode of these models is form-dependent:
community replication found the `choice` form badly miscalibrated — a 70/30 event
answered as heads **98%** of the time, with probabilities that moved when the
options were reordered — while the predicate form returned 70% correctly. So the
choice form is *measured as a hazard*, not trusted as a signal.

## How the probabilities become a portfolio

The mapping is fixed in code before any result is seen, so a run cannot be tuned
after the fact. The default is **graded**:

1. only `noul` probabilities are used;
2. half the book is an even spread across the whole universe, so every asset is
   always held at some weight;
3. the other half is allocated in proportion to each asset's probability above
   the prior (`base_weight`);
4. the scenario's position limit applies per asset, and cash absorbs any remainder.

An earlier policy held only the top three names above 0.5, equal-weighted. It could
emit nothing but 0% or 33% per position, rewrote the entire book every week
(~50% turnover), and left a single confident asset taking the whole book. `graded`
exists to remove that: a weak forecast becomes a mild tilt rather than a binary
switch. `selection = "top_n"` restores the old behaviour for reproducing runs
recorded under it.

An unanswered question is backfilled at the neutral, so a provider failure is
recorded as missing coverage and can never become a directional bet.

## How it is scored

**Calibration.** Every probability is labeled by whether that ticker actually beat
the benchmark over the horizon — a *relative* outcome, so a rising market cannot
mark every prediction correct. Reported per form: Brier, Brier skill (versus a
constant at the sample rate), log loss, AUC with ties handled by the Mann-Whitney
U statistic, accuracy, and 10-bin reliability with expected and maximum
calibration error. The headline number is `miscalibration_gap`: the mean absolute
distance between the noul and choice answers to the same question.

**P&L.** The portfolio runs through the same accounting engine, on the same frozen
prices, against SPY buy-and-hold and three deterministic references: equal weight,
60/40, and 12-minus-1 momentum. Results are reported with the observation count
and answer coverage beside them, because a return without its coverage is not
interpretable.

## Latest benchmark

Rolling six months to the latest completed session (`2026-6m`), weekly decisions,
25 stocks and ETFs, one probability question per asset per week — 625 predictions
per model. Frozen adjusted prices throughout.

| Decision model | Return | vs SPY | vs equal weight | Sharpe | Max DD | Brier skill | AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `clef-flash` | **+11.28%** | −7.22% | **+1.89%** | 2.02 | −3.75% | −0.318 | 0.496 |
| `gpt-6-luna` | +8.67% | −9.83% | −0.71% | 1.63 | −4.04% | −0.022 | 0.474 |
| `clef` | +7.28% | −11.22% | −2.11% | 1.30 | −5.32% | −0.301 | 0.499 |
| `jev-latest` | +7.01% | −11.49% | −2.38% | 1.28 | −6.06% | −0.077 | 0.490 |
| *SPY buy-and-hold* | *+18.50%* | — | — | — | — | — | — |
| *equal weight, same 25* | *+9.39%* | — | — | — | — | — | — |

Read this carefully, because the honest conclusion is narrower than the returns
suggest:

- **No decision model beat SPY.** SPY is cap-weighted and holds the mega-caps at
  full weight, so an equal-weighted book of the same names starts roughly nine
  points behind. That is a property of the benchmark, not of the models.
- **One model beat the equal-weight reference** (`clef-flash`, +1.89 points). That
  is the comparison that isolates the model, and one window is not enough to call
  it skill.
- **The probabilities carry no measurable signal.** Every model has *negative*
  Brier skill — worse than predicting the sample average — and an AUC of 0.474 to
  0.499, i.e. chance or slightly below. The returns come from diversifying across
  25 names and trading rarely, not from forecasting.
- Two providers are markedly underconfident: `clef` answered a mean 0.29 where the
  realized rate was 0.45.

An earlier configuration asked about 10 momentum-ranked candidates and held only
the top three above 0.5, equal-weighted. Because that can emit nothing but 0% or
33% per position, it rewrote the whole book weekly at ~50% turnover, and returned
−9.3% to +0.7%. One decision per asset with graded sizing roughly halved turnover
and added 8 to 17 points of return — a portfolio-construction gain, not a
forecasting one.

## Two things worth knowing before you trust a number

**One window is one draw.** A single deterministic momentum rule swings across a
~41-point range between sub-windows of one six-month period. Any single-window
ranking is a sample from a distribution wider than most differences between
models. Report the dispersion, not the peak.

**Brier alone cannot say which way a model is wrong.** It is invariant under
flipping both the probability and the outcome distribution. The reliability bins
and accuracy are what distinguish "confidently wrong" from "hedging".

**Beating SPY is partly structural.** SPY is cap-weighted; any equal-weighted
selection of the same universe tends to lag it in a mega-cap-led market. Compare
against equal weight and momentum on the same universe to isolate what the model
contributed. The dashboard reports both.

## Requirements

- Python 3.11+ and `uv`. Dependencies: pydantic, pandas, numpy, httpx,
  exchange-calendars, yfinance.
- A key for whichever decision model you run. TypeSafe needs `TYPESAFE_API_KEY`;
  OpenAI needs `OPENAI_API_KEY`; Cloudflare needs `CLOUDFLARE_ACCOUNT_ID` plus
  `CLOUDFLARE_AUTH_TOKEN`. Each also has a `BEATSPY_`-prefixed form that wins.
- Keys live in the environment, a project `.env`, or `~/.beatspy/secrets.env`.
  The dashboard contains no credentials, and `.env` is git-ignored.

Everything is HTTP-based, so the Clef path works from Python through the Workers AI
REST endpoint — no Cloudflare Worker or TypeScript build step is required.

## Development

```bash
uv sync --extra dev
uv run --no-sync pytest
uv run --no-sync ruff check .
node --test tests/dashboard.test.cjs
```

The whole test suite is hermetic: no network, no credentials, and no real models.
A scripted offline client answers every question form, so complete runs are
executed, scored, replayed, and rendered inside the tests.

Apache-2.0 · [LICENSE](LICENSE)