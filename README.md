<div align="center">

# BeatSPY

**Decision models, not chat models.**

Four decision models read one frozen point-in-time market state, answer the same
typed probability questions, and are scored twice: were the probabilities
calibrated, and did the portfolio they built beat SPY?

[Live dashboard](https://beatspy.vercel.app) · [Architecture](ARCHITECTURE.md) · [Roadmap](ROADMAP.md) · [Contributing](CONTRIBUTING.md)

[![CI](https://github.com/kingabzpro/beatspy/actions/workflows/ci.yml/badge.svg)](https://github.com/kingabzpro/beatspy/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Full 4-model run](https://img.shields.io/badge/full%204--model%20run-%240.07-brightgreen.svg)](#the-models)

</div>

> [!IMPORTANT]
> **The honest result.** In the published window no model beat SPY, and no model's
> probabilities show measurable skill: every Brier skill is negative and every AUC
> permutation p-value is far from significant. That window also punished the
> scenario's own momentum selection, so the table's most useful line is the one
> that compares each model against simply holding the names it was given.

## Latest benchmark

Rolling six months to the latest completed session (`2026-6m`), weekly decisions,
frozen adjusted prices, window 2026-04-09 to 2026-10-09. The pool is 25 assets:
large companies across all 11 sectors, plus broad equities and long Treasuries.
Each decision may trade **only the pool's twelve strongest trailing six-month
performers versus SPY**, ranked point-in-time from prices up to that date, so the
book is the model's choice among real assets. SPY is the yardstick and is never
held. That narrowing is itself a momentum rule, and this window punished it: its
own equal-weight return is the reference line below.

| Decision model | Return | vs its own set | vs SPY | Sharpe | Max DD | Turnover | Brier skill | AUC | p(AUC) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `clef-flash` | **+0.46%** | **+0.84%** | −14.63% | 0.13 | −10.26% | 31.1% | −0.289 | 0.460 | 0.91 |
| `gpt-6-luna` | +0.36% | +0.74% | −14.73% | 0.12 | −9.24% | 32.0% | −0.050 | 0.428 | 0.96 |
| `clef` | −0.54% | −0.16% | −15.63% | −0.02 | −11.15% | 33.1% | −0.245 | 0.477 | 0.83 |
| `jev-latest` | −0.60% | −0.22% | −15.69% | −0.04 | −10.79% | 34.8% | −0.078 | 0.478 | 0.76 |
| *its investable set, equal weight* | *−0.38%* | *n/a* | *−15.47%* | *n/a* | *n/a* | *n/a* | *n/a* | *n/a* | *n/a* |
| *whole pool, equal weight* | *+9.09%* | *n/a* | *−6.00%* | *n/a* | *n/a* | *n/a* | *n/a* | *n/a* | *n/a* |
| *SPY buy-and-hold* | *+15.09%* | *n/a* | *n/a* | *n/a* | *n/a* | *n/a* | *n/a* | *n/a* | *n/a* |

Read this carefully, because the story is in the reference lines, not the models:

- **The selection rule lost the money, not the models.** Holding the same twelve
  names equally, with no model input at all, returned −0.38% while the full pool
  made +9.09% and SPY +15.09%. Trailing six-month winners gave back most of
  their lead in this window, at every setting tested (3, 6 and 9-month
  lookbacks, top 8 to 20).
- **Two models beat the set they were given, barely.** `clef-flash` (+0.84) and
  `gpt-6-luna` (+0.74) added less than a point over equal weight of the same
  names; `clef` and `jev-latest` were slightly behind. With no measurable
  signal, that is what the numbers should look like.
- **Turnover is the visible cost of a rotating universe**: 31 to 35% per
  decision, against 5% for a fixed pool, because names enter and leave the
  top twelve every week.
- **The probabilities still carry no measurable signal**: every Brier skill is
  negative and every AUC permutation p-value is far from significant
  (0.76 to 0.96).

<details>
<summary><b>Universe and policy history, and why the numbers moved</b></summary>

Four configurations have been published, and only one of them was ever about
forecasting:

1. **Top three above 0.5** (the earliest cut): rewrote the whole book weekly at
   ~50% turnover and returned −9.3% to +0.7%.
2. **Graded over the whole pool** (protocol 2): halved turnover, added 8 to 17
   points, and left the book about 7 points behind SPY, because an equal-weighted
   selection of a mega-cap-led market lags the cap-weighted index itself.
3. **SPY core plus sleeve** (protocol 3, withdrawn): tracked SPY to within a
   point, but 85% of the book was the index, so the score stopped measuring the
   model's choices.
4. **Trailing-performer pool** (protocol 4, current): no index, the model
   chooses among real assets, and the report adds the reference line that
   isolates the rule's own return. In this window the rule was the problem,
   which is a result worth publishing rather than tuning away.

Every gain in that list is portfolio construction or universe selection, not
forecasting: the calibration columns are unmoved by all of it.

</details>

<details>
<summary><b>Is the momentum selection worth keeping?</b></summary>

The scenario makes it a one-line change, and the offline sweep (no model calls,
equal weight so the rule is measured on its own) says:

| Lookback | Top N | Rule's own return | Turnover |
| --- | --- | --- | --- |
| 63 days | 12 | +6.75% | 18.2% |
| 126 days | 12 | −0.38% | 14.6% |
| 189 days | 20 | +7.48% | 6.4% |
| none, whole pool | 25 | +9.09% | 3.3% |

One window is one draw, so this is not proof that momentum loses. It is proof
that the rule can dominate the result, which is why the model-vs-its-own-set
column exists. Widen `selection_top`, lengthen `selection_lookback_days`, or drop
`selection_top` to measure the models over the whole pool.

</details>

## The models

| Model | Provider | Endpoint | Input price |
| --- | --- | --- | --- |
| `gpt-6-luna` | OpenAI [Decisions API](https://developers.openai.com/api/docs/guides/decisions) (public beta) | `api.openai.com/v1/decisions` | $0.10 / 1M |
| `jev-latest` | [TypeSafe Jev](https://docs.typesafe.ai/api-reference) | `api.typesafe.ai/v1/systemone` | $0.042 / 1M |
| `clef` | Cloudflare [Clef](https://developers.cloudflare.com/workers-ai/models/clef/) (27B) | Workers AI | $0.24 / 1M |
| `clef-flash` | Cloudflare Clef-flash (9B) | Workers AI | $0.09 / 1M |

All four bill **input tokens only**: output is free because there is no output
text, and none offers prompt caching (pricing checked 2026-10-08). One run asks
one question per asset and one call answers them all, so a complete 25-week
benchmark of all four models costs about **$0.07**.

## Quick start

```bash
git clone https://github.com/kingabzpro/beatspy.git
cd beatspy
uv sync --extra dev
uv run beatspy setup                 # pick a model and store its key
uv run beatspy run --model clef-flash --scenario 2026-6m
uv run beatspy report --open         # render the dashboard locally
```

No keys yet? `uv run beatspy demo` builds synthetic runs and renders the same
dashboard offline. `uv run beatspy models` shows which credentials resolve, and
`uv run beatspy doctor --probe` makes one live call per model.

## How it works

```mermaid
flowchart LR
    A["Freeze prices<br/>content-addressed snapshot"] --> B["Build state<br/>only data on or before as_of"]
    B --> C["Ask typed questions<br/>one per asset"]
    C --> D["Model returns probabilities"]
    D --> E["Fixed sizing policy<br/>SPY core + conviction sleeve"]
    E --> F["Fill at the next session's open"]
    F --> G["Label vs benchmark<br/>Brier, AUC, reliability"]
    F --> H["Score portfolio<br/>vs SPY and baselines"]
```

<details>
<summary><b>One decision, step by step</b></summary>

1. The scenario resolves its dates and loads a content-addressed snapshot, or
   freezes a new one; a 15-minute buffer keeps unfinished sessions out.
2. The state is built from data dated on or before the decision date only, and
   kept inside a token budget that is enforced, not assumed.
3. The model returns one typed answer per question: `noul` (a yes/no
   probability), `choice`, or `score`. Refusals and missing answers are recorded
   as missing coverage, never invented.
4. The fixed policy turns the probabilities into target weights. The model never
   states an allocation.
5. Orders fill at the next session's open. Every probability is then labeled by
   whether that asset actually beat the benchmark over the same horizon.

</details>

<details>
<summary><b>How probabilities become a portfolio</b></summary>

The default policy is **graded_cs**: the whole book spreads across the
investable set, half evenly and half by each probability's distance above the
decision's own mean, so the same ranking produces the same weights under any
scale a provider answers on. A scenario may narrow the investable set to its
strongest trailing performers (`selection_top`); the ranking is point-in-time and
the report adds an equal-weight line for that same set, so the rule's own return
stays visible instead of being credited to the model.

The index is never investable under any policy: SPY is the yardstick the labels
and the comparison are measured against. `graded` (conviction above a fixed 0.5
prior) and `top_n` (the earliest top-three cut) stay selectable so recorded runs
remain reproducible, and under every graded policy an unanswered question carries
no conviction share, so a provider failure can never become a directional bet.

</details>

<details>
<summary><b>How it is scored</b></summary>

**Calibration.** Each probability is labeled by whether the asset beat the
benchmark over the horizon, a relative outcome, so a rising market cannot mark
every prediction correct. Per form: Brier, Brier skill against a constant at the
sample rate, log loss, tie-aware AUC, accuracy, and 10-bin reliability with
expected and maximum calibration error. `miscalibration_gap` compares the noul
and choice forms on the same question and is reported only when a run asks both
(`--form twin` or `all`).

**P&L.** The portfolio runs through the same accounting engine, on the same
frozen prices, against SPY buy-and-hold and three deterministic references:
equal weight, 60/40, and 12-minus-1 momentum. Sharpe uses a zero risk-free rate
(`risk_free_annual = 0.0`), recorded per run.

**Uncertainty.** Samples are clustered by decision date, so a run's effective
sample is closer to its number of decisions than to its number of predictions.
`metrics.json` carries a 95% block-bootstrap interval for the return and for the
paired excess over SPY, and a two-sided permutation p-value for each signal's
Brier skill and AUC, with outcomes shuffled within decision dates. Both are
seeded, so replay reproduces them exactly.

</details>

<details>
<summary><b>What one window can prove</b></summary>

`examples/power_analysis.py` simulates the benchmark's own tests. For a book with
about 1% daily volatility, the share of runs whose 95% interval excludes a true
excess of a given size:

| Decision windows | 2%/yr | 5%/yr | 10%/yr | 20%/yr |
| --- | --- | --- | --- | --- |
| 25 (the published run) | 6% | 6% | 7% | 14% |
| 50 | 5% | 6% | 10% | 26% |
| 100 | 4% | 6% | 14% | 44% |
| 250 | 5% | 8% | 28% | 79% |

A six-month, 25-decision benchmark cannot separate a 20%/year edge from luck;
roughly 250 weekly windows are needed for 80% power at that effect size. The
table describes the sample, it does not test the model.

For calibration at the observed probability spread (sd about 0.05), the two
p-values behave very differently, which is why every signal reports both. A
perfectly calibrated informative model (AUC 0.55) is rejected by the Brier-skill
test only 3% of the time even at 100 windows, because the Brier score is
dominated by its calibration term near 0.5; the AUC test finds the same model 68%
of the time at 25 windows. In short: read `auc_p_value` when probabilities sit
near 0.5. And Brier alone cannot say which way a model is wrong: it is invariant
under flipping both the probability and the outcome distribution, so reliability
bins and accuracy are what separate "confidently wrong" from "hedging".

</details>

<details>
<summary><b>Trust: replay, no look-ahead, no credentials in the dashboard</b></summary>

- `beatspy verify` replays a finished run offline: artifact hashes, the snapshot
  manifest, the decision schedule, and the recorded weights through the
  accounting engine, and requires exact agreement with the stored metrics.
- `examples/leakage_audit.py` is an adversarial check: it tampers with every
  price after a run's midpoint and requires every earlier decision to be
  byte-identical while later ones change. On the published runs, 12 of 12
  pre-cut decisions were identical and 13 of 13 post-cut decisions changed.
- Prices are frozen into content-addressed snapshots; the state contains only
  data dated on or before the decision date, and the token budget is enforced
  before the state is sent.
- The dashboard contains no credentials and reads reviewed JSON artifacts only.

</details>

<details>
<summary><b>Commands</b></summary>

```bash
uv run beatspy run --model clef-flash clef jev-latest gpt-6-luna --scenario 2026-6m
uv run beatspy run --model gpt-6-luna --max-decisions 12
uv run beatspy run --model clef --form all        # ask every question form
uv run beatspy run --max-cost-usd 0.05            # abort a run that would overspend
uv run beatspy state --scenario 2026-6m           # what a decision would read, and its token cost
uv run beatspy calibrate <run-id>                 # calibration table for one run
uv run beatspy verify results/<run-id>            # offline replay of a finished run
uv run beatspy compare                            # latest run per decision model
```

</details>

<details>
<summary><b>Requirements and credentials</b></summary>

- Python 3.11+ and [uv](https://docs.astral.sh/uv/). Dependencies: pydantic,
  pandas, numpy, httpx, exchange-calendars, yfinance.
- A key for whichever model you run. TypeSafe needs `TYPESAFE_API_KEY`; OpenAI
  needs `OPENAI_API_KEY`; Cloudflare needs `CLOUDFLARE_ACCOUNT_ID` plus
  `CLOUDFLARE_AUTH_TOKEN`. Each also has a `BEATSPY_`-prefixed form that wins.
- Keys come from the environment, a project `.env`, or `~/.beatspy/secrets.env`.
  `.env` is git-ignored and never imported by the dashboard.
- Everything is HTTP-based, so the Clef path works through the Workers AI REST
  endpoint with no Worker or TypeScript build step.

</details>

<details>
<summary><b>Development</b></summary>

```bash
uv sync --extra dev
uv run --no-sync pytest              # 270+ tests, no network and no credentials
uv run --no-sync ruff check .
node --test tests/dashboard.test.cjs
uv run python examples/leakage_audit.py   # adversarial look-ahead audit on a finished run
```

The suite is hermetic: a scripted offline client answers every question form, so
complete runs are executed, scored, replayed, rendered, and audited inside the
tests.

</details>

## License

Apache-2.0. See [LICENSE](LICENSE).
