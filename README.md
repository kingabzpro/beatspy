<div align="center">

# BeatSPY

**Decision models, not chat models.**

Thirty-one decision models across two protocol versions read the same frozen
point-in-time market state, answer the same typed probability questions, and are
scored twice: were the probabilities calibrated, and did the portfolio they built
beat SPY? Protocol v2 (2026-10-23) adds risk and context to the state: RSI-14,
30-day realized volatility, distance from the 52-week high, a volume ratio, and a
sector label, because under v1 the models mostly echoed the momentum column.

[Live dashboard](https://beatspy.vercel.app) · [Architecture](ARCHITECTURE.md) · [Roadmap](ROADMAP.md) · [Contributing](CONTRIBUTING.md)

[![CI](https://github.com/kingabzpro/beatspy/actions/workflows/ci.yml/badge.svg)](https://github.com/kingabzpro/beatspy/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![v2 26-model run](https://img.shields.io/badge/v2%2026--model%20run-%240.35-brightgreen.svg)](#the-models)

</div>

> [!IMPORTANT]
> **The honest result.** Across the three v2 windows, the best average gap to
> SPY in the table below is +1.59pp (`gemini-decision-1`) and no Brier skill is
> positive. Fourteen model-window cells reach a nominal AUC p-value under 0.05
> (smallest: `kairos` p 0.002), against roughly four expected by chance at
> this many tests; the cells share windows and models, so none survives a
> proper correction. Five models beat SPY in two of three windows
> (`gemini-decision-1`, `mammoth-1`, `kairos`, `subq-snap-1.1`,
> `drex-max-finance`), and every margin sits inside its bootstrap interval. A
> single-window leader can sit mid-table once the other windows run, which is
> why the board scores three. v2 moved mean discrimination above a coin flip
> for eleven models (mean AUC up to 0.526, from 0.39 on the momentum-only
> state): a reason to keep testing, not a claim of edge.

## Latest benchmark: three windows under protocol v2

Protocol v2, scored on three consecutive six-month windows (oldest to newest:
Oct 2025, Apr 2026, Oct 2026 endpoints), weekly decisions, frozen adjusted
prices. The pool is 25 large companies across all 11 sectors plus long
Treasuries; the book spreads across them by conviction, and SPY is the
yardstick, never held.

| Decision model | Oct 25 | Apr 26 | Oct 26 | Average | Beat SPY |
| --- | --- | --- | --- | --- | --- |
| `gemini-decision-1` (Google) | +3.32pp | +7.75pp | -6.28pp | **+1.59pp** | **2/3** |
| `span-01-free` (first-party) | -3.52pp | +6.29pp | -1.34pp | +0.48pp | 1/3 |
| `clef` via Requesty | -0.10pp | **+8.28pp** | -6.92pp | +0.42pp | 1/3 |
| `mammoth-1` (Baseten, free) | +0.35pp | +8.03pp | -7.54pp | +0.28pp | **2/3** |
| `gpt-6-luna` | -0.11pp | +6.48pp | -6.15pp | +0.07pp | 1/3 |
| `clef` | -0.77pp | +7.36pp | -6.53pp | +0.02pp | 1/3 |
| `clef-omni` | -1.21pp | +4.74pp | -4.10pp | -0.19pp | 1/3 |
| `kairos` (Minara) | +1.63pp | +5.27pp | -8.04pp | -0.38pp | **2/3** |
| `subq-snap-1.1` | +2.06pp | +4.94pp | -8.35pp | -0.45pp | **2/3** |
| `elephant-1` (Baseten, free) | -1.31pp | +6.35pp | -6.83pp | -0.60pp | 1/3 |
| `bison-1` (Baseten) | -1.76pp | +5.82pp | -6.71pp | -0.89pp | 1/3 |
| `clef-flash` | -1.68pp | +3.74pp | -4.79pp | -0.91pp | 1/3 |
| `groundhog-1` (Baseten) | -2.53pp | +6.27pp | -6.95pp | -1.07pp | 1/3 |
| `drex-max-finance` (Nace) | +1.45pp | +5.22pp | -10.01pp | -1.11pp | **2/3** |
| `d1.2-ultra` (Liquid) | -1.62pp | +6.17pp | -7.96pp | -1.14pp | 1/3 |
| `d3-turbo` (free) | -0.59pp | +5.26pp | -8.77pp | -1.36pp | 1/3 |
| `ms-decision-1` | -3.85pp | +7.23pp | -7.49pp | -1.37pp | 1/3 |
| `drex-v2.0` (Nace) | -0.45pp | +4.98pp | -8.93pp | -1.47pp | 1/3 |
| `jev-1.14` | -0.70pp | +3.50pp | -7.87pp | -1.69pp | 1/3 |
| `joey-1` (Baseten) | -6.39pp | +5.71pp | -5.76pp | -2.15pp | 1/3 |
| `pplx-decider` | -1.20pp | -0.51pp | -5.39pp | -2.37pp | 0/3 |
| `d1.2` (Liquid) | -1.12pp | +4.44pp | -10.50pp | -2.39pp | 1/3 |
| `drex-max-v1.1` (Nace) | -2.23pp | +6.75pp | -11.79pp | -2.42pp | 1/3 |
| `d3-edge` (free) | -3.47pp | +3.98pp | -7.82pp | -2.44pp | 1/3 |
| `d3-mini` (free) | -2.54pp | +3.12pp | -8.31pp | -2.58pp | 1/3 |
| `standard-one-8b` | -1.44pp | +3.05pp | -11.44pp | -3.28pp | 1/3 |

Each column is the model's gap to SPY in one six-month window (SPY made
+23.04%, +1.88% and +15.09%). The Apr 26 window lifted 25 of 26 models above
SPY because the pool itself outran a flat index there; the other two windows
are the honest test, and there the gaps are negative or inside noise. The
single-window returns live on the dashboard, one chip per window, with the
three-window average as the default lens. Protocol-5 results are never mixed
in: they live on the site's archive page.

## The models

Twelve providers are wired natively: OpenAI Decisions (`gpt-6-luna`), Google
Decisions (`gemini-decision-1`, $0.10/M input), TypeSafe (`jev-1.14`),
Cloudflare Workers AI (`clef`, `clef-flash`, `clef-omni`), Requesty
(`jev-1.14`, `pplx-decider`, `ms-decision-1`), Standard Thinking
(`standard-one-8b`), Liquid (`d1.2`, `d1.2-ultra`), Nace (`drex-max-v1.1`,
`drex-max-finance`, `drex-v2.0`), Baseten (`mammoth-1`, `elephant-1`,
`bison-1`, `groundhog-1`, `joey-1`), vLLM SR (`d3-turbo`, `d3-mini`, `d3-edge`,
all free), Subquadratic (`subq-snap-1.1`), and Minara Cloud (`kairos`, a
Jev-family deployment at $0.042/M input). Respan also answers first-party:
`span-01-free` via api.respan.ai's behavior-scoring endpoint (free, each noul
question mapped to one scored proposition) ran all three windows at 100%
coverage, no daily-cap trouble (`span-02` first-party needs paid Respan
credits). The paid OpenRouter routes (`kev-4b`, `liquid-d1`,
`microsoft-decision-1.1`, `heygen-decision-1`, `morph/reflex`) still wait on a
zero balance. All bill input tokens only. Most answer every question in one
call; `kairos` takes eight questions per request and re-reads the state per
question (about 98,500 tokens per decision), as do the two small d3s, so
compare cost per decision rather than per token. The 26-model v2 board above
cost about **$0.35** for its latest window. `beatspy models` prints every
runnable id, endpoint, and price.

### Timestral forecasting benchmark

`timestral-209m` forecasts each asset's adjusted closing price relative to SPY
from up to 256 trading sessions of history through the decision date. BeatSPY
interpolates the forecast quantiles at the current relative price to estimate
the probability of beating SPY over the scheduled holding period. The run uses
the published frozen snapshots, weekly schedule, portfolio policy, and scoring.
The checkpoint's exposure to these market windows during training is unknown.

| Window end | Timestral return | SPY return | Gap to SPY | AUC |
| --- | ---: | ---: | ---: | ---: |
| Oct 2025 | +25.69% | +23.04% | +2.65pp | 0.506 |
| Apr 2026 | +9.09% | +1.88% | +7.21pp | 0.527 |
| Oct 2026 | +10.68% | +15.09% | -4.41pp | 0.467 |

Timestral beat SPY in two of three windows and averaged +1.82 percentage
points versus SPY. Version 5 of the [Kaggle notebook](https://www.kaggle.com/code/kingabzpro/beatspy-timestral)
ran [all three windows](examples/kaggle_timestral.py) on a Tesla T4. Each run
had 100% answer coverage and passed BeatSPY replay. The CLI model supports
the `per_asset`, `noul`, `choice`, and `twin` question forms.

### LightGBM baseline

`lightgbm` refits a gradient-boosted classifier at each decision. It uses the
protocol v2 price, risk, volume, and sector features from earlier Fridays and
includes a training label only once the corresponding forward return is known.
Training labels use the same trading-session horizon as the current decision.
The benchmark keeps the published snapshots, schedule, sizing, and scoring.

| Window end | LightGBM return | SPY return | Gap to SPY | AUC |
| --- | ---: | ---: | ---: | ---: |
| Oct 2025 | +28.15% | +23.04% | +5.11pp | 0.506 |
| Apr 2026 | +6.93% | +1.88% | +5.05pp | 0.490 |
| Oct 2026 | +8.20% | +15.09% | -6.89pp | 0.523 |

LightGBM beat SPY in two of three windows, averaging +1.09 percentage points
versus SPY. Brier skill was negative in each window. All three runs had 100%
answer coverage and passed replay.

The leaderboard's **Show all model types** toggle includes LightGBM, Timestral,
and other strategies in both the single-window and all-window views. The
[Kaggle notebook](https://www.kaggle.com/code/kingabzpro/beatspy-lightgbm) runs
and verifies all three windows with the [LightGBM runner](examples/kaggle_lightgbm.py).
For the CLI, install `lightgbm==4.6.0` and `scikit-learn` in the environment
before running `beatspy run --model lightgbm`.

### Logistic regression and the LightGBM blend

`logistic-regression` fits an L2-regularized classifier to the same features,
dates, and settled outperformance labels as LightGBM. Imputation, scaling, and
sector encoding are fitted from each decision's training data. The
`logistic-lightgbm` model averages the two outperformance probabilities with
fixed 50/50 weights. Both use the same sizing, costs, and three frozen windows.

| Model | Oct 2025 vs SPY | Apr 2026 vs SPY | Oct 2026 vs SPY | Average gap |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | -0.89pp | +8.17pp | -8.76pp | -0.49pp |
| Logit + LightGBM | +1.54pp | +7.07pp | -7.79pp | +0.27pp |
| LightGBM | +5.11pp | +5.05pp | -6.89pp | +1.09pp |

The blend improved Brier skill over standalone LightGBM in every window, and
its Brier skill stayed negative in all three. All six new runs passed replay
with full answer coverage.

On the leaderboard, **Show all model types** reveals **LINEAR** for Logistic
Regression and **ENSEMBLE** for Logit + LightGBM. The
[Kaggle notebook](https://www.kaggle.com/code/kingabzpro/beatspy-logistic-benchmarks)
uses the [benchmark runner](examples/kaggle_logistic.py).

### Hierarchical Risk Parity

`hrp` estimates covariance from up to 126 aligned daily returns through the
decision close. It uses single-linkage clustering on correlation distances
and recursive bisection by cluster variance. The existing position cap,
weekly schedule, next-open execution, and trading costs apply to the
allocation. Equal weight rebalances on the same schedule, and SPY is held
as the benchmark. The run records weights directly for replay.

| Window end | HRP | Equal weight | SPY |
| --- | ---: | ---: | ---: |
| Oct 2025 | +10.31% | +21.33% | +23.04% |
| Apr 2026 | +7.31% | +8.19% | +1.88% |
| Oct 2026 | +2.84% | +9.09% | +15.09% |

HRP beat SPY in one window and equal weight in zero. Its maximum drawdown
was smaller than SPY's in all three windows. All three runs passed replay.

Use **Show all model types** to find **HRP** with the **ALLOC** tag. The
[Kaggle notebook](https://www.kaggle.com/code/kingabzpro/beatspy-hrp)
uses the [HRP runner](examples/kaggle_hrp.py). The CLI uses SciPy for clustering.

### Stock-JEPA

BeatSPY's independent adaptation of [STOCK-JEPA](https://arxiv.org/abs/2610.07006)
uses 32 sessions of stock and SPY returns and stock volume changes. A ridge
model predicts market statistics at 5- and 21-session horizons. A GRU and an
exponential moving-average target learn a latent correction to the ridge prior.
A probability readout on the frozen representation estimates whether each
stock will beat SPY over the upcoming decision period.

Each model fits once from the frozen snapshot's history through the day before
its window opens. The benchmark uses BeatSPY's existing portfolio policy.

| Window end | Stock-JEPA | Equal weight | SPY |
| --- | ---: | ---: | ---: |
| Oct 2025 | +19.04% | +21.33% | +23.04% |
| Apr 2026 | +8.27% | +8.19% | +1.88% |
| Oct 2026 | +10.41% | +9.09% | +15.09% |

Stock-JEPA beat SPY in one of three windows and equal weight in two. Its average
gap to SPY was −0.76 percentage points. All three runs replayed with full answer
coverage. Use **Show all model types** to find **Stock-JEPA** with the **JEPA** tag.
The [Kaggle notebook](https://www.kaggle.com/code/kingabzpro/beatspy-stock-jepa-l4)
uses the [GPU benchmark runner](examples/kaggle_stock_jepa.py). The CLI uses
PyTorch with `--model stock-jepa --form per_asset`.

### Chronos-2, TimesFM 3.0, and T0-Gamma

These pretrained models use Timestral's input and portfolio rules. Each model
forecasts the stock-to-SPY log price ratio from up to 256 trading sessions of
history. BeatSPY interpolates the terminal deciles at today's ratio to estimate
the probability of outperforming SPY over the scheduled decision horizon.
Models run without fine-tuning or additional covariates. The frozen snapshots
limit inputs at inference time. Checkpoint pretraining cutoffs were not audited
against the historical windows.

| Model | Oct 2025 vs SPY | Apr 2026 vs SPY | Oct 2026 vs SPY | Mean gap |
| --- | ---: | ---: | ---: | ---: |
| Timestral | +2.65 pp | +7.21 pp | −4.41 pp | +1.82 pp |
| T0-Gamma | −1.00 pp | +6.72 pp | −6.35 pp | −0.21 pp |
| TimesFM 3.0 | −1.14 pp | +5.78 pp | −6.77 pp | −0.71 pp |
| Chronos-2 | −3.78 pp | +4.05 pp | −8.88 pp | −2.87 pp |

Each new model beat SPY in one window. All nine runs replayed with full answer
coverage. Use **Show all model types** to find the **FORECAST** rows.

The [Kaggle runner](examples/kaggle_foundation_forecast.py) pins checkpoint
revisions and records the package version and GPU in each result summary.
The notebooks ran sequentially on Tesla T4 GPUs.
[Chronos-2](https://www.kaggle.com/code/kingabzpro/beatspy-chronos-2) ·
[TimesFM 3.0](https://www.kaggle.com/code/kingabzpro/beatspy-timesfm-3-0) ·
[T0-Gamma](https://www.kaggle.com/code/kingabzpro/beatspy-t0-gamma)

TimesFM 3.0 was used for non-commercial research evaluation under
[Google's model license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE).

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
dashboard offline.

## How it works

1. Prices are frozen into a content-addressed snapshot.
2. The state is built from data dated on or before the decision date only,
   inside an enforced token budget.
3. Each model answers one typed question per asset; refusals are recorded as
   missing coverage, never invented.
4. A fixed policy turns probabilities into target weights. The model never
   states an allocation.
5. Orders fill at the next session's open, then every answer is labeled by
   whether that asset beat SPY and scored for calibration (Brier, AUC,
   reliability) and P&L (vs SPY, equal weight, 60/40, momentum).

Trust: `beatspy verify` replays a finished run offline and requires exact
agreement with the stored metrics; `examples/leakage_audit.py` tampers with
prices after the midpoint and requires every earlier decision to stay
byte-identical; the dashboard reads reviewed JSON artifacts only, never
credentials. [ARCHITECTURE.md](ARCHITECTURE.md) has the full detail.

## Development

```bash
uv sync --extra dev
uv run --no-sync pytest              # 270+ tests, no network, no credentials
uv run --no-sync ruff check .
node --test tests/dashboard.test.cjs
```

Python 3.11+ and [uv](https://docs.astral.sh/uv/). Keys come from the
environment, a project `.env`, or `~/.beatspy/secrets.env`; `.env` is
git-ignored.

## License

Apache-2.0. See [LICENSE](LICENSE).
