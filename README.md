<div align="center">

# BeatSPY

**Decision models, not chat models.**

Twenty-five decision models across two protocol versions read the same frozen
point-in-time market state, answer the same typed probability questions, and are
scored twice: were the probabilities calibrated, and did the portfolio they built
beat SPY? Protocol v2 (2026-10-23) adds risk and context to the state: RSI-14,
30-day realized volatility, distance from the 52-week high, a volume ratio, and a
sector label, because under v1 the models mostly echoed the momentum column.

[Live dashboard](https://beatspy.vercel.app) · [Architecture](ARCHITECTURE.md) · [Roadmap](ROADMAP.md) · [Contributing](CONTRIBUTING.md)

[![CI](https://github.com/kingabzpro/beatspy/actions/workflows/ci.yml/badge.svg)](https://github.com/kingabzpro/beatspy/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![v2 21-model run](https://img.shields.io/badge/v2%2021--model%20run-%240.30-brightgreen.svg)](#the-models)

</div>

> [!IMPORTANT]
> **The honest result.** Across the three v2 windows, the best average gap to
> SPY among the decision models in the table below is +0.48pp and no Brier skill is
> positive. Eleven model-window cells reach a nominal AUC p-value under 0.05
> (smallest: `kairos` p 0.002), against roughly three expected by chance at
> this many tests; the cells share windows and models, so none survives a
> proper correction. Four models beat SPY in two of three windows
> (`mammoth-1`, `jev-latest`, `subq-snap-1.1`, `kairos`), and every margin sits
> inside its bootstrap interval. The single-window leader (Clef Omni) fell to
> mid-table once the other windows ran, which is exactly why the board scores
> three. v2 did move mean discrimination above a coin flip for nine models
> (mean AUC up to 0.526, from 0.39 on the momentum-only state), which is a
> reason to keep testing, not a claim of edge.

## Latest benchmark: three windows under protocol v2

Protocol v2, scored on three consecutive six-month windows (oldest to newest:
Oct 2025, Apr 2026, Oct 2026 endpoints), weekly decisions, frozen adjusted
prices. The pool is 25 large companies across all 11 sectors plus long
Treasuries; the book spreads across them by conviction, and SPY is the
yardstick, never held.

| Decision model | Oct 25 | Apr 26 | Oct 26 | Average | Beat SPY |
| --- | --- | --- | --- | --- | --- |
| `span-01-free` (first-party) | -3.52pp | +6.29pp | -1.34pp | **+0.48pp** | 1/3 |
| `clef` via Requesty | -0.10pp | **+8.28pp** | -6.92pp | +0.42pp | 1/3 |
| `mammoth-1` (Baseten, free) | +0.35pp | +8.03pp | -7.54pp | +0.28pp | **2/3** |
| `gpt-6-luna` | -0.11pp | +6.48pp | -6.15pp | +0.07pp | 1/3 |
| `clef` | -0.77pp | +7.36pp | -6.53pp | +0.02pp | 1/3 |
| `clef-omni` | -1.21pp | +4.74pp | -4.10pp | -0.19pp | 1/3 |
| `kairos` (Minara) | +1.63pp | +5.27pp | -8.04pp | -0.38pp | **2/3** |
| `subq-snap-1.1` | +2.06pp | +4.94pp | -8.35pp | -0.45pp | **2/3** |
| `jev-latest` | +1.39pp | +3.52pp | -6.59pp | -0.56pp | **2/3** |
| `elephant-1` (Baseten, free) | -1.31pp | +6.35pp | -6.83pp | -0.60pp | 1/3 |
| `clef-flash` | -1.68pp | +3.74pp | -4.79pp | -0.91pp | 1/3 |
| `d1.2-ultra` (Liquid) | -1.62pp | +6.17pp | -7.96pp | -1.14pp | 1/3 |
| `d3-turbo` (free) | -0.59pp | +5.26pp | -8.77pp | -1.36pp | 1/3 |
| `ms-decision-1` | -3.85pp | +7.23pp | -7.49pp | -1.37pp | 1/3 |
| `jev-1.14` | -0.70pp | +3.50pp | -7.87pp | -1.69pp | 1/3 |
| `pplx-decider` | -1.20pp | -0.51pp | -5.39pp | -2.37pp | 0/3 |
| `d1.2` (Liquid) | -1.12pp | +4.44pp | -10.50pp | -2.39pp | 1/3 |
| `drex-max-v1.1` (Nace) | -2.23pp | +6.75pp | -11.79pp | -2.42pp | 1/3 |
| `d3-edge` (free) | -3.47pp | +3.98pp | -7.82pp | -2.44pp | 1/3 |
| `d3-mini` (free) | -2.54pp | +3.12pp | -8.31pp | -2.58pp | 1/3 |
| `standard-one-8b` | -1.44pp | +3.05pp | -11.44pp | -3.28pp | 1/3 |

Each column is the model's gap to SPY in one six-month window (SPY made
+23.04%, +1.88% and +15.09%). The Apr 26 window lifted 20 of 21 models above
SPY because the pool itself outran a flat index there; the other two windows
are the honest test, and there the gaps are negative or inside noise. The
single-window returns live on the dashboard, one chip per window, with the
three-window average as the default lens. Protocol-5 results are never mixed
in: they live on the site's archive page.

## The models

Ten providers are wired natively: OpenAI Decisions (`gpt-6-luna`), TypeSafe
(`jev-latest`), Cloudflare Workers AI (`clef`, `clef-flash`, `clef-omni`),
Requesty (`jev-1.14`, `pplx-decider`, `ms-decision-1`), Standard Thinking
(`standard-one-8b`), Liquid (`d1.2`, `d1.2-ultra`), Nace (`drex-max-v1.1`,
$0.03/M input), vLLM SR (`d3-turbo`, `d3-mini`,
`d3-edge`, all free), Subquadratic (`subq-snap-1.1`), and Minara Cloud
(`kairos`, a Jev-family deployment at $0.042/M input). Respan also answers
first-party: `span-01-free` via api.respan.ai's behavior-scoring endpoint
(free, each noul question mapped to one scored proposition) ran all three
windows at 100% coverage, no daily-cap trouble (`span-02` first-party needs
paid Respan credits). The paid routes (`kev-4b`, `liquid-d1`)
still wait on a zero balance, and `mercury-decide:free` keeps answering 403
upstream. All bill input tokens only. Most answer every question in one call;
`kairos` takes eight questions per request and re-reads the state per question
(about 98,500 tokens per decision), as do the two small d3s, so compare cost
per decision rather than per token. The 21-model v2 board above cost about
**$0.30** for its latest window. `beatspy models` prints every runnable id,
endpoint, and price.

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
