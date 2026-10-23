<div align="center">

# BeatSPY

**Decision models, not chat models.**

Seventeen decision models across two protocol versions read the same frozen
point-in-time market state, answer the same typed probability questions, and are
scored twice: were the probabilities calibrated, and did the portfolio they built
beat SPY? Protocol v2 (2026-10-23) adds risk and context to the state: RSI-14,
30-day realized volatility, distance from the 52-week high, a volume ratio, and a
sector label, because under v1 the models mostly echoed the momentum column.

[Live dashboard](https://beatspy.vercel.app) · [Architecture](ARCHITECTURE.md) · [Roadmap](ROADMAP.md) · [Contributing](CONTRIBUTING.md)

[![CI](https://github.com/kingabzpro/beatspy/actions/workflows/ci.yml/badge.svg)](https://github.com/kingabzpro/beatspy/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![v2 15-model run](https://img.shields.io/badge/v2%2015--model%20run-%240.19-brightgreen.svg)](#the-models)

</div>

> [!IMPORTANT]
> **The honest result.** Across all three v2 windows (18 months of weekly
> decisions), the best average gap to SPY is +0.48pp, no Brier skill is
> positive, and only one window of one model reaches a nominal AUC p-value
> under 0.05 (`span-01-free`, latest window, p 0.026 - expected once by chance
> at this many runs, and not significant after correction). Only `jev-latest` and
> `subq-snap-1.1` beat SPY in two of three windows, and both margins sit inside
> their bootstrap intervals. The single-window leader (Clef Omni) dropped to
> fourth once the other windows ran, which is exactly why the board scores
> three. v2 did move discrimination above a coin flip for six models (AUC up
> to 0.519, from 0.39 on the momentum-only state), which is a reason to keep
> testing, not a claim of edge.

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
| `gpt-6-luna` | -0.11pp | +6.48pp | -6.15pp | +0.07pp | 1/3 |
| `clef` | -0.77pp | +7.36pp | -6.53pp | +0.02pp | 1/3 |
| `clef-omni` | -1.21pp | +4.74pp | -4.10pp | -0.19pp | 1/3 |
| `subq-snap-1.1` | +2.06pp | +4.94pp | -8.35pp | -0.45pp | **2/3** |
| `jev-latest` | +1.39pp | +3.52pp | -6.59pp | -0.56pp | **2/3** |
| `clef-flash` | -1.68pp | +3.74pp | -4.79pp | -0.91pp | 1/3 |
| `d3-turbo` (free) | -0.59pp | +5.26pp | -8.77pp | -1.36pp | 1/3 |
| `ms-decision-1` | -3.85pp | +7.23pp | -7.49pp | -1.37pp | 1/3 |
| `jev-1.14` | -0.70pp | +3.50pp | -7.87pp | -1.69pp | 1/3 |
| `pplx-decider` | -1.20pp | -0.51pp | -5.39pp | -2.37pp | 0/3 |
| `d3-edge` (free) | -3.47pp | +3.98pp | -7.82pp | -2.44pp | 1/3 |
| `d3-mini` (free) | -2.54pp | +3.12pp | -8.31pp | -2.58pp | 1/3 |
| `liquid-d1-ultra` | -0.60pp | +2.74pp | -11.43pp | -3.10pp | 1/3 |
| `standard-one-8b` | -1.44pp | +3.05pp | -11.44pp | -3.28pp | 1/3 |

Each column is the model's gap to SPY in one six-month window (SPY made
+23.04%, +1.88% and +15.09%). The Apr 26 window lifted 15 of 16 models above
SPY because the pool itself outran a flat index there; the other two windows
are the honest test, and there the gaps are negative or inside noise. The
single-window returns live on the dashboard, one chip per window, with the
three-window average as the default lens. Protocol-5 results are never mixed
in: they live on the site's archive page.

## The models

Eight providers are wired natively: OpenAI Decisions (`gpt-6-luna`), TypeSafe
(`jev-latest`), Cloudflare Workers AI (`clef`, `clef-flash`, `clef-omni`),
Requesty (`jev-1.14`, `pplx-decider`, `ms-decision-1`), Standard Thinking
(`standard-one-8b`), Liquid (`liquid-d1-ultra`), vLLM SR (`d3-turbo`, `d3-mini`,
`d3-edge`, all free), and Subquadratic (`subq-snap-1.1`). Respan also answers first-party: `span-01-free` via
api.respan.ai's behavior-scoring endpoint (free, each noul question mapped to
one scored proposition) ran all three windows at 100% coverage, no daily-cap
trouble. The OpenRouter `span-01-lite` route remains quota-capped at 50 free
requests a day and has one v2 window; `span-02` first-party needs paid Respan
credits. The paid routes
(`kev-4b`, `liquid-d1`, `drex-v1.5`) still wait on a zero balance, and
`mercury-decide:free` keeps answering 403 upstream. All bill input tokens only,
and one call answers every question, so the full 15-model v2 board above cost
about **$0.19**. `beatspy models` prints every runnable id, endpoint, and price.

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
