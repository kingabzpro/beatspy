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
> **The honest result.** Under protocol v2, in the published window no model beat
> SPY (+15.09%), no Brier skill is positive, and no AUC p-value reaches
> significance (the closest sits at 0.10). The equal-weight pool made +9.09%, so
> most of every model's return is the pool's beta, not forecasting. v2 moved
> several models' discrimination above a coin flip (AUC up to 0.519, from 0.39
> under the momentum-only form comparison), which is a reason to keep testing,
> not a claim of edge.

## Latest benchmark

Rolling six months to 2026-10-09, weekly decisions, frozen adjusted prices. The
pool is 25 large companies across all 11 sectors plus long Treasuries; the book
spreads across them by conviction, and SPY is the yardstick, never held.

| Decision model | Return | vs SPY |
| --- | --- | --- |
| `clef-omni` | **+10.99%** | −4.10% |
| `clef-flash` | +10.30% | −4.79% |
| `pplx-decider` (Requesty) | +9.70% | −5.39% |
| `gpt-6-luna` | +8.94% | −6.15% |
| `clef` | +8.56% | −6.53% |
| `jev-latest` | +8.50% | −6.59% |
| `clef` via Requesty | +8.17% | −6.92% |
| `ms-decision-1` (Requesty) | +7.60% | −7.49% |
| `d3-edge` (free) | +7.27% | −7.82% |
| `jev-1.14` (Requesty) | +7.22% | −7.87% |
| `d3-mini` (free) | +6.78% | −8.31% |
| `subq-snap-1.1` | +6.74% | −8.35% |
| `d3-turbo` (free) | +6.32% | −8.77% |
| `liquid-d1-ultra` | +3.66% | −11.43% |
| `standard-one-8b` | +3.65% | −11.44% |
| *pool, equal weight* | *+9.09%* | *−6.00%* |
| *SPY buy-and-hold* | *+15.09%* | *n/a* |

The spread between models (+3.65% to +10.99%) straddles the pool's +9.09%, and
the return ordering still does not track probability quality. The SPY gap is
structural: SPY is cap-weighted and this book is not. Four models that ran v1
through OpenRouter (`span-01-lite`, `kev-4b`, `liquid-d1`, `drex-v1.5`) wait on a
zero aggregator balance and are absent from the v2 board; their v1 windows stay
published.

## The models

Eight providers are wired natively: OpenAI Decisions (`gpt-6-luna`), TypeSafe
(`jev-latest`), Cloudflare Workers AI (`clef`, `clef-flash`, `clef-omni`),
Requesty (`jev-1.14`, `pplx-decider`, `ms-decision-1`), Standard Thinking
(`standard-one-8b`), Liquid (`liquid-d1-ultra`), vLLM SR (`d3-turbo`, `d3-mini`,
`d3-edge`, all free), and Subquadratic (`subq-snap-1.1`). Seven more routes run
through OpenRouter while its balance is zero: `span-01-lite` (free), `kev-4b`,
`liquid-d1`, `pplx-decider`, `clef-omni`, `drex-v1.5`, and `ms-decision-1`. All
bill input tokens only, and one call answers every question, so the full 15-model
v2 board above cost about **$0.19**. `beatspy models` prints every runnable id,
endpoint, and price.

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
