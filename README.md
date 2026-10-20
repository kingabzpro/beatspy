<div align="center">

# BeatSPY

**Decision models, not chat models.**

Thirteen decision models read the same frozen point-in-time market state, answer
the same typed probability questions, and are scored twice: were the
probabilities calibrated, and did the portfolio they built beat SPY?

[Live dashboard](https://beatspy.vercel.app) · [Architecture](ARCHITECTURE.md) · [Roadmap](ROADMAP.md) · [Contributing](CONTRIBUTING.md)

[![CI](https://github.com/kingabzpro/beatspy/actions/workflows/ci.yml/badge.svg)](https://github.com/kingabzpro/beatspy/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Full 13-model run](https://img.shields.io/badge/full%2013--model%20run-%240.23-brightgreen.svg)](#the-models)

</div>

> [!IMPORTANT]
> **The honest result.** In the published window no model beat SPY (+15.09%), and
> no model's probabilities show measurable skill: every Brier skill is negative
> and no AUC p-value reaches significance (the closest, `span-01-lite`, sits at
> 0.08). The equal-weight pool made +9.09%, so most of every model's return is
> the pool's beta, not forecasting.

## Latest benchmark

Rolling six months to 2026-10-09, weekly decisions, frozen adjusted prices. The
pool is 25 large companies across all 11 sectors plus long Treasuries; the book
spreads across them by conviction, and SPY is the yardstick, never held.

| Decision model | Return | vs SPY |
| --- | --- | --- |
| `span-01-lite` (free) | **+11.67%** | −3.42% |
| `kev-4b` | +10.54% | −4.55% |
| `gpt-6-luna` | +9.84% | −5.25% |
| `clef` | +9.82% | −5.27% |
| `clef-flash` | +9.78% | −5.31% |
| `jev-latest` | +8.29% | −6.80% |
| `liquid-d1` | +8.23% | −6.86% |
| `pplx-decider` | +8.16% | −6.93% |
| `clef-omni` | +7.72% | −7.37% |
| `liquid-d1-ultra` | +7.70% | −7.39% |
| `drex-v1.5` | +7.08% | −8.01% |
| `ms-decision-1` | +5.96% | −9.13% |
| `standard-one-8b` | +4.38% | −10.71% |
| *pool, equal weight* | *+9.09%* | *−6.00%* |
| *SPY buy-and-hold* | *+15.09%* | *n/a* |

The spread between models (+4.38% to +11.67%) straddles the pool's +9.09%, and
the return ordering does not track probability quality. The SPY gap is
structural: SPY is cap-weighted and this book is not. Three more models (both
Upstage Solar builds and Together Tev1) stalled on a zero aggregator balance and
are off the board.

## The models

Six providers are wired natively: OpenAI Decisions (`gpt-6-luna`), TypeSafe
(`jev-latest`), Cloudflare Workers AI (`clef`, `clef-flash`), Requesty
(`jev-1.14`), Standard Thinking (`standard-one-8b`), and Liquid
(`liquid-d1-ultra`). Seven more run through OpenRouter: `span-01-lite` (free),
`kev-4b`, `liquid-d1`, `pplx-decider`, `clef-omni`, `drex-v1.5`, and
`ms-decision-1`. All bill input tokens only, and
one call answers every question, so the full 13-model benchmark above cost about
**$0.23**. `beatspy models` prints every runnable id, endpoint, and price.

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
