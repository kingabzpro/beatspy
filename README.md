# BeatSPY

**One model. Twelve decisions. Can it beat SPY?**

[Open dashboard →](https://beatspy.vercel.app) · [Contribution guide](CONTRIBUTING.md)

The fast benchmark supplies frozen Yahoo price indicators directly to one portfolio
manager call per decision, including its realized performance versus SPY and a
fixed large-company stock-core candidate. The core is scored separately; models
can choose other stocks and hedges based on dated evidence. Python handles trading, risk rules, and scoring.

## Run a benchmark

Install Python 3.11+ and [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/kingabzpro/beatspy.git
cd beatspy
uv sync
uv run beatspy setup
uv run beatspy run
uv run beatspy results
uv run beatspy report --open
```

The default benchmark is a continuous portfolio from **2026-01-01 through the
latest completed NYSE session** available at the run's release. Weekly decisions
are capped at **12 per model** and span the full year-to-date period. Decisions
use frozen adjusted prices, with extra warmup history excluded from scoring.
23 large, established companies cover all 11 sectors. TLT and GLD are permitted hedges;
SPY is comparison-only and cannot be bought by the model portfolio.
This fixed surviving universe
has survivorship bias; it is not historical index membership.

The default uses cached Yahoo Finance adjusted prices and one model request per
decision, with no Olostep, Finnhub, or TimeGPT calls. Output is limited to 4,096
tokens; requests time out after 90 seconds without automatic retries. Failed
decisions hold the previous portfolio and remain visible in the results.

For the slower five-agent pipeline, use `--team --max-decisions 36`.
Its optional Finnhub news and forecasting integrations remain available for local
experiments. Models may already know historical outcomes; frozen inputs do not
eliminate that limitation.

For a local Olostep comparison, set `OLOSTEP_API_KEY` and run:

```bash
uv run beatspy run --web-research
```

This experimental mode uses Olostep's [Google parser](https://docs.olostep.com/features/structured-content/parsers)
and [page scrapes](https://docs.olostep.com/api-reference/scrapes/create), with no generated web answers.
Each decision allows two searches and three scrapes, with cached responses.
Search snippets are withheld; page text requires a publication date and must not
be published or updated after the decision date. Publisher metadata is not an
authenticated historical archive, so these runs are excluded from the official leaderboard.
Finnhub monthly recommendation trends are analyst sentiment, not weekly forecasts;
they are unavailable for past decisions without an authenticated archive.
Its separate price-target endpoint may require a paid entitlement.

Any OpenAI-compatible chat endpoint works in fast mode; the optional team mode requires tool calling. To avoid setup prompts:

```bash
uv run beatspy setup --model your-model --base-url https://your-endpoint/v1 --api-key-env YOUR_MODEL_KEY
```

`--max-decisions 60` selects 60 decisions across the same period for a local experiment.
`--scenario`, `--model`, `--jobs`, and `--concurrency` remain available for custom
experiments. Historical scenarios and 2025/2026 recent windows remain accessible.
The dashboard has no filters and shows only the latest execution per model, with
each result's period displayed. The active leaderboard contains only new benchmark results.

## Submit in one command

```bash
uv run beatspy submit --send
# Or run and request verification together:
uv run beatspy run --submit
```

Install and authenticate [GitHub CLI](https://cli.github.com/) first. Submission
validates the latest real run and sends a verification request; it never shares
API keys. `beatspy submit` prepares the request locally without sending it.
The dashboard's **Submit for verification** button opens a prefilled request too.

Accepted new leaderboard results come from the trusted main-branch runner using
approved endpoints and owner-controlled credentials. It reruns the model, checks
code integrity, replays scores, signs the exact exported artifacts with Ed25519,
and automatically prepares a results PR. Each accepted run has a unique
verification ID and downloadable signature. Reruns may differ from local scores.
Local runs remain unsigned until the trusted workflow reruns and signs them.

## Development

```bash
uv sync --extra dev --extra browser
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
node --test tests/dashboard.test.cjs
uv build
```

Browser tests require `uv run playwright install chromium`. Ordinary tests use
scripted agents; live forecast checks are opt-in. Chronos and TimeGPT remain
optional extras. [Architecture](ARCHITECTURE.md) · [Apache-2.0](LICENSE)

## Benchmark stock mix

The fixed universe balances sector coverage with large, established companies.
It is not a live ranking or a historical reconstruction of index membership.

| Sector | Companies |
| --- | --- |
| Technology | Apple (AAPL), Microsoft (MSFT), Nvidia (NVDA) |
| Communication services | Alphabet (GOOGL), Meta (META) |
| Consumer discretionary | Amazon (AMZN), Home Depot (HD) |
| Consumer staples | Walmart (WMT), Procter & Gamble (PG) |
| Financials | JPMorgan Chase (JPM), Visa (V) |
| Healthcare | Eli Lilly (LLY), Johnson & Johnson (JNJ) |
| Energy | ExxonMobil (XOM), Chevron (CVX) |
| Industrials | Caterpillar (CAT), GE Aerospace (GE) |
| Utilities | NextEra Energy (NEE), Southern Company (SO) |
| Materials | Linde (LIN), Sherwin-Williams (SHW) |
| Real estate | Prologis (PLD), American Tower (AMT) |

TLT and GLD are hedges; cash is allowed. SPY remains in frozen prices, research,
forecasts, and baseline charts solely for comparison. Protocol 4 separates the
market-data universe from the investable list, drops forbidden allocations with
recorded violations, and rejects forbidden orders at execution and offline replay.
