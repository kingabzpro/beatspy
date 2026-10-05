# BeatSPY

**One model. Five trading agents. Can it beat SPY?**

[Open dashboard →](https://beatspy.vercel.app) · [Contribution guide](CONTRIBUTING.md)

Research, analyst, and forecasting agents work in parallel, then a critic and
portfolio manager make the decision. Python handles trading, risk rules, and scoring.

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
are capped at **36 per model** and span the full year-to-date period. Decisions
use frozen adjusted prices, with extra warmup history excluded from scoring.
23 stocks cover all 11 sectors; TLT, GLD, and SPY add bonds, gold, and the baseline.
This fixed surviving universe
has survivorship bias; it is not historical index membership.

Finnhub financials/news activate when their key is available. Olostep is disabled
for the default benchmark. TimeGPT forecasting is optional: install with
`uv sync --extra timegpt`, configure `NIXTLA_API_KEY`, and select `timegpt` during setup
or set `BEATSPY_FORECAST_PROVIDER=timegpt`. Use `FINNHUB_API_KEY`, the `BEATSPY_` equivalents,
or enter them during setup. Secrets stay in environment variables or
`~/.beatspy/secrets.env`. The dashboard contains no credentials.
Live fundamentals can include later information; the runner records
that risk. Date-constrained news is not a complete historical archive, and models
may already know historical outcomes.

Any OpenAI-compatible endpoint with tool calling works. To avoid setup prompts:

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
