# Architecture

## Six-month version family

Protocol 9 adds a reproducible **version family**: three scenarios that share one
window, one snapshot, one universe, and one accounting engine, and differ only in
decision cadence. `2026-6m-buyhold` makes a single decision then holds (with no
reference allocation and no performance feedback, so the model originates its own
portfolio); `2026-6m-monthly` rebalances at month-ends; `2026-6m-weekly` trades
weekly and may hold gold, oil, and bond hedges. Because the universe and scoring
rule are held constant, a return difference between versions is attributable to
the number and timing of decisions and their costs, not to a different test.

Protocol 9 replaces three name-keyed behaviours with scenario-declared fields:
`reference_kind` (`momentum`/`core`/`none`), `perf_feedback`, and
`require_min_names` (a diversification floor recorded as a violation, never a
rejected portfolio). It also defers single-call mode to the scenario's declared
`pipeline`; protocols 6–8 keep their historical numeric meaning so archived runs
replay byte-identically. `score_assets` scores non-tradable buy-and-hold assets
beside SPY, so a hedge sleeve's own market return is a replayable number rather
than a claim in a report. Hedges are capped and crypto is excluded: a
non-NYSE-calendar asset would break the single-calendar snapshot invariant.

## Fast default

The default `2026-comparison` uses protocol 5 and 12 dates spread across
July 6–October 2, 2026. It preserves the archived five-agent momentum workflow,
six-company basket, and local statistical forecasts while keeping SPY
comparison-only. Three tool calls per agent, 4,096 output tokens, 90-second
timeouts, and zero automatic retries bound the work. The requested period,
actual holding horizon, and exact frozen data cutoff are recorded in every run.

The optional `2026-ytd` experiment uses protocol 8 and one
portfolio manager call per date, without tools. Frozen Yahoo indicators, holdings,
transaction costs, a fixed six-company stock core reference, and realized performance versus SPY are supplied directly.
Feedback is calculated only through each decision date and replayed from portfolio accounting. Each request
has a 4,096-token default cap, a 90-second timeout, and no automatic retries.
Daily scoring and offline replay still cover the full period. The five-agent
protocol below is the default for the fixed-period comparison.

The core is AAPL, MSFT, NVDA, GOOGL, META, and AMZN, equally weighted within the
position limits. It is also scored separately as a non-model baseline. Its sector
concentration is explicit; models can depart based on dated price evidence.

## Components

BeatSPY has two independent parts: a Python benchmark package in `src/beatspy`, and
static HTML/CSS/JavaScript in `dashboard`. The website reads reviewed artifact files;
it does not run models, store credentials, or require an application server.

## Benchmark flow

Resolve the scenario's concrete dates → prepare an immutable market snapshot → run
research, analyst, and forecasting agents concurrently → critic → portfolio manager →
validate allocations → fill at the next session's open → score through the cutoff.
Trading dates depend on portfolio state and stay sequential. Independent model/scenario
runs share bounded agent/tool concurrency and snapshot preparation, but never budgets,
portfolios, or mutable model settings. Optional network tools reuse a batch HTTP client;
blocking provider/forecast work runs off the event loop. Chronos inference is serialized.

The accounting engine is shared by AI portfolios, equal weight, 60/40 (equity exposure
plus cash), and momentum. Prices are adjusted OHLCV. Source hashes, actual data dates,
settings, errors, complete agent outputs, usage, and the exact snapshot are recorded.
Protocol 3 corrects month-end scheduling and open/close ordering, uses the actual holding
horizon, and records replayable market briefs. Bulk evidence tools keep the entire
universe within budget; unavailable tools are omitted. The manager receives a feasible
momentum reference and a benchmark-relative objective. Protocol 2 scores retain their
original schedule and accounting order through the versioned replay path.

## Data and publication

NYSE session closes resolve recent windows; a 15-minute provider-finalization buffer
excludes unfinished bars. Fresh sessions create content-addressed snapshots under local
`.beatspy-data/snapshots/`; scenario pointers select the latest without changing old files.

`validate` reparses agent outputs and replays decisions offline against the included
prices. `submit --send` requests an approved trusted rerun on main-branch code, using
owner-controlled endpoints and credentials. The runner exports replayed artifacts,
redacts known credentials, deduplicates snapshots, and signs the result with Ed25519.
The pinned public key validates a unique verification ID, artifact bytes, benchmark
version, code/dependency digest, and execution provenance. A separate base-branch
validator checks signatures and replay before accepting new leaderboard records.
The default continuous comparison spans July 6–October 2, 2026 and includes
the original six large companies, with TLT/GLD hedges and cash.
Protocols 4 and 5 use the explicit tradable list in prompts, allocation validation,
execution, and replay. SPY is comparison-only for this benchmark. Decisions are capped at 12, retaining
both ends of the selected schedule and daily equity scoring through
the cutoff. The same cap is recorded in the scenario and enforced during replay.
External research is disabled for official comparison results; local forecasts remain enabled.
Protocol 5 adds optional direct Olostep search and scraping (`--web-research`),
bounded to two searches and three scrapes per decision. Undated, future-published,
or future-updated page content is withheld before reaching the model. Search
results are navigation only. Accepted text and source metadata are logged in
`events.jsonl`. Live metadata cannot authenticate past content; these experiments
remain outside the official leaderboard. Current fundamentals and unarchived
historical analyst ratings are excluded from protocol 5 agents. Trusted comparison runs use Yahoo prices, local forecasts, and the sparse dated event feed. The dashboard shows one latest execution per model, with
no filters. Old runs have been removed from the active catalog.

The static dashboard uses safe text rendering and native SVG charts. A local report is a
copy of that same frontend with a local catalog. The leaderboard keys on model **and
scenario version**, so the three six-month cadences appear as separate rows and no version
is hidden behind another run of the same model. See [CONTRIBUTING](CONTRIBUTING.md) for
trust controls, limitations, development, and Vercel deployment.
