# Architecture

## Fast default

Protocol 6 uses 12 dates spread across the full 2026 evaluation period and one
portfolio manager call per date, without tools. Frozen Yahoo indicators, holdings,
transaction costs, and the momentum reference are supplied directly. Each request
has a 4,096-token default cap, a 90-second timeout, and no automatic retries.
Daily scoring and offline replay still cover the full period. The five-agent
protocol below remains available with `--team` for local experiments.

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
The default continuous benchmark spans 2026 through the latest completed market session
and includes 23 large companies across 11 sectors, with TLT/GLD hedges and cash.
Protocols 4 and 5 use the explicit tradable list in prompts, allocation validation,
execution, and replay. SPY is comparison-only for this benchmark. Default single-call decisions are capped at 12, retaining
both ends of the year-to-date schedule and daily equity scoring through
the cutoff. The same cap is recorded in the scenario and enforced during replay.
External research is disabled for official fast results.
Protocol 5 adds optional direct Olostep search and scraping (`--web-research`),
bounded to two searches and three scrapes per decision. Undated, future-published,
or future-updated page content is withheld before reaching the model. Search
results are navigation only. Accepted text and source metadata are logged in
`events.jsonl`. Live metadata cannot authenticate past content; these experiments
remain outside the official leaderboard. Current fundamentals and unarchived
historical analyst ratings are excluded from protocol 5 agents. Trusted fast runs use prices only. The dashboard shows one latest execution per model, with
no filters. Old runs have been removed from the active catalog.

The static dashboard uses safe text rendering and native SVG charts. A local report is a
copy of that same frontend with a local catalog. See [CONTRIBUTING](CONTRIBUTING.md) for
trust controls, limitations, development, and Vercel deployment.
