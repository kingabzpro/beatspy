# Architecture

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
prices. `submit` prepares reviewed public artifacts under `dashboard/data/`, redacts known
credentials, deduplicates snapshots, and rebuilds the catalog. Validation proves recorded
score consistency, not model identity. Maintainer attestations are owner-controlled hashes
of separately generated run records. The dashboard shows one board for the selected
window and the latest result per model. Exact protocol, snapshot, tools, budgets, and
model settings stay in each artifact; earlier runs remain immutable. Synthetic data stays separate.

The static dashboard uses safe text rendering and native SVG charts. A local report is a
copy of that same frontend with a local catalog. See [CONTRIBUTING](CONTRIBUTING.md) for
trust controls, limitations, development, and Vercel deployment.
