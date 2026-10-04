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
Protocol 2 ends scoring at the requested cutoff; old results may have included trailing
downloaded days and therefore do not share a comparison group with new runs.

## Data and publication

NYSE session closes resolve recent windows; a 15-minute provider-finalization buffer
excludes unfinished bars. Fresh sessions create content-addressed snapshots under local
`.beatspy-data/snapshots/`; scenario pointers select the latest without changing old files.

`validate` reparses agent outputs and replays decisions offline against the included
prices. `submit` prepares reviewed public artifacts under `dashboard/data/`, redacts known
credentials, deduplicates snapshots, and rebuilds the catalog. Validation proves recorded
score consistency, not model identity. Maintainer attestations are owner-controlled hashes
of separately generated run records. Comparison groups include dates, snapshot, protocol,
tools, budgets, model settings, and execution rules. Synthetic data stays separate.

The static dashboard uses safe text rendering and native SVG charts. A local report is a
copy of that same frontend with a local catalog. See [CONTRIBUTING](CONTRIBUTING.md) for
trust controls, limitations, development, and Vercel deployment.
