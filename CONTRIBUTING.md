# Contributing

## Development

Use Python 3.11+ and `uv sync --extra dev`. Run `uv run --extra dev pytest`, `uv run --extra dev ruff check .`,
`uv run --extra dev ruff format --check .`, `node --test tests/dashboard.test.cjs`, and `uv build`.
Commit `uv.lock` when dependencies change. Unit tests must not call model or data APIs.

For browser integration checks: `uv sync --extra dev --extra browser`, then
`uv run --extra dev --extra browser playwright install chromium` and `uv run --extra dev --extra browser pytest tests/test_browser.py`.
Live forecast checks require `BEATSPY_LIVE_TESTS=1`, the optional package, credentials,
and suitable local frozen prices; they are excluded from ordinary runs.

## Submit a result

1. Fork and clone the repository. Configure your endpoint with `beatspy setup`.
2. Run `uv run beatspy run --scenario 2026-recent` (or another bundled scenario).
3. Validate `uv run beatspy validate --run results/<run-id>`.
4. Prepare `uv run beatspy submit --run results/<run-id>`.
5. Review and commit the generated files in `dashboard/data/`, then open a PR.

Include the model/provider, exact command, optional tools, and any integrity caveats.
The command prepares files locally; it never sends a PR or credentials for you.
Review free-text outputs before sharing them; redact personal content and rerun validation.

Submission includes `run.json`, `metrics.json`, `equity_curve.csv`, `trades.csv`,
`decisions.jsonl`, `events.jsonl`, and the exact snapshot's `prices.csv` and `MANIFEST.json`.
Snapshots are deduplicated by hash. Existing published run directories are immutable.
Git preserves exported artifact bytes across platforms; do not normalize their line endings.
Legacy local runs remain viewable but must be rerun to produce protocol 2 evidence.
Synthetic demos cannot be submitted. Individual artifacts are limited to 25 MB.

### What validation establishes

The trusted validator checks hashes, ticker coverage, finite values, dates, the decision
schedule, and complete five-agent outputs. It reparses model outputs, reapplies risk rules,
replays next-session orders, and recomputes equity, trades, baselines, and all metrics.
A self-consistent fabricated transcript can still pass: this does **not** authenticate
provider/model identity, model requests, or token usage.

Submitted PR data is parsed using base-branch code, without secrets or executing submitted
files. The catalog is recomputed; submitted trust labels are not authoritative. Protect
`.github/`, the validator/catalog code, and the maintainer registry with required CODEOWNER
review in GitHub branch protection. Repository settings must be enabled by the owner.

### Maintainer runs

Community results stay labeled **Community submitted** after review and merge.
To obtain a **Maintainer run**, the owner runs the claimed model separately using trusted
code and controlled endpoint credentials, then exports that new result. The owner adds
its exported `run.json` SHA-256 under its run ID in `dashboard/data/maintainer-runs.json`.
Only an owner-authored PR from this repository may modify that registry. Rebuild the
catalog using `build_catalog(Path('dashboard/data'))` from `beatspy.reporting.catalog`.
Changing a registered artifact removes its designation until the owner attests again.
A different rerun score is normal for nondeterministic models and is not proof of fraud.

## Recent windows and reproducibility

`2026-recent` resolves a rolling 90-calendar-day window using completed NYSE sessions;
`2025-recent` ends at the final completed session of 2025. NYSE holidays and early closes
are handled by `exchange-calendars`; daily bars become eligible 15 minutes after close.
Older scenarios use explicit dates. `--start`, `--end`, and `--freq` override the window.

Prices include extra warmup history, excluded from scoring. Missing required tickers or
late provider data fail clearly. No automatic stale-data fallback is used. Once another
session completes, new recent runs download a fresh snapshot. Within the same cutoff,
the existing snapshot is reused; `--refresh-data` explicitly checks revised prices.
Both versions remain immutable. Legacy files are preserved. Fetch time is metadata,
not part of the snapshot identity. Runs retain an exact local copy of the snapshot.

`--jobs` limits independent runs; `--concurrency` separately limits active agents and
external tools across the batch. Defaults are 2 and 6. Lower them for provider limits.
The three initial agents overlap, then critic and manager run in order. Portfolio dates
stay sequential. Timeouts and bounded retries prevent indefinite requests. Completed
runs survive another run's failure. Seeded Chronos inference is serialized.

Recent event feeds are sparse and source-linked, not comprehensive news archives.
Their latest event date is displayed. Optional current fundamentals and web search are
flagged for look-ahead risk; they belong to separate comparison groups. Historical model
knowledge and adjusted prices remain limitations even when tools obey date restrictions.

## Add a scenario or tool

Scenarios are TOML files in `src/beatspy/scenarios/builtin/` or `~/.beatspy/scenarios/`.
Use fixed `start`/`end`, or `year` and `window_days`; include universe, costs, and limits.
Add a dated `<name>.events.toml` with source URLs. Test it with scripted agents.
Tools must charge the correct agent budget and clamp data to the harness's `as_of` date.
Heavy forecasting packages remain optional. Metrics must be deterministic and have a
hand-calculated or regression check. Never add an LLM judge.

## Dashboard and Vercel

The live dashboard is https://beatspy.vercel.app. The frontend lives in `dashboard/`
and reads `data/index.json`; details load on demand.
No backend, API keys, framework, package installation, or build step is needed.

Preview the public site:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory dashboard
```

`uv run beatspy report --open` prepares and serves local results instead. Local runs stay
unverified; synthetic runs have their own label. Nothing local is published automatically.

In Vercel, import the repository, set Root Directory to `dashboard`, Framework Preset to
**Other**, and leave Build/Install Commands blank. `dashboard/vercel.json` includes static
security headers. Deployments follow approved merges on the production branch. This
project is connected to the GitHub repository; production updates follow pushes to `main`.
PR preview deployments are disabled; only production publishes approved changes.
Do not serve the whole repository or expose credentials. Public results are explicitly
exported and reviewed before publication. The initial maintainer comparison uses GLM-5.3,
GLM-5.3-Flash, and DeepSeek-V4.1-Flash with monthly decisions, temperature 0, baseline
forecasts, tool budget 10 per agent, and `BEATSPY_MAX_TURNS=16` for every model. Exact
settings, dates, endpoint names, and request telemetry are included in each run record.
