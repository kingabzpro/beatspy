# Contributing

## Run the benchmark

1. Clone the repo and run `uv sync --extra dev` (Python 3.11+ required).
2. Run `uv run beatspy models` to see the decision models and which keys resolve.
3. Run `uv run beatspy setup` to pick a model and store its key, or export the key
   yourself (`OPENAI_API_KEY`, `TYPESAFE_API_KEY`, or `CLOUDFLARE_ACCOUNT_ID` plus
   `CLOUDFLARE_AUTH_TOKEN`). Each also has a `BEATSPY_`-prefixed form that wins.
4. Run `uv run beatspy run --model <id>`.
5. Inspect the result with `uv run beatspy calibrate <run-id>`, `uv run beatspy
   compare`, or `uv run beatspy report --open`.

Add `--probe` to `beatspy doctor` to make one live call per model and see exactly
what each provider returns, including its question limits and latency. Without
`--probe`, `doctor` makes no network calls at all.

## What a contribution should not do

- **Do not tune the sizing policy to a result.** `min_probability`, `top_n` and
  `min_names` live in `DecisionConfig` and are fixed before a run. If you change
  them, say so in the PR and expect every previously recorded number to be
  incomparable. `DCN_PROTOCOL` in `src/beatspy/dcn/__init__.py` exists so a run
  records the rules it executed under; bump it when the question set, the state
  layout, or the sizing policy changes.
- **Do not put an inference into a scored number.** Unanswered questions are
  backfilled at the neutral 0.5 so the portfolio can be sized, but calibration
  reads only `answered_noul` / `answered_choice` / `answered_rank`. If you add a
  scoring path, keep that separation.
- **Do not reach forward from state construction.** Only `DataService.close_on_offset`
  may look past the decision date, and only for labeling. A test builds the same
  decision from two price series that diverge after the decision date and requires
  identical state — keep it passing.
- **Do not let the state grow.** Workers AI truncates long text state at roughly
  2K tokens, which would give Clef a different question from everyone else. The
  state budget is enforced by tests for both a small and a 23-name universe.
- **Do not ship a fake provider in `src/`.** The scripted offline client lives in
  `tests/conftest.py` on purpose.

## Develop

```bash
uv sync --extra dev
uv run --no-sync pytest
uv run --no-sync ruff check .
node --test tests/dashboard.test.cjs
```

`src/beatspy/cli.py` is a large hand-rolled argparse module; wrap help strings
rather than raising the line limit. Commit `uv.lock` with dependency changes.

### Tests must stay hermetic

No network, no credentials, no real models, and no live price downloads. Build
runs with the fixtures in `tests/conftest.py`:

- `scenario`, `data_service`, `settings` — a dated scenario over synthetic prices;
- `frozen_data` — a real content-addressed snapshot on disk (valid
  `prices_sha256`, row count and coverage) so a run copies it exactly as a live
  run would, without the network;
- `offline_client` — a deterministic decision model that answers all three forms;
- `dcn_run_dir` — a complete run produced by the real runner, for report,
  catalog and verification tests;
- `run_offline(...)` — the helper behind them.

Calibration constants are verified against hand-computed values, so a change to
`dcn/calibration.py` that silently shifts Brier, log loss, AUC or the reliability
bins fails immediately.

### Adding a provider

1. Add an entry to `DECISION_MODELS` in `config.py` with `provider`, `model`,
   `label`, `base_url`, `api_key_env`, `cost_per_m_input`, `cost_per_m_output`,
   `context_window`, `max_questions`, and a `state_note` if the transport has a
   quirk worth recording.
2. Subclass `HttpDcnClient` in `dcn/clients.py`: implement `build_request` and
   `parse_response`, and register it in `_CLIENTS`.
3. Add a contract test asserting on the exact request body and the exact response
   envelope, plus a retry test for a transient status.
4. Cite the pricing source in the PR. Never guess a price; an unpriced model must
   report no cost rather than a made-up one.

## Limitations to state plainly

- Prices are real adjusted OHLCV, but the universe is a fixed set of large
  companies chosen after the fact. It has survivorship bias and is **not**
  point-in-time index membership.
- `2026-ytd` and the recent windows end at the latest completed session, so they
  are frozen at snapshot time and do not move.
- A single window is one draw: one deterministic momentum rule moves across a
  ~41-point range between sub-windows of a six-month period. Report dispersion.
- Workers AI's state truncation means Clef may read less of the state than the
  other providers if the budget is ever exceeded. The budget is enforced, but the
  caveat belongs in any comparison.
- A decision model may already know historical outcomes from pretraining. Frozen
  prices prevent look-ahead *in the harness*; they cannot prevent it in the
  weights.

## Dashboard

The static frontend in `dashboard/` reads `data/index.json`. It shows the latest
execution per decision model and never selects the best score. Details include
the equity curve with baselines and a reliability curve per run, with the
noul/choice/rank curves and the miscalibration gap. Never render data as HTML;
the frontend already escapes it.

Preview with `python -m http.server 8000 --bind 127.0.0.1 --directory dashboard`.
`beatspy report --open` serves a local copy separately. Branding lives in
`dashboard/assets/`; keep image URLs, dimensions and alternate text synchronized.
Private credentials must never be deployed.