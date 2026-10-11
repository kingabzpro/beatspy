# Roadmap

Working plan for BeatSPY, derived from the full architecture and results review on
2026-10-11. Items are ordered **from low priority up to very important**: the last
section (P1) is the one that matters most.

If you are skimming: start at **P1 - Very important** at the bottom, then come back
for the sequencing section at the very end.

## Conventions

- Priority buckets: **P4 Low** (polish), **P3 Medium** (hygiene and robustness),
  **P2 High** (measurable improvement levers), **P1 Very important** (benchmark
  validity and credibility).
- Effort: `S` hours, `M` a day or two, `L` a week or more.
- Status markers: `todo`, `doing`, `done`, `dropped`.
- **Evidence rule.** Every item states why it exists, with a file reference or a
  measured number. No speculative work without evidence.
- **Protocol rule.** Any change to the state, the questions, the sizing policy, or
  the label definition increments `DCN_PROTOCOL` and starts a new comparison
  group (`reporting/catalog.py` already hashes protocol into `comparison_group`).
  Never mix protocol versions in one leaderboard, and never tune a change on the
  same window you report it on.

---

## P4 - Low (polish)

### L1. Remove or wire the dead `concurrency` knob
`effort: S` · `status: done`
- **Why:** `DecisionConfig.concurrency` (`src/beatspy/config.py:110`) feeds a
  semaphore in `DcnPipeline` (`src/beatspy/dcn/pipeline.py:203`), but one run asks
  its decisions strictly sequentially (each decision depends on the previous
  book), so the semaphore never blocks. The knob is configurable, documented, and
  does nothing for the DCN path.
- **Change:** either delete it from config, `config.toml` writing, `.env.example`,
  and docs, or repurpose it as a cross-run client-pool limit with a real effect.
- **Done when:** the documented config surface matches observable behavior.

### L2. Delete `.github/benchmark-models.json`
`effort: S` · `status: done`
- **Why:** it is a leftover agent-era registry (gpt-6-luna, mimo, GLM, Kimi,
  DeepSeek) that contradicts `config.py`: it lists gpt-6-luna output at
  `$0.5 / 1M` while the DCN config bills input only (`cost_per_m_output: 0.0`,
  verified 2026-10-08). Two pricing registries will drift.
- **Change:** delete the file; `config.py` `DECISION_MODELS` is the single source
  of pricing.
- **Done when:** `grep -r benchmark-models` finds nothing, and no docs reference it.

### L3. Clean local residue from the deleted spy-core package
`effort: S` · `status: partial`
- **Why:** `src/beatspy/spy_core/__pycache__` holds compiled modules (budget,
  experiment, paper, policy, research, runner, sec, verify) whose sources were
  never committed and are now gone; `.beatspy-data/*.py` helper scripts
  (`replay-spy-core.py`, `resume-spy-core.py`, `summarize-spy-core.py`,
  `focus-gpt-jev.py`) import the missing package; `results/` holds about two dozen
  run directories plus `experiments/` and `research/`. All of it is gitignored, so
  this is local hygiene only.
- **Change:** archive the results directories outside the repository first (the
  last commit already established this convention for superseded runs), then
  remove the orphaned `__pycache__` and helper scripts. Confirm with the owner
  before deleting any data artifacts.
- **Done when:** nothing on disk imports a package that no longer exists.
- **Progress (2026-10-11):** removed the orphaned
  `src/beatspy/spy_core/__pycache__` (compiled modules only, no sources). The
  `results/` directories and `.beatspy-data/*.py` helper scripts are
  deliberately left in place: they are gitignored and irreversible to delete, so
  they wait for an explicit go-ahead.

### L4. Record one full state sample per run
`effort: S` · `status: done`
- **Why:** `decisions.jsonl` stores `state_sha256` and `state_chars` but never the
  state text, so after a run you can prove the input never changed, but you cannot
  read what the model actually saw. External audit and debugging (especially the
  token budget work in V1) both need at least one readable sample.
- **Change:** write the first decision's rendered state to `state-sample.json` in
  the run directory, or store every state gzipped if the size is acceptable.
  Include the hash so the sample itself is verifiable.
- **Done when:** an auditor can read the exact input for at least one decision and
  match it against `state_sha256`.

### L5. Record data library versions in the snapshot manifest
`effort: S` · `status: done`
- **Why:** `data/freeze.py` records `source`, `fetched_at_utc`, and content hashes,
  but not the versions of yfinance, pandas, or exchange-calendars. Adjusted-price
  conventions can shift between yfinance releases, which would silently change a
  re-frozen snapshot.
- **Change:** add a `libraries` block to `MANIFEST.json`, and a test that
  `snapshot_id` is unaffected by metadata-only additions.
- **Done when:** every manifest names the versions that produced it.

### L6. Document the zero risk-free rate
`effort: S` · `status: done`
- **Why:** `BenchConfig.risk_free_annual` defaults to `0.0`, so the reported Sharpe
  is a raw return-over-volatility ratio. That is a legitimate choice, but it is
  not stated in the README scoring section.
- **Change:** one sentence in the README scoring section; the value is already
  recorded per run in `run.json`.
- **Done when:** docs and `settings.bench` agree.

---

## P3 - Medium (hygiene and robustness)

### M1. Fix docs drift
`effort: S` · `status: done`
- **Why:** three known mismatches.
  - `ARCHITECTURE.md:21` says OpenAI uses `POST /v1/responses` with a
    `decision.questions` block. The client posts to `/v1/decisions` with a
    `questions` array and the `predicate` type (commit `aa657cb` fixed the code,
    not the doc).
  - The `dcn/pipeline.py` module docstring describes the legacy `top_n` policy as
    if it were the current default.
  - `README.md:117` calls `miscalibration_gap` "the headline number" while all
    four published runs have it `null` (see V5).
- **Change:** update all three to match the code.
- **Done when:** `ARCHITECTURE.md`, the pipeline docstring, and the README describe
  what `clients.py` and `pipeline.py` actually do.

### M2. Make the scenario schema strict
`effort: S` · `status: done`
- **Why:** `Scenario` silently ignores unknown TOML keys, so four built-in
  scenarios still carry `allow_finnhub` / `allow_web_search` from the deleted
  agent era, including `2026-ytd.toml` with `allow_finnhub = true`. A future typo
  in a scenario key would be ignored just as quietly.
- **Change:** set `extra="forbid"` on the `Scenario` model, strip the vestigial
  keys, and add a test that an unknown key fails with a clear message.
- **Done when:** a misspelled scenario key stops the run instead of silently
  doing nothing.

### M3. Record the full sizing policy in `run.json`
`effort: S` · `status: done`
- **Why:** the policy block (`src/beatspy/dcn/runner.py:237-242`) records
  `selection`, `min_probability`, `top_n`, `min_names`, but not `base_weight`, so
  a graded run's own artifact cannot fully describe how its weights were derived.
- **Change:** add `base_weight` (and any future policy fields) to the block;
  confirm `verify.py` tolerates the added key.
- **Done when:** `run.json` alone explains the weight construction for every
  policy.

### M4. Collapse the CLI duplication and unify the catalog
`effort: M` · `status: todo`
- **Why:** `cli.py` is 1,276 lines with five duplicate implementations:
  `_write_demo_runs` (955) fabricates a run instead of driving the real runner,
  `_read_run_row` (805) duplicates `reporting/catalog.py:114`,
  `_offline_fixture` (429) duplicates `data/freeze.py` manifest and snapshot-id
  creation, `_demo_decision_days` (946) duplicates the schedule, and
  `_local_verify` (741) is a partial `dcn/verify.py`. Separately,
  `catalog.build_catalog` (153) is used only by tests while
  `reporting/dashboard.py:70` scans runs with its own logic, so "valid run" has
  two definitions.
- **Change:** promote the scripted offline client already proven in
  `tests/conftest.py` into `src/`, make `demo` call `run_dcn` with it, have the
  dashboard consume `catalog.summary`, and delete the duplicates.
- **Done when:** `cli.py` shrinks by roughly 300 lines, `beatspy demo` artifacts
  pass `beatspy verify`, and the full test suite passes.

### M5. Add run-level guardrails
`effort: M` · `status: done`
- **Why:** retries are bounded per call (4 attempts), but nothing stops a run from
  grinding against an exhausted quota decision after decision. The archived
  spy-core experiment shows the failure mode: 2,823 HTTP attempts, $3.77 reserved,
  `provider_quota_exhausted`, and a run directory left with only a checkpoint.
- **Change:** add `--max-cost-usd` and a consecutive-failure abort to `run_dcn`;
  record the abort reason and spent estimate in `run.json`.
- **Done when:** a simulated quota exhaustion stops the run after a bounded number
  of decisions and states why in the artifact.
- **Progress (2026-10-11):** `run_dcn` takes `max_cost_usd` and
  `max_consecutive_failures`, both exposed as `beatspy run` flags. An aborted run
  writes no artifacts at all (so a partial result cannot be mistaken for a
  finished one) and the error names the spend, the decision index, and the last
  provider error. Both guardrail values are recorded in `run.json` for completed
  runs. Tests cover the cost abort, the failure abort, and the recorded shape.

### M6. Make the baselines say what they are
`effort: S` · `status: done`
- **Why:** `sixty_forty_weight_fn` (`src/beatspy/engine/backtest.py:245-249`)
  returns `{SPY: 0.6}`, which is 60% equity and 40% cash, not a 60/40 portfolio,
  even though TLT trades in the same universe. The momentum baseline silently
  falls back to 100% SPY when nothing scores, and that fallback is not visible in
  the artifacts.
- **Change:** either implement SPY/TLT as the 60/40 leg or rename the baseline to
  `sixty_cash`; record momentum fallback dates in the metrics or events.
- **Done when:** every baseline name matches its construction, and any fallback is
  visible in the artifacts.
- **Progress (2026-10-11):** `sixty_forty_weight_fn` now holds 60% benchmark and
  40% TLT when TLT is investable (all seven bundled scenarios), falling back to
  60% benchmark plus 40% cash only when it is not. Three engine tests pin the
  equity/bond weights, the no-TLT fallback, and the momentum fallback to 100%
  benchmark on short history, which is also documented in the function. The
  fallback is not surfaced as a metric: adding a field to `RunMetrics` would
  make recorded runs fail replay comparison, so visibility comes from the test
  and docstring instead.

### M7. Measure choice-form order sensitivity
`effort: M` · `status: todo`
- **Why:** the architecture and README assert the `choice` form is reorder
  sensitive ("probabilities that moved when the options were reordered"), citing
  community replication. The repo stores the answer but never tests the claim.
- **Change:** add an optional run flag that shuffles `CHOICE_OPTIONS` order
  deterministically per decision and reports the paired movement (for example mean
  absolute change in the outperform probability) as its own metric.
- **Done when:** one run produces a reorder-delta number with an observation
  count.

### M8. Publish a null-model row
`effort: S` · `status: todo`
- **Why:** the Brier skill reference is an internal constant, and no published row
  anchors the pipeline end to end. A model that always answers 0.5 must produce
  BSS exactly 0 and P&L equal to the equal-weight baseline. If it does not, the
  plumbing has a bug.
- **Change:** add a scripted uniform-0.5 client and publish its run per scenario
  as "null model (0.5)".
- **Done when:** the null row reproduces the equal-weight return and BSS 0 within
  float tolerance.

### M9. Add a `beatspy state` inspector command
`effort: M` · `status: done`
- **Why:** today the only way to see a state is inside `decisions.jsonl` after a
  paid run. Budget work (V1) and debugging both need an offline view.
- **Change:** a command that builds the state for a scenario and date from the
  frozen snapshot, prints the JSON plus the token estimate and per-section sizes,
  and requires no credentials.
- **Done when:** it runs offline and its estimate matches what a real run records.
- **Progress (2026-10-11):** `beatspy state` prints the per-section and total
  character/token cost against the budget, with `--json` for the exact document,
  `--held` to simulate a fully invested book, and `--offline-fixture` for a
  credential-free check. Diagnostics go to stderr so `--json` stays parseable.

### M10. Write the power-analysis note
`effort: M` · `status: done` · `depends on: V3`
- **Why:** the README warns "one window is one draw" but gives no scale. With 25
  weekly clusters, the minimum detectable BSS or excess return is probably much
  larger than the differences between models.
- **Change:** simulate the detectable effect size at 25, 50, and 100 decision
  clusters using the V3 bootstrap machinery, and put the numbers in the docs.
- **Done when:** the docs state the smallest effect the current schedule can
  distinguish from noise.
- **Progress (2026-10-11):** `examples/power_analysis.py` simulates both the
  block bootstrap and both permutation tests. Findings now in the README: at
  ~1% daily volatility a 25-window run detects a 20%/year excess 14% of the time
  (250 windows for 80% power), and at the observed probability spread the
  Brier-skill test is nearly blind (3% power against a perfectly calibrated
  AUC-0.55 model even at 100 windows) while the AUC test finds it 68% of the
  time at 25 windows. Both tests hold their nominal 5% rate under the null. This
  is the evidence for V4: the published single window is a description, not a
  test, and `auc_p_value` is the sensitive statistic to read.

---

## P2 - High (measurable improvement levers)

### H1. Add CI
`effort: S` · `status: done`
- **Why:** there is no `.github/workflows`. The suite is hermetic (217 test
  functions across 12 modules, plus 13 node render tests, no network, no keys),
  so continuous integration is free and would have caught V1 the moment the
  universe changed.
- **Change:** a workflow running `uv sync --extra dev`, `pytest`,
  `ruff check .`, and `node --test tests/dashboard.test.cjs` on push and pull
  request, with a Python version matrix.
- **Done when:** green on `main` and enforced on PRs.

### H2. Give the models a base rate
`effort: M` · `status: todo`
- **Why:** underconfidence is systematic. `clef` answered a mean 0.29 and
  `clef-flash` 0.31 against a realized base rate of 0.45, and expected calibration
  error runs from 0.05 (gpt-6-luna) to 0.24 (clef-flash). The state contains no
  prior at all; nothing tells the model how often this universe beats SPY.
- **Change:** add a point-in-time, trailing-only base rate to the state (for
  example, the share of the universe that beat SPY over the trailing 63
  sessions), identically for every provider. Bump `DCN_PROTOCOL`, re-run 2026-6m,
  and report the ECE and BSS change.
- **Done when:** a protocol-bumped run shows the before/after calibration numbers
  side by side.

### H3. Add a horizon ladder
`effort: M` · `status: todo`
- **Why:** the published benchmark asks 3, 4, 5, or 9 trading day relative
  questions (19 of 25 decisions are 5-day). That is near-noise territory on
  mega-caps, and every model measures at chance. A 21-day question is more
  tractable and lets the benchmark show skill decay with horizon instead of
  assuming one.
- **Change:** ask both a 5-day and a 21-day question per asset in the same call
  (input cost roughly doubles, still cents per run), store observations per
  horizon, and extend the calibration block with a horizon dimension.
- **Done when:** one published run reports per-horizon Brier, BSS, and AUC.

### H4. Score a recalibration layer, raw versus mapped
`effort: M` · `status: todo`
- **Why:** if the miscalibration is mostly scale (H2 suggests it is), a monotone
  map fitted on a development window will improve Brier skill without adding any
  forecasting power, and that is worth measuring explicitly rather than
  conflating. The deleted spy-core research already showed the mechanism works
  (`gpt_shrink` improved development return over raw forecasts).
- **Change:** fit Platt or isotonic calibration on a development window only,
  freeze the map into `run.json`, and report Brier/BSS both raw and recalibrated
  on the untouched evaluation window.
- **Done when:** both numbers are published with the fit window named.

### H5. Decompose the P&L into base and tilt
`effort: S` · `status: todo`
- **Why:** the graded policy is half uniform spread, half tilt
  (`pipeline.py:132-136`). The equal-weight baseline already isolates the active
  half in the comparison table, but the metrics and dashboard never show the
  split, so the reader cannot see that the tilts contributed roughly nothing.
  That split is the actual finding.
- **Change:** add per-run base-half and tilt-half return contributions to
  `metrics.json` and the dashboard.
- **Done when:** every run shows how much of its excess came from the tilts.

### H6. Revive the SPY-core evaluation as a documented series
`effort: L` · `status: done`
- **Why:** the deleted spy-core package ran a SPY-core-plus-satellite design
  (80% SPY minimum, 5% satellite cap, 20-session horizon, magnitude buckets,
  frozen gates) and its archived reports concluded that no model cleared the
  skill gate and simple momentum won the historical window. Those results live
  only in untracked local artifacts; the design deserves to be either reproduced
  in-tree or explicitly retired.
- **Change:** re-implement the SPY-core policy as a scenario-level `policy`
  option with its own protocol version, run it, and publish the outcome either
  way.
- **Done when:** the SPY-core design has a reproducible run or a documented
  decision to drop it.
- **Progress (2026-10-11):** implemented as the `core_satellite` policy (not a
  scenario series): `core_weight` of the book holds the benchmark itself, exempt
  from the per-name cap, and the rest is the cross-sectional sleeve. It is the
  default under `DCN_PROTOCOL = 3`, `--core-weight` adjusts it per run, and the
  verifier reads the recorded policy back. Live jev-latest runs on 2026-6m:
  whole-book graded_cs returned +7.77% against SPY +15.09% (excess -7.33%,
  turnover 23.3%); core 0.85 returned +13.69% (excess -1.40%, CI [-3.65%,
  +0.32%], turnover 5.2%); core 0.95 returned +14.33% (excess -0.76%, CI
  [-2.05%, +0.03%]). The structural cap-weighting gap is now confined to the
  sleeve, which is what the benchmark should measure. Publishing protocol-3 runs
  to the dashboard remains part of V4.

---

## P1 - Very important (validity and credibility)

### V1. Enforce the state token budget
`effort: S` · `status: done`
- **Why:** `ARCHITECTURE.md:51` says the state is budgeted to stay inside
  `DecisionConfig.max_state_tokens` (~1,800), and the test comment says "the
  budget is enforced" (`tests/test_dcn_state.py:93`). It is not enforced anywhere
  at runtime. In the published 2026-6m runs the state grows from 4,259 characters
  (empty book) to 5,826 characters once 25 holdings plus `recent_decisions` are
  included: about 1,942 estimated tokens, over budget and at Cloudflare's ~2K
  truncation threshold. The test only covers `holdings={}`, so it passes. This is
  exactly the like-for-like break the architecture doc says is impossible: Clef
  may be reading a truncated state while the other providers read the full one.
- **Change:** enforce the budget in `build_state` (raise, or trim deterministically
  and record what was trimmed), record the final estimate on each decision, and
  extend the test to a fully invested 25-name book with recent decisions.
- **Done when:** a full-book run stays inside the budget or fails loudly, and the
  test that claims enforcement actually covers the real case.
- **Progress (2026-10-11):** the budget is enforced in the live path by
  `enforce_budget` (called from `DcnPipeline.decide`), which trims the recent
  history first and then market rows the decision does not ask about, failing
  loudly if the state still does not fit. Reproduced the original bug in a test:
  a fully invested 25-name book with three history entries estimates at 1,933
  tokens against the 1,800 budget. `state_trimmed` is recorded per decision in
  `decisions.jsonl`, and the pipeline test proves a full book is sent inside the
  configured budget.

### V2. Make the sizing policy scale-invariant
`effort: M` · `status: done`
- **Why:** tilt is `max(0, p - 0.5)` (`pipeline.py:132-136`), so the portfolio
  depends on each provider's absolute probability scale, not just its ranking.
  Measured across the published runs: probability standard deviation is 0.047
  (gpt-6-luna) versus 0.253 (clef-flash), and only 23% to 34% of answers exceed
  0.5. A provider that ranks perfectly but answers in [0.1, 0.45] gets a fully
  uniform book; the same ranking at a higher scale gets a concentrated tilt book
  that can hit the 35% position cap. The cross-model P&L spread is therefore
  partly a scale artifact.
- **Change:** add a cross-sectional policy (demean or rank probabilities within
  each decision) behind a new `selection` value, keep `graded` for reproducing
  recorded runs, record the policy in `run.json` (M3), then re-run 2026-6m and
  quantify how much of the spread was scale.
- **Done when:** a property test proves the same ranking at different scales
  yields the same weights.
- **Progress (2026-10-11):** `graded_cs` is the new default: the conviction share
  follows each probability's distance above the decision's own mean, which makes
  the book invariant to any positive affine transform of a provider's scale. A
  name the provider did not answer carries no conviction share, preserving the
  never-bet-on-a-backfill invariant. `graded` and `top_n` remain selectable for
  reproducing recorded runs. Property tests cover four scales of the same
  ranking, the below-0.5 case the old policy discarded, and the unanswered-name
  rule. `DCN_PROTOCOL` is bumped to 2 (V1 and V2 both change recorded rules), so
  `beatspy verify` refuses old runs instead of re-scoring them under new rules.
  Still open: the live 2026-6m re-run to quantify how much of the former
  cross-model spread was scale, which needs provider credentials.

### V3. Add the statistical uncertainty layer
`effort: M` · `status: done`
- **Why:** there is not a single confidence interval or p-value in the repo, yet
  the headline claim is "no measurable signal". The 625 observations are clustered
  in about 25 decision dates, so the effective sample is far smaller than it
  looks, and no reported number carries its uncertainty.
- **Change:** date-block bootstrap for the portfolio metrics, a permutation test
  for Brier skill and AUC (shuffle outcomes within dates to build the null), and
  an effective-sample-size note. Surface intervals in `metrics.json`, the README,
  and the dashboard.
- **Done when:** every headline number ships with an interval, and "no signal" is
  a measured p-value rather than an assertion.
- **Progress (2026-10-11):** `dcn/statistics.py` implements the decision-window
  block bootstrap (portfolio return and paired excess over SPY, 95%, 1,000 draws,
  fixed seed) and the within-date permutation test (Brier skill and AUC, two-sided,
  1,000 draws). `metrics.json` carries `return_ci`, `excess_return_ci`, and
  `brier_skill_p_value` / `auc_p_value` per signal; the catalog row flattens them
  so published data carries them too. Replay reproduces every value exactly
  (pinned by a test that rebuilds the report from `decisions.jsonl`), the fast
  rank-based AUC is cross-checked against the Mann-Whitney one, and the README and
  ARCHITECTURE explain the interpretation. Still open: the dashboard does not yet
  draw the intervals or p-values, only the JSON payload carries them.

### V4. Publish the multi-window suite
`effort: M` · `status: todo`
- **Why:** seven built-in scenarios exist (2020-covid, 2022-bear, 2023-recovery,
  2025-recent, 2026-recent, 2026-ytd, 2026-6m) and exactly one is published. The
  README itself warns that a single deterministic momentum rule swings across a
  41-point range between sub-windows, so ranking models on one six-month window
  is ranking noise. The catalog already has `comparison_group` for keeping
  windows separate. Cost is trivial: the README puts a full four-model scenario at
  about $0.07.
- **Change:** run all four models across all seven scenarios, publish a scenario
  by model matrix with a pooled view, and keep per-window detail.
- **Done when:** the dashboard shows the matrix with pooled uncertainty from V3.
- **Progress (2026-10-11):** the single-window leaderboard is refreshed:
  `dashboard/data/` now carries four protocol-3 `core_satellite` runs on 2026-6m
  (2026-04-09 to 2026-10-09), one per model, all `beatspy verify` clean, total
  spend about $0.07. Returns +13.69% to +14.06% against SPY +15.09%, every
  excess interval straddling zero, no calibration skill (AUC p 0.25 to 0.67).
  The dashboard's policy line now describes the core/sleeve split instead of the
  legacy top-N route, and a duplicated sentence in the coverage caption is gone.
  Still open: the other six scenarios and the pooled multi-window view.

### V5. Populate `miscalibration_gap` in the headline runs
`effort: S` · `status: todo`
- **Why:** `README.md:117` and the architecture doc call the noul-versus-choice
  gap the headline number, but all four published `metrics.json` files have
  `miscalibration_gap: null`, because `per_asset` asks only the noul form. The
  claim that the benchmark exists to measure is currently unmeasured in the
  published data.
- **Change:** publish at least one `--form twin` or `--form all` run per model
  (questions per call roughly double; still cents), or soften the docs.
- **Done when:** the headline runs carry a non-null gap with an observation count,
  or the docs stop calling it the headline.

---

## Parking lot (unscored ideas)

Revisit these only after P1 and P2 are in motion; they are not yet backed by
evidence strong enough to schedule.

- Sector and industry labels in the state (point-in-time, static mapping is fine).
- Dispersion and regime features: cross-sectional return dispersion, realized
  correlation, benchmark drawdown state.
- Alternative universes: international ETFs, duration, commodities, so results
  are not one asset class.
- Intraday or VWAP fills; today everything fills at the next session's open.
- A second price source to cross-check adjusted closes.
- Signed or attested submissions again, if third-party runs are ever accepted.
- New decision-model providers as they launch; the adapter surface is one class.
- Abstention scoring: reward or measure a model that declines to answer.
- Doubled-cost sensitivity as a standard table (spy-core recorded this once).
- Factor attribution of the tilt half (size, value, momentum exposures).
- A cost and latency leaderboard alongside calibration.
- Multi-horizon portfolio blending (weights from 5d and 21d answers combined by a
  fixed rule).

## Suggested sequencing

The priority order above is by importance, not execution order. Do it this way:

1. **V1, V2, H1.** Fix the validity bug, decouple sizing from provider scale, and
   stand up CI so nothing regresses. Days of work, and it makes every later
   number trustworthy.
2. **V3, V4, V5.** Add uncertainty, publish the multi-window suite, and make the
   headline claim real. After this the leaderboard is defensible.
3. **H2, H3, H4.** The experiments that could actually move the numbers. One
   variable per protocol version, before/after reported, never fitted on the
   reported window.
4. **M items between experiments**, **L items any time**, **H5 and H6** once the
   measurement layer can tell you whether the portfolio changes did anything.

How to extend this file: keep the ascending order (low at the top, very important
at the bottom), one item per heading, and every item needs a Why with evidence, a
Change, and a Done-when. Log completed items by flipping `status: done` rather
than deleting them, so the history of what was tried stays readable.
