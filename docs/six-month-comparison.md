# Six-month, three-version benchmark

One six-month window. Three decision cadences. The same frozen prices, the same
universe, the same accounting. The only thing that changes between versions is
how often a model is allowed to act.

| Version | Scenario | Cadence | Decisions | Portfolio rule |
|---|---|---|---:|---|
| **V1 · Buy once** | `2026-6m-buyhold` | one decision, then hold | 1 | Fund ≥22 individual names at the start and hold them untouched |
| **V2 · Monthly** | `2026-6m-monthly` | monthly | 7 | Rebalance at each month-end |
| **V3 · Weekly** | `2026-6m-weekly` | weekly | 24 | Trade weekly, including gold, oil, and bond hedges |

## What is fixed across all three

- **Window:** `2026-04-06` → `2026-10-02`, scored daily to the cutoff.
- **Capital:** 100,000 simulated dollars, no leverage, no shorting.
- **Costs:** 5 bps fee + 5 bps slippage charged on traded notional each way.
- **Universe:** 24 individual companies across 11 sectors plus `GLD`, `USO`, and
  `TLT` hedges. Identical in every version, drawn from one snapshot
  (`463ccf3104cdb644…`) so SPY and every baseline line reconcile exactly.
- **SPY is comparison-only.** It is data, never an investable ticker, so no
  version can buy the index it is being measured against.
- **Workflow:** the five-agent pipeline (research, analyst, forecaster → critic →
  portfolio manager), three tool calls per agent, 4,096 output tokens.
- **No external research:** no Finnhub, Olostep, or TimeGPT calls. Forecasts use
  the local statistical `baseline` provider.

## What changes per version, and why

The three versions answer different questions, which is the point of running
them side by side:

- **V1 measures selection.** One decision means entry cost is paid once and no
  subsequent trade can rescue or damage the book. To keep this honest the
  manager receives **no reference allocation** — it must originate the portfolio
  from the frozen evidence — and **no performance feedback**, because no prior
  window exists. The 22-name floor is enforced as a recorded violation, so a
  concentrated portfolio is measured rather than silently accepted.
- **V2 measures rebalancing.** Seven decisions capture month-scale shifts at low
  turnover cost.
- **V3 measures active trading and hedging.** Twenty-four decisions pay roughly
  3.4× V2's turnover cost; whether the extra reactivity is worth it is exactly
  what the comparison shows.

Because only cadence differs, a return gap between versions is attributable to
the number and timing of decisions (and their costs), not to a different
universe or a different scoring rule. That is a strong claim, and it depends on
all three versions sharing one snapshot and one window — which they do.

## Hedges are scored on their own

The weekly version lets a model hold gold (`GLD`), oil (`USO`), and bonds
(`TLT`). A hedge that loses money while the portfolio wins is still a losing
hedge, and a sleeve too small to move the portfolio cannot be judged from the
portfolio's return. So `GLD`, `USO`, `TLT`, and `CVX` are additionally scored as
**buy-and-hold market assets beside SPY** and recorded in `metrics.baselines`
and `equity_curve.csv`. Any claim that a hedge helped can be checked against
that asset's own return.

Crypto is deliberately excluded. `BTC-USD` trades on a different calendar than
the NYSE (roughly 200 bars a year against 251 sessions), which would break the
single-calendar assumption the snapshot validator enforces and make
cross-version comparison unreliable.

## Protocols

These scenarios use **protocol 9**, which makes three harness behaviours
scenario-declared instead of keyed to a scenario name:

- `reference_kind` — `momentum`, `core`, or `none` (V1 uses `none`).
- `perf_feedback` — whether realized performance history is supplied.
- `require_min_names` — a diversification floor, recorded as a violation.

Protocol 9 also defers single-call mode to the scenario's declared `pipeline`,
so the multi-agent family can share the protocol number with single-call
scenarios. Protocols 2–8 keep their historical meaning and their exact market
brief bytes, so every archived run still replays byte-for-byte.

## Reproduce

```bash
uv run beatspy run --scenario 2026-6m-buyhold --scenario 2026-6m-monthly --scenario 2026-6m-weekly
uv run beatspy results
uv run beatspy report --open
```

## Limitations

- The window has already been inspected during development, so it is not an
  untouched holdout. Models may also know historical outcomes from training.
- The universe is a fixed surviving basket and therefore has survivorship bias.
- The dated event feed is sparse and curated, not a complete news history.
- These versions are **not leaderboard-eligible**; only `2026-comparison` accepts
  a trusted verification request. `beatspy submit` reports this explicitly.
- Turnover costs differ by design across versions. A higher-return version is
  not automatically a better strategy: check `avg_turnover` and the cost
  columns before drawing that conclusion.