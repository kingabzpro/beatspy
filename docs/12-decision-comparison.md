# Twelve decisions over the archived 2026 period

The benchmark covers **July 6 through October 2, 2026**, the same scored period
as the original four-decision archive. Twelve dates are selected from the weekly
schedule, retaining its first and last executable dates. Daily portfolio scoring
continues through October 2. This is a fixed historical comparison, not 2026
year-to-date or a rolling latest-release window.

The original basket is AAPL, MSFT, NVDA, META, JPM, and XOM, plus TLT/GLD hedges
and cash. The exact archived Yahoo snapshot is reused:
`29c6cb831a75ad36c1d29541696b319d63e8b20eb911432f8a307b1f474f0e95`.
Warmup data begins June 2, 2025 and is excluded from scoring.

The five-agent workflow is preserved: research, analyst, and forecaster run in
parallel, then the critic reviews their reports, then the portfolio manager
chooses the allocation. The manager receives a feasible 12-minus-1-month
momentum candidate, not the later fixed-stock-core reference or performance
feedback. Forecasts use the local statistical `baseline` provider, batched once
for the universe and scaled to each actual holding horizon. The original sparse,
dated event feed is reused. No Finnhub, TimeGPT, Olostep, or external web tools
are used.

Speed controls cap each agent at three tool calls, eight turns, and 4,096 output
tokens, with a 90-second request timeout and no automatic retries. Independent
models and the first three agents run concurrently; a portfolio's decisions
stay sequential. Twelve decisions involve substantially more than twelve model
requests. Models retain control of their stock allocations, hedges, and cash.

SPY is comparison-only under current protocol-5 eligibility rules. The original
protocol-3 archive accidentally permitted it. This correction, output/tool
limits, and model reasoning settings mean the comparison is not an isolated
test of decision count. No old score has been relabeled as a new result.

Every result retains native responses, errors, usage, trades, exact prices,
source revision, version, and hashes. The harness is frozen at `1efe431`,
version 0.2.6. All completed results are replay-checked before publication;
they remain unsigned until the owner's signing key is configured.

## Reproduce

```bash
uv run beatspy run
# Equivalent explicit profile:
uv run beatspy run --scenario 2026-comparison
uv run beatspy results
```

The default prints its exact evaluation timeline and decision count on completion.
`--scenario 2026-ytd` remains the separate full-year, single-call experiment.

The period has already been inspected and used for development. Historical
outperformance here is not an untouched holdout, independent evidence of future
outperformance, or proof that models did not know historical outcomes from training.

## Completed six-model batch

SPY returned **2.70%**. **5 of 6** models exceeded it; four of the five models
with an archived four-decision counterpart improved their recorded return.

| Model | Archived 4-decision return | New 12-decision return | Runtime | Recorded cost estimate | Invalid portfolios |
|---|---:|---:|---:|---:|---:|
| Kimi-K3 | — | 7.50% | 7.7 min | $1.0224 | 0 |
| mimo-v2.6-pro | 3.26% | 6.35% | 14.9 min | $0.1660 | 0 |
| GLM-5.3 | 4.39% | 5.55% | 2.9 min | $0.3173 | 0 |
| DeepSeek-V4.1-Flash | 2.64% | 4.36% | 4.1 min | $0.1722 | 0 |
| gpt-6-luna | 3.65% | 3.98% | 5.3 min | $0.0521 | 1 |
| GLM-5.3-Flash | 3.05% | 1.94% | 5.8 min | $0.0707 | 0 |

Total recorded native token estimate: **$1.8007**. These use the OpenRouter
uncached input/output rates checked on October 5, 2026 and the native provider
usage returned for this batch. They are not invoices and exclude probes, earlier
experiments, cached-input discounts, subscriptions, and hosting costs.

GPT returned one malformed portfolio response on July 6, retaining cash until
the next decision. One critic report on September 4 lacked required fields.
No provider exceptions were recorded in this batch. Errors remain in the
artifacts; no model was rerun to conceal a failure or select a higher score.

The same-accounting non-model baselines still outperform every model:
**equal weight 11.27%**, **12-minus-1 momentum 12.62%**. Beating SPY here does
not establish that the models add value over these simpler strategies.

All six runs used source revision `1efe431`. After the batch completed, the
CLI defaults were aligned with its already configured GLM low-reasoning and
MiMo disabled-thinking settings; this did not change the completed artifacts.
