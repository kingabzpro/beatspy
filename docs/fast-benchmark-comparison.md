# Fast benchmark comparison

The latest six runs use 12 decisions, one portfolio-manager call per decision,
frozen Yahoo prices, and no Olostep, Finnhub, or TimeGPT calls. The manager receives
past realized portfolio performance versus SPY, with a prompt focused on net relative
returns and evidence for hedges. A fixed six-company core replaces the lagging momentum anchor and is also scored separately.
GLM models use low reasoning effort; MiMo thinking is disabled; GPT's native
completion-token limit is mapped correctly. Outputs are capped at 4,096 tokens.

All runs cover 2026-01-01 through 2026-10-02, the latest completed session when
frozen, using the same 26-ticker snapshot as the prior 36-decision batch.
23 stocks cover 11 sectors; TLT/GLD and cash are available. SPY is comparison-only.
The default changes decision count, prompt, pipeline, and some model reasoning
settings, so this is a workflow comparison rather than an isolated provider test.

| Model | Previous 36-decision return | Latest 12-decision return | Previous runtime | Latest runtime | Latest estimated cost | Invalid outputs |
|---|---:|---:|---:|---:|---:|---:|
| gpt-6-luna | 6.79% | 1.54% | 27.3 min | 43.0 sec | $0.0064 | 0 |
| mimo-v2.6-pro | 6.55% | -4.11% | 199.0 min | 80.7 sec | $0.0299 | 0 |
| GLM-5.3 | 8.19% | 3.05% | 176.6 min | 41.9 sec | $0.0401 | 0 |
| GLM-5.3-Flash | 3.05% | -0.49% | 228.9 min | 165.4 sec | $0.0091 | 1 |
| DeepSeek-V4.1-Flash | 7.59% | 1.46% | 14.3 min | 23.5 sec | $0.0196 | 0 |
| Kimi-K3 | 2.99% | -0.63% | 30.1 min | 98.8 sec | $0.1443 | 0 |

SPY buy-and-hold returned **13.54%**. **0 of 6** latest models exceeded it.
Previous batch token estimate: **$27.71**; latest batch: **$0.2494**.
Costs use the configured OpenRouter token rates with native providers. They exclude
probe calls, stopped trials, provider cache discounts, and subscription billing.
Runtime includes cached snapshot loading and scoring, excluding the subsequent offline replay.

This is a development comparison on an already inspected historical period, not an
independent holdout or a promise of future returns. Alternative price-only references were checked locally and not adopted: recent
strength 2.25%, equal stocks 11.18%, trend stocks 4.48%, pullback stocks 4.35%,
and past-window selection 1.43%. The fixed six-company core returned 16.67%.
This core was chosen during development after inspecting the test period; the
comparison therefore carries strategy-selection bias and needs a fresh holdout.
No results were selected by best return. The dashboard shows each model's latest run.

Every latest run passed artifact hashing and offline accounting replay. They remain
**unsigned** because the owner's existing signing key has not been configured.

GLM-5.3-Flash had one provider timeout. That decision retained its prior portfolio.
Its cost estimate covers returned token usage only; any provider charge for the
timed-out request is unknown and is not treated as a known zero.

## Earlier 12-call development batch

The earlier protocol-7 batch used the momentum anchor and performance feedback,
with MiMo thinking still enabled. It also had zero models exceeding SPY.

| Model | Protocol-7 return | Latest protocol-8 return |
|---|---:|---:|
| gpt-6-luna | 0.74% | 1.54% |
| mimo-v2.6-pro | 2.62% | -4.11% |
| GLM-5.3 | -4.11% | 3.05% |
| GLM-5.3-Flash | -7.63% | -0.49% |
| DeepSeek-V4.1-Flash | -1.96% | 1.46% |
| Kimi-K3 | -1.09% | -0.63% |
