# Changelog

## 0.2.6

- Default to the archived five-agent momentum workflow with 12 decisions over July 6–October 2, 2026.
- Reuse the original six-company basket, local batch forecasts, and dated event feed; SPY stays comparison-only.
- Bound agent output, tool calls, timeouts, and retries; keep the full-year single-call experiment as an optional scenario.
- Show the fixed timeline in benchmark output, documentation, and the dashboard.

## 0.2.1

- Protocol 4 separates tradable assets from benchmark data and blocks non-tradable orders.
- SPY is comparison-only; the stock mix uses 23 large companies across 11 sectors, plus TLT/GLD hedges.
- Remove trades, decisions, and per-decision token detail from the website.
- Retire results generated under the rule that accidentally allowed SPY allocations.

## 0.2.0

- 2026 year-to-date default: 2026 through the latest completed session, with 23 stocks across 11 sectors.
- Latest result per model, no dashboard filters or yearly averages, and aligned branding.
- The 2026-ytd benchmark uses weekly decisions capped at 36 per model, Finnhub where available, and no Olostep.
- Optional TimeGPT forecasts and OpenRouter reference token prices; the active catalog has been reset for new runs.
- Simple setup, run, results, and submit commands, plus one-command verification requests.
- Trusted main-branch reruns, code/dependency integrity checks, replay, Ed25519 signatures,
  unique verification IDs, and automated result PRs. Existing results remain unsigned.

## Unreleased

- Separate static dashboard and Python benchmark CLI.
- Recent 90-day scenarios for 2025 and 2026; completed-session data with immutable snapshots.
- Bounded parallel runs, agents, and external tool requests.
- Offline artifact replay and community/maintainer result designations.
- Correct evaluation cutoff: trailing downloaded days no longer affect scores.
- Correct Chronos quantiles to use all sampled paths at the forecast's final step.
- Simplified setup, documentation, dependencies, and repository layout.
