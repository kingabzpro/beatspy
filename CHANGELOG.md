# Changelog

## 0.3.0 — decision-model benchmark

**Breaking: the five-agent pipeline is gone.** This release replaces the chat-model
agent benchmark with a benchmark for decision models. Nothing from the agent
harness remains: no agents, no tool calling, no prompts, no news or web-research
integrations, no forecast providers, and no signed-submission path. `main` and
`main-backup-2026-10-06` still carry them.

Added

- Four decision models behind one contract: OpenAI Decisions (`gpt-6-luna`),
  TypeSafe Jev (`jev-latest`), Cloudflare Clef and Clef-flash. All bill input
  tokens only; pricing recorded in the model registry and cited in the README.
- A point-in-time state builder that serves only data dated on or before the
  decision, and stays inside a ~1,800-token budget because Workers AI silently
  truncates long state at roughly 2K tokens.
- Three question forms — `noul`, `choice`, `rank` — with a `twin` mode that asks
  the first two of the same ticker in one call, so the form itself can be measured.
- Calibration scoring: Brier, Brier skill, log loss, tie-aware AUC, accuracy, and
  10-bin reliability with expected and maximum calibration error, plus
  `miscalibration_gap` between the noul and choice answers.
- Labeling that is *benchmark-relative* — did the ticker beat the benchmark over
  the same horizon — so a rising market cannot mark every prediction correct.
- A fixed sizing policy: only noul probabilities, a 0.5 threshold, top three
  equal-weighted, cash only below two qualifying names. Unanswered questions
  backfill at the neutral 0.5 and are never scored, so a failure is missing
  coverage rather than a directional bet.
- Commands: `models`, `doctor [--probe]`, `run`, `calibrate`, `verify`, `compare`,
  `demo`, and `report`.
- Offline replay verification that re-checks artifact hashes and the snapshot
  manifest, replays recorded weights, and requires an exact metrics match.
- A reliability-curve chart and per-run calibration columns on the dashboard.

Changed

- Dependencies pruned to pydantic, pandas, numpy, httpx, exchange-calendars and
  yfinance. Removed: openai-agents, cryptography, and the chronos/timegpt/browser
  extras.
- `directional_accuracy` and the agent/tool telemetry are gone from metrics and
  from the UI.

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
