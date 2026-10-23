# Changelog

## 0.4.0 - protocol 6 (v2): a state worth reading

**Breaking for recorded runs:** `DCN_PROTOCOL` is 6. Protocol-5 runs stay readable
and stay published on their own windows, but `beatspy verify` no longer re-scores
them under the new rules.

Why: measured across every 2026-6m decision, model probabilities tracked the last
five days of returns at rank correlation +0.75 (d3-turbo) and +0.55 (SubQ Snap):
with only momentum fields in the state, the models echoed the freshest column, and
no answer type changed their below-random ordering (every AUC 0.39-0.47).

Added

- Risk and context in every market row: RSI-14, 30-day annualized realized
  volatility, distance from the 52-week high, a 10-vs-50-session volume ratio, and
  a sector label. Computed point-in-time from the frozen snapshot; fields that
  cannot be computed are omitted rather than invented.
- New decision models: vLLM SR D3 Turbo 27B, D3 Mini 4B, D3 Edge 0.6B (free API at
  decision.vllm-sr.ai) and SubQ Snap 1.1 (api.subq.ai, $0.05/M). Keys resolve from
  `VLLM_SR_API_KEY` and `SUBQ_API_KEY`.
- Question batching in the shared DCN client (`question_batch` in the model spec):
  endpoints that re-read the whole state once per question (the smaller d3 models)
  get their 25 questions split across smaller requests and one merged result.
- The full three-window v2 sweep: all 15 runnable models on every six-month
  window, 45 protocol-6 runs beside the 33 recorded under v5. The single-window
  leader did not survive out of sample; the best 18-month average gap to SPY is
  +0.42pp, and only Jev and SubQ Snap beat SPY in two of three windows, inside
  their bootstrap intervals.
- `dcn_protocol` on every catalog and dashboard row, and the site boards separate
  windows by protocol, so a v2 run never replaces a v5 row it is not comparable
  with. The board opens on the v2 window when one is present.

Changed

- The state budget rose from 1,800 to 2,900 tokens for the wider rows. No provider
  truncated the wider state; each one's own reported input-token count (about 6.5K
  to 7K per request) confirms the full state was read.
- Question text: a one-day horizon now reads "the next 1 trading day", and the
  meaningless `state_fields` key is gone from the instructions.

## 0.3.0 - decision-model benchmark

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
- Three question forms (`noul`, `choice`, `rank`) with a `twin` mode that asks
  the first two of the same ticker in one call, so the form itself can be measured.
- Calibration scoring: Brier, Brier skill, log loss, tie-aware AUC, accuracy, and
  10-bin reliability with expected and maximum calibration error, plus
  `miscalibration_gap` between the noul and choice answers.
- Labeling that is *benchmark-relative* (did the ticker beat the benchmark over
  the same horizon), so a rising market cannot mark every prediction correct.
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
