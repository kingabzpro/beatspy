# Architecture

BeatSPY has two independent parts: a Python benchmark package in `src/beatspy`, and
static HTML/CSS/JavaScript in `dashboard`. The website reads reviewed artifact files;
it does not run models, store credentials, or require an application server.

This is a benchmark for **decision models**. There are no agents, no tool calling,
no prompts, and no conversation. A decision model reads a `state` and a map of
typed `questions`, and returns one typed `answer` per question.

## Two contracts, one shape

Every provider speaks the same request: `state` plus `questions`, where each
question is `noul` (a yes/no probability), `choice` (an option plus a
distribution) or `score` (a probability-weighted position on ordered levels). Only
the transport differs, so `dcn/clients.py` holds one adapter per provider behind a
single `DcnClient` protocol:

| Provider | Transport |
| --- | --- |
| OpenAI Decisions | `POST /v1/responses` with a `decision.questions` block |
| TypeSafe Jev | `POST https://api.typesafe.ai/v1/systemone` |
| Cloudflare Clef | `POST /accounts/{id}/ai/run/@cf/cloudflare/{model}` |

429, 529, 5xx and connection errors retry with exponential backoff; any other 4xx
raises immediately, because retrying a schema error only burns quota.

## Benchmark flow

Resolve the scenario's concrete dates → load or freeze a content-addressed price
snapshot → for each decision date, build the point-in-time state, ask the typed
questions in one call, convert the answers to target weights, validate them →
execute at the next session's open → label every probability against the realized
benchmark-relative outcome → score calibration and P&L → write artifacts.

Decision dates stay strictly sequential because later decisions depend on earlier
positions. Runs are independent of one another and share only the read-only
snapshot.

## The state is the only evidence channel

`dcn/state.py` builds one compact JSON object from `DataService`, which never
serves a bar later than `as_of`. Two rules are load-bearing:

- **No leakage.** Only data dated on or before the decision date enters the state.
  A test builds the same decision from two price series that are identical up to
  the decision date and completely different afterwards, and requires
  byte-identical state.
- **A hard token budget.** Workers AI silently truncates long text state to
  roughly its first 2K tokens, which would quietly give Clef a different question
  from the one the other providers read. The state is therefore budgeted to stay
  inside `DecisionConfig.max_state_tokens` (~1,800) for *every* provider — the
  full 23-name universe is covered by a test — and the size is recorded on each
  decision as `state_chars` plus a `state_sha256`.

The state describes the market; it never states an allocation. Allocation rules
live in the question set and the sizing policy, where they can be audited.

## Calibration is half the benchmark

`dcn/calibration.py` is provider-independent and deterministic. Labels are
*relative* — did the ticker beat the benchmark over the same horizon — so a rising
market cannot make every prediction correct. Both legs are measured
close-to-close strictly after the decision date, through
`DataService.close_on_offset`, which is the only accessor that deliberately looks
forward and is unreachable from state construction.

Per form it reports Brier, Brier skill against a constant at the sample rate, log
loss (clipped at 1e-15), AUC via the Mann-Whitney U statistic so ties are counted
as half and never inflate the score, accuracy, and 10-bin reliability with
expected and maximum calibration error. `miscalibration_gap` is the mean absolute
difference between the `noul` and `choice` answers to the same question, which is
the specific failure mode the benchmark exists to measure.

**An inferred number is never scored.** Unanswered questions are backfilled at the
neutral 0.5 so the portfolio can be sized, but `answered_noul` / `answered_choice`
/ `answered_rank` hold only what the provider actually returned, and only those
become observations. A test proves that a provider answering half the questions
produces calibration statistics over exactly half of them.

## Sizing policy

`dcn/pipeline.py` maps probabilities to a portfolio under fixed rules: only `noul`
probabilities are used, a name must exceed `min_probability` (0.5) to be held, the
strongest `top_n` are equal-weighted, and fewer than `min_names` qualifying names
means cash only. `schemas.validate_weights` then clamps to the scenario's position
limit and records every correction as a violation rather than applying it
silently.

A provider failure yields `invalid = True` and no weights, which the run loop
treats as **hold** — a model failure must never become a liquidation.

## Scoring and replay

`dcn/scoring.py` scores the portfolio through the shared accounting engine against
SPY buy-and-hold, equal weight, 60/40 and 12-minus-1 momentum on the same frozen
prices, and assembles the calibration report. `dcn/verify.py` replays a finished
run offline: it re-checks artifact hashes and the snapshot manifest, rebuilds the
decision schedule, replays the *recorded* weights through the accounting engine,
recomputes metrics from the recorded observations, and requires an exact match
with the stored `metrics.json`, equity curve and trades.

Because decisions record their questions, answers, state hash and observations,
replay needs no network and never re-derives a label differently from the run.

## Data and publication

NYSE session closes resolve recent windows; a 15-minute provider-finalization
buffer excludes unfinished bars. Fresh sessions create content-addressed snapshots
under local `.beatspy-data/snapshots/`; scenario pointers select the latest without
changing old files.

The static dashboard uses safe text rendering and native SVG charts, and shows the
leaderboard, the equity curve with baselines, and a reliability curve per run with
the noul/choice/rank curves and the miscalibration gap. A local report is a copy of
the same frontend with a local catalog. See [CONTRIBUTING](CONTRIBUTING.md) for
trust controls, limitations, development, and deployment.

## What was removed

The five-agent relay, the tool layer, the news and web-research integrations, the
forecast providers, and the Ed25519 signed-submission path were all deleted
together with the agent benchmark. They existed to serve a chat model that
gathered evidence and emitted JSON; a decision model needs none of it. `main` and
`main-backup-2026-10-06` still carry them.