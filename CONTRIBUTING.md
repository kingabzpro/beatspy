# Contributing

## Run and submit

1. Clone the repo and run `uv sync` (Python 3.11+ required).
2. Run `uv run beatspy setup` to select a model. Fast runs only require its API key.
3. Run `uv run beatspy run`. The benchmark starts on January 1, 2026 and ends at the latest completed NYSE session, with 12 decisions spread across the full period. Each uses one model call and cached Yahoo prices; no research API keys are needed.
4. View `uv run beatspy results` or `uv run beatspy report --open`.
5. Run `uv run beatspy submit --send`. Authenticate GitHub CLI first with `gh auth login`.

No run IDs, artifact copying, commits, or forks are needed to request verification.
`beatspy run --submit` combines running and submission. `beatspy submit` only prepares
a request file; `beatspy validate` replays the latest real run without sending anything.

To compare direct Olostep search and scraping locally, set `OLOSTEP_API_KEY` and
run `uv run beatspy run --web-research`. It is capped at two searches and three
scrapes per decision. Pages with missing, future, or future-updated dates are
withheld. Live publisher dates cannot authenticate historical content, so this
experimental mode does not qualify for the official leaderboard.

Submission creates a GitHub issue containing the model name and local run ID.
The owner adds the `verify-benchmark` label to start an approved trusted rerun.
Unknown models need an owner-reviewed entry in `.github/benchmark-models.json`.
Requests cannot set endpoints, scenarios, scripts, or signing credentials.
Once the rerun finishes, the workflow checks and signs its own new result and opens
a leaderboard PR. Publication follows its review and merge. The public dashboard
also offers a prefilled verification request button.

## Owner setup (once)

- Commit the **public** Ed25519 PEM key to `.github/verification-key.pem`.
- Put the matching existing **private** PEM key in the `BEATSPY_SIGNING_KEY` secret
  of the `leaderboard-signing` GitHub environment. Never commit the private key.
- Configure that environment to allow only protected `main` and require owner approval.
- Add the model API secrets listed in `.github/benchmark-models.json`;
  no financial research API keys are required for the trusted fast benchmark.
- Enable Actions to create PRs. Require the result-validation check and CODEOWNER
  review for protected code/workflows/public-key changes before merging.
- Optionally set `BEATSPY_SUBMISSION_TOKEN` to a repository-scoped GitHub App/PAT
  token so generated PR checks start automatically. With the default Actions token,
  approve the generated PR's workflow runs before merging ([GitHub behavior](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)).

The trusted workflow only checks out main-branch code; it never executes submitted
code or downloads a user's proposed result to sign. Its signing key is available
only in the signing step, after model execution. New submissions must pass the
pinned public-key signature, benchmark version/code digest checks, full historical
period check, artifact hashes, and deterministic replay using base-branch code.
A unique verification ID binds the result, release, source revision, and execution URL.
Modified unsigned legacy records cannot be accepted as new verified results.
Only an owner-authored PR can reset the leaderboard to an empty catalog; that exception
accepts no result artifacts. Every subsequent addition still needs a trusted signature.

The private key signs; the public key validates. Signatures attest the controlled
execution and exact artifact bytes, not predictive skill or freedom from model
knowledge leakage. Nondeterministic reruns can legitimately produce different scores.

## Develop

```bash
uv sync --extra dev --extra browser
uv run playwright install chromium
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
node --test tests/dashboard.test.cjs
uv build
```

Commit `uv.lock` for dependency changes. Unit tests must not call model/data APIs.
Live forecast checks require `BEATSPY_LIVE_TESTS=1` and provider credentials.
Scenarios live in `src/beatspy/scenarios/builtin/`; tools must enforce their decision
date and budget. Prices include warmup data, but scoring starts at the stated start.
Snapshots remain immutable and must cover every ticker/session. Recent scenarios
remain available for quick experiments; leaderboard requests use `2026-ytd`.

The fixed stock universe has survivorship bias. Curated events and Finnhub history
may be sparse. Current fundamentals and unarchived historical analyst ratings
are withheld; adjusted prices and model knowledge remain historical-test limitations.

## Dashboard

The static frontend in `dashboard/` reads `data/index.json`. It displays the latest
execution per model, never selects the best score, and lists each evaluation period.
Details include performance, allocations, cost estimates, provenance, and verification IDs.
Decision and trade evidence remains downloadable for replay; detailed tables are omitted from the interface.
Cost estimates use OpenRouter's standard uncached input/output token prices, checked
2026-10-05. They exclude provider cache discounts, subscriptions, and external tool fees.
Custom rates can be entered in the dashboard. Run estimates are not provider invoices.

Preview with `python -m http.server 8000 --bind 127.0.0.1 --directory dashboard`.
Vercel uses Root Directory `dashboard`, Framework Preset **Other**, and no build or
install command. `beatspy report --open` serves local results separately.
Production follows approved merges; private credentials must never be deployed.

Website branding lives in `dashboard/assets/`: SVG source artwork, a 1200×630 PNG
social preview, and home-screen icons. `index.html` includes Open Graph/X sharing
metadata and the canonical public URL. Keep image URLs, dimensions, and alternate
text synchronized when replacing the preview. All branding assets ship in local
reports and the Python wheel too.
