## Summary

What changes and why, in one or two sentences.

## Integrity checklist (leave only what applies)

- [ ] No agent-facing tool can serve data past the decision date (`ToolContext.as_of`).
- [ ] New tools consume the per-agent budget via `_guard`.
- [ ] Metrics remain deterministic; any randomness is seeded.
- [ ] Submitted text is rendered as text, never interpreted as HTML.
- [ ] Backtest outputs keep the hypothetical-results disclaimer; synthetic data stays labeled.

## Tests

- [ ] `uv run --extra dev pytest` passes locally (hermetic tests only; no network, no keys).
- [ ] New logic has a test, or the change is covered by existing ones.

## Artifacts

If benchmark behavior changed, attach `metrics.json` from a before/after run on the
same scenario and data snapshot.

For result submissions:

- [ ] `beatspy validate --run results/<run-id>` passes.
- [ ] Artifacts and exact frozen data are included, without credentials.
- [ ] State the model/provider and command used; label look-ahead risks.
- [ ] This is a community submission, not a claim of verified model identity.
