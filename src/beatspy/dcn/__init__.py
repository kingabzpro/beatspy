"""DCN: a benchmark for decision models, not chat models.

A decision model reads a `state` and a schema of typed `questions`, then returns
a probability for every allowed answer. There is no text, no tool calling, and no
conversation. This package builds the historical state, asks the questions, and
scores both halves of the result:

- `calibration`: were the probabilities right, on the frozen outcomes?
- P&L: what did a portfolio built only from those probabilities earn, scored by
  the same accounting engine and against the same deterministic baselines?

The benchmark is deliberately tool-free: evidence reaches the model only through
the state, so every provider sees byte-identical input.
"""

from __future__ import annotations

# Bump when the question set, state layout, or sizing policy changes: recorded
# runs must never be silently reinterpreted under new rules.
#
# 2: the state budget is enforced (recent history may be trimmed), and the
#    default sizing policy is graded_cs, which is invariant to a provider's
#    probability scale. Runs recorded under 1 keep their own policy in run.json
#    and are rejected by `beatspy verify` rather than re-scored under new rules.
# 3: the default sizing policy was core_satellite: most of the book held the
#    benchmark itself. Superseded, because a book that is 85% the index is not
#    making choices.
# 4: the benchmark is never investable again. The default policy is graded_cs
#    over the scenario's own assets, so the score reflects the model's choices;
#    SPY is the yardstick only.
# 5: the state carries a point-in-time base rate, the sizing signal defaults to
#    a consensus across every question form answered, and the policy trades with
#    a rebalance band so small weekly moves are not turned into fees.
DCN_PROTOCOL = 5

__all__ = ["DCN_PROTOCOL"]