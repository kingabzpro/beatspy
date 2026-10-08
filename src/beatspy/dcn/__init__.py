"""DCN: a benchmark for decision models, not chat models.

A decision model reads a `state` and a schema of typed `questions`, then returns
a probability for every allowed answer. There is no text, no tool calling, and no
conversation. This package builds the historical state, asks the questions, and
scores both halves of the result:

- `calibration` — were the probabilities right, on the frozen outcomes?
- P&L — what did a portfolio built only from those probabilities earn, scored by
  the same accounting engine and against the same deterministic baselines?

The benchmark is deliberately tool-free: evidence reaches the model only through
the state, so every provider sees byte-identical input.
"""

from __future__ import annotations

# Bump when the question set, state layout, or sizing policy changes: recorded
# runs must never be silently reinterpreted under new rules.
DCN_PROTOCOL = 1

__all__ = ["DCN_PROTOCOL"]