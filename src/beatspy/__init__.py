"""BeatSPY: benchmark decision models, not chat models.

A decision model reads a point-in-time state plus a schema of typed questions and
returns probabilities. This package asks those questions about frozen historical
markets and scores the answers twice: on calibration, and on a portfolio built
from them.
"""

__version__ = "0.3.0"