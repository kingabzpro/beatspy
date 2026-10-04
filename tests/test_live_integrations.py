"""Live integration tests for optional forecasting extras.

These run only when the corresponding extra is installed AND credentials are
available; CI and hermetic runs skip them automatically.
"""

from __future__ import annotations

import os
from importlib.util import find_spec
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PRICES = REPO_ROOT / ".beatspy-data" / "2022-bear" / "prices.csv"
pytestmark = pytest.mark.skipif(
    os.environ.get("BEATSPY_LIVE_TESTS") != "1", reason="live tests require explicit opt-in"
)


def _spy_closes(n: int = 260) -> list[float]:
    import pandas as pd

    frame = pd.read_csv(PRICES, parse_dates=["date"])
    frame = frame[frame["ticker"] == "SPY"].sort_values("date")
    return [float(v) for v in frame["close"].tail(n)]


requires_prices = pytest.mark.skipif(not PRICES.exists(), reason="frozen 2022-bear snapshot not present")


@pytest.mark.skipif(not os.environ.get("BEATSPY_TIMEGPT_API_KEY"), reason="requires BEATSPY_TIMEGPT_API_KEY")
@pytest.mark.skipif(find_spec("nixtla") is None, reason="nixtla extra not installed")
@requires_prices
def test_timegpt_live_forecast():
    from beatspy.tools.forecast import timegpt_forecast

    try:
        result = timegpt_forecast("SPY", _spy_closes(), 20, api_key=os.environ["BEATSPY_TIMEGPT_API_KEY"])
    except Exception as exc:
        if "request limit" in str(exc) or "429" in str(exc):
            pytest.skip("TimeGPT monthly quota exhausted for this API key")
        raise
    assert result is not None
    assert result.method == "timegpt-1"
    assert result.p10_pct <= result.p50_pct <= result.p90_pct
    print("\nTimeGPT live:", result.as_dict())


@pytest.mark.skipif(find_spec("chronos") is None, reason="chronos extra not installed")
@requires_prices
def test_chronos_live_forecast():
    from beatspy.tools.forecast import chronos_forecast

    result = chronos_forecast("SPY", _spy_closes(), 20)
    assert result is not None
    assert result.method == "chronos-t5-small"
    assert result.p10_pct <= result.p50_pct <= result.p90_pct
    # seeded sampling must be deterministic
    again = chronos_forecast("SPY", _spy_closes(), 20)
    assert again.as_dict() == result.as_dict()
    print("\nChronos live:", result.as_dict())
