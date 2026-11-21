import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from beatspy.data.service import DataService
from beatspy.dcn.questions import build_questions


def test_timestral_forecasts_relative_closes_with_the_benchmark_horizon():
    from beatspy.dcn.timestral import TimestralClient

    dates = pd.bdate_range("2025-01-01", periods=80)
    as_of = dates[69]
    spy = np.linspace(100.0, 110.0, len(dates))
    asset = 2.0 * spy
    asset[70:] *= 3.0
    prices = pd.concat(
        [
            pd.DataFrame(
                {"date": dates, "ticker": ticker, "open": close, "high": close,
                 "low": close, "close": close, "volume": 100}
            )
            for ticker, close in (("AAA", asset), ("SPY", spy))
        ],
        ignore_index=True,
    )

    class FixedForecast:
        def forecast(self, context, prediction_length):
            assert context.shape == (1, 70)
            np.testing.assert_allclose(context, 100.0 * np.log(2.0), rtol=1e-6)
            assert prediction_length == 4
            # At the terminal step, 75% of the forecast exceeds today's ratio.
            quantiles = np.full((1, 1, 4, 3), 200.0)
            quantiles[0, 0, -1] = [context[0, -1], 80.0, 90.0]
            return SimpleNamespace(quantiles=quantiles, quantile_levels=(0.25, 0.5, 0.75))

    client = TimestralClient(DataService(prices), forecaster=FixedForecast())
    questions = build_questions("twin", ["AAA", "MISSING"], "SPY", 4).questions
    result = asyncio.run(client.decide({"as_of": str(as_of.date()), "horizon_trading_days": 4}, questions))
    assert result.answers["noul_AAA"].noul == pytest.approx(0.75)
    assert result.answers["choice_AAA"].probabilities == pytest.approx({"outperform": 0.75, "underperform": 0.25})
    assert result.missing_ids == ["noul_MISSING", "choice_MISSING"]
    assert result.requests == 1
