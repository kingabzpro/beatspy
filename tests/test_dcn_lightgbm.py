from datetime import date

import numpy as np
import pandas as pd

from beatspy.data.service import DataService


def test_lightgbm_training_includes_only_matured_forward_outcomes():
    from beatspy.dcn.lightgbm import LightGBMClient

    dates = pd.bdate_range("2025-01-01", "2025-02-21")
    as_of = date(2025, 2, 12)

    def service(future_multiplier):
        frames = []
        for ticker, closes in {"AAA": 100 + np.arange(len(dates), dtype=float), "SPY": np.full(len(dates), 100.)}.items():
            closes[dates.date > as_of] *= future_multiplier
            frames.append(pd.DataFrame({
                "ticker": ticker, "date": dates, "open": closes, "high": closes,
                "low": closes, "close": closes, "volume": 100,
            }))
        return DataService(pd.concat(frames, ignore_index=True))

    features, outcomes = LightGBMClient(service(1)).training_data(as_of, 5, ["AAA"])
    changed_features, changed_outcomes = LightGBMClient(service(8)).training_data(as_of, 5, ["AAA"])
    assert list(features.index) == [(date(2025, 1, 31), "AAA")]
    assert list(outcomes) == [1]
    pd.testing.assert_frame_equal(features, changed_features)
    np.testing.assert_array_equal(outcomes, changed_outcomes)
