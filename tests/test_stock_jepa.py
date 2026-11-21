from datetime import date

import numpy as np
import pandas as pd

from beatspy.data.service import DataService


def test_jepa_training_panel_excludes_post_cutoff_prices():
    from beatspy.dcn.stock_jepa import market_panel

    dates = pd.date_range("2025-01-01", periods=5)
    prices = []
    for ticker in ("AAA", "SPY"):
        close = [100.0, 102.0, 105.0, 1000.0, 1.0] if ticker == "AAA" else [100.0] * 5
        prices.append(pd.DataFrame({
            "ticker": ticker, "date": dates, "close": close, "open": close, "high": close, "low": close,
            "volume": 100,
        }))
    data = DataService(pd.concat(prices, ignore_index=True))
    panel = market_panel(data, ["AAA"], date(2025, 1, 3), limit=100)
    assert panel.shape == (1, 2, 3)
    np.testing.assert_allclose(panel[0, :, 0], np.log([1.02, 105 / 102]), rtol=1e-6)
