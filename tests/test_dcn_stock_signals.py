import numpy as np
import pandas as pd


def test_rank_targets_group_by_rebalance_date():
    from beatspy.dcn.stock_signals import rank_targets

    origins = pd.to_datetime(["2025-04-04"] * 5 + ["2025-04-11"] * 5)
    targets, groups = rank_targets(np.array([5, 4, 3, 2, 1, -5, -4, -3, -2, -1]), origins)
    np.testing.assert_array_equal(targets, [4, 3, 2, 1, 0, 0, 1, 2, 3, 4])
    assert groups == [5, 5]


def test_residual_signals_remove_market_beta_and_use_distinct_lookbacks():
    from beatspy.dcn.stock_signals import residual_signals

    market = np.tile([-0.01, 0.01], 63)
    stock = 2.0 * market
    stock[-65:-5] += 0.001
    stock[-5:] -= 0.005
    momentum, reversal = residual_signals(stock, market)
    assert momentum > 0
    assert reversal > 0
    np.testing.assert_allclose(residual_signals(2.0 * market, market), (0.0, 0.0), atol=1e-12)
