import numpy as np
import pandas as pd
import pytest


def test_hrp_allocates_inverse_variance_for_two_assets():
    pytest.importorskip("scipy")
    from beatspy.dcn.hrp import hrp_weights

    returns = pd.DataFrame({"AAA": [-0.01, 0.01, -0.01, 0.01], "BBB": [-0.02, -0.02, 0.02, 0.02]})
    weights = hrp_weights(returns)
    assert weights == pytest.approx({"AAA": 0.8, "BBB": 0.2})
    assert np.isclose(sum(weights.values()), 1.0)
