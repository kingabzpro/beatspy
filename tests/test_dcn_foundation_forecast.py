from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.mark.parametrize("name", ["chronos-2", "timesfm-3.0", "t0-gamma"])
def test_foundation_adapter_returns_terminal_quantiles_in_input_order(name):
    from beatspy.dcn.foundation_forecast import FORECAST_CLIENTS, QUANTILE_LEVELS

    model = Mock()
    context = np.arange(130, dtype=np.float32).reshape(2, 65)
    quantiles = np.arange(54).reshape(2, 3, 9)
    predictions = [SimpleNamespace(quantiles=item) for item in quantiles]
    if name == "chronos-2":
        model.predict_quantiles.return_value = ([item[None, ...] for item in quantiles], None)
    elif name == "timesfm-3.0":
        model.predict_batch.return_value = iter(predictions)
    else:
        model.predict.return_value = SimpleNamespace(quantiles=Mock())
        model.predict.return_value.quantiles.__getitem__ = Mock(
            return_value=SimpleNamespace(cpu=lambda: quantiles[:, -1]),
        )
    client = FORECAST_CLIENTS[name](None, forecaster=model)
    knots, levels = client.terminal_quantiles(context, 3)
    np.testing.assert_array_equal(knots, quantiles[:, -1])
    assert levels == QUANTILE_LEVELS
