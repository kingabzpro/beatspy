from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from beatspy.dcn import hybrids


def test_hrp_and_control_hold_the_same_top_five():
    pytest.importorskip("scipy")
    index = pd.bdate_range("2024-01-01", periods=127)
    rng = np.random.default_rng(84)
    frames = {ticker: pd.DataFrame({"close": np.exp(rng.normal(0, i / 1000, len(index)).cumsum())}, index=index)
              for i, ticker in enumerate("ABCDEF", start=1)}
    data = SimpleNamespace(window=lambda ticker, day, size: frames[ticker].iloc[-size:])
    signals = dict(zip("ABCDEF", [0.4, 0.4, 0.8, 0.9, 0.6, 0.7], strict=True))
    hrp = hybrids.ranked_allocation(signals, data, index[-1].date(), risk_parity=True)
    control = hybrids.ranked_allocation(signals, data, index[-1].date(), risk_parity=False)
    assert set(hrp) == set(control) == set("ACDEF")
    assert all(value == 0.2 for value in control.values())
    assert hrp != control


def test_jepa_meta_features_start_after_encoder_training(monkeypatch):
    calendar = [day.date() for day in pd.bdate_range("2024-01-01", periods=180)]
    data = SimpleNamespace(calendar=calendar)
    client = hybrids.JepaElasticNetClient(data, "cpu")
    scored = []

    def panel(data, tickers, day, *, limit):
        if limit == hybrids.CONTEXT + 1:
            scored.append(day)
        else:
            assert day == calendar[2 * 179 // 3 - 1]
        return np.ones((len(tickers), hybrids.CONTEXT, 3))

    monkeypatch.setattr(hybrids, "market_panel", panel)
    monkeypatch.setattr(hybrids, "train_predictor", lambda samples, device: (
        lambda inputs: np.full((len(inputs), hybrids.HORIZON), 0.6)
    ))
    before, after, now = calendar[90], calendar[140], calendar[-1]
    training = pd.DataFrame(index=pd.MultiIndex.from_tuples([(before, "ABC"), (after, "ABC")]))
    current = pd.DataFrame(index=["ABC"])
    assert client.extra_features(training, current, now, 5, ["ABC"]) == ["stock_jepa"]
    assert pd.isna(training.iloc[0]["stock_jepa"])
    assert training.iloc[1]["stock_jepa"] == current.iloc[0]["stock_jepa"] == 0.6
    assert scored == [after, now]
