import numpy as np
import pandas as pd

from beatspy.dcn.lightgbm import LightGBMClient


def test_ensemble_averages_the_same_rows_with_equal_weights(monkeypatch):
    from beatspy.dcn.logistic import LogisticEnsembleClient, LogisticRegressionClient

    training = pd.DataFrame({"feature": [1.0, 2.0]})
    labels = [0, 1]
    current = pd.DataFrame({"feature": [4.0, 3.0]}, index=["Z", "A"])

    def predictor(values):
        def predict(self, features, outcomes, inputs):
            assert features is training and outcomes is labels and inputs is current
            return np.array(values)

        return predict

    monkeypatch.setattr(LightGBMClient, "_predict_probabilities", predictor([0.2, 0.8]), raising=False)
    monkeypatch.setattr(LogisticRegressionClient, "_predict_probabilities", predictor([0.4, 0.6]))
    client = LogisticEnsembleClient(None)
    np.testing.assert_allclose(client._predict_probabilities(training, labels, current), [0.3, 0.7])
