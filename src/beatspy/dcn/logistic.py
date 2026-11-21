"""Linear and blended predictors on LightGBM's point-in-time training rows."""

from __future__ import annotations

import pandas as pd

from .clients import DcnError
from .lightgbm import FEATURES, LightGBMClient


class LogisticRegressionClient(LightGBMClient):
    model = "logistic-regression"
    endpoint = "local://logistic-regression"

    def _predict_probabilities(self, training: pd.DataFrame, labels: list[int], inputs: pd.DataFrame):
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
        except ImportError as exc:
            raise DcnError("Logistic regression requires scikit-learn.") from exc

        numeric = [feature for feature in FEATURES if feature != "sector"]
        preprocessor = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("sector", OneHotEncoder(handle_unknown="ignore"), ["sector"]),
        ])
        model = make_pipeline(preprocessor, LogisticRegression(C=1.0, max_iter=500))
        model.fit(training, labels)
        return model.predict_proba(inputs)[:, 1]


class LogisticEnsembleClient(LogisticRegressionClient):
    model = "logistic-lightgbm"
    endpoint = "local://logistic-lightgbm"

    def _predict_probabilities(self, training: pd.DataFrame, labels: list[int], inputs: pd.DataFrame):
        linear = super()._predict_probabilities(training, labels, inputs)
        boosted = LightGBMClient._predict_probabilities(self, training, labels, inputs)
        return 0.5 * (linear + boosted)
