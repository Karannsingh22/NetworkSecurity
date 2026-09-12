from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME, EXPECTED_FEATURE_COLUMNS

import os
import sys

import pandas as pd

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


class NetworkModel:
    """
    Thin wrapper around (preprocessor, model) that is used everywhere a
    prediction is made - training-time evaluation, batch CSV prediction,
    and the URL scanner - so inference always goes through the exact same
    preprocessing the model was trained with.
    """

    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def _align_columns(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Reindexes the input to exactly the column set/order the fitted
        preprocessor expects. Fitted scikit-learn transformers store the
        column names they were fit on (`feature_names_in_`) and raise if
        given a DataFrame with different names or a different order - so
        this is a safety net, not just a convenience: it guarantees the
        URL scanner, the advanced manual-input form, and batch CSV upload
        all feed the model identically, regardless of what order their
        code happened to build the dict/DataFrame in.
        """
        expected_columns = list(
            getattr(self.preprocessor, "feature_names_in_", EXPECTED_FEATURE_COLUMNS)
        )
        missing = [c for c in expected_columns if c not in x.columns]
        if missing:
            raise ValueError(f"Input is missing required feature columns: {missing}")
        return x[expected_columns]

    def predict(self, x: pd.DataFrame):
        try:
            x = self._align_columns(x)
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict_proba(self, x: pd.DataFrame):
        """
        Returns class probabilities if the underlying model supports it
        (e.g. RandomForestClassifier does; some models don't). Returns
        None if not supported, so callers can fall back to showing just
        the predicted class.
        """
        try:
            if not hasattr(self.model, "predict_proba"):
                return None
            x = self._align_columns(x)
            x_transform = self.preprocessor.transform(x)
            return self.model.predict_proba(x_transform)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
