import os
import sys

import numpy as np
import pandas as pd

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.constant.training_pipeline import (
    FINAL_MODEL_DIR,
    FINAL_MODEL_FILE_NAME,
    FINAL_PREPROCESSOR_FILE_NAME,
    EXPECTED_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

LABELS = {0: "Normal Traffic", 1: "Potential Attack / Malicious Traffic"}


class BatchPredictionPipeline:
    """
    Loads the locally trained model + preprocessor once and reuses them to
    score new data - a single record (dict, e.g. from the URL scanner or
    the Advanced Feature Input form), or a whole CSV file. This is the
    single inference entry point every UI mode goes through, so
    training-time and inference-time preprocessing always stay in sync.
    """

    def __init__(
        self,
        model_path: str = os.path.join(FINAL_MODEL_DIR, FINAL_MODEL_FILE_NAME),
        preprocessor_path: str = os.path.join(FINAL_MODEL_DIR, FINAL_PREPROCESSOR_FILE_NAME),
    ):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Trained model not found at '{model_path}'. Run the training "
                    "pipeline first (python main.py) to generate it."
                )
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(
                    f"Preprocessing object not found at '{preprocessor_path}'. Run the "
                    "training pipeline first (python main.py) to generate it."
                )

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            self.network_model = NetworkModel(preprocessor=preprocessor, model=model)
            logging.info("Loaded trained model and preprocessor for inference")
        except Exception as e:
            print("BATCH PIPELINE ERROR:", repr(e))
            raise

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts on a dataframe of raw feature columns (any column order;
        NetworkModel aligns them to what the preprocessor expects).
        Returns the input dataframe with extra columns: predicted_result
        (0/1), prediction_label, and confidence (model probability of the
        predicted class, if the model supports predict_proba - else NaN).
        """
        try:
            feature_df = df.copy()
            if TARGET_COLUMN in feature_df.columns:
                feature_df = feature_df.drop(columns=[TARGET_COLUMN])

            predictions = self.network_model.predict(feature_df)
            probabilities = self.network_model.predict_proba(feature_df)

            result_df = df.copy()
            result_df["predicted_result"] = predictions
            result_df["prediction_label"] = [LABELS.get(int(p), str(p)) for p in predictions]
            if probabilities is not None:
                result_df["confidence"] = [
                    probabilities[i, int(predictions[i])] for i in range(len(predictions))
                ]
            else:
                result_df["confidence"] = np.nan
            return result_df
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict_single(self, feature_dict: dict):
        """
        Predicts on a single record represented as a dict of
        {feature_name: value}. Values may be numpy.nan for features that
        genuinely could not be determined (e.g. from the URL scanner) -
        the preprocessing pipeline's KNNImputer handles those, exactly as
        it was designed and fit to do.

        Returns (predicted_class: int, label: str, confidence: float|None)
        """
        try:
            ordered = {col: feature_dict.get(col, np.nan) for col in EXPECTED_FEATURE_COLUMNS}
            df = pd.DataFrame([ordered])
            result_df = self.predict_dataframe(df)
            predicted_class = int(result_df["predicted_result"].iloc[0])
            label = result_df["prediction_label"].iloc[0]
            confidence = result_df["confidence"].iloc[0]
            confidence = None if pd.isna(confidence) else float(confidence)
            return predicted_class, label, confidence
        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":
    # Simple CLI smoke test: python -m networksecurity.pipeline.batch_prediction <csv_path>
    try:
        csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("valid_data", "test.csv")
        pipeline = BatchPredictionPipeline()
        input_df = pd.read_csv(csv_path)
        output_df = pipeline.predict_dataframe(input_df)
        os.makedirs("prediction_output", exist_ok=True)
        out_path = os.path.join("prediction_output", "output.csv")
        output_df.to_csv(out_path, index=False)
        print(f"Predictions written to {out_path}")
        print(output_df["prediction_label"].value_counts())
    except Exception as e:
        raise NetworkSecurityException(e, sys)
