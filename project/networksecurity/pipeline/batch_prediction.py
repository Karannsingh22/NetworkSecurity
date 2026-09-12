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

LABELS = {
    0: "Normal Traffic",
    1: "Potential Attack / Malicious Traffic",
}


def find_file(filename, configured_path):
    """
    Find model/preprocessor files in common locations.

    Streamlit Cloud may use a different working directory than local Windows.
    """

    current_file = os.path.abspath(__file__)

    # project/ directory
    project_dir = os.path.abspath(
        os.path.join(os.path.dirname(current_file), "..", "..", "..")
    )

    # Repository root
    repo_dir = os.path.dirname(project_dir)

    candidates = [
        # Original configured path
        configured_path,

        # Relative to current working directory
        os.path.join(os.getcwd(), configured_path),

        # Relative to project/
        os.path.join(project_dir, configured_path),

        # Relative to repository root
        os.path.join(repo_dir, configured_path),

        # Common artifact locations
        os.path.join(project_dir, "artifacts", filename),
        os.path.join(project_dir, "final_model", filename),
        os.path.join(project_dir, "model", filename),
        os.path.join(project_dir, "models", filename),

        os.path.join(repo_dir, "artifacts", filename),
        os.path.join(repo_dir, "final_model", filename),
        os.path.join(repo_dir, "model", filename),
        os.path.join(repo_dir, "models", filename),
    ]

    # Remove duplicates
    candidates = list(dict.fromkeys(os.path.abspath(p) for p in candidates))

    for path in candidates:
        if os.path.isfile(path):
            logging.info(f"Found required file: {path}")
            return path

    # Last resort: recursively search project and repository
    for root_dir in [project_dir, repo_dir]:
        for root, dirs, files in os.walk(root_dir):
            # Avoid unnecessary directories
            dirs[:] = [
                d for d in dirs
                if d not in {
                    ".git",
                    ".venv",
                    "venv",
                    "__pycache__",
                    "node_modules"
                }
            ]

            if filename in files:
                path = os.path.join(root, filename)
                logging.info(f"Found required file by recursive search: {path}")
                return path

    return None


class BatchPredictionPipeline:
    """
    Loads the trained model and preprocessor and uses them for inference.
    """

    def __init__(
        self,
        model_path=None,
        preprocessor_path=None,
    ):
        try:

            # ---------------------------------------------------------
            # Build the paths from the training pipeline constants
            # ---------------------------------------------------------

            if model_path is None:
                configured_model_path = os.path.join(
                    FINAL_MODEL_DIR,
                    FINAL_MODEL_FILE_NAME,
                )
            else:
                configured_model_path = model_path

            if preprocessor_path is None:
                configured_preprocessor_path = os.path.join(
                    FINAL_MODEL_DIR,
                    FINAL_PREPROCESSOR_FILE_NAME,
                )
            else:
                configured_preprocessor_path = preprocessor_path

            # ---------------------------------------------------------
            # Find model
            # ---------------------------------------------------------

            actual_model_path = find_file(
                FINAL_MODEL_FILE_NAME,
                configured_model_path,
            )

            if actual_model_path is None:
                raise FileNotFoundError(
                    "\nTrained model file could not be found.\n\n"
                    f"Expected filename: {FINAL_MODEL_FILE_NAME}\n"
                    f"Configured path: {configured_model_path}\n"
                    f"Current working directory: {os.getcwd()}\n"
                )

            # ---------------------------------------------------------
            # Find preprocessor
            # ---------------------------------------------------------

            actual_preprocessor_path = find_file(
                FINAL_PREPROCESSOR_FILE_NAME,
                configured_preprocessor_path,
            )

            if actual_preprocessor_path is None:
                raise FileNotFoundError(
                    "\nPreprocessor file could not be found.\n\n"
                    f"Expected filename: {FINAL_PREPROCESSOR_FILE_NAME}\n"
                    f"Configured path: {configured_preprocessor_path}\n"
                    f"Current working directory: {os.getcwd()}\n"
                )

            logging.info(
                f"Loading model from: {actual_model_path}"
            )

            logging.info(
                f"Loading preprocessor from: {actual_preprocessor_path}"
            )

            # ---------------------------------------------------------
            # Load trained objects
            # ---------------------------------------------------------

            model = load_object(actual_model_path)

            preprocessor = load_object(actual_preprocessor_path)

            self.network_model = NetworkModel(
                preprocessor=preprocessor,
                model=model,
            )

            logging.info(
                "Loaded trained model and preprocessor successfully."
            )

        except Exception as e:

            print("BATCH PIPELINE ERROR:", repr(e))

            raise

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:

        try:

            feature_df = df.copy()

            if TARGET_COLUMN in feature_df.columns:
                feature_df = feature_df.drop(
                    columns=[TARGET_COLUMN]
                )

            predictions = self.network_model.predict(
                feature_df
            )

            probabilities = self.network_model.predict_proba(
                feature_df
            )

            result_df = df.copy()

            result_df["predicted_result"] = predictions

            result_df["prediction_label"] = [
                LABELS.get(int(p), str(p))
                for p in predictions
            ]

            if probabilities is not None:

                result_df["confidence"] = [
                    probabilities[i, int(predictions[i])]
                    for i in range(len(predictions))
                ]

            else:

                result_df["confidence"] = np.nan

            return result_df

        except Exception as e:

            raise NetworkSecurityException(e, sys)

    def predict_single(self, feature_dict: dict):

        try:

            ordered = {
                col: feature_dict.get(col, np.nan)
                for col in EXPECTED_FEATURE_COLUMNS
            }

            df = pd.DataFrame([ordered])

            result_df = self.predict_dataframe(df)

            predicted_class = int(
                result_df["predicted_result"].iloc[0]
            )

            label = result_df["prediction_label"].iloc[0]

            confidence = result_df["confidence"].iloc[0]

            confidence = (
                None
                if pd.isna(confidence)
                else float(confidence)
            )

            return predicted_class, label, confidence

        except Exception as e:

            raise NetworkSecurityException(e, sys)


if __name__ == "__main__":

    try:

        csv_path = (
            sys.argv[1]
            if len(sys.argv) > 1
            else os.path.join("valid_data", "test.csv")
        )

        pipeline = BatchPredictionPipeline()

        input_df = pd.read_csv(csv_path)

        output_df = pipeline.predict_dataframe(input_df)

        os.makedirs(
            "prediction_output",
            exist_ok=True
        )

        out_path = os.path.join(
            "prediction_output",
            "output.csv"
        )

        output_df.to_csv(
            out_path,
            index=False
        )

        print(
            f"Predictions written to {out_path}"
        )

        print(
            output_df[
                "prediction_label"
            ].value_counts()
        )

    except Exception as e:

        raise NetworkSecurityException(e, sys)