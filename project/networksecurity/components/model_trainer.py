import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.constant.training_pipeline import (
    FINAL_MODEL_DIR,
    FINAL_MODEL_FILE_NAME,
    MODEL_REPORT_FILE_NAME,
)

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data, evaluate_models, write_yaml_file
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)


class ModelTrainer:
    """
    Trains several classical ML classifiers on the transformed phishing
    dataset, picks the best one by held-out test accuracy, and saves it
    together with the preprocessing pipeline so it can be reused for
    inference (batch prediction and the Streamlit app) without retraining.

    All experiment metrics are written to a local YAML report
    (Artifacts/<timestamp>/model_trainer/model_report.yaml) instead of an
    external experiment tracker, so nothing here needs internet access or
    third-party accounts.
    """

    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, y_train, x_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "AdaBoost": AdaBoostClassifier(random_state=42),
        }
        params = {
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"],
            },
            "Random Forest": {
                "n_estimators": [8, 16, 32, 128, 256]
            },
            "Gradient Boosting": {
                "learning_rate": [.1, .01, .05, .001],
                "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                "n_estimators": [8, 16, 32, 64, 128, 256]
            },
            "Logistic Regression": {},
            "AdaBoost": {
                "learning_rate": [.1, .01, .001],
                "n_estimators": [8, 16, 32, 64, 128, 256]
            }
        }
        model_report: dict = evaluate_models(
            X_train=X_train, y_train=y_train, X_test=x_test, y_test=y_test, models=models, param=params
        )

        # Best model = highest accuracy on the held-out test split
        best_model_score = max(model_report.values())
        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]

        logging.info(f"Best model selected: {best_model_name} (test accuracy={best_model_score:.4f})")

        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        # Persist a local, human-readable experiment report (no external tracker required)
        report_path = os.path.join(self.model_trainer_config.model_trainer_dir, MODEL_REPORT_FILE_NAME)
        write_yaml_file(
            file_path=report_path,
            content={
                "candidate_model_test_accuracy": {k: float(v) for k, v in model_report.items()},
                "best_model": best_model_name,
                "train_metrics": {
                    "f1_score": float(classification_train_metric.f1_score),
                    "precision_score": float(classification_train_metric.precision_score),
                    "recall_score": float(classification_train_metric.recall_score),
                },
                "test_metrics": {
                    "f1_score": float(classification_test_metric.f1_score),
                    "precision_score": float(classification_test_metric.precision_score),
                    "recall_score": float(classification_test_metric.recall_score),
                },
            },
            replace=True,
        )

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
        # Save the full wrapper (preprocessor + model) as the versioned artifact
        save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

        # Also refresh the "final_model" copy used directly by the Streamlit app
        os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
        save_object(os.path.join(FINAL_MODEL_DIR, FINAL_MODEL_FILE_NAME), obj=best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
