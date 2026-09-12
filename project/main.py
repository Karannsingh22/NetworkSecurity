"""
Entry point for running the full, local training pipeline.

Usage (from the project root, after installing requirements.txt):

    python main.py

This will:
  1. Read Network_Data/phisingData.csv
  2. Validate it against data_schema/schema.yaml and check for data drift
  3. Impute missing values and fit the preprocessing pipeline
  4. Train and compare several classifiers, keep the best one
  5. Save the trained model + preprocessor to final_model/ so the
     Streamlit app (app.py) can load them for predictions

No AWS, MongoDB Atlas, or other cloud/remote services are used.
"""
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    try:
        logging.info("Starting the local Network Security training pipeline")
        pipeline = TrainingPipeline()
        model_trainer_artifact = pipeline.run_pipeline()
        logging.info("Training pipeline completed successfully")
        print("Training complete.")
        print(model_trainer_artifact)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
