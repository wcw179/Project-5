"""Script to load and inspect a saved model artifact."""

import sys
from pathlib import Path

import joblib

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.logger import logger

MODEL_PATH = project_root / "data" / "models" / "EURUSDm_model_20250830_150301.joblib"


def main():
    """Loads the model and prints its properties."""
    logger.info(f"--- Loading Model from {MODEL_PATH} ---")

    if not MODEL_PATH.exists():
        logger.error(f"Model file not found at {MODEL_PATH}")
        return

    try:
        model = joblib.load(MODEL_PATH)
        logger.success("Model loaded successfully.")

        logger.info("--- Model Inspection ---")
        logger.info(f"Model Type: {type(model)}")

        # For MultiOutputClassifier, inspect the individual estimators
        if hasattr(model, "estimators_"):
            logger.info(f"Number of estimators (targets): {len(model.estimators_)}")
            first_estimator = model.estimators_[0]
            logger.info(f"Type of individual estimator: {type(first_estimator)}")
            logger.info("Parameters of the first estimator:")
            logger.info(first_estimator.get_params())
        else:
            logger.info("Model parameters:")
            logger.info(model.get_params())

    except Exception as e:
        logger.error(f"Failed to load or inspect model: {e}")


if __name__ == "__main__":
    main()
