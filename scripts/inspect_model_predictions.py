"""Script to inspect the prediction probabilities of the trained model."""

import sys
from pathlib import Path

import joblib
import pandas as pd

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine

from src.features.pipeline import create_all_features
from src.logger import logger

DB_PATH = project_root / "data" / "m5_trading.db"
MODEL_PATH = project_root / "data" / "models" / "EURUSDm_model_20250901_065505.joblib"
SYMBOL = "EURUSDm"
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"


def main():
    """Loads the model and inspects its prediction distribution."""
    logger.info("--- Starting Model Prediction Inspection ---")

    # 1. Load Data
    engine = create_engine(f"sqlite:///{DB_PATH}")
    query = f"""
        SELECT time, open, high, low, close, volume
        FROM bars
        WHERE symbol = '{SYMBOL}' AND time BETWEEN '{START_DATE}' AND '{END_DATE}'
        ORDER BY time ASC
    """
    raw_data = pd.read_sql(query, engine, index_col="time", parse_dates=["time"])
    if raw_data.index.has_duplicates:
        raw_data = raw_data[~raw_data.index.duplicated(keep="last")]

    X = create_all_features(raw_data, SYMBOL)

    # 2. Load Model
    try:
        model = joblib.load(MODEL_PATH)
        logger.success("Model loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Model not found at {MODEL_PATH}.")
        return

    # 3. Get Prediction Probabilities
    logger.info("Calculating prediction probabilities...")
    feature_names = model.estimators_[0].feature_names_in_

    # predict_proba returns a list of arrays (one per target)
    # each array is [prob_class_0, prob_class_1]
    all_probs = model.predict_proba(X[feature_names])

    # We only care about the probability of the positive class (1)
    positive_probs = [p[:, 1] for p in all_probs]

    # For each sample, find the highest probability across all 12 targets
    max_probs = pd.Series([max(probs) for probs in zip(*positive_probs)], index=X.index)

    # 4. Analyze and report the distribution
    logger.info("--- Prediction Probability Distribution ---")
    logger.info(f"\n{max_probs.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])}")


if __name__ == "__main__":
    main()
