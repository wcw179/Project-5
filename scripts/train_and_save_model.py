"""Script to train a final model with the best parameters and save it."""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine

from src.features.pipeline import create_all_features
from src.labeling.dynamic_labels import create_dynamic_labels
from src.logger import logger

DB_PATH = project_root / "data" / "m5_trading.db"
MODEL_DIR = project_root / "data" / "models"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2025-08-15"

# Best parameters from Financial Objective Optuna Study
BEST_PARAMS = {
    "n_estimators": 489,
    "max_depth": 6,
    "learning_rate": 0.08840168831576564,
    "subsample": 0.8684907067369135,
    "colsample_bytree": 0.6930181682497625,
    "gamma": 0.12421432420241219,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "n_jobs": -1,
    "random_state": 42,
    "tree_method": "hist",
    "base_score": 0.5,
}


def main():
    """Main function to train and save the final model."""
    logger.info("--- Starting Final Model Training ---")

    # 1. Load and prepare data
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

    featured_data = create_all_features(raw_data, SYMBOL)
    labels, _ = create_dynamic_labels(raw_data)
    aligned_index = featured_data.index.intersection(labels.index)
    X = featured_data.loc[aligned_index]
    # Select all target columns
    y = labels.loc[aligned_index].filter(like="hit_")

    # 2. Train the model with the best parameters
    logger.info("Training final model on full dataset...")
    # Wrap the XGBoost model with MultiOutputClassifier for multi-label training
    xgb_model = xgb.XGBClassifier(**BEST_PARAMS)
    model = MultiOutputClassifier(xgb_model)
    model.fit(X, y)
    logger.success("Final model training complete.")

    # 3. Save the model artifact
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"{SYMBOL}_model_{timestamp}.joblib"

    try:
        joblib.dump(model, model_path)
        logger.success(f"Model saved successfully to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")


if __name__ == "__main__":
    main()
