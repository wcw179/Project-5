"""Script to train a final model with the best parameters and save it."""

import sys
from datetime import datetime
from pathlib import Path

import json
import joblib
import pandas as pd
import xgboost as xgb
from sqlalchemy import create_engine

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.labeling.trend_scanning_labeling import generate_true_labels
from src.logger import logger

DB_PATH = project_root / "data" / "m5_trading.db"
MODEL_DIR = project_root / "models"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2025-08-31"

def main():
    """Main function to train and save the final model."""
    logger.info("--- Starting Final Model Training ---")

    # 1. Load and prepare data using the corrected labeling function
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

    X, y, _ = generate_true_labels(
        raw_data,
        pt_mult=20.0,
        sl_mult=1.5,
        num_days=5
    )

    # 2. Load best hyperparameters from the optimization run
    params_path = MODEL_DIR / f"{SYMBOL}_best_primary_params.json"
    if not params_path.exists():
        logger.error(f"Hyperparameter file not found at {params_path}. Please run the optimization script first.")
        return

    with open(params_path, 'r') as f:
        best_params = json.load(f)

    # Add multiclass objective params
    best_params['objective'] = 'multi:softmax'
    best_params['num_class'] = 3
    best_params['eval_metric'] = 'mlogloss'

    logger.info(f"Loaded best parameters: {best_params}")

    # 3. Train the model with the best parameters
    logger.info("Training final model on full dataset...")

    # Map labels from {-1, 0, 1} to {0, 1, 2} for XGBoost
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})

    model = xgb.XGBClassifier(**best_params)
    model.fit(X, y_mapped)
    logger.success("Final model training complete.")

    # 4. Save the model artifact
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
