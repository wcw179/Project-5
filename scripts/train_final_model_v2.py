"""
This script trains the final v2 XGBoost model (MultiOutputClassifier) on the
entire dataset using the meta-labeling pipeline.
"""
import sys
from pathlib import Path
import json
import joblib
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.multioutput import MultiOutputClassifier

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from scripts.build_training_dataset_v2 import build_dataset_v2

# --- Logger Configuration ---
log_file_path = project_root / "logs" / "final_model_training_v2.log"
logger.add(log_file_path, rotation="10 MB", retention="10 days")

# --- Constants ---
MODEL_DIR = project_root / "models"
SYMBOL = "EURUSDm"

def main():
    """Main function to train and save the final v2 model."""
    logger.info("--- Starting Final Model Training (v2) ---")

    # 1. Build the full dataset using the v2 pipeline
    X, y, _ = build_dataset_v2()
    if X.empty:
        logger.error("v2 Dataset creation failed. Exiting training.")
        return

    # 2. Load the best hyperparameters (assuming they are saved from run_optimization)
    params_path = MODEL_DIR / f"{SYMBOL}_best_primary_params.json"
    if not params_path.exists():
        logger.warning(f"Best parameters file not found at {params_path}. Using default parameters.")
        best_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_estimators': 500,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1,
            'tree_method': 'hist',
            'random_state': 42,
        }
    else:
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        logger.info(f"Loaded best hyperparameters: {best_params}")

    # 3. Train the model on the entire dataset
    logger.info(f"Training final v2 model on {len(X)} samples...")
    
    estimator = xgb.XGBClassifier(**best_params)
    model = MultiOutputClassifier(estimator=estimator, n_jobs=-1)
    model.fit(X, y)

    logger.success("Final v2 model training complete.")

    # 4. Save the trained model
    model_path = MODEL_DIR / "final_model_EURUSD_v2.joblib"
    joblib.dump(model, model_path)
    logger.success(f"Saved final trained v2 model to {model_path}")

if __name__ == "__main__":
    main()

