"""
This script trains the final XGBoost model on the entire dataset using the
best hyperparameters found during optimization and saves the trained model.
"""
import sys
from pathlib import Path
import json
import joblib
import pandas as pd
import xgboost as xgb
from loguru import logger

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from scripts.build_training_dataset import build_dataset

# --- Logger Configuration ---
log_file_path = project_root / "logs" / "final_model_training.log"
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>")
logger.add(log_file_path, rotation="10 MB", retention="10 days", enqueue=True, serialize=False)

# --- Constants ---
MODEL_DIR = project_root / "models"
SYMBOL = "EURUSDm"

def main():
    """Main function to train and save the final model."""
    logger.info("--- Starting Final Model Training ---")

    # 1. Build the full dataset
    X, y, _ = build_dataset()
    if X.empty:
        logger.error("Dataset creation failed. Exiting training.")
        return

    # 2. Load the best hyperparameters
    params_path = MODEL_DIR / f"{SYMBOL}_final_best_params.json"
    if not params_path.exists():
        logger.error(f"Best parameters file not found at {params_path}. Run optimization first.")
        return

    with open(params_path, 'r') as f:
        best_params = json.load(f)
    logger.info(f"Loaded best hyperparameters: {best_params}")

    # Add parameters not tuned by Optuna
    best_params['objective'] = 'multi:softmax'
    best_params['num_class'] = 3
    best_params['eval_metric'] = 'mlogloss'
    best_params['tree_method'] = 'hist'
    best_params['random_state'] = 42

    # 3. Train the model on the entire dataset
    logger.info(f"Training final model on {len(X)} samples...")

    # Map labels from {-1, 0, 1} to {0, 1, 2} for XGBoost
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})

    model = xgb.XGBClassifier(**best_params)
    model.fit(X, y_mapped, verbose=False)

    logger.success("Final model training complete.")

    # 4. Save the trained model
    model_path = MODEL_DIR / "final_model_EURUSD.joblib"
    joblib.dump(model, model_path)
    logger.success(f"Saved final trained model to {model_path}")

if __name__ == "__main__":
    main()

