"""
This script orchestrates the end-to-end pipeline for building and training a multi-symbol trading model.
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
log_file_path = project_root / "logs" / "multi_symbol_pipeline.log"
logger.add(log_file_path, rotation="10 MB", retention="10 days")

# --- Constants ---
MODEL_DIR = project_root / "models"
VALID_SYMBOLS = ['EURUSDm', 'USDJPYm', 'GBPUSDm', 'AUDUSDm' 'XAUUSDm']

def main():
    """Main function to run the multi-symbol training pipeline."""
    logger.info("--- Starting Multi-Symbol Model Training Pipeline ---")

    # 1. Generate and combine data for all symbols
    all_features = []
    all_labels = []
    symbol_map = {name: i for i, name in enumerate(VALID_SYMBOLS)}

    for symbol in VALID_SYMBOLS:
        logger.info(f"Processing data for symbol: {symbol}")
        try:
            X, y, _ = build_dataset(symbol=symbol)
            if X.empty:
                logger.warning(f"No data generated for {symbol}. Skipping.")
                continue
            
            # Add the symbol_id as a categorical feature
            X['symbol_id'] = symbol_map[symbol]
            
            all_features.append(X)
            all_labels.append(y)
            logger.success(f"Successfully processed {symbol}.")
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")

    if not all_features:
        logger.error("No data was generated for any symbol. Exiting.")
        return

    final_X = pd.concat(all_features)
    final_y = pd.concat(all_labels)
    logger.success(f"Combined dataset created with {len(final_X)} total samples.")

    # 2. Train the model on the combined dataset
    # For simplicity, we'll use the best params from the single-symbol optimization
    params_path = MODEL_DIR / "EURUSDm_final_best_params.json"
    if not params_path.exists():
        logger.error(f"Best parameters file not found at {params_path}. Run single-symbol optimization first.")
        return

    with open(params_path, 'r') as f:
        best_params = json.load(f)
    logger.info(f"Loaded best hyperparameters: {best_params}")

    # Add parameters not tuned by Optuna
    best_params.update({
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'random_state': 42
    })

    logger.info(f"Training final multi-symbol model on {len(final_X)} samples...")
    y_mapped = final_y.map({-1: 0, 0: 1, 1: 2})

    model = xgb.XGBClassifier(**best_params)
    model.fit(final_X, y_mapped, verbose=False)
    logger.success("Multi-symbol model training complete.")

    # 4. Save the trained model
    model_path = MODEL_DIR / "multi_symbol_final_model.joblib"
    joblib.dump(model, model_path)
    logger.success(f"Saved final multi-symbol model to {model_path}")

if __name__ == "__main__":
    main()

