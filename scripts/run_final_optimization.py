"""
Script to run hyperparameter optimization for the final model using Optuna,
Purged K-Fold CV, and the full feature/label generation pipeline.
"""
import sys
from pathlib import Path
import json
import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.metrics import f1_score

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from scripts.build_training_dataset import build_dataset
from sklearn.model_selection import TimeSeriesSplit

# --- Logger Configuration ---
log_file_path = project_root / "logs" / "final_optimization.log"
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>")
logger.add(log_file_path, rotation="10 MB", retention="10 days", enqueue=True, serialize=False)

# --- Constants ---
MODEL_DIR = project_root / "models"
SYMBOL = "EURUSDm"
N_TRIALS = 100

def objective(trial: optuna.trial.Trial, X: pd.DataFrame, y: pd.Series, cv_splits: list) -> float:
    """Optuna objective function to train and evaluate models using F1-score."""
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'tree_method': 'hist',
        'random_state': 42,
    }

    logger.info(f"Trial {trial.number}: Starting with params {params}")
    scores = []
    for i, (train_idx, val_idx) in enumerate(cv_splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
        y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2})

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train_mapped, eval_set=[(X_val, y_val_mapped)], early_stopping_rounds=10, verbose=False)

        preds = model.predict(X_val)
        preds_mapped_back = pd.Series(preds).map({0: -1, 1: 0, 2: 1}).values

        score = f1_score(y_val, preds_mapped_back, average='weighted')
        scores.append(score)
    
    mean_score = float(np.mean(scores))
    logger.info(f"Trial {trial.number}: Finished with mean F1-score = {mean_score:.4f}")
    return mean_score

def main():
    """Main function to run the optimization study."""
    logger.info(f"--- Starting Final Model Optimization ({N_TRIALS} Trials) ---")
    
    # 1. Build Dataset
    X, y, sample_info = build_dataset()
    if X.empty:
        logger.error("Dataset creation failed. Exiting optimization.")
        return

    # 2. Setup TimeSeriesSplit Cross-Validation
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_splits = list(tscv.split(X))

    # 3. Run Optuna Study
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X, y, cv_splits),
        n_trials=N_TRIALS,
        n_jobs=-1
    )

    # 4. Save Best Results
    logger.info(f"Best trial score: {study.best_value}")
    logger.info(f"Best trial params: {study.best_params}")

    params_path = MODEL_DIR / f"{SYMBOL}_final_best_params.json"
    with open(params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    logger.success(f"Saved best hyperparameters to {params_path}")

if __name__ == "__main__":
    main()

