"""
Script to run hyperparameter optimization for EURUSDm using Optuna with AFML triple-barrier labels (long + short),
Purged K-Fold CV, and a financial objective function.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.features.pipeline import create_all_features
from src.labeling.trend_scanning_labeling import generate_true_labels
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
<<<<<<< HEAD
from src.labeling.trend_scanning_labeling import generate_trend_scanning_meta_labels as generate_primary_labels
from mlfinpy.cross_validation.cross_validation import PurgedKFold
from src.modeling.optimization import financial_objective_function
=======
>>>>>>> 62ce9b2e17fee7f24ee56398ea656e5178723856

# --- Logger Configuration ---
log_file_path = project_root / "logs" / "optimization.log"
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
)
logger.add(
    log_file_path, rotation="10 MB", retention="10 days", enqueue=True, serialize=False
)

# --- Constants ---
DB_PATH = project_root / "data" / "m5_trading.db"
MODEL_DIR = project_root / "models"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2025-08-31"
VALIDATION_SPLIT_DATE = "2025-01-01"
HOLD_PERIOD_BARS = [288]  # 24-hour horizon
N_TRIALS = 100


def load_data():
    """Loads data and generates true {-1, 0, 1} labels."""
    logger.info("Loading and preparing data...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    query = (
        f"SELECT * FROM bars WHERE symbol = '{SYMBOL}' AND time BETWEEN "
        f"'{START_DATE}' AND '{END_DATE}' ORDER BY time ASC"
    )
    data = pd.read_sql(query, engine, index_col="time", parse_dates=["time"])
    if data.index.has_duplicates:
        data = data[~data.index.duplicated(keep="last")]

    # Generate true labels using our corrected function
    X, y, sample_info = generate_true_labels(
        data,
        pt_mult=20.0,
        sl_mult=1.5,
        num_days=5
    )

    logger.success("Data loading and preparation finished.")
    return X, y, sample_info


def objective(
    trial: optuna.trial.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: list,
) -> float:
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

    logger.info(f"Trial {trial.number}: Starting")
    scores = []
    for i, (train_idx, val_idx) in enumerate(cv_splits):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
<<<<<<< HEAD
        trade_data_val = trade_data.iloc[val_idx].join(X_val['side'])
=======
>>>>>>> 62ce9b2e17fee7f24ee56398ea656e5178723856

        # Map labels from {-1, 0, 1} to {0, 1, 2} for XGBoost
        y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
        y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2})

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train_mapped, eval_set=[(X_val, y_val_mapped)], early_stopping_rounds=10, verbose=False)

<<<<<<< HEAD
            if y_train_label.nunique() < 2:
                continue
=======
        preds = model.predict(X_val)
        # Map predictions back to original labels for scoring
        preds_mapped_back = pd.Series(preds).map({0: -1, 1: 0, 2: 1}).values
>>>>>>> 62ce9b2e17fee7f24ee56398ea656e5178723856

        score = f1_score(y_val, preds_mapped_back, average='weighted')
        scores.append(score)
        logger.info(f"Trial {trial.number}, Fold {i+1}: F1-Score = {score:.4f}")

<<<<<<< HEAD
            if (
                hasattr(estimator, "classes_")
                and len(estimator.classes_) > 1
                and estimator.classes_[1] == 1
            ):
                y_pred_proba_fold[label] = estimator.predict_proba(X_val)[:, 1]
            else:
                y_pred_proba_fold[label] = 0.0

        if not y_pred_proba_fold.empty:
            score = financial_objective_function(trial, y_pred_proba_fold, trade_data_val)
            scores.append(score)
            logger.info(f"Trial {trial.number}, Fold {i+1}: Score = {score}")

    mean_score = float(np.mean(scores)) if scores else -1.0
    logger.info(f"Trial {trial.number}: Finished with mean score = {mean_score}")
=======
    mean_score = float(np.mean(scores)) if scores else 0.0
    logger.info(f"Trial {trial.number}: Finished with mean F1-score = {mean_score:.4f}")
>>>>>>> 62ce9b2e17fee7f24ee56398ea656e5178723856
    return mean_score


def main():
    """Main function to run the optimization study."""
    logger.info(f"Using XGBoost version: {xgb.__version__}")
    X, y, sample_info = load_data()

    # Now y is a Series, no need to filter columns
    y_filtered = y

    # --- TimeSeriesSplit Cross-Validation ---
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_splits = list(tscv.split(X))

    logger.debug(f"Shapes before PurgedKFold: X={X.shape}, y={y.shape}, sample_info={sample_info.shape}")
    logger.debug(f"Sample of X index:\n{X.head().index}")
    logger.debug(f"Sample of y index:\n{y_filtered.head().index}")
    logger.debug(f"Sample of sample_info index:\n{sample_info.head().index}")

    # --- Optuna Study ---
    logger.info(f"--- Starting Optimization ({N_TRIALS} Trials) ---")

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=30, interval_steps=10
    )
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        lambda trial: objective(trial, X, y, cv_splits),
        n_trials=N_TRIALS,
        n_jobs=-1,  # Use all available CPU cores
    )

    logger.info(f"Best trial score: {study.best_value}")
    logger.info(f"Best trial params: {study.best_params}")

    # Save the best hyperparameters to a file
    best_params = {k: v for k, v in study.best_params.items() if "threshold" not in k}
    params_path = MODEL_DIR / f"{SYMBOL}_best_primary_params.json"
    import json

    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=4)
    logger.success(f"Saved best primary model hyperparameters to {params_path}")
    logger.success("Optimization script finished successfully.")


if __name__ == "__main__":
    main()
