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
from src.labeling.triple_barrier_afml import (
    TripleBarrierConfig,
    generate_primary_labels,
)
from src.model_validation.purged_k_fold import PurgedKFold
from src.modeling.optimization import financial_objective_function

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
END_DATE = "2025-08-15"
VALIDATION_SPLIT_DATE = "2025-01-01"
HOLD_PERIOD_BARS = [12, 48, 144, 288]  # 1h, 4h, 12h, 24h
N_TRIALS = 100


def load_data():
    """Loads and prepares data from the database and generates AFML labels."""
    logger.info("Loading and preparing data...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    query = (
        f"SELECT * FROM bars WHERE symbol = '{SYMBOL}' AND time BETWEEN "
        f"'{START_DATE}' AND '{END_DATE}' ORDER BY time ASC"
    )
    data = pd.read_sql(query, engine, index_col="time", parse_dates=["time"])
    if data.index.has_duplicates:
        data = data[~data.index.duplicated(keep="last")]

    # Feature engineering
    featured_data = create_all_features(data, SYMBOL)

    # AFML triple-barrier labels (long + short)
    tb_cfg = TripleBarrierConfig(
        horizons=HOLD_PERIOD_BARS,  # list of bars
        rr_multiples=[3.0, 5.0, 10.0, 15.0],
        vol_method="atr",
        atr_period=100,
        vol_window=288,
        atr_mult_base=2.0,
        include_short=True,  # ensure both sides are considered
        spread_pips=2.0,
    )
    labels, sample_info = generate_primary_labels(data, SYMBOL, tb_cfg)

    # Targets: long/short hit labels
    label_cols = [
        col
        for col in labels.columns
        if col.startswith("long_hit_") or col.startswith("short_hit_")
    ]

    aligned_index = featured_data.index.intersection(labels.index)
    X = featured_data.loc[aligned_index]
    y = labels.loc[aligned_index][label_cols]
    trade_data = data.loc[aligned_index][["high", "low", "close"]]
    sample_info = sample_info.loc[aligned_index]

    logger.success("Data loading and preparation finished.")
    return X, y, trade_data, sample_info


def objective(
    trial: optuna.trial.Trial,
    X: pd.DataFrame,
    y: pd.DataFrame,
    trade_data: pd.DataFrame,
    cv_splits: list,
) -> float:
    """Optuna objective function to train and evaluate models using cross-validation."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "use_label_encoder": False,
        "tree_method": "hist",
        "random_state": 42,
    }

    logger.info(f"Trial {trial.number}: Starting")
    scores = []
    for i, (train_idx, val_idx) in enumerate(cv_splits):
        logger.info(f"Trial {trial.number}: Processing fold {i+1}/{len(cv_splits)}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        trade_data_val = trade_data.iloc[val_idx]

        y_pred_proba_fold = pd.DataFrame(index=X_val.index)

        # Fit a classifier per target label (multi-label)
        for j, label in enumerate(y.columns):
            y_train_label = y_train.iloc[:, j]
            y_val_label = y_val.iloc[:, j]

            # Skip if only one class in training fold
            if y_train_label.nunique() < 2:
                continue

            estimator = xgb.XGBClassifier(**params, early_stopping_rounds=10)
            estimator.fit(
                X_train, y_train_label, eval_set=[(X_val, y_val_label)], verbose=False
            )

            # Ensure positive-class probabilities
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
    return mean_score


def main():
    """Main function to run the optimization study."""
    logger.info(f"Using XGBoost version: {xgb.__version__}")
    X, y, trade_data, sample_info = load_data()

    # Filter out labels with only one class across the entire dataset
    valid_labels = [col for col in y.columns if y[col].nunique() > 1]
    removed = [col for col in y.columns if col not in valid_labels]
    if removed:
        logger.warning(f"Removed {len(removed)} single-class labels.")
    y_filtered = y[valid_labels]

    # --- Purged K-Fold Cross-Validation Setup ---
    n_splits = 5
    pkf = PurgedKFold(n_splits=n_splits, embargo_td=pd.Timedelta(days=1))
    cv_splits = list(pkf.split(X, sample_info=sample_info))

    # --- Optuna Study ---
    logger.info(f"--- Starting Optimization ({N_TRIALS} Trials) ---")

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=30, interval_steps=10
    )
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(
        lambda trial: objective(trial, X, y_filtered, trade_data, cv_splits),
        n_trials=N_TRIALS,
        n_jobs=-1,
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
