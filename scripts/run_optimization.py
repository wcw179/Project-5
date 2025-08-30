"""Script to run Optuna hyperparameter optimization for XGBoost."""

import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine

from src.features.pipeline import create_all_features
from src.labeling.dynamic_labels import create_dynamic_labels
from src.logger import logger
from src.model_validation.purged_k_fold import PurgedKFold

DB_PATH = project_root / "data" / "m5_trading.db"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2025-08-15"

# --- Data Loading (cached for performance during optimization) ---


def load_and_prepare_data():
    """Loads, features, and labels data once."""
    logger.info("Loading and preparing data for optimization...")
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
    labels, sample_info = create_dynamic_labels(raw_data)

    aligned_index = featured_data.index.intersection(labels.index)
    X = featured_data.loc[aligned_index]
    y = labels.loc[aligned_index]["hit_5R_1"]  # Single target for now
    sample_info = sample_info.loc[aligned_index]
    return X, y, sample_info


X, y, sample_info = load_and_prepare_data()


def objective(trial: optuna.Trial) -> float:
    """Defines one optimization trial."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "n_jobs": -1,
        "random_state": 42,
        "tree_method": "hist",
        "base_score": 0.5,
    }

    model = xgb.XGBClassifier(**params)
    pkf = PurgedKFold(n_splits=5, embargo_td=pd.Timedelta(hours=24))
    scores = []

    for fold, (train_idx, test_idx) in enumerate(pkf.split(X, sample_info=sample_info)):
        if len(train_idx) == 0:  # Skip folds with no training data
            continue
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        scores.append(accuracy_score(y_test, preds))

    return np.mean(scores)


def main():
    """Main function to run the optimization."""
    logger.info("--- Starting Optuna Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    logger.success("--- Optimization Finished ---")
    logger.info(f"Best trial score: {study.best_value:.4f}")
    logger.info("Best trial parameters:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")


if __name__ == "__main__":
    main()
