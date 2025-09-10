"""Script to evaluate the performance of a trained XGBoost model."""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine

from src.features.pipeline import create_all_features
from src.labeling.dynamic_labels import create_dynamic_labels
from src.logger import logger
from src.modeling.financial_objective import calculate_financial_objective

# --- Configuration ---
DB_PATH = project_root / "data" / "m5_trading.db"
MODEL_PATH = "C:/Users/wcw17/Documents/GitHub/project-5/Project-5/data/models/EURUSDm_model_20250901_065505.joblib"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2025-08-15"
TEST_SPLIT_DATE = "2025-06-01"  # Use last ~2.5 months for testing


def load_and_prepare_data():
    """Loads, features, and labels the entire dataset."""
    logger.info("Loading and preparing data for evaluation...")
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
    y = labels.loc[aligned_index].filter(like="hit_")
    return X, y


def main():
    """Main evaluation function."""
    logger.info(f"--- Starting Model Evaluation for {MODEL_PATH} ---")

    # Load model and data
    model = joblib.load(MODEL_PATH)
    X, y = load_and_prepare_data()

    # Time-based split
    X_test = X[X.index >= TEST_SPLIT_DATE]
    y_test = y[y.index >= TEST_SPLIT_DATE]

    logger.info(f"Test set shape: {X_test.shape}")

    # Generate predictions (probabilities)
    pred_probas_list = [est.predict_proba(X_test)[:, 1] for est in model.estimators_]
    pred_probas_df = pd.DataFrame(
        np.array(pred_probas_list).T, index=X_test.index, columns=y.columns
    )

    # Calculate financial objective
    trade_data = X_test[["close"]].copy()
    score = calculate_financial_objective(y_test, pred_probas_df, trade_data)
    logger.success(f"Financial Objective Score on Test Set: {score:.4f}")
    print(f"Financial Objective Score on Test Set: {score:.4f}")

    # Feature Importance
    # Average the feature importances from all estimators in the MultiOutputClassifier
    feature_importances = (
        pd.DataFrame(
            [est.feature_importances_ for est in model.estimators_],
            columns=X_test.columns,
        )
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12.0, 10.0))
    sns.barplot(
        x=feature_importances.values[:20],
        y=feature_importances.index[:20],
        palette="viridis",
    )
    plt.title("Top 20 Feature Importances (Gain)")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    # Create reports directory if it doesn't exist
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    plt.tight_layout()
    plt.savefig(reports_dir / "feature_importance.png")
    logger.info(f"Feature importance plot saved to reports/feature_importance.png")


if __name__ == "__main__":
    main()
