"""Script to run a full training pipeline for a single model and symbol."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine

from src.features.pipeline import create_all_features
from src.labeling.dynamic_labels import create_dynamic_labels
from src.logger import logger
from src.model_validation.purged_k_fold import PurgedKFold
from src.modeling.models import get_balanced_model

DB_PATH = project_root / "data" / "m5_trading.db"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2025-08-15"


def load_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Loads data for a specific symbol and date range."""
    logger.info(f"Loading data for {symbol} from {start_date} to {end_date}...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    try:
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM bars
            WHERE symbol = '{symbol}' AND time BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY time ASC
        """
        df = pd.read_sql(query, engine, index_col="time", parse_dates=["time"])
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]
        logger.success(f"Loaded {len(df)} bars.")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None


def main():
    """Main function to run the training pipeline."""
    # 1. Load Data
    raw_data = load_data(SYMBOL, START_DATE, END_DATE)
    if raw_data is None or raw_data.empty:
        logger.error("No data loaded, cannot run training.")
        return

    # 2. Generate Features
    featured_data = create_all_features(raw_data, SYMBOL)

    # 3. Generate Labels
    labels, sample_info = create_dynamic_labels(raw_data)

    # 4. Align Data
    # Ensure that we only use features and labels that have a corresponding entry
    aligned_index = featured_data.index.intersection(labels.index)
    X = featured_data.loc[aligned_index]
    y = labels.loc[aligned_index]
    sample_info = sample_info.loc[aligned_index]

    # For this test, we'll train on a single target label
    y_single_target = y["hit_5R_1"]

    logger.info(f"Aligned data shapes: X={X.shape}, y={y_single_target.shape}")

    # 5. Train Model with Purged K-Fold CV
    pkf = PurgedKFold(n_splits=5, embargo_td=pd.Timedelta(hours=24))
    model = get_balanced_model()

    scores = []
    for fold, (train_idx, test_idx) in enumerate(pkf.split(X, sample_info=sample_info)):
        logger.info(f"--- Starting Fold {fold+1}/5 ---")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = (
            y_single_target.iloc[train_idx],
            y_single_target.iloc[test_idx],
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        scores.append(accuracy)
        logger.info(f"Fold {fold+1} Accuracy: {accuracy:.4f}")

    logger.success(
        f"Cross-validation finished. Average Accuracy: {pd.Series(scores).mean():.4f}"
    )


if __name__ == "__main__":
    main()
