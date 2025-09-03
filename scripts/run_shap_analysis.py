"""Script to run SHAP explainability analysis on the optimized model."""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from loguru import logger
from sqlalchemy import create_engine

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.features.pipeline import create_all_features
from src.labeling.dynamic_labels import create_dynamic_labels

# --- Logger Configuration ---
log_file_path = project_root / "logs" / "shap_analysis.log"
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
MODEL_PATH = (
    project_root / "data" / "models" / "EURUSDm_optimized_model_20250902_081724.joblib"
)
REPORTS_DIR = project_root / "reports"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2025-08-15"
HOLD_PERIOD_BARS = 4 * 12  # 4h hold period


def load_data():
    """Loads and prepares data from the database."""
    logger.info("Loading and preparing data...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    query = f"SELECT * FROM bars WHERE symbol = '{SYMBOL}' AND time BETWEEN '{START_DATE}' AND '{END_DATE}' ORDER BY time ASC"
    data = pd.read_sql(query, engine, index_col="time", parse_dates=["time"])
    if data.index.has_duplicates:
        data = data[~data.index.duplicated(keep="last")]

    featured_data = create_all_features(data, SYMBOL)
    labels, _ = create_dynamic_labels(data, horizons=[HOLD_PERIOD_BARS])

    aligned_index = featured_data.index.intersection(labels.index)
    X = featured_data.loc[aligned_index]

    logger.success("Data loading and preparation finished.")
    return X


def main():
    """Main function to run the SHAP analysis."""
    REPORTS_DIR.mkdir(exist_ok=True)
    X = load_data()

    logger.info(f"Loading optimized model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    # The model is a MultiOutputClassifier, with one XGBoost model per output (long/short)
    long_model = model.estimators_[0]
    short_model = model.estimators_[1]

    explainer_long = shap.TreeExplainer(long_model)
    explainer_short = shap.TreeExplainer(short_model)

    logger.info("Calculating SHAP values for long model...")
    shap_values_long = explainer_long.shap_values(X)

    logger.info("Calculating SHAP values for short model...")
    shap_values_short = explainer_short.shap_values(X)

    # --- Generate and Save Global Feature Importance Plots ---
    # For Long Model
    plt.figure()
    shap.summary_plot(shap_values_long, X, show=False, plot_type="bar")
    plt.title("SHAP Feature Importance - Long Model")
    plt.tight_layout()
    long_plot_path = REPORTS_DIR / "shap_importance_long.png"
    plt.savefig(long_plot_path)
    plt.close()
    logger.success(f"Saved long model SHAP plot to {long_plot_path}")

    # For Short Model
    plt.figure()
    shap.summary_plot(shap_values_short, X, show=False, plot_type="bar")
    plt.title("SHAP Feature Importance - Short Model")
    plt.tight_layout()
    short_plot_path = REPORTS_DIR / "shap_importance_short.png"
    plt.savefig(short_plot_path)
    plt.close()
    logger.success(f"Saved short model SHAP plot to {short_plot_path}")


if __name__ == "__main__":
    main()
