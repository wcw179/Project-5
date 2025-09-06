"""
Script to test the Trend-Scanning algorithm for generating t_events.

This script focuses solely on Step 1 of the meta-labeling strategy:
identifying moments of clear trends in the market.
"""
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from loguru import logger

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from mlfinpy.labeling.trend_scanning import trend_scanning_labels

# --- Constants ---
DB_PATH = project_root / "data" / "m5_trading.db"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2025-08-15"
LOOK_FORWARD_WINDOW = 288  # 24 hours
T_VALUE_THRESHOLD = 2.0 # Standard threshold for statistical significance

def main():
    """Loads data and runs the trend-scanning analysis."""
    logger.info("Loading data for trend-scanning test...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    query = f"SELECT * FROM bars WHERE symbol = '{SYMBOL}' AND time BETWEEN '{START_DATE}' AND '{END_DATE}' ORDER BY time ASC"
    data = pd.read_sql(query, engine, index_col="time", parse_dates=["time"])
    if data.index.has_duplicates:
        data = data[~data.index.duplicated(keep="last")]
    logger.info(f"Loaded {len(data)} total bars.")

    # --- Run Trend-Scanning ---
    logger.info(f"Running trend-scanning with a look-forward window of {LOOK_FORWARD_WINDOW} bars...")
    ts_labels = trend_scanning_labels(data['close'], look_forward_window=LOOK_FORWARD_WINDOW)
    ts_labels.dropna(inplace=True)
    logger.info(f"Generated {len(ts_labels)} trend-scanning results.")

    # --- Analyze Results ---
    logger.info("--- Analysis of Trend-Scanning t-values ---")
    logger.info(f"Mean t-value: {ts_labels['t_value'].mean():.2f}")
    logger.info(f"Std Dev of t-values: {ts_labels['t_value'].std():.2f}")
    logger.info(f"Min t-value: {ts_labels['t_value'].min():.2f}")
    logger.info(f"Max t-value: {ts_labels['t_value'].max():.2f}")
    logger.info(f"Median t-value: {ts_labels['t_value'].median():.2f}")

    # --- Filter for Significant t_events ---
    significant_trends = ts_labels[ts_labels['t_value'].abs() > T_VALUE_THRESHOLD]
    t_events = significant_trends.index
    side_prediction = significant_trends['t_value'].apply(np.sign)

    logger.info(f"--- Analysis of Significant t_events (Threshold > {T_VALUE_THRESHOLD}) ---")
    logger.success(f"Found {len(t_events)} significant events ({len(t_events)/len(data)*100:.2f}% of total bars).")
    
    if not t_events.empty:
        up_trends = (side_prediction == 1).sum()
        down_trends = (side_prediction == -1).sum()
        logger.info(f"  - Upward trends: {up_trends} ({up_trends/len(t_events)*100:.2f}%)")
        logger.info(f"  - Downward trends: {down_trends} ({down_trends/len(t_events)*100:.2f}%)")

if __name__ == "__main__":
    main()

