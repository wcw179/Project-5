"""
Script to test the Trend-Scanning algorithm for generating t_events.

This script focuses solely on Step 1 of the meta-labeling strategy:
identifying moments of clear trends in the market.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from loguru import logger

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.labeling.robust_trend_scanning import robust_trend_scanning_labels as trend_scanning_labels

# --- Constants ---
DB_PATH = project_root / "data" / "m5_trading.db"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"  # Shorten date range for faster testing
LOOK_FORWARD_WINDOW = 288  # 24 hours
T_VALUE_THRESHOLD = 2.0 # Standard threshold for statistical significance
EVENT_SAMPLING_RATE = 12 # Run trend-scanning once per hour (12 * 5 mins)

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
    # To improve performance, we sample the events we run the expensive algorithm on.
    t_events_sampled = data.index[::EVENT_SAMPLING_RATE]
    logger.info(f"Running trend-scanning on {len(t_events_sampled)} sampled events (1 every {EVENT_SAMPLING_RATE*5} mins)...")
    ts_labels = trend_scanning_labels(data['close'], t_events=t_events_sampled, look_forward_window=LOOK_FORWARD_WINDOW)
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
    # Instead of a fixed threshold, let's use a dynamic one based on the distribution.
    # We'll filter for t-values in the top 20% (above the 80th percentile).
    dynamic_threshold = ts_labels['t_value'].abs().quantile(0.80)
    logger.info(f"Dynamic t-value threshold (80th percentile): {dynamic_threshold:.2f}")

    significant_trends = ts_labels[ts_labels['t_value'].abs() > dynamic_threshold]
    t_events = significant_trends.index
    side_prediction = significant_trends['t_value'].apply(np.sign)

    logger.info(f"--- Analysis of Significant t_events (Threshold > {dynamic_threshold:.2f}) ---")
    logger.success(f"Found {len(t_events)} significant events ({len(t_events)/len(t_events_sampled)*100:.2f}% of sampled events).")

    if not t_events.empty:
        up_trends = (side_prediction == 1).sum()
        down_trends = (side_prediction == -1).sum()
        logger.info(f"  - Upward trends: {up_trends} ({up_trends/len(t_events)*100:.2f}%)")
        logger.info(f"  - Downward trends: {down_trends} ({down_trends/len(t_events)*100:.2f}%)")

    # --- Save Results for Analysis ---
    results_df = pd.concat([significant_trends, side_prediction.rename('side')], axis=1)
    output_path = project_root / "reports" / "trend_scanning_analysis.csv"
    results_df.to_csv(output_path)
    logger.success(f"Saved analysis results to {output_path}")

if __name__ == "__main__":
    main()

