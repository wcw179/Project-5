"""
This script builds the final, clean training dataset by combining the best of
the feature engineering and trend-scanning meta-labeling pipelines.
It is designed to be importable for use in other scripts.
"""
import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from loguru import logger

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import the necessary functions from the project's modules
from src.features.pipeline import create_all_features
from src.labeling.robust_trend_scanning import robust_trend_scanning_labels
from mlfinpy.labeling import get_events, add_vertical_barrier, get_bins
from mlfinpy.util import get_daily_vol

# --- Constants ---
DB_PATH = project_root / "data" / "m5_trading.db"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2025-08-31"

def build_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Runs the full pipeline to generate features, labels, and sample info.

    Returns:
        A tuple of (X, y, sample_info).
        X: DataFrame of features.
        y: Series of labels.
        sample_info: DataFrame containing t1 timestamps for purging.
    """
    # 1. Load Raw Data
    logger.info(f"Loading raw data for {SYMBOL} from {START_DATE} to {END_DATE}...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    query = f"SELECT * FROM bars WHERE symbol = '{SYMBOL}' AND time BETWEEN '{START_DATE}' AND '{END_DATE}' ORDER BY time ASC"
    data = pd.read_sql(query, engine, index_col="time", parse_dates=["time"])
    if data.index.has_duplicates:
        data = data[~data.index.duplicated(keep="last")]
    logger.info(f"Loaded {len(data)} total bars.")

    # 2. Create Features
    logger.info("Generating full feature set...")
    featured_data = create_all_features(data, symbol=SYMBOL)
    close = featured_data['close']

    # 3. Create Labels
    logger.info("Generating labels using Trend-Scanning and Triple-Barrier Method...")
    LOOK_FORWARD_WINDOW = 288
    EVENT_SAMPLING_RATE = 12
    t_events_sampled = close.index[::EVENT_SAMPLING_RATE]
    ts_labels = robust_trend_scanning_labels(close, t_events=t_events_sampled, look_forward_window=LOOK_FORWARD_WINDOW)
    ts_labels.dropna(inplace=True)

    dynamic_threshold = ts_labels['t_value'].abs().quantile(0.80)
    significant_trends = ts_labels[ts_labels['t_value'].abs() > dynamic_threshold]
    t_events = significant_trends.index
    side = significant_trends['t_value'].apply(np.sign)
    side.name = 'side'
    logger.success(f"Found {len(t_events)} significant events to label.")

    vol = get_daily_vol(close, lookback=50).ffill().bfill()
    vertical_barriers = add_vertical_barrier(t_events=t_events, close=close, num_days=5)

    pt_sl = [15.0, 1.5]
    long_events = side[side == 1].index
    short_events = side[side == -1].index

    all_bins = []
    all_events = [] # To correctly capture t1

    if not long_events.empty:
        events_long = get_events(close=close, t_events=long_events, pt_sl=pt_sl, target=vol, min_ret=0, num_threads=1, vertical_barrier_times=vertical_barriers)
        bins_long = get_bins(triple_barrier_events=events_long, close=close).dropna()
        bins_long['side'] = 1
        all_bins.append(bins_long)
        all_events.append(events_long)

    if not short_events.empty:
        events_short = get_events(close=close, t_events=short_events, pt_sl=pt_sl, target=vol, min_ret=0, num_threads=1, vertical_barrier_times=vertical_barriers, side_prediction=pd.Series(-1, index=short_events))
        bins_short = get_bins(triple_barrier_events=events_short, close=close).dropna()
        if not bins_short.empty:
            short_labels = pd.Series(0, index=bins_short.index)
            short_labels.loc[bins_short['ret'] > 0] = -1
            short_labels.loc[bins_short['ret'] < 0] = 1
            bins_short['bin'] = short_labels
            bins_short['side'] = -1
            all_bins.append(bins_short)
            all_events.append(events_short)

    if not all_bins:
        logger.error("No labels were generated.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    labels_df = pd.concat(all_bins).sort_index()
    events_df = pd.concat(all_events).sort_index()

    # 4. Combine and Finalize
    logger.info("Combining features and labels...")
    common_index = featured_data.index.intersection(labels_df.index)

    X = featured_data.loc[common_index]
    y = labels_df.loc[common_index, 'bin']


    # Create sample_info from events_df which reliably contains 't1'
    sample_info = pd.DataFrame(index=y.index)
    sample_info['t1'] = events_df.loc[y.index, 't1']

    logger.success(f"Final dataset created with {len(X)} samples.")
    return X, y, sample_info

def main():
    """Runs the data generation and saves multiple samples for verification."""
    logger.info("--- Starting Final Training Dataset Creation (Standalone Run) ---")
    X, y, _ = build_dataset()

    if X.empty:
        logger.error("Pipeline finished with no data.")
        return

    # Create reports directory if it doesn't exist
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    # 5. Create and Save Samples
    sample_size = min(100, len(X))
    random_indices = X.sample(n=sample_size, random_state=42).index

    X_sample = X.loc[random_indices]
    y_sample = y.loc[random_indices]
    combined_sample = X_sample.copy()
    combined_sample['label'] = y_sample

    # Define paths
    features_path = reports_dir / "features_sample.csv"
    labels_path = reports_dir / "labels_sample.csv"
    combined_path = reports_dir / "combined_sample.csv"

    # Save files
    X_sample.to_csv(features_path)
    y_sample.to_csv(labels_path, header=True)
    combined_sample.to_csv(combined_path)

    logger.success(f"Saved {sample_size} random samples to the 'reports' directory:")
    logger.info(f"- Features: {features_path}")
    logger.info(f"- Labels: {labels_path}")
    logger.info(f"- Combined: {combined_path}")

if __name__ == "__main__":
    main()

