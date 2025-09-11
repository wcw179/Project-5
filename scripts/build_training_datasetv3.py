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

START_DATE = "2024-09-05"
END_DATE = "2025-09-05"

def build_dataset(symbol: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Runs the full pipeline to generate features, labels, and sample info.

    Returns:
        A tuple of (X, y, sample_info).
        X: DataFrame of features.
        y: Series of labels.
        sample_info: DataFrame containing t1 timestamps for purging.
    """
    # 1. Load Raw Data
    logger.info(f"Loading raw data for {symbol} from {START_DATE} to {END_DATE}...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    query = f"SELECT * FROM bars WHERE symbol = '{symbol}' AND time BETWEEN '{START_DATE}' AND '{END_DATE}' ORDER BY time ASC"
    data = pd.read_sql(query, engine, index_col="time", parse_dates=["time"])
    if data.index.has_duplicates:
        data = data[~data.index.duplicated(keep="last")]
    logger.info(f"Loaded {len(data)} total bars for {symbol}.")

    # 2. Create Features
    logger.info(f"Generating full feature set for {symbol}...")
    featured_data = create_all_features(data, symbol=symbol)
    close = featured_data['close']

    # 3. Create Labels
    logger.info(f"Generating labels for {symbol}...")
    LOOK_FORWARD_WINDOW = 288
    EVENT_SAMPLING_RATE = 1
    t_events_sampled = close.index[::EVENT_SAMPLING_RATE]
    ts_labels = robust_trend_scanning_labels(close, t_events=t_events_sampled, look_forward_window=LOOK_FORWARD_WINDOW)
    ts_labels.dropna(inplace=True)

    dynamic_threshold = ts_labels['t_value'].abs().quantile(0.80)
    significant_trends = ts_labels[ts_labels['t_value'].abs() > dynamic_threshold]
    t_events = significant_trends.index
    side = significant_trends['t_value'].apply(np.sign)
    side.name = 'side'
    logger.success(f"Found {len(t_events)} significant events to label for {symbol}.")

    vol = get_daily_vol(close, lookback=50).ffill().bfill()
    vertical_barriers = add_vertical_barrier(t_events=t_events, close=close, num_days=5)

    pt_sl = [20.0, 1.5]
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
        logger.error(f"No labels were generated for {symbol}.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    labels_df = pd.concat(all_bins).sort_index()
    events_df = pd.concat(all_events).sort_index()

    # 4. Combine and Finalize
    logger.info(f"Combining features and labels for {symbol}...")
    common_index = featured_data.index.intersection(labels_df.index)

    X = featured_data.loc[common_index]

    # Create a comprehensive y DataFrame
    y_df = labels_df.loc[common_index].copy()
    y_df['t1'] = events_df.loc[common_index, 't1']
    y_df.rename(columns={'bin': 'label'}, inplace=True)

    # Add symbol column for multi-symbol backtesting
    X['symbol'] = symbol
    y_df['symbol'] = symbol

    # sample_info is now effectively the same as y_df
    sample_info = y_df[['t1', 'symbol']].copy()

    logger.success(f"Final dataset created for {symbol} with {len(X)} samples.")
    return X, y_df, sample_info

def main():
    """Runs the data generation for a list of symbols and saves the full dataset."""
    logger.info("--- Starting Full Dataset Creation (Multi-Symbol) ---")

    symbols = ["EURUSDm", "GBPUSDm", "USDJPYm", "AUDUSDm", "XAUUSDm"]
    all_X = []
    all_y = []
    all_sample_info = []

    for sym in symbols:
        X, y, sample_info = build_dataset(symbol=sym)
        if not X.empty:
            all_X.append(X)
            all_y.append(y)
            all_sample_info.append(sample_info)

    if not all_X:
        logger.error("Pipeline finished with no data for any symbol.")
        return

    # Combine all dataframes
    X_full = pd.concat(all_X)
    y_full = pd.concat(all_y)
    sample_info_full = pd.concat(all_sample_info)

    # --- Imbalance Handling ---
    logger.info("Checking for label imbalance...")
    label_counts = y_full.value_counts()
    logger.info(f"Label distribution:\n{label_counts}")
    imbalance_ratio = label_counts.min() / label_counts.max()
    if imbalance_ratio < 0.5:
        logger.warning(f"Significant label imbalance detected! Ratio: {imbalance_ratio:.2f}")

    # --- Saving Final Dataset ---
    data_dir = project_root / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)

    X_full_path = data_dir / "X_full_v4.parquet"
    y_full_path = data_dir / "y_full_v4.parquet"
    sample_info_path = data_dir / "sample_info_full_v4.parquet"

    # Save features
    X_full.to_parquet(X_full_path)

    y_full.to_parquet(y_full_path)

    # Save sample info
    sample_info_full.to_parquet(sample_info_path)

    logger.success("Full dataset saved successfully in Parquet format:")
    logger.info(f"- Features: {X_full_path}")
    logger.info(f"- Labels: {y_full_path}")
    logger.info(f"- Sample Info: {sample_info_path}")

if __name__ == "__main__":
    main()
