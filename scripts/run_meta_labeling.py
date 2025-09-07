"""
Script to run the full meta-labeling pipeline:
1. Generate t_events using the Trend-Scanning method.
2. Apply the Triple-Barrier Method to the t_events to generate primary labels.
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

from src.labeling.robust_trend_scanning import robust_trend_scanning_labels
from mlfinpy.labeling import get_events, add_vertical_barrier, get_bins
from mlfinpy.util import get_daily_vol

# --- Constants ---
DB_PATH = project_root / "data" / "m5_trading.db"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

def main():
    """Loads data and runs the meta-labeling pipeline."""
    logger.info("Starting meta-labeling pipeline...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    query = f"SELECT * FROM bars WHERE symbol = '{SYMBOL}' AND time BETWEEN '{START_DATE}' AND '{END_DATE}' ORDER BY time ASC"
    data = pd.read_sql(query, engine, index_col="time", parse_dates=["time"])
    if data.index.has_duplicates:
        data = data[~data.index.duplicated(keep="last")]
    logger.info(f"Loaded {len(data)} total bars.")

    # --- Step 1: Generate t_events with Trend-Scanning ---
    logger.info("Step 1: Generating t_events with Trend-Scanning...")

    # Constants from the test script
    LOOK_FORWARD_WINDOW = 288  # 24 hours
    EVENT_SAMPLING_RATE = 12 # Run trend-scanning once per hour (12 * 5 mins)

    t_events_sampled = data.index[::EVENT_SAMPLING_RATE]
    ts_labels = robust_trend_scanning_labels(data['close'], t_events=t_events_sampled, look_forward_window=LOOK_FORWARD_WINDOW)
    ts_labels.dropna(inplace=True)

    # Filter for significant events
    dynamic_threshold = ts_labels['t_value'].abs().quantile(0.80)
    significant_trends = ts_labels[ts_labels['t_value'].abs() > dynamic_threshold]
    t_events = significant_trends.index
    side = significant_trends['t_value'].apply(np.sign)
    side.name = 'side'

    logger.success(f"Found {len(t_events)} significant events to label.")

    # --- Step 2: Apply Triple-Barrier Method ---
    # --- Apply Triple-Barrier Method ---
    logger.info("Step 2: Applying Triple-Barrier Method...")
    vol = get_daily_vol(data['close'], lookback=50).ffill().bfill()
    vertical_barriers = add_vertical_barrier(t_events=t_events, close=data['close'], num_days=5)

    pt_multiple = 15.0  # Using a single representative PT multiple as requested
    sl_mult = 1.5

    long_events = side[side == 1].index
    short_events = side[side == -1].index

    all_bins = []

    # --- Process Long Signals ---
    if not long_events.empty:
        events_long = get_events(
            close=data['close'], t_events=long_events, pt_sl=[pt_multiple, sl_mult], target=vol,
            min_ret=0, num_threads=1, vertical_barrier_times=vertical_barriers
        )
        bins_long = get_bins(triple_barrier_events=events_long, close=data['close']).dropna()
        if not bins_long.empty:
            bins_long['side'] = 1
            all_bins.append(bins_long)

    # --- Process Short Signals ---
    if not short_events.empty:
        events_short = get_events(
            close=data['close'], t_events=short_events, pt_sl=[pt_multiple, sl_mult], target=vol,
            min_ret=0, num_threads=1, vertical_barrier_times=vertical_barriers, side_prediction=pd.Series(-1, index=short_events)
        )
        bins_short = get_bins(triple_barrier_events=events_short, close=data['close']).dropna()
        if not bins_short.empty:
            short_labels = pd.Series(0, index=bins_short.index)
            short_labels.loc[bins_short['ret'] > 0] = -1
            short_labels.loc[bins_short['ret'] < 0] = 1
            bins_short['bin'] = short_labels
            bins_short['side'] = -1
            all_bins.append(bins_short)

    # --- Combine, Format, Sample, and Save ---
    if all_bins:
        final_df = pd.concat(all_bins).sort_index()
        final_df['trgt'] = vol.loc[final_df.index]

        # Select and order columns as requested
        final_df = final_df[['ret', 'trgt', 'bin', 'side']]
        final_df.index.name = 'time'

        logger.success(f"Successfully generated detailed labels for {len(final_df)} events.")

        sample_size = min(100, len(final_df))
        result_sample = final_df.sample(n=sample_size, random_state=42)

        output_path = project_root / "reports" / "detailed_meta_labeling_sample.csv"
        result_sample.to_csv(output_path)
        logger.success(f"Saved a random sample of {sample_size} results to {output_path}")
        logger.info("--- Result Sample Analysis ---")
        logger.info(f"\n{result_sample.head().to_string()}")
    else:
        logger.error("No labels were generated in the entire process.")

if __name__ == "__main__":
    main()

