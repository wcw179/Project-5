"""
Script to test and debug different triple-barrier labeling configurations.
"""
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from loguru import logger

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.labeling.triple_barrier_afml import TripleBarrierConfig
from src.labeling.mlfinpy_labeling import generate_labels_with_mlfinpy
from mlfinpy.util import get_daily_vol

# --- Constants ---
DB_PATH = project_root / "data" / "m5_trading.db"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2025-08-15"
HOLD_PERIOD_BARS = [288]  # 24-hour horizon
DEBUG_SAMPLE_COUNT = 5 # Number of samples to print for debugging

# --- Configurations to Test ---
TEST_CONFIGS = [
    {"rr_multiples": [3.0, 5.0, 10.0, 15.0, 20.0], "atr_mult_base": 1.2},
    {"rr_multiples": [3.0, 5.0, 10.0, 15.0, 20.0], "atr_mult_base": 1.0},
    {"rr_multiples": [3.0, 5.0, 10.0, 15.0, 20.0], "atr_mult_base": 1.5}
]

def debug_labeling_sample(event_start_time, label_info, data, vol, config):
    """Prints detailed debug information for a single labeling event."""
    try:
        entry_price = data.loc[event_start_time, 'close']
        volatility = vol.loc[event_start_time]
        rr = float(label_info['label_name'].split('_rr')[-1])
        is_long = 'long' in label_info['label_name']

        if pd.isna(volatility) or volatility == 0:
            logger.warning(f"Skipping debug for {event_start_time} due to invalid volatility.")
            return

        # Calculate barriers based on trade direction
        if is_long:
            take_profit_price = entry_price + (volatility * rr * config.atr_mult_base)
            stop_loss_price = entry_price - (volatility * config.atr_mult_base)
        else:  # Short
            take_profit_price = entry_price - (volatility * rr * config.atr_mult_base)
            stop_loss_price = entry_price + (volatility * config.atr_mult_base)

        # Get price action within the horizon
        horizon_end_time = event_start_time + pd.Timedelta(minutes=5 * config.horizons[0])
        horizon_data = data.loc[event_start_time:horizon_end_time]

        price_low = horizon_data['low'].min()
        price_high = horizon_data['high'].max()

        logger.debug(f"--- Debugging Event at {event_start_time} for Label {label_info['label_name']} ---")
        logger.debug(f"  Direction: {'LONG' if is_long else 'SHORT'}")
        logger.debug(f"  Entry Price: {entry_price:.5f}, Volatility (ATR): {volatility:.5f}")
        logger.debug(f"  R/R: {rr}, ATR Base: {config.atr_mult_base}")
        logger.debug(f"  >> Take Profit Price: {take_profit_price:.5f}")
        logger.debug(f"  >> Stop Loss Price: {stop_loss_price:.5f}")
        logger.debug(f"  Price Action in Horizon: High={price_high:.5f}, Low={price_low:.5f}")
        logger.debug(f"  Assigned Label: {label_info['hit']}\n")

    except Exception as e:
        logger.error(f"Error debugging event at {event_start_time}: {e}")

def main():
    """Loads data and runs labeling tests for each configuration."""
    logger.info("Loading data for labeling test...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    query = f"SELECT * FROM bars WHERE symbol = '{SYMBOL}' AND time BETWEEN '{START_DATE}' AND '{END_DATE}' ORDER BY time ASC"
    data = pd.read_sql(query, engine, index_col="time", parse_dates=["time"])
    if data.index.has_duplicates:
        data = data[~data.index.duplicated(keep="last")]
    logger.info(f"Loaded {len(data)} bars.")

    for i, test in enumerate(TEST_CONFIGS):
        logger.info(f"--- Running Test Configuration {i+1}/{len(TEST_CONFIGS)} ---")
        logger.info(f"Parameters: R/R Multiples={test['rr_multiples']}, ATR Base={test['atr_mult_base']}")

        tb_cfg = TripleBarrierConfig(
            horizons=HOLD_PERIOD_BARS,
            rr_multiples=test['rr_multiples'],
            vol_method="atr",
            atr_period=100,
            vol_window=288,
            atr_mult_base=test['atr_mult_base'],
            include_short=True,
        )

        vol = get_daily_vol(data['close'], lookback=tb_cfg.vol_window).ffill().bfill()
        labels, _ = generate_labels_with_mlfinpy(data, tb_cfg)

        # --- Analysis and Debugging ---
        total_samples = len(labels)

        for col in labels.columns:
            hits = labels[col].sum()
            hit_rate = (hits / total_samples) * 100
            logger.info(f"  - Label '{col}': {hits} hits ({hit_rate:.2f}%)")

            # Debug the first few hits for each label type
            hit_indices = labels[labels[col] == 1].index
            for hit_time in hit_indices[:DEBUG_SAMPLE_COUNT]:
                debug_label_info = {'label_name': col, 'hit': 1}
                debug_labeling_sample(hit_time, debug_label_info, data, vol, tb_cfg)

        total_hits = labels.sum().sum()
        total_hit_rate = (total_hits / (total_samples * len(labels.columns))) * 100
        logger.success(f"Total Hits for Config {i+1}: {total_hits} ({total_hit_rate:.2f}% overall hit rate)")

if __name__ == "__main__":
    main()

