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


# --- Constants ---
DB_PATH = project_root / "data" / "m5_trading.db"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2025-08-31"
HOLD_PERIOD_BARS = [288]  # 24-hour horizon
DEBUG_SAMPLE_COUNT = 5 # Number of samples to print for debugging

# --- Configurations to Test ---
TEST_CONFIGS = [
    {"rr_multiples": [3.0, 5.0, 10.0, 15.0, 20.0], "atr_mult_base": 1.2},
    {"rr_multiples": [3.0, 5.0, 10.0, 15.0, 20.0], "atr_mult_base": 1.0},
    {"rr_multiples": [3.0, 5.0, 10.0, 15.0, 20.0], "atr_mult_base": 1.5},
]

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
            spread_pips=2.0,
        )

        labels, _ = generate_labels_with_mlfinpy(data, tb_cfg)

        # Analyze the generated labels
        total_samples = len(labels)
        long_cols = [c for c in labels.columns if 'long' in c]
        short_cols = [c for c in labels.columns if 'short' in c]

        total_long_hits = labels[long_cols].sum().sum()
        total_short_hits = labels[short_cols].sum().sum()

        logger.info(f"  Total Long Hits: {total_long_hits}")
        logger.info(f"  Total Short Hits: {total_short_hits}")

        for col in labels.columns:
            hits = labels[col].sum()
            hit_rate = (hits / total_samples) * 100
            logger.debug(f"    - Label '{col}': {hits} hits ({hit_rate:.2f}%)")

        total_hits = labels.sum().sum()
        logger.success(f"Total Hits for Config {i+1}: {total_hits}")

if __name__ == "__main__":
    main()

