"""
Script to test the full Trend-Scanning Meta-Labeling pipeline.
This creates a new, isolated test to verify the logic in trend_scanning_labeling.py
without modifying any existing user files.
"""
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from loguru import logger
import numpy as np

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.labeling.trend_scanning_labeling import generate_trend_scanning_meta_labels
from src.labeling.triple_barrier_afml import TripleBarrierConfig

# --- Constants ---
DB_PATH = project_root / "data" / "m5_trading.db"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2023-03-31"  # Use a shorter range for a quicker test

def main():
    """Loads data and runs the full labeling pipeline for a test."""
    logger.info("--- Starting New Labeling Logic Test ---")
    
    # --- Load Data ---
    logger.info("Loading data...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    query = f"SELECT * FROM bars WHERE symbol = '{SYMBOL}' AND time BETWEEN '{START_DATE}' AND '{END_DATE}' ORDER BY time ASC"
    data = pd.read_sql(query, engine, index_col="time", parse_dates=["time"])
    if data.index.has_duplicates:
        data = data[~data.index.duplicated(keep="last")]
    logger.info(f"Loaded {len(data)} total bars.")

    # --- Configure Labeling ---
    logger.info("Configuring triple-barrier settings...")
    config = TripleBarrierConfig(
        horizons=[288],  # 24-hour horizon
        rr_multiples=[15.0], # R/R of 15
        vol_window=288,
        atr_mult_base=1.5
    )

    # --- Generate Labels ---
    logger.info("Generating trend-scanning meta-labels...")
    labels_df, sample_info_df = generate_trend_scanning_meta_labels(
        data=data,
        config=config,
        ts_window=20,
        ts_t_value_threshold=2.0
    )

    # --- Analyze Results ---
    if labels_df.empty:
        logger.error("Label generation failed. Resulting DataFrame is empty.")
    else:
        logger.success("Label generation successful!")
        logger.info(f"Labels DataFrame shape: {labels_df.shape}")
        logger.info(f"Sample Info DataFrame shape: {sample_info_df.shape}")
        logger.info("Labels DataFrame head:")
        logger.info(f"\n{labels_df.head()}")
        logger.info("Sample Info DataFrame head:")
        logger.info(f"\n{sample_info_df.head()}")
        
        # Check for actual labels
        if not labels_df.empty:
            label_col = labels_df.columns[0]
            num_labels = labels_df[label_col].sum()
            logger.info(f"Number of '1' labels generated in '{label_col}': {num_labels}")

if __name__ == "__main__":
    main()

