"""
This script builds the final, clean training dataset using the meta-labeling strategy.

It is designed to be importable for use in other scripts (v2 pipeline).
"""
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from loguru import logger

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.features.pipeline import create_all_features
from src.labeling.triple_barrier_afml import TripleBarrierConfig
from src.labeling.trend_scanning_labeling import generate_trend_scanning_meta_labels

# --- Constants ---
DB_PATH = project_root / "data" / "m5_trading.db"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2025-08-15"
HOLD_PERIOD_BARS = [288]  # 24-hour horizon

def build_dataset_v2():
    """Loads and prepares data using the meta-labeling pipeline."""
    logger.info("Loading and preparing data for v2 pipeline...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    query = f"SELECT * FROM bars WHERE symbol = '{SYMBOL}' AND time BETWEEN '{START_DATE}' AND '{END_DATE}' ORDER BY time ASC"
    data = pd.read_sql(query, engine, index_col="time", parse_dates=["time"])
    if data.index.has_duplicates:
        data = data[~data.index.duplicated(keep="last")]

    featured_data = create_all_features(data, SYMBOL)

    tb_cfg = TripleBarrierConfig(
        horizons=HOLD_PERIOD_BARS,
        rr_multiples=[1.0, 1.5, 2.0, 2.5],
        vol_method="atr",
        atr_period=100,
        vol_window=288,
        atr_mult_base=1.5,
        include_short=True,
        spread_pips=2.0,
    )
    labels, sample_info = generate_trend_scanning_meta_labels(data, tb_cfg)

    label_cols = [c for c in labels.columns if c.startswith('meta_label_')]
    if not label_cols:
        raise ValueError("No meta-labels were generated.")

    valid_sample_info = sample_info.dropna(subset=['t1', 'side'])
    common_index = featured_data.index.intersection(labels.index).intersection(valid_sample_info.index)

    X_features = featured_data.loc[common_index]
    y = labels.loc[common_index][label_cols]
    sample_info = valid_sample_info.loc[common_index]

    X = X_features.join(sample_info['side'])

    logger.success(f"v2 dataset created successfully with {len(X)} samples.")
    return X, y, sample_info

if __name__ == '__main__':
    build_dataset_v2()

