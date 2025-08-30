"""Script to generate all features for the entire dataset and save it."""

import sys
from pathlib import Path

import pandas as pd

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine

from src.config import BASE_DIR, config
from src.features.pipeline import create_all_features
from src.logger import logger

DB_PATH = BASE_DIR / "data" / "m5_trading.db"
OUTPUT_PATH = BASE_DIR / "data" / "featured_data.parquet"


def load_full_data_for_symbol(symbol: str, engine) -> pd.DataFrame | None:
    """Loads the full historical data for a single symbol."""
    logger.info(f"Loading full historical data for {symbol}...")
    try:
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM bars
            WHERE symbol = '{symbol}' AND time BETWEEN '2023-01-01' AND '2025-08-15'
            ORDER BY time ASC
        """
        df = pd.read_sql(query, engine, index_col="time", parse_dates=["time"])

        # Remove duplicate timestamps, keeping the last entry
        if df.index.has_duplicates:
            logger.warning(
                f"Found {df.index.duplicated().sum()} duplicate timestamps in {symbol}. Removing them."
            )
            df = df[~df.index.duplicated(keep="last")]

        logger.success(f"Loaded {len(df)} bars for {symbol}.")
        return df
    except Exception as e:
        logger.error(f"Failed to load data for {symbol}: {e}")
        return None


def main():
    """Loads all data, generates features, and saves the result."""
    logger.info("--- Starting Full Feature Generation Process ---")

    engine = create_engine(f"sqlite:///{DB_PATH}")
    all_data = {
        symbol: load_full_data_for_symbol(symbol, engine)
        for symbol in config.data.symbols
    }

    all_featured_dfs = []

    # 2. Generate features for each symbol, processing year by year to manage memory
    for symbol, df in all_data.items():
        if df is None or df.empty:
            logger.warning(f"Skipping feature generation for empty symbol: {symbol}")
            continue

        # Process data in yearly chunks
        years = df.index.year.unique()
        for year in years:
            logger.info(f"Processing {symbol} for year {year}...")
            yearly_df = df[df.index.year == year]

            featured_df = create_all_features(yearly_df, f"{symbol}-{year}")
            featured_df["symbol"] = symbol  # Add symbol identifier
            all_featured_dfs.append(featured_df)

    if not all_featured_dfs:
        logger.error("No data was processed. Aborting.")
        return

    # 3. Combine and save the final dataset
    logger.info("Combining all featured dataframes...")
    final_df = pd.concat(all_featured_dfs)

    logger.info(f"Saving final featured dataset to {OUTPUT_PATH}...")
    try:
        final_df.to_parquet(OUTPUT_PATH, index=True)
        logger.success(
            f"Successfully saved featured data. Final shape: {final_df.shape}"
        )
    except Exception as e:
        logger.error(f"Failed to save data to Parquet: {e}")


if __name__ == "__main__":
    main()
