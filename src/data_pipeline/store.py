"""In-memory data store for historical and real-time data."""

from typing import Dict

import pandas as pd
from sqlalchemy import create_engine

from src.config import BASE_DIR, config
from src.logger import logger

DB_PATH = BASE_DIR / "data" / "m5_trading.db"


class DataStore:
    """Manages loading and accessing of time-series data for all symbols."""

    def __init__(self):
        self._engine = create_engine(f"sqlite:///{DB_PATH}")
        self.buffer_size = config.data.rolling_window_hours * 12  # M5 bars
        self._data: Dict[str, pd.DataFrame] = {}
        self.load_initial_data()

    def load_initial_data(self):
        """Loads the initial data window for each symbol."""
        hours = config.data.rolling_window_hours
        limit = hours * 12

        logger.info(
            f"Loading initial data store: last {hours} hours ({limit} bars) per symbol."
        )

        for symbol in config.data.symbols:
            logger.debug(f"Loading data for {symbol}...")
            try:
                query = f"""
                    SELECT time, open, high, low, close, volume
                    FROM bars
                    WHERE symbol = '{symbol}'
                    ORDER BY time DESC
                    LIMIT {limit}
                """
                df = pd.read_sql(
                    query, self._engine, index_col="time", parse_dates=["time"]
                )
                df = df.iloc[::-1]  # Oldest first

                # Remove duplicate timestamps, keeping the last entry
                if df.index.has_duplicates:
                    logger.warning(
                        f"Found {df.index.duplicated().sum()} duplicate timestamps in {symbol}. Removing them."
                    )
                    df = df[~df.index.duplicated(keep="last")]

                if self.validate_data(df, symbol):
                    self._data[symbol] = df
                    logger.success(f"Loaded and validated {len(df)} bars for {symbol}.")

            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")

    def validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """Validates the integrity of the incoming data."""
        if df.empty:
            logger.warning(f"No data found for {symbol}. Validation skipped.")
            return False

        # 1. Schema Check
        expected_columns = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in expected_columns):
            logger.error(f"Schema validation failed for {symbol}. Missing columns.")
            return False

        # 2. Gap Check (for M5 data)
        time_diffs = df.index.to_series().diff().dropna()
        gaps = time_diffs[time_diffs > pd.Timedelta(minutes=5)]
        if not gaps.empty:
            logger.warning(f"Detected {len(gaps)} gaps in data for {symbol}.")

        # 3. Outlier Check (simple z-score)
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            if (z_scores.abs() > 3).any():
                logger.warning(f"Outlier detected in '{col}' for {symbol}.")

        return True

    def get_data(self, symbol: str) -> pd.DataFrame | None:
        """Returns the DataFrame for a given symbol."""
        return self._data.get(symbol)

    def get_all_data(self) -> Dict[str, pd.DataFrame]:
        """Returns the entire data store as DataFrames."""
        return {symbol: self.get_data(symbol) for symbol in self._data}


_data_store_instance = None


def get_data_store() -> DataStore:
    """Returns the singleton instance of the DataStore."""
    global _data_store_instance
    if _data_store_instance is None:
        _data_store_instance = DataStore()
    return _data_store_instance
