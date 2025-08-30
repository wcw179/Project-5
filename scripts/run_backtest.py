"""Script to run a full backtest for a single symbol and strategy."""

import sys
from pathlib import Path

import pandas as pd

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine

from src.backtesting.engine import Backtester
from src.backtesting.metrics import get_performance_summary
from src.backtesting.strategies import MovingAverageCrossover
from src.features.pipeline import create_all_features
from src.logger import logger

DB_PATH = project_root / "data" / "m5_trading.db"
SYMBOL = "EURUSDm"
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"


def load_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """Loads data for a specific symbol and date range."""
    logger.info(f"Loading data for {symbol} from {start_date} to {end_date}...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    try:
        query = f"""
            SELECT time, open, high, low, close, volume
            FROM bars
            WHERE symbol = '{symbol}' AND time BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY time ASC
        """
        df = pd.read_sql(query, engine, index_col="time", parse_dates=["time"])
        if df.index.has_duplicates:
            df = df[~df.index.duplicated(keep="last")]
        logger.success(f"Loaded {len(df)} bars.")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None


def main():
    """Main function to run the backtest."""
    # 1. Load Data
    raw_data = load_data(SYMBOL, START_DATE, END_DATE)
    if raw_data is None or raw_data.empty:
        logger.error("No data loaded, cannot run backtest.")
        return

    # 2. Generate Features
    featured_data = create_all_features(raw_data, SYMBOL)

    # 3. Initialize Strategy
    strategy = MovingAverageCrossover(short_window=20, long_window=50)

    # 4. Run Backtest
    backtester = Backtester(featured_data, strategy)
    equity_curve = backtester.run()

    # 5. Evaluate Performance
    get_performance_summary(equity_curve)


if __name__ == "__main__":
    main()
