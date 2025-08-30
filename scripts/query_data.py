"""Utility to query and summarize bar data for a specific date range."""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import config

DB_PATH = project_root / "data" / "m5_trading.db"


def query_date_range(db_path: Path, start_date: str, end_date: str):
    """Queries the database for a specific date range and summarizes the findings."""
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}")
        return

    print(f"--- Querying Database: {db_path.name} ---")
    print(f"Date Range: {start_date} to {end_date}\n")
    engine = create_engine(f"sqlite:///{db_path}")

    try:
        for symbol in config.data.symbols:
            print(f"--- Symbol: {symbol} ---")
            query = f"""
                SELECT time, open, high, low, close, volume
                FROM bars
                WHERE symbol = '{symbol}' AND time BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY time ASC
            """
            df = pd.read_sql(query, engine, parse_dates=["time"])

            if df.empty:
                print("No data found in this date range.")
            else:
                print(f"Found {len(df)} bars.")
                print(f"Data from: {df['time'].min()}")
                print(f"Data to:   {df['time'].max()}")
            print()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        engine.dispose()


if __name__ == "__main__":
    start = "2023-01-01"
    end = "2025-08-15"
    query_date_range(DB_PATH, start, end)
