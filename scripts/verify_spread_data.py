"""
Script to verify the state of the 'spread' column in the 'bars' table.
"""
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, inspect
from loguru import logger

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

DB_PATH = project_root / "data" / "m5_trading.db"

def verify_data():
    """Connects to the database and checks the spread column."""
    engine = create_engine(f"sqlite:///{DB_PATH}")
    inspector = inspect(engine)
    columns = [col['name'] for col in inspector.get_columns('bars')]

    if 'spread' not in columns:
        logger.error("Verification Failed: The 'spread' column does not exist in the 'bars' table.")
        return

    logger.info("Column 'spread' exists in the 'bars' table.")

    query = """
    SELECT 
        symbol,
        COUNT(spread) as non_null_spread_count,
        COUNT(*) as total_records
    FROM 
        bars
    GROUP BY
        symbol;
    """
    
    df = pd.read_sql(query, engine)

    print("\n--- Spread Column Verification Results ---")
    print(df.to_markdown(index=False))

    if df['non_null_spread_count'].sum() == 0:
        logger.warning("Verification Result: The 'spread' column exists but contains no data.")
    else:
        logger.success("Verification Result: The 'spread' column contains data.")

if __name__ == "__main__":
    verify_data()

