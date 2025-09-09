"""
This script adds the 'spread' column to the 'bars' table in the database if it doesn't already exist.
"""
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
from loguru import logger

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- Constants ---
DB_PATH = project_root / "data" / "m5_trading.db"

def add_spread_column():
    """Adds the spread column to the bars table."""
    engine = create_engine(f"sqlite:///{DB_PATH}")
    
    alter_sql = text("ALTER TABLE bars ADD COLUMN spread REAL")
    
    with engine.connect() as conn:
        try:
            logger.info("Attempting to add 'spread' column to 'bars' table...")
            conn.execute(alter_sql)
            logger.success("Column 'spread' added successfully.")
        except Exception as e:
            if "duplicate column name" in str(e):
                logger.warning("Column 'spread' already exists. No action taken.")
            else:
                logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    add_spread_column()

