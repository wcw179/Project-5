"""
A simplified and robust script to import historical data from CSV files into the SQLite database.
This version prioritizes clarity and correctness over complex SQL operations.
"""
import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from loguru import logger

# --- Setup ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
DB_PATH = project_root / "data" / "m5_trading.db"
DATA_DIR = project_root / "data"
CSV_FILES = ["AUDUSDm.csv", "GBPUSDm.csv", "USDJPYm.csv", "EURUSDm.csv", "XAUUSDm.csv"]

def import_data():
    """Main function to process CSV files and upsert into the database."""
    engine = create_engine(f"sqlite:///{DB_PATH}")
    inspector = inspect(engine)
    table_name = "bars"

    for file_name in CSV_FILES:
        file_path = DATA_DIR / file_name
        symbol = file_name.replace('.csv', '')
        logger.info(f"--- Processing {symbol} from {file_path} ---")

        if not file_path.exists():
            logger.warning(f"File not found: {file_path}. Skipping.")
            continue

        try:
            # --- Step 1: Load and Clean Data ---
            # Load new data from CSV
            df_csv = pd.read_csv(file_path)
            df_csv['time'] = pd.to_datetime(df_csv['time'], errors='coerce', utc=True)
            df_csv.dropna(subset=['time'], inplace=True)
            df_csv['symbol'] = symbol # CRITICAL FIX: Assign the symbol to the new data
            # CRITICAL FIX: Remove duplicates from the source file itself
            df_csv.drop_duplicates(subset=['symbol', 'time'], keep='first', inplace=True)

            # --- Step 2: Filter to match database schema ---
            db_columns = [col['name'] for col in inspector.get_columns(table_name)]
            df_to_insert = df_csv[[col for col in db_columns if col in df_csv.columns]]

            # --- Step 3: Overwrite data in a single transaction ---
            logger.info(f"Writing {len(df_to_insert)} records for {symbol}. This will overwrite existing data.")
            with engine.begin() as conn:
                # Clear all old data for the symbol
                conn.execute(text(f"DELETE FROM {table_name} WHERE symbol = '{symbol}'"))
                # Write the new, clean data
                df_to_insert.to_sql(table_name, conn, if_exists='append', index=False, chunksize=10000)
            logger.success(f"Successfully wrote data for {symbol}.")

        except Exception as e:
            logger.error(f"Database operation failed for {symbol}: {e}")

    logger.info("--- Data ingestion process finished. ---")

if __name__ == "__main__":
    import_data()

