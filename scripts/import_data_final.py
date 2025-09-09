"""
A final, robust script to import and merge historical data from CSV files into the SQLite database.
This version ensures data integrity by performing a safe merge-and-overwrite.
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
    """Main function to process CSV files and safely upsert into the database."""
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
            # Define column names as the file has no header
            col_names = ['date', 'time_str', 'open', 'high', 'low', 'close', 'tick_volume', 'volume', 'spread']

            # Load new data from CSV
            df_csv = pd.read_csv(file_path, header=None, names=col_names)

            # Combine date and time_str into a single datetime column and clean the data
            df_csv['time'] = pd.to_datetime(df_csv['date'].astype(str) + ' ' + df_csv['time_str'], errors='coerce', utc=True)
            df_csv.dropna(subset=['time'], inplace=True)
            df_csv['symbol'] = symbol
            df_csv.drop_duplicates(subset=['symbol', 'time'], keep='first', inplace=True)

            # Load existing data from DB
            df_db = pd.read_sql(f"SELECT * FROM {table_name} WHERE symbol = '{symbol}'", engine, parse_dates=['time'])
            if not df_db.empty and df_db['time'].dt.tz is None:
                df_db['time'] = df_db['time'].dt.tz_localize('UTC')
            logger.info(f"Loaded {len(df_csv)} new/updated records from CSV and {len(df_db)} existing records from DB.")

            # --- Step 2: Combine and Deduplicate Safely ---
            # Combine, with new data first, then drop duplicates, keeping the new data.
            combined_df = pd.concat([df_csv, df_db], ignore_index=True)
            combined_df.drop_duplicates(subset=['symbol', 'time'], keep='first', inplace=True)

            # --- Step 3: Filter to match database schema ---
            db_columns = [col['name'] for col in inspector.get_columns(table_name)]
            final_df = combined_df[[col for col in db_columns if col in combined_df.columns]]

            # --- Step 4: Overwrite data in a single transaction ---
            logger.info(f"Writing {len(final_df)} total records for {symbol}...")
            with engine.begin() as conn:
                conn.execute(text(f"DELETE FROM {table_name} WHERE symbol = '{symbol}'"))
                final_df.to_sql(table_name, conn, if_exists='append', index=False, chunksize=10000)
            logger.success(f"Successfully wrote data for {symbol}.")

        except Exception as e:
            logger.error(f"Database operation failed for {symbol}: {e}")

    logger.info("--- Data ingestion process finished. ---")

if __name__ == "__main__":
    import_data()

