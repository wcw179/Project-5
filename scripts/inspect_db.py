"""Utility to inspect the structure of the SQLite database."""

import sys
from pathlib import Path

# Add project root to the Python path to allow importing from src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


from sqlalchemy import create_engine, inspect

from src.config import BASE_DIR

DB_PATH = BASE_DIR / "data" / "m5_trading.db"


def inspect_database(db_path: Path):
    """Connects to the database and prints its schema."""
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path}")
        return

    print(f"--- Inspecting Database: {db_path.name} ---")
    engine = create_engine(f"sqlite:///{db_path}")

    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        if not table_names:
            print("No tables found in the database.")
            return

        print(f"Found {len(table_names)} tables: {', '.join(table_names)}\n")

        for table_name in table_names:
            print(f"--- Table: {table_name} ---")
            columns = inspector.get_columns(table_name)
            for column in columns:
                print(
                    f"  - {column['name']} ({column['type']}, "
                    f"nullable={column['nullable']})"
                )
            print()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        engine.dispose()


if __name__ == "__main__":
    inspect_database(DB_PATH)
