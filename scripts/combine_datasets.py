"""
This script combines the pre-generated X and y datasets into a single
Parquet file required for the training script.
"""
import pandas as pd
from pathlib import Path
from loguru import logger

# --- Configuration ---
project_root = Path(__file__).resolve().parent.parent
DATA_DIR = project_root / "data" / "processed"

X_FILE_PATH = DATA_DIR / "X_full_v3.parquet"
Y_FILE_PATH = DATA_DIR / "y_full_v3.parquet"
OUTPUT_PATH = DATA_DIR / "full_dataset_v3.parquet"

def main():
    """Loads, combines, and saves the datasets."""
    logger.info("--- Starting Dataset Combination ---")

    # Check if source files exist
    if not X_FILE_PATH.exists() or not Y_FILE_PATH.exists():
        logger.error(f"Source file not found. Please ensure both {X_FILE_PATH.name} and {Y_FILE_PATH.name} exist.")
        return

    # Load datasets
    logger.info(f"Loading features from {X_FILE_PATH.name}...")
    X_full = pd.read_parquet(X_FILE_PATH)
    logger.info(f"Loading labels from {Y_FILE_PATH.name}...")
    y_full = pd.read_parquet(Y_FILE_PATH)

    # Combine
    logger.info("Joining features and labels...")
    # Drop the overlapping 'symbol' column from y_full before joining
    combined_df = X_full.join(y_full.drop(columns=['symbol'], errors='ignore'))

    if combined_df.isnull().values.any():
        logger.warning("NaN values found after join. Check for index mismatches.")

    # Save the combined dataset
    logger.info(f"Saving combined dataset to {OUTPUT_PATH.name}...")
    combined_df.to_parquet(OUTPUT_PATH)

    logger.success(f"Successfully created combined dataset at: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

