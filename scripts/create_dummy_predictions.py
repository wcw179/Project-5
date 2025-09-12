"""
Creates a dummy primary_model_predictions.parquet file for meta-labeling development.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

# --- Configuration ---
project_root = Path(__file__).resolve().parent.parent
DATA_DIR = project_root / "data" / "processed"

Y_FILE_PATH = DATA_DIR / "y_full_v3.parquet"
OUTPUT_PATH = DATA_DIR / "primary_model_predictions.parquet"

def main():
    """Creates and saves the dummy prediction file."""
    logger.info("--- Creating Dummy Primary Model Predictions ---")

    if not Y_FILE_PATH.exists():
        logger.error(f"Source file {Y_FILE_PATH.name} not found. Cannot create dummy predictions.")
        return

    logger.info(f"Reading index from {Y_FILE_PATH.name}...")
    y_df = pd.read_parquet(Y_FILE_PATH)

    # Create dummy predictions (random 0s and 1s)
    logger.info("Generating random predictions...")
    predictions = np.random.randint(0, 2, size=len(y_df))
    
    # Create the new DataFrame
    pred_df = pd.DataFrame({'prediction': predictions}, index=y_df.index)

    # Save the dummy predictions
    logger.info(f"Saving dummy predictions to {OUTPUT_PATH.name}...")
    pred_df.to_parquet(OUTPUT_PATH)

    logger.success(f"Successfully created dummy predictions at: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

