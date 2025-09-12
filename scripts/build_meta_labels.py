"""
This script uses the results from the Triple-Barrier method to create meta-labels
for the predictions of a primary model.
"""
import pandas as pd
from pathlib import Path
from loguru import logger

# --- Configuration ---
project_root = Path(__file__).resolve().parent.parent
DATA_DIR = project_root / "data" / "processed"

TRIPLE_BARRIER_PATH = DATA_DIR / "y_full_v3.parquet"
PREDICTIONS_PATH = DATA_DIR / "primary_model_predictions.parquet"
OUTPUT_PATH = DATA_DIR / "meta_labeled_dataset.parquet"

def apply_meta_labeling(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the meta-labeling logic to the DataFrame."""
    logger.info("Applying meta-labeling logic...")

    # Conditions for meta_label
    # 1. If primary model predicts 0, meta_label is 0
    # 2. If primary model predicts 1 and the bet was correct (label > 0), meta_label is 1
    # 3. If primary model predicts 1 and the bet was incorrect (label <= 0), meta_label is 0
    df['meta_label'] = 0  # Default to 0

    # Use the correct column name 'label'
    correct_bet_mask = (df['prediction'] == 1) & (df['label'] > 0)
    df.loc[correct_bet_mask, 'meta_label'] = 1

    return df

def main():
    """Main function to orchestrate the meta-labeling process."""
    logger.info("--- Starting Meta-Labeling Process ---")

    # Check for source files
    if not TRIPLE_BARRIER_PATH.exists() or not PREDICTIONS_PATH.exists():
        logger.error(f"Source files not found. Ensure '{TRIPLE_BARRIER_PATH.name}' and '{PREDICTIONS_PATH.name}' exist.")
        return

    # Load data
    logger.info(f"Loading triple-barrier results from '{TRIPLE_BARRIER_PATH.name}'...")
    barrier_df = pd.read_parquet(TRIPLE_BARRIER_PATH)
    logger.info(f"Loading primary model predictions from '{PREDICTIONS_PATH.name}'...")
    preds_df = pd.read_parquet(PREDICTIONS_PATH)

    # Join the dataframes
    logger.info("Joining datasets...")
    combined_df = barrier_df.join(preds_df, how='inner')

    if len(combined_df) == 0:
        logger.error("The join resulted in an empty DataFrame. Please check for index alignment.")
        return

    # Apply the meta-labeling logic
    meta_labeled_df = apply_meta_labeling(combined_df)

    # Save the final dataset
    logger.info(f"Saving meta-labeled dataset to '{OUTPUT_PATH.name}'...")
    meta_labeled_df.to_parquet(OUTPUT_PATH)

    # Print statistics
    logger.info("Meta-label distribution:")
    print(meta_labeled_df['meta_label'].value_counts())

    logger.success(f"Successfully created meta-labeled dataset at: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

