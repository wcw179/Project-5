"""
This script runs the SHAP analysis on the final trained v3 ensemble model.
"""
import sys
from pathlib import Path
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- Logger Configuration ---
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>")

# --- Constants ---
MODEL_DIR = project_root / "models"
REPORTS_DIR = project_root / "reports"
DATA_DIR = project_root / "data" / "processed"

def main():
    """Main function to run the SHAP analysis for the v3 model."""
    logger.info("--- Starting SHAP Analysis (v3 Ensemble Model) ---")

    # 1. Load the trained model and dataset
    logger.info("Loading v3 model and dataset...")
    model_path = MODEL_DIR / "enhanced_final_ensemble_model.joblib"
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}. Train the model first.")
        return
    model = joblib.load(model_path)

    X_path = DATA_DIR / "X_full.parquet"
    if not X_path.exists():
        logger.error(f"Dataset file not found at {X_path}. Build the v3 dataset first.")
        return
    X = pd.read_parquet(X_path)

    REPORTS_DIR.mkdir(exist_ok=True)
    logger.success("Successfully loaded v3 model and dataset.")

    # 2. Create SHAP Explainer and calculate values
    logger.info("Creating SHAP TreeExplainer...")
    # Use the underlying booster object for better compatibility with SHAP
    explainer = shap.TreeExplainer(model.get_booster())

    # Use a smaller sample for faster computation
    sample_size = min(1000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)

    logger.info(f"Calculating SHAP values for {sample_size} samples...")
    shap_values = explainer.shap_values(X_sample)
    logger.success("SHAP values calculated.")
    # 3. Generate and Save Plots for Each Class
    class_names = {0: 'Sell (-1)', 1: 'Hold (0)', 2: 'Buy (1)'}
    for i, class_name in class_names.items():
        safe_class_name = class_name.replace(' ', '_').replace('(', '').replace(')', '')
        logger.info(f"Generating plots for class: {class_name}...")

        # Global Feature Importance (Bar Plot)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[:, :, i], X_sample, plot_type="bar", show=False)
        plt.title(f"SHAP Global Feature Importance for Class: {class_name}")
        plot_path = REPORTS_DIR / f"shap_global_bar_{safe_class_name}_v3.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logger.success(f"Saved global bar plot to {plot_path}")

        # Summary Plot (Beeswarm)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[:, :, i], X_sample, show=False)
        plt.title(f"SHAP Summary Plot for Class: {class_name}")
        plot_path = REPORTS_DIR / f"shap_summary_beeswarm_{safe_class_name}_v3.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        logger.success(f"Saved summary beeswarm plot to {plot_path}")

    logger.info("--- SHAP Analysis (v3) Finished ---")

if __name__ == "__main__":
    main()
