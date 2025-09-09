"""
This script runs the SHAP analysis on the final trained v2 model.
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from scripts.build_training_dataset_v2 import build_dataset_v2

# --- Logger Configuration ---
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>")

# --- Constants ---
MODEL_DIR = project_root / "models"
REPORTS_DIR = project_root / "reports"

def main():
    """Main function to run the SHAP analysis for the v2 model."""
    logger.info("--- Starting SHAP Analysis (v2) ---")

    # 1. Load the trained model and dataset
    logger.info("Loading v2 model and dataset...")
    model_path = MODEL_DIR / "final_model_EURUSD_v2.joblib"
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}. Train the v2 model first.")
        return
    model = joblib.load(model_path)

    X, y, _ = build_dataset_v2()
    if X.empty:
        logger.error("v2 Dataset creation failed. Exiting SHAP analysis.")
        return

    REPORTS_DIR.mkdir(exist_ok=True)
    logger.success("Successfully loaded v2 model and dataset.")

    # 2. Select a single target label to analyze
    target_label = y.columns[0]
    logger.info(f"Analyzing SHAP values for target label: {target_label}")
    target_idx = list(y.columns).index(target_label)
    single_estimator = model.estimators_[target_idx]

    # 3. Create SHAP Explainer and calculate values
    logger.info("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(single_estimator)

    sample_size = min(500, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    
    logger.info(f"Calculating SHAP values for {sample_size} samples...")
    shap_values = explainer.shap_values(X_sample)
    logger.success("SHAP values calculated.")

    # 4. Generate and Save Global Feature Importance Plot (Bar Plot)
    logger.info("Generating global feature importance bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title(f"SHAP Global Feature Importance for {target_label}")
    plot_path = REPORTS_DIR / f"shap_global_importance_bar_{target_label}_v2.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    logger.success(f"Saved global feature importance plot to {plot_path}")

    # 5. Generate and Save Summary Plot (Beeswarm)
    logger.info("Generating summary plot (beeswarm)...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"SHAP Summary Plot for {target_label}")
    plot_path = REPORTS_DIR / f"shap_summary_beeswarm_{target_label}_v2.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    logger.success(f"Saved summary plot to {plot_path}")

    logger.info("--- SHAP Analysis (v2) Finished ---")

if __name__ == "__main__":
    main()

