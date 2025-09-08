"""
This script runs the SHAP (SHapley Additive exPlanations) analysis on the
final trained model to explain its predictions.
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

from scripts.build_training_dataset import build_dataset

# --- Logger Configuration ---
log_file_path = project_root / "logs" / "shap_analysis.log"
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>")
logger.add(log_file_path, rotation="10 MB", retention="10 days", enqueue=True, serialize=False)

# --- Constants ---
MODEL_DIR = project_root / "models"
REPORTS_DIR = project_root / "reports"

def main():
    """Main function to run the SHAP analysis."""
    logger.info("--- Starting SHAP Analysis ---")

    # 1. Load the trained model and dataset
    logger.info("Loading model and dataset...")
    model_path = MODEL_DIR / "final_model.joblib"
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}. Train the model first.")
        return
    model = joblib.load(model_path)

    X, y, _ = build_dataset()
    if X.empty:
        logger.error("Dataset creation failed. Exiting SHAP analysis.")
        return

    REPORTS_DIR.mkdir(exist_ok=True)
    logger.success("Successfully loaded model and dataset.")

    # 2. Create SHAP Explainer using the modern API
    logger.info("Creating SHAP TreeExplainer and calculating values...")
    explainer = shap.TreeExplainer(model)

    # Use a smaller sample for faster analysis
    sample_size = min(500, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)

    # The modern API returns a single Explanation object, which is easier to work with
    shap_explanation = explainer(X_sample)
    logger.success("SHAP values calculated.")

    # 3. Generate and Save Global Feature Importance Plot
    logger.info("Generating global feature importance plot...")
    plt.figure()
    class_names = ["Short", "Neutral", "Long"] # Corresponds to {-1, 0, 1} mapped to {0, 1, 2}
    shap.summary_plot(shap_explanation.values, X_sample, plot_type="bar", show=False, class_names=class_names)
    plot_path = REPORTS_DIR / "shap_global_feature_importance.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    logger.success(f"Saved global feature importance plot to {plot_path}")

    logger.info("--- SHAP Analysis Finished ---")



if __name__ == "__main__":
    main()
