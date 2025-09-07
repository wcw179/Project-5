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
    # For bar plots, we pass the shap_values component of the explanation object
    shap.summary_plot(shap_explanation.values, X_sample, plot_type="bar", show=False, class_names=["Short", "Neutral", "Long"])
    plot_path = REPORTS_DIR / "shap_global_feature_importance.png"
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    logger.success(f"Saved global feature importance plot to {plot_path}")

    # 4. Generate and Save Local Prediction Explanations (Waterfall Plots)
    logger.info("Generating local prediction explanation plots...")
    class_names = {0: "Short", 1: "Neutral", 2: "Long"}

    # Explain the first 3 predictions in the sample
    for i in range(3):
        for class_idx, class_name in class_names.items():
            plt.figure()
            # The new API makes slicing for plots much simpler and more robust
            shap.waterfall_plot(shap_explanation[i, :, class_idx], show=False)
            plt.title(f"SHAP Waterfall for Prediction {i} - Class: {class_name}")
            local_plot_path = REPORTS_DIR / f"shap_waterfall_pred_{i}_class_{class_name}.png"
            plt.savefig(local_plot_path, bbox_inches='tight')
            plt.close()
            logger.success(f"Saved waterfall plot to {local_plot_path}")


    # 5. Generate and Save Feature Interaction Plots (Dependence Plots)
    logger.info("Generating feature interaction dependence plots...")

    # First, find the top 5 most important features globally
    # We average the mean absolute SHAP values across all classes
    global_shap_values = np.abs(shap_explanation.values).mean(axis=0)
    feature_importances = pd.DataFrame(
        list(zip(X_sample.columns, global_shap_values.sum(axis=1))),
        columns=['feature_name', 'importance']
    ).sort_values(by='importance', ascending=False)

    top_features = feature_importances['feature_name'].head(5).tolist()

    for feature in top_features:
        for class_idx, class_name in class_names.items():
            plt.figure()
            # Use the legacy API for dependence_plot for more stability
            shap.dependence_plot(
                feature,
                shap_explanation.values[:, :, class_idx],
                X_sample,
                show=False
            )
            plt.title(f"SHAP Dependence Plot for '{feature}' - Class: {class_name}")
            interaction_plot_path = REPORTS_DIR / f"shap_dependence_{feature}_class_{class_name}.png"
            plt.savefig(interaction_plot_path, bbox_inches='tight')
            logger.success(f"Saved dependence plot to {interaction_plot_path}")
            plt.close()

    # 6. Perform and Save Global Importance Analysis from Data
    logger.info("Calculating and saving global feature importance...")

    # Calculate mean absolute SHAP value for each feature across all classes
    shap_values_all_classes = shap_explanation.values
    mean_abs_shap = np.abs(shap_values_all_classes).mean(axis=(0, 2))

    feature_importance_df = pd.DataFrame({
        'feature': X_sample.columns,
        'mean_abs_shap': mean_abs_shap
    }).sort_values(by='mean_abs_shap', ascending=False)

    importance_path = REPORTS_DIR / "global_feature_importance.csv"
    feature_importance_df.to_csv(importance_path, index=False)
    logger.success(f"Saved global feature importance data to {importance_path}")
    logger.info("--- Top 10 Most Important Features ---")
    logger.info(f"\n{feature_importance_df.head(10).to_string()}")

    # 7. Export SHAP Values to CSV
    logger.info("Exporting SHAP values for the sample to CSV...")

    # Create a DataFrame for each class's SHAP values
    shap_dfs = []
    for i, name in class_names.items():
        df = pd.DataFrame(shap_explanation.values[:, :, i], columns=[f"{col}_shap_{name}" for col in X_sample.columns], index=X_sample.index)
        shap_dfs.append(df)

    # Concatenate all SHAP value DataFrames
    shap_df_combined = pd.concat(shap_dfs, axis=1)

    # Join with the original feature values
    export_df = X_sample.join(shap_df_combined)

    export_path = REPORTS_DIR / "shap_values_sample.csv"
    export_df.to_csv(export_path)
    logger.success(f"Saved SHAP values sample to {export_path}")



if __name__ == "__main__":
    main()
