"""Script to verify that a unique model was trained for each target label."""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.logger import logger

# --- Configuration ---
MODEL_PATH = "C:/Users/wcw17/Documents/GitHub/project-5/Project-5/data/models/EURUSDm_model_20250901_065505.joblib"


def main():
    """Main verification function."""
    logger.info(f"--- Verifying Label Usage for {MODEL_PATH} ---")

    # Load model
    model_artifact = joblib.load(MODEL_PATH)

    # Check if the model is a MultiOutputClassifier and has estimators
    if not hasattr(model_artifact, "estimators_"):
        logger.error(
            "The loaded object is not a scikit-learn MultiOutputClassifier or has no estimators."
        )
        return

    estimators = model_artifact.estimators_
    # Assuming target names are stored or can be inferred. For this project, we know the structure.
    r_multiples = ["5R", "10R", "15R", "20R"]
    horizons = ["1", "2", "3"]
    target_names = [f"hit_{r}_{h}" for r in r_multiples for h in horizons]

    if len(estimators) != len(target_names):
        logger.error(
            f"Mismatch between number of estimators ({len(estimators)}) and expected targets ({len(target_names)})."
        )
        return

    # Create a 4x3 subplot grid
    fig, axes = plt.subplots(4, 3, figsize=(20, 25), sharex=True)
    fig.suptitle("Feature Importance for Each Target Label Model", fontsize=20)
    axes = axes.flatten()

    for i, (est, target_name) in enumerate(zip(estimators, target_names)):
        if hasattr(est, "feature_importances_"):
            importances = pd.Series(
                est.feature_importances_, index=model_artifact.feature_names_in_
            )
            top_10 = importances.nlargest(10)

            sns.barplot(x=top_10.values, y=top_10.index, ax=axes[i], palette="mako")
            axes[i].set_title(f"Target: {target_name}")
            axes[i].tick_params(axis="y", labelsize=10)
        else:
            logger.warning(
                f"Estimator for {target_name} has no 'feature_importances_'."
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Create reports directory if it doesn't exist
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    save_path = reports_dir / "label_specific_feature_importance.png"

    plt.savefig(save_path)
    logger.success(f"Verification plot saved to {save_path}")
    print(f"Verification plot saved to {save_path}")


if __name__ == "__main__":
    main()
