"""
Script to analyze feature importance from the saved models of a cross-validation run.
"""
import sys
import pandas as pd
import json
import joblib
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def main():
    """Main function to load models and analyze feature importance."""
    print("--- Starting Feature Importance Analysis ---")
    
    MODEL_DIR = project_root / "models"
    
    # Load the validation results to get model paths and feature names
    try:
        X = pd.read_parquet(project_root / "data" / "processed" / "X_full.parquet")
        feature_names = X.columns.tolist()
        
        results_path = MODEL_DIR / "final_validation_results.json"
        with open(results_path, 'r') as f:
            results = json.load(f)
        model_paths = results.get('saved_models', [])
    except FileNotFoundError as e:
        print(f"Error: Required file not found. {e}")
        return
    except KeyError:
        # Fallback for older results format
        try:
            model_paths = [str(MODEL_DIR / f"final_model_fold_{i}.joblib") for i in range(4)]
        except Exception as e:
            print(f"Could not find model paths in results file. Error: {e}")
            return

    if not model_paths:
        print("Error: No model paths found in the results file.")
        return

    all_importances = []

    for i, model_path in enumerate(model_paths):
        try:
            model = joblib.load(model_path)
            importances = pd.Series(model.feature_importances_, index=feature_names)
            all_importances.append(importances)
            print(f"Successfully loaded model from fold {i}")
        except FileNotFoundError:
            print(f"Warning: Model file not found for fold {i} at {model_path}. Skipping.")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")

    if not all_importances:
        print("Could not load any models. Aborting analysis.")
        return

    # Calculate average importance
    avg_importance = pd.concat(all_importances, axis=1).mean(axis=1)
    avg_importance.sort_values(ascending=False, inplace=True)

    print("\n--- Top 15 Most Influential Features (Averaged Across Folds) ---")
    print(avg_importance.head(15))

if __name__ == "__main__":
    main()
