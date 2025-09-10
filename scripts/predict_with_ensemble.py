"""
Script to make predictions using an ensemble of models trained across CV folds.
"""
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

class FoldEnsemble:
    """An ensemble of models trained on different folds of the data."""
    def __init__(self, model_paths):
        print(f"Initializing ensemble with {len(model_paths)} models...")
        self.models = self._load_models(model_paths)

    def _load_models(self, model_paths):
        models = []
        for path in model_paths:
            try:
                model = joblib.load(path)
                models.append(model)
                print(f"Successfully loaded model from: {path}")
            except FileNotFoundError:
                print(f"Warning: Model file not found at {path}. Skipping.")
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
        return models

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Averages the probability predictions from all models in the ensemble.
        Returns the class with the highest average probability.
        """
        if not self.models:
            raise RuntimeError("No models were loaded for the ensemble.")

        # Get probability predictions from each model
        all_probas = [model.predict_proba(X) for model in self.models]

        # Average the probabilities across all models
        avg_probas = np.mean(all_probas, axis=0)

        # Get the class with the highest average probability
        final_predictions_mapped = np.argmax(avg_probas, axis=1)
        
        # Map back to original labels (-1, 0, 1)
        label_map = {0: -1, 1: 0, 2: 1}
        final_predictions = np.vectorize(label_map.get)(final_predictions_mapped)
        
        return final_predictions, avg_probas

def main():
    """Main function to demonstrate ensemble prediction."""
    print("--- Starting Ensemble Prediction Demonstration ---")
    MODEL_DIR = project_root / "models"

    # Load model paths from the results file
    try:
        results_path = MODEL_DIR / "final_validation_results.json"
        with open(results_path, 'r') as f:
            results = json.load(f)
        model_paths = results.get('saved_models', [])
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_path}")
        return

    if not model_paths:
        print("Error: No saved model paths found in results file.")
        return

    # Initialize the ensemble
    ensemble = FoldEnsemble(model_paths)

    # Load some sample data for prediction (e.g., the last 100 rows)
    X = pd.read_parquet(project_root / "data" / "processed" / "X_full.parquet")
    sample_X = X.tail(100)

    print(f"\nMaking predictions on {len(sample_X)} sample data points...")
    predictions, probabilities = ensemble.predict(sample_X)

    print("\n--- Prediction Results ---")
    results_df = pd.DataFrame(probabilities, columns=['prob_sell', 'prob_hold', 'prob_buy'], index=sample_X.index)
    results_df['prediction'] = predictions
    print(results_df)

if __name__ == "__main__":
    main()
