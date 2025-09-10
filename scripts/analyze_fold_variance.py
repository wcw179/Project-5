"""
Script to analyze the data variance between the best and worst performing folds
from a cross-validation run.
"""
import sys
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection._split import _BaseKFold

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Define the same PurgedKFold class used in the optimization script
class PurgedKFold(_BaseKFold):
    def __init__(self, n_splits=10, t1=None, pct_embargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError('t1 must be a pd.Series.')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        if X.shape[0] != self.t1.shape[0]:
            raise ValueError('X and t1 must have the same length.')

        indices = list(X.index)
        embargo = int(X.shape[0] * self.pct_embargo)
        test_ranges = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]

        for i, (start_ix, end_ix) in enumerate(test_ranges):
            test_indices = indices[start_ix:end_ix]

            t0 = self.t1.index[start_ix]
            t1_ = self.t1.iloc[end_ix - 1]

            # Purge from training set
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if t1_ is not pd.NaT:
                train_indices = np.setdiff1d(train_indices, self.t1.index.searchsorted(self.t1[self.t1 >= t1_ - pd.Timedelta(microseconds=1)].index))

            # Embargo
            if embargo > 0:
                embargo_start_ix = end_ix
                embargo_end_ix = min(embargo_start_ix + embargo, X.shape[0])
                train_indices = np.setdiff1d(train_indices, np.arange(embargo_start_ix, embargo_end_ix))

            yield train_indices, test_indices

def analyze_fold_data(fold_name: str, train_idx, val_idx, X, y):
    """Analyzes and prints statistics for a given fold."""
    print(f"\n--- Analysis for {fold_name} ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    print(f"\n[+] {fold_name} - Label Distribution:")
    print("Train labels:\n", y_train.value_counts(normalize=True))
    print("\nValidation labels:\n", y_val.value_counts(normalize=True))

    print(f"\n[+] {fold_name} - Feature Summary Statistics (Train):")
    print(X_train.describe())

    print(f"\n[+] {fold_name} - Feature Summary Statistics (Validation):")
    print(X_val.describe())

def main():
    """Main function to load data, recreate folds, and run analysis."""
    print("--- Starting Fold Variance Analysis ---")

    # Configuration
    CONFIG = {
        'n_splits': 5,
        'pct_embargo': 0.01,
    }
    BEST_FOLD_INDEX = 3 # Corresponds to Fold 4
    WORST_FOLD_INDEX = 1 # Corresponds to Fold 2

    # Load data
    data_dir = project_root / "data" / "processed"
    X = pd.read_parquet(data_dir / "X_full.parquet")
    y = pd.read_parquet(data_dir / "y_full.parquet").squeeze()
    sample_info = pd.read_parquet(data_dir / "sample_info_full.parquet")

    # Align datasets
    common_idx = X.index.intersection(y.index).intersection(sample_info.index)
    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]
    sample_info_aligned = sample_info.loc[common_idx]
    t1 = sample_info_aligned['t1']

    # Reset index for CV compatibility
    X_reindexed = X_aligned.reset_index(drop=True)
    y_reindexed = y_aligned.reset_index(drop=True)

    # Recreate CV splits
    cv = PurgedKFold(n_splits=CONFIG['n_splits'], t1=t1, pct_embargo=CONFIG['pct_embargo'])
    cv_splits = list(cv.split(X_reindexed, y_reindexed))

    # Analyze the worst fold
    worst_fold_train_idx, worst_fold_val_idx = cv_splits[WORST_FOLD_INDEX]
    analyze_fold_data("Worst Fold (Fold 2)", worst_fold_train_idx, worst_fold_val_idx, X_reindexed, y_reindexed)

    # Analyze the best fold
    best_fold_train_idx, best_fold_val_idx = cv_splits[BEST_FOLD_INDEX]
    analyze_fold_data("Best Fold (Fold 4)", best_fold_train_idx, best_fold_val_idx, X_reindexed, y_reindexed)

if __name__ == "__main__":
    main()
