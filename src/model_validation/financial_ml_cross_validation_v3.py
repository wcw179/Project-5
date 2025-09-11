"""
Enhanced version addressing the identified issues in financial ML cross-validation pipeline V1
"""

from typing import Callable, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss, accuracy_score, f1_score
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import joblib
import logging
from pathlib import Path


class PurgedStratifiedKFold(KFold):
    """
    A robust cross-validator that combines purging, embargoing, and stratification.
    It first stratifies the data, then applies purging and embargoing based on time.
    A final check ensures that the training set for each fold contains all necessary classes.
    """
    def __init__(self, n_splits: int = 3, samples_info_sets: pd.Series = None,
                 pct_embargo: float = 0.0, min_samples_per_class: int = 1):
        if not isinstance(samples_info_sets, pd.Series):
            raise ValueError("samples_info_sets must be a pandas Series.")
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo
        self.min_samples_per_class = min_samples_per_class

    def split(self, X: pd.DataFrame, y: pd.Series, groups: np.ndarray = None):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        embargo = int(n_samples * self.pct_embargo)

        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=False)

        for train_indices_initial, test_indices in skf.split(X, y):
            t1 = self.samples_info_sets.iloc[test_indices]
            test_start, test_end = t1.index.min(), t1.max()

            train_times = self.samples_info_sets.iloc[train_indices_initial]

            purge_mask_1 = (train_times >= test_start) & (train_times <= test_end)
            purge_mask_2 = (train_times.index >= test_start) & (train_times.index <= test_end)
            purge_mask_3 = (train_times.index <= test_start) & (train_times >= test_end)

            final_purge_mask = ~(purge_mask_1 | purge_mask_2 | purge_mask_3)
            train_indices_purged = train_indices_initial[final_purge_mask]

            if embargo > 0:
                last_test_index_pos = test_indices.max()
                embargo_start_pos = last_test_index_pos + 1
                embargo_end_pos = embargo_start_pos + embargo
                embargo_indices = indices[embargo_start_pos:embargo_end_pos]
                train_indices = np.setdiff1d(train_indices_purged, embargo_indices)
            else:
                train_indices = train_indices_purged

            # Final check to ensure all validation classes are in the training set
            train_classes = np.unique(y.iloc[train_indices])
            val_classes = np.unique(y.iloc[test_indices])
            if np.isin(val_classes, train_classes).all():
                yield train_indices, test_indices
            else:
                logging.warning(f"Skipping a fold due to missing classes after purging. "
                              f"Train classes: {train_classes}, Val classes: {val_classes}")