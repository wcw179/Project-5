"""Implementation of Purged K-Fold Cross-Validation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.logger import logger


class PurgedKFold(KFold):
    """
    A K-Fold cross-validator that purges and embargoes overlapping samples.

    This is crucial for financial time-series data where samples may have
    overlapping start and end times, leading to data leakage.
    """

    def __init__(
        self, n_splits: int = 10, embargo_td: pd.Timedelta = pd.Timedelta(hours=24)
    ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.embargo_td = embargo_td

    def split(
        self, X: pd.DataFrame, y: pd.Series = None, sample_info: pd.DataFrame = None
    ):
        """
        Generates purged and embargoed train/test splits.

        Args:
            X: The feature DataFrame.
            y: The target series (optional).
            sample_info: DataFrame with 'start_time' and 'end_time' for each sample.
        """
        if sample_info is None:
            raise ValueError("sample_info must be provided for PurgedKFold.")

        logger.info(
            f"Starting Purged K-Fold split with {self.n_splits} splits and embargo {self.embargo_td}."
        )

        indices = np.arange(X.shape[0])
        super_split = super().split(indices)

        for i, (train_idx, test_idx) in enumerate(super_split):
            logger.debug(f"Processing fold {i+1}/{self.n_splits}...")

            # Get the start and end times for the test set
            test_start_time = sample_info.iloc[test_idx].start_time.min()
            test_end_time = sample_info.iloc[test_idx].end_time.max()

            # --- Purging Logic ---
            # Purge training samples that overlap with the test set
            train_sample_info = sample_info.iloc[train_idx]

            # Samples starting before the test set ends
            purge_mask_1 = train_sample_info.start_time <= test_end_time
            # Samples ending after the test set starts
            purge_mask_2 = train_sample_info.end_time >= test_start_time

            # Combine masks to find overlapping samples
            overlapping_mask = purge_mask_1 & purge_mask_2
            purged_train_idx = train_idx[~overlapping_mask]

            # --- Embargo Logic ---
            # Apply an embargo period after the test set
            embargo_start_time = test_end_time + self.embargo_td
            embargo_mask = (
                sample_info.iloc[purged_train_idx].start_time < embargo_start_time
            )
            final_train_idx = purged_train_idx[~embargo_mask]

            logger.debug(
                f"Fold {i+1}: Original train size: {len(train_idx)}, Purged size: {len(final_train_idx)}"
            )
            yield final_train_idx, test_idx
