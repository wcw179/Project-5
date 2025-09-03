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
            train_times = sample_info.iloc[train_idx]
            test_times = sample_info.iloc[test_idx]

            # Determine the full time span of the test set
            test_start = test_times.index.min()
            test_end = test_times["t1"].max()

            # Identify training samples contaminated by the test set
            # 1. Purge: Training samples whose labels overlap with the test period
            train_starts = train_times.index
            train_ends = train_times["t1"]

            # Overlap conditions
            overlap_c1 = (train_starts >= test_start) & (train_starts <= test_end)
            overlap_c2 = (train_ends >= test_start) & (train_ends <= test_end)
            overlap_c3 = (train_starts <= test_start) & (train_ends >= test_end)
            purged_mask = overlap_c1 | overlap_c2 | overlap_c3

            # 2. Embargo: Training samples immediately following the test period
            embargo_end = test_end + self.embargo_td
            embargo_mask = (train_starts > test_end) & (train_starts < embargo_end)

            # Combine masks to get all contaminated indices
            contaminated_mask = purged_mask | embargo_mask
            final_train_indices = train_times[~contaminated_mask].index

            # Convert final time indices back to integer positions
            final_train_idx = X.index.get_indexer(final_train_indices)

            logger.debug(
                f"Fold {i+1}: Original train size: {len(train_idx)}, Purged size: {len(final_train_idx)}"
            )
            yield final_train_idx, test_idx
