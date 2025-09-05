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

        if X.shape[0] == 0:
            logger.warning("Input X is empty, PurgedKFold will not yield any splits.")
            return

        indices = np.arange(X.shape[0])
        super_split = super().split(indices)

        for i, (train_idx, test_idx) in enumerate(super_split):
            logger.debug(f"Processing fold {i+1}/{self.n_splits}...")

            train_times = sample_info.iloc[train_idx]
            test_times = sample_info.iloc[test_idx]

            # Get test set time boundaries
            test_start_time = test_times.index.min()
            test_end_time = test_times["t1"].max()

            # Purge training samples that overlap with the test set
            train_sample_starts = train_times.index
            train_sample_ends = train_times["t1"]

            # Condition 1: Train sample starts during the test period
            purge_cond1 = (train_sample_starts >= test_start_time) & (train_sample_starts <= test_end_time)
            # Condition 2: Train sample ends during the test period
            purge_cond2 = (train_sample_ends >= test_start_time) & (train_sample_ends <= test_end_time)
            # Condition 3: Train sample envelops the test period
            purge_cond3 = (train_sample_starts <= test_start_time) & (train_sample_ends >= test_end_time)

            purged_mask = purge_cond1 | purge_cond2 | purge_cond3

            # Embargo: Purge training samples that start right after the test set
            embargo_start_time = test_end_time + self.embargo_td
            embargo_mask = (train_sample_starts > test_end_time) & (train_sample_starts < embargo_start_time)

            # Combine masks and get the final training indices
            contaminated_mask = purged_mask | embargo_mask
            final_train_mask = ~contaminated_mask

            final_train_idx = train_idx[final_train_mask.to_numpy()]

            logger.debug(
                f"Fold {i+1}: Original train size: {len(train_idx)}, Purged size: {len(final_train_idx)}"
            )
            yield final_train_idx, test_idx
