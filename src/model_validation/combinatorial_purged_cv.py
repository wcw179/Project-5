"""
Combinatorial Purged Cross-Validation (CPCV)

Implements a CPCV splitter suitable for financial time-series. It generates
multiple combinations of non-overlapping test sets, and purges/embargoes the
training indices around those test sets to avoid leakage.

References: López de Prado (2018), AFML.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Generator, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass
class CPCVConfig:
    n_splits: int = 10
    n_test_splits: int = 2
    embargo_td: pd.Timedelta = pd.Timedelta(hours=24)


class CombinatorialPurgedCV:
    def __init__(self, n_splits: int = 10, n_test_splits: int = 2, embargo_td: pd.Timedelta = pd.Timedelta(hours=24)) -> None:
        if n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be < n_splits")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_td = embargo_td

    def split(
        self,
        X: pd.DataFrame,
        sample_info: pd.DataFrame,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Args:
            X: features with DatetimeIndex
            sample_info: DataFrame with at least a column 't1' (end time) and index as start time
        Yields:
            (train_idx, test_idx) as integer positions into X
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex")
        if "t1" not in sample_info.columns:
            raise ValueError("sample_info must contain column 't1'")

        n = len(X)
        indices = np.arange(n)
        # split equally by index order
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            folds.append((start, stop))
            current = stop

        # precompute time spans for each fold
        fold_spans = []
        for (start, stop) in folds:
            idx = X.index[start:stop]
            if len(idx) == 0:
                fold_spans.append((None, None))
                continue
            f_start = idx.min()
            f_end = idx.max()
            fold_spans.append((f_start, f_end))

        # build combinations of test folds
        for test_combo in combinations(range(self.n_splits), self.n_test_splits):
            # collect test indices
            test_mask = np.zeros(n, dtype=bool)
            test_start_time = None
            test_end_time = None
            for t in test_combo:
                start, stop = folds[t]
                test_mask[start:stop] = True
                span = fold_spans[t]
                if test_start_time is None or (span[0] is not None and span[0] < test_start_time):
                    test_start_time = span[0]
                if test_end_time is None or (span[1] is not None and span[1] > test_end_time):
                    test_end_time = span[1]

            test_idx = indices[test_mask]

            # purge + embargo around test period
            if test_start_time is None or test_end_time is None:
                continue
            embargo_end = test_end_time + self.embargo_td

            # map sample_info to boolean mask of contaminated training samples
            s_idx = sample_info.index
            s_t1 = sample_info["t1"]
            purged = ((s_idx >= test_start_time) & (s_idx <= test_end_time)) \
                | ((s_t1 >= test_start_time) & (s_t1 <= test_end_time)) \
                | ((s_idx <= test_start_time) & (s_t1 >= test_end_time))
            embargo = (s_idx > test_end_time) & (s_idx < embargo_end)
            contaminated = purged | embargo

            # Convert remaining start times to integer positions in X
            final_train_indices = s_idx[~contaminated]
            train_idx = X.index.get_indexer(final_train_indices)
            train_idx = train_idx[train_idx >= 0]

            yield train_idx, test_idx

