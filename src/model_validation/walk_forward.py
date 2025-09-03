"""Implementation of Walk-Forward Analysis Cross-Validation."""

from typing import Generator

import numpy as np
import pandas as pd

from src.logger import logger


class WalkForwardValidator:
    """
    A walk-forward cross-validator for financial time-series.

    This validator simulates a realistic trading scenario where a model is
    periodically retrained on new data and tested on the subsequent period.
    """

    def __init__(
        self,
        train_window_hours: int = 2000,
        test_window_hours: int = 500,
        step_size_hours: int = 100,
        embargo_hours: int = 24,
        m5_bars_per_hour: int = 12,
    ):
        self.train_window = train_window_hours * m5_bars_per_hour
        self.test_window = test_window_hours * m5_bars_per_hour
        self.step_size = step_size_hours * m5_bars_per_hour
        self.embargo = embargo_hours * m5_bars_per_hour
        logger.info(
            f"Initialized WalkForwardValidator: "
            f"Train={self.train_window}, Test={self.test_window}, "
            f"Embargo={self.embargo}, Step={self.step_size} (in bars)."
        )

    def split(
        self, X: pd.DataFrame
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates walk-forward train/test splits with an embargo period.

        Args:
            X: The feature DataFrame.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        start_pos = 0
        while (
            start_pos + self.train_window + self.embargo + self.test_window <= n_samples
        ):
            train_end_pos = start_pos + self.train_window
            test_start_pos = train_end_pos + self.embargo
            test_end_pos = test_start_pos + self.test_window

            train_idx = indices[start_pos:train_end_pos]
            test_idx = indices[test_start_pos:test_end_pos]

            logger.debug(
                f"Yielding split: Train {train_idx[0]}-{train_idx[-1]}, "
                f"Test {test_idx[0]}-{test_idx[-1]}"
            )
            yield train_idx, test_idx

            start_pos += self.step_size
