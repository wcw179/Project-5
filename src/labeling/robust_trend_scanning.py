"""
Implementation of Trend-Scanning labels described in `Advances in Financial Machine Learning: Lecture 3/10
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>`_

This version includes a modification for numerical stability.
"""

from typing import Optional

import numpy as np
import pandas as pd

from mlfinpy.structural_breaks.sadf import get_betas


def robust_trend_scanning_labels(
    price_series: pd.Series,
    t_events: Optional[list] = None,
    look_forward_window: int = 20,
    min_sample_length: int = 5,
    step: int = 1,
    epsilon: float = 1e-8  # Small value to prevent division by zero
) -> pd.DataFrame:
    """
    A numerically robust implementation of the trend-scanning labeling technique.

    This version adds a small epsilon to the denominator of the t-value calculation
    to prevent division-by-zero errors when the standard error of the regression
    slope is close to zero (i.e., for perfectly linear trends).

    Parameters
    ----------
    price_series : pd.Series
        Close prices used to label the data set.
    t_events : Optional[list]
        Filtered events, array of pd.Timestamps.
    look_forward_window : int
        Maximum look forward window used to get the trend value.
    min_sample_length : int
        Minimum sample length used to fit regression.
    step : int
        Optimal t-value index is searched every 'step' indices.
    epsilon : float
        A small constant added to the t-value denominator for stability.

    Returns
    -------
    pd.DataFrame
        Consists of t1, t-value, ret, bin (label information).
    """
    # pylint: disable=invalid-name

    if t_events is None:
        t_events = price_series.index

    t1_array = []  # Array of label end times
    t_values_array = []  # Array of trend t-values

    for index in t_events:
        subset = price_series.loc[index:].iloc[:look_forward_window]  # Take t:t+L window
        if subset.shape[0] >= look_forward_window:
            max_abs_t_value = -np.inf
            max_t_value_index = None
            max_t_value = None

            for forward_window in np.arange(min_sample_length, subset.shape[0], step):
                y_subset = subset.iloc[:forward_window].values.reshape(-1, 1)
                X_subset = np.ones((y_subset.shape[0], 2))
                X_subset[:, 1] = np.arange(y_subset.shape[0])

                b_mean_, b_std_ = get_betas(X_subset, y_subset)
                
                # MODIFICATION: Add epsilon for numerical stability
                std_error = np.sqrt(b_std_[1, 1])
                t_beta_1 = (b_mean_[1] / (std_error + epsilon))[0]

                if abs(t_beta_1) > max_abs_t_value:
                    max_abs_t_value = abs(t_beta_1)
                    max_t_value = t_beta_1
                    max_t_value_index = forward_window

            label_endtime_index = subset.index[max_t_value_index - 1]
            t1_array.append(label_endtime_index)
            t_values_array.append(max_t_value)

        else:
            t1_array.append(None)
            t_values_array.append(None)

    labels = pd.DataFrame({"t1": t1_array, "t_value": t_values_array}, index=t_events)
    labels.loc[:, "ret"] = price_series.reindex(labels.t1).values / price_series.reindex(labels.index).values - 1
    labels["bin"] = labels.t_value.apply(np.sign)

    return labels
