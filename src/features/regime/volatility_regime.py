"""Module for determining the market regime based on volatility."""

import numpy as np
import pandas as pd


def add_volatility_regime(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """
    Adds a multi-level volatility regime feature based on rolling quantiles.

    Regimes:
    - 0: Low Volatility (<25th percentile)
    - 1: Medium Volatility (25th-75th percentile)
    - 2: High Volatility (75th-95th percentile)
    - 3: Crisis Volatility (>95th percentile)

    Args:
        df (pd.DataFrame): The input DataFrame with a 'close' column.
        window (int): The rolling window for volatility and quantiles.

    Returns:
        pd.DataFrame: DataFrame with the 'volatility_regime' column added.
    """
    df_copy = df.copy()

    # Calculate rolling volatility of log returns
    log_returns = pd.Series(np.log(df_copy["close"]).diff())
    rolling_vol = log_returns.rolling(window=window, min_periods=window).std()

    # Define regime thresholds using rolling quantiles
    low_thresh = rolling_vol.rolling(window=window).quantile(0.25)
    high_thresh = rolling_vol.rolling(window=window).quantile(0.75)
    crisis_thresh = rolling_vol.rolling(window=window).quantile(0.95)

    # Classify the regime (default to Medium)
    df_copy["volatility_regime"] = 1
    df_copy.loc[rolling_vol < low_thresh, "volatility_regime"] = 0  # Low
    df_copy.loc[rolling_vol >= high_thresh, "volatility_regime"] = 2  # High
    df_copy.loc[rolling_vol >= crisis_thresh, "volatility_regime"] = 3  # Crisis

    return df_copy
