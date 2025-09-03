"""Module for calculating fractal dimension features."""

import numpy as np
import pandas as pd


def fractal_dimension(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculates the Fractal Dimension Index over a rolling window.

    Args:
        df (pd.DataFrame): DataFrame with 'high' and 'low' columns.
        window (int): The rolling window period.

    Returns:
        pd.Series: The Fractal Dimension Index.
    """
    high_roll = df["high"].rolling(window=window).max()
    low_roll = df["low"].rolling(window=window).min()

    # Calculate the slope of the line connecting the start and end of the window
    price_range = high_roll - low_roll

    # Avoid division by zero
    price_range[price_range == 0] = 0.0001

    # The fractal dimension is related to the log of the price range
    fd = 1 + (np.log(price_range / price_range.rolling(window=2).mean()) / np.log(2))
    return fd.replace([np.inf, -np.inf], np.nan)


def add_fractal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds fractal dimension features to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with OHLC data.

    Returns:
        pd.DataFrame: The DataFrame with added fractal features.
    """
    df_copy = df.copy()
    for window in [14, 30, 60]:
        df_copy[f"fractal_dim_{window}"] = fractal_dimension(df_copy, window)

    return df_copy
