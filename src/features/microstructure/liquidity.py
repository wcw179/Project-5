"""Functions for calculating market microstructure features."""

import pandas as pd
import pandas_ta as ta

from src.logger import logger


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds microstructure features related to liquidity and order flow."""

    # Spread Proxy
    # A simple proxy using the high-low range of a bar.
    df["spread_proxy"] = df["high"] - df["low"]

    # Volume Regime
    # Classify volume as low, normal, or high based on rolling quantiles.
    rolling_volume = df["volume"].rolling(window=100)
    df["volume_regime"] = 0  # Default to normal
    df.loc[df["volume"] < rolling_volume.quantile(0.25), "volume_regime"] = -1  # Low
    df.loc[df["volume"] > rolling_volume.quantile(0.75), "volume_regime"] = 1  # High

    # On-Balance Volume (OBV) Slope
    # We use pandas-ta to get OBV, then calculate its slope.
    obv = ta.obv(close=df["close"], volume=df["volume"])
    df["obv_slope"] = obv.diff().rolling(window=10).mean()

    # --- Placeholder Features ---
    # These features require tick-level data, which is not available in the current dataset.
    logger.warning(
        "Tick volume and order flow features are not implemented due to data limitations. Using neutral placeholders."
    )
    df["tick_volume_ratio"] = 1.0  # STUB: Ratio of tick volume to real volume
    df["order_flow_imbalance"] = 0.0  # STUB: Net buy vs. sell pressure

    return df
