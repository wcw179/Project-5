"""Functions for calculating market microstructure features."""

import pandas as pd
import pandas_ta as ta

from src.logger import logger


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds microstructure features related to liquidity and order flow."""
    df_copy = df.copy()

    # 1. Spread Feature (Dynamic)
    # Use the real spread if available, otherwise fall back to the proxy.
    if 'spread' in df_copy.columns and df_copy['spread'].notna().any():
        df_copy["spread_proxy"] = df_copy["spread"]
    else:
        df_copy["spread_proxy"] = df_copy["high"] - df_copy["low"]

    # 2. Volume Regime
    rolling_volume = df_copy["volume"].rolling(window=100)
    df_copy["volume_regime"] = 0  # Default to normal
    df_copy.loc[df_copy["volume"] < rolling_volume.quantile(0.25), "volume_regime"] = -1  # Low
    df_copy.loc[df_copy["volume"] > rolling_volume.quantile(0.75), "volume_regime"] = 1  # High

    # 3. New Dynamic Volume Features
    df_copy['volume_roc'] = ta.roc(df_copy['volume'], length=14)
    rolling_vol_50 = df_copy['volume'].rolling(window=50, min_periods=10).mean()
    df_copy['volume_abnormality'] = df_copy['volume'] / rolling_vol_50


    # On-Balance Volume (OBV) Slope
    # We use pandas-ta to get OBV, then calculate its slope.
    obv = ta.obv(close=df_copy["close"], volume=df_copy["volume"])
    df_copy["obv_slope"] = obv.diff().rolling(window=10).mean()

    # --- Placeholder Features ---
    # These features require tick-level data, which is not available in the current dataset.
    logger.warning(
        "Tick volume and order flow features are not implemented due to data limitations. Using neutral placeholders."
    )
    df_copy["tick_volume_ratio"] = 1.0  # STUB: Ratio of tick volume to real volume
    df_copy["order_flow_imbalance"] = 0.0  # STUB: Net buy vs. sell pressure

    return df_copy
