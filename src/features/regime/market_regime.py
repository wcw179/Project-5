"""
Features designed to detect the prevailing market regime, such as trend strength
and volatility stress.
"""
import pandas as pd
import pandas_ta as ta

def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds features to help identify the current market regime.

    Features:
    - ADX (Average Directional Index): Measures trend strength.
    - Volatility Stress: Measures if recent volatility is high or low compared
      to a longer-term baseline.

    Args:
        df: DataFrame with OHLC data.

    Returns:
        DataFrame with added regime features.
    """
    df_copy = df.copy()

    # 1. ADX for Trend Strength
    # A high ADX indicates a strong trend (either up or down).
    # A low ADX indicates a weak trend or a ranging market.
    adx = df_copy.ta.adx(length=14)
    if isinstance(adx, pd.DataFrame) and 'ADX_14' in adx.columns:
        df_copy['adx_14'] = adx['ADX_14']

    # 2. Volatility Stress Indicator
    # Compares short-term volatility to long-term volatility.
    # A high value indicates recent volatility is much higher than the baseline.
    short_term_vol = df_copy['close'].rolling(window=20).std()
    long_term_vol = df_copy['close'].rolling(window=100).std()
    df_copy['volatility_stress'] = (short_term_vol / long_term_vol) - 1

    return df_copy
