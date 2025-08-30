"""Functions for calculating volatility and distribution features."""

import pandas as pd
import pandas_ta as ta


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds volatility and statistical distribution features."""

    # Calculate daily returns for distribution analysis
    returns = df["close"].pct_change()

    # Rolling Skewness and Kurtosis
    # These measure the asymmetry and 'tailedness' of the returns distribution
    df["returns_skew_rolling"] = returns.rolling(window=100).skew()
    df["returns_kurt_rolling"] = returns.rolling(window=100).kurt()

    # Average True Range (ATR) as a primary volatility measure
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # --- Placeholder Features ---
    df["volatility_regime"] = 0  # STUB: -1 for low, 0 for medium, 1 for high
    df["vol_expansion_rate"] = 0.0  # STUB: Rate of change of ATR
    df["garch_forecast"] = 0.0  # STUB: GARCH(1,1) forecast of next period's volatility
    df["var_breach_frequency"] = 0.0  # STUB: How often returns exceed 95% VaR
    df["extreme_move_frequency"] = 0.0  # STUB: Frequency of > 3-sigma moves

    return df
