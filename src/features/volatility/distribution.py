"""Functions for calculating volatility and distribution features."""

import pandas as pd
import pandas_ta as ta

from src.logger import logger


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

    # --- Advanced Volatility & Distribution Features ---
    # Volatility Expansion Rate (rate of change of ATR)
    df["vol_expansion_rate"] = df["atr_14"].diff().rolling(window=10).mean()

    # Value-at-Risk (VaR) Breach Frequency
    rolling_std = returns.rolling(window=100).std()
    var_95 = 1.65 * rolling_std  # 95% VaR for a normal distribution
    df["var_breach"] = (returns < -var_95).astype(int)
    df["var_breach_frequency"] = df["var_breach"].rolling(window=200).mean()

    # Extreme Move Frequency (frequency of > 3-sigma moves)
    extreme_move = (returns.abs() > (3 * rolling_std)).astype(int)
    df["extreme_move_frequency"] = extreme_move.rolling(window=200).mean()

    # --- Placeholder Features ---
    logger.warning(
        "GARCH forecast feature is not implemented due to dependency requirements. Using a neutral placeholder."
    )
    df["garch_forecast"] = 0.0  # STUB: GARCH(1,1) forecast

    return df
