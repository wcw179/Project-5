"""Functions for calculating trend and momentum features."""

import pandas as pd
import pandas_ta as ta

from src.logger import logger


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds trend and momentum features, including multi-timeframe alignment."""

    # --- Multi-Timeframe EMA Alignment ---
    # --- M5 EMAs for baseline strategy ---
    df["ema_20"] = ta.ema(df["close"], length=20)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["ema_200"] = ta.ema(df["close"], length=200)

    # Resample to H1 and H4 to calculate longer-term EMAs
    df_h1 = df["close"].resample("h").last()
    df_h4 = df["close"].resample("4h").last()

    # Calculate EMAs on H1 data
    ema20_h1 = ta.ema(df_h1, length=20)
    ema50_h1 = ta.ema(df_h1, length=50)
    ema200_h1 = ta.ema(df_h1, length=200)

    # Calculate EMAs on H4 data
    ema20_h4 = ta.ema(df_h4, length=20)
    ema50_h4 = ta.ema(df_h4, length=50)

    # Forward-fill the resampled EMAs and align back to the original M5 index
    df["ema20_h1"] = ema20_h1.reindex(df.index, method="ffill")
    df["ema50_h1"] = ema50_h1.reindex(df.index, method="ffill")
    df["ema200_h1"] = ema200_h1.reindex(df.index, method="ffill")
    df["ema20_h4"] = ema20_h4.reindex(df.index, method="ffill")
    df["ema50_h4"] = ema50_h4.reindex(df.index, method="ffill")

    # EMA Alignment Features
    df["ema_algnment"] = (
        (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"])
    ).astype(int) - (
        (df["ema_20"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"])
    ).astype(
        int
    )

    df["ema_alignment_h1"] = (
        (df["ema20_h1"] > df["ema50_h1"]) & (df["ema50_h1"] > df["ema200_h1"])
    ).astype(int) - (
        (df["ema20_h1"] < df["ema50_h1"]) & (df["ema50_h1"] < df["ema200_h1"])
    ).astype(
        int
    )

    df["ema_alignment_h4"] = ((df["ema20_h4"] > df["ema50_h4"])).astype(int) - (
        (df["ema20_h4"] < df["ema50_h4"])
    ).astype(int)

    # --- Advanced Trend & Momentum Features ---
    # EMA Slope Acceleration (2nd derivative of a short-term EMA)
    ema_slope = df["ema_20"].diff()
    df["ema_slope_acceleration"] = ema_slope.diff().rolling(window=10).mean()

    # Trend Strength Index (using ADX)
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["trend_strength_index"] = adx["ADX_14"]

    # Momentum Regime (using RSI slope)
    rsi = ta.rsi(df["close"], length=14)
    rsi_slope = rsi.diff().rolling(window=10).mean()
    df["momentum_regime"] = 0
    df.loc[rsi_slope > 0.5, "momentum_regime"] = 1  # Strong positive momentum
    df.loc[rsi_slope < -0.5, "momentum_regime"] = -1  # Strong negative momentum

    # --- Placeholder Features ---
    logger.warning(
        "MACD and RSI divergence features are not yet implemented. Using neutral placeholders."
    )
    df["macd_divergence"] = 0  # STUB: -1 for bearish, 1 for bullish, 0 for none
    df["rsi_divergence"] = 0  # STUB: -1 for bearish, 1 for bullish, 0 for none

    return df
