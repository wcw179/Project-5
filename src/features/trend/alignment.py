"""Functions for calculating trend and momentum features."""

import pandas as pd
import pandas_ta as ta


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

    # --- Placeholder Features ---
    df["ema_slope_acceleration"] = 0.0  # STUB: 2nd derivative of a short-term EMA
    df["trend_strength_index"] = 0.0  # STUB: Composite of ADX, etc.
    df["macd_divergence"] = 0  # STUB: -1 for bearish, 1 for bullish, 0 for none
    df["rsi_divergence"] = 0  # STUB: -1 for bearish, 1 for bullish, 0 for none
    df["momentum_regime"] = 0  # STUB: -1 for weak, 0 for neutral, 1 for strong

    return df
