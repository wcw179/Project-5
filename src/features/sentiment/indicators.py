"""Functions for calculating sentiment and positioning features."""

import pandas as pd


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds sentiment and market positioning features (STUBS)."""

    # --- Placeholder Features ---
    # These features require external data sources and are stubbed for now.
    df["vix_regime"] = 0  # STUB: -1 for low, 0 for normal, 1 for high VIX
    df["put_call_ratio"] = 0.0  # STUB: CBOE put/call ratio
    df["commitment_traders"] = (
        0.0  # STUB: Net non-commercial positioning from COT reports
    )
    df["sentiment_extremes"] = 0  # STUB: -1 for extreme fear, 1 for extreme greed

    return df
