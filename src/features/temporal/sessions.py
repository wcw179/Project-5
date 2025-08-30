"""Functions for calculating temporal and session-based features."""

import pandas as pd


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds trading session and time-based features to the DataFrame."""
    # Ensure the index is a DatetimeIndex in UTC
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Basic time features
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek  # Monday=0, Sunday=6

    # Trading session features (UTC times)
    # Note: These are simplified representations.
    df["is_asian_session"] = ((df.index.hour >= 0) & (df.index.hour < 9)).astype(int)
    df["is_london_session"] = ((df.index.hour >= 7) & (df.index.hour < 16)).astype(int)
    df["is_ny_session"] = ((df.index.hour >= 12) & (df.index.hour < 21)).astype(int)
    df["is_overlap_session"] = (
        (df["is_london_session"] == 1) & (df["is_ny_session"] == 1)
    ).astype(int)

    # --- Placeholder Features ---
    # These require more complex logic or external data.
    df["session_transition"] = 0  # STUB: 1 if within 1h of a session change
    df["news_release_proximity"] = 1.0  # STUB: Normalized time to next major news event

    return df
