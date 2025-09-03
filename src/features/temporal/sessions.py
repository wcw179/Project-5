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

    # --- Advanced Temporal Features ---
    # Detects periods within 1 hour of a major session open/close
    transition_hours = [
        0,
        7,
        9,
        12,
        16,
        21,
    ]  # Asian open, London open, Asian close, NY open, London close, NY close
    df["session_transition"] = df.index.hour.isin(transition_hours).astype(int)

    # STUB: This feature requires an external news calendar data source.
    # For now, it's a placeholder that won't affect the model.
    df["news_release_proximity"] = 1.0

    return df
