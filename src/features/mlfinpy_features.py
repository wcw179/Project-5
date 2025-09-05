"""
Features created using the mlfinpy library.
"""
import pandas as pd
from mlfinpy.filters import cusum_filter, z_score_filter
from mlfinpy.structural_breaks import get_sadf

def add_mlfinpy_filter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features based on mlfinpy filters.

    Args:
        df: DataFrame with at least 'close' prices.

    Returns:
        DataFrame with new filter-based features.
    """
    out = df.copy()
    close = out['close']

    # 1. CUSUM Filter
    daily_std = close.pct_change().rolling(288).std().mean()
    threshold = daily_std * 2 if pd.notna(daily_std) and daily_std > 0 else 0.001
    event_timestamps_cusum = cusum_filter(close, threshold=threshold)
    cusum_series = pd.Series(1, index=event_timestamps_cusum)
    out['mlfinpy_cusum_event'] = cusum_series.reindex(out.index).fillna(0).astype(int)

    # 2. Z-Score Filter
    event_timestamps_zscore = z_score_filter(close, mean_window=288, std_window=288, z_score=3.0)
    z_score_series = pd.Series(1, index=event_timestamps_zscore)
    out['mlfinpy_z_score_event'] = z_score_series.reindex(out.index).fillna(0).astype(int)

    return out

def add_mlfinpy_structural_break_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features based on mlfinpy structural break tests.

    Args:
        df: DataFrame with at least 'close' prices.

    Returns:
        DataFrame with new structural break features.
    """
    out = df.copy()

    # Supremum Augmented Dickey-Fuller (SADF) - Computationally intensive
    # This test is currently disabled to improve pipeline speed.
    # To enable, uncomment the following lines and ensure 'close' is defined.
    # close = out['close']
    # sadf_stats = get_sadf(close, min_length=288, add_const=True, model='SADF')
    # out['mlfinpy_sadf'] = sadf_stats

    return out

