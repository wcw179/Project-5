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
    close = out['close']

    # Initialize placeholder columns
    out['sadf_log_tstat'] = 0.0
    out['sadf_log_crit'] = 0.0
    out['sadf_log_break'] = 0.0

    try:
        # To manage performance, run SADF on a sampled subset of the data
        # We run it once every 4 hours (48 * 5 mins)
        sampling_rate = 48
        close_sampled = close.iloc[::sampling_rate]

        print(f"[Info] Running SADF on {len(close_sampled)} sampled points...")
        # min_length is reduced as we are working with a smaller, sampled dataset
        sadf_stats = get_sadf(series=close_sampled.dropna(), model='linear', lags=10, min_length=60, add_const=True, num_threads=1, verbose=False)

        # Create full-length series and forward-fill the results
        sadf_tstat = sadf_stats['stat'].reindex(out.index).ffill()
        sadf_crit = sadf_stats['critical_value'].reindex(out.index).ffill()

        # If successful, populate the columns
        out['sadf_log_tstat'] = sadf_tstat
        out['sadf_log_crit'] = sadf_crit
        out['sadf_log_break'] = (sadf_tstat > sadf_crit).astype(int)
        print("[Info] SADF calculation successful.")

    except Exception as e:
        # If the test fails, log a warning but proceed with placeholder values
        print(f"[Warning] SADF calculation failed: {e}. Using neutral placeholders.")

    return out

