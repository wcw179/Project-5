"""
Labeling functions using the mlfinpy library, adapted for the project's multi-label format.
"""
import pandas as pd
from mlfinpy.labeling import get_events, add_vertical_barrier, get_bins
from mlfinpy.util import get_daily_vol

from src.labeling.triple_barrier_afml import TripleBarrierConfig

def generate_labels_with_mlfinpy(data: pd.DataFrame, config: TripleBarrierConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates primary model labels using mlfinpy, returning a multi-label format.

    Args:
        data: DataFrame with OHLCV data.
        symbol: The symbol being processed.
        config: TripleBarrierConfig object.

    Returns:
        A tuple of (labels_df, sample_info_df).
    """
    close = data['close']
    num_days = config.horizons[0] / 288  # Convert bars to days

    # 1. Calculate daily volatility and add vertical barrier
    vol = get_daily_vol(close, lookback=config.vol_window).ffill().bfill()
    vertical_barriers = add_vertical_barrier(t_events=close.index, close=close, num_days=num_days)

    all_labels = []
    # Get events once for all R/R multiples
    base_events = get_events(
        close=close,
        t_events=close.index,
        pt_sl=[1, 1],  # Dummy values, will be replaced by binning logic
        target=vol,
        min_ret=0,
        num_threads=1,
        vertical_barrier_times=vertical_barriers
    )

    for rr in config.rr_multiples:
        # 2. Get bins for the specific R/R multiple
        # We manually set the pt/sl levels on the events df for get_bins
        events_with_rr = base_events.copy()
        events_with_rr['pt'] = rr
        events_with_rr['sl'] = config.atr_mult_base

        bins = get_bins(triple_barrier_events=events_with_rr, close=close)

        # 3. Convert to multi-label format, ensuring proper alignment and filling NaNs
        horizon = config.horizons[0]
        long_hit_col = f'long_hit_h{horizon}_rr{rr}'
        short_hit_col = f'short_hit_h{horizon}_rr{rr}'

        labels = pd.DataFrame(index=close.index)

        long_hits = (bins['bin'] == 1)
        labels[long_hit_col] = long_hits.reindex(close.index).fillna(0).astype(int)

        if config.include_short:
            short_hits = (bins['bin'] == -1)
            labels[short_hit_col] = short_hits.reindex(close.index).fillna(0).astype(int)

        all_labels.append(labels)

    labels_df = pd.concat(all_labels, axis=1)

    # Create a compatible sample_info DataFrame from the base events
    sample_info = pd.DataFrame({'t1': base_events['t1'], 'trgt': base_events['trgt']})

    return labels_df, sample_info

