"""
Labeling functions using the mlfinpy library, adapted for the project's multi-label format.
"""
import pandas as pd
from mlfinpy.labeling import get_events, add_vertical_barrier, get_bins
from mlfinpy.util import get_daily_vol

from src.labeling.triple_barrier_afml import TripleBarrierConfig

def generate_labels_with_mlfinpy(data: pd.DataFrame, config: TripleBarrierConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates primary model labels using mlfinpy, returning a multi-label format.

    This implementation correctly separates the logic for long and short positions
    by using the 'side_prediction' parameter in get_events, which is crucial for
    meta-labeling and avoiding logical errors in barrier calculation.

    Args:
        data: DataFrame with OHLCV data.
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
    all_events = []  # To store events for sample_info

    for rr in config.rr_multiples:
        pt_sl = [rr, config.atr_mult_base]

        # --- 2. Process LONG positions ---
        # Create a series of side predictions (1 for long)
        long_sides = pd.Series(1, index=close.index)
        events_long = get_events(
            close=close, t_events=close.index, pt_sl=pt_sl, target=vol, min_ret=0,
            num_threads=1, vertical_barrier_times=vertical_barriers, side_prediction=long_sides
        )
        bins_long = get_bins(triple_barrier_events=events_long, close=close)

        labels = pd.DataFrame(index=close.index)
        long_hit_col = f'long_hit_h{config.horizons[0]}_rr{rr}'
        labels[long_hit_col] = (bins_long['bin'] == 1).reindex(close.index).fillna(0).astype(int)

        # --- 3. Process SHORT positions (if included) ---
        if config.include_short:
            short_sides = pd.Series(-1, index=close.index)
            events_short = get_events(
                close=close, t_events=close.index, pt_sl=pt_sl, target=vol, min_ret=0,
                num_threads=1, vertical_barrier_times=vertical_barriers, side_prediction=short_sides
            )
            bins_short = get_bins(triple_barrier_events=events_short, close=close)
            short_hit_col = f'short_hit_h{config.horizons[0]}_rr{rr}'
            labels[short_hit_col] = (bins_short['bin'] == 1).reindex(close.index).fillna(0).astype(int) # bin is 1 if TP is hit
            all_events.append(events_short)

        all_labels.append(labels)
        all_events.append(events_long)

    labels_df = pd.concat(all_labels, axis=1)

    # 4. Create a compatible sample_info DataFrame
    if all_events:
        full_events_df = pd.concat(all_events)
        sample_info = full_events_df.groupby(full_events_df.index).last()[['t1', 'trgt']]
    else:
        sample_info = pd.DataFrame(columns=['t1', 'trgt'])

    return labels_df, sample_info

