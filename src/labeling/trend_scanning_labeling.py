"""
Implements the Trend-Scanning Meta-Labeling strategy from AFML Chapter 3.

This method uses trend-scanning to identify potential trading opportunities (t_events)
and then applies the triple-barrier method to determine the outcome of those trades.
"""
import pandas as pd
from mlfinpy.labeling import get_events, add_vertical_barrier, get_bins
from mlfinpy.labeling.trend_scanning import trend_scanning_labels
from mlfinpy.util import get_daily_vol

from src.labeling.triple_barrier_afml import TripleBarrierConfig

def generate_trend_scanning_meta_labels(
    data: pd.DataFrame, 
    config: TripleBarrierConfig,
    ts_window: int = 20, 
    ts_t_value_threshold: float = 2.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates meta-labels using the trend-scanning method.

    Args:
        data: DataFrame with OHLCV data.
        config: TripleBarrierConfig object.
        ts_window: The look-forward window for the trend-scanning algorithm.
        ts_t_value_threshold: The minimum absolute t-value to consider a trend significant.

    Returns:
        A tuple of (labels_df, sample_info_df).
    """
    close = data['close']
    
    # Step 1: Generate trend-scanning labels to identify significant events (t_events)
    ts_labels = trend_scanning_labels(close, look_forward_window=ts_window)
    
    # Filter for events with a t-value above the threshold
    significant_trends = ts_labels[ts_labels['t_value'].abs() > ts_t_value_threshold]
    t_events = significant_trends.index
    side_prediction = significant_trends['t_value'].apply(np.sign)

    if t_events.empty:
        logger.warning("No significant trends found by trend-scanning. Returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    # Step 2: Apply the triple-barrier method to these significant t_events
    num_days = config.horizons[0] / 288
    vol = get_daily_vol(close, lookback=config.vol_window).ffill().bfill()
    vertical_barriers = add_vertical_barrier(t_events=t_events, close=close, num_days=num_days)

    all_labels = []
    for rr in config.rr_multiples:
        pt_sl = [rr, config.atr_mult_base]
        
        events = get_events(
            close=close, 
            t_events=t_events, 
            pt_sl=pt_sl, 
            target=vol.loc[t_events], 
            min_ret=0,
            num_threads=1, 
            vertical_barrier_times=vertical_barriers,
            side_prediction=side_prediction
        )
        
        bins = get_bins(triple_barrier_events=events, close=close)
        
        # Note: For meta-labeling, the final label is just the outcome (1 for win, 0 for loss/timeout)
        # The side is determined by the primary model (trend-scanning)
        meta_label_col = f'meta_label_h{config.horizons[0]}_rr{rr}'
        labels = pd.DataFrame(index=close.index)
        labels[meta_label_col] = (bins['bin'] == 1).reindex(close.index).fillna(0).astype(int)
        all_labels.append(labels)

    labels_df = pd.concat(all_labels, axis=1)
    
    # sample_info needs to be aligned with the final labels' index
    sample_info = pd.DataFrame({'t1': vertical_barriers, 'trgt': vol, 'side': side_prediction})
    sample_info = sample_info.loc[labels_df.index.intersection(sample_info.index)]

    return labels_df, sample_info
