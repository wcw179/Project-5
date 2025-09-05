"""
Structural Breaks features (V4 §2.6)

Implements lightweight proxies for structural change and explosiveness:
- CUSUM event rate over rolling windows
- Time since last CUSUM event
- Positive/negative CUSUM intensities (normalized by rolling std)
- Drawup/Drawdown magnitudes as regime-shift proxies
- One-sided cumulative deviation statistics (sub-/super-martingale proxies)

All computations assume df.index is DatetimeIndex (UTC) on a 5-minute grid and
columns include at least ['close'].
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

from src.features.structural_breaks.afml import get_chu_stinchcombe_white_statistics

def _rolling_std_eps(x: pd.Series, w: int) -> pd.Series:
    s = x.rolling(w, min_periods=max(8, w // 4)).std(ddof=0)
    return s.replace(0, np.nan)


def _cusum_events(ret: pd.Series, k: float) -> pd.Series:
    """CUSUM filter (Brownlees & Gallo-style) on returns.
    Emits 1 when an event occurs (positive or negative excursion beyond k).
    """
    # normalized return
    r = ret.fillna(0.0)
    # use rolling std on a moderate horizon (e.g., 1 day = 288 bars) for normalization
    vol = _rolling_std_eps(r, 288)
    z = r / vol

    pos = 0.0
    neg = 0.0
    out = np.zeros(len(z), dtype=np.int8)
    for i, val in enumerate(z.to_numpy()):
        if np.isnan(val):
            pos = 0.0
            neg = 0.0
            continue
        pos = max(0.0, pos + val)
        neg = min(0.0, neg + val)
        if pos > k or neg < -k:
            out[i] = 1
            pos = 0.0
            neg = 0.0
    return pd.Series(out, index=ret.index)


def add_structural_break_features(
    df: pd.DataFrame,
    *,
    windows: Sequence[int] = (288, 1440),  # 1d, 5d on M5
    cusum_k: float = 5.0,
) -> pd.DataFrame:
    out = df.copy()
    close = out['close']
    ret = close.pct_change()
    # Chu-Stinchcombe-White Test for breaks
    csw_stats = get_chu_stinchcombe_white_statistics(close, test_type='one_sided')
    csw_break = (csw_stats['stat'] > csw_stats['critical_value']).astype(int)
    out = out.join(csw_break.rename('csw_break'))
    out['csw_break'].fillna(0, inplace=True)


    # CUSUM event indicator
    ev = _cusum_events(ret, cusum_k)
    out['cusum_event'] = ev

    # Event rate per window and time since last event
    for w in windows:
        out[f'cusum_event_rate_w{w}'] = ev.rolling(w, min_periods=max(8, w // 4)).mean()
        # time since last event in bars
        idx = (~ev.astype(bool)).astype(int)
        # cumulative count of consecutive non-events
        tsle = idx.groupby((ev == 1).cumsum()).cumcount()
        out[f'time_since_last_event_w{w}'] = tsle

    # Positive/negative cumulative deviations (martingale proxies)
    # cumulative sum of demeaned returns over rolling window
    for w in windows:
        r = ret - ret.rolling(w, min_periods=max(8, w // 4)).mean()
        s = r.rolling(w, min_periods=max(8, w // 4)).apply(np.nansum, raw=True)
        vol = _rolling_std_eps(ret, w)
        z = s / (vol * np.sqrt(w))
        out[f'one_sided_dev_z_w{w}'] = z

    # Drawup/Drawdown magnitudes
    for w in windows:
        roll = close.rolling(w, min_periods=max(8, w // 4))
        max_up = (close / roll.min()) - 1.0
        max_dn = (close / roll.max()) - 1.0
        out[f'max_drawup_w{w}'] = max_up
        out[f'max_drawdown_w{w}'] = max_dn

    return out

