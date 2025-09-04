"""
AFML-inspired microstructural features (Chapter 19)

Implements classic microstructure proxies using bar-level OHLCV:
- Kyle's lambda (rolling): cov(r, signed_volume) / var(signed_volume)
- Amihud illiquidity (rolling): E[ |r| / Vol ]
- Roll's effective spread (rolling): 2 * sqrt( max(0, -cov(r_t, r_{t-1})) )
- Corwin–Schultz spread estimator (daily): closed-form estimate from daily highs/lows

Notes:
- r is close-to-close log return by default.
- signed_volume = sign(r) * volume (bar approximation of order flow)
- Daily CS spread is resampled from intraday bars and ffilled back to M5 index.
"""
from __future__ import annotations

from typing import Sequence, Optional
import numpy as np
import pandas as pd


def _logret(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def _rolling_cov(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return (
        a.rolling(window, min_periods=max(8, window // 4)).cov(b)
    )


def _rolling_var(a: pd.Series, window: int) -> pd.Series:
    return a.rolling(window, min_periods=max(8, window // 4)).var(ddof=0)


def add_afml_microstructure_features(
    df: pd.DataFrame,
    *,
    windows: Sequence[int] = (144, 288),  # 12h, 1d
    use_logret: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    close = out['close']
    vol = out.get('volume', pd.Series(0.0, index=out.index))

    r = _logret(close) if use_logret else close.pct_change()
    s_vol = np.sign(r.fillna(0.0)) * vol.fillna(0.0)

    for w in windows:
        cov = _rolling_cov(r, s_vol, w)
        var = _rolling_var(s_vol, w)
        out[f'kyle_lambda_w{w}'] = cov / var.replace(0, np.nan)
        out[f'amihud_illiq_w{w}'] = (
            (r.abs() / vol.replace(0, np.nan)).rolling(w, min_periods=max(8, w // 4)).mean()
        )
        # Roll spread
        r_lag = r.shift(1)
        cov_rr = _rolling_cov(r, r_lag, w)
        out[f'roll_spread_w{w}'] = 2.0 * np.sqrt(np.clip(-cov_rr, 0.0, np.inf))

    return out


def _corwin_schultz_spread_daily(df: pd.DataFrame) -> pd.Series:
    """Compute Corwin–Schultz spread estimator from daily highs/lows.
    Returns a daily series reindexed to df.index (M5) via ffill.
    """
    # Resample to daily high/low
    daily = pd.DataFrame({
        'high': df['high'].resample('1D').max(),
        'low': df['low'].resample('1D').min(),
    }).dropna()
    if daily.empty:
        return pd.Series(np.nan, index=df.index)

    hl = np.log(daily['high'] / daily['low']).clip(lower=0)
    beta = (hl**2).rolling(2).sum().shift(-1)  # 2-day window per paper
    gamma = (np.log(daily['high'].rolling(2).max() / daily['low'].rolling(2).min()))**2
    alpha = (np.sqrt(2*beta) - np.sqrt(beta)) / (3 - 2*np.sqrt(2))
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    # Map back to intraday index
    daily_spread = spread.reindex(df.index.date, method=None)
    # Reindex by date mapping
    ds = pd.Series(index=df.index, dtype=float)
    ds.loc[:] = np.nan
    # forward fill by day
    ds = spread.reindex(pd.to_datetime(daily.index.date)).reindex(df.index.date, method='ffill').to_numpy()
    return pd.Series(ds, index=df.index)


def add_corwin_schultz_spread(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        out['cs_spread_daily'] = _corwin_schultz_spread_daily(out)
    except Exception:
        out['cs_spread_daily'] = np.nan
    return out

