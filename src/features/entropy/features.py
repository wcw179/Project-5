"""
Entropy feature set (V4 §2.6)

Implements practical, efficient proxies of entropy/complexity on rolling windows.
- shannon_entropy_sign_wN: Shannon entropy of returns sign over window N
- sign_change_rate_wN: proportion of sign flips (complexity proxy similar to LZ)
- price_range_entropy_wN: entropy over binned intrawindow returns

Notes:
- Windows are in bars (M5). Choose windows that map to sensible horizons (e.g.,
  144 bars ≈ 12h, 288 bars ≈ 1 day).
"""
from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np
import pandas as pd


def _shannon_entropy_from_probs(p: np.ndarray) -> float:
    p = p[(p > 0) & (p <= 1)]
    if p.size == 0:
        return np.nan
    return float(-(p * np.log2(p)).sum())


def _rolling_entropy_sign(ret: pd.Series, window: int) -> pd.Series:
    def _ent(x: np.ndarray) -> float:
        # states: {-1, 0, +1}
        vals, counts = np.unique(x, return_counts=True)
        probs = counts / counts.sum()
        return _shannon_entropy_from_probs(probs)

    return (
        ret.fillna(0.0)
        .apply(np.sign)
        .rolling(window, min_periods=max(8, window // 4))
        .apply(lambda a: _ent(a.astype(int)), raw=True)
    )


def _rolling_sign_change_rate(ret: pd.Series, window: int) -> pd.Series:
    s = np.sign(ret.fillna(0.0).to_numpy())
    flips = np.abs(np.diff(s)) > 0
    flips = np.concatenate([[False], flips])  # align length
    sr = pd.Series(flips, index=ret.index).rolling(window, min_periods=max(8, window // 4)).mean()
    return sr


def _rolling_binned_entropy(ret: pd.Series, window: int, bins: int = 7) -> pd.Series:
    # entropy of intrawindow return distribution after binning
    def _bin_ent(x: np.ndarray) -> float:
        x_finite = x[np.isfinite(x)]
        if len(x_finite) == 0:
            return np.nan
        hist, _ = np.histogram(x_finite, bins=bins)
        p = hist / max(1, hist.sum())
        return _shannon_entropy_from_probs(p)

    return ret.rolling(window, min_periods=max(8, window // 4)).apply(lambda a: _bin_ent(a), raw=True)


def add_entropy_features(
    df: pd.DataFrame,
    *,
    windows: Sequence[int] = (144, 288),  # 12h, 1d on M5
) -> pd.DataFrame:
    out = df.copy()
    ret = out['close'].pct_change()

    for w in windows:
        out[f'shannon_entropy_sign_w{w}'] = _rolling_entropy_sign(ret, w)
        out[f'sign_change_rate_w{w}'] = _rolling_sign_change_rate(ret, w)
        out[f'price_range_entropy_w{w}'] = _rolling_binned_entropy(ret, w, bins=9)

    return out

