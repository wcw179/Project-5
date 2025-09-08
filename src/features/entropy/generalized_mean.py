"""
AFML §18.7 Entropy and the Generalized Mean

This module provides rolling features based on the generalized (power) mean of
nonnegative series (e.g., |returns|), and Rényi entropies computed from
histogram plug-in probabilities. The connection between Rényi entropies and
power means allows exploring a spectrum of tail sensitivity.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


def _power_mean(x: np.ndarray, p: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if p == 0.0:
        # Geometric mean
        xp = x[x > 0]
        if xp.size == 0:
            return np.nan
        return float(np.exp(np.mean(np.log(xp))))
    m = np.mean(np.power(x, p))
    if m < 0:
        return np.nan
    return float(np.power(m, 1.0 / p))


def rolling_power_means(
    series: pd.Series,
    *,
    orders: Sequence[float] = (-2.0, -1.0, 0.0, 1.0, 2.0),
    window: int = 288,
) -> pd.DataFrame:
    """Rolling generalized means of |series| for various orders p.

    Args:
        series: input series (e.g., returns) which will be absoluted and used
        orders: list of power mean orders; include 0.0 for geometric mean
        window: rolling window length (bars)
    Returns:
        DataFrame with columns power_mean_p{p}
    """
    x = series.abs()

    def _calc(a: np.ndarray, p: float) -> float:
        return _power_mean(a, p)

    out = pd.DataFrame(index=series.index)
    roll = x.rolling(window, min_periods=max(8, window // 4))
    for p in orders:
        out[f'power_mean_p{p}'] = roll.apply(lambda a: _calc(a, p), raw=True)
    return out


def _renyi_entropy_from_hist(prob: np.ndarray, q: float) -> float:
    prob = prob[(prob > 0) & (prob <= 1)]
    if prob.size == 0:
        return np.nan
    if np.isclose(q, 1.0):
        # Shannon limit
        return float(-np.sum(prob * np.log(prob)))
    s = np.sum(np.power(prob, q))
    if s <= 0:
        return np.nan
    return float(np.log(s) / (1.0 - q))


def rolling_renyi_entropy(
    series: pd.Series,
    *,
    window: int = 288,
    q_list: Sequence[float] = (0.5, 1.0, 2.0),
    bins: int = 11,
) -> pd.DataFrame:
    """Rolling Rényi entropies of order q for the given series (plug-in).

    Args:
        series: input series (e.g., returns)
        window: rolling window length
        q_list: Rényi orders to compute. Use 1.0 to approximate Shannon.
        bins: histogram bins for plug-in probabilities
    Returns:
        DataFrame with columns renyi_q{q}
    """
    out = pd.DataFrame(index=series.index)
    roll = series.rolling(window, min_periods=max(8, window // 4))

    def _calc_entropy(a: np.ndarray, q: float) -> float:
        if not np.isfinite(a).any():
            return np.nan
        hist, _ = np.histogram(a[np.isfinite(a)], bins=bins)
        p = hist / max(1, hist.sum())
        return _renyi_entropy_from_hist(p, q)

    for q in q_list:
        out[f'renyi_entropy_q{q}'] = roll.apply(lambda a: _calc_entropy(a, q), raw=True)
    return out


def add_generalized_mean_entropy_features(
    df: pd.DataFrame,
    *,
    window_power_mean: int = 144,  # Reduced for performance
    power_orders: Sequence[float] = (-1.0, 1.0, 2.0), # Reduced for performance
    window_renyi: int = 144,  # Reduced for performance
    renyi_q_list: Sequence[float] = (0.5, 2.0), # Reduced for performance
    renyi_bins: int = 10, # Reduced for performance
) -> pd.DataFrame:
    """Combine rolling power means of |returns| and Rényi entropies of returns."""
    out = df.copy()
    ret = out['close'].pct_change()

    pm = rolling_power_means(ret, orders=power_orders, window=window_power_mean)
    ry = rolling_renyi_entropy(ret, window=window_renyi, q_list=renyi_q_list, bins=renyi_bins)

    for c in pm.columns:
        out[c] = pm[c]
    for c in ry.columns:
        out[c] = ry[c]
    return out

