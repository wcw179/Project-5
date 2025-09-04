"""
AFML Chapter 18: Entropy Features (practical implementations)

Implements rolling Approximate Entropy (ApEn), Sample Entropy (SampEn), and
Permutation Entropy (PE) on close-to-close returns.

Notes:
- Window-based computation; complexity grows with window size and embedding dim.
- Defaults are chosen for stability on intraday data.
- If performance is an issue, downsample or reduce windows/embedding dims.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


def _approximate_entropy(x: np.ndarray, m: int, r: float) -> float:
    # Pincus (1991) ApEn implementation using Chebyshev distance
    # Returns NaN if insufficient length or degenerate tolerance
    N = x.size
    if N <= m + 1 or r <= 0 or not np.isfinite(r):
        return np.nan

    def _phi(mm: int) -> float:
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)
        # Chebyshev distance across pairs; threshold <= r
        # Compute count of matches for each template
        counts = []
        for i in range(emb.shape[0]):
            d = np.max(np.abs(emb - emb[i]), axis=1)
            c = np.mean(d <= r)
            counts.append(c)
        counts = np.array(counts)
        counts = counts[counts > 0]
        if counts.size == 0:
            return np.inf
        return np.mean(np.log(counts))

    return float(_phi(m) - _phi(m + 1))


def _sample_entropy(x: np.ndarray, m: int, r: float) -> float:
    # Richman & Moorman (2000) SampEn
    N = x.size
    if N <= m + 1 or r <= 0 or not np.isfinite(r):
        return np.nan
    emb_m = np.lib.stride_tricks.sliding_window_view(x, m)
    emb_mp1 = np.lib.stride_tricks.sliding_window_view(x, m + 1)

    def _count_sim(emb: np.ndarray) -> int:
        cnt = 0
        for i in range(emb.shape[0]):
            d = np.max(np.abs(emb - emb[i]), axis=1)
            cnt += int(np.sum(d <= r) - 1)  # exclude self-match
        return cnt

    B = _count_sim(emb_m)
    A = _count_sim(emb_mp1)
    if B == 0 or A == 0:
        return np.nan
    return float(-np.log(A / B))


def _permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1) -> float:
    # Bandt & Pompe (2002) Permutation Entropy
    N = x.size
    L = N - (m - 1) * tau
    if L <= 0:
        return np.nan
    patterns = {}
    for i in range(L):
        window = x[i : i + m * tau : tau]
        ranks = tuple(np.argsort(window))
        patterns[ranks] = patterns.get(ranks, 0) + 1
    counts = np.array(list(patterns.values()), dtype=float)
    p = counts / counts.sum()
    h = -np.sum(p * np.log2(p))
    # normalized by log2(m!)
    h_norm = h / np.log2(np.math.factorial(m))
    return float(h_norm)


def add_afml_entropy_features(
    df: pd.DataFrame,
    *,
    windows: Sequence[int] = (144, 288),
    m_apen: int = 2,
    r_apen: float | None = None,  # if None, set to 0.2 * std window
    m_sampen: int = 2,
    r_sampen: float | None = None,  # if None, set to 0.2 * std window
    m_perm: int = 3,
    tau_perm: int = 1,
) -> pd.DataFrame:
    out = df.copy()
    ret = out['close'].pct_change().fillna(0.0)

    for w in windows:
        roll = ret.rolling(w, min_periods=max(16, w // 4))
        std_w = roll.std().replace(0, np.nan)

        # ApEn
        rA = (0.2 * std_w) if r_apen is None else r_apen
        out[f'apen_w{w}'] = roll.apply(
            lambda a: _approximate_entropy(a, m_apen, float(np.nanmean(rA.loc[roll._get_window_indexer(a.shape[0])])) if r_apen is None else rA),  # type: ignore  # noqa: E501
            raw=True,
        )

        # SampEn
        rS = (0.2 * std_w) if r_sampen is None else r_sampen
        out[f'sampen_w{w}'] = roll.apply(
            lambda a: _sample_entropy(a, m_sampen, float(np.nanmean(rS.loc[roll._get_window_indexer(a.shape[0])])) if r_sampen is None else rS),  # type: ignore  # noqa: E501
            raw=True,
        )

        # Permutation Entropy (normalized)
        out[f'perm_entropy_w{w}'] = roll.apply(
            lambda a: _permutation_entropy(a, m=m_perm, tau=tau_perm), raw=True
        )

    return out

