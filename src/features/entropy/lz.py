"""
Lempel–Ziv entropy rate estimators (AFML §18.4)

Provides functions to:
- Encode returns to a discrete alphabet (sign or quantile bins)
- Compute LZ76/Kontoyiannis-style entropy rate estimates
- Rolling-window LZ entropy rate features

Caveats:
- LZ estimators are computationally heavy. Use moderate windows (<= 288) and
  a small alphabet (e.g., 3 to 7 bins). These are proxies for complexity.
"""
from __future__ import annotations

from typing import Literal, Sequence
import math
import numpy as np
import pandas as pd

AlphabetMethod = Literal["sign", "quantile"]


def encode_returns_to_symbols(
    close: pd.Series,
    *,
    method: AlphabetMethod = "quantile",
    n_bins: int = 5,
) -> pd.Series:
    ret = close.pct_change().fillna(0.0)
    if method == "sign":
        # Map to {0,1,2} for {-1,0,+1}
        s = np.sign(ret)
        sym = (s + 1).astype(int)
        return pd.Series(sym, index=close.index)
    # Quantile binning
    q = pd.qcut(ret.rank(method="first"), n_bins, labels=False, duplicates="drop")
    return q.astype(int)


def _lz76_match_length(seq: Sequence[int], start: int) -> int:
    """Return the length of the shortest substring starting at `start` that has
    not appeared before. Based on LZ76 parsing idea.
    """
    n = len(seq)
    L = 1
    while start + L <= n:
        sub = tuple(seq[start : start + L])
        # search in prefix [0:start]
        found = False
        for j in range(max(0, start - L), start):
            if tuple(seq[j : j + L]) == sub:
                found = True
                break
        if not found:
            return L
        L += 1
    return L


def lz76_entropy_rate(seq: Sequence[int], alphabet_size: int) -> float:
    """Kontoyiannis/LZ76-style entropy rate estimate (bits/symbol).

    h ≈ (n * log_a n) / sum_i Λ_i  where Λ_i is match length at i, a is alphabet size.
    We convert to bits/symbol by multiplying log base change factor (log2 / log a).
    """
    arr = list(seq)
    n = len(arr)
    if n <= 1 or alphabet_size <= 1:
        return float("nan")
    Ls = []
    for i in range(n):
        Ls.append(_lz76_match_length(arr, i))
    denom = sum(Ls)
    if denom <= 0:
        return float("nan")
    h_a = (n * math.log(n, alphabet_size)) / denom
    # convert to bits (base 2)
    return h_a * math.log(alphabet_size, 2)


def rolling_lz_entropy_rate(
    close: pd.Series,
    *,
    window: int = 288,
    method: AlphabetMethod = "quantile",
    n_bins: int = 5,
) -> pd.Series:
    syms = encode_returns_to_symbols(close, method=method, n_bins=n_bins).astype(int)

    def _calc(a: np.ndarray) -> float:
        return lz76_entropy_rate(a.tolist(), alphabet_size=n_bins)

    return syms.rolling(window, min_periods=max(16, window // 4)).apply(_calc, raw=True)

