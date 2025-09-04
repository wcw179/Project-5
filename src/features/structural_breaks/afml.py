"""
AFML Chapter 17: Structural Break Tests (practical implementations)

Includes:
- 17.2 Types of Structural Break Tests (summary via helpers)
- 17.3.1 Brown–Durbin–Evans CUSUM on recursive residuals (simplified proxy)
- 17.3.2 Chu–Stinchcombe–White CUSUM on levels
- 17.4.1 Chow-type Dickey–Fuller
- 17.4.2 Supremum Augmented Dickey–Fuller (SADF)
- 17.4.3 Sub- and Super-Martingale tests (one-sided cumulative deviation Z)

Notes:
- Implementations are designed to be dependency-light (numpy/pandas only) and
  fast enough for intraday usage. For research-grade tests, consider statsmodels
  implementations and critical values from the literature.
- All functions assume a DatetimeIndex at 5-minute UTC when fed bar data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class CusumResult:
    stat: pd.Series  # cumulative statistic
    crit_upper: Optional[pd.Series] = None
    crit_lower: Optional[pd.Series] = None


def _demean(x: pd.Series) -> pd.Series:
    return x - x.mean()


def csw_cusum_levels(x: pd.Series) -> CusumResult:
    """17.3.2 Chu–Stinchcombe–White CUSUM Test on levels (simplified).

    C_t = cumsum((x - mean(x)) / std(x))
    Returns the cumulative path; rejection requires critical boundaries which
    depend on sample and desired size. We leave crit bands None (caller can set).
    """
    x = x.dropna()
    if x.empty:
        return CusumResult(stat=x)
    z = (x - x.mean()) / (x.std(ddof=0) or np.nan)
    c = z.cumsum()
    return CusumResult(stat=c)


def bde_cusum_recursive_residuals(y: pd.Series) -> CusumResult:
    """17.3.1 Brown–Durbin–Evans CUSUM on recursive residuals (proxy).

    Strict BDE requires recursive residuals from an OLS model. As a light proxy
    for a univariate series, we fit an expanding mean recursively and compute
    standardized one-step-ahead residuals.
    """
    y = y.dropna()
    if y.empty:
        return CusumResult(stat=y)

    resid = []
    mu = 0.0
    n = 0
    var = 0.0
    for v in y.to_numpy():
        # one-step-ahead prediction = mu
        r = v - (mu if n > 0 else v)
        resid.append(r)
        # update running mean/variance (Welford)
        n += 1
        delta = v - mu
        mu += delta / n
        var += delta * (v - mu)
    resid = pd.Series(resid, index=y.index)
    std = np.sqrt((var / max(n - 1, 1))) or np.nan
    w = resid / (std if std and np.isfinite(std) else resid.std(ddof=0))
    return CusumResult(stat=w.cumsum())


def _adf_t_stat(y: pd.Series, max_lags: int = 0) -> float:
    """Compute ADF t-statistic for rho in Δy_t = rho*y_{t-1} + Σ φ_i Δy_{t-i} + e_t.
    Minimal implementation with OLS via numpy. No constant/trend (Chow-type).
    """
    y = y.dropna()
    dy = y.diff().dropna()
    y_lag = y.shift(1).reindex(dy.index)

    X = [y_lag.to_numpy().reshape(-1, 1)]
    for i in range(1, max_lags + 1):
        X.append(dy.shift(i).to_numpy().reshape(-1, 1))
    X = np.hstack([x for x in X])
    Y = dy.to_numpy().reshape(-1, 1)

    # drop rows with NaN due to lags
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    Y = Y[mask]
    if X.size == 0:
        return np.nan

    # OLS
    XtX = X.T @ X
    try:
        beta = np.linalg.solve(XtX, X.T @ Y)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtX) @ (X.T @ Y)
    resid = Y - X @ beta
    s2 = float((resid.T @ resid) / max(len(Y) - X.shape[1], 1))
    var_beta = s2 * np.linalg.pinv(XtX)
    # t-stat for rho = first coefficient
    rho = float(beta[0])
    se_rho = float(np.sqrt(var_beta[0, 0])) if var_beta[0, 0] > 0 else np.nan
    return rho / se_rho if se_rho and np.isfinite(se_rho) else np.nan


def chow_type_df(y: pd.Series, max_lags: int = 0) -> float:
    """17.4.1 Chow-type Dickey–Fuller test: ADF without constant/trend.
    Returns t-statistic (left tail)."""
    return _adf_t_stat(y, max_lags=max_lags)


def sadf(y: pd.Series, r0: float = 0.2, max_lags: int = 0) -> Tuple[float, pd.Series]:
    """17.4.2 Supremum ADF (SADF) per PSY (2011, 2015) simplified.

    Args:
        y: series
        r0: minimal window fraction
        max_lags: ADF lag order
    Returns:
        (sup_t, t_series) where t_series indexed by end points contains sup over
        expanding starting points.
    """
    y = y.dropna()
    n = len(y)
    w0 = max(int(np.floor(r0 * n)), 10)
    t_vals = []
    idx = []
    for end in range(w0, n + 1):
        sup_t = -np.inf
        for start in range(0, end - w0 + 1):
            t_stat = _adf_t_stat(y.iloc[start:end], max_lags=max_lags)
            if np.isfinite(t_stat):
                sup_t = max(sup_t, t_stat)
        t_vals.append(sup_t)
        idx.append(y.index[end - 1])
    tser = pd.Series(t_vals, index=idx)
    return float(np.nanmax(tser.values)), tser


def sub_super_martingale_tests(ret: pd.Series, windows: Sequence[int] = (288,)) -> pd.DataFrame:
    """17.4.3 Sub-/Super-martingale tests (one-sided cumulative deviation Z).

    Returns a DataFrame with columns:
      sub_martingale_z_w{w}: tests E[ret]<=0 (negative drift)
      super_martingale_z_w{w}: tests E[ret]>=0 (positive drift)
    """
    out = pd.DataFrame(index=ret.index)
    for w in windows:
        mu = ret.rolling(w, min_periods=max(8, w // 4)).mean()
        sig = ret.rolling(w, min_periods=max(8, w // 4)).std(ddof=0)
        cum = (ret - mu).rolling(w, min_periods=max(8, w // 4)).sum()
        z = cum / (sig * np.sqrt(w))
        out[f'sub_martingale_z_w{w}'] = -z  # H0: super-martingale (drift >= 0)
        out[f'super_martingale_z_w{w}'] = z  # H0: sub-martingale (drift <= 0)
    return out

