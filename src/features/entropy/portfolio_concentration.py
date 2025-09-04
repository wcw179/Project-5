"""
AFML §18.8.3 Portfolio Concentration

Computes concentration metrics from portfolio weights at each timestamp.
Inputs are expected as a DataFrame `W` with index=DatetimeIndex and columns=assets,
values are nonnegative weights that (ideally) sum to 1 per row.

Metrics (per timestamp):
- HHI: Herfindahl–Hirschman Index = sum(w_i^2)
- shannon_entropy_w: Shannon entropy of weights (nats)
- effective_n: exp(shannon_entropy_w) (effective number of holdings)
- normalized_concentration: (HHI - 1/n) / (1 - 1/n) when n is number of nonzero assets

If rows contain NaNs or do not sum to 1, we normalize by row sum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def _normalize_rows(W: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    W = W.copy()
    W = W.fillna(0.0)
    rs = W.sum(axis=1).replace(0.0, np.nan)
    W = W.div(rs, axis=0)
    return W.fillna(0.0)


def portfolio_concentration_metrics(W: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    if W is None or not isinstance(W, pd.DataFrame) or W.empty:
        raise ValueError("portfolio weights DataFrame is required and must be non-empty")
    Wn = _normalize_rows(W, eps=eps)

    # HHI
    hhi = (Wn ** 2).sum(axis=1)

    # Shannon entropy of weights (nats)
    Wc = Wn.clip(lower=eps)
    sh_entropy = -(Wc * np.log(Wc)).sum(axis=1)

    # Effective number of holdings
    eff_n = np.exp(sh_entropy)

    # Normalized concentration relative to equal-weight over nonzero assets
    nonzero_n = (Wn > eps).sum(axis=1).clip(lower=1)
    norm_conc = (hhi - 1.0 / nonzero_n) / (1.0 - 1.0 / nonzero_n)

    out = pd.DataFrame(
        {
            "hhi": hhi,
            "shannon_entropy_w": sh_entropy,
            "effective_n": eff_n,
            "normalized_concentration": norm_conc,
        },
        index=W.index,
    )
    return out


def add_portfolio_concentration_features(
    df: pd.DataFrame,
    weights: Optional[pd.DataFrame],
) -> pd.DataFrame:
    out = df.copy()
    if weights is None or not isinstance(weights, pd.DataFrame) or weights.empty:
        # No weights provided; fill NaN columns for traceability
        out["hhi"] = np.nan
        out["shannon_entropy_w"] = np.nan
        out["effective_n"] = np.nan
        out["normalized_concentration"] = np.nan
        return out

    # Align weights to df index
    W = weights.copy()
    W.index = pd.to_datetime(W.index)
    W = W.reindex(out.index).ffill()

    conc = portfolio_concentration_metrics(W)
    out = out.join(conc, how="left")
    return out

