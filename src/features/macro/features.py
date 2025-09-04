"""
Macro & Cross-Asset features (V4 §2.6)

Design
- This module accepts the local symbol OHLCV dataframe and (optionally) a
  `context` dict with external macro/cross-asset series aligned on the same
  bar clock (index: pandas.DatetimeIndex at 5-minute UTC).
- If external series are missing, the functions will populate NaNs for the
  corresponding macro features and log an info message to guide integration.

Expected context keys (optional)
- 'dxy': DataFrame with column 'close' (US Dollar Index)
- 'yield_10y', 'yield_2y': DataFrames with column 'close' (yields)
- 'credit_spread': DataFrame with 'close' (e.g., HY-OAS)
- 'equity', 'bond', 'commodity', 'crypto': DataFrames with 'close' for proxies

Features (examples)
- yield_curve_slope: 10Y - 2Y (synced to bar clock)
- dxy_momentum: rolling z-score of DXY returns
- risk_on_off: sign of correlation equity vs bond (rolling window)
- equity_correlation / bond_correlation / commodity_correlation / crypto_correlation
"""
from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import pandas as pd

from src.logger import logger


def _safe_align(df: pd.DataFrame, ctx: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if ctx is None:
        return None
    if not isinstance(ctx, pd.DataFrame) or 'close' not in ctx.columns:
        return None
    s = ctx['close'].copy()
    s.index = pd.to_datetime(s.index)
    # Align to df index (forward-fill intraday if needed)
    return s.reindex(df.index).ffill()


def add_macro_features(
    df: pd.DataFrame,
    context: Optional[Dict[str, pd.DataFrame]] = None,
    *,
    corr_window: int = 288,
    momentum_window: int = 288,
) -> pd.DataFrame:
    """
    Add macro and cross-asset features. If `context` is missing, columns will be NaN.

    Args:
        df: OHLCV dataframe with index=DatetimeIndex (UTC) and columns [open, high, low, close, volume]
        context: optional dict of external series as described in module docstring
        corr_window: rolling window for cross-asset correlations (bars)
        momentum_window: rolling window for momentum z-scores (bars)
    """
    out = df.copy()

    ctx = context or {}
    dxy = _safe_align(out, ctx.get('dxy'))
    y10 = _safe_align(out, ctx.get('yield_10y'))
    y2 = _safe_align(out, ctx.get('yield_2y'))
    credit = _safe_align(out, ctx.get('credit_spread'))
    eq = _safe_align(out, ctx.get('equity'))
    bd = _safe_align(out, ctx.get('bond'))
    cmd = _safe_align(out, ctx.get('commodity'))
    cpt = _safe_align(out, ctx.get('crypto'))

    # yield curve slope
    if y10 is not None and y2 is not None:
        out['yield_curve_slope'] = y10 - y2
    else:
        out['yield_curve_slope'] = np.nan

    # credit spreads
    out['credit_spreads'] = credit if credit is not None else np.nan

    # DXY momentum as rolling z-score of returns
    if dxy is not None:
        dxy_ret = dxy.pct_change()
        mu = dxy_ret.rolling(momentum_window, min_periods=momentum_window // 4).mean()
        sig = dxy_ret.rolling(momentum_window, min_periods=momentum_window // 4).std()
        out['dxy_momentum'] = (dxy_ret - mu) / (sig.replace(0, np.nan))
    else:
        out['dxy_momentum'] = np.nan

    # risk_on_off: correlation between equity and bond returns (negative => risk-off)
    def _roll_corr(a: Optional[pd.Series], b: Optional[pd.Series]) -> pd.Series:
        if a is None or b is None:
            return pd.Series(np.nan, index=out.index)
        ar = a.pct_change()
        br = b.pct_change()
        return ar.rolling(corr_window, min_periods=corr_window // 4).corr(br)

    out['equity_bond_corr'] = _roll_corr(eq, bd)
    out['risk_on_off'] = out['equity_bond_corr']

    # cross-asset correlations with the local symbol (close returns)
    loc_ret = out['close'].pct_change()

    def _asset_corr(asset: Optional[pd.Series]) -> pd.Series:
        if asset is None:
            return pd.Series(np.nan, index=out.index)
        r = asset.pct_change()
        return loc_ret.rolling(corr_window, min_periods=corr_window // 4).corr(r)

    out['equity_correlation'] = _asset_corr(eq)
    out['bond_correlation'] = _asset_corr(bd)
    out['commodity_correlation'] = _asset_corr(cmd)
    out['crypto_correlation'] = _asset_corr(cpt)

    # Market stress proxies (if not provided, leave NaN to be backfilled by data layer)
    out['systemic_risk'] = np.nan
    out['liquidity_stress'] = np.nan
    out['correlation_breakdown'] = np.nan
    out['contagion_risk'] = np.nan

    if not context:
        logger.info("Macro context not provided; macro features filled with NaNs where applicable.")
    return out

