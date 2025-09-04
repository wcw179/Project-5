"""
Meta-labeling utilities (V4, Phase 1A)

- extract_meta_features(X, primary_proba): build meta-feature matrix from primary
  predictions and market/context features.
- generate_meta_labels(primary_proba, context): optional rule-based labels for
  bootstrapping (trade_quality, position_size_multiplier, entry_timing, risk_level).

Notes:
- In production, meta labels should come from realized performance and
  historical meta-analytics; here we provide a practical placeholder.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


def _agreement_features(proba: pd.DataFrame) -> pd.DataFrame:
    """Agreement metrics across label columns: mean/max/var/skew/kurt."""
    if proba.empty:
        return pd.DataFrame(index=proba.index)
    stats = pd.DataFrame(index=proba.index)
    stats["p_mean"] = proba.mean(axis=1)
    stats["p_max"] = proba.max(axis=1)
    stats["p_min"] = proba.min(axis=1)
    stats["p_var"] = proba.var(axis=1)
    stats["p_std"] = proba.std(axis=1)
    return stats


def _confidence_features(proba: pd.DataFrame) -> pd.DataFrame:
    """Per-sample confidence proxies: top-k spread, entropy of probs normalized."""
    if proba.empty:
        return pd.DataFrame(index=proba.index)
    arr = proba.to_numpy()
    sorted_p = np.sort(arr, axis=1)[:, ::-1]
    top1 = sorted_p[:, 0]
    top2 = sorted_p[:, 1] if proba.shape[1] > 1 else np.zeros_like(top1)
    spread = top1 - top2
    # entropy per row normalized by log(#labels)
    row_sum = arr.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    pnorm = arr / row_sum
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(pnorm * np.log(pnorm + 1e-12)).sum(axis=1)
    ent_norm = ent / np.log(proba.shape[1] + 1e-9)
    out = pd.DataFrame({"top1": top1, "top2": top2, "p_spread": spread, "p_entropy": ent_norm}, index=proba.index)
    return out


def extract_meta_features(
    X: pd.DataFrame,
    primary_proba: pd.DataFrame,
    market_cols: Optional[list[str]] = None,
    hist_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Construct meta-feature matrix from primary outputs and selected X columns.

    Args:
        X: original feature matrix
        primary_proba: DataFrame of primary label probabilities (columns=labels)
        market_cols: subset of X columns to include (e.g., regime, volatility)
        hist_cols: historical performance/aggregation features (if available)
    """
    feats = []
    feats.append(_agreement_features(primary_proba))
    feats.append(_confidence_features(primary_proba))

    if market_cols:
        feats.append(X[market_cols].copy())
    if hist_cols:
        feats.append(X[hist_cols].copy())

    meta_X = pd.concat(feats, axis=1).fillna(0.0)
    return meta_X


def _rule_trade_quality(top1: pd.Series, spread: pd.Series) -> pd.Series:
    q = pd.Series(0, index=top1.index, dtype=int)
    q[(top1 >= 0.80) & (spread >= 0.25)] = 3
    q[(top1 >= 0.65) & (spread >= 0.15) & (q == 0)] = 2
    q[(top1 >= 0.55) & (q == 0)] = 1
    return q


def _rule_position_size_multiplier(q: pd.Series) -> pd.Series:
    # Map quality to baseline multipliers
    mapping = {0: 0.0, 1: 1.0, 2: 1.5, 3: 2.5}
    return q.map(mapping).astype(float)


def _rule_entry_timing(spread: pd.Series) -> pd.Series:
    # 2=immediate, 1=wait_pullback, 0=skip
    t = pd.Series(1, index=spread.index, dtype=int)
    t[spread >= 0.25] = 2
    t[spread < 0.10] = 0
    return t


def _rule_risk_level(vol_regime: Optional[pd.Series]) -> pd.Series:
    if vol_regime is None:
        return pd.Series(1, index=vol_regime.index if vol_regime is not None else None, dtype=int)
    # assume vol_regime in {0,1,2,3}
    r = vol_regime.clip(lower=0, upper=2).astype(int)
    return r


def generate_meta_labels(
    X: pd.DataFrame,
    primary_proba: pd.DataFrame,
    market_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Rule-based meta-labels for bootstrapping training of meta-models.

    Returns columns: trade_quality (0..3), position_size_multiplier (float),
    entry_timing (0..2), risk_level (0..2)
    """
    feats = _confidence_features(primary_proba)
    q = _rule_trade_quality(feats["top1"], feats["p_spread"])
    m = _rule_position_size_multiplier(q)
    t = _rule_entry_timing(feats["p_spread"])
    vol_reg = X[market_cols[0]] if market_cols else None
    r = _rule_risk_level(vol_reg)
    out = pd.DataFrame(
        {
            "trade_quality": q,
            "position_size_multiplier": m,
            "entry_timing": t,
            "risk_level": r,
        },
        index=primary_proba.index,
    )
    return out

