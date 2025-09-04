"""
AFML-style Triple-Barrier Labeling (multi-RR, multi-horizon) for EURUSDm Phase 1A

Implements:
- get_events: defines vertical barrier (time horizon) per index; target volatility via rolling ATR or std
- get_bins: computes labels (hit PT/SL/none) and times t1 per sample
- generate_primary_labels: multi-RR and multi-horizon matrix of binary labels
- compute_sample_weights: sample weights to address overlapping outcomes (optional)

Notes:
- This is a practical simplified variant suited for M5 bars with fixed horizons (in bars)
- For Phase 1A, we focus on primary labels. Meta-labels are produced by a separate module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class TripleBarrierConfig:
    horizons: Sequence[int]  # in bars, e.g., [12, 48, 144, 288]
    rr_multiples: Sequence[float]  # e.g., [5.0, 10.0, 15.0, 20.0]
    vol_method: str = "atr"  # or "std"
    atr_period: int = 100
    vol_window: int = 288
    atr_mult_base: float = 2.0  # base stop distance in ATR units
    include_short: bool = False  # primary labels for long only by default (V4 targets usually long-side)
    spread_pips: float = 2.0


def _pip_value(symbol: str) -> float:
    # Simple mapping; adjust as needed per symbol
    return 0.0001 if symbol.endswith("USDm") else 0.01


def _target_vol(df: pd.DataFrame, cfg: TripleBarrierConfig) -> pd.Series:
    if cfg.vol_method == "std":
        return df["close"].pct_change().rolling(cfg.vol_window).std()
    # ATR default
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / cfg.atr_period, adjust=False).mean()
    return atr


def get_events(df: pd.DataFrame, cfg: TripleBarrierConfig) -> pd.DataFrame:
    """Compute base distances (PT/SL per index) and max horizon t1 for each bar.
    Returns a DataFrame with columns ['pt_dist', 'sl_dist', 't1'].
    """
    out = pd.DataFrame(index=df.index)
    vol = _target_vol(df, cfg)
    out["pt_dist"] = cfg.atr_mult_base * vol
    out["sl_dist"] = cfg.atr_mult_base * vol

    # For each index, t1 = index shifted by max horizon for convenience (we'll use per-horizon later)
    max_h = max(cfg.horizons)
    out["t1"] = df.index.to_series().shift(-max_h)
    return out


def _first_touch(pt: float, sl: float, hi: np.ndarray, lo: np.ndarray) -> int:
    """Return +1 if PT is hit first, -1 if SL first, 0 if none."""
    for k in range(len(hi)):
        if hi[k] >= pt:
            return 1
        if lo[k] <= sl:
            return -1
    return 0


def get_bins(
    df: pd.DataFrame,
    events: pd.DataFrame,
    horizon_bars: int,
    rr: float,
    *,
    spread: float,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute binary labels for a single horizon and R/R multiple for BOTH sides.
    Returns (y_long, y_short, t1_series) aligned with df.index.
    """
    n = len(df)
    y_long = np.zeros(n, dtype=np.int8)
    y_short = np.zeros(n, dtype=np.int8)
    t1 = df.index.to_numpy().copy()

    close = df["close"].to_numpy()
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()

    pt_dist = events["pt_dist"].to_numpy()
    sl_dist = events["sl_dist"].to_numpy()

    for i in range(n - horizon_bars):
        entry = close[i]
        # Long side PT/SL
        long_pt = entry + (pt_dist[i] * rr) + spread
        long_sl = entry - sl_dist[i]
        # Short side PT/SL (profit when price goes down)
        short_pt = entry - (pt_dist[i] * rr) - spread
        short_sl = entry + sl_dist[i]

        # slice next horizon_bars candles
        hi_slice = hi[i + 1 : i + 1 + horizon_bars]
        lo_slice = lo[i + 1 : i + 1 + horizon_bars]

        # Long outcome
        out_long = _first_touch(long_pt, long_sl, hi_slice, lo_slice)
        y_long[i] = 1 if out_long == 1 else 0

        # Short outcome (mirror)
        out_short = _first_touch(short_sl, short_pt, hi_slice, lo_slice)  # first touch of SL (up) vs PT (down)
        y_short[i] = 1 if out_short == -1 else 0

        t1[i] = df.index[i + horizon_bars]

    return (
        pd.Series(y_long, index=df.index, dtype=np.int8),
        pd.Series(y_short, index=df.index, dtype=np.int8),
        pd.Series(t1, index=df.index),
    )


def generate_primary_labels(
    df: pd.DataFrame,
    symbol: str,
    cfg: TripleBarrierConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a label matrix for multiple horizons and R/R multiples for BOTH sides.
    Returns (labels_df, sample_info) where sample_info['t1'] is the furthest horizon.
    Columns produced per (rr, h):
      - long_hit_{int(rr)}R_h{h}
      - short_hit_{int(rr)}R_h{h}
    """
    events = get_events(df, cfg)
    pip = _pip_value(symbol)
    spread = cfg.spread_pips * pip

    labels = {}
    t1_all = {}
    for rr in cfg.rr_multiples:
        for h in cfg.horizons:
            y_long, y_short, t1 = get_bins(df, events, horizon_bars=h, rr=rr, spread=spread)
            labels[f"long_hit_{int(rr)}R_h{h}"] = y_long
            labels[f"short_hit_{int(rr)}R_h{h}"] = y_short
            t1_all[(rr, h)] = t1

    labels_df = pd.DataFrame(labels, index=df.index)
    # Use max horizon t1 for purging
    max_h = max(cfg.horizons)
    sample_info = pd.DataFrame(index=df.index)
    sample_info["t1"] = df.index.to_series().shift(-max_h)
    return labels_df, sample_info


def compute_sample_weights(sample_info: pd.DataFrame) -> pd.Series:
    """Simple sample weights inversely proportional to overlap density.
    For Phase 1A, a placeholder approximation using uniform weights.
    """
    w = pd.Series(1.0, index=sample_info.index)
    return w

