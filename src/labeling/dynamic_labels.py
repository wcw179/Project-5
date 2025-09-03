"""Functions for creating dynamic, multi-horizon, multi-label targets."""

import numpy as np
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm

from src.logger import logger


def create_dynamic_labels(
    df: pd.DataFrame,
    horizons: list[int],
    pt_sl_ratios: list[float],
    atr_period: int = 100,
    atr_multiplier: float = 2.0,
    spread_cost_pips: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates bi-directional labels using a vectorized triple-barrier method."""
    logger.info("Starting vectorized bi-directional label generation...")

    pip_value = 0.0001
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=atr_period)
    df.dropna(inplace=True)

    labels = pd.DataFrame(index=df.index)
    stop_loss_atr = df["atr"] * atr_multiplier

    close_prices = df["close"].to_numpy()
    high_prices = df["high"].to_numpy()
    low_prices = df["low"].to_numpy()

    for ratio in tqdm(pt_sl_ratios, desc="Processing R/R Ratios"):
        profit_take_atr = stop_loss_atr * ratio

        for h in horizons:
            # Vectorized barrier calculation
            long_pt = close_prices + profit_take_atr + (spread_cost_pips * pip_value)
            long_sl = close_prices - stop_loss_atr
            short_pt = close_prices - profit_take_atr - (spread_cost_pips * pip_value)
            short_sl = close_prices + stop_loss_atr

            long_outcomes = np.zeros(len(df))
            short_outcomes = np.zeros(len(df))

            for i in range(len(df) - h):
                # Long trade
                for j in range(1, h + 1):
                    if high_prices[i + j] >= long_pt[i]:
                        long_outcomes[i] = 1
                        break
                    if low_prices[i + j] <= long_sl[i]:
                        break
                # Short trade
                for j in range(1, h + 1):
                    if low_prices[i + j] <= short_pt[i]:
                        short_outcomes[i] = 1
                        break
                    if high_prices[i + j] >= short_sl[i]:
                        break

            labels[f"long_hit_{h}_{int(ratio)}R"] = long_outcomes
            labels[f"short_hit_{h}_{int(ratio)}R"] = short_outcomes

    # --- Sample Info for Purging ---
    max_horizon = max(horizons)
    t1_series = df.index.to_series().shift(-max_horizon).fillna(method="ffill")
    sample_info = pd.DataFrame(t1_series, index=df.index, columns=["t1"])

    df.drop(columns=["atr"], inplace=True, errors="ignore")
    logger.success(f"Generated {len(labels.columns)} bi-directional labels.")
    return labels, sample_info
