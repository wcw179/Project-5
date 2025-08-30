"""Functions for creating dynamic, multi-horizon, multi-label targets."""

import pandas as pd
import pandas_ta as ta

from src.logger import logger


def create_dynamic_labels(
    df: pd.DataFrame,
    atr_length: int = 14,
    r_multiples: list[float] = [5.0, 10.0, 15.0, 20.0],
    horizons: list[int] = [4 * 12, 12 * 12, 24 * 12],  # 4h, 12h, 24h in M5 bars
    spread_cost: float = 0.0002,  # Example spread cost
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates multi-horizon, multi-label profit targets based on ATR multiples.

    Args:
        df: DataFrame with OHLCV data.
        atr_length: The period for ATR calculation.
        r_multiples: A list of R-multiples for profit targets (e.g., 5R, 10R).
        horizons: A list of look-forward periods (in bars) for each horizon.
        spread_cost: The estimated cost of spread and slippage.

    Returns:
        A tuple containing:
        - labels_df: DataFrame with multi-label targets.
        - sample_info_df: DataFrame with start and end times for each sample.
    """
    logger.info("Starting dynamic multi-label generation...")

    # 1. Calculate ATR for dynamic targets
    atr = ta.atr(df["high"], df["low"], df["close"], length=atr_length)

    labels = []
    sample_info = []

    # 2. Iterate through each bar to generate forward-looking labels
    for i in range(len(df) - max(horizons)):
        entry_time = df.index[i]
        entry_price = df["close"].iloc[i]
        atr_value = atr.iloc[i]

        if pd.isna(atr_value):
            continue

        stop_loss_price = entry_price - atr_value

        label_row = {"time": entry_time}
        sample_info_row = {"start_time": entry_time}

        max_end_time = entry_time

        # 3. Check each profit target
        for r_mult in r_multiples:
            profit_target_price = entry_price + r_mult * atr_value + spread_cost

            # 4. Check each horizon
            for h_idx, horizon in enumerate(horizons):
                window = df.iloc[i + 1 : i + 1 + horizon]

                # Check if profit target is hit
                hit_mask = window["high"] >= profit_target_price
                # Check if stop loss is hit
                sl_mask = window["low"] <= stop_loss_price

                # Combine masks to find first touch
                first_hit = hit_mask.idxmax() if hit_mask.any() else None
                first_sl = sl_mask.idxmax() if sl_mask.any() else None

                label_key = f"hit_{int(r_mult)}R_{h_idx+1}"
                label_row[label_key] = 0

                if first_hit and (not first_sl or first_hit <= first_sl):
                    label_row[label_key] = 1
                    if first_hit > max_end_time:
                        max_end_time = first_hit
                elif first_sl:
                    if first_sl > max_end_time:
                        max_end_time = first_sl

        sample_info_row["end_time"] = max_end_time
        labels.append(label_row)
        sample_info.append(sample_info_row)

    labels_df = pd.DataFrame(labels).set_index("time")
    sample_info_df = pd.DataFrame(sample_info).set_index(labels_df.index)

    logger.success(f"Generated {len(labels_df)} labels and sample info records.")
    return labels_df, sample_info_df
