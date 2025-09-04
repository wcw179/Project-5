"""
Module for hyperparameter optimization and financial objective functions.

This module contains a more realistic backtesting engine and an objective
function aligned with the project's financial goals for use with Optuna.
"""

import numpy as np
import optuna
import pandas as pd
import pandas_ta as ta

from src.backtesting.metrics import get_performance_summary


def run_atr_based_backtest(
    predictions: pd.DataFrame,
    trade_data: pd.DataFrame,
    long_threshold: float,
    short_threshold: float,
    atr_multiplier_sl: float,
    atr_multiplier_tp: float,
    position_size_pct: float,
    atr_period: int = 100,
    initial_cash: float = 100000.0,
) -> tuple[pd.Series, list[dict]]:
    """
    Runs a more realistic backtest using ATR-based stops and targets.
    """
    long_signals = predictions["long_label"] > long_threshold
    short_signals = predictions["short_label"] > short_threshold

    atr = ta.atr(
        trade_data["high"], trade_data["low"], trade_data["close"], length=atr_period
    )

    trades = []
    cash = initial_cash
    position = 0
    entry_price = 0.0
    stop_loss_price = 0.0
    take_profit_price = 0.0

    equity = [initial_cash] * len(trade_data)

    for i in range(1, len(trade_data)):
        current_price = trade_data["close"].iloc[i]

        # Update equity with unrealized PnL
        if position != 0:
            unrealized_pnl = (current_price - entry_price) * position
            equity[i] = cash + unrealized_pnl
        else:
            equity[i] = cash

        # Exit conditions
        if position > 0:  # Long position
            if current_price <= stop_loss_price or current_price >= take_profit_price:
                pnl = (current_price - entry_price) * position
                cash += pnl
                trades.append(
                    {"pnl": pnl, "return": pnl / (equity[i - 1] or initial_cash)}
                )
                position = 0
        elif position < 0:  # Short position
            if current_price >= stop_loss_price or current_price <= take_profit_price:
                pnl = (current_price - entry_price) * position
                cash += pnl
                trades.append(
                    {"pnl": pnl, "return": pnl / (equity[i - 1] or initial_cash)}
                )
                position = 0

        # Update equity with realized PnL
        if position == 0:
            equity[i] = cash

        # Entry conditions
        if position == 0:
            current_atr = atr.iloc[i]
            if pd.isna(current_atr):
                continue

            if long_signals.iloc[i]:
                position_size_units = (equity[i] * position_size_pct) / current_price
                position = position_size_units
                entry_price = current_price
                stop_loss_price = entry_price - (current_atr * atr_multiplier_sl)
                take_profit_price = entry_price + (current_atr * atr_multiplier_tp)
            elif short_signals.iloc[i]:
                position_size_units = (equity[i] * position_size_pct) / current_price
                position = -position_size_units
                entry_price = current_price
                stop_loss_price = entry_price + (current_atr * atr_multiplier_sl)
                take_profit_price = entry_price - (current_atr * atr_multiplier_tp)

    return pd.Series(equity, index=trade_data.index), trades


def financial_objective_function(
    trial: optuna.trial.Trial, y_pred_proba: pd.DataFrame, trade_data: pd.DataFrame
) -> float:
    """
    Calculates a weighted financial score based on the project's specific goals.
    """
    # Hyperparameters for the backtest
    long_threshold = trial.suggest_float("long_threshold", 0.31, 0.95)
    short_threshold = trial.suggest_float("short_threshold", 0.31, 0.95)
    atr_multiplier_sl = trial.suggest_float("atr_multiplier_sl", 1.0, 5.0)
    atr_multiplier_tp = trial.suggest_float("atr_multiplier_tp", 1.0, 10.0)
    position_size_pct = trial.suggest_float(
        "position_size_pct", 0.01, 0.1
    )  # Risk 1-10% of equity

    # Aggregate multi-label predictions
    # Support both legacy ('long_*'/'short_*') and AFML triple-barrier ('hit_*') labels
    long_cols = [c for c in y_pred_proba.columns if ("long" in c) or c.startswith("hit_")]
    short_cols = [c for c in y_pred_proba.columns if "short" in c]

    agg_predictions = pd.DataFrame(index=y_pred_proba.index)
    agg_predictions["long_label"] = (
        y_pred_proba[long_cols].max(axis=1) if len(long_cols) > 0 else 0
    )
    agg_predictions["short_label"] = (
        y_pred_proba[short_cols].max(axis=1) if len(short_cols) > 0 else 0
    )

    equity_curve, trades = run_atr_based_backtest(
        predictions=agg_predictions,
        trade_data=trade_data,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        atr_multiplier_sl=atr_multiplier_sl,
        atr_multiplier_tp=atr_multiplier_tp,
        position_size_pct=position_size_pct,
    )

    if len(trades) < 20:
        return -10.0

    metrics = get_performance_summary(equity_curve, trades, return_dict=True)
    if not metrics:
        return -10.0

    # --- Multi-Objective Score based on TASK.md ---
    sharpe = metrics.get("Sharpe Ratio", -1.0)
    avg_return = np.mean([t["return"] for t in trades]) if trades else 0.0
    pnl_stability = -np.std([t["return"] for t in trades]) if trades else -1.0
    max_dd = metrics.get("Max Drawdown", 1.0)
    hit_rate = metrics.get("Hit Rate", 0.0)

    # Clean up potential inf/-inf values
    sharpe = sharpe if np.isfinite(sharpe) else -1.0
    avg_return = avg_return if np.isfinite(avg_return) else 0.0
    pnl_stability = pnl_stability if np.isfinite(pnl_stability) else -1.0

    # Constraint: Max Drawdown < 8%
    if max_dd > 0.08:
        return -10.0 - (max_dd * 100)  # Heavy penalty for violating constraint

    # Weights from TASK.md
    weights = {
        "sharpe": 0.35,
        "avg_return": 0.25,
        "stability": 0.20,
        "max_dd": 0.15,
        "hit_rate": 0.05,
    }

    score = (
        sharpe * weights["sharpe"]
        + avg_return * 100 * weights["avg_return"]  # Scale avg return
        + pnl_stability * weights["stability"]
        + (1 - max_dd) * weights["max_dd"]
        + hit_rate * weights["hit_rate"]
    )

    return score if np.isfinite(score) else -10.0
