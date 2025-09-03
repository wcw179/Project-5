"""Functions for calculating backtest performance metrics."""

from typing import Optional

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    equity_curve: pd.Series, risk_free_rate: float = 0.0
) -> float:
    """Calculates the Sharpe Ratio."""
    returns = equity_curve.pct_change().dropna()
    excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def calculate_sortino_ratio(
    equity_curve: pd.Series, risk_free_rate: float = 0.0
) -> float:
    """Calculates the Sortino Ratio."""
    returns = equity_curve.pct_change().dropna()
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    if downside_std == 0:
        return np.inf
    return np.sqrt(252) * excess_returns.mean() / downside_std


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Calculates the Maximum Drawdown."""
    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    return drawdown.min()


def calculate_calmar_ratio(equity_curve: pd.Series) -> float:
    """Calculates the Calmar Ratio."""
    returns = equity_curve.pct_change().dropna()
    annual_return = returns.mean() * 252
    max_drawdown = calculate_max_drawdown(equity_curve)
    if max_drawdown == 0:
        return np.inf
    return annual_return / abs(max_drawdown)


def calculate_profit_factor(trades: list[dict]) -> float:
    """Calculates the Profit Factor."""
    if not trades:
        return 0.0

    pnl = np.array([trade["pnl"] for trade in trades])
    gross_profit = np.sum(pnl[pnl > 0])
    gross_loss = np.abs(np.sum(pnl[pnl < 0]))

    if gross_loss == 0:
        return np.inf  # Undefined or infinite profit factor

    return gross_profit / gross_loss


def get_performance_summary(
    equity_curve: pd.Series,
    trades: Optional[list[dict]] = None,
    return_dict: bool = False,
) -> Optional[dict[str, float]]:
    """
    Calculates and displays a summary of key performance metrics.

    Args:
        equity_curve (pd.Series): The equity curve of the backtest.
        trades (list[dict]): A list of trades, each a dict with 'pnl'.
        return_dict (bool): If True, returns a dictionary of the metrics.

    Returns:
        A dictionary of performance metrics if return_dict is True, otherwise None.
    """
    sharpe = calculate_sharpe_ratio(equity_curve)
    max_dd = calculate_max_drawdown(equity_curve)
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

    metrics = {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Total Trades": 0,
        "Hit Rate": 0,
        "Avg Profit/Trade": 0,
        "Profit Factor": 0,
    }

    if trades is not None and len(trades) > 0:
        trade_pnl = [t["pnl"] for t in trades]
        metrics["Total Trades"] = len(trades)
        metrics["Hit Rate"] = (
            (np.sum(np.array(trade_pnl) > 0) / len(trades)) if len(trades) > 0 else 0
        )
        metrics["Avg Profit/Trade"] = np.mean(trade_pnl)
        metrics["Profit Factor"] = calculate_profit_factor(trades)

    if return_dict:
        return metrics

    print("--- Backtest Performance Summary ---")
    print(f"Total Return: {metrics['Total Return']:.2%}")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {metrics['Max Drawdown']:.2%}")
    print(f"Total Trades: {metrics['Total Trades']}")
    print(f"Hit Rate: {metrics['Hit Rate']:.2%}")
    print(f"Avg Profit/Trade: {metrics['Avg Profit/Trade']:.2f}")
    print(f"Profit Factor: {metrics['Profit Factor']:.2f}")
    return None
