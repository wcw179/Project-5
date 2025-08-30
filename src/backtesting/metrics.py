"""Functions for calculating backtest performance metrics."""

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


def get_performance_summary(equity_curve: pd.Series):
    """Prints a summary of key performance metrics."""
    print("--- Backtest Performance Summary ---")
    print(f"Sharpe Ratio: {calculate_sharpe_ratio(equity_curve):.2f}")
    print(f"Sortino Ratio: {calculate_sortino_ratio(equity_curve):.2f}")
    print(f"Max Drawdown: {calculate_max_drawdown(equity_curve):.2%}")
    print(f"Calmar Ratio: {calculate_calmar_ratio(equity_curve):.2f}")
    print(f"Total Return: {(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1):.2%}")
