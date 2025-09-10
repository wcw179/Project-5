"""Script to backtest a trained model on a specific period and report PnL."""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from sqlalchemy import create_engine

from src.backtesting.metrics import calculate_max_drawdown
from src.features.pipeline import create_all_features
from src.labeling.dynamic_labels import create_dynamic_labels
from src.logger import logger

# --- Configuration ---
DB_PATH = project_root / "data" / "m5_trading.db"
MODEL_PATH = "C:/Users/wcw17/Documents/GitHub/project-5/Project-5/data/models/EURUSDm_model_20250901_065505.joblib"
SYMBOL = "EURUSDm"
START_DATE = "2025-01-01"
END_DATE = "2025-08-15"


def load_and_prepare_data():
    """Loads, features, and labels data for the backtest period."""
    logger.info(f"Loading data from {START_DATE} to {END_DATE}...")
    engine = create_engine(f"sqlite:///{DB_PATH}")
    query = f"""
        SELECT time, open, high, low, close, volume
        FROM bars
        WHERE symbol = '{SYMBOL}' AND time BETWEEN '{START_DATE}' AND '{END_DATE}'
        ORDER BY time ASC
    """
    raw_data = pd.read_sql(query, engine, index_col="time", parse_dates=["time"])
    if raw_data.index.has_duplicates:
        raw_data = raw_data[~raw_data.index.duplicated(keep="last")]

    featured_data = create_all_features(raw_data, SYMBOL)
    labels, _ = create_dynamic_labels(raw_data)

    aligned_index = featured_data.index.intersection(labels.index)
    X = featured_data.loc[aligned_index]
    y = labels.loc[aligned_index].filter(like="hit_")
    return X, y


def run_backtest(model, X_test, y_test):
    """Runs the backtest simulation and returns performance metrics and equity curve."""
    # --- Configuration ---
    threshold = 0.214
    initial_cash = 100000.0
    hold_period = 12  # 1 hour

    # Generate predictions
    pred_probas_list = [est.predict_proba(X_test)[:, 1] for est in model.estimators_]
    pred_probas_df = pd.DataFrame(
        np.array(pred_probas_list).T, index=X_test.index, columns=y_test.columns
    )
    max_probs = pred_probas_df.max(axis=1).values.astype(float)
    close_prices = X_test["close"].values

    # --- Simulation Loop ---
    cash = initial_cash
    position_value = 0.0
    equity_curve = []
    pnl_per_trade = []
    entry_bar = -1
    entry_price = 0.0

    for i in range(len(X_test)):
        current_price = close_prices[i]
        current_equity = cash + position_value
        if position_value > 0 and entry_price > 0:
            current_equity = cash + (position_value / entry_price) * current_price
        equity_curve.append(current_equity)

        if entry_bar != -1 and (i - entry_bar) >= hold_period:
            exit_value = (position_value / entry_price) * current_price
            pnl = exit_value - position_value
            pnl_per_trade.append(pnl)
            cash += exit_value
            position_value = 0
            entry_bar = -1

        if position_value == 0 and max_probs[i] > threshold:
            position_value = cash
            entry_price = current_price
            entry_bar = i
            cash = 0

    # --- Performance Metrics ---
    equity_series = pd.Series(equity_curve, index=X_test.index)
    total_pnl = sum(pnl_per_trade)
    num_trades = len(pnl_per_trade)
    wins = [p for p in pnl_per_trade if p > 0]
    win_rate = len(wins) / num_trades if num_trades > 0 else 0
    avg_profit = np.mean(pnl_per_trade) if num_trades > 0 else 0
    max_dd = calculate_max_drawdown(equity_series)
    total_signals = (max_probs > threshold).sum()

    report = {
        "Total PnL": f"${total_pnl:,.2f}",
        "Total Trades": num_trades,
        "Total Signals Generated": total_signals,
        "Win Rate": f"{win_rate:.2%}",
        "Average PnL per Trade": f"${avg_profit:,.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
    }

    return report, pd.Series(equity_curve, index=X_test.index)


def main():
    """Main backtest function."""
    logger.info(f"--- Starting Backtest for {MODEL_PATH} on 2025 data ---")

    model = joblib.load(MODEL_PATH)
    X, y = load_and_prepare_data()

    report, equity_curve = run_backtest(model, X, y)

    logger.success("--- Backtest Report ---")
    for key, value in report.items():
        logger.info(f"{key}: {value}")
        print(f"{key}: {value}")

    # Plot Equity Curve
    plt.figure(figsize=(12.0, 7.0))
    equity_curve.plot(title="2025 Backtest Equity Curve", legend=False)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.tight_layout()

    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    save_path = reports_dir / "backtest_2025_equity_curve.png"
    plt.savefig(save_path)
    logger.success(f"Equity curve plot saved to {save_path}")


if __name__ == "__main__":
    main()
