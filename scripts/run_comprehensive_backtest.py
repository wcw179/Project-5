"""
This script runs a comprehensive backtest on the final ensemble model.
"""
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- Logger Configuration ---
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>")

# --- Constants ---
MODEL_DIR = project_root / "models"
REPORTS_DIR = project_root / "reports"
DATA_DIR = project_root / "data" / "processed"

def main():
    """Main function to run the backtest."""
    logger.info("--- Starting Comprehensive Backtest ---")

    # 1. Load Model and Data
    logger.info("Loading model and dataset...")
    model_path = MODEL_DIR / "enhanced_final_ensemble_model.joblib"
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}.")
        return
    model = joblib.load(model_path)

    X_path = DATA_DIR / "X_full.parquet"
    if not X_path.exists():
        logger.error(f"Dataset file not found at {X_path}.")
        return
    X = pd.read_parquet(X_path)
    
    # We need the original prices to calculate returns, which are not in X.
    # We will load the original database for this.
    db_path = project_root / "data" / "m5_trading.db"
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{db_path}")
    
    # NOTE: This backtest uses EURUSDm as a proxy for market returns for all signals.
    # A more robust backtest would require symbol information to be carried through the pipeline.
    logger.warning("Using 'EURUSDm' as a proxy for market returns for all signals.")
    prices = pd.read_sql("SELECT time, close FROM bars WHERE symbol = 'EURUSDm'", engine, index_col='time', parse_dates=['time'])

    # Ensure timezone consistency (Parquet is tz-aware, raw DB might not be)
    if prices.index.tz is None:
        prices = prices.tz_localize('UTC')

    # Normalize both indexes to UTC to prevent any mismatch
    X.index = X.index.tz_convert('UTC')
    prices.index = prices.index.tz_convert('UTC')
    
    logger.success(f"Loaded model and {len(X)} data points.")

    # 2. Generate Predictions
    logger.info("Generating predictions for the full dataset...")
    pred_mapped = model.predict(X)
    predictions = pd.Series(pred_mapped, index=X.index).map({0: -1, 1: 0, 2: 1})
    logger.success("Predictions generated.")

    # 3. Align Data and Calculate Returns
    logger.info("Aligning data and calculating returns...")
    price_returns = prices['close'].pct_change()
    df_backtest = pd.DataFrame({'signal': predictions}).join(price_returns.rename('market_return'), how='inner')
    df_backtest['position'] = df_backtest['signal'].shift(1)
    df_backtest['strategy_return'] = df_backtest['position'] * df_backtest['market_return']
    df_backtest.dropna(inplace=True)
    logger.success("Strategy returns calculated.")

    # 4. Calculate and Report Key Performance Metrics
    logger.info("Calculating performance metrics...")
    df_backtest['cumulative_strategy_return'] = (1 + df_backtest['strategy_return']).cumprod()
    df_backtest['cumulative_market_return'] = (1 + df_backtest['market_return']).cumprod()

    periods_per_year = 252 * 24 * 12 # 5-minute bars
    sharpe_ratio = (df_backtest['strategy_return'].mean() / df_backtest['strategy_return'].std()) * np.sqrt(periods_per_year)

    equity_curve = df_backtest['cumulative_strategy_return'] + 1
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve / peak) - 1
    max_drawdown = drawdown.min()

    logger.info("--- Backtest Performance Results ---")
    logger.info(f"Total Return (Strategy): {df_backtest['cumulative_strategy_return'].iloc[-1]:.2%}")
    logger.info(f"Total Return (Market): {df_backtest['cumulative_market_return'].iloc[-1]:.2%}")
    logger.info(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")

    # 5. Plot Equity Curve
    logger.info("Generating equity curve plot...")
    REPORTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(12, 8))
    (df_backtest['cumulative_strategy_return'] + 1).plot(label='Strategy')
    (df_backtest['cumulative_market_return'] + 1).plot(label='Market (Buy and Hold)')
    plt.title('Comprehensive Backtest - Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plot_path = REPORTS_DIR / "comprehensive_backtest_equity_curve.png"
    plt.savefig(plot_path)
    plt.close()
    logger.success(f"Equity curve plot saved to {plot_path}")

if __name__ == "__main__":
    main()
