"""
Implements a simple primary model for generating trading signals.
"""
import pandas as pd

def get_primary_model_signals(
    close: pd.Series,
    fast_window: int = 12,  # 1 hour on 5-min data
    slow_window: int = 72,  # 6 hours on 5-min data
) -> pd.Series:
    """
    Generates trading signals based on a simple moving average (SMA) crossover.

    Args:
        close: Series of close prices.
        fast_window: The lookback window for the fast SMA.
        slow_window: The lookback window for the slow SMA.

    Returns:
        A Series with side predictions (1 for long, -1 for short).
    """
    # Calculate fast and slow moving averages
    fast_sma = close.rolling(window=fast_window).mean()
    slow_sma = close.rolling(window=slow_window).mean()

    # Generate signals
    signals = pd.Series(0, index=close.index)
    signals[fast_sma > slow_sma] = 1
    signals[fast_sma < slow_sma] = -1

    # Forward-fill signals to represent a held position, then remove leading zeros
    signals = signals.replace(0, pd.NA).ffill().dropna().astype(int)
    
    return signals

