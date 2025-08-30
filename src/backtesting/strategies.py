"""A collection of simple trading strategies for backtesting."""

import pandas as pd


class MovingAverageCrossover:
    """A simple moving average crossover strategy."""

    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window
        self.short_mavg_prev = None
        self.long_mavg_prev = None

    def generate_signal(self, bar: pd.Series) -> str | None:
        """Generates a signal based on the current bar."""
        signal = None
        short_mavg = bar[f"ema_{self.short_window}"]
        long_mavg = bar[f"ema_{self.long_window}"]

        if self.short_mavg_prev is not None and self.long_mavg_prev is not None:
            # Crossover condition
            if short_mavg > long_mavg and self.short_mavg_prev <= self.long_mavg_prev:
                signal = "buy"
            # Crossunder condition
            elif short_mavg < long_mavg and self.short_mavg_prev >= self.long_mavg_prev:
                signal = "sell"

        self.short_mavg_prev = short_mavg
        self.long_mavg_prev = long_mavg

        return signal
