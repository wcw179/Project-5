"""Core event-driven backtesting engine."""

import pandas as pd

from src.logger import logger


class Backtester:
    """A simple, event-driven backtesting engine."""

    def __init__(self, data: pd.DataFrame, strategy, initial_cash: float = 100000.0):
        self.data = data
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0.0
        self.equity_curve: list[float] = []

    def run(self):
        """Runs the backtest simulation."""
        logger.info("--- Starting Backtest --- ")
        for i, bar in self.data.iterrows():
            # 1. Update portfolio value (Mark-to-Market)
            equity = self.cash + self.position * bar["close"]
            self.equity_curve.append(equity)

            # 2. Generate signal from strategy
            signal = self.strategy.generate_signal(bar)

            # 3. Execute trades based on signal
            if signal == "buy" and self.position == 0:
                self.buy(bar)
            elif signal == "sell" and self.position > 0:
                self.sell(bar)

        logger.info("--- Backtest Finished --- ")
        return pd.Series(self.equity_curve, index=self.data.index)

    def buy(self, bar):
        """Executes a buy order."""
        # Simple example: invest all cash
        shares_to_buy = self.cash / bar["close"]
        self.position += shares_to_buy
        self.cash -= shares_to_buy * bar["close"]
        logger.debug(
            f"{bar.name}: BOUGHT {shares_to_buy:.2f} shares at {bar['close']:.2f}"
        )

    def sell(self, bar):
        """Executes a sell order."""
        logger.debug(
            f"{bar.name}: SOLD {self.position:.2f} shares at {bar['close']:.2f}"
        )
        self.cash += self.position * bar["close"]
        self.position = 0
