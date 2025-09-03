"""Core event-driven backtesting engine."""

import pandas as pd

from src.logger import logger


class Backtester:
    """A simple, event-driven backtesting engine for Forex."""

    def __init__(
        self,
        data: pd.DataFrame,
        strategy,
        initial_cash: float = 100000.0,
        position_size_units: float = 10000.0,
    ):
        self.data = data
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0.0  # Number of units of base currency
        self.position_size_units = position_size_units
        self.entry_price = 0.0
        self.equity_curve: list[float] = []
        self.trades: list[dict] = []

    def run(self):
        """Runs the backtest simulation."""
        logger.info("--- Starting Backtest --- ")
        for i, (timestamp, bar) in enumerate(self.data.iterrows()):
            # 1. Update portfolio value (Mark-to-Market)
            unrealized_pnl = (bar["close"] - self.entry_price) * self.position
            equity = self.cash + unrealized_pnl
            self.equity_curve.append(equity)

            # 2. Generate signal from strategy
            signal = self.strategy.generate_signal(bar, i)

            # 3. Execute trades based on signal
            if self.position == 0:  # Can only enter a new position if flat
                if signal == "buy":
                    self.buy(bar)
                elif signal == "sell_short":
                    self.sell_short(bar)
            elif (
                self.position > 0 and signal == "sell"
            ):  # Can only sell a long position
                self.sell(bar)
            elif (
                self.position < 0 and signal == "buy_to_cover"
            ):  # Can only cover a short position
                self.buy_to_cover(bar)

        logger.info("--- Backtest Finished --- ")
        return pd.Series(self.equity_curve, index=self.data.index), self.trades

    def buy(self, bar):
        """Executes a buy order."""
        self.position = self.position_size_units
        self.entry_price = bar["close"]
        # In a real forex system, margin would be deducted here. We simplify and just track PnL in cash.
        logger.debug(
            f"{bar.name}: BOUGHT {self.position:.2f} units at {bar['close']:.2f}"
        )

    def sell(self, bar):
        """Closes a long position."""
        pnl = (bar["close"] - self.entry_price) * self.position
        self.trades.append(
            {"entry_time": None, "exit_time": bar.name, "pnl": pnl, "type": "long"}
        )
        self.cash += pnl
        logger.debug(
            f"{bar.name}: SOLD {self.position:.2f} units at {bar['close']:.2f} for a PnL of {pnl:.2f}"
        )
        self.position = 0
        self.entry_price = 0

    def sell_short(self, bar):
        """Executes a short sell order."""
        self.position = -self.position_size_units
        self.entry_price = bar["close"]
        logger.debug(
            f"{bar.name}: SOLD SHORT {abs(self.position):.2f} units at {bar['close']:.2f}"
        )

    def buy_to_cover(self, bar):
        """Closes a short position."""
        pnl = (self.entry_price - bar["close"]) * abs(self.position)
        self.trades.append(
            {"entry_time": None, "exit_time": bar.name, "pnl": pnl, "type": "short"}
        )
        self.cash += pnl
        logger.debug(
            f"{bar.name}: BOUGHT TO COVER {abs(self.position):.2f} units at {bar['close']:.2f} for a PnL of {pnl:.2f}"
        )
        self.position = 0
        self.entry_price = 0
