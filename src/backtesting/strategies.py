"""A collection of simple trading strategies for backtesting."""

from typing import Optional

import pandas as pd


class MovingAverageCrossover:
    """A simple moving average crossover strategy."""

    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window
        self.short_mavg_prev = None
        self.long_mavg_prev = None

    def generate_signal(self, bar: pd.Series, current_bar_index: int) -> str | None:
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


class XGBoostStrategy:
    """A strategy that uses a trained XGBoost model to generate signals."""

    def __init__(self, model, threshold: float = 0.5, hold_period: int = 12):
        self.model = model
        self.threshold = threshold
        self.hold_period = hold_period
        self.entry_bar = -1

    def generate_signal(self, bar: pd.Series, current_bar_index: int) -> str | None:
        """Generates a signal based on the model's prediction."""
        # Exit condition
        if (
            self.entry_bar != -1
            and (current_bar_index - self.entry_bar) >= self.hold_period
        ):
            self.entry_bar = -1
            return "sell"

        # Entry condition
        if self.entry_bar == -1:  # Only consider entry if not already in a position
            # Reshape the feature row to be 2D for the model's predict_proba method
            feature_names = self.model.estimators_[0].feature_names_in_
            features = bar[feature_names].values.reshape(1, -1)

            # Predict probabilities for all 12 targets
            # The result is a list of 12 arrays, one for each target classifier
            # Each array has shape (n_samples, n_classes), so we want the prob of class 1
            probabilities = [p[:, 1][0] for p in self.model.predict_proba(features)]

            # If any target's probability is above the threshold, generate a buy signal
            if any(p > self.threshold for p in probabilities):
                self.entry_bar = current_bar_index
                return "buy"

        return None


class PrecomputedSignalStrategy:
    """A strategy that uses pre-computed predictions for long and short signals."""

    def __init__(
        self,
        predictions: pd.DataFrame,
        long_threshold: float,
        short_threshold: float,
        hold_period: int,
    ):
        self.predictions = predictions
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.hold_period = hold_period
        self.entry_bar = -1
        self.position_type: Optional[str] = None  # Can be 'long' or 'short'

    def generate_signal(self, bar: pd.Series, current_bar_index: int) -> str | None:
        """Generates a buy, sell, or hold signal based on pre-computed probabilities."""
        timestamp = bar.name

        # Exit condition: close position after hold_period
        if (
            self.position_type is not None
            and (current_bar_index - self.entry_bar) >= self.hold_period
        ):
            signal = "sell" if self.position_type == "long" else "buy_to_cover"
            self.position_type = None
            self.entry_bar = -1
            return signal

        # Entry condition: only enter if flat
        if self.position_type is None:
            if timestamp in self.predictions.index:
                long_prob = self.predictions.loc[timestamp, "long_label"]
                short_prob = self.predictions.loc[timestamp, "short_label"]

                if long_prob > self.long_threshold:
                    self.position_type = "long"
                    self.entry_bar = current_bar_index
                    return "buy"
                elif short_prob > self.short_threshold:
                    self.position_type = "short"
                    self.entry_bar = current_bar_index
                    return "sell_short"

        return None
