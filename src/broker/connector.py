"""MT5 Broker Connector."""

import backoff
import MetaTrader5 as mt5
import pandas as pd

from src.broker.exceptions import AuthenticationError, ConnectionError
from src.config import config
from src.logger import logger


class MT5Connector:
    """Handles connection and data retrieval from MetaTrader 5."""

    def __init__(self, dry_run: bool = config.broker.dry_run):
        self.dry_run = dry_run
        self._is_connected = False
        mode = "DRY RUN" if dry_run else "LIVE"
        logger.info(f"MT5 Connector initialized in {mode} mode.")

    @backoff.on_exception(backoff.expo, ConnectionError, max_tries=3, factor=2)
    def connect(self) -> None:
        """Establishes connection to the MT5 terminal."""
        if self.dry_run:
            logger.info("Dry run mode: Skipping actual MT5 connection.")
            self._is_connected = True
            return

        if self._is_connected:
            logger.warning("Already connected to MT5.")
            return

        logger.info("Connecting to MT5 terminal...")
        if not mt5.initialize():
            raise ConnectionError(
                f"MT5 initialize() failed, error code: {mt5.last_error()}"
            )

        account_info = mt5.account_info()
        if not account_info:
            mt5.shutdown()
            raise AuthenticationError("Failed to retrieve account info.")

        self._is_connected = True
        logger.success(f"Connected to MT5 account: {account_info.login}")

    def disconnect(self) -> None:
        """Shuts down the connection to the MT5 terminal."""
        if self.dry_run:
            logger.info("Dry run mode: Skipping actual MT5 disconnection.")
            self._is_connected = False
            return

        if self._is_connected:
            logger.info("Disconnecting from MT5 terminal...")
            mt5.shutdown()
            self._is_connected = False

    def is_healthy(self) -> bool:
        """Checks if the connection is active and responsive."""
        if self.dry_run:
            return True
        return self._is_connected and mt5.terminal_info() is not None

    def get_m5_bars(self, symbol: str, count: int) -> pd.DataFrame:
        """(STUB) Retrieves M5 OHLCV bars for a given symbol."""
        if not self.is_healthy():
            raise ConnectionError("Broker is not connected or unhealthy.")

        logger.info(f"Retrieving {count} M5 bars for {symbol}...")
        # In a real implementation, you would call mt5.copy_rates_from_pos()
        # This is a stub returning dummy data.
        if self.dry_run:
            dummy_data = {
                "time": pd.to_datetime(
                    pd.date_range(end=pd.Timestamp.now(), periods=count, freq="5min")
                ),
                "open": [1.0] * count,
                "high": [1.1] * count,
                "low": [0.9] * count,
                "close": [1.05] * count,
                "tick_volume": [100] * count,
            }
            return pd.DataFrame(dummy_data)

        raise NotImplementedError("Live data retrieval not yet implemented.")

    def place_order(self, symbol: str, order_type: str, volume: float) -> dict:
        """(STUB) Places a trade order."""
        if not self.is_healthy():
            raise ConnectionError("Broker is not connected or unhealthy.")

        logger.info(f"Placing {order_type} order for {volume} of {symbol}...")
        if self.dry_run:
            return {
                "status": "success",
                "order_id": 12345,
                "message": "Dry run order placed.",
            }

        raise NotImplementedError("Live order placement not yet implemented.")

    def get_open_positions(self) -> list:
        """(STUB) Retrieves all open positions."""
        if not self.is_healthy():
            raise ConnectionError("Broker is not connected or unhealthy.")

        logger.info("Retrieving open positions...")
        if self.dry_run:
            return []  # No open positions in dry run stub

        raise NotImplementedError("Live position retrieval not yet implemented.")
