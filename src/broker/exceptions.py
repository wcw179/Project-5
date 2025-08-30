"""Custom exceptions for the broker module."""


class BrokerError(Exception):
    """Base exception for broker-related errors."""

    pass


class ConnectionError(BrokerError):
    """Raised when connection to the broker fails."""

    pass


class AuthenticationError(BrokerError):
    """Raised when authentication with the broker fails."""

    pass


class OrderError(BrokerError):
    """Raised for errors related to order placement or modification."""

    pass


class DataError(BrokerError):
    """Raised for errors related to data retrieval."""

    pass
