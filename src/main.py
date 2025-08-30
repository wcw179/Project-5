"""Main application entry point."""

from src.config import config
from src.logger import logger


def process_symbol(symbol: str, request_id: str):
    """
    Example function to demonstrate logging with context.
    """
    # Add context to the logger for this specific request
    log = logger.bind(request_id=request_id, symbol=symbol)

    log.info("Starting processing for symbol.")

    try:
        # Simulate some work
        log.debug("Fetching data...")
        if symbol == "EURUSDm":
            log.info("Processing EURUSDm specific logic.")
        else:
            log.warning(
                "Symbol has no specific logic.", extra_details="Not implemented"
            )

        # Simulate an error
        if symbol == "XAUUSDm":
            raise ValueError("Invalid price data for XAUUSDm")

        log.success("Successfully processed symbol.")

    except Exception:
        log.exception("An error occurred during processing.")


def main():
    """Main application entry point."""
    logger.info(f"Starting {config.project_name} v{config.version}")

    # Example usage
    process_symbol("EURUSDm", "req-123")
    process_symbol("GBPUSDm", "req-124")
    process_symbol("XAUUSDm", "req-125")

    logger.info("Application finished.")


if __name__ == "__main__":
    main()
