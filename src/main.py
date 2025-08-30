"""Main application entry point."""

from src.config import config
from src.data_pipeline.store import get_data_store
from src.features.pipeline import create_all_features
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
    data_store = get_data_store()  # Initialize the data store
    """Main application entry point."""
    logger.info(f"Starting {config.project_name} v{config.version}")

    # Verify data store loading
    for symbol, data in data_store.get_all_data().items():
        if data is not None and not data.empty:
            logger.info(
                f"DataStore loaded for {symbol}: {len(data)} bars, "
                f"from {data.index.min()} to {data.index.max()}"
            )
        else:
            logger.warning(f"DataStore is empty for {symbol}.")

    # --- Feature Engineering Example ---
    eurusd_data = data_store.get_data("EURUSDm")
    if eurusd_data is not None:
        logger.info("Running feature engineering pipeline for EURUSDm...")
        eurusd_featured = create_all_features(eurusd_data, "EURUSDm")
        logger.info(f"Feature engineering complete. New shape: {eurusd_featured.shape}")
        logger.info(
            f"First 5 rows of featured data:\n{eurusd_featured.head().to_string()}"
        )

    logger.info("Application finished.")


if __name__ == "__main__":
    main()
