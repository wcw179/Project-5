"""Master feature engineering pipeline that orchestrates all feature creation."""

import pandas as pd

from src.features.microstructure.liquidity import add_microstructure_features
from src.features.sentiment.indicators import add_sentiment_features
from src.features.temporal.sessions import add_session_features
from src.features.trend.alignment import add_trend_features
from src.features.volatility.distribution import add_volatility_features
from src.logger import logger


def create_all_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Orchestrates the entire feature engineering process for a given symbol.

    Args:
        df: Raw DataFrame with OHLCV data for a single symbol.
        symbol: The symbol being processed, for logging purposes.

    Returns:
        DataFrame enriched with all new features.
    """
    logger.info(f"Starting feature engineering pipeline for {symbol}...")

    # Make a copy to avoid modifying the original DataFrame
    df_featured = df.copy()

    # The order of feature creation can be important.
    logger.debug(f"({symbol}) Adding temporal features...")
    df_featured = add_session_features(df_featured)
    logger.debug(f"({symbol}) ...Temporal features OK.")

    logger.debug(f"({symbol}) Adding microstructure features...")
    df_featured = add_microstructure_features(df_featured)
    logger.debug(f"({symbol}) ...Microstructure features OK.")

    logger.debug(f"({symbol}) Adding trend features...")
    df_featured = add_trend_features(df_featured)
    logger.debug(f"({symbol}) ...Trend features OK.")

    logger.debug(f"({symbol}) Adding volatility features...")
    df_featured = add_volatility_features(df_featured)
    logger.debug(f"({symbol}) ...Volatility features OK.")

    logger.debug(f"({symbol}) Adding sentiment features...")
    df_featured = add_sentiment_features(df_featured)
    logger.debug(f"({symbol}) ...Sentiment features OK.")

    # Drop rows with NaN values created by rolling indicators
    initial_rows = len(df_featured)
    logger.debug(f"({symbol}) Dropping NaN values...")
    df_featured.dropna(inplace=True)
    final_rows = len(df_featured)
    logger.info(f"({symbol}) Dropped {initial_rows - final_rows} rows with NaN values.")

    logger.success(
        f"Feature engineering pipeline completed for {symbol}. Shape: {df_featured.shape}"
    )
    return df_featured
