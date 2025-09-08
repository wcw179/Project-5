"""Master feature engineering pipeline that orchestrates all feature creation."""

import pandas as pd

from src.features.fractal.dimension import add_fractal_features
from src.features.microstructure.liquidity import add_microstructure_features
from src.features.regime.volatility_regime import add_volatility_regime
from src.features.sentiment.indicators import add_sentiment_features
from src.features.temporal.sessions import add_session_features
from src.features.trend.alignment import add_trend_features
from src.features.volatility.distribution import add_volatility_features
from src.features.macro.features import add_macro_features
from src.features.entropy.features import add_entropy_features
from src.features.entropy.generalized_mean import add_generalized_mean_entropy_features
from src.features.entropy.portfolio_concentration import add_portfolio_concentration_features
from src.features.structural_breaks.features import add_structural_break_features
from src.features.mlfinpy_features import (
    add_mlfinpy_filter_features,
    add_mlfinpy_structural_break_features,
)
from src.logger import logger


def create_all_features(
    df: pd.DataFrame,
    symbol: str,
    *,
    macro_context: dict | None = None,
    portfolio_weights: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Orchestrates the entire feature engineering process for a given symbol.

    Args:
        df: Raw DataFrame with OHLCV data for a single symbol.
        symbol: The symbol being processed, for logging purposes.
        macro_context: Optional dict of external macro/cross-asset series. If None,
            macro features will be NaN and can be filled downstream.
        portfolio_weights: Optional portfolio weights (index timestamps, columns assets)
            for concentration features (AFML §18.8.3). If provided, concentration
            metrics will be joined.

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

    if macro_context is not None:
        logger.debug(f"({symbol}) Adding macro & cross-asset features (V4 §2.6)...")
        df_featured = add_macro_features(df_featured, context=macro_context)
        logger.debug(f"({symbol}) ...Macro features OK.")
    else:
        logger.debug(f"({symbol}) Skipping macro features: no context provided.")

    logger.debug(f"({symbol}) Adding entropy features (V4 §2.6)...")
    df_featured = add_entropy_features(df_featured)
    logger.debug(f"({symbol}) ...Entropy features OK.")

    logger.debug(f"({symbol}) Adding generalized mean & Rényi entropy features (AFML §18.7)...")
    df_featured = add_generalized_mean_entropy_features(df_featured)
    logger.debug(f"({symbol}) ...Generalized mean & Rényi features OK.")

    logger.debug(f"({symbol}) Adding fractal features...")
    df_featured = add_fractal_features(df_featured)
    logger.debug(f"({symbol}) ...Fractal features OK.")

    logger.debug(f"({symbol}) Adding regime features...")
    df_featured = add_volatility_regime(df_featured)
    logger.debug(f"({symbol}) ...Regime features OK.")

    logger.debug(f"({symbol}) Adding mlfinpy filter features...")
    df_featured = add_mlfinpy_filter_features(df_featured)
    logger.debug(f"({symbol}) ...mlfinpy filter features OK.")

    logger.debug(f"({symbol}) Adding mlfinpy structural break features...")
    df_featured = add_mlfinpy_structural_break_features(df_featured)
    logger.debug(f"({symbol}) ...mlfinpy structural break features OK.")

    logger.debug(f"({symbol}) Adding custom structural break features...")
    df_featured = add_structural_break_features(df_featured)
    logger.debug(f"({symbol}) ...Custom structural break features OK.")


    if portfolio_weights is not None:
        logger.debug(f"({symbol}) Adding portfolio concentration features (AFML §18.8.3)...")
        df_featured = add_portfolio_concentration_features(
            df_featured, weights=portfolio_weights
        )
        logger.debug(f"({symbol}) ...Portfolio concentration features OK.")
    else:
        logger.debug(
            f"({symbol}) Skipping portfolio concentration features: no weights provided."
        )

    # Drop rows with NaN values created by rolling indicators
    initial_rows = len(df_featured)
    logger.debug(f"({symbol}) Dropping NaN values...")
    df_featured.dropna(inplace=True)
    final_rows = len(df_featured)
    logger.info(f"({symbol}) Dropped {initial_rows - final_rows} rows with NaN values.")

    # Drop the symbol column if it exists, as it's not a feature
    if "symbol" in df_featured.columns:
        df_featured = df_featured.drop(columns=["symbol"])

    logger.success(
        f"Feature engineering pipeline completed for {symbol}. Shape: {df_featured.shape}"
    )
    return df_featured
