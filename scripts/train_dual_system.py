"""
Train Dual-Model System (Primary + Meta) for EURUSDm (Phase 1A)

Pipeline:
1) Load EURUSDm bars from SQLite over given date range
2) Feature engineering via create_all_features
3) AFML triple-barrier labels (multi-RR, multi-horizon) for primary tasks
4) Cross-validation (Purged K-Fold) training for primary models (Fast/Deep/Balanced)
5) Ensemble primary predict_proba with configured weights
6) Build meta-features and bootstrap meta-labels (rule-based)
7) Train meta-models (trade_quality, position_sizing, entry_timing, risk_assessment)
8) Save artifacts and simple report
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# project path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import json
import xgboost as xgb
from src.features.pipeline import create_all_features
from src.labeling.triple_barrier_afml import (
    TripleBarrierConfig,
    generate_primary_labels,
    compute_sample_weights,
)
from src.labeling.meta_labeling import extract_meta_features, generate_meta_labels
from src.modeling.meta_models import get_meta_models
from src.logger import logger

DB_PATH = project_root / "data" / "m5_trading.db"
MODEL_DIR = project_root / "models"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2025-08-31"
HOLD_PERIOD_BARS = [12, 48, 144, 288]  # 1h, 4h, 12h, 24h


def load_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{DB_PATH}")
    q = f"SELECT * FROM bars WHERE symbol = '{symbol}' AND time BETWEEN '{start}' AND '{end}' ORDER BY time ASC"
    df = pd.read_sql(q, engine, index_col="time", parse_dates=["time"])
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


def fit_primary_model(
    X: pd.DataFrame, y: pd.DataFrame, params: dict
) -> Dict[str, xgb.XGBClassifier]:
    """Trains one XGBoost classifier per label using the best hyperparameters."""
    logger.info("Training optimized primary model on full data...")
    trained_models = {}
    for col in y.columns:
        if y[col].nunique() < 2:
            logger.warning(f"Skipping label '{col}' as it has only one class.")
            continue

        model = xgb.XGBClassifier(**params)
        model.fit(X, y[col])
        trained_models[col] = model

    logger.success("Primary model training complete.")
    return trained_models


def predict_primary_proba(
    models: Dict[str, xgb.XGBClassifier], X: pd.DataFrame
) -> pd.DataFrame:
    """Generates prediction probabilities from the trained primary models."""
    probas = {}
    for label, model in models.items():
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            # Ensure we get the probability of the positive class (1)
            proba_col = p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()
            probas[label] = proba_col

    return pd.DataFrame(probas, index=X.index)


def main():
    logger.info("--- Starting Dual-Model Training Pipeline ---")

    # 1. Load Data and Features
    logger.info("Loading EURUSDm bars...")
    bars = load_bars(SYMBOL, START_DATE, END_DATE)
    logger.info("Feature engineering...")
    X = create_all_features(bars, SYMBOL)

    # 2. Generate Primary Labels (AFML Triple-Barrier)
    logger.info("Generating AFML triple-barrier primary labels (long & short)...")
    tb_cfg = TripleBarrierConfig(
        horizons=HOLD_PERIOD_BARS,
        rr_multiples=[5.0, 10.0, 15.0, 20.0],
        vol_method="atr",
        atr_period=100,
        vol_window=288,
        atr_mult_base=2.0,
        include_short=True,  # Generate both long and short labels
        spread_pips=2.0,
    )
    y, sample_info = generate_primary_labels(bars, SYMBOL, tb_cfg)

    # Align dataframes
    idx = X.index.intersection(y.index)
    X, y, sample_info = X.loc[idx], y.loc[idx], sample_info.loc[idx]

    # 3. Load Best Hyperparameters
    params_path = MODEL_DIR / f"{SYMBOL}_best_primary_params.json"
    if not params_path.exists():
        logger.error(f"Hyperparameter file not found at {params_path}")
        logger.error("Please run 'scripts/run_optimization.py' first.")
        return
    logger.info(f"Loading best hyperparameters from {params_path}")
    with open(params_path, "r") as f:
        best_params = json.load(f)

    # 4. Train Primary Model
    primary_model = fit_primary_model(X, y, best_params)

    # 5. Get Primary Probabilities
    logger.info("Generating primary model probabilities...")
    primary_proba = predict_primary_proba(primary_model, X)

    # 6. Train Meta-Models
    logger.info("Building meta-features & generating bootstrap meta-labels...")
    market_cols = [c for c in X.columns if "volatility_regime" in c][:1]
    meta_X = extract_meta_features(X, primary_proba, market_cols=market_cols)
    meta_y = generate_meta_labels(X, primary_proba, market_cols=market_cols)

    logger.info("Training meta-models...")
    meta_models = get_meta_models()
    trained_meta_models = {}
    for name, model in meta_models.items():
        target = meta_y[name]
        model.fit(meta_X, target)
        trained_meta_models[name] = model

    # 7. Save Artifacts
    MODEL_DIR.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    primary_path = MODEL_DIR / f"{SYMBOL}_primary_model_{ts}.joblib"
    meta_path = MODEL_DIR / f"{SYMBOL}_meta_models_{ts}.joblib"

    joblib.dump(primary_model, primary_path)
    joblib.dump(trained_meta_models, meta_path)

    logger.success(f"Saved PRIMARY model to {primary_path}")
    logger.success(f"Saved META models to {meta_path}")
    logger.info("--- Dual-Model Training Pipeline Finished ---")


if __name__ == "__main__":
    main()
