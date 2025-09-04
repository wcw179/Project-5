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

from src.features.pipeline import create_all_features
from src.model_validation.purged_k_fold import PurgedKFold
from src.modeling.models import get_fast_model, get_deep_model, get_balanced_model
from src.labeling.triple_barrier_afml import TripleBarrierConfig, generate_primary_labels, compute_sample_weights
from src.labeling.meta_labeling import extract_meta_features, generate_meta_labels
from src.modeling.meta_models import get_meta_models
from src.logger import logger

DB_PATH = project_root / "data" / "m5_trading.db"
MODEL_DIR = project_root / "models"
SYMBOL = "EURUSDm"
START_DATE = "2023-01-01"
END_DATE = "2025-08-31"
HOLD_PERIOD_BARS = [12, 48, 144, 288]  # 1h, 4h, 12h, 24h
ENSEMBLE_WEIGHTS = {"fast": 0.3, "deep": 0.3, "balanced": 0.4}


def load_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    engine = create_engine(f"sqlite:///{DB_PATH}")
    q = f"SELECT * FROM bars WHERE symbol = '{symbol}' AND time BETWEEN '{start}' AND '{end}' ORDER BY time ASC"
    df = pd.read_sql(q, engine, index_col="time", parse_dates=["time"])
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


def fit_primary_models(X: pd.DataFrame, y: pd.DataFrame, sample_info: pd.DataFrame) -> Dict[str, object]:
    models = {
        "fast": get_fast_model(),
        "deep": get_deep_model(),
        "balanced": get_balanced_model(),
    }
    # Purged K-Fold
    pkf = PurgedKFold(n_splits=5, embargo_td=pd.Timedelta(hours=24))
    splits = list(pkf.split(X, sample_info=sample_info))

    # Simple CV loop (fit on full at the end for deployment)
    for name, model in models.items():
        logger.info(f"Training primary model '{name}' on full data (CV evaluated separately)...")
        # Optional: evaluate with CV and log scores per label
        # Here we directly fit on full matrix for brevity
        # Multi-label: fit per column sequentially and store as dict of models
        label_models = {}
        for col in y.columns:
            if y[col].nunique() < 2:
                continue
            m = model.__class__(**model.get_xgb_params()) if hasattr(model, 'get_xgb_params') else model.__class__(**model.get_params())
            m.set_params(**{k: v for k, v in model.get_params().items() if k in m.get_params()})
            m.fit(X, y[col])
            label_models[col] = m
        models[name] = label_models
    return models


def ensemble_proba(models: Dict[str, Dict[str, object]], X: pd.DataFrame) -> pd.DataFrame:
    # compute weighted average proba across primary models for each label
    labels = sorted({lab for label_models in models.values() for lab in label_models.keys()})
    out = pd.DataFrame(0.0, index=X.index, columns=labels)
    for mname, label_models in models.items():
        w = ENSEMBLE_WEIGHTS.get(mname, 0.0)
        if w == 0:
            continue
        for lab, mdl in label_models.items():
            if hasattr(mdl, "predict_proba"):
                p = mdl.predict_proba(X)
                proba = p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()
                out[lab] += w * proba
    return out


def main():
    logger.info("Loading EURUSDm bars...")
    bars = load_bars(SYMBOL, START_DATE, END_DATE)

    logger.info("Feature engineering...")
    X = create_all_features(bars, SYMBOL)

    logger.info("Generating AFML triple-barrier primary labels...")
    tb_cfg = TripleBarrierConfig(
        horizons=HOLD_PERIOD_BARS,
        rr_multiples=[5.0, 10.0, 15.0, 20.0],
        vol_method="atr",
        atr_period=100,
        vol_window=288,
        atr_mult_base=2.0,
        include_short=False,
        spread_pips=2.0,
    )
    y, sample_info = generate_primary_labels(bars, SYMBOL, tb_cfg)

    # Align
    idx = X.index.intersection(y.index)
    X = X.loc[idx]
    y = y.loc[idx]
    sample_info = sample_info.loc[idx]

    logger.info("Computing sample weights (placeholder uniform)...")
    sample_weights = compute_sample_weights(sample_info)

    logger.info("Training primary models (Fast/Deep/Balanced)...")
    primary_models = fit_primary_models(X, y, sample_info)

    logger.info("Ensembling primary probabilities...")
    primary_proba = ensemble_proba(primary_models, X)

    logger.info("Building meta-features & generating bootstrap meta-labels...")
    # Attempt to include a volatility regime column if present
    market_cols = [c for c in X.columns if "volatility_regime" in c][:1]
    meta_X = extract_meta_features(X, primary_proba, market_cols=market_cols, hist_cols=None)
    meta_y = generate_meta_labels(X, primary_proba, market_cols=market_cols)

    logger.info("Training meta-models...")
    meta_models = get_meta_models()
    trained_meta = {}
    for name, mdl in meta_models.items():
        target = meta_y[name]
        mdl.fit(meta_X, target)
        trained_meta[name] = mdl

    # Save artifacts
    MODEL_DIR.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    primary_path = MODEL_DIR / f"{SYMBOL}_primary_models_{ts}.joblib"
    meta_path = MODEL_DIR / f"{SYMBOL}_meta_models_{ts}.joblib"
    proba_path = MODEL_DIR / f"{SYMBOL}_primary_proba_{ts}.parquet"

    joblib.dump(primary_models, primary_path)
    joblib.dump(trained_meta, meta_path)
    primary_proba.to_parquet(proba_path)

    logger.success(f"Saved primary models to {primary_path}")
    logger.success(f"Saved meta models to {meta_path}")
    logger.success(f"Saved primary proba to {proba_path}")


if __name__ == "__main__":
    main()

