"""
Meta-model definitions for V4 dual-model architecture.

Provides helper constructors for:
- Trade quality classifier (multi-class)
- Position sizing regressor (continuous)
- Entry timing classifier (multi-class)
- Risk assessment classifier (multi-class)
"""
from __future__ import annotations

from typing import Dict

from xgboost import XGBClassifier, XGBRegressor


def get_meta_models() -> Dict[str, object]:
    meta_models = {
        "trade_quality": XGBClassifier(
            objective="multi:softprob",
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
        ),
        "position_sizing": XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
        ),
        "entry_timing": XGBClassifier(
            objective="multi:softprob",
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
        ),
        "risk_assessment": XGBClassifier(
            objective="multi:softprob",
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            random_state=42,
        ),
    }
    return meta_models

