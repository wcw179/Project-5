"""Definitions for the XGBoost model ensemble."""

import xgboost as xgb


def get_fast_model() -> xgb.XGBClassifier:
    """Returns the Fast XGBoost model, optimized for speed."""
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        objective="binary:logistic",  # Base objective for multi-label
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,  # Use all available CPU cores
        random_state=42,
        tree_method="hist",  # Fast histogram-based method
        base_score=0.5,
    )


def get_deep_model() -> xgb.XGBClassifier:
    """Returns the Deep XGBoost model, for complex pattern detection."""
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.2,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        base_score=0.5,
    )


def get_balanced_model() -> xgb.XGBClassifier:
    """Returns the Balanced XGBoost model, an optimal trade-off."""
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.075,
        subsample=0.75,
        colsample_bytree=0.75,
        gamma=0.15,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        base_score=0.5,
    )
