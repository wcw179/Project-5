"""
Script to run hyperparameter optimization for the final model using Optuna,
Purged K-Fold CV, and the full feature/label generation pipeline.
"""
import sys
from pathlib import Path
import json
import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import gc
from loguru import logger
from sklearn.metrics import f1_score, classification_report

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from mlfinpy.cross_validation import PurgedKFold

# --- Logger Configuration ---
log_file_path = project_root / "logs" / "final_optimization_multi.log"
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>")
logger.add(log_file_path, rotation="10 MB", retention="10 days", enqueue=True, serialize=False)

# --- Constants & Config ---
MODEL_DIR = project_root / "models"
CONFIG = {
    'n_trials': 100,
    'n_splits': 5,
    'early_stopping_patience': 20,
    'improvement_threshold': 0.001,
    'random_state': 42,
    'n_jobs': -1,  # Safer for complex operations
}

def objective(trial: optuna.trial.Trial, X: pd.DataFrame, y: pd.Series, cv_splits: list) -> float:
    """Optuna objective function to train and evaluate models using F1-score."""
    try:
        params = {
            'objective': 'multi:softprob',  # Use softprob for probability scores
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 2),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'tree_method': 'hist',
            'random_state': CONFIG['random_state'],
        }

        logger.info(f"Trial {trial.number}: Starting with params")
        scores = []
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            try:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
                y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2})

                if len(np.unique(y_train_mapped)) < 2:
                    logger.warning(f"Fold {fold}: Insufficient class diversity in training data. Skipping.")
                    continue

                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train_mapped, eval_set=[(X_val, y_val_mapped)], early_stopping_rounds=10, verbose=False)

                pred_probs = model.predict_proba(X_val)
                preds = np.argmax(pred_probs, axis=1)
                preds_mapped_back = pd.Series(preds).map({0: -1, 1: 0, 2: 1}).values

                score = f1_score(y_val, preds_mapped_back, average='weighted')
                scores.append(score)

                del model, preds, preds_mapped_back
                gc.collect()

            except Exception as e:
                logger.error(f"Error in fold {fold}: {e}")
                continue

        if not scores:
            logger.error(f"Trial {trial.number}: No valid scores obtained from any fold.")
            return -1.0

        mean_score = float(np.mean(scores))
        logger.info(f"Trial {trial.number}: Finished with mean F1-score = {mean_score:.4f}")
        return mean_score

    except Exception as e:
        logger.error(f"Trial {trial.number} failed entirely: {e}")
        return -1.0

def save_optimization_results(study: optuna.study.Study, model_dir: Path):
    """Save detailed optimization results."""
    joblib.dump(study, model_dir / "optimization_study.pkl")
    trials_df = study.trials_dataframe()
    trials_df.to_csv(model_dir / "optimization_trials.csv", index=False)
    try:
        import optuna.visualization as vis
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(model_dir / "optimization_history.html"))
    except ImportError:
        logger.warning("Optuna visualization not available")

def validate_best_model(study: optuna.study.Study, X: pd.DataFrame, y: pd.Series, cv_splits: list):
    """Validate the best model with detailed metrics."""
    logger.info("--- Starting Final Validation of Best Model ---")
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'random_state': CONFIG['random_state'],
    })

    all_y_true, all_y_pred = [], []
    for train_idx, val_idx in cv_splits:
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
        y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2})

        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train_mapped, eval_set=[(X_val, y_val_mapped)], early_stopping_rounds=10, verbose=False)

        pred_probs = model.predict_proba(X_val)
        preds = np.argmax(pred_probs, axis=1)
        preds_mapped_back = pd.Series(preds).map({0: -1, 1: 0, 2: 1}).values

        all_y_true.extend(y_val.values)
        all_y_pred.extend(preds_mapped_back)

    logger.info("--- Final Validation Results ---")
    report = classification_report(all_y_true, all_y_pred, digits=4)
    logger.info(f"\n{report}")

def main():
    """Main function to run the optimization study."""
    logger.info(f"--- Starting Multi-Symbol Model Optimization ({CONFIG['n_trials']} Trials) ---")

    data_dir = project_root / "data" / "processed"
    X_full_path = data_dir / "X_full.parquet"
    y_full_path = data_dir / "y_full.parquet"

    if not X_full_path.exists() or not y_full_path.exists():
        logger.error("Parquet dataset files not found. Please run the build script first.")
        return

    X = pd.read_parquet(X_full_path)
    y = pd.read_parquet(y_full_path).squeeze()
    logger.info(f"Loaded {len(X)} samples from Parquet files.")

    sample_info_path = data_dir / "sample_info_full.parquet"
    if not sample_info_path.exists():
        logger.error("Sample info file not found. Please run the build script first.")
        return
    sample_info = pd.read_parquet(sample_info_path)
    t1 = sample_info.loc[X.index, 't1']

    # Đã sửa: Proper alignment checking
    common_idx = X.index.intersection(sample_info.index)
    if len(common_idx) == 0:
        raise ValueError("No common indices between X and sample_info")
    X_aligned = X.loc[common_idx]
    t1 = sample_info.loc[common_idx, 't1']

    cv = PurgedKFold(n_splits=CONFIG['n_splits'], t1=t1, pct_embargo=0.01)
    cv_splits = list(cv.split(X))

    def early_stopping_callback(study, _):
        if len(study.trials) >= CONFIG['early_stopping_patience']:
            recent_trials = study.trials[-CONFIG['early_stopping_patience']:]
            recent_values = [t.value for t in recent_trials if t.value is not None]

            if len(recent_values) >= CONFIG['early_stopping_patience']:
                improvement = max(recent_values) - min(recent_values)
                if improvement < CONFIG['improvement_threshold']:
                    logger.info(f"Early stopping: No significant improvement in last {CONFIG['early_stopping_patience']} trials.")
                    study.stop()

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X, y, cv_splits),
        n_trials=CONFIG['n_trials'],
        callbacks=[early_stopping_callback],
        n_jobs=CONFIG['n_jobs']
    )

    logger.info(f"Best trial score: {study.best_value}")
    logger.info(f"Best trial params: {study.best_params}")

    params_path = MODEL_DIR / "multi_symbol_final_best_params.json"
    with open(params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    logger.success(f"Saved best hyperparameters to {params_path}")

    save_optimization_results(study, MODEL_DIR)
    validate_best_model(study, X, y, cv_splits)
    best_model = xgb.XGBClassifier(**best_params)
    best_model.fit(X, y.map({-1:0,0:1,1:2}))
    joblib.dump(best_model, MODEL_DIR / "multi_symbol_final_model.pkl")

if __name__ == "__main__":
    main()

