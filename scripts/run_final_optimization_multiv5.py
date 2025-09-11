"""
Comprehensive script (v5) for hyperparameter optimization using Optuna.

This version incorporates several key enhancements for robust financial ML modeling:
1.  **PurgedStratifiedKFold:** Uses a sophisticated cross-validator to prevent data
    leakage and preserve class balance, crucial for time-series data.
2.  **Configurable Financial Metrics:** Allows switching between proxy returns and
    true returns (`ret` column) for calculating Sharpe, Sortino, and Hit Ratio.
3.  **Flexible Combined Metric:** The primary optimization metric is a weighted
    combination of accuracy and a financial metric, with weights configurable
    in the `CONFIG` dictionary.
4.  **Detailed Results Storage:** Saves not only the best trial but also detailed
    metrics for each fold of each trial to a CSV for in-depth analysis.
5.  **Safe Column Handling:** Robustly handles the presence or absence of a
    'symbol' column in the feature set.
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
from sklearn.metrics import f1_score, classification_report, accuracy_score
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import our enhanced cross-validation
from src.model_validation.financial_ml_cross_validation import PurgedStratifiedKFold

# --- Logger Configuration ---
log_file_path = project_root / "logs" / "enhanced_optimization_multi.log"
# Ensure log directory exists before adding file sink
log_file_path.parent.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
)
logger.add(
    log_file_path,
    rotation="10 MB",
    retention="10 days",
    enqueue=True,
    serialize=False
)

# --- Enhanced Configuration ---
MODEL_DIR = project_root / "models"
CONFIG = {
    'n_trials': 100,
    'n_splits': 5,
    'early_stopping_patience': 20,
    'improvement_threshold': 0.001,
    'random_state': 42,
    'n_jobs_optuna': 1,  # Conservative for memory safety in Optuna
    'n_jobs_xgb': -1, # Use all available cores for XGBoost
    'xgb_early_stopping': 50,
    'pct_embargo': 0.01,
    'min_samples_per_class': 10,
    'use_true_returns': True, # Switch to use 'ret' column for financial metrics
    'evaluation_metrics': ['f1_weighted', 'accuracy', 'sharpe', 'sortino', 'hit_ratio', 'combined'],
    'primary_metric': 'combined',
    'metric_weights': { # Weights for the 'combined' metric
        'accuracy': 0.2,
        'sharpe': 0.8
    }
}


def evaluate_predictions(y_true: pd.Series, y_pred: np.array,
                         true_returns: Optional[pd.Series] = None) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics using either true returns or a proxy."""
    metrics = {}
    annualization_factor = 252 * 288 # 288 5-minute bars per 24h day

    # --- Classification Metrics ---
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # --- Financial Metrics ---
    positions = pd.Series(y_pred, index=y_true.index).map({1: 1, -1: -1, 0: 0})

    if CONFIG['use_true_returns'] and true_returns is not None:
        strategy_returns = positions * true_returns
    else:
        # Fallback to proxy returns if true returns are not available or configured off
        market_proxy = y_true.map({1: 0.001, -1: -0.001, 0: 0})
        strategy_returns = positions * market_proxy

    strategy_returns.fillna(0, inplace=True)
    active_trades = strategy_returns[strategy_returns != 0]

    # Sharpe Ratio
    if np.std(strategy_returns) > 0:
        metrics['sharpe'] = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(annualization_factor)
    else:
        metrics['sharpe'] = 0.0

    # Sortino Ratio
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0
    if downside_std > 0:
        metrics['sortino'] = (np.mean(strategy_returns) / downside_std) * np.sqrt(annualization_factor)
    else:
        metrics['sortino'] = 0.0

    # Hit Ratio
    if not active_trades.empty:
        metrics['hit_ratio'] = (active_trades > 0).sum() / len(active_trades)
    else:
        metrics['hit_ratio'] = 0.0

    # --- Combined Metric (from CONFIG) ---
    weights = CONFIG['metric_weights']
    normalized_sharpe = max(0, min(1, (metrics['sharpe'] + 2) / 5)) # Normalize Sharpe in a reasonable range
    metrics['combined'] = (weights['accuracy'] * metrics['accuracy'] +
                         weights['sharpe'] * normalized_sharpe)

    return metrics


def objective(trial: optuna.trial.Trial, X: pd.DataFrame, y_df: pd.DataFrame,
              cv_splits: List[Tuple[np.array, np.array]]) -> float:
    """Enhanced Optuna objective function with comprehensive evaluation."""
    try:
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 2.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'tree_method': 'hist',
            'random_state': CONFIG['random_state'],
            'n_jobs': CONFIG['n_jobs_xgb'],
        }

        fold_metrics = {metric: [] for metric in CONFIG['evaluation_metrics']}
        valid_folds = 0

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            try:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train_df, y_val_df = y_df.iloc[train_idx], y_df.iloc[val_idx]
                y_train, y_val = y_train_df['label'], y_val_df['label']

                X_train_numeric = X_train.drop(columns=['symbol'], errors='ignore')
                X_val_numeric = X_val.drop(columns=['symbol'], errors='ignore')

                y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
                y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2})

                if len(np.unique(y_train_mapped)) < 2: continue

                model = xgb.XGBClassifier(**params)
                model.fit(X_train_numeric, y_train_mapped, eval_set=[(X_val_numeric, y_val_mapped)],
                          early_stopping_rounds=CONFIG['xgb_early_stopping'], verbose=False)

                preds = np.argmax(model.predict_proba(X_val_numeric), axis=1)
                preds_mapped_back = pd.Series(preds).map({0: -1, 1: 0, 2: 1}).values

                metrics = evaluate_predictions(y_val, preds_mapped_back, true_returns=y_val_df['ret'])
                for name, score in metrics.items(): fold_metrics[name].append(score)
                valid_folds += 1
                del model, preds, preds_mapped_back; gc.collect()
            except Exception as e:
                logger.error(f"Error in fold {fold}: {e}")
                continue

        if valid_folds == 0: return -1.0

        mean_scores = {metric: np.mean(scores) for metric, scores in fold_metrics.items()}
        trial.set_user_attr("fold_metrics", fold_metrics)
        for metric, score in mean_scores.items():
            trial.set_user_attr(f"mean_{metric}", score)

        primary_score = mean_scores.get(CONFIG['primary_metric'], -1.0)
        return float(primary_score)

    except Exception as e:
        logger.error(f"Trial {trial.number} failed entirely: {e}")
        return -1.0

def save_comprehensive_results(study: optuna.study.Study, model_dir: Path):
    """Save comprehensive optimization results, including detailed fold metrics."""
    logger.info("Saving comprehensive optimization results...")
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- Save Fold Metrics ---
    all_fold_metrics = []
    for trial in study.trials:
        if 'fold_metrics' in trial.user_attrs:
            fm = trial.user_attrs['fold_metrics']
            for fold_num, metric_scores in enumerate(zip(*fm.values())):
                res = {'trial': trial.number, 'fold': fold_num}
                res.update({metric: score for metric, score in zip(fm.keys(), metric_scores)})
                all_fold_metrics.append(res)
    if all_fold_metrics:
        fold_metrics_df = pd.DataFrame(all_fold_metrics)
        fold_metrics_df.to_csv(model_dir / "fold_metrics_v5.csv", index=False)
        logger.success(f"Fold metrics saved to {model_dir / 'fold_metrics_v5.csv'}")

    # --- Save Standard Results ---
    joblib.dump(study, model_dir / "optimization_study_v5.pkl")
    study.trials_dataframe().to_csv(model_dir / "optimization_trials_v5.csv", index=False)

    best_trial_results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial_number': study.best_trial.number,
        'optimization_config': CONFIG
    }
    with open(model_dir / "best_trial_results_v5.json", "w") as f:
        json.dump(best_trial_results, f, indent=4, default=str)

    logger.success("All optimization results saved.")

def final_model_validation(study: optuna.study.Study, X: pd.DataFrame, y_df: pd.DataFrame,
                          cv_splits: List[Tuple[np.array, np.array]]) -> Dict[str, Any]:
    """Comprehensive final model validation with all metrics"""
    logger.info("=== Starting Comprehensive Final Model Validation ===")

    # Prepare best parameters
    best_params = study.best_params.copy()
    best_params.update({
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'random_state': CONFIG['random_state'],
        'n_jobs': -1,
    })

    # Cross-validation results storage
    all_predictions = []
    all_true_labels = []
    fold_results = []
    saved_models = []

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        logger.info(f"Validating fold {fold + 1}/{len(cv_splits)}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_df, y_val_df = y_df.iloc[train_idx], y_df.iloc[val_idx]
        y_train, y_val = y_train_df['label'], y_val_df['label']

        # Safe drop of 'symbol' column
        X_train_numeric = X_train.drop(columns=['symbol'], errors='ignore')
        X_val_numeric = X_val.drop(columns=['symbol'], errors='ignore')

        # Label mapping
        y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
        y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2})

        # Train final model
        model = xgb.XGBClassifier(**best_params)
        model.fit(
            X_train_numeric, y_train_mapped,
            eval_set=[(X_val_numeric, y_val_mapped)],
            early_stopping_rounds=CONFIG['xgb_early_stopping'],
            verbose=False
        )

        # Predictions
        pred_probs = model.predict_proba(X_val_numeric)
        preds = np.argmax(pred_probs, axis=1)
        preds_mapped_back = pd.Series(preds).map({0: -1, 1: 0, 2: 1}).values

        # Store predictions
        all_predictions.extend(preds_mapped_back)
        all_true_labels.extend(y_val.values)

        # Calculate fold metrics
        fold_metrics = evaluate_predictions(y_val, preds_mapped_back, true_returns=y_val_df['ret'])
        fold_results.append(fold_metrics)

        # Save individual fold model
        model_path = MODEL_DIR / f"final_model_fold_{fold}.joblib"
        joblib.dump(model, model_path)
        saved_models.append(str(model_path))

        logger.info(f"Fold {fold} metrics: " +
                   ", ".join([f"{k}: {v:.4f}" for k, v in fold_metrics.items()]))

    # Aggregate results
    # For overall metrics, we need to align true returns with predictions
    all_true_labels_s = pd.Series(all_true_labels)
    all_true_returns_s = y_df.loc[all_true_labels_s.index, 'ret']
    overall_metrics = evaluate_predictions(
        all_true_labels_s,
        np.array(all_predictions),
        true_returns=all_true_returns_s
    )

    # Calculate cross-validation statistics
    cv_stats = {}
    for metric in CONFIG['evaluation_metrics']:
        scores = [fold[metric] for fold in fold_results]
        cv_stats[metric] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }

    # Classification report
    class_report = classification_report(
        all_true_labels, all_predictions,
        output_dict=True, digits=4
    )

    logger.info("=== Final Validation Results ===")
    logger.info("Overall Metrics:")
    for metric, score in overall_metrics.items():
        logger.info(f"  {metric}: {score:.4f}")

    logger.info("\nCross-Validation Statistics:")
    for metric, stats in cv_stats.items():
        logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    logger.info(f"\nClassification Report:\n{classification_report(all_true_labels, all_predictions, digits=4)}")

    return {
        'overall_metrics': overall_metrics,
        'cv_statistics': cv_stats,
        'fold_results': fold_results,
        'classification_report': class_report,
        'saved_models': saved_models,
        'best_parameters': best_params
    }


def train_final_ensemble_model(X: pd.DataFrame, y_df: pd.DataFrame,
                              best_params: Dict[str, Any]) -> str:
    """Train final ensemble model on full dataset"""
    logger.info("Training final ensemble model on full dataset...")

    # Prepare final parameters
    final_params = best_params.copy()
    final_params.update({
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'tree_method': 'hist',
        'random_state': CONFIG['random_state'],
        'n_jobs': -1,
    })

    # Map labels
    y_mapped = y_df['label'].map({-1: 0, 0: 1, 1: 2})

    # Train on full dataset
    final_model = xgb.XGBClassifier(**final_params)
    # Safe drop of 'symbol' column
    X_numeric = X.drop(columns=['symbol'], errors='ignore')
    final_model.fit(X_numeric, y_mapped)

    # Save final model
    final_model_path = MODEL_DIR / "enhanced_final_ensemble_model.joblib"
    joblib.dump(final_model, final_model_path)

    logger.success(f"Final ensemble model saved to: {final_model_path}")
    return str(final_model_path)


def main():
    """Main function to run the full optimization and validation pipeline."""
    logger.info(f"=== Starting Comprehensive Model Optimization (v5) ===")
    logger.info(f"Configuration: {json.dumps(CONFIG, indent=2)}")

    # --- 1. Load Data ---
    data_dir = project_root / "data" / "processed"
    X = pd.read_parquet(data_dir / "X_full_v3.parquet")
    y_df = pd.read_parquet(data_dir / "y_full_v3.parquet")

    # --- 2. Align and Prepare Data ---
    common_idx = X.index.intersection(y_df.index)
    X_aligned, y_df_aligned = X.loc[common_idx], y_df.loc[common_idx]
    t1 = y_df_aligned['t1']

    X_for_split = X_aligned.reset_index(drop=True)
    y_df_for_split = y_df_aligned.reset_index(drop=True)
    y_for_split = y_df_for_split['label']

    logger.info(f"Aligned datasets. Number of samples: {len(X_for_split)}")
    logger.info(f"Label distribution: {y_for_split.value_counts().sort_index().to_dict()}")

    # --- 3. Create CV Splits ---
    cv = PurgedStratifiedKFold(
        n_splits=CONFIG['n_splits'],
        samples_info_sets=t1,
        pct_embargo=CONFIG['pct_embargo'],
        min_samples_per_class=CONFIG['min_samples_per_class']
    )
    cv_splits = list(cv.split(X_for_split, y_for_split))

    # --- 4. Run Optuna Optimization ---
    study = optuna.create_study(direction="maximize", study_name="financial_ml_model_v5")
    study.optimize(
        lambda trial: objective(trial, X_for_split, y_df_for_split, cv_splits),
        n_trials=CONFIG['n_trials'],
        n_jobs=CONFIG['n_jobs_optuna']
    )

    # --- 5. Save Results ---
    logger.success(f"Optimization completed! Best {CONFIG['primary_metric']} score: {study.best_value:.4f}")
    save_comprehensive_results(study, MODEL_DIR)

    # --- 6. Final Validation ---
    validation_results = final_model_validation(study, X_for_split, y_df_for_split, cv_splits)
    with open(MODEL_DIR / "final_validation_results_v5.json", "w") as f:
        json.dump(validation_results, f, indent=4, default=str)

    # --- 7. Train Final Model ---
    final_model_path = train_final_ensemble_model(X_for_split, y_df_for_split, study.best_params)

    logger.success("=== Optimization Pipeline (v5) Completed Successfully ===")
    logger.success(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main()
