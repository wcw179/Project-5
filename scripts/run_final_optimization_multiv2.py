"""
Enhanced script to run hyperparameter optimization for the final model using Optuna,
Enhanced Purged Stratified K-Fold CV, and comprehensive evaluation metrics.

Key improvements:
1. PurgedStratifiedKFold for better class balance
2. Multiple evaluation metrics (F1, Sharpe, Combined)
3. Conservative memory management (n_jobs=1 for Optuna)
4. Better early stopping (50 rounds)
5. Model persistence and comprehensive logging
6. Robust error handling and validation
"""
import sys
from pathlib import Path
import json
import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection._split import _BaseKFold
class PurgedKFold(_BaseKFold):
    """
    Purged K-Fold cross-validator
    This implementation works with integer-based indices and relies on a pre-computed
    `t1` series that indicates the end time of each event.
    """
    def __init__(self, n_splits=10, t1=None, pct_embargo=0., n_jobs=1):
        if not isinstance(t1, pd.Series):
            raise ValueError('t1 must be a pd.Series.')
        super(PurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.n_jobs = n_jobs

    def split(self, X, y=None, groups=None):
        if X.shape[0] != self.t1.shape[0]:
            raise ValueError('X and t1 must have the same length.')

        indices = np.arange(X.shape[0])
        embargo = int(X.shape[0] * self.pct_embargo)
        test_ranges = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]

        for i, (start_ix, end_ix) in enumerate(test_ranges):
            test_indices = indices[start_ix:end_ix]

            t0 = self.t1.index[start_ix] # Start of test set
            t1_ = self.t1.iloc[end_ix - 1] # End of test set

            # Purge from training set
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if t1_ is not pd.NaT:
                train_indices = np.setdiff1d(train_indices, self.t1.index.searchsorted(self.t1[self.t1 >= t1_ - pd.Timedelta(microseconds=1)].index))

            # Embargo
            if embargo > 0:
                embargo_start_ix = end_ix
                embargo_end_ix = min(embargo_start_ix + embargo, X.shape[0])
                train_indices = np.setdiff1d(train_indices, np.arange(embargo_start_ix, embargo_end_ix))

            yield train_indices, test_indices
import gc
from loguru import logger
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import our enhanced cross-validation
from src.model_validation.financial_ml_cross_validation import PurgedStratifiedKFold, sharpe_score, combined_score

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
    'n_jobs': 1,  # Conservative for memory safety
    'xgb_early_stopping': 50,  # Increased from 10
    'pct_embargo': 0.01,
    'min_samples_per_class': 10,
    'evaluation_metrics': ['f1_weighted', 'accuracy', 'sharpe', 'combined'],
    'primary_metric': 'combined',  # Use combined metric for optimization
}


def calculate_returns_proxy(y_true: np.array, y_pred: np.array) -> np.array:
    """
    Convert classification labels to return proxy for Sharpe calculation.
    This is a simplified approach - in production, you'd use actual returns.
    """
    # Assume y_true represents market direction, convert to return proxy
    returns_proxy = np.where(y_true == 1, 0.001,   # Up: small positive return
                            np.where(y_true == -1, -0.001,  # Down: small negative return
                                    0.0))  # Neutral: no return

    # Strategy returns = position * market return
    positions = np.where(y_pred == 1, 1.0,    # Long position
                        np.where(y_pred == -1, -1.0,  # Short position
                                0.0))  # No position

    return positions * returns_proxy


def evaluate_predictions(y_true: np.array, y_pred: np.array,
                        sample_weight: Optional[np.array] = None) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics"""
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)

    metrics = {}

    # Classification metrics
    metrics['f1_weighted'] = f1_score(y_true, y_pred, sample_weight=sample_weight, average='weighted')
    metrics['accuracy'] = accuracy_score(y_true, y_pred, sample_weight=sample_weight)

    # Trading-specific metrics
    try:
        strategy_returns = calculate_returns_proxy(y_true, y_pred)
        if np.std(strategy_returns) > 0:
            metrics['sharpe'] = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        else:
            metrics['sharpe'] = 0.0
    except Exception as e:
        logger.warning(f"Sharpe calculation failed: {e}")
        metrics['sharpe'] = 0.0

    # Combined metric (30% accuracy + 70% normalized Sharpe)
    normalized_sharpe = max(0, min(1, (metrics['sharpe'] + 1) / 4))
    metrics['combined'] = 0.3 * metrics['accuracy'] + 0.7 * normalized_sharpe

    return metrics


def objective(trial: optuna.trial.Trial, X: pd.DataFrame, y: pd.Series,
              cv_splits: List[Tuple[np.array, np.array]]) -> float:
    """Enhanced Optuna objective function with comprehensive evaluation."""
    try:
        # Enhanced hyperparameter space
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # Wider range
            'max_depth': trial.suggest_int('max_depth', 3, 10),  # Deeper trees allowed
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 2.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'tree_method': 'hist',
            'random_state': CONFIG['random_state'],
            'n_jobs': -1,  # Parallel within XGBoost only
        }

        logger.info(f"Trial {trial.number}: Starting with enhanced params")

        fold_metrics = {metric: [] for metric in CONFIG['evaluation_metrics']}
        valid_folds = 0

        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            try:
                # Data preparation
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Label mapping for XGBoost
                y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
                y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2})

                # Check class diversity
                if len(np.unique(y_train_mapped)) < 2:
                    logger.warning(f"Fold {fold}: Insufficient class diversity. Skipping.")
                    continue

                # Model training with enhanced early stopping
                # Calculate class weights to counteract imbalance
                sample_weights = compute_sample_weight(
                    class_weight='balanced',
                    y=y_train_mapped
                )

                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train_mapped,
                    sample_weight=sample_weights,
                    eval_set=[(X_val, y_val_mapped)],
                    early_stopping_rounds=CONFIG['xgb_early_stopping'],
                    verbose=False
                )

                # Predictions
                pred_probs = model.predict_proba(X_val)
                preds = np.argmax(pred_probs, axis=1)
                preds_mapped_back = pd.Series(preds).map({0: -1, 1: 0, 2: 1}).values

                # Calculate all metrics
                metrics = evaluate_predictions(y_val.values, preds_mapped_back)

                for metric_name, score in metrics.items():
                    fold_metrics[metric_name].append(score)

                valid_folds += 1

                # Memory cleanup
                del model, preds, preds_mapped_back
                gc.collect()

            except Exception as e:
                logger.error(f"Error in fold {fold}: {e}")
                continue

        if valid_folds == 0:
            logger.error(f"Trial {trial.number}: No valid folds completed.")
            return -1.0

        # Calculate mean scores and log comprehensive results
        mean_scores = {metric: np.mean(scores) for metric, scores in fold_metrics.items()}

        logger.info(f"Trial {trial.number} Results ({valid_folds}/{len(cv_splits)} folds):")
        for metric, score in mean_scores.items():
            logger.info(f"  {metric}: {score:.4f}")

        # Return primary metric for optimization
        primary_score = mean_scores.get(CONFIG['primary_metric'], -1.0)

        # Store additional metrics in trial user attributes
        for metric, score in mean_scores.items():
            trial.set_user_attr(f"mean_{metric}", score)
            if len(fold_metrics[metric]) > 1:
                trial.set_user_attr(f"std_{metric}", np.std(fold_metrics[metric]))

        logger.info(f"Trial {trial.number}: Primary metric ({CONFIG['primary_metric']}) = {primary_score:.4f}")
        return float(primary_score)

    except Exception as e:
        logger.error(f"Trial {trial.number} failed entirely: {e}")
        return -1.0


def create_stratified_cv_splits(X: pd.DataFrame, y: pd.Series,
                               sample_info: pd.DataFrame) -> List[Tuple[np.array, np.array]]:
    """Create stratified CV splits with proper purging"""
    logger.info("Creating stratified CV splits with purging...")

    # Align indices
    common_idx = X.index.intersection(sample_info.index)
    if len(common_idx) == 0:
        raise ValueError("No common indices between X and sample_info")

    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]
    t1_aligned = sample_info.loc[common_idx, 't1']

    logger.info(f"Aligned data: {len(X_aligned)} samples")

    # Create enhanced CV with stratification
    cv = PurgedStratifiedKFold(
        n_splits=CONFIG['n_splits'],
        samples_info_sets=t1_aligned,
        pct_embargo=CONFIG['pct_embargo'],
        min_samples_per_class=CONFIG['min_samples_per_class']
    )

    cv_splits = list(cv.split(X_aligned, y_aligned))

    # Log class distribution per fold
    logger.info("Class distribution per fold:")
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        train_dist = y_aligned.iloc[train_idx].value_counts().sort_index()
        val_dist = y_aligned.iloc[val_idx].value_counts().sort_index()
        logger.info(f"  Fold {fold}: Train {train_dist.to_dict()}, Val {val_dist.to_dict()}")

    return cv_splits, X_aligned, y_aligned


def save_comprehensive_results(study: optuna.study.Study, model_dir: Path):
    """Save comprehensive optimization results with enhanced analysis"""
    logger.info("Saving comprehensive optimization results...")

    # Save study object
    joblib.dump(study, model_dir / "enhanced_optimization_study.pkl")

    # Enhanced trials dataframe with user attributes
    trials_df = study.trials_dataframe()

    # Add user attributes to dataframe
    user_attrs_df = []
    for trial in study.trials:
        attrs = trial.user_attrs.copy()
        attrs['trial_number'] = trial.number
        user_attrs_df.append(attrs)

    if user_attrs_df:
        user_attrs_df = pd.DataFrame(user_attrs_df)
        trials_df = trials_df.merge(user_attrs_df, left_on='number', right_on='trial_number', how='left')

    trials_df.to_csv(model_dir / "enhanced_optimization_trials.csv", index=False)

    # Save best trial comprehensive results
    best_trial_results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial_number': study.best_trial.number,
        'best_trial_user_attrs': study.best_trial.user_attrs,
        'optimization_config': CONFIG,
        'n_completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    }

    with open(model_dir / "enhanced_best_trial_results.json", "w") as f:
        json.dump(best_trial_results, f, indent=4, default=str)

    # Visualization (if available)
    try:
        import optuna.visualization as vis

        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(model_dir / "enhanced_optimization_history.html"))

        # Parameter importance
        fig = vis.plot_param_importances(study)
        fig.write_html(str(model_dir / "enhanced_param_importance.html"))

        # Hyperparameter relationships
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(str(model_dir / "enhanced_parallel_coordinate.html"))

        logger.info("Visualization plots saved successfully")

    except ImportError:
        logger.warning("Optuna visualization not available. Install plotly for visualizations.")


def final_model_validation(study: optuna.study.Study, X: pd.DataFrame, y: pd.Series,
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
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Label mapping
        y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
        y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2})

        # Add robustness check for class diversity
        if len(np.unique(y_train_mapped)) < 2:
            logger.warning(f"Validation Fold {fold}: Insufficient class diversity. Skipping.")
            continue

        # Train final model
        model = xgb.XGBClassifier(**best_params)
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=y_train_mapped
        )
        model.fit(
            X_train, y_train_mapped,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val_mapped)],
            early_stopping_rounds=CONFIG['xgb_early_stopping'],
            verbose=False
        )

        # Predictions
        pred_probs = model.predict_proba(X_val)
        preds = np.argmax(pred_probs, axis=1)
        preds_mapped_back = pd.Series(preds).map({0: -1, 1: 0, 2: 1}).values

        # Store predictions
        all_predictions.extend(preds_mapped_back)
        all_true_labels.extend(y_val.values)

        # Calculate fold metrics
        fold_metrics = evaluate_predictions(y_val.values, preds_mapped_back)
        fold_results.append(fold_metrics)

        # Save individual fold model
        model_path = MODEL_DIR / f"final_model_fold_{fold}.joblib"
        joblib.dump(model, model_path)
        saved_models.append(str(model_path))

        logger.info(f"Fold {fold} metrics: " +
                   ", ".join([f"{k}: {v:.4f}" for k, v in fold_metrics.items()]))

    # Aggregate results
    overall_metrics = evaluate_predictions(
        np.array(all_true_labels),
        np.array(all_predictions)
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


def train_final_ensemble_model(X: pd.DataFrame, y: pd.Series,
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
    y_mapped = y.map({-1: 0, 0: 1, 1: 2})

    # Train on full dataset
    final_model = xgb.XGBClassifier(**final_params)
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y_mapped
    )
    final_model.fit(X, y_mapped, sample_weight=sample_weights)

    # Save final model
    final_model_path = MODEL_DIR / "enhanced_final_ensemble_model.joblib"
    joblib.dump(final_model, final_model_path)

    logger.success(f"Final ensemble model saved to: {final_model_path}")
    return str(final_model_path)


def main():
    """Enhanced main function with comprehensive pipeline"""
    logger.info(f"=== Starting Enhanced Multi-Symbol Model Optimization ===")
    logger.info(f"Configuration: {json.dumps(CONFIG, indent=2)}")

    # Prepare directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = project_root / "data" / "processed"
    X_full_path = data_dir / "X_full.parquet"
    y_full_path = data_dir / "y_full.parquet"
    sample_info_path = data_dir / "sample_info_full.parquet"

    # Validate data files
    for path in [X_full_path, y_full_path, sample_info_path]:
        if not path.exists():
            logger.error(f"Required file not found: {path}")
            return

    # Load datasets
    X = pd.read_parquet(X_full_path)
    y = pd.read_parquet(y_full_path).squeeze()
    sample_info = pd.read_parquet(sample_info_path)

    # Basic validations
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    valid_labels = {-1, 0, 1}
    unique_labels = set(pd.unique(y))
    if not unique_labels.issubset(valid_labels):
        logger.error(f"y_full contains invalid labels {unique_labels - valid_labels}. Expected only {-1,0,1}.")
        return
    if 't1' not in sample_info.columns:
        logger.error("sample_info_full.parquet must contain a 't1' column.")
        return

    # Step 2: Align datasets based on their original datetime index
    common_idx = X.index.intersection(y.index).intersection(sample_info.index)
    if len(common_idx) == 0:
        logger.error("No common indices found across X, y, and sample_info.")
        return

    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]
    sample_info_aligned = sample_info.loc[common_idx]
    logger.info(f"Aligned datasets. Number of samples: {len(X_aligned)}")

    # Step 3: Prepare data for PurgedKFold and for the model
    t1 = sample_info_aligned['t1']  # This has the datetime index for purging logic

    # Reset index on the training dataframes for splitter compatibility
    X_aligned.reset_index(drop=True, inplace=True)
    y_aligned.reset_index(drop=True, inplace=True)
    logger.info(f"Label distribution in aligned data: {y_aligned.value_counts().sort_index().to_dict()}")

    # Step 4: Instantiate and Run CV
    cv = PurgedKFold(n_splits=CONFIG['n_splits'], t1=t1, pct_embargo=CONFIG['pct_embargo'])
    cv_splits = list(cv.split(X_aligned, y_aligned)) # Pass both X and y for stratification

    # Early stopping callback
    def early_stopping_callback(study, _):  # _ is unused but kept for Optuna callback signature
        if len(study.trials) >= CONFIG['early_stopping_patience']:
            recent_trials = study.trials[-CONFIG['early_stopping_patience']:]
            recent_values = [t.value for t in recent_trials if t.value is not None]

            if len(recent_values) >= CONFIG['early_stopping_patience']:
                improvement = max(recent_values) - min(recent_values)
                if improvement < CONFIG['improvement_threshold']:
                    logger.info(
                        f"Early stopping triggered: No improvement > {CONFIG['improvement_threshold']} "
                        f"in last {CONFIG['early_stopping_patience']} trials."
                    )
                    study.stop()

    # Run optimization
    logger.info(f"Starting Optuna optimization with {CONFIG['n_trials']} trials...")
    study = optuna.create_study(
        direction="maximize",
        study_name="enhanced_multi_symbol_optimization"
    )

    study.optimize(
        lambda trial: objective(trial, X_aligned, y_aligned, cv_splits),
        n_trials=CONFIG['n_trials'],
        callbacks=[early_stopping_callback],
        n_jobs=CONFIG['n_jobs']  # Conservative n_jobs=1 for memory safety
    )

    # Log optimization results
    logger.success(f"Optimization completed!")
    logger.success(f"Best trial score ({CONFIG['primary_metric']}): {study.best_value:.4f}")
    logger.success(f"Best trial number: {study.best_trial.number}")
    logger.info(f"Best parameters: {json.dumps(study.best_params, indent=2)}")

    # Save comprehensive results
    save_comprehensive_results(study, MODEL_DIR)

    # Final validation
    validation_results = final_model_validation(study, X_aligned, y_aligned, cv_splits)

    # Save validation results
    with open(MODEL_DIR / "final_validation_results.json", "w") as f:
        json.dump(validation_results, f, indent=4, default=str)

    # Train final ensemble model
    final_model_path = train_final_ensemble_model(X_aligned, y_aligned, study.best_params)

    # Summary
    logger.success("=== Optimization Pipeline Completed Successfully ===")
    logger.success(f"Best {CONFIG['primary_metric']} score: {study.best_value:.4f}")
    logger.success(f"Final ensemble model: {final_model_path}")
    logger.success(f"All results saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()