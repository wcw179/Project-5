"""
Script to run hyperparameter optimization for the final model using Optuna,
Purged K-Fold CV, and the full feature/label generation pipeline.
Improved version with better error handling and validation.
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
import warnings
warnings.filterwarnings('ignore')

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from mlfinpy.cross_validation import PurgedKFold
except ImportError:
    logger.warning("mlfinpy not found, trying alternative import")
    try:
        from scripts.purged_cross_validation import PurgedKFold
    except ImportError:
        logger.error("PurgedKFold not found in any expected location")
        sys.exit(1)

# --- Logger Configuration ---
log_file_path = project_root / "logs" / "final_optimization_multi.log"
log_file_path.parent.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>")
logger.add(log_file_path, rotation="10 MB", retention="10 days", enqueue=True, serialize=False)

# --- Constants & Config ---
MODEL_DIR = project_root / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'n_trials': 100,
    'n_splits': 3,  # Reduced for stability
    'early_stopping_patience': 20,
    'improvement_threshold': 0.001,
    'random_state': 42,
    'n_jobs': 1,  # Changed to 1 to avoid threading issues with XGBoost
    'min_samples_per_fold': 50,  # Minimum samples required per fold
}

def validate_fold_data(y_train_mapped: pd.Series, y_val_mapped: pd.Series, fold: int) -> bool:
    """Validate that fold has sufficient data quality for training."""
    try:
        # Check minimum sample size
        if len(y_train_mapped) < CONFIG['min_samples_per_fold'] or len(y_val_mapped) < 10:
            logger.warning(f"Fold {fold}: Insufficient samples (train={len(y_train_mapped)}, val={len(y_val_mapped)})")
            return False
        
        # Check class diversity in training set
        train_classes = len(np.unique(y_train_mapped))
        if train_classes < 2:
            logger.warning(f"Fold {fold}: Insufficient class diversity in training data ({train_classes} classes)")
            return False
        
        # Check that validation set has at least 2 classes for meaningful F1 score
        val_classes = len(np.unique(y_val_mapped))
        if val_classes < 2:
            logger.warning(f"Fold {fold}: Insufficient class diversity in validation data ({val_classes} classes)")
            return False
        
        # Check class distribution balance (no class should be >95% of data)
        train_dist = y_train_mapped.value_counts(normalize=True)
        if train_dist.max() > 0.95:
            logger.warning(f"Fold {fold}: Highly imbalanced training data (max class: {train_dist.max():.2%})")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating fold {fold}: {e}")
        return False

def objective(trial: optuna.trial.Trial, X: pd.DataFrame, y: pd.Series, cv_splits: list) -> float:
    """Optuna objective function to train and evaluate models using F1-score."""
    try:
        params = {
            'objective': 'multi:softmax',  # Changed back to softmax for stability
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),  # Reduced upper bound
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 2),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'tree_method': 'hist',
            'random_state': CONFIG['random_state'],
            'verbosity': 0,  # Suppress XGBoost warnings
        }

        logger.debug(f"Trial {trial.number}: Starting with {len(cv_splits)} folds")
        scores = []
        valid_folds = 0
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            try:
                # Ensure indices are valid
                train_idx = np.array(train_idx)
                val_idx = np.array(val_idx)
                
                # Check for index out of bounds
                if len(train_idx) == 0 or len(val_idx) == 0:
                    logger.warning(f"Fold {fold}: Empty indices")
                    continue
                
                if train_idx.max() >= len(X) or val_idx.max() >= len(X):
                    logger.error(f"Fold {fold}: Index out of bounds")
                    continue

                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Map labels to XGBoost format
                y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
                y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2})

                # Check for NaN values after mapping
                if y_train_mapped.isna().any() or y_val_mapped.isna().any():
                    logger.warning(f"Fold {fold}: NaN values found after label mapping")
                    continue

                # Validate fold data quality
                if not validate_fold_data(y_train_mapped, y_val_mapped, fold):
                    continue

                # Train model with error handling
                model = xgb.XGBClassifier(**params)
                
                # Fit with early stopping but handle potential errors
                try:
                    model.fit(
                        X_train, y_train_mapped,
                        eval_set=[(X_val, y_val_mapped)],
                        early_stopping_rounds=10,
                        verbose=False
                    )
                except Exception as fit_error:
                    logger.error(f"Fold {fold}: Model fitting failed - {fit_error}")
                    continue

                # Make predictions
                preds = model.predict(X_val)
                
                # Map predictions back to original labels
                pred_mapping = {0: -1, 1: 0, 2: 1}
                preds_mapped_back = np.array([pred_mapping.get(p, 0) for p in preds])

                # Calculate F1 score with error handling
                try:
                    score = f1_score(y_val.values, preds_mapped_back, average='weighted', zero_division=0)
                    if np.isnan(score) or np.isinf(score):
                        logger.warning(f"Fold {fold}: Invalid F1 score")
                        continue
                    scores.append(score)
                    valid_folds += 1
                    logger.debug(f"Fold {fold}: F1 score = {score:.4f}")
                except Exception as score_error:
                    logger.error(f"Fold {fold}: F1 score calculation failed - {score_error}")
                    continue

                # Cleanup
                del model, preds, preds_mapped_back
                
            except Exception as fold_error:
                logger.error(f"Fold {fold}: Unexpected error - {fold_error}")
                continue
            finally:
                gc.collect()

        # Validate results
        if not scores or valid_folds == 0:
            logger.error(f"Trial {trial.number}: No valid scores obtained from any fold.")
            return 0.0  # Return neutral score instead of negative

        if valid_folds < CONFIG['n_splits'] // 2:  # Less than half the folds succeeded
            logger.warning(f"Trial {trial.number}: Only {valid_folds}/{CONFIG['n_splits']} folds succeeded")

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores)) if len(scores) > 1 else 0.0
        
        logger.info(f"Trial {trial.number}: {valid_folds} valid folds, F1 = {mean_score:.4f} ± {std_score:.4f}")
        return mean_score

    except Exception as e:
        logger.error(f"Trial {trial.number} failed entirely: {e}")
        return 0.0  # Return neutral score

def create_samples_info_sets_from_data(X: pd.DataFrame) -> pd.Series:
    """Create sample info sets if not available."""
    try:
        # Check for datetime columns
        datetime_cols = []
        for col in X.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(X[col])
                    datetime_cols.append(col)
                except:
                    continue
        
        if datetime_cols:
            dates = pd.to_datetime(X[datetime_cols[0]])
        elif isinstance(X.index, pd.DatetimeIndex):
            dates = X.index
        else:
            # Create sequential timestamps
            logger.warning("No datetime information found, creating sequential timestamps")
            dates = pd.date_range(start='2023-01-01', periods=len(X), freq='H')
        
        # Create info sets (assuming 1-hour lookahead for each sample)
        t1 = dates + pd.Timedelta(hours=1)
        samples_info_sets = pd.Series(data=t1.values, index=dates.values)
        
        return samples_info_sets
        
    except Exception as e:
        logger.error(f"Error creating samples_info_sets: {e}")
        # Return dummy series
        dates = pd.date_range(start='2023-01-01', periods=len(X), freq='H')
        t1 = dates + pd.Timedelta(hours=1)
        return pd.Series(data=t1.values, index=dates.values)

def setup_cross_validation(X: pd.DataFrame, data_dir: Path) -> list:
    """Setup cross-validation splits with proper error handling."""
    try:
        # Try to load sample info
        sample_info_path = data_dir / "sample_info_full.parquet"
        
        if sample_info_path.exists():
            logger.info("Loading sample info from file")
            sample_info = pd.read_parquet(sample_info_path)
            
            # Ensure alignment
            common_idx = X.index.intersection(sample_info.index)
            if len(common_idx) == 0:
                raise ValueError("No common indices between X and sample_info")
            
            # Use only common indices
            X_aligned = X.loc[common_idx]
            t1 = sample_info.loc[common_idx, 't1']
            
            logger.info(f"Using {len(common_idx)} aligned samples")
            
        else:
            logger.warning("Sample info file not found, creating from data")
            samples_info_sets = create_samples_info_sets_from_data(X)
            t1 = samples_info_sets
            X_aligned = X
        
        # Create PurgedKFold with error handling
        try:
            cv = PurgedKFold(n_splits=CONFIG['n_splits'], t1=t1, pct_embargo=0.01)
            cv_splits = list(cv.split(X_aligned))
            
            logger.info(f"Created {len(cv_splits)} cross-validation splits")
            
            # Validate splits
            valid_splits = []
            for i, (train_idx, val_idx) in enumerate(cv_splits):
                if len(train_idx) >= CONFIG['min_samples_per_fold'] and len(val_idx) >= 10:
                    valid_splits.append((train_idx, val_idx))
                else:
                    logger.warning(f"Split {i} has insufficient samples, skipping")
            
            if len(valid_splits) == 0:
                raise ValueError("No valid cross-validation splits created")
            
            logger.info(f"Using {len(valid_splits)} valid splits")
            return valid_splits, X_aligned
            
        except Exception as cv_error:
            logger.error(f"PurgedKFold failed: {cv_error}")
            raise
            
    except Exception as e:
        logger.error(f"Cross-validation setup failed: {e}")
        raise

def save_optimization_results(study: optuna.study.Study, model_dir: Path):
    """Save detailed optimization results."""
    try:
        # Save study object
        joblib.dump(study, model_dir / "optimization_study.pkl")
        
        # Save trials dataframe
        trials_df = study.trials_dataframe()
        trials_df.to_csv(model_dir / "optimization_trials.csv", index=False)
        
        # Save summary statistics
        summary = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'study_direction': 'maximize'
        }
        
        with open(model_dir / "optimization_summary.json", 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Try to save visualizations
        try:
            import optuna.visualization as vis
            fig = vis.plot_optimization_history(study)
            fig.write_html(str(model_dir / "optimization_history.html"))
            
            fig_param = vis.plot_param_importances(study)
            fig_param.write_html(str(model_dir / "param_importances.html"))
            
        except ImportError:
            logger.warning("Optuna visualization not available")
        except Exception as viz_error:
            logger.warning(f"Visualization creation failed: {viz_error}")
            
        logger.success("Optimization results saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving optimization results: {e}")

def validate_best_model(study: optuna.study.Study, X: pd.DataFrame, y: pd.Series, cv_splits: list):
    """Validate the best model with detailed metrics."""
    logger.info("--- Starting Final Validation of Best Model ---")
    
    try:
        best_params = study.best_params.copy()
        best_params.update({
            'objective': 'multi:softmax',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'random_state': CONFIG['random_state'],
            'verbosity': 0,
        })

        all_y_true, all_y_pred = [], []
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            try:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                y_train_mapped = y_train.map({-1: 0, 0: 1, 1: 2})
                y_val_mapped = y_val.map({-1: 0, 0: 1, 1: 2})

                if not validate_fold_data(y_train_mapped, y_val_mapped, fold):
                    continue

                model = xgb.XGBClassifier(**best_params)
                model.fit(
                    X_train, y_train_mapped,
                    eval_set=[(X_val, y_val_mapped)],
                    early_stopping_rounds=10,
                    verbose=False
                )

                preds = model.predict(X_val)
                pred_mapping = {0: -1, 1: 0, 2: 1}
                preds_mapped_back = np.array([pred_mapping.get(p, 0) for p in preds])

                fold_score = f1_score(y_val.values, preds_mapped_back, average='weighted', zero_division=0)
                fold_scores.append(fold_score)

                all_y_true.extend(y_val.values)
                all_y_pred.extend(preds_mapped_back)
                
                logger.info(f"Validation fold {fold}: F1 = {fold_score:.4f}")
                
            except Exception as e:
                logger.error(f"Validation fold {fold} failed: {e}")
                continue

        if all_y_true and all_y_pred:
            logger.info("--- Final Validation Results ---")
            logger.info(f"Cross-validation F1 scores: {[f'{s:.4f}' for s in fold_scores]}")
            logger.info(f"Mean CV F1: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
            
            report = classification_report(all_y_true, all_y_pred, digits=4)
            logger.info(f"\nOverall Classification Report:\n{report}")
            
            # Save validation results
            validation_results = {
                'cv_scores': fold_scores,
                'mean_cv_score': float(np.mean(fold_scores)),
                'std_cv_score': float(np.std(fold_scores)),
                'classification_report': report
            }
            
            with open(MODEL_DIR / "validation_results.json", 'w') as f:
                json.dump(validation_results, f, indent=4, default=str)
                
        else:
            logger.error("No validation results obtained")
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")

def main():
    """Main function to run the optimization study."""
    logger.info(f"--- Starting Multi-Symbol Model Optimization ({CONFIG['n_trials']} Trials) ---")

    try:
        # Check data availability
        data_dir = project_root / "data" / "processed"
        X_full_path = data_dir / "X_multi_symbol.parquet"  # Updated path
        y_full_path = data_dir / "y_multi_symbol.parquet"  # Updated path

        if not X_full_path.exists() or not y_full_path.exists():
            logger.error("Multi-symbol dataset files not found. Please run the pipeline script first.")
            logger.info(f"Looking for: {X_full_path} and {y_full_path}")
            return

        # Load data
        X = pd.read_parquet(X_full_path)
        y = pd.read_parquet(y_full_path).squeeze()
        logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
        
        # Basic data validation
        if X.empty or y.empty:
            logger.error("Empty dataset loaded")
            return
        
        if len(X) != len(y):
            logger.error(f"Shape mismatch: X={len(X)}, y={len(y)}")
            return
        
        # Check class distribution
        class_dist = y.value_counts()
        logger.info(f"Class distribution: {class_dist.to_dict()}")
        
        if len(class_dist) < 2:
            logger.error("Insufficient class diversity in target variable")
            return

        # Setup cross-validation
        cv_splits, X_aligned = setup_cross_validation(X, data_dir)
        y_aligned = y.loc[X_aligned.index]
        
        logger.info(f"Using {len(X_aligned)} aligned samples for optimization")

        # Early stopping callback
        def early_stopping_callback(study, _):
            if len(study.trials) >= CONFIG['early_stopping_patience']:
                recent_trials = study.trials[-CONFIG['early_stopping_patience']:]
                recent_values = [t.value for t in recent_trials if t.value is not None and t.value > 0]

                if len(recent_values) >= CONFIG['early_stopping_patience']:
                    improvement = max(recent_values) - min(recent_values)
                    if improvement < CONFIG['improvement_threshold']:
                        logger.info(f"Early stopping: No significant improvement in last {CONFIG['early_stopping_patience']} trials.")
                        study.stop()

        # Run optimization
        study = optuna.create_study(direction="maximize", study_name="multi_symbol_optimization")
        study.optimize(
            lambda trial: objective(trial, X_aligned, y_aligned, cv_splits),
            n_trials=CONFIG['n_trials'],
            callbacks=[early_stopping_callback],
            n_jobs=CONFIG['n_jobs']
        )

        # Log results
        logger.info(f"Optimization completed with {len(study.trials)} trials")
        logger.info(f"Best trial score: {study.best_value:.4f}")
        logger.info(f"Best trial params: {study.best_params}")

        # Save results
        params_path = MODEL_DIR / "multi_symbol_final_best_params.json"
        with open(params_path, "w") as f:
            json.dump(study.best_params, f, indent=4)
        logger.success(f"Saved best hyperparameters to {params_path}")

        save_optimization_results(study, MODEL_DIR)
        validate_best_model(study, X_aligned, y_aligned, cv_splits)
        
        logger.success("Optimization pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Optimization pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()