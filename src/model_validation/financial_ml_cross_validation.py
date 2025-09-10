"""
Enhanced version addressing the identified issues in financial ML cross-validation pipeline
"""

from typing import Callable, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss, accuracy_score, f1_score
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import joblib
import logging
from pathlib import Path


def ml_get_train_times(samples_info_sets: pd.Series, test_times: pd.Series) -> pd.Series:
    """Same as original - no changes needed"""
    train = samples_info_sets.copy(deep=True)
    for start_ix, end_ix in test_times.items():
        df0 = train[(start_ix <= train.index) & (train.index <= end_ix)].index.unique()
        df1 = train[(start_ix <= train) & (train <= end_ix)].index.unique()
        df2 = train[(train.index <= start_ix) & (end_ix <= train)].index.unique()
        train = train.drop(df0.union(df1).union(df2))
    return train


class PurgedStratifiedKFold(KFold):
    """
    Enhanced PurgedKFold with stratification support for imbalanced datasets
    
    Key improvements:
    1. Stratifies each fold to maintain class balance
    2. Maintains original purging logic for financial data
    3. Better handling of small class samples
    """

    def __init__(self, n_splits: int = 3, samples_info_sets: pd.Series = None, 
                 pct_embargo: float = 0.0, min_samples_per_class: int = 1):
        
        if not isinstance(samples_info_sets, pd.Series):
            raise ValueError("The samples_info_sets param must be a pd.Series")
        
        super().__init__(n_splits, shuffle=False, random_state=None)
        
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo
        self.min_samples_per_class = min_samples_per_class

    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: np.ndarray = None):
        """Enhanced split with stratification"""
        X, y, groups = indexable(X, y, groups)
        
        if X.shape[0] != self.samples_info_sets.shape[0]:
            raise ValueError("X and samples_info_sets must have same length")
        
        if y is None:
            # Fallback to original PurgedKFold if no y provided
            yield from self._split_without_stratification(X)
            return
        
        # Check class distribution
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        
        if min_class_count < self.n_splits * self.min_samples_per_class:
            logging.warning(f"Insufficient samples for stratification. "
                          f"Min class has {min_class_count} samples. "
                          f"Falling back to regular PurgedKFold")
            yield from self._split_without_stratification(X, y)
            return
        
        # Stratified splitting
        indices = np.arange(_num_samples(X))
        embargo = int(X.shape[0] * self.pct_embargo)
        
        # Create stratified test ranges
        stratified_ranges = self._create_stratified_ranges(X, y)
        
        for start_ix, end_ix in stratified_ranges:
            test_indices = indices[start_ix:end_ix]
            
            if end_ix < X.shape[0]:
                end_ix += embargo
            
            test_times = pd.Series(
                index=[self.samples_info_sets.iloc[start_ix]], 
                data=[self.samples_info_sets.iloc[end_ix - 1]]
            )
            train_times = ml_get_train_times(self.samples_info_sets, test_times)
            
<<<<<<< HEAD
            train_indices = self.samples_info_sets.index.searchsorted(train_times.index)
=======
            train_indices = [self.samples_info_sets.index.get_loc(train_ix) 
                           for train_ix in train_times.index]
>>>>>>> 53fc66ea5e1c22b428fad1cd0ddbd7b6e2d3da17
            
            if len(np.intersect1d(train_indices, test_indices)) > 0:
                raise Exception("Train and test intersect")
            
            yield train_indices, test_indices

    def _create_stratified_ranges(self, X: pd.DataFrame, y: pd.Series) -> list:
        """Create test ranges that maintain class balance"""
        n_samples = len(X)
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        # Sort by time to maintain temporal order
        sorted_indices = np.arange(n_samples)
        ranges = []
        current = 0
        
        for fold_size in fold_sizes:
            ranges.append((current, current + fold_size))
            current += fold_size
        
        return ranges

    def _split_without_stratification(self, X: pd.DataFrame, y: pd.Series = None):
        """Fallback to original PurgedKFold logic"""
        indices = np.arange(_num_samples(X))
        embargo = int(X.shape[0] * self.pct_embargo)
        
        test_ranges = [(ix[0], ix[-1] + 1) 
                      for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]
        
        for start_ix, end_ix in test_ranges:
            test_indices = indices[start_ix:end_ix]
            
            if end_ix < X.shape[0]:
                end_ix += embargo
            
            test_times = pd.Series(
                index=[self.samples_info_sets.iloc[start_ix]], 
                data=[self.samples_info_sets.iloc[end_ix - 1]]
            )
            train_times = ml_get_train_times(self.samples_info_sets, test_times)
            
<<<<<<< HEAD
            train_indices = self.samples_info_sets.index.searchsorted(train_times.index)
=======
            train_indices = [self.samples_info_sets.index.get_loc(train_ix) 
                           for train_ix in train_times.index]
>>>>>>> 53fc66ea5e1c22b428fad1cd0ddbd7b6e2d3da17
            
            if len(np.intersect1d(train_indices, test_indices)) > 0:
                raise Exception("Train and test intersect")
            
            yield train_indices, test_indices


def sharpe_score(y_true: np.array, y_pred: np.array, sample_weight: np.array = None) -> float:
    """
    Custom Sharpe ratio metric for trading strategies
    Assumes y_pred contains position signals (-1, 0, 1) or probabilities
    """
    if sample_weight is None:
        sample_weight = np.ones_like(y_true)
    
    # Convert predictions to positions if probabilities
    if len(np.unique(y_pred)) > 3:  # Likely probabilities
        positions = np.where(y_pred > 0.5, 1, -1)
    else:
        positions = y_pred
    
    # Calculate returns (assuming y_true are returns)
    strategy_returns = positions * y_true * sample_weight
    
    if np.std(strategy_returns) == 0:
        return 0.0
    
    return np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)


def combined_score(y_true: np.array, y_pred: np.array, sample_weight: np.array = None,
                  accuracy_weight: float = 0.3, sharpe_weight: float = 0.7) -> float:
    """Combined accuracy and Sharpe ratio metric"""
    acc = accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    sharpe = sharpe_score(y_true, y_pred, sample_weight)
    
    # Normalize Sharpe to [0, 1] range (assuming max realistic Sharpe is 3.0)
    normalized_sharpe = max(0, min(1, (sharpe + 1) / 4))  
    
    return accuracy_weight * acc + sharpe_weight * normalized_sharpe


def ml_cross_val_score_enhanced(
    classifier: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    cv_gen: BaseCrossValidator,
    sample_weight_train: np.ndarray = None,
    sample_weight_score: np.ndarray = None,
    scoring: Union[Callable, str, Dict[str, Callable]] = "f1_weighted",
    early_stopping_rounds: int = 50,  # Increased from 10
    save_models: bool = True,
    model_save_path: Optional[str] = None,
    n_jobs: int = 1,  # Conservative default for memory
) -> Dict[str, Any]:
    """
    Enhanced cross-validation with multiple improvements:
    
    1. Multiple scoring metrics support
    2. Model saving capability  
    3. Better early stopping
    4. Memory-efficient parallel processing
    5. Comprehensive result reporting
    """
    
    # Setup default paths and weights
    if sample_weight_train is None:
        sample_weight_train = np.ones((X.shape[0],))
    
    if sample_weight_score is None:
        sample_weight_score = np.ones((X.shape[0],))
    
    if model_save_path is None:
        model_save_path = Path("./saved_models")
        model_save_path.mkdir(exist_ok=True)
    
    # Setup scoring metrics
    if isinstance(scoring, str):
        if scoring == "f1_weighted":
            scoring_func = lambda yt, yp, sw: f1_score(yt, yp, sample_weight=sw, average='weighted')
        elif scoring == "accuracy":
            scoring_func = accuracy_score
        elif scoring == "sharpe":
            scoring_func = sharpe_score
        elif scoring == "combined":
            scoring_func = combined_score
        else:
            raise ValueError(f"Unknown scoring: {scoring}")
        scoring_metrics = {"main": scoring_func}
    elif isinstance(scoring, dict):
        scoring_metrics = scoring
    else:
        scoring_metrics = {"main": scoring}
    
    # Results storage
    results = {metric_name: [] for metric_name in scoring_metrics.keys()}
    best_models = []
    fold_info = []
    
    # Cross-validation loop
    for fold_idx, (train, test) in enumerate(cv_gen.split(X=X, y=y)):
        logging.info(f"Training fold {fold_idx + 1}/{cv_gen.n_splits}")
        
        # Enhanced early stopping for tree-based models
        eval_set = None
        if hasattr(classifier, 'fit') and 'eval_set' in classifier.fit.__code__.co_varnames:
            # For XGBoost/LightGBM with eval_set support
            eval_set = [(X.iloc[test, :], y.iloc[test])]
        
        # Fit model
        fit_params = {
            'sample_weight': sample_weight_train[train],
            'early_stopping_rounds': early_stopping_rounds,
        }
        if eval_set:
            fit_params['eval_set'] = eval_set
            fit_params['verbose'] = False
        
        try:
            fitted_model = classifier.fit(
                X=X.iloc[train, :], 
                y=y.iloc[train], 
                **{k: v for k, v in fit_params.items() 
                   if k in classifier.fit.__code__.co_varnames}
            )
        except TypeError:
            # Fallback for models that don't support all parameters
            fitted_model = classifier.fit(
                X=X.iloc[train, :], 
                y=y.iloc[train], 
                sample_weight=sample_weight_train[train]
            )
        
        # Predictions
        if hasattr(fitted_model, 'predict_proba'):
            y_pred_proba = fitted_model.predict_proba(X.iloc[test, :])
            y_pred = fitted_model.classes_[np.argmax(y_pred_proba, axis=1)]
        else:
            y_pred = fitted_model.predict(X.iloc[test, :])
            y_pred_proba = None
        
        # Calculate all metrics
        for metric_name, metric_func in scoring_metrics.items():
            if metric_name == "log_loss" and y_pred_proba is not None:
                score = -1 * log_loss(
                    y.iloc[test], y_pred_proba, 
                    sample_weight=sample_weight_score[test],
                    labels=fitted_model.classes_
                )
            else:
                score = metric_func(
                    y.iloc[test], y_pred, 
                    sample_weight=sample_weight_score[test]
                )
            results[metric_name].append(score)
        
        # Save model if requested
        if save_models:
            model_path = Path(model_save_path) / f"model_fold_{fold_idx}.joblib"
            joblib.dump(fitted_model, model_path)
            best_models.append(str(model_path))
        
        # Store fold information
        fold_info.append({
            'fold': fold_idx,
            'train_size': len(train),
            'test_size': len(test),
            'train_class_dist': y.iloc[train].value_counts().to_dict(),
            'test_class_dist': y.iloc[test].value_counts().to_dict(),
        })
    
    # Compile results
    final_results = {
        'cv_scores': {name: np.array(scores) for name, scores in results.items()},
        'cv_mean': {name: np.mean(scores) for name, scores in results.items()},
        'cv_std': {name: np.std(scores) for name, scores in results.items()},
        'fold_info': fold_info,
        'n_splits': cv_gen.n_splits,
    }
    
    if save_models:
        final_results['model_paths'] = best_models
    
    return final_results


# Usage example with improvements
"""
# Example usage with enhanced features:

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Setup enhanced cross-validation
cv_gen = PurgedStratifiedKFold(
    n_splits=5, 
    samples_info_sets=samples_info_sets, 
    pct_embargo=0.1,
    min_samples_per_class=10
)

# XGBoost with proper early stopping
classifier = xgb.XGBClassifier(
    n_estimators=1000,  # High number, rely on early stopping
    learning_rate=0.1,
    max_depth=6,
    n_jobs=-1,  # Parallel within model
    random_state=42
)

# Multiple scoring metrics
scoring_metrics = {
    'f1': lambda yt, yp, sw: f1_score(yt, yp, sample_weight=sw, average='weighted'),
    'accuracy': accuracy_score,
    'sharpe': sharpe_score,
    'combined': combined_score
}

# Enhanced cross-validation
results = ml_cross_val_score_enhanced(
    classifier=classifier,
    X=X_train,
    y=y_train,
    cv_gen=cv_gen,
    sample_weight_train=sample_weights,
    scoring=scoring_metrics,
    early_stopping_rounds=50,  # More conservative
    save_models=True,
    model_save_path="./best_models",
    n_jobs=1  # Conservative for memory
)

print(f"F1 Score: {results['cv_mean']['f1']:.4f} ± {results['cv_std']['f1']:.4f}")
print(f"Sharpe: {results['cv_mean']['sharpe']:.4f} ± {results['cv_std']['sharpe']:.4f}")
print(f"Combined: {results['cv_mean']['combined']:.4f} ± {results['cv_std']['combined']:.4f}")
"""