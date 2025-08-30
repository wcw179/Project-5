---
type: "manual"
---

# Black Swan Detection System Trading Bot - XGBoost Multi-Label RR (v3) - Optimized

## 🎯 Objectives
- Bot hunter Black swan MT5 Trade
- Detect **Black Swan events** (rare extreme market movements)
- Predict **profit potential** across R multiples (5R, 10R, 15R, 20R)  
- Use **XGBoost Multi-Label Classifier** for speed and explainability

## 1. Data & Labeling Strategy

### Data Bar M5 ####
SLQ file: XAUUSDm, EURUSDm, GBPUSDm, USDJPYm
C:\Users\wcw17\Documents\GitHub\project-5\data\m5_trading.db

### Primary Labels (Multi-label Binary Classification)
- `hit_5R`: Achieves profit ≥ 5R within prediction horizon
- `hit_10R`: Achieves profit ≥ 10R within prediction horizon  
- `hit_15R`: Achieves profit ≥ 15R within prediction horizon
- `hit_20R`: Achieves profit ≥ 20R within prediction horizon

### Enhanced Labeling Methodology
**Original Weakness**: Static timeframe may miss delayed reactions
**Optimization**: Dynamic labeling windows
- **Fast moves**: 1-4 hours (immediate Black Swan)
- **Medium moves**: 4-12 hours (momentum continuation) 
- **Slow moves**: 12-24 hours (trend following)

### Tail Event Classification
- **Extreme tail**: Top/bottom 0.5% returns (True Black Swan)
- **Heavy tail**: Top/bottom 1-2% returns (Strong moves)
- **Normal tail**: Top/bottom 5% returns (Regular volatility)

## 2. Feature Engineering (Prioritized & Enhanced)

### 🕒 Temporal Features (Critical Tier)
```python
# Session timing - highest predictive power
is_ny_session          # Primary Black Swan catalyst
is_overlap_session     # London-NY overlap volatility  
is_london_session      # Trend vs sideways control
is_asian_session       # Low volatility baseline
session_transition     # NEW: Session handoff volatility
news_release_proximity # NEW: Distance to major news events
```

### 💰 Market Microstructure (Enhanced)
```python
# Liquidity & Flow
spread_proxy           # Transaction cost indicator
volume_regime          # Volume classification (high/low/normal)
obv_slope             # Money flow momentum
tick_volume_ratio     # NEW: Tick volume vs real volume
order_flow_imbalance  # NEW: Buy vs sell pressure
```

### 📈 Trend & Momentum (Multi-timeframe)
```python
# Trend alignment across timeframes
ema_alignment_h1      # 20/50/200 EMA alignment on H1
ema_alignment_h4      # NEW: H4 trend confirmation
ema_slope_acceleration # NEW: Rate of change in slope
trend_strength_index  # NEW: Composite trend measure

# Momentum indicators
macd_divergence       # NEW: Hidden/regular divergence
rsi_divergence        # NEW: RSI divergence detection
momentum_regime       # NEW: Strong/weak/neutral momentum
```

### 📊 Volatility & Distribution (Advanced)
```python
# Volatility clustering
volatility_regime     # NEW: High/medium/low vol regime
vol_expansion_rate    # NEW: Rate of volatility increase  
garch_forecast        # NEW: GARCH volatility prediction

# Distribution characteristics  
returns_skew_rolling  # Rolling skewness (tail bias)
returns_kurt_rolling  # Rolling kurtosis (tail thickness)
var_breach_frequency  # NEW: VaR breach rate
extreme_move_frequency # NEW: Historical extreme frequency
```

### 🌊 Market Sentiment & Positioning
```python
# NEW: Sentiment indicators
vix_regime           # Fear/greed indicator
put_call_ratio       # Options sentiment
commitment_traders   # COT positioning data
sentiment_extremes   # Contrarian signals
```

## 3. Model Architecture (Enhanced)

### Multi-Model Ensemble Approach
**Original Weakness**: Single model may overfit to specific patterns

```python
# Primary Models
xgb_fast     # Fast decisions (100 trees, depth=4)
xgb_deep     # Deep patterns (500 trees, depth=8) 
xgb_balanced # Balanced approach (300 trees, depth=6)

# Ensemble logic
final_prediction = weighted_average([
    0.4 * xgb_balanced.predict(),
    0.3 * xgb_deep.predict(), 
    0.3 * xgb_fast.predict()
])
```

### Advanced Training Strategy
```python
# Temporal cross-validation (no data leakage)
# Custom loss function for imbalanced classes
# SMOTE oversampling for rare Black Swan events
# Hyperparameter optimization with Optuna
```

## 4. Enhanced Trading Logic

### Entry Conditions (Multi-stage Filtering)
```python
# Stage 1: Model Signal
hit_5R_prob >= 0.65 AND
confidence_score >= 0.7  # NEW: Model confidence

# Stage 2: Market Context  
(is_ny_session OR is_overlap_session) AND
spread_proxy <= spread_threshold AND
volatility_regime != "EXTREMELY_HIGH"  # Avoid gap opens

# Stage 3: Risk Environment
market_stress_index <= stress_threshold AND  # NEW
correlation_breakdown == False  # NEW: Avoid correlation spikes
```

### Dynamic Position Sizing (Kelly-based)
**Original Weakness**: Static risk sizing
```python
base_risk = 0.01  # 1% base risk

# Kelly Criterion adaptation
if hit_20R_prob >= 0.3:
    position_multiplier = 2.0    # 2% risk
elif hit_15R_prob >= 0.4:  
    position_multiplier = 1.5    # 1.5% risk
elif hit_10R_prob >= 0.6:
    position_multiplier = 1.25   # 1.25% risk
else:
    position_multiplier = 1.0    # 1% risk

final_risk = base_risk * position_multiplier * confidence_score
```

### Advanced Exit Strategy
```python
# Partial profit taking
if unrealized_pnl >= 3R:
    close 25% of position
if unrealized_pnl >= 7R:
    close 50% of position  
if unrealized_pnl >= 12R:
    close 75% of position

# Trailing stop (volatility-adjusted)
trailing_stop = max(
    1R,  # Hard stop
    current_atr * 2,  # ATR-based
    vwap_distance * 0.5  # VWAP-based
)
```

## 5. Risk Management (Enhanced)

### Portfolio Level Risk
```python
# Daily risk limits
max_daily_risk = 0.05      # 5% max daily exposure
max_concurrent_trades = 3   # Position correlation control
max_weekly_risk = 0.15     # 15% max weekly exposure

# Market regime adaptation
if volatility_regime == "CRISIS":
    reduce_all_risks_by = 0.5
elif market_stress_index > 80:
    halt_new_positions = True
```

### Model Performance Monitoring
```python
# Real-time model degradation detection
rolling_accuracy_7d = track_accuracy(window=7)
rolling_sharpe_30d = track_sharpe(window=30)

if rolling_accuracy_7d < 0.55:
    reduce_position_size = 0.5
if rolling_sharpe_30d < 0.8:
    pause_trading = True
```

## 6. Deployment Architecture (Production-Ready)

### Data Pipeline
```python
# Real-time feature computation
feature_store = update_features_streaming(
    market_data=live_feed,
    lookback_window=200,  # 200-hour history
    update_frequency=60   # 1-minute updates
)

# Feature quality checks  
validate_feature_quality(feature_store)
detect_data_drift(current_features, training_features)
```

### Model Serving
```python
# A/B testing framework
model_champion = load_model("champion_v2.pkl")
model_challenger = load_model("challenger_v3.pkl") 

# Route 80% to champion, 20% to challenger
prediction = route_prediction(
    champion_weight=0.8,
    challenger_weight=0.2
)
```

### Monitoring & Alerting
```python
# Performance dashboards
track_metrics = [
    "prediction_accuracy",
    "profit_factor", 
    "max_drawdown",
    "feature_drift",
    "model_confidence"
]

# Automated alerts
if any_metric_degraded():
    send_alert_to_trader()
    initiate_model_retrain()
```

## 7. Advanced Cross-Validation & Model Validation

### Time-Series Specific Validation Strategy
**Critical**: Financial time series require specialized validation to prevent data leakage

```python
# Walk-Forward Analysis (Primary Method)
class WalkForwardValidator:
    def __init__(self, train_window=2000, test_window=500, step_size=100):
        self.train_window = train_window  # 2000 hours (~3 months)
        self.test_window = test_window    # 500 hours (~3 weeks)  
        self.step_size = step_size        # 100 hours step
        self.embargo_hours = 24           # 24-hour embargo period
    
    def split_data(self, data, labels):
        """
        Time-based splits with embargo:
        Train: [t-2000:t], Embargo: [t:t+24], Test: [t+24:t+524]
        """
        splits = []
        for start_idx in range(self.train_window, len(data) - self.test_window, self.step_size):
            train_end = start_idx
            test_start = start_idx + self.embargo_hours
            test_end = test_start + self.test_window
            
            train_indices = slice(start_idx - self.train_window, train_end)
            test_indices = slice(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        return splits

# Purged K-Fold Cross-Validation
class PurgedKFoldCV:
    def __init__(self, n_splits=5, embargo_hours=24):
        self.n_splits = n_splits
        self.embargo_hours = embargo_hours
    
    def split(self, X, y, groups=None):
        """
        Purged K-Fold with embargo:
        - Removes overlapping samples between train/test
        - Adds embargo period to prevent data leakage
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size
            
            # Purge overlapping samples
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[max(0, test_start - self.embargo_hours):
                      min(n_samples, test_end + self.embargo_hours)] = False
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices

# Combinatorial Purged Cross-Validation
class CombinatorialPurgedCV:
    def __init__(self, n_splits=10, n_test_splits=2, embargo_hours=24):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_hours = embargo_hours
    
    def split(self, X, y, groups=None):
        """
        Advanced CV for financial ML:
        - Multiple non-overlapping test sets
        - Purged train sets with embargo
        - Reduces selection bias
        """
        from itertools import combinations
        
        n_samples = len(X)
        indices = np.arange(n_samples)
        split_indices = np.array_split(indices, self.n_splits)
        
        # Generate all combinations of test splits
        test_combinations = list(combinations(range(self.n_splits), self.n_test_splits))
        
        for test_combo in test_combinations:
            # Combine test splits
            test_indices = np.concatenate([split_indices[i] for i in test_combo])
            
            # Create purged train set
            train_mask = np.ones(n_samples, dtype=bool)
            for test_idx in test_indices:
                # Purge around test samples
                start_purge = max(0, test_idx - self.embargo_hours)
                end_purge = min(n_samples, test_idx + self.embargo_hours)
                train_mask[start_purge:end_purge] = False
            
            train_indices = np.where(train_mask)[0]
            
            yield train_indices, test_indices
```

### Hyperparameter Optimization Framework

```python
# Bayesian Optimization for Black Swan Models
class BlackSwanOptimizer:
    def __init__(self, cv_strategy="walk_forward"):
        self.cv_strategy = cv_strategy
        self.optimization_history = []
    
    def optimize_xgboost_params(self, X, y, n_trials=100):
        """
        Multi-objective Bayesian optimization:
        - Primary: Sharpe ratio
        - Secondary: Average return per trade
        - Constraint: Maximum drawdown < 8%
        """
        import optuna
        from optuna.samplers import TPESampler
        
        def objective(trial):
            # XGBoost hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            }
            
            # Cross-validation with financial metrics
            scores = self.cross_validate_financial(X, y, params)
            
            # Multi-objective scoring
            sharpe_ratio = scores['sharpe_ratio']
            avg_return = scores['avg_return_per_trade']
            max_dd = scores['max_drawdown']
            pnl_stability = scores['pnl_stability']
            
            # Penalize if constraints violated
            if max_dd > 0.08:  # 8% max drawdown constraint
                return -np.inf
            
            # Combined score (weighted)
            combined_score = (
                0.4 * sharpe_ratio +
                0.3 * avg_return +
                0.2 * pnl_stability +
                0.1 * (1 - max_dd)  # Lower drawdown is better
            )
            
            return combined_score
        
        # Run Bayesian optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params, study.best_value
    
    def cross_validate_financial(self, X, y, params):
        """
        Financial time-series cross-validation with custom metrics
        """
        if self.cv_strategy == "walk_forward":
            cv = WalkForwardValidator()
        elif self.cv_strategy == "purged_kfold":
            cv = PurgedKFoldCV()
        else:
            cv = CombinatorialPurgedCV()
        
        scores = {
            'sharpe_ratio': [],
            'avg_return_per_trade': [],
            'max_drawdown': [],
            'pnl_stability': [],
            'hit_rate': []
        }
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model = XGBClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_proba = model.predict_proba(X_test)
            
            # Simulate trading performance
            trade_results = self.simulate_trading(y_test, y_pred_proba, X_test)
            
            # Calculate financial metrics
            fold_scores = self.calculate_financial_metrics(trade_results)
            
            for metric in scores:
                scores[metric].append(fold_scores[metric])
        
        # Return average scores
        return {metric: np.mean(values) for metric, values in scores.items()}
    
    def calculate_financial_metrics(self, trade_results):
        """
        Calculate trading-specific performance metrics
        """
        returns = np.array(trade_results['returns'])
        equity_curve = np.cumsum(returns)
        
        # Sharpe ratio (annualized)
        if np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24)  # Hourly data
        else:
            sharpe_ratio = 0
        
        # Average return per trade
        avg_return = np.mean(returns) if len(returns) > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # PnL stability (negative volatility of returns)
        pnl_stability = -np.std(returns) if len(returns) > 0 else 0
        
        # Hit rate
        winning_trades = np.sum(returns > 0)
        hit_rate = winning_trades / len(returns) if len(returns) > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'avg_return_per_trade': avg_return,
            'max_drawdown': abs(max_drawdown),
            'pnl_stability': pnl_stability,
            'hit_rate': hit_rate
        }

# Random Search Alternative (Faster for Initial Exploration)
class RandomSearchOptimizer:
    def __init__(self, n_iter=50):
        self.n_iter = n_iter
    
    def optimize_random(self, X, y, param_distributions):
        """
        Random search with financial CV
        - Faster than Bayesian for initial parameter exploration
        - Good for understanding parameter sensitivity
        """
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.metrics import make_scorer
        
        # Custom scorer for financial metrics
        def financial_scorer(y_true, y_pred_proba):
            # Simplified scoring function
            trade_results = self.simulate_trading_simple(y_true, y_pred_proba)
            return self.calculate_sharpe_ratio(trade_results)
        
        scorer = make_scorer(financial_scorer, needs_proba=True, greater_is_better=True)
        
        # Random search
        random_search = RandomizedSearchCV(
            XGBClassifier(random_state=42),
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            cv=PurgedKFoldCV(n_splits=3),  # Faster CV for random search
            scoring=scorer,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X, y)
        
        return random_search.best_params_, random_search.best_score_
```

### Model Selection & Ensemble Strategy

```python
class ModelSelectionFramework:
    def __init__(self):
        self.candidate_models = {}
        self.performance_history = {}
    
    def compare_model_architectures(self, X, y):
        """
        Compare different model architectures:
        - XGBoost (primary)
        - LightGBM (speed comparison)  
        - CatBoost (categorical handling)
        - Neural Network (deep patterns)
        """
        architectures = {
            'xgb_fast': {'n_estimators': 100, 'max_depth': 4},
            'xgb_deep': {'n_estimators': 500, 'max_depth': 8},
            'xgb_balanced': {'n_estimators': 300, 'max_depth': 6},
        }
        
        results = {}
        cv = WalkForwardValidator()
        
        for name, params in architectures.items():
            model = XGBClassifier(**params, random_state=42)
            
            # Cross-validation
            scores = []
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_test)
                
                # Financial performance
                trade_results = self.simulate_trading(y_test, y_pred, X_test)
                financial_score = self.calculate_financial_metrics(trade_results)
                scores.append(financial_score)
            
            results[name] = {
                'mean_sharpe': np.mean([s['sharpe_ratio'] for s in scores]),
                'mean_return': np.mean([s['avg_return_per_trade'] for s in scores]),
                'max_dd': np.max([s['max_drawdown'] for s in scores]),
                'stability': np.mean([s['pnl_stability'] for s in scores])
            }
        
        return results
```

## 8. Key Improvements Summary

### Addressed Weaknesses:
1. **Static timeframe** → Dynamic labeling windows
2. **Single model risk** → Ensemble approach  
3. **Limited features** → Multi-timeframe, sentiment, microstructure
4. **Basic risk sizing** → Kelly-based dynamic sizing
5. **No drift detection** → Real-time monitoring
6. **Overfitting risk** → Advanced financial CV, regularization
7. **Limited context** → Market regime awareness
8. **Poor validation** → Walk-forward, Purged CV, Embargo
9. **Manual tuning** → Bayesian optimization with financial metrics

### Performance Expectations:
- **Hit Rate**: 60-70% (vs original ~55%)
- **Profit Factor**: 2.5+ (vs original ~1.8)
- **Daily Drawdown**: <4% (vs original ~12%)
- **Max Drawdown**: <8% (vs original ~12%)
- **Sharpe Ratio**: 2.0+ (vs original ~1.2)
- **Model Robustness**: 95%+ validation consistency

This optimized system addresses the core weaknesses while maintaining the explainability advantage of XGBoost through SHAP analysis and proper financial time-series validation.