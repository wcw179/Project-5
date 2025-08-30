# Compliance Checklist – Black Swan Detection System (v3) Rules

Scope: Validate adherence to .augment/rules/Black Swan Detection System - XGBoost Multi-Label RR (v3) - Optimized.md
Owner: Engineering Lead
Run: At PRs that touch data, features, modeling, signals, risk, serving, or monitoring

## Data & Cadence
- [ ] Canonical historical source: SQLite at data/m5_trading.db (M5 bars)
- [ ] Supported symbols: EURUSDm, GBPUSDm, USDJPYm, XAUUSDm
- [ ] Rolling feature window: 200 hours per symbol
- [ ] Feature store refresh: 1-minute cadence
- [ ] Data validation: schema, gaps, outliers implemented

## Targets & Labeling
- [ ] Multi-label binary targets: hit_5R, hit_10R, hit_15R, hit_20R
- [ ] Dynamic horizons: 1–4h, 4–12h, 12–24h
- [ ] Spread and slippage incorporated into label computation
- [ ] ATR-based dynamic target adjustment
- [ ] Regime-aware label modifications
- [ ] Leakage checks on labeling pass

## Validation (Time-Series Safe)
- [ ] Walk-Forward splits: train=2000h, test=500h, step=100h, embargo=24h
- [ ] Purged K-Fold with 24h embargo
- [ ] Combinatorial Purged CV with 24h embargo
- [ ] Statistical confirmation of no overlap/leakage

## Feature Engineering
- [ ] Temporal sessions: NY, London, Asian, Overlap, session_transition
- [ ] Market context: news_release_proximity scaffolded
- [ ] Trend/Momentum: EMA(20/50/200) H1/H4, MACD, RSI, Stoch, Bollinger, ATR, OBV, VWAP
- [ ] Volatility/Microstructure/Sentiment (enhancement phase) planned and versioned
- [ ] Feature versioning, validation, and outlier detection in pipeline

## Modeling & Explainability
- [ ] XGBoost models: fast(100,d4), balanced(300,d6), deep(500,d8)
- [ ] Weighted ensemble: 0.4 balanced, 0.3 deep, 0.3 fast
- [ ] Probability calibration implemented
- [ ] Class weighting for imbalance
- [ ] SHAP global/local/interaction; SHAP values exported and trended

## Optimization
- [ ] Optuna TPE with 50–100 trials
- [ ] Multi-objective: Sharpe (35%), Avg return/trade (25%), Stability (20%), MaxDD constraint (<=8%), Hit-rate bonus (5%)
- [ ] Early stop for unpromising trials; parallel execution configured

## Signals & Risk
- [ ] Signal Gate Stage 1: hit_5R_prob ≥ 0.65 AND confidence_score ≥ 0.7
- [ ] Signal Gate Stage 2: (is_ny_session OR is_overlap_session) AND spread_proxy ≤ threshold AND volatility_regime != EXTREMELY_HIGH
- [ ] Signal Gate Stage 3: market_stress_index ≤ threshold AND correlation_breakdown == False
- [ ] Kelly-based sizing: base 1%; multipliers by probabilities; final_risk = base × multiplier × confidence_score
- [ ] Position controls: hard 1R SL, ATR/VWAP trailing, partial profits at 3R/7R/12R
- [ ] Portfolio limits: daily ≤5%, weekly ≤15%, max 3 concurrent, correlation limits

## Serving & Monitoring
- [ ] A/B serving: champion 80%, challenger 20%
- [ ] Real-time monitoring: rolling_accuracy_7d, rolling_sharpe_30d with actions
- [ ] Data/feature drift detection present
- [ ] Alerts: Email/SMS/Slack configured
- [ ] Audit logging of predictions and trades; explainability DB persisted

## Documentation & Governance
- [ ] ADR-000 present and current
- [ ] Compliance checklist integrated in CI
- [ ] Traceability matrix up-to-date

