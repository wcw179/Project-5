# ADR-000: Architecture and Compliance Baseline

Status: Accepted
Date: 2025-08-29

Context
- Project: Black Swan Trading Bot (BlackSwanHunter)
- Canonical specification: .augment/rules/Black Swan Detection System - XGBoost Multi-Label RR (v3) - Optimized.md
- Execution roadmap: TASK.md

Decision
We adopt the following non-negotiable architectural constraints:
1) Data Source & Cadence
   - Canonical historical and (optional) near-real-time M5 OHLCV bars from SQLite at data/m5_trading.db
   - Rolling feature window = 200 hours per symbol, refreshed each minute
   - Supported symbols: EURUSDm, GBPUSDm, USDJPYm, XAUUSDm

2) Targets & Labeling
   - Multi-label binary targets: hit_5R, hit_10R, hit_15R, hit_20R
   - Dynamic horizons: 1–4h (fast), 4–12h (medium), 12–24h (slow)
   - Include spread and slippage; ATR-based dynamic target adjustment; regime-aware modifiers

3) Validation (No Data Leakage)
   - Walk-Forward (train=2000h, test=500h, step=100h) with 24h embargo
   - Purged K-Fold and Combinatorial Purged CV with 24h embargo

4) Modeling & Serving
   - Three-model XGBoost ensemble: fast(100,depth=4), balanced(300,depth=6), deep(500,depth=8)
   - Weighted ensemble: 0.4 balanced, 0.3 deep, 0.3 fast; probability calibration; class weighting
   - SHAP explainability: global/local/interaction; SHAP value export and trending
   - A/B serving: champion 80%, challenger 20%

5) Signals & Risk
   - Gate Stage 1: hit_5R_prob ≥ 0.65 AND confidence_score ≥ 0.7
   - Gate Stage 2: (NY OR overlap session) AND spread_proxy ≤ threshold AND volatility_regime != EXTREMELY_HIGH
   - Gate Stage 3: market_stress_index ≤ threshold AND correlation_breakdown == False
   - Sizing: Kelly-based adaptation with base_risk 1% and probability-driven multipliers; final_risk = base × multiplier × confidence_score
   - Portfolio limits: daily ≤5%, weekly ≤15%, max 3 concurrent positions
   - Position controls: hard 1R stop, ATR/VWAP trailing stops, partial profits at 3R/7R/12R

6) Monitoring & Governance
   - Model performance monitoring: rolling_accuracy_7d, rolling_sharpe_30d
   - Alerting on degradation; drift detection; retrain triggers
   - Audit logging of predictions and trades; explainability artifacts saved

Consequences
- All designs, code, and experiments must pass a compliance checklist derived from this ADR.
- Any deviation requires a new ADR with explicit justification and rollback plan.

References
- TASK.md (project plan)
- .augment/rules/Black Swan Detection System - XGBoost Multi-Label RR (v3) - Optimized.md (system rules)

