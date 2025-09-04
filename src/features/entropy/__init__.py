"""
Entropy feature set (V4 §2.6)

Family of features measuring complexity/information of price processes:
- Information-theoretic measures (Shannon, Rényi)
- Lempel–Ziv / Kontoyiannis entropy rate
- Gaussian process entropy & entropy-implied volatility
- Entropy & generalized mean (risk concentration)
- Market microstructure entropy

Notes:
- Implementations should operate on rolling windows aligned to bar clock (M5) and
  optionally on higher timeframes for stability.
- Provide numerically stable estimators and clear parameterization (window, base,
  normalization).
"""

