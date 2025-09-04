"""
Cross-Asset & Macro feature set (V4 §2.6)

Provides features capturing macro environment and cross-asset dynamics:
- Cross-asset correlations (equity, bond, commodity, crypto)
- Yield curve slope (10Y-2Y), credit spreads
- Dollar strength (DXY momentum), risk-on/risk-off indicator
- Systemic risk, liquidity stress, correlation breakdown, contagion risk

Note: Implementations should fetch/derive inputs from appropriate data sources
and align to the bar clock used by the trading system (M5 aggregation + HTF rollups).
"""

