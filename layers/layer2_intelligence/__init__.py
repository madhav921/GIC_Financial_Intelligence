"""
Layer 2 — Predictive Intelligence

Responsibilities:
  - Commodity price forecasting: 4 methods for 12 JLR materials
      Method 1: SARIMAX(1,1,1)(1,1,1,12) — seasonal ARIMA, stable commodities
      Method 2: XGBoost with 50+ features — macro-driven, volatile commodities
      Method 3: Futures Curve Extraction — market-implied forward prices
      Method 4: Scenario Analysis — Bear/Base/Bull expert hybrid
  - Regime Detection (Hurst exponent) — adaptive ensemble weighting
      H < 0.45 → Mean-reverting → SARIMAX dominant
      H > 0.55 → Trending       → XGBoost dominant
      H ≈ 0.50 → Volatile       → Scenario dominant
  - Demand Forecasting — XGBoost per vehicle segment (Luxury, Premium, Performance, EV)
  - Price Elasticity — log-log Ridge regression
  - 5-fold walk-forward cross-validation (no look-ahead bias)
  - Model Registry — joblib versioned model persistence

Key modules (in src/models/):
  commodity_forecast.py         — Orchestrator: trains, routes, generates forecasts
  regime_detector.py            — Hurst exponent adaptive weighting
  commodity_scenarios.py        — Bear/Base/Bull with macro probability shifting
  futures_curve.py              — CME/NYMEX term structure extraction
  demand_forecast.py            — Segment-level demand (XGBoost)
  price_elasticity.py           — Log-log Ridge regression
  model_registry.py             — Versioned model persistence (joblib)
  backtesting.py                — Walk-forward validation
"""
from layers.layer2_intelligence.controller import IntelligenceLayerController
__all__ = ["IntelligenceLayerController"]
