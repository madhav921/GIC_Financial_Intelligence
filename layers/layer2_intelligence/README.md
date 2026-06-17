# Layer 2 — Predictive Intelligence

**Controller:** `IntelligenceLayerController` (`controller.py`)
**Entry point:** `controller.train_all_models(commodity_df, macro_df)` → metrics dict

## What this layer does

Trains and operates four commodity price forecasting methods for 12 JLR materials.
Uses Hurst exponent regime detection to adaptively blend model outputs.
Also produces demand forecasts by vehicle segment and price elasticity estimates.

## Forecast method routing

```
H < 0.45  →  Mean-reverting regime  →  SARIMAX dominant
H > 0.55  →  Trending regime        →  XGBoost dominant
H ≈ 0.50  →  Volatile regime        →  Scenario dominant
```

## Key src/ modules

| Module | Role |
|---|---|
| `src/models/commodity_forecast.py` | Orchestrator: trains, routes, generates ensemble forecasts |
| `src/models/regime_detector.py` | Hurst exponent computation and regime classification |
| `src/models/commodity_scenarios.py` | Bear/Base/Bull scenarios with macro probability shifting |
| `src/models/futures_curve.py` | CME/NYMEX forward curve term structure extraction |
| `src/models/demand_forecast.py` | XGBoost per vehicle segment (Luxury, Premium, Performance, EV) |
| `src/models/price_elasticity.py` | Log-log Ridge regression for price sensitivity |
| `src/models/model_registry.py` | Versioned joblib model persistence |
| `src/models/backtesting.py` | 5-fold walk-forward cross-validation (no look-ahead bias) |
