# Advanced Forecasting Capabilities (Additive)

These modules are **purely additive**. They live under `src/models/` and add
state-of-the-art uncertainty calibration, explainability, and regime awareness
on top of the existing SARIMAX + XGBoost ensemble in `commodity_forecast.py`.
**No existing file is modified.** Each module imports heavy/optional
dependencies lazily and degrades gracefully (documented fallback) so it never
hard-crashes the platform. They are designed to be wired in opportunistically —
the current pipeline runs unchanged if they are never called.

| Module | SOTA Technique | Research Reference | Concrete Improvement to GIC |
|---|---|---|---|
| `conformal.py` — `ConformalForecaster` | Split-conformal prediction intervals + Adaptive Conformal Inference (ACI) | Vovk et al.; Angelopoulos & Bates 2021 ("A Gentle Intro to Conformal Prediction"); Gibbs & Candès 2021 (ACI) | **Provable marginal coverage** regardless of model misspecification — guaranteed ~80% bands vs SARIMAX's empirical ~79% that assumes Gaussian errors. ACI re-tunes alpha online to hold coverage under regime drift. |
| `explainability_shap.py` — `ShapExplainer` | SHAP feature attribution (TreeExplainer) with gain/permutation fallback | Lundberg & Lee 2017 ("A Unified Approach to Interpreting Model Predictions", NeurIPS); Lundberg et al. 2020 | **Quantified per-driver attribution** feeding LLM narratives ("Manufacturing PMI contributed +2.3% to the forecast") instead of opaque importance scores. Falls back to XGBoost gain + permutation importance if `shap` is absent. |
| `change_point.py` — `ChangePointDetector` | CUSUM control chart + Bayesian Online Change-Point Detection (BOCPD) | Page 1954 (CUSUM); Adams & MacKay 2007 (BOCPD, arXiv:0710.3742) | **Faster regime-shift alerts than Hurst alone** — fires on the step a structural break occurs and emits a calibrated change-probability confidence, whereas the window-averaged Hurst exponent reacts with a multi-month lag. |
| `quantile_forecast.py` — `QuantileForecaster` | Gradient-boosted quantile (pinball) regression | Koenker & Bassett 1978; Friedman 2001; Meinshausen 2006 (Quantile Regression Forests) | **Asymmetric, fat-tail intervals** that SARIMAX's symmetric ±1.96·sigma band misses — captures the skewed downside risk of metals (Lithium/Cobalt/Rhodium). Uses XGBoost quantile objective, falls back to sklearn `GradientBoostingRegressor(loss="quantile")`. |

## How they compose

- **Conformal + Ensemble**: wrap the existing ensemble point forecast; feed
  held-out residuals to `calibrate()`, then emit guaranteed-coverage bands.
- **Quantile GBR**: a drop-in alternative interval source where downside
  asymmetry matters; complements (not replaces) conformal coverage.
- **SHAP**: explain any trained XGBoost commodity model; output is LLM-ready
  driver text for the narrative layer.
- **Change-point**: augments `regime_detector.py` (Hurst) with abrupt-break
  detection for dashboard alerting.

## Quick verification

Each module ships a `quick_demo()` (run `python -m src.models.<module>`). They
tolerate missing optional deps (`shap`, and either `xgboost` or `scikit-learn`)
by falling back to a documented alternative path.
