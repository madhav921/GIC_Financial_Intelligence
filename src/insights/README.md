# Actionable-Intelligence Layer (`src/insights/`)

This layer turns the platform's **descriptive** analytics (commodity forecasts,
P&L, Monte Carlo) into **prescriptive, prioritised, £-quantified** intelligence —
the "so what / now what" on top of the "what".

## Components

| Module | Responsibility |
|---|---|
| `schemas.py` | `InsightCard` + `VarianceBridge` dataclasses (the shared contracts). |
| `variance_bridge.py` | `VarianceBridgeAnalyzer` — decomposes the plan→actual EBIT gap into Volume, Price/Mix, Commodity, FX, Warranty, Other. |
| `anomaly_detector.py` | `AnomalyDetector` — rolling z-score / IQR flags on any series. |
| `early_warning.py` | `EarlyWarningSystem` — composite 0–100 cross-domain risk score. |
| `recommendation_engine.py` | `RecommendationEngine` — hedge / inventory / pricing actions with £ impact. |
| `insight_engine.py` | `InsightEngine` — orchestrator that emits the prioritised `InsightCard` feed. |

## `InsightCard` schema

Each card pairs a quantified **finding** with the **reasoning**, the £ **impact**,
a concrete **recommended_action**, and the £ **expected_action_savings**:

```
id, category (commodity_risk|margin|demand|warranty|hedging|cost|opportunity),
severity (info|warning|critical), priority (1..5), title, finding, reasoning,
impact_gbp, impact_label, confidence (0-1), recommended_action,
expected_action_savings_gbp, affected_segments[], supporting_metrics{}
```

`InsightEngine.generate_insights(context)` works with rich context **or none at
all** — with empty/partial context it backfills a curated, specific demo feed so
the API always returns high-quality content. Results are sorted by priority then
impact magnitude and capped to the top N (default 8).

## How Layer 5 (LLM governance) consumes this

The LLM-governance layer reads the structured `InsightCard` / `VarianceBridge`
dicts (via `/insights/feed` and `/insights/variance-bridge`) and writes the
executive **narrative**: it never invents numbers — it grounds every sentence in
the `impact_gbp`, `recommended_action` and `supporting_metrics` already computed
here, keeping the narrative auditable and consistent with the quantitative layer.
