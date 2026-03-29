"""
Commodity Forecast Pipeline — End-to-End

Runs the complete commodity price forecasting pipeline:
  Stage 1: Data Generation (12 JLR-relevant commodities)
  Stage 2: Model Training (SARIMAX + XGBoost for each commodity)
  Stage 3: Forecast Generation (4 methods per commodity)
  Stage 4: Scenario Analysis (Bear/Base/Bull with macro assumptions)
  Stage 5: Futures Curve Extraction (for eligible commodities)
  Stage 6: Variance Tracking & Monthly Update Simulation
  Stage 7: Commodity Index → L2 Financial Driver Engine
  Stage 8: Governance (Audit trail + Explainability)

Usage:
    python scripts/run_commodity_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.data_loader import DataLoader
from src.data.synthetic_generator import generate_all_synthetic_data
from src.governance.audit_trail import AuditTrail
from src.governance.explainability import ExplainabilityEngine
from src.logging_setup import setup_logging
from src.models.commodity_forecast import CommodityForecastModel, ForecastResult
from src.models.commodity_scenarios import (
    CommodityScenarioEngine,
    MacroAssumptions,
    MonthlyUpdatePipeline,
    VarianceTracker,
)
from src.models.futures_curve import FuturesCurveExtractor


def _print_header(text: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def _print_forecast_summary(commodity: str, result: ForecastResult) -> None:
    """Print a compact forecast summary."""
    print(
        f"    {commodity:16s} | {result.model_type:16s} "
        f"| Next: {result.point_forecast[0]:>10,.1f} "
        f"| +12m: {result.point_forecast[-1]:>10,.1f}"
    )


def main():
    setup_logging()
    audit = AuditTrail()

    _print_header("GIC Commodity Price Forecast Pipeline")
    print("  12 JLR-relevant commodities | 4 forecasting methods")
    print("  3-18 month horizon | Bear/Base/Bull scenarios")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 1: DATA GENERATION
    # ═══════════════════════════════════════════════════════════════
    _print_header("STAGE 1: Data Generation")
    datasets = generate_all_synthetic_data(seed=42)
    for name, df in datasets.items():
        audit.log_data_ingestion("synthetic", name, len(df))
        print(f"    {name:25s}: {df.shape}")

    loader = DataLoader()
    commodity_df = loader.load_commodity_prices()
    macro_df = loader.load_macro_indicators()

    commodity_cols = [c for c in commodity_df.columns if c != "date"]
    print(f"\n  Commodities loaded: {len(commodity_cols)}")
    print(f"  Date range: {commodity_df['date'].min()} to {commodity_df['date'].max()}")
    print(f"  Commodities: {', '.join(commodity_cols)}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 2: MODEL TRAINING (SARIMAX + XGBoost)
    # ═══════════════════════════════════════════════════════════════
    _print_header("STAGE 2: Model Training (SARIMAX + XGBoost)")
    cfm = CommodityForecastModel()
    train_results = cfm.train_all_commodities(commodity_df, macro_df)

    print("\n  Training Results:")
    print(f"  {'Commodity':16s} | {'XGB MAE':>10s} | {'XGB MAPE':>10s} | {'CV MAPE':>10s} | {'SARIMAX AIC':>12s}")
    print(f"  {'-'*16}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
    for commodity, metrics in train_results.items():
        xgb = metrics.get("xgboost", {})
        cv = metrics.get("cross_validation", {})
        sar = metrics.get("sarimax", {})
        xgb_mae = xgb.get("mae", float("nan"))
        xgb_mape = xgb.get("mape", float("nan"))
        cv_mape = cv.get("cv_mape_mean", float("nan"))
        sar_aic = sar.get("aic", float("nan"))
        print(
            f"  {commodity:16s} | {xgb_mae:>10.1f} | {xgb_mape:>9.1f}% | {cv_mape:>9.1f}% | {sar_aic:>12.0f}"
        )

        # Log to audit
        audit.log_forecast("xgboost", commodity, [], xgb)

    # ═══════════════════════════════════════════════════════════════
    # STAGE 3: FORECAST GENERATION (All Methods)
    # ═══════════════════════════════════════════════════════════════
    _print_header("STAGE 3: Forecast Generation")

    print("\n  [3.1] Method 1 — SARIMAX (Baseline)")
    for commodity in train_results:
        if commodity in cfm.sarimax_models:
            result = cfm.forecast_sarimax(commodity)
            _print_forecast_summary(commodity, result)

    print("\n  [3.2] Method 2 — XGBoost (Macro-Driven)")
    xgb_forecasts = {}
    for commodity in train_results:
        if commodity in cfm.xgb_models:
            result = cfm.forecast_xgboost(commodity, commodity_df, macro_df)
            xgb_forecasts[commodity] = result
            _print_forecast_summary(commodity, result)

    print("\n  [3.3] Method 3 — Futures Curve Extraction (Market-Implied)")
    futures_results = {}
    latest_prices = {}
    for commodity in commodity_cols:
        latest_prices[commodity] = float(commodity_df[commodity].dropna().iloc[-1])

    for commodity in commodity_cols:
        if FuturesCurveExtractor.is_eligible(commodity):
            result = cfm.forecast_futures_curve(commodity, latest_prices[commodity])
            futures_results[commodity] = result
            _print_forecast_summary(commodity, result)

    if not futures_results:
        print("    (No eligible commodities in current data)")

    print("\n  [3.4] Method 4 — Scenario Analysis (Bear/Base/Bull)")
    scenario_engine = CommodityScenarioEngine()
    macro = MacroAssumptions()  # Default base case
    scenario_table = scenario_engine.scenario_comparison_table(latest_prices, macro)
    if not scenario_table.empty:
        print(scenario_table.to_string(index=False))
    else:
        print("    (No scenario config — using basic scenario analysis)")
        for commodity in commodity_cols[:5]:
            result = cfm.forecast_scenario(commodity, latest_prices.get(commodity), macro)
            _print_forecast_summary(commodity, result)

    # ═══════════════════════════════════════════════════════════════
    # STAGE 4: MULTI-METHOD COMPARISON
    # ═══════════════════════════════════════════════════════════════
    _print_header("STAGE 4: Multi-Method Comparison")
    print(f"\n  {'Commodity':16s} | {'Preferred':>10s} | {'SARIMAX':>10s} | {'XGBoost':>10s} | {'Futures':>10s} | {'Scenario':>10s}")
    print(f"  {'-'*16}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for commodity in train_results:
        preferred = cfm.get_preferred_method(commodity)
        sar_12m = "—"
        xgb_12m = "—"
        fut_12m = "—"
        scn_12m = "—"

        if commodity in cfm.sarimax_models:
            r = cfm.forecast_sarimax(commodity)
            sar_12m = f"{r.point_forecast[-1]:,.0f}"

        if commodity in xgb_forecasts:
            xgb_12m = f"{xgb_forecasts[commodity].point_forecast[-1]:,.0f}"

        if commodity in futures_results:
            fut_12m = f"{futures_results[commodity].point_forecast[-1]:,.0f}"

        if commodity in latest_prices:
            r = cfm.forecast_scenario(commodity, latest_prices[commodity], macro)
            scn_12m = f"{r.point_forecast[-1]:,.0f}"

        print(
            f"  {commodity:16s} | {preferred:>10s} | {sar_12m:>10s} | {xgb_12m:>10s} | {fut_12m:>10s} | {scn_12m:>10s}"
        )

    # ═══════════════════════════════════════════════════════════════
    # STAGE 5: COMMODITY INDEX (Feed → L2 COGS Driver)
    # ═══════════════════════════════════════════════════════════════
    _print_header("STAGE 5: BOM-Weighted Commodity Index")
    commodity_index = cfm.generate_commodity_index(commodity_df)
    print(f"\n  Index range: {commodity_index['commodity_index'].min():.1f} to {commodity_index['commodity_index'].max():.1f}")
    print(f"  Latest index value: {commodity_index['commodity_index'].iloc[-1]:.1f}")
    print(f"  Index base = 100.0 (normalized at first observation)")
    print(f"\n  This feeds into Layer 2: COGS = Base_COGS × (1 + Material_45% × Index_Change)")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 6: VARIANCE TRACKING (Monthly Update Simulation)
    # ═══════════════════════════════════════════════════════════════
    _print_header("STAGE 6: Variance Tracking & Monthly Update")

    # Simulate a monthly update: compare mid-period forecasts to end-period actuals
    if len(commodity_df) >= 24:
        mid_idx = len(commodity_df) // 2
        # Simulate "prior forecast" as prices from midpoint
        prior_row = commodity_df.iloc[mid_idx]
        actual_row = commodity_df.iloc[-1]

        prior_forecasts_sim = {}
        for col in commodity_cols:
            if pd.notna(prior_row.get(col)):
                prior_forecasts_sim[col] = float(prior_row[col])

        update_result = cfm.run_monthly_update(
            commodity_df, prior_forecasts_sim, macro
        )

        print(f"\n  Update month: {update_result['update_month']}")
        print(f"  Commodities updated: {update_result['num_commodities_updated']}")

        if update_result["escalations"]:
            print(f"  ESCALATIONS (>10% variance): {update_result['escalations']}")
        else:
            print("  No escalations required")

        # Print variance records
        print(f"\n  {'Commodity':16s} | {'Prior':>10s} | {'Actual':>10s} | {'Variance':>8s} | {'Level':>18s}")
        print(f"  {'-'*16}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*18}")
        for vr in update_result["variance_records"]:
            print(
                f"  {vr.commodity:16s} | {vr.prior_forecast:>10,.1f} | {vr.actual_price:>10,.1f} "
                f"| {vr.variance_pct:>+7.1f}% | {vr.escalation_level:>18s}"
            )

    # ═══════════════════════════════════════════════════════════════
    # STAGE 7: SCENARIO STRESS TEST (Bear / Bull)
    # ═══════════════════════════════════════════════════════════════
    _print_header("STAGE 7: Macro Scenario Stress Tests")

    scenarios = {
        "Base Case": MacroAssumptions(),
        "Bear (Contraction)": MacroAssumptions(
            global_manufacturing_pmi=47,
            china_gdp_growth=3.5,
            usd_dxy=108,
            energy_ttf=50,
            supply_disruption_risk="high",
        ),
        "Bull (Expansion)": MacroAssumptions(
            global_manufacturing_pmi=54,
            china_gdp_growth=5.8,
            usd_dxy=96,
            energy_ttf=22,
            ev_demand_growth=0.28,
        ),
    }

    for scenario_name, macro_input in scenarios.items():
        print(f"\n  [{scenario_name}]")
        table = scenario_engine.scenario_comparison_table(latest_prices, macro_input)
        if not table.empty:
            for _, row in table.iterrows():
                print(
                    f"    {row['commodity']:16s} | Bear: {row['bear_12m']:>10,.0f} "
                    f"| Base: {row['base_12m']:>10,.0f} | Bull: {row['bull_12m']:>10,.0f} "
                    f"| Weighted: {row['weighted_12m']:>10,.0f}"
                )

        audit.log_scenario_run(
            f"commodity_{scenario_name.lower().replace(' ', '_')}",
            {
                "pmi": macro_input.global_manufacturing_pmi,
                "china_gdp": macro_input.china_gdp_growth,
                "dxy": macro_input.usd_dxy,
            },
            {"commodities_analyzed": len(table) if not table.empty else 0},
        )

    # ═══════════════════════════════════════════════════════════════
    # STAGE 8: GOVERNANCE & AUDIT
    # ═══════════════════════════════════════════════════════════════
    _print_header("STAGE 8: Governance & Audit")

    # Feature importance for top commodities
    print("\n  Top Feature Importance (by commodity):")
    for commodity in list(train_results.keys())[:4]:
        fi = cfm.get_feature_importance(commodity)
        if not fi.empty:
            top3 = fi.head(3)
            features_str = ", ".join(
                f"{row['feature']}({row['importance']:.3f})"
                for _, row in top3.iterrows()
            )
            print(f"    {commodity:16s}: {features_str}")

    # CV metrics summary
    cv_df = cfm.get_cv_metrics()
    if not cv_df.empty:
        print("\n  Cross-Validation Summary:")
        print(
            cv_df[["commodity", "cv_mape_mean", "cv_mape_std", "cv_directional_accuracy"]]
            .round(1)
            .to_string(index=False)
        )

    entries = audit.get_entries()
    print(f"\n  Audit trail entries: {len(entries)}")

    # ═══════════════════════════════════════════════════════════════
    _print_header("Pipeline Complete!")
    print(f"\n  Models trained:     {len(train_results)} commodities")
    print(f"  Methods available:  SARIMAX, XGBoost, Futures Curve, Scenario")
    print(f"  Scenarios:          Bear / Base / Bull with macro-driven weights")
    print(f"  Variance tracking:  Enabled (>5% alert, >10% escalation)")
    print(f"  Commodity Index:    Ready for L2 COGS driver")
    print(f"\n  Next steps:")
    print(f"    1. Monthly update: python scripts/run_commodity_pipeline.py")
    print(f"    2. Full pipeline:  python scripts/run_pipeline.py")
    print(f"    3. Dashboard:      streamlit run src/dashboard/app.py")
    print(f"    4. API:            uvicorn src.api.app:app --reload")


if __name__ == "__main__":
    main()
