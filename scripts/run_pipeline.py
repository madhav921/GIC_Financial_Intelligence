"""
End-to-end pipeline: data generation → training → financial model → simulation.

This script runs the complete Plan-to-Perform pipeline locally.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.data_loader import DataLoader
from src.data.synthetic_generator import generate_all_synthetic_data
from src.drivers.financial_model import FinancialModel
from src.governance.audit_trail import AuditTrail
from src.governance.bias_tracking import BiasTracker
from src.logging_setup import setup_logging
from src.models.commodity_forecast import CommodityForecastModel
from src.models.demand_forecast import DemandForecastModel
from src.models.inventory_risk import InventoryRiskModel
from src.models.price_elasticity import PriceElasticityModel
from src.simulation.scenario_engine import ScenarioDefinition, ScenarioEngine


def main():
    setup_logging()
    audit = AuditTrail()

    print("=" * 70)
    print("  GIC Plan-to-Perform | End-to-End Pipeline")
    print("=" * 70)

    # ===================================================================
    # STAGE 1: DATA GENERATION
    # ===================================================================
    print("\n[STAGE 1] Generating Synthetic Data...")
    datasets = generate_all_synthetic_data(seed=42)
    for name, df in datasets.items():
        audit.log_data_ingestion("synthetic", name, len(df))
        print(f"    {name}: {df.shape}")

    loader = DataLoader()
    commodity_df = loader.load_commodity_prices()
    sales_df = loader.load_sales_data()
    macro_df = loader.load_macro_indicators()
    production_df = loader.load_production_inventory()

    # ===================================================================
    # STAGE 2: MODEL TRAINING (Layer 3)
    # ===================================================================
    print("\n[STAGE 2] Training AI Models (Layer 3)...")

    # Commodity Forecast
    print("  [2.1] Commodity Forecast Models")
    cfm = CommodityForecastModel()
    commodity_results = cfm.train_all_commodities(commodity_df, macro_df)
    commodity_index = cfm.generate_commodity_index(commodity_df)

    for commodity, metrics in commodity_results.items():
        xgb = metrics["xgboost"]
        print(f"        {commodity:12s} → MAPE: {xgb['mape']:.1f}%")
        audit.log_forecast("xgboost", commodity, [], xgb)

    # Demand Forecast
    print("  [2.2] Demand Forecast Models")
    dfm = DemandForecastModel()
    demand_results = dfm.train_all_segments(sales_df, macro_df, commodity_df)
    for seg, m in demand_results.items():
        if "error" not in m:
            print(f"        {seg:16s} → MAPE: {m['mape']:.1f}%")

    # Price Elasticity
    print("  [2.3] Price Elasticity Estimation")
    pem = PriceElasticityModel()
    pem.fit_all_segments(sales_df, macro_df, commodity_df)
    print(pem.summary_table().to_string(index=False))

    # Inventory Risk
    print("  [2.4] Inventory & Warranty Risk")
    irm = InventoryRiskModel()
    irm.analyze_all_segments(production_df)
    print(irm.summary_table().to_string(index=False))

    # ===================================================================
    # STAGE 3: FINANCIAL MODEL (Deterministic Driver Model)
    # ===================================================================
    print("\n[STAGE 3] Building Financial Model...")
    fm = FinancialModel()
    pnl = fm.build_pnl(sales_df, commodity_index)
    annual = fm.annual_summary(pnl)
    print("\n  Annual P&L Summary:")
    print(annual[["year", "segment", "volume", "net_revenue", "gross_margin", "gross_margin_pct", "operating_income"]].to_string(index=False))

    # ===================================================================
    # STAGE 4: SCENARIO SIMULATION (Layer 4)
    # ===================================================================
    print("\n[STAGE 4] Running Scenario Simulations...")
    engine = ScenarioEngine()
    presets = ScenarioEngine.preset_scenarios()
    comparison = engine.compare_scenarios(presets, sales_df, commodity_index)

    print("\n  Scenario Comparison:")
    print(comparison.to_string(index=False))

    # Run Monte Carlo on key scenarios
    print("\n  Running Monte Carlo on key scenarios...")
    for scenario_name in ["Base Case", "Commodity Crisis", "Stagflation"]:
        scenario = next(s for s in presets if s.name == scenario_name)
        result = engine.run_scenario(scenario, sales_df, commodity_index, n_simulations=5000)
        sim = result["simulation"]
        stats = sim.stats["gross_margin"]
        print(f"    {scenario_name:20s} | Mean Margin: ${stats['mean']:>14,.0f} | VaR(95%): ${stats['var_95']:>14,.0f}")

        audit.log_scenario_run(
            scenario_name, {"demand": scenario.demand_shock, "commodity": scenario.commodity_shock},
            {"mean_margin": stats["mean"], "var_95": stats["var_95"]},
        )

    # ===================================================================
    # STAGE 5: GOVERNANCE (Layer 5)
    # ===================================================================
    print("\n[STAGE 5] Governance & Audit...")

    # Bias tracking (simulate with in-sample data)
    tracker = BiasTracker()
    for commodity in ["Lithium", "Steel", "Aluminum"]:
        if commodity in cfm.sarimax_models:
            fitted = cfm.sarimax_models[commodity].model_fit.fittedvalues
            actuals = commodity_df[commodity].iloc[-len(fitted):]
            report = tracker.compute_bias(actuals, fitted, "sarimax", commodity)
            alert_flag = " [ALERT]" if report.is_alert else " [OK]"
            print(f"    Bias {commodity:12s}: {report.mean_bias_pct:+.2f}% ({report.bias_direction}-forecasting){alert_flag}")

    # Audit trail summary
    entries = audit.get_entries()
    print(f"\n    Audit trail entries: {len(entries)}")

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  Pipeline Complete!")
    print("=" * 70)
    print("\n  Next steps:")
    print("    1. Start API:      uvicorn src.api.app:app --reload")
    print("    2. View docs:      http://localhost:8000/docs")
    print("    3. Run tests:      pytest tests/ -v")
    print("    4. Dashboard:      streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
