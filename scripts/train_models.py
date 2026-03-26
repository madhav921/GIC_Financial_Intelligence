"""Train all models in the pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_loader import DataLoader
from src.logging_setup import setup_logging
from src.models.commodity_forecast import CommodityForecastModel
from src.models.demand_forecast import DemandForecastModel
from src.models.inventory_risk import InventoryRiskModel
from src.models.price_elasticity import PriceElasticityModel


def main():
    setup_logging()
    print("=" * 60)
    print("  Training All Models — GIC Plan-to-Perform")
    print("=" * 60)

    loader = DataLoader()
    data = loader.load_all()

    commodity_df = data["commodity_prices"]
    sales_df = data["sales_data"]
    macro_df = data["macro_indicators"]
    production_df = data["production_inventory"]

    # ── 1. Commodity Forecast Models ─────────────────────────────
    print("\n[1/4] Training Commodity Forecast Models...")
    cfm = CommodityForecastModel()
    commodity_results = cfm.train_all_commodities(commodity_df, macro_df)

    print("\nCommodity Model Results:")
    for commodity, metrics in commodity_results.items():
        xgb = metrics["xgboost"]
        cv = metrics["cross_validation"]
        print(f"  {commodity:12s} | XGB MAE: {xgb['mae']:>10.1f} | CV MAPE: {cv['cv_mape_mean']:>6.1f}% ± {cv['cv_mape_std']:.1f}%")

    # ── 2. Demand Forecast Models ────────────────────────────────
    print("\n[2/4] Training Demand Forecast Models...")
    dfm = DemandForecastModel()
    demand_results = dfm.train_all_segments(sales_df, macro_df, commodity_df)

    print("\nDemand Model Results:")
    for segment, metrics in demand_results.items():
        if "error" not in metrics:
            print(f"  {segment:16s} | MAE: {metrics['mae']:>8.0f} | MAPE: {metrics['mape']:>6.1f}%")
        else:
            print(f"  {segment:16s} | Error: {metrics['error']}")

    # ── 3. Price Elasticity Models ───────────────────────────────
    print("\n[3/4] Estimating Price Elasticities...")
    pem = PriceElasticityModel()
    elasticity_results = pem.fit_all_segments(sales_df, macro_df, commodity_df)

    print("\nElasticity Estimates:")
    print(pem.summary_table().to_string(index=False))

    # ── 4. Inventory & Warranty Risk ─────────────────────────────
    print("\n[4/4] Analyzing Inventory & Warranty Risk...")
    irm = InventoryRiskModel()
    risk_results = irm.analyze_all_segments(production_df)

    print("\nInventory Risk Analysis:")
    print(irm.summary_table().to_string(index=False))

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  All models trained successfully!")
    print("=" * 60)
    print(f"  Commodity models: {len(commodity_results)}")
    print(f"  Demand models:    {len(demand_results)}")
    print(f"  Elasticity models: {len(elasticity_results)}")
    print(f"  Risk analyses:    {len(risk_results)}")


if __name__ == "__main__":
    main()
