# Layer 3 — Financial Drivers

**Controller:** `FinancialLayerController` (`controller.py`)
**Entry point:** `controller.build_pnl(sales_df, commodity_index_df)` → monthly P&L DataFrame

## What this layer does

Builds the deterministic financial model: Revenue → COGS → Gross Margin → EBIT → Net Income.
Supports real-time commodity shock injection (<1 second recalculation) and scenario P&L.

## Core financial equations

```
Revenue          = Σ(Volume × Net_Price × (1 - Incentive%))
COGS             = Revenue × 77.5% × (1 + Commodity_Impact × 45%)
Gross Margin     = Revenue - COGS
Warranty Reserve = Revenue × 2.5%
Depreciation     = CapEx / Useful_Life / 12
Operating Income = Gross Margin - Warranty - Depreciation
Tax              = max(0, Op_Income × 21%)
Net Income       = Op_Income - Tax
```

## Key src/ modules

| Module | Role |
|---|---|
| `src/drivers/financial_model.py` | P&L assembly engine — stitches all drivers together |
| `src/drivers/revenue_drivers.py` | Volume × Net Price model with elasticity adjustment |
| `src/drivers/cost_drivers.py` | BOM-weighted COGS with commodity index integration |
| `src/drivers/capital_drivers.py` | CapEx scheduling and straight-line depreciation |
