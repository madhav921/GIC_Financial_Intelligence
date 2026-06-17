"""
Layer 3 — Financial Drivers

Responsibilities:
  - Revenue model: Revenue = Σ(Volume_segment × Net_Price × (1 - Incentive%))
  - COGS model:    COGS = Revenue × Base_COGS% × (1 + Commodity_Impact × Material_Fraction)
  - P&L assembly:  Gross Margin → Warranty → Depreciation → EBIT → Tax → Net Income
  - Commodity shock injection: <1 second real-time EBIT recalculation
  - Scenario P&L:  Apply demand + commodity shocks deterministically

Financial equations:
  Revenue          = Σ(Volume × Net_Price × (1 - Incentive%))
  COGS             = Revenue × 77.5% × (1 + Commodity_Impact × 45%)
  Gross Margin     = Revenue - COGS
  Warranty Reserve = Revenue × 2.5%
  Depreciation     = CapEx / Useful_Life / 12
  Operating Income = Gross Margin - Warranty - Depreciation
  Tax              = max(0, Op_Income × 21%)
  Net Income       = Op_Income - Tax

Key modules (in src/drivers/):
  revenue_drivers.py  — Volume × Net Price with elasticity
  cost_drivers.py     — BOM-weighted COGS with commodity index
  capital_drivers.py  — CapEx scheduling, straight-line depreciation
  financial_model.py  — P&L assembly engine
"""
from layers.layer3_financial.controller import FinancialLayerController
__all__ = ["FinancialLayerController"]
