# Build Action Items
### *Exact steps. No fluff. Hackathon in 4 weeks.*

---

## Before You Start — File Inventory

Current state of the codebase you're building on:

| Module | Status | Action |
|---|---|---|
| `src/data/polars_pipeline.py` | ✅ Working | Wrap behind Protocol |
| `src/data/synthetic_generator.py` | ✅ Working | Wrap behind Protocol |
| `src/data/connectors/yfinance_connector.py` | ✅ Working (9/12 commodities) | Wrap behind Protocol |
| `src/data/connectors/fred_connector.py` | ✅ Working | Wrap behind Protocol |
| `src/data/connectors/erp_connector.py` | 🔧 Stub | Implement Protocol interface |
| `src/models/commodity_forecast.py` | ✅ Working | Add RegimeDetector hook |
| `src/models/demand_forecast.py` | ✅ Working | No change needed |
| `src/drivers/financial_model.py` | ✅ Working | Add shock injection method |
| `src/simulation/monte_carlo.py` | ✅ Working | Expose decomposed variance |
| `src/dashboard/pages/executive_summary.py` | ✅ Working | Add fan chart |
| `src/dashboard/pages/commodity_intelligence.py` | ✅ Working | Add shock sliders + hedge panel |
| `src/dashboard/pages/scenario_simulation.py` | ✅ Working | Wire Monte Carlo decomposition |
| `src/dashboard/pages/financial_pnl.py` | ✅ Working | No change needed |

**New files to create:**
- `src/data/data_source_protocol.py`
- `src/data/data_router.py`
- `src/models/regime_detector.py`
- `src/models/commodity_shock.py`
- `src/models/hedge_optimizer.py`

---

## WEEK 1 — Plug-and-Play Data Layer
**Goal: Any data source swaps via config. All downstream code is source-agnostic.**

### Task 1.1 — Create the Protocol file
**File:** `src/data/data_source_protocol.py`

Create this file from scratch with two Protocol classes:

```python
from __future__ import annotations
from typing import Protocol
import polars as pl

class MarketDataSource(Protocol):
    def get_commodity_prices(self, from_date: str, to_date: str) -> pl.DataFrame: ...
    def get_macro_indicators(self, from_date: str, to_date: str) -> pl.DataFrame: ...
    def get_fx_rates(self, from_date: str, to_date: str) -> pl.DataFrame: ...

class OperationalDataSource(Protocol):
    def get_sales(self, from_date: str, to_date: str) -> pl.DataFrame: ...
    def get_production(self, from_date: str, to_date: str) -> pl.DataFrame: ...
    def get_cogs_detail(self, from_date: str, to_date: str) -> pl.DataFrame: ...
    def get_bom(self) -> pl.DataFrame: ...
    def get_inventory(self, from_date: str, to_date: str) -> pl.DataFrame: ...
```

### Task 1.2 — Wrap existing sources as Protocol implementations
**File:** `src/data/data_router.py`

Create 4 classes that implement the protocols:

**`SyntheticOperationalSource`** — wraps `src/data/synthetic_generator.py`
- `get_sales()` → call `SyntheticDataGenerator().generate_sales_data()`
- `get_production()` → call `SyntheticDataGenerator().generate_production_data()`
- `get_cogs_detail()` → call `SyntheticDataGenerator().generate_bom_data()`
- `get_bom()` → load `data/synthetic/bom_data.csv` via polars
- `get_inventory()` → call `SyntheticDataGenerator().generate_production_inventory()`

**`ParquetOperationalSource`** — reads from `data/parquet/` or `data/raw/`
- Each method: `pl.scan_parquet(...)` with date filter; fallback to CSV if parquet missing
- If file missing → raise `FileNotFoundError` with helpful message pointing to `generate_data.py`

**`YFinanceMarketSource`** — wraps `src/data/connectors/yfinance_connector.py`
- `get_commodity_prices()` → call `fetch_commodity_prices()`, filter by date
- `get_macro_indicators()` → call `fred_connector.fetch_macro_indicators()`
- `get_fx_rates()` → call `fetch_fx_rates()` from yfinance

**`SAPOperationalSource`** — stub for production
```python
class SAPOperationalSource:
    """
    SAP S/4HANA connector. Implement using pyrfc or SAP REST API.
    See: src/data/connectors/erp_connector.py for connection template.
    Required env vars: SAP_HOST, SAP_USER, SAP_PASS, SAP_CLIENT
    """
    def get_sales(self, from_date: str, to_date: str) -> pl.DataFrame:
        raise NotImplementedError(
            "SAP connector not implemented. "
            "Set operational_data_source: parquet in settings.yaml "
            "and provide data/raw/sales_data.csv"
        )
    # same pattern for all other methods
```

**`DataRouter`** — reads `settings.yaml` and returns the right source
```python
def get_operational_source() -> OperationalDataSource:
    settings = get_settings()
    match settings.get("operational_data_source", "synthetic"):
        case "synthetic":  return SyntheticOperationalSource()
        case "parquet":    return ParquetOperationalSource()
        case "sap":        return SAPOperationalSource()
        case _:            raise ValueError(f"Unknown source: {source}")

def get_market_source() -> MarketDataSource:
    settings = get_settings()
    match settings.get("market_data_source", "live"):
        case "live":    return YFinanceMarketSource()
        case "parquet": return ParquetMarketSource()
        case _:         raise ValueError(...)
```

### Task 1.3 — Update settings.yaml
**File:** `config/settings.yaml`

Add at the top of the file:
```yaml
# Data source selection (synthetic | parquet | sap | salesforce)
operational_data_source: synthetic
# Market data source (live | parquet | synthetic)  
market_data_source: live
```

### Task 1.4 — Update FinancialModel to accept injected source
**File:** `src/drivers/financial_model.py`

Change `__init__` to accept an optional `OperationalDataSource`:
```python
def __init__(self, data_source: OperationalDataSource | None = None):
    self.data_source = data_source or DataRouter.get_operational_source()
```

Where the model currently loads data directly, replace with `self.data_source.get_sales(...)` etc.

### Task 1.5 — Verify it works
```bash
# Run with synthetic (default)
python -c "from src.data.data_router import get_operational_source; s = get_operational_source(); print(s.get_sales('2024-01-01', '2025-12-31').head())"

# Change config to parquet, run again — same output structure, different data
```

**Week 1 done when:** Changing `operational_data_source` in settings.yaml between `synthetic` and `parquet` produces the same downstream output structure.

---

## WEEK 2 — The 4 High-Impact Features

### Task 2.1 — Commodity Shock Calculator
**New file:** `src/models/commodity_shock.py`

```python
class CommodityShockCalculator:
    """
    Given a commodity price shock (%), computes the P&L cascade.
    Used by the live dashboard sliders.
    """
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.bom_weights = self._load_bom_weights()  # from settings or bom_data
    
    def _load_bom_weights(self) -> dict[str, float]:
        """BOM weight of each commodity in COGS. Sums to ~1.0."""
        # From settings.yaml commodities[].bom_weight_pct
        # e.g. {"steel": 0.35, "aluminum": 0.20, "lithium": 0.12, ...}
    
    def compute_shock(
        self,
        commodity: str,
        shock_pct: float,
        base_revenue: float,
        base_cogs_pct: float = 0.62,
        material_fraction: float = 0.45,
        tax_rate: float = 0.25,
    ) -> dict:
        """
        Returns:
            cogs_impact: float  (£ change in COGS)
            ebit_impact: float  (£ change in EBIT, after tax shield)
            margin_impact_bps: float (basis points change in gross margin %)
            pct_of_ebit: float  (impact as % of base EBIT)
        """
        bom_weight = self.bom_weights.get(commodity, 0.0)
        base_cogs = base_revenue * base_cogs_pct
        material_cogs = base_cogs * material_fraction
        
        cogs_impact = material_cogs * bom_weight * shock_pct
        ebit_impact = -cogs_impact  # COGS up → EBIT down
        after_tax_impact = ebit_impact * (1 - tax_rate)
        margin_impact_bps = (-cogs_impact / base_revenue) * 10_000
        
        return {
            "commodity": commodity,
            "shock_pct": shock_pct,
            "cogs_impact": cogs_impact,
            "ebit_impact": after_tax_impact,
            "margin_impact_bps": margin_impact_bps,
            "pct_of_base_ebit": after_tax_impact / (base_revenue * 0.10),  # assume 10% EBIT margin
        }
    
    def waterfall(self, shocks: dict[str, float], base_revenue: float) -> list[dict]:
        """
        Compute waterfall for multiple simultaneous shocks.
        Returns ordered list for waterfall chart rendering.
        """
        results = []
        for commodity, shock_pct in shocks.items():
            if abs(shock_pct) > 0.001:
                results.append(self.compute_shock(commodity, shock_pct, base_revenue))
        return sorted(results, key=lambda x: abs(x["ebit_impact"]), reverse=True)
```

### Task 2.2 — Hedge Optimizer
**New file:** `src/models/hedge_optimizer.py`

```python
from scipy.optimize import minimize_scalar
import numpy as np

class HedgeOptimizer:
    """
    Computes optimal hedge ratio for a commodity exposure.
    Balances: expected savings vs. hedge cost vs. VaR reduction.
    """
    
    def optimize(
        self,
        forecast_mean: float,      # ML forecast price
        forecast_std: float,       # uncertainty (from CI)
        futures_price: float,      # current futures price
        exposure_units: float,     # physical units exposed (tonnes, etc.)
        hedge_cost_bps: float = 30, # cost of hedging in basis points
        confidence: float = 0.95,
    ) -> dict:
        """
        Returns optimal hedge ratio h* in [0, 1].
        """
        def total_expected_cost(h):
            # Unhedged portion: exposed to forecast price
            unhedged_exposure = exposure_units * (1 - h)
            # Hedged portion: locked at futures price + cost
            hedged_cost = exposure_units * h * futures_price * (1 + hedge_cost_bps / 10_000)
            # Expected total cost
            expected_market_cost = unhedged_exposure * forecast_mean
            return expected_market_cost + hedged_cost
        
        def var_at_confidence(h):
            # At (1-confidence) worst case: price is mean + z * std
            z = 1.645  # 95% one-tail
            worst_price = forecast_mean + z * forecast_std
            unhedged = exposure_units * (1 - h) * worst_price
            hedged = exposure_units * h * futures_price * (1 + hedge_cost_bps / 10_000)
            return unhedged + hedged
        
        # Minimize: (alpha) * expected_cost + (1-alpha) * VaR
        alpha = 0.5
        result = minimize_scalar(
            lambda h: alpha * total_expected_cost(h) + (1 - alpha) * var_at_confidence(h),
            bounds=(0, 1),
            method='bounded'
        )
        h_star = result.x
        
        unhedged_var = var_at_confidence(0)
        hedged_var = var_at_confidence(h_star)
        unhedged_expected = total_expected_cost(0)
        hedged_expected = total_expected_cost(h_star)
        
        return {
            "optimal_hedge_ratio": round(h_star, 3),
            "expected_savings": unhedged_expected - hedged_expected,
            "var_reduction": unhedged_var - hedged_var,
            "hedge_cost": exposure_units * h_star * futures_price * (hedge_cost_bps / 10_000),
            "recommendation": f"Hedge {h_star*100:.0f}% of exposure",
        }
```

### Task 2.3 — Regime Detector
**New file:** `src/models/regime_detector.py`

```python
import numpy as np
import polars as pl
from enum import Enum

class Regime(str, Enum):
    MEAN_REVERTING = "mean_reverting"
    TRENDING       = "trending"
    VOLATILE       = "volatile"

class RegimeDetector:
    """
    Classifies commodity price regime using Hurst exponent.
    H < 0.45: mean-reverting (SARIMAX dominant)
    H > 0.55: trending      (XGBoost dominant)
    else:     volatile      (Scenarios dominant)
    """
    
    def detect(self, prices: np.ndarray, window: int = 24) -> dict:
        """
        Args:
            prices: Array of price values (at least 24 months recommended)
            window: Rolling window for Hurst estimation
        Returns:
            regime: Regime enum
            hurst: float
            ensemble_weights: dict[str, float]
        """
        h = self._hurst_exponent(prices[-window:])
        rolling_vol = np.std(np.diff(prices[-12:]) / prices[-13:-1])
        
        if h < 0.45:
            regime = Regime.MEAN_REVERTING
            weights = {"sarimax": 0.45, "xgboost": 0.20, "futures": 0.25, "scenarios": 0.10}
        elif h > 0.55:
            regime = Regime.TRENDING
            weights = {"sarimax": 0.15, "xgboost": 0.45, "futures": 0.30, "scenarios": 0.10}
        else:
            regime = Regime.VOLATILE
            weights = {"sarimax": 0.20, "xgboost": 0.20, "futures": 0.20, "scenarios": 0.40}
        
        return {
            "regime": regime,
            "hurst": round(h, 3),
            "rolling_vol_pct": round(rolling_vol * 100, 2),
            "ensemble_weights": weights,
            "confidence": "high" if abs(h - 0.5) > 0.1 else "medium",
        }
    
    def _hurst_exponent(self, prices: np.ndarray) -> float:
        """
        Estimate Hurst exponent using rescaled range (R/S) analysis.
        Returns float in [0, 1].
        """
        n = len(prices)
        if n < 8:
            return 0.5  # Not enough data; assume random walk
        
        lags = range(2, min(n // 2, 20))
        rs_values = []
        
        for lag in lags:
            chunks = [prices[i:i+lag] for i in range(0, n - lag, lag)]
            rs_chunk = []
            for chunk in chunks:
                mean = np.mean(chunk)
                deviations = np.cumsum(chunk - mean)
                R = np.max(deviations) - np.min(deviations)
                S = np.std(chunk)
                if S > 0:
                    rs_chunk.append(R / S)
            if rs_chunk:
                rs_values.append((lag, np.mean(rs_chunk)))
        
        if len(rs_values) < 2:
            return 0.5
        
        lags_arr = np.log([x[0] for x in rs_values])
        rs_arr   = np.log([x[1] for x in rs_values])
        H = np.polyfit(lags_arr, rs_arr, 1)[0]
        return float(np.clip(H, 0.01, 0.99))
```

### Task 2.4 — Wire Regime Detector into Commodity Forecast
**File:** `src/models/commodity_forecast.py`

In the `forecast` method, before computing ensemble:
```python
from src.models.regime_detector import RegimeDetector

# After loading price history:
detector = RegimeDetector()
regime_result = detector.detect(price_history.to_numpy())
weights = regime_result["ensemble_weights"]

# Replace hardcoded weights with regime_result["ensemble_weights"]
ensemble_forecast = (
    sarimax_forecast * weights["sarimax"] +
    xgb_forecast    * weights["xgboost"] +
    futures_forecast * weights["futures"] +
    scenario_forecast * weights["scenarios"]
)

# Store regime info for dashboard
self.last_regime = regime_result
```

### Task 2.5 — Add shock injection to FinancialModel
**File:** `src/drivers/financial_model.py`

Add a method:
```python
def apply_commodity_shock(
    self,
    base_pnl: pd.DataFrame,
    shocks: dict[str, float],      # {"lithium": 0.20, "steel": -0.05}
) -> pd.DataFrame:
    """
    Apply commodity shocks to a base P&L DataFrame.
    Returns modified P&L with shock effects.
    Used by the live dashboard sliders.
    """
    from src.models.commodity_shock import CommodityShockCalculator
    calc = CommodityShockCalculator()
    
    shocked_pnl = base_pnl.copy()
    total_revenue = base_pnl["net_revenue"].sum()
    
    for commodity, shock_pct in shocks.items():
        impact = calc.compute_shock(commodity, shock_pct, total_revenue)
        # Distribute COGS impact proportionally across months
        monthly_weight = base_pnl["net_revenue"] / total_revenue
        shocked_pnl["total_cogs"] += impact["cogs_impact"] * monthly_weight
    
    # Recompute downstream items
    shocked_pnl["gross_margin"] = shocked_pnl["net_revenue"] - shocked_pnl["total_cogs"]
    shocked_pnl["operating_income"] = (
        shocked_pnl["gross_margin"] - shocked_pnl["warranty_reserve"] - shocked_pnl["depreciation"]
    )
    shocked_pnl["tax"] = shocked_pnl["operating_income"].clip(lower=0) * shocked_pnl["tax_rate"].fillna(0.25)
    shocked_pnl["net_income"] = shocked_pnl["operating_income"] - shocked_pnl["tax"]
    
    return shocked_pnl
```

**Week 2 done when:** Can call `CommodityShockCalculator().compute_shock("lithium", 0.20, 30e9)` and get a dict with `ebit_impact` populated correctly.

---

## WEEK 3 — Dashboard Integration & Polish

### Task 3.1 — Commodity Intelligence page: Add Shock Sliders
**File:** `src/dashboard/pages/commodity_intelligence.py`

Add a new tab "Live Shock Calculator" with:
```python
with tab_shock:
    st.subheader("Commodity → P&L Impact Calculator")
    
    base_revenue = _get_base_revenue()  # from FinancialModel
    
    # Sliders for each commodity
    commodities = get_settings()["commodities"]
    shocks = {}
    cols = st.columns(3)
    for i, commodity in enumerate(commodities):
        with cols[i % 3]:
            shock = st.slider(
                f"{commodity['name']} (%)",
                min_value=-50, max_value=50, value=0, step=1,
                key=f"shock_{commodity['name']}"
            ) / 100
            shocks[commodity["name"].lower()] = shock
    
    # Compute combined impact
    calc = CommodityShockCalculator()
    waterfall_data = calc.waterfall(shocks, base_revenue)
    
    if waterfall_data:
        # Animated waterfall chart
        total_ebit_impact = sum(d["ebit_impact"] for d in waterfall_data)
        
        st.metric(
            label="Total EBIT Impact",
            value=f"£{total_ebit_impact/1e6:.1f}M",
            delta=f"{total_ebit_impact/base_revenue*100:.2f}% of revenue"
        )
        
        # Plotly waterfall
        fig = go.Figure(go.Waterfall(
            name="EBIT Impact",
            orientation="v",
            measure=["relative"] * len(waterfall_data) + ["total"],
            x=[d["commodity"] for d in waterfall_data] + ["Total EBIT"],
            y=[d["ebit_impact"]/1e6 for d in waterfall_data] + [total_ebit_impact/1e6],
            texttemplate="%{value:.1f}M",
            connector={"line": {"color": "rgb(63,63,63)"}},
        ))
        fig.update_layout(title="P&L Impact Waterfall (£M)", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Hedge recommendations
        st.subheader("Hedge Recommendations")
        optimizer = HedgeOptimizer()
        for d in waterfall_data[:3]:  # Top 3 by impact
            commodity = d["commodity"]
            exposure = base_revenue * 0.62 * 0.45 * calc.bom_weights.get(commodity, 0)
            hedge = optimizer.optimize(
                forecast_mean=1.0,  # relative to current price
                forecast_std=abs(d["shock_pct"]) * 0.5,
                futures_price=1.0,
                exposure_units=exposure,
            )
            st.write(f"**{commodity}:** {hedge['recommendation']} — "
                    f"saves £{hedge['expected_savings']/1e6:.1f}M, "
                    f"reduces VaR by £{hedge['var_reduction']/1e6:.1f}M")
```

### Task 3.2 — Commodity Intelligence page: Add Regime Panel
**File:** `src/dashboard/pages/commodity_intelligence.py`

In the existing commodity tab, add a Regime section:
```python
st.subheader("Market Regime Classification")
detector = RegimeDetector()

regime_rows = []
for commodity in commodities:
    prices = _load_price_series(commodity["name"])  # from commodity_df
    if prices is not None and len(prices) >= 12:
        result = detector.detect(prices)
        regime_rows.append({
            "Commodity": commodity["name"],
            "Regime": result["regime"].value.replace("_", " ").title(),
            "Hurst": result["hurst"],
            "Confidence": result["confidence"],
            "Dominant Model": max(result["ensemble_weights"], key=result["ensemble_weights"].get),
        })

if regime_rows:
    df = pd.DataFrame(regime_rows)
    # Color-code by regime
    st.dataframe(df.style.apply(
        lambda row: ["background-color: #d4edda" if row["Regime"] == "Mean Reverting"
                    else "background-color: #fff3cd" if row["Regime"] == "Trending"
                    else "background-color: #f8d7da" for _ in row],
        axis=1
    ))
```

### Task 3.3 — Executive Summary: Add P&L Fan Chart
**File:** `src/dashboard/pages/executive_summary.py`

Replace the single-line revenue trend with a Monte Carlo fan chart:
```python
def _render_pnl_fan_chart(base_pnl, n_sims=2000):
    """Fan chart showing P&L uncertainty over 12 months."""
    from src.simulation.monte_carlo import MonteCarloSimulator
    
    mc = MonteCarloSimulator()
    results = mc.run(base_pnl, n_simulations=n_sims)
    
    months = base_pnl["date"].dt.strftime("%b %Y")
    
    fig = go.Figure()
    
    # 95% CI band
    fig.add_trace(go.Scatter(
        x=months, y=results["p95_oi"] / 1e6,
        fill=None, line_color="rgba(0,100,255,0.2)", name="p95"
    ))
    fig.add_trace(go.Scatter(
        x=months, y=results["p5_oi"] / 1e6,
        fill="tonexty", line_color="rgba(0,100,255,0.2)",
        fillcolor="rgba(0,100,255,0.1)", name="80% CI"
    ))
    
    # 80% CI band
    fig.add_trace(go.Scatter(
        x=months, y=results["p90_oi"] / 1e6,
        fill=None, line_color="rgba(0,100,255,0.4)", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=months, y=results["p10_oi"] / 1e6,
        fill="tonexty", fillcolor="rgba(0,100,255,0.25)",
        line_color="rgba(0,100,255,0.4)", name="80% CI"
    ))
    
    # Mean line
    fig.add_trace(go.Scatter(
        x=months, y=results["mean_oi"] / 1e6,
        line=dict(color="blue", width=2), name="Forecast (Mean)"
    ))
    
    fig.update_layout(
        title="Operating Income — 12-Month Forecast with Uncertainty",
        yaxis_title="Operating Income (£M)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # VaR summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Annual EBIT (Mean)", f"£{results['annual_mean_oi']/1e9:.2f}B")
    col2.metric("VaR (95%)", f"-£{results['var_95']/1e6:.0f}M")
    col3.metric("CVaR (95%)", f"-£{results['cvar_95']/1e6:.0f}M")
```

### Task 3.4 — Monte Carlo: Add variance decomposition
**File:** `src/simulation/monte_carlo.py`

After running simulations, add decomposition output:
```python
def decompose_variance(self, results: np.ndarray) -> dict:
    """
    Decompose output variance by source: commodity, demand, FX.
    Run 3 partial simulations holding 2 sources fixed each time.
    """
    # Total variance
    total_var = np.var(results)
    
    # Variance from commodity only (hold demand + FX at mean)
    comm_only = self._run_partial(shock_demand=False, shock_fx=False)
    comm_var = np.var(comm_only)
    
    # Variance from demand only
    demand_only = self._run_partial(shock_commodity=False, shock_fx=False)
    demand_var = np.var(demand_only)
    
    # Variance from FX only
    fx_only = self._run_partial(shock_commodity=False, shock_demand=False)
    fx_var = np.var(fx_only)
    
    total_measured = comm_var + demand_var + fx_var
    
    return {
        "commodity_pct": round(comm_var / total_measured * 100, 1),
        "demand_pct":    round(demand_var / total_measured * 100, 1),
        "fx_pct":        round(fx_var / total_measured * 100, 1),
    }
```

**Week 3 done when:** Dashboard shows animated waterfall on slider movement, fan chart on executive summary, and regime table on commodity page.

---

## WEEK 4 — Demo Polish & Production Readiness

### Task 4.1 — Synthetic data quality check
Run the data pipeline and verify numbers look like a real automotive company:
```bash
python scripts/generate_data.py
python scripts/run_pipeline.py
```
Check in `src/dashboard/app.py`:
- Revenue should be in the £20–35B range (annual)
- Gross margin should be ~25–35%
- EBIT should be ~5–12%
- Commodity spend should be 40–50% of COGS

If numbers are off, adjust `config/settings.yaml`:
```yaml
vehicle_segments:
  - name: premium_suv
    annual_volume: 50000
    avg_price_usd: 65000
    ...
```

### Task 4.2 — FastAPI routes: verify all 4 key endpoints work
**File:** `src/api/routes/forecast.py`

Confirm these routes are implemented and return correct schemas:
```
POST /forecast/commodity   → uses CommodityForecastXGBoost, returns with regime_info
POST /forecast/demand      → uses DemandForecast, returns by segment
POST /simulation/scenario  → uses MonteCarloSimulator, returns with var_decomposition
POST /pnl/shock            → NEW: accepts {shocks: {"lithium": 0.20}}, returns waterfall
```

Add the shock endpoint if missing:
```python
@router.post("/pnl/shock")
def compute_pnl_shock(request: ShockRequest) -> ShockResponse:
    calc = CommodityShockCalculator()
    base_revenue = get_base_revenue_from_latest_pnl()
    waterfall = calc.waterfall(request.shocks, base_revenue)
    return ShockResponse(waterfall=waterfall, total_ebit_impact=sum(w["ebit_impact"] for w in waterfall))
```

### Task 4.3 — Add Pydantic schemas for new features
**File:** `src/api/schemas.py`

Add:
```python
class ShockRequest(BaseModel):
    shocks: dict[str, float]  # {"lithium": 0.20, "steel": -0.05}
    base_revenue: float | None = None  # optional override

class ShockWaterfallItem(BaseModel):
    commodity: str
    shock_pct: float
    cogs_impact: float
    ebit_impact: float
    margin_impact_bps: float

class ShockResponse(BaseModel):
    waterfall: list[ShockWaterfallItem]
    total_ebit_impact: float
    total_cogs_impact: float
    hedge_recommendations: list[dict]

class RegimeInfo(BaseModel):
    commodity: str
    regime: str  # mean_reverting | trending | volatile
    hurst: float
    ensemble_weights: dict[str, float]
    confidence: str
```

### Task 4.4 — Demo data: Make synthetic data look like JLR
**File:** `config/settings.yaml`

Ensure these calibrations match JLR's public financials:
```yaml
financial:
  base_cogs_pct: 0.775         # JLR: ~77.5% COGS ratio
  material_fraction: 0.45      # Materials = 45% of COGS
  warranty_reserve_pct: 0.018  # 1.8% of revenue
  tax_rate: 0.19               # UK corporation tax
  monthly_depreciation_usd: 95_000_000  # ~£1.14B/year

vehicle_segments:
  - name: defender
    annual_volume: 95000
    avg_price_usd: 55000
    incentive_pct: 0.04
  - name: discovery
    annual_volume: 60000
    avg_price_usd: 62000
    incentive_pct: 0.05
  - name: range_rover
    annual_volume: 110000
    avg_price_usd: 95000
    incentive_pct: 0.03
  - name: jaguar_ev
    annual_volume: 18000
    avg_price_usd: 75000
    incentive_pct: 0.08
```

### Task 4.5 — Smoke tests for new features
Add to `tests/test_commodity_forecast.py`:
```python
def test_regime_detector_hurst():
    from src.models.regime_detector import RegimeDetector
    det = RegimeDetector()
    # Random walk: H ≈ 0.5
    rw = np.cumsum(np.random.randn(100))
    result = det.detect(rw)
    assert 0.3 < result["hurst"] < 0.7
    assert result["regime"] in ["mean_reverting", "trending", "volatile"]

def test_shock_calculator():
    from src.models.commodity_shock import CommodityShockCalculator
    calc = CommodityShockCalculator()
    result = calc.compute_shock("lithium", 0.20, 30e9)
    assert result["ebit_impact"] < 0     # price up → EBIT down
    assert abs(result["ebit_impact"]) > 1e6  # at least £1M impact

def test_hedge_optimizer():
    from src.models.hedge_optimizer import HedgeOptimizer
    opt = HedgeOptimizer()
    result = opt.optimize(
        forecast_mean=12000, forecast_std=1200,
        futures_price=11500, exposure_units=50000
    )
    assert 0 <= result["optimal_hedge_ratio"] <= 1
    assert result["expected_savings"] > 0  # hedging should be worth something when price risk exists
```

### Task 4.6 — Run everything end to end before demo
```bash
# 1. Generate synthetic data
python scripts/generate_data.py

# 2. Fetch live market data (Yahoo Finance)
python scripts/fetch_data.py

# 3. Train models
python scripts/train_models.py

# 4. Run full pipeline
python scripts/run_pipeline.py

# 5. Run tests
pytest tests/ -v

# 6. Start API (keep running)
uvicorn src.api.app:create_app --factory --reload --port 8000

# 7. Start dashboard (separate terminal)
streamlit run src/dashboard/app.py --server.port 8501

# 8. Visit: http://localhost:8501
```

---

## Feature Priority If Time Runs Short

| Priority | Feature | Drop it if... |
|---|---|---|
| P0 — Must have | Shock Calculator (sliders → waterfall) | Never |
| P0 — Must have | Probabilistic P&L fan chart | Never |
| P0 — Must have | Plug-and-play config switch | Never |
| P1 — Should have | Hedge Optimizer panel | < 1 week left |
| P1 — Should have | Regime Detector table | < 1 week left |
| P2 — Nice to have | Variance decomposition | < 2 weeks left |
| P2 — Nice to have | Narrative explainability | < 2 weeks left |
| P3 — Cut | SAP stub implementation | Exists already |

---

## Common Mistakes to Avoid

**Don't** copy-paste the old `DataLoader` pattern into new features. Use `DataRouter.get_operational_source()` everywhere.

**Don't** hard-code revenue figures in the shock calculator. Always pull from `FinancialModel.build_pnl()` or the latest stored P&L.

**Don't** run 5000 Monte Carlo sims on every Streamlit re-render. Cache with `@st.cache_data(ttl=60)` and only re-run when parameters change.

**Don't** use `pd.DataFrame` in new code. Use `pl.DataFrame` (Polars) consistently; only convert to pandas at the Plotly/Streamlit boundary.

**Do** ensure the Hurst exponent calculation handles edge cases: fewer than 12 price points → return 0.5 (random walk assumption).

**Do** validate that the BOM weights in `CommodityShockCalculator` sum to approximately 1.0 for a typical vehicle. If not, the shock magnitudes will be wrong.
