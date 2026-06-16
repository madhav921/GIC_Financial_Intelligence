# **DATA GAP ANALYSIS & SYNTHETIC DATA GENERATION STRATEGY**
## **Jaguar Land Rover | GIC Platform Data Architecture**

**Document Version:** 1.0  
**Date:** April 20, 2026  
**Status:** Strategic Planning  

---

## **EXECUTIVE SUMMARY**

The GIC Platform is currently **data-complete at the architectural level** but **data-starved in production reality**. The system runs on synthetic data generated from mathematical distributions that approximate JLR's business, but lacks real production/sales/cost data.

This document:
1. **Maps the data universe:** What data exists (real vs. synthetic)
2. **Identifies critical gaps:** Which data types are missing or partial
3. **Defines approximation strategy:** How to synthetically model missing data with high fidelity
4. **Provides connector roadmap:** Phased integration of real JLR data from SAP, Salesforce, Treasury
5. **Validates synthetic calibration:** Reconciliation against FY2024 actual results

---

## **DATA INVENTORY**

### **Current Data Landscape**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA ECOSYSTEM: GIC PLATFORM                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TIER 1: REAL-WORLD DATA (Partial)                                │
│  ├─ Commodity Prices (Yahoo Finance) ✅ OPERATIONAL                │
│  │  └─ 12 commodities × 7 years daily data                        │
│  ├─ Macro Indicators (FRED placeholder) ⚠️ PARTIAL                │
│  │  └─ GDP, unemployment, yields available; some gaps              │
│  ├─ FX Rates (Yahoo Finance) ✅ OPERATIONAL                       │
│  │  └─ GBP/USD, EUR/USD, USD/JPY, USD/CNY                        │
│  ├─ Crypto (CCXT/Binance) ⚠️ PLACEHOLDER                         │
│  │  └─ Not currently used in JLR models                           │
│  └─ Market Indices (Yahoo Finance) ✅ OPERATIONAL                │
│     └─ S&P 500, DXY, Oil, Gold, VIX                              │
│                                                                     │
│  TIER 2: JLR OPERATIONAL DATA (Missing)                          │
│  ├─ Production Data ❌ NOT YET CONNECTED                         │
│  │  ├─ Units by vehicle segment (Premium SUV, Luxury SUV, etc.)  │
│  │  ├─ Plant utilization % by factory                            │
│  │  ├─ Manufacturing cost per unit (by plant, segment)           │
│  │  └─ Warranty defect rates by segment                          │
│  ├─ Sales Data ❌ NOT YET CONNECTED                              │
│  │  ├─ Units sold by segment, region, month                      │
│  │  ├─ ASP (Average Selling Price) by segment                    │
│  │  ├─ Incentive spend by segment                                │
│  │  └─ Sales channel mix (dealer, fleet, direct)                 │
│  ├─ Cost Data ❌ NOT YET CONNECTED                               │
│  │  ├─ COGS by segment (materials, labor, overhead)              │
│  │  ├─ Commodity cost per unit (e.g., steel kg/vehicle)          │
│  │  ├─ Fixed vs. variable cost breakdown                         │
│  │  └─ Supply chain costs (logistics, quality)                   │
│  ├─ Inventory Data ❌ NOT YET CONNECTED                          │
│  │  ├─ Raw material stock levels (steel, lithium, etc.)          │
│  │  ├─ WIP and finished goods inventory                          │
│  │  ├─ Supplier lead times by commodity                          │
│  │  └─ Obsolescence rates by segment                             │
│  ├─ Warranty Data ❌ NOT YET CONNECTED                           │
│  │  ├─ Monthly warranty claims by segment                        │
│  │  ├─ Cost per claim (labor + parts)                            │
│  │  ├─ Severity distribution (minor repairs vs. major)           │
│  │  └─ Claims trend by manufacturing year                        │
│  ├─ Capital Assets ❌ NOT YET CONNECTED                          │
│  │  ├─ CapEx schedule (plants, equipment, tooling)               │
│  │  ├─ Depreciation by asset class                               │
│  │  ├─ Asset retirement schedules                                │
│  │  └─ Plant capacity & planned expansions                       │
│  ├─ Treasury & Finance ❌ NOT YET CONNECTED                      │
│  │  ├─ Commodity hedging positions                               │
│  │  ├─ FX hedging positions                                      │
│  │  ├─ Debt covenants & interest rates                           │
│  │  ├─ Tax positions & provisions                                │
│  │  └─ GL accounts (all balance sheet & P&L)                     │
│  ├─ Supply Chain ❌ NOT YET CONNECTED                            │
│  │  ├─ Supplier financial health (credit ratings)                │
│  │  ├─ Single-source dependency risks                            │
│  │  ├─ Geopolitical supply disruption flags                      │
│  │  └─ Diversification strategies by commodity                   │
│  └─ HR & Headcount ❌ NOT YET CONNECTED                          │
│     ├─ Labor costs (hourly, salaried, overhead allocation)       │
│     ├─ Headcount by plant, function                              │
│     ├─ Union contract terms (wage escalation, benefits)          │
│     └─ Restructuring plans                                       │
│                                                                     │
│  TIER 3: SYNTHETIC DATA (Generated) 🤖                            │
│  ├─ Commodity Prices (if real data unavailable) ✅                │
│  │  └─ Ornstein-Uhlenbeck process for 3 commodities              │
│  ├─ Sales Data (Fully Synthetic) ✅                              │
│  │  ├─ Monthly units by segment (trend + seasonality)            │
│  │  └─ Calibrated to plausible JLR volumes                       │
│  ├─ Macro Indicators (Fully Synthetic) ✅                        │
│  │  ├─ 12 macro series (GDP, rates, PMI, DXY, etc.)              │
│  │  └─ Cross-correlated realistic behavior                       │
│  ├─ BOM Data (Fully Synthetic) ✅                                │
│  │  ├─ Commodity kg per vehicle by segment                       │
│  │  ├─ Segment-specific material usage                           │
│  │  └─ EV premiums (2.5× battery materials)                      │
│  ├─ Production/Inventory (Fully Synthetic) ✅                    │
│  │  ├─ Monthly production, sales, inventory, utilization         │
│  │  ├─ Warranty claims distribution                              │
│  │  └─ Lead time and stockout probabilities                      │
│  └─ Cost Drivers (Synthetic Approximations) ✅                   │
│     ├─ COGS as % of revenue                                      │
│     ├─ Commodity material fraction (45% of COGS)                 │
│     ├─ Warranty reserve as % of revenue (2%)                     │
│     └─ Tax rate (21% UK corporate tax)                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## **DETAILED GAP ANALYSIS**

### **Tier 1: Real-World Data (Status: Mostly Complete)**

#### **Commodity Prices** ✅ OPERATIONAL
- **Source:** Yahoo Finance (yfinance library)
- **Commodities:** 9/12 covered
  - ✅ Copper (HG=F)
  - ✅ Aluminum (AA)
  - ✅ Nickel (VALE proxy)
  - ✅ Platinum (PL=F)
  - ✅ Palladium (PA=F)
  - ✅ Natural Gas (NG=F)
  - ✅ Lithium (LIT)
  - ✅ Steel (SLX proxy)
  - ✅ Cobalt (GLNCY proxy)
  - ❌ Polypropylene (No direct ticker → synthetic fallback)
  - ❌ ABS Resin (No direct ticker → synthetic fallback)
  - ❌ Rhodium (Expensive/illiquid → synthetic fallback)

- **Data Quality:** 
  - Daily data available back 7 years
  - Intraday volatility captured
  - Good for input to commodity forecasting models

- **Gap:** 3 specialty materials (Polypropylene, ABS Resin, Rhodium) lack direct market data
  - **Synthetic Model:** Ornstein-Uhlenbeck with parameters calibrated to
    - Historical volatility ~2× that of commodity average
    - Mean reversion strength: 0.15 (low drift)
    - Result: Realistic daily returns within bounds of specialty plastics/PGMs market

---

#### **Macro Indicators** ⚠️ PARTIAL
- **Currently Working:**
  - GDP proxy (S&P 500 VIX implied volatility)
  - Unemployment (via FRED, but requires API key)
  - Interest rates (10-year yield available via yfinance)
  - CPI inflation (via FRED proxy)
  - Commodity indices (DXY, crude oil, gold)

- **Currently Stubbed:**
  - PMI (Purchasing Manager Index) — FRED requires subscription
  - Baltic Dry Index (shipping costs) — some days missing data
  - Consumer Sentiment — FRED subscription
  - Confidence Indices — requires multiple sources

- **Solution:** 
  - **Phase 1 (Months 1–2):** Add FRED API integration with API key management
  - **Phase 2 (Months 3–6):** Fallback to Bloomberg (if license exists)
  - **Fallback (Always Available):** Synthetic PMI/Baltic based on historical distributions

---

#### **FX Rates** ✅ OPERATIONAL
- **Covered Pairs:** GBP/USD, EUR/USD, USD/JPY, USD/CNY
- **Data Quality:** Daily rates, highly accurate
- **Gap:** None significant
- **Usage:** Used to convert commodity prices from USD to GBP for revenue/COGS impact

---

#### **Crypto Data** ⚠️ OPERATIONAL (Not JLR-Relevant)
- **Source:** CCXT/Binance
- **Symbols:** BTC, ETH, SOL, XRP, BNB, AVAX
- **Current Usage:** None (JLR doesn't manufacture cryptocurrency)
- **Future Potential:** Could proxy for market risk sentiment (crypto = high-risk assets)
- **Status:** Currently disconnected from JLR models (can remove from scope)

---

### **Tier 2: JLR Operational Data (Status: Completely Missing)**

This is the **critical gap**. Platform architecture supports these data sources but no connectors exist.

#### **Production Data** ❌ NOT CONNECTED
- **What's Needed:**
  - Monthly units produced by vehicle segment (Premium SUV, Luxury SUV, Performance, EV)
  - Plant utilization % (Solihull, Castle Bromwich, etc.)
  - Manufacturing cost per unit by plant, segment (material + labor + overhead allocation)
  - Warranty defect rates at birth by segment

- **Current Synthetic Model:**
  ```
  Production[segment, month] = Base_Volume × (1 + Trend) × Seasonality × Shock
  - Base volumes: Premium SUV 8K/mo, Luxury SUV 5K/mo, Performance 2K/mo, EV 1K/mo
  - Trend: 1–2% quarterly
  - Seasonality: Q4 +20%, Q1 -15% (automotive calendar)
  - Shock: None (deterministic)
  ```

- **Real Data Source:** SAP Production Planning (PP module)
  - Tables: AUFK (orders), RESB (reservations), IOOB (inventory by plant)
  - Frequency: Daily (can aggregate to monthly)
  - Availability: Likely Yes (SAP HANA accessible)
  - Effort to Connect: 2–3 weeks (SAP specialist)

- **Validation Approach:**
  - Compare synthetic volumes vs. FY2024 actual production
  - Calibrate Trend + Seasonality to match actuals
  - Benchmark utilization % against industry (target: 75–85% is typical for auto)

---

#### **Sales Data** ❌ NOT CONNECTED
- **What's Needed:**
  - Monthly units sold by vehicle segment, region (EMEA, APAC, AMERICAS, China)
  - ASP (Average Selling Price) by segment, month
  - Incentive spend ($M/month) by segment
  - Channel mix: dealer sales % vs. fleet vs. direct

- **Current Synthetic Model:**
  ```
  Sales[segment, month] = Production[segment, month] × (1 + Demand_Shock)
  - Demand shocks: Normal(0, 0.08) = 8% standard deviation
  - Regional allocation: EMEA 40%, APAC 25%, AMERICAS 20%, China 15%
  - ASP: Premium SUV £65K, Luxury SUV £75K, Performance £85K, EV £55K
  - Incentive: 8–12% of ASP by segment
  ```

- **Real Data Source:** Salesforce (CRM system) + Finance GL (revenue recognition)
  - Tables: Salesforce Objects (Opportunity, Order), Finance GL Account 4000–4999 (revenue)
  - Frequency: Daily (order capture), monthly GL posting
  - Availability: Likely Yes
  - Effort to Connect: 2–3 weeks (Salesforce admin + finance analyst)

- **Validation Approach:**
  - Extract reported sales units from FY2024 GL + Salesforce history
  - Compare synthetic volumes vs. actual
  - Calibrate ASP + incentive to match reported net revenue

---

#### **Cost Data** ❌ NOT CONNECTED
- **What's Needed:**
  - COGS by segment, month (broken down: materials, labor, factory overhead)
  - Commodity cost per unit (kg of steel per vehicle, etc.)
  - Material scrap/waste rates
  - Labor hours per vehicle by segment
  - Hourly labor rates (union vs. salaried)
  - Factory fixed costs (rent, utilities, depreciation allocated to manufacturing)

- **Current Synthetic Model:**
  ```
  COGS = Revenue × Base_COGS_Pct + Commodity_Impact
  - Base_COGS_Pct: 60% (industry standard for automotive)
  - Commodity_Impact: (Commodity_Index - Base) × Material_Fraction × Revenue
  - Material_Fraction: 45% (commodities represent 45% of base COGS)
  - By segment: Premium SUV 58% COGS, Luxury 62%, Performance 55%, EV 70% (battery costs)
  ```

- **Real Data Sources:**
  - **Material Costs:** SAP MM (Materials Management), cost accounting (COPC tables)
  - **Labor:** HR system (SAP HCM), union contracts, timekeeping
  - **Factory Overhead:** SAP CO (Controlling), cost center allocation
  - **Supplier Invoices:** SAP AP (Accounts Payable), logistics costs from supply chain system
  - Frequency: Monthly (cost accounting close)
  - Availability: Likely Yes (standard SAP modules)
  - Effort to Connect: 4–6 weeks (SAP Controlling specialist + supply chain analyst)

- **Validation Approach:**
  - Reconstruct FY2024 COGS from GL detail (materials, labor, overhead)
  - Compare to synthetic model
  - Identify material fraction (what % of COGS is commodity-driven?)
  - Measure labor cost per unit, utilization impact

---

#### **Inventory Data** ❌ NOT CONNECTED
- **What's Needed:**
  - Raw material inventory (days of supply by commodity)
  - WIP (Work-in-Process) inventory (units, days)
  - Finished goods inventory (units by segment, plant)
  - Supplier lead times by commodity (days)
  - Obsolescence rates (% of inventory scrapped annually)
  - Safety stock levels (current vs. modeled optimal)

- **Current Synthetic Model:**
  ```
  Inventory[month] = Sales[month] × Days_of_Supply + Safety_Stock
  - Days_of_Supply: 60 days (2 months of supply, typical for auto)
  - Safety_Stock: 15% of monthly sales (buffer for demand spikes)
  - Stockout_Probability: Calculated from Normal demand distribution
  - Lead_Time: 45 days for most commodities (some PGMs 60–90 days)
  ```

- **Real Data Source:** SAP MM (Materials Management), SAP PM (Plant Maintenance)
  - Tables: MARD (stock by storage location), MARC (material master), EICP (purchase requisitions)
  - Frequency: Daily (material movements), monthly accounting close
  - Availability: Likely Yes
  - Effort to Connect: 2–3 weeks (SAP logistics/supply chain analyst)

- **Validation Approach:**
  - Extract actual inventory balances from FY2024 month-end
  - Calculate actual days of supply by commodity
  - Compare to synthetic assumption (60 days)
  - Identify true safety stock levels + lead times

---

#### **Warranty Data** ❌ NOT CONNECTED
- **What's Needed:**
  - Monthly warranty claims (units) by segment
  - Cost per claim (labor + parts) by segment, claim severity
  - Claims trend by manufacturing year (older vehicles = higher defects)
  - Regional warranty exposure (some regions have higher claim rates)

- **Current Synthetic Model:**
  ```
  Warranty_Reserve = Net_Revenue × Warranty_Reserve_Rate
  - Warranty_Reserve_Rate: 2.0% (JLR industry benchmark: 1.5–2.5%)
  - Claims distribution: Lognormal(mean_claim=£450, std=£300)
  - Segment-specific rates: Premium SUV 1.8%, Luxury 2.2%, Performance 1.5%, EV 2.5% (new tech)
  ```

- **Real Data Source:**
  - Warranty Management System (SAP SCM WM or dedicated warranty platform)
  - GL warranty provision accounts
  - Frequency: Monthly claim processing, monthly accrual
  - Availability: Likely Yes (warranty is material for automotive)
  - Effort to Connect: 3–4 weeks (warranty team + finance analyst)

- **Validation Approach:**
  - Extract FY2024 actual warranty costs from GL
  - Calculate warranty cost as % of revenue by segment
  - Compare to synthetic 2.0% assumption
  - Build actual claims distribution (mean, std, tail risk)

---

#### **Capital Assets** ❌ NOT CONNECTED
- **What's Needed:**
  - CapEx schedule (plants, equipment, tooling) by year, asset class
  - Depreciation schedule (straight-line, useful lives, residual values)
  - Asset retirement schedule (plant closures, equipment end-of-life)
  - Planned CapEx commitments (EV battery plant, paint shop modernization, etc.)

- **Current Synthetic Model:**
  ```
  Depreciation[month] = Fixed_Rate × Gross_Assets
  - Fixed_Rate: 0.5% monthly (6% annual, typical for auto manufacturing)
  - Sample_CapEx: Battery Plant £500M/10yr, Paint £120M/8yr, Robotics £80M/7yr, Press £45M/12yr
  - Total Depreciation: ~£40M/month (illustrative)
  ```

- **Real Data Source:** SAP AA (Asset Accounting), PP (Plant Planning)
  - Tables: ANLA (asset ledger), ANLC (depreciation), ANKA (retirement)
  - Frequency: Annual (asset plan), monthly (depreciation accrual)
  - Availability: Definitely Yes (standard SAP module)
  - Effort to Connect: 1–2 weeks (SAP asset accounting specialist)

- **Validation Approach:**
  - Extract actual depreciation from FY2024 GL (typically GL 6200–6299)
  - Compare to synthetic assumption
  - Build actual CapEx pipeline (announced/committed investments)
  - Project depreciation for next 3 years based on schedule

---

#### **Treasury & Finance** ❌ NOT CONNECTED
- **What's Needed:**
  - **Hedging positions:** Commodity hedges (forward contracts, futures, options), FX hedges
  - **Debt:** Outstanding debt balance, interest rates, maturity schedule, covenant levels
  - **Tax:** Tax rate, provisions, deferred tax assets/liabilities
  - **GL Detail:** Full chart of accounts (revenue, COGS, SG&A, operating expenses, interest, tax)

- **Current Synthetic Model:**
  ```
  Operating_Income = Gross_Margin - Warranty - Depreciation
  Tax = Operating_Income × Tax_Rate (21%)
  Net_Income = Operating_Income - Tax
  (No hedging, no debt interest, no provisions modeled)
  ```

- **Real Data Source:**
  - SAP FM (Financial Management) / GL
  - Treasury Management System (TMS) for hedge positions
  - Debt agreement schedules
  - Tax accounting system
  - Frequency: Daily (cash position), monthly (close), annual (debt schedule)
  - Availability: Definitely Yes
  - Effort to Connect: 3–4 weeks (finance controller + treasury team)

- **Validation Approach:**
  - Extract actual interest expense, tax, and operating items from FY2024 P&L
  - Compare implied tax rate, debt interest rate
  - Validate debt covenant calculations (leverage ratio, interest coverage)
  - Map hedging gains/losses to commodity impact

---

#### **Supply Chain** ❌ NOT CONNECTED
- **What's Needed:**
  - Supplier financial health (credit ratings, financial statements, default risk)
  - Commodity sourcing concentrations (% from top 3 suppliers)
  - Geopolitical supply risk flags (e.g., cobalt from DRC, lithium from China)
  - Alternative supplier availability (dual-source cost premiums)
  - Strategic diversification plans (e.g., European lithium mining partnerships)

- **Current Synthetic Model:**
  ```
  Supply_Chain[status] = All_suppliers_healthy (no disruption modeled)
  - Single-source risk: Not modeled
  - Geopolitical shocks: Not modeled
  - Lead time variability: Fixed 45 days (no shock distribution)
  ```

- **Real Data Source:**
  - Supply Chain Finance System (e.g., Coupa, Ariba, or custom ERP module)
  - Strategic sourcing team database (supplier scorecards, diversification plans)
  - Treasury counterparty risk system
  - Frequency: Quarterly (supplier reviews), event-driven (disruptions)
  - Availability: Likely Yes (strategic sourcing typically has supplier database)
  - Effort to Connect: 3–4 weeks (sourcing director + data analyst)

- **Validation Approach:**
  - Map current supplier concentrations by commodity
  - Identify single-source dependencies (>30% from one supplier)
  - Assess geopolitical exposure (which materials sourced from geopolitically risky regions?)
  - Model supply chain shock scenarios (top supplier bankruptcy, logistics disruption)

---

#### **HR & Headcount** ❌ NOT CONNECTED
- **What's Needed:**
  - Headcount by plant, function (manufacturing, engineering, sales, admin)
  - Labor costs (hourly rate, salaried salary, benefits, payroll taxes)
  - Union contracts (wage escalation, benefit provisions, layoff restrictions)
  - Planned restructuring (plant closures, headcount reductions, outsourcing)
  - Labor productivity (units per FTE)

- **Current Synthetic Model:**
  ```
  Labor_Cost = Fixed % of COGS (embedded in 60% COGS assumption)
  - No labor cost inflation modeled
  - No headcount variability
  - No restructuring scenarios
  ```

- **Real Data Source:**
  - SAP HCM (Human Capital Management)
  - Payroll system
  - Union contract documents
  - Headcount plan / HR forecast
  - Frequency: Monthly (payroll), annual (headcount planning)
  - Availability: Likely Yes
  - Effort to Connect: 2–3 weeks (HR analytics + finance team)

- **Validation Approach:**
  - Extract actual labor costs from FY2024 GL
  - Calculate labor cost per FTE, per unit
  - Verify union contract wage escalation terms
  - Map to planned restructuring (which plants closing? when?)

---

## **SYNTHETIC DATA CALIBRATION STRATEGY**

### **Philosophy**
Use synthetic data as a **bridge until real data is connected**. The goal is not perfect accuracy but **directional correctness**:
- Order of magnitude correct (e.g., COGS 60% of revenue, not 40% or 80%)
- Seasonality patterns realistic (automotive strong in Q4, weak in Q1)
- Volatility realistic (commodity ±15%, demand ±8%)
- Cross-correlations reasonable (higher oil prices → higher transportation costs)

### **Calibration Process**

#### **Step 1: Establish Real Data Baseline** (Month 1)
```
Source: Finance GL + Salesforce history
Extract:
- FY2024 Reported P&L: Revenue, COGS, Gross Margin, Operating Income, Net Income
- FY2024 by Segment: Units sold, Revenue, Gross Margin %
- FY2024 by Month: Revenue, COGS, Gross Margin trend
- Historical Volatility: Monthly growth rate std deviation

Result: "Target P&L" to benchmark synthetic model against
```

#### **Step 2: Run Synthetic Model** (Month 1)
```
Input: Current synthetic data generation (Ornstein-Uhlenbeck commodity, normal demand, etc.)
Output: Synthetic P&L for FY2024 time period

Compare: Synthetic P&L vs. Real P&L
- Total revenue: Error = (Synthetic - Real) / Real
- COGS: Error %
- Gross margin: Error %
- By segment: Compare synthetic vs. real unit volumes, ASP, margin %
```

#### **Step 3: Adjust Parameters** (Month 1–2)
```
If synthetic COGS 55% but real COGS 60%, adjust:
- material_fraction from 40% to 45%
- base_cogs_pct from 58% to 60%
- Warranty reserve from 1.5% to 2.0%

If synthetic demand volatility 15% but real is 8%, adjust:
- Normal(0, 0.08) instead of Normal(0, 0.15)

Iterate until synthetic P&L within 5% of real P&L for full year
```

#### **Step 4: Cross-Validate by Segment** (Month 2)
```
For each segment (Premium SUV, Luxury SUV, Performance, EV):
- Compare synthetic volumes vs. real FY2024 sales
- Compare synthetic ASP vs. real average transaction price
- Compare synthetic margin % vs. real margin % (if segmented in GL)

Calibrate segment-specific parameters:
- EV: Higher COGS (battery), lower ASP (price parity target), lower volume (nascent)
- Performance: Lower volume, higher margin, more commodity-sensitive
- Luxury: Higher ASP, higher margin %, mature demand pattern
- Premium: Mid-range on all metrics
```

#### **Step 5: Scenario Sensitivity Validation** (Month 2–3)
```
Run historical scenarios:
- Oil price shock (2022): Model should show COGS increase, margin compression
- EV demand surge (2023): Model should show EV volume lift, overall margin mix shift
- Lithium spike (2021): Model should show COGS increase for EV, less impact for ICE

Compare model outputs to actual P&L impact from that period
- If lithium doubled in Q3 2022 and actual EV margin fell 300 bps, synthetic model should predict similar
```

---

## **PHASED DATA INTEGRATION ROADMAP**

### **Phase 1: Data Audit & Validation (Months 1–2)**

**Workstreams:**

| System | Owner | Effort | Deliverable |
|--------|-------|--------|-------------|
| **SAP Production (PP)** | Supply Chain Lead | 2 weeks | Data extraction script, FY2024 production by segment/plant |
| **Salesforce Sales** | Sales Finance | 2 weeks | Sales units, ASP, incentive by segment/region/month |
| **SAP HANA COGS** | Controller | 3 weeks | Material, labor, overhead costs by segment; COGS % validation |
| **SAP Inventory (MM)** | Logistics Manager | 2 weeks | Days of supply, lead times, safety stock by commodity |
| **Warranty Database** | Quality/Warranty Team | 2 weeks | Claims volume, cost per claim, trend by segment |
| **SAP Asset Accounting** | FP&A | 1 week | CapEx schedule, depreciation, asset list |
| **Treasury System** | Treasurer | 2 weeks | Hedging positions, debt schedule, interest costs |
| **Supply Chain Database** | Sourcing Director | 2 weeks | Supplier concentrations, geopolitical flags |
| **HR / Payroll** | CHRO | 2 weeks | Headcount, labor costs, union contracts |

**Deliverables:**
- [ ] Data Quality Report: % completeness, timeliness, outliers for each system
- [ ] FY2024 P&L Reconstruction: Full GL detail extracted, reconciled to reported
- [ ] Synthetic Calibration Baseline: Identify parameter adjustments needed

---

### **Phase 2: Connector Development (Months 2–4)**

#### **Priority 1 (Weeks 1–4): Production + Sales + COGS**
```
Why: These drive 90% of P&L variance
Development:
- SAP Data Extract (MM/COPC/PP): Load production, COGS, by segment/month
- Salesforce API: Pull sales units, revenue, incentive
- Reconciliation logic: Production → Sales → Inventory flow validation

Test against FY2024 data:
- Production volume by segment: Match within 2%
- Revenue by segment: Match within 1%
- COGS by segment: Match within 3%

Outcome: Real volume + revenue + COGS data feeding forecasting models
```

#### **Priority 2 (Weeks 5–8): Inventory + Warranty + Capital**
```
Why: Secondary P&L drivers + important for scenario modeling (CapEx impact on depreciation)
Development:
- SAP MM: Inventory by commodity, days of supply
- Warranty system: Monthly claims, cost per claim
- SAP AA: CapEx schedule, depreciation detail

Outcome: Inventory risk, warranty reserve, depreciation accuracy improved
```

#### **Priority 3 (Weeks 9–12): Treasury + Supply Chain + HR**
```
Why: Risk management + strategic planning
Development:
- Treasury: Hedge positions, debt, interest costs
- Supply Chain: Supplier concentrations, lead times, geopolitical flags
- HR: Headcount, labor costs, restructuring plans

Outcome: Risk dashboard populated, supply chain scenario modeling enabled
```

---

### **Phase 3: Real Data Operationalization (Months 4–6)**

**Activities:**
- [ ] Switch production forecasts from synthetic to real historical data
- [ ] Backtest demand forecast models on real sales history
- [ ] Validate price elasticity model with real pricing/volume data
- [ ] Recalibrate P&L engine using actual COGS, warranty, depreciation
- [ ] Publish monthly forecast using real data as baseline

**Governance:**
- [ ] Data lineage documented (which GL accounts feed which forecasts?)
- [ ] SLAs for data freshness (e.g., production data loaded by 5th business day after month-end)
- [ ] Data quality monitoring dashboard (% completeness, timeliness alerts)

---

## **SYNTHETIC DATA QUALITY STANDARDS**

### **Acceptable Ranges for Synthetic Approximations**

When real data is unavailable, synthetic approximations must meet these standards:

| Metric | Real Data | Synthetic | Acceptable Error |
|--------|-----------|-----------|------------------|
| **Revenue** | Actual GL | Synthetic | <2% of annual |
| **COGS** | GL detail | Synthetic | <3% of annual |
| **Gross Margin %** | Calculated | Synthetic | <200 bps |
| **Volume by Segment** | Salesforce | Synthetic | <5% annual; <8% monthly |
| **ASP by Segment** | Revenue/Volume | Synthetic | <3% |
| **Warranty % Revenue** | GL + Claims | Synthetic | <50 bps (e.g., 2.0% ± 0.5%) |
| **Days of Supply** | MM system | Synthetic | <10 days (e.g., 60 ± 10) |
| **COGS Volatility** | Historical std | Synthetic | <1% per annum |

### **Forbidden Practices**
- ❌ Use synthetic data for capital allocation decisions (use real data or scenario ranges only)
- ❌ Publish synthetic forecasts to external stakeholders (always use real data baseline for external guidance)
- ❌ Mix synthetic + real data without clear labels (auditors require transparency)
- ❌ Run models on synthetic data beyond 12-month horizon (use scenarios instead)

### **Acceptable Practices**
- ✅ Use synthetic data for model development / testing (before real data available)
- ✅ Use synthetic data for scenario / sensitivity analysis (labeled as "illustrative")
- ✅ Use synthetic data to fill short-term gaps (e.g., if April actuals delayed, use April synthetic forecast)
- ✅ Use synthetic data for performance benchmarking (compare model accuracy vs. synthetic baseline)

---

## **DATA QUALITY METRICS**

### **Completeness**
```
Completeness % = (Records with all required fields) / (Total records) × 100

Target: >95% for production, sales, cost data
Risk: If <90%, model outputs become unreliable
Action: Investigate missing data source; implement data validation in SAP
```

### **Timeliness**
```
Timeliness % = (Data loaded by SLA date) / (Total months) × 100

Target: >95% (e.g., production data loaded by 5th business day after month-end)
Risk: If timeliness <80%, forecasts become stale
Action: Implement SAP batch job scheduling, escalation alerts
```

### **Accuracy (for Synthetic Data)**
```
Accuracy = 1 - (Synthetic - Real) / Real) (absolute value)

Target: >95% (i.e., within 5% of real data)
Risk: If <90%, model outputs biased
Action: Recalibrate synthetic parameters against real data
```

### **Consistency**
```
Consistency = (Cross-checks passed) / (Total cross-checks) × 100

Example cross-checks:
- Production = Sales + Δ Inventory (should balance)
- Revenue units × ASP = GL revenue (should reconcile within rounding)
- COGS components (materials + labor + OH) = GL COGS (should match)

Target: 100%
Risk: Any imbalance indicates data quality issue
Action: Investigate imbalance; correct source data
```

---

## **SYNTHETIC DATA GENERATOR CODE STRUCTURE**

Current location: `src/data/synthetic_generator.py`

### **Key Functions**

#### **1. Commodity Prices (Ornstein-Uhlenbeck Process)**
```python
def generate_commodity_prices(seed=42, months=84):
    """
    Generates realistic commodity price time series for 12 JLR materials
    
    Parameters:
    - seed: Reproducibility
    - months: 84 = 7 years (2019–2026)
    
    Output: DataFrame with columns [date, commodity, price_usd_per_unit]
    
    Model:
    dP/dt = -κ(P - μ) dt + σ dW
    - κ: Mean reversion speed (0.05–0.15 per month)
    - μ: Long-term mean price
    - σ: Volatility (5–20% annualized)
    - dW: Random shock
    
    Calibration:
    - κ: Metals ~0.08 (slow mean reversion), Energy ~0.12 (faster)
    - μ: Based on current spot prices
    - σ: From historical 5-year daily returns
    """
```

#### **2. Sales Data (Trend + Seasonality + Noise)**
```python
def generate_sales_data(seed=42, months=84):
    """
    Generates monthly vehicle sales by segment
    
    Model:
    Sales[t] = BaseVolume × Trend[t] × Seasonality[t] × (1 + Noise[t])
    - BaseVolume: Premium SUV 8K/mo, Luxury 5K, Performance 2K, EV 1K
    - Trend: Linear + quarterly (EV ramps +15%/yr, ICE flat to -2%/yr)
    - Seasonality: Q4 +20%, Q1 -15%, others neutral
    - Noise: Normal(0, 0.08) = 8% monthly volatility
    
    Calibration:
    - Base volumes: Set to JLR FY2024 actual monthly average (from audit)
    - Trend: Align to announced strategy (EV growth target)
    - Seasonality: Match dealer inventory replenishment cycle
    """
```

#### **3. Macro Indicators (Correlated Random Walks)**
```python
def generate_macro_indicators(seed=42, months=84):
    """
    Generates 12 macro series with realistic correlations
    
    Series: GDP, CPI, Unemployment, Interest Rates, Oil, PMI, DXY, 
            Baltic Dry Index, EV Penetration, Consumer Sentiment
    
    Model: Multivariate normal with calibrated covariance matrix
    - Positive correlations: (Oil, DXY), (Unemployment, CPI in stagflation), 
                             (GDP, DXY if USD strong)
    - Negative correlations: (PMI, Unemployment), (EV penetration, Oil price)
    
    Calibration: Covariance matrix from actual 2015–2025 data
    """
```

#### **4. BOM Data (Bill of Materials)**
```python
def generate_bom_data(seed=42):
    """
    Maps commodity kg per vehicle by segment
    
    Assumptions:
    - Average vehicle weight: 1500 kg
    - Commodity content varies by segment + material
    - EV: +2.5× battery metals (lithium, cobalt, nickel)
    
    Example (for Premium SUV):
    - Steel: 900 kg (60% of weight)
    - Aluminum: 150 kg (10%)
    - Copper: 25 kg (wiring, motors)
    - Lithium: 0.8 kg (if EV)
    - etc.
    
    Calibration: Aligned to industry benchmarks + JLR product specs
    """
```

---

## **RECOMMENDED NEXT STEPS**

### **Immediate (Month 1)**
1. [ ] Schedule data audit kickoff with SAP owner + Finance team
2. [ ] Assign data owners for each 9 systems (Production, Sales, Costs, etc.)
3. [ ] Extract FY2024 actual P&L from GL; reconcile to reported results
4. [ ] Benchmark synthetic P&L against actual (identify biggest gaps)

### **Short-term (Months 1–2)**
1. [ ] Complete data quality assessment (% completeness, timeliness)
2. [ ] Document current data lineage (GL account → Forecast input)
3. [ ] Adjust synthetic parameters to match FY2024 actuals within 5%
4. [ ] Prepare "Real vs. Synthetic" comparison dashboard for CFO

### **Medium-term (Months 2–4)**
1. [ ] Develop SAP/Salesforce connectors (Priority 1: Production, Sales, COGS)
2. [ ] Test real data feeds on Q1 2026 actuals
3. [ ] Rerun demand/commodity models using real sales history
4. [ ] Backtest financial model against FY2024 (how well did model predict P&L?)

### **Long-term (Months 4–12)**
1. [ ] Complete all 9 data connectors (Priority 2 & 3)
2. [ ] Operationalize real data in monthly close process
3. [ ] Retire synthetic data generation (use only for scenario testing)
4. [ ] Publish monthly GIC Dashboard to finance leadership (using real data)

---

## **SUMMARY TABLE: Data Gap Status**

| Data Source | Current Status | Priority | Real Data Available | Effort | Impact on P&L |
|-------------|--------|----------|---------------------|--------|--------------|
| **Commodity Prices** | ✅ Connected | P0 | 9/12 | Done | High |
| **Production Volumes** | ❌ Synthetic | P1 | Yes (SAP) | 2w | High |
| **Sales Volumes** | ❌ Synthetic | P1 | Yes (Salesforce) | 2w | High |
| **COGS/Materials** | ❌ Synthetic | P1 | Yes (SAP GL) | 3w | High |
| **ASP/Pricing** | ❌ Synthetic | P1 | Yes (GL revenue) | 1w | High |
| **Inventory Levels** | ❌ Synthetic | P2 | Yes (SAP MM) | 2w | Medium |
| **Warranty Costs** | ❌ Synthetic | P2 | Yes (GL + WM sys) | 2w | Medium |
| **CapEx/Depreciation** | ❌ Synthetic | P2 | Yes (SAP AA) | 1w | Medium |
| **Hedging Positions** | ❌ None | P3 | Yes (Treasury) | 2w | Medium |
| **Supply Chain Risk** | ❌ None | P3 | Yes (Sourcing DB) | 2w | Medium |
| **Labor Costs** | ❌ Synthetic | P3 | Yes (HR/Payroll) | 2w | Low–Medium |
| **Macro Indicators** | ⚠️ Partial | P1 | Yes (FRED) | 2w | Medium |

---

**Document Owner:** Data Analytics Team  
**Approved By:** [CFO/CIO Name]  
**Next Review:** Month 2 (post-audit completion)
