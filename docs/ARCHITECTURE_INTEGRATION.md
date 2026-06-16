# **ARCHITECTURE & INTEGRATION BLUEPRINT**
## **GIC Platform | Technical Infrastructure & Data Connectors**

**Document Version:** 1.0  
**Date:** April 20, 2026  
**Status:** Technical Design  

---

## **EXECUTIVE SUMMARY**

This document details:
1. **System Architecture** — How the GIC Platform's 5 layers integrate
2. **Data Flow Pipelines** — Real JLR data → Forecasting → P&L → Risk outputs
3. **Integration Points** — Connectors to SAP, Salesforce, Treasury, Supply Chain
4. **Deployment Model** — On-prem, containerization, scaling
5. **Governance & Audit** — Immutable logs, data lineage, compliance controls

---

## **OVERALL ARCHITECTURE**

### **The GIC Platform Stack (5 Layers)**

```
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 5: PRESENTATION & GOVERNANCE                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐ │
│  │  Streamlit      │  │  FastAPI REST    │  │  Audit Dashboard   │ │
│  │  Dashboard      │  │  API Endpoints   │  │  (JSONL Viewer)    │ │
│  │                 │  │                  │  │                    │ │
│  │- 7 pages        │  │- /health         │  │- Forecast history  │ │
│  │- Real-time      │  │- /forecast       │  │- Override log      │ │
│  │- Scenario       │  │- /simulation     │  │- Bias tracking     │ │
│  │  comparison     │  │- /pnl            │  │- Explainability    │ │
│  └────────┬────────┘  └────────┬─────────┘  └──────────┬─────────┘ │
│           │                    │                        │           │
└───────────┼────────────────────┼────────────────────────┼───────────┘
            │                    │                        │
            └────────────────────┼────────────────────────┘
                                 │
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 4: SIMULATION & RISK QUANTIFICATION                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐  ┌────────────────────────────────────────┐  │
│  │ Monte Carlo      │  │ Scenario Engine (7 Presets)           │  │
│  │ Simulator        │  │                                        │  │
│  │                  │  │ - Base Case                            │  │
│  │- 5000 sims      │  │ - Commodity Bull                       │  │
│  │- Fat-tailed      │  │ - Lithium Shock                       │  │
│  │  shocks          │  │ - Demand Collapse                     │  │
│  │- VaR/CVaR       │  │ - Perfect Storm                        │  │
│  │  metrics         │  │ - EV Acceleration                     │  │
│  │- Risk KPIs       │  │ - Restructuring                       │  │
│  └────────┬─────────┘  └────────────┬───────────────────────────┘  │
│           │                        │                              │
└───────────┼────────────────────────┼──────────────────────────────┘
            │                        │
            └────────────┬───────────┘
                         │
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 3: FINANCIAL DRIVER MODEL (DETERMINISTIC)                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │ Revenue Model  │  │ Cost Model    │  │ Financial Close      │  │
│  │                │  │               │  │                      │  │
│  │ Vol × ASP ×    │  │ COGS =        │  │ - Gross Margin       │  │
│  │ (1-Incentive)  │  │ f(Vol, Comm,  │  │ - Warranty Reserve   │  │
│  │                │  │ Utilization)  │  │ - Depreciation       │  │
│  │ Elasticity     │  │               │  │ - Operating Income   │  │
│  │ adjustment     │  │ Material %    │  │ - Tax & Net Income   │  │
│  │                │  │ commodity     │  │                      │  │
│  │ By segment,    │  │ shock         │  │ Month-by-month &     │  │
│  │ region, month  │  │               │  │ annual summaries     │  │
│  └────────┬───────┘  └───────┬───────┘  └──────────┬───────────┘  │
│           │                  │                    │              │
└───────────┼──────────────────┼────────────────────┼──────────────┘
            │                  │                    │
            └──────────────────┼────────────────────┘
                               │
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 2: PREDICTIVE MODELS (AI/ML)                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────┐  │
│  │ Demand Model   │  │ Commodity      │  │ Price Elasticity &  │  │
│  │ (XGBoost)      │  │ Forecast       │  │ Other Models        │  │
│  │                │  │ (4-method)     │  │                      │  │
│  │ Vol forecast   │  │                │  │ - Price Elasticity   │  │
│  │ by segment     │  │ SARIMAX        │  │ - Inventory Risk     │  │
│  │ + region       │  │ XGBoost        │  │ - Warranty Cost      │  │
│  │ + confidence   │  │ Futures Curve  │  │ - Backtesting &      │  │
│  │  intervals     │  │ Scenario       │  │   validation         │  │
│  │                │  │ Ensemble       │  │                      │  │
│  │ + feature      │  │                │  │                      │  │
│  │  importance    │  │ + CI per       │  │                      │  │
│  │                │  │  commodity     │  │                      │  │
│  │ Walk-forward   │  │                │  │                      │  │
│  │ backtesting    │  │ Commodity Index│  │                      │  │
│  │ metrics        │  │ (BOM-weighted) │  │                      │  │
│  └────────┬───────┘  └───────┬────────┘  └──────────┬───────────┘  │
│           │                  │                     │              │
└───────────┼──────────────────┼─────────────────────┼──────────────┘
            │                  │                     │
            └──────────────────┼─────────────────────┘
                               │
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 1: DATA INGESTION & FEATURE ENGINEERING                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────────┐    │
│  │ Data         │  │ Feature        │  │ Data Validation &    │    │
│  │ Connectors   │  │ Engineering    │  │ Quality              │    │
│  │              │  │                │  │                      │    │
│  │ SAP HANA     │  │ Lag features   │  │ Completeness checks  │    │
│  │ - Production │  │ Rolling stats  │  │ Outlier detection    │    │
│  │ - Sales      │  │ Momentum       │  │ Freshness monitoring │    │
│  │ - COGS       │  │ RSI, MACD      │  │ Data lineage         │    │
│  │ - GL         │  │ Macro context  │  │                      │    │
│  │ - Inventory  │  │ Calendar       │  │                      │    │
│  │              │  │ Event flags    │  │                      │    │
│  │ Salesforce   │  │                │  │                      │    │
│  │ - Sales      │  │                │  │                      │    │
│  │ - Revenue    │  │                │  │                      │    │
│  │ - Customers  │  │                │  │                      │    │
│  │              │  │                │  │                      │    │
│  │ Market Data  │  │                │  │                      │    │
│  │ - Yahoo Fin. │  │                │  │                      │    │
│  │ - FRED API   │  │                │  │                      │    │
│  │ - Futures    │  │                │  │                      │    │
│  │              │  │                │  │                      │    │
│  │ Treasury     │  │                │  │                      │    │
│  │ - FX Rates   │  │                │  │                      │    │
│  │ - Hedges     │  │                │  │                      │    │
│  │ - Debt       │  │                │  │                      │    │
│  │              │  │                │  │                      │    │
│  │ Supply Chain │  │                │  │                      │    │
│  │ - Lead times │  │                │  │                      │    │
│  │ - Suppliers  │  │                │  │                      │    │
│  │              │  │                │  │                      │    │
│  │ Synthetic    │  │                │  │                      │    │
│  │ - Fallback   │  │                │  │                      │    │
│  └──────┬───────┘  └────────┬───────┘  └──────────┬───────────┘    │
│         │                   │                    │                 │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ Polars Data Pipeline (Parquet-based, lazy eval)      │          │
│  │ - Auto CSV → Parquet conversion                       │          │
│  │ - Priority resolution (parquet > external > raw)      │          │
│  │ - Compression (zstd)                                 │          │
│  │ - Data quality metrics                                │          │
│  └──────────────────────────────────────────────────────┘          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## **DATA FLOW PIPELINES**

### **Daily Pipeline (Morning Close)**

```
06:00 AM    Start Scheduler Job
            ├─ Trigger: Daily cron (Mondays-Fridays 6 AM UK time)
            └─ Context: After NY markets close (next day commodity data available)

06:05 AM    Fetch Real-Time Market Data
            ├─ Yahoo Finance: Commodity prices (12 commodities)
            ├─ Yahoo Finance: FX rates (GBP/USD, EUR/USD, USD/JPY, USD/CNY)
            ├─ Yahoo Finance: Market indices (S&P, VIX, Oil, Gold)
            ├─ FRED API: Macro indicators (if available, with fallback)
            ├─ CCXT/Binance: Crypto prices (BTC, ETH)
            ├─ LME/NYMEX API: Futures curves (if connected)
            └─ Timeout: 30 sec (fail gracefully to synthetic)

06:15 AM    Load JLR Operational Data (if real-time SAP available)
            ├─ SAP HANA: Today's production units
            ├─ SAP HANA: Yesterday's sales volumes
            ├─ SAP HANA: Current inventory by commodity
            ├─ Salesforce: Yesterday's sales revenue
            └─ Timeout: 2 min (use prior day actual if delayed)

06:20 AM    Data Validation
            ├─ Completeness: All 12 commodities received? If not, flag warning
            ├─ Freshness: Data newer than 4 hours old? If not, use cached
            ├─ Outliers: Any commodity > 5% daily move? Flag for review
            ├─ Cross-checks: FX pairs consistent? (BID/ASK tight?)
            └─ Log all validation results to data_quality.log

06:25 AM    Feature Engineering (for forecasting)
            ├─ Compute lagged returns (1, 3, 6, 12 months)
            ├─ Compute rolling volatility
            ├─ Compute momentum indicators (RSI, MACD)
            ├─ Merge with macro data (aligned on same date)
            └─ Create feature matrix for next 12 months

06:30 AM    Run Forecasting Models
            ├─ Demand Forecast (all 4 segments): 12-month outlook
            │  ├─ Load demand model (XGBoost, saved from last training)
            │  ├─ Use latest macro forecast (from Bloomberg/consensus)
            │  ├─ Generate point forecast + 80%/95% CIs
            │  └─ Log forecast to output_demand.parquet
            │
            ├─ Commodity Forecast (all 12 commodities): 12-month outlook
            │  ├─ Load 4 models (SARIMAX, XGBoost, Futures, Scenario)
            │  ├─ Execute ensemble (weighted average)
            │  ├─ Generate commodity index (BOM-weighted)
            │  └─ Log forecast to output_commodities.parquet
            │
            └─ Price Elasticity: Apply elasticity to demand → ASP impact

06:40 AM    Build P&L Model (Deterministic)
            ├─ Input: Demand forecast + Commodity forecast + ASP
            ├─ Compute: Month-by-month P&L for next 12 months
            │  ├─ Revenue = Units × ASP × (1 - Incentive)
            │  ├─ COGS = Revenue × Base_COGS% + Commodity_Impact + Util_Impact
            │  ├─ Gross Margin = Revenue - COGS
            │  ├─ Warranty = Revenue × Warranty%
            │  ├─ Operating Income = Gross Margin - Warranty - Depreciation
            │  └─ Net Income = OI - Tax
            ├─ Aggregate to annual by segment
            └─ Log P&L to output_pnl.parquet

06:50 AM    Run Monte Carlo Simulation
            ├─ Execute 5000 simulation runs
            │  ├─ Draw commodity shock (t-dist, df=5)
            │  ├─ Draw demand shock (normal)
            │  ├─ Draw FX shock (normal)
            │  ├─ Compute P&L for each run
            │  └─ Store to simulation_results.parquet
            ├─ Compute statistics
            │  ├─ Mean, median, std
            │  ├─ p5/p10/p25/p75/p90/p95
            │  ├─ VaR(95%), CVaR(95%)
            │  └─ Margin-at-Risk
            └─ Log to output_simulation.parquet

07:00 AM    Run 7 Preset Scenarios
            ├─ For each scenario (Base, Bull, Lithium Shock, etc.):
            │  ├─ Set scenario parameters
            │  ├─ Run deterministic P&L
            │  ├─ Run 1000-sim Monte Carlo (faster than full 5K)
            │  └─ Capture results
            └─ Log to output_scenarios.parquet

07:10 AM    Governance & Audit Logging
            ├─ Log forecast event to audit_log.jsonl
            │  ├─ timestamp, model_version, data_version
            │  ├─ commodity_forecast_accuracy (if prior actual available)
            │  ├─ demand_forecast_accuracy
            │  └─ notes (any anomalies)
            │
            ├─ Bias Tracking: Compare forecast vs. actual from prior month
            │  ├─ If actual available: Compute forecast error
            │  ├─ Store to bias_history.parquet
            │  ├─ If error > threshold: Log alert
            │  └─ Trend check: Is bias improving?
            │
            └─ Model Card: Publish governance doc
               ├─ Model versions used
               ├─ Training data period
               ├─ Key metrics (MAPE, RMSE)
               └─ Limitations & caveats

07:20 AM    Generate Alerts (if thresholds breached)
            ├─ Commodity Alert: Any commodity > ±15% vs. forecast?
            ├─ Demand Alert: Any segment > ±10% vs. forecast?
            ├─ P&L Alert: EBIT > ±5% vs. prior month forecast?
            ├─ Risk Alert: VaR(95%) exceeded threshold?
            └─ Quality Alert: Data freshness issue?

07:25 AM    Load Results to Dashboard
            ├─ Refresh Streamlit cache
            ├─ Update FastAPI endpoints
            ├─ Send notification: "GIC Daily Update Complete"
            └─ Publish time: 07:25 AM UK

Dashboard Ready (07:30 AM)
┌──────────────────────────────┐
│ Finance team checks:          │
│ ✓ Executive summary page      │
│ ✓ Commodity forecast chart    │
│ ✓ P&L forecast               │
│ ✓ Scenario comparison         │
│ ✓ Risk dashboard (VaR/CVaR)  │
└──────────────────────────────┘
```

### **Monthly Cycle (Close + Guidance)**

```
Day 1–3:    Actual Results Received
            ├─ SAP GL: Actual P&L (revenue, COGS, margin, OI, tax)
            ├─ Salesforce: Actual sales units + ASP
            ├─ SAP MM: Actual inventory movements
            └─ Treasury: Actual FX impact, hedging results

Day 4:      Variance Analysis
            ├─ Forecast vs. Actual comparison (month-by-month from 12-month forecast)
            ├─ Segment-level: Units, revenue, margin variance
            ├─ Driver analysis:
            │  ├─ Was volume shortfall demand-driven or supply-driven?
            │  ├─ Was margin miss commodity shock or utilization?
            │  ├─ Was COGS impact larger than forecast? Recalibrate material_fraction
            │  └─ Was warranty reserve adequate?
            ├─ Bias tracking: Log actual vs. forecast to variance_log.jsonl
            └─ Output: Variance Analysis Report (PowerPoint-ready)

Day 5:      Model Retraining (if needed)
            ├─ Walk-forward update: New month of actuals added to training set
            ├─ Check if model should be retrained
            │  ├─ If forecast error > 10%: Trigger retraining
            │  ├─ If actual data pattern shifted: Trigger retraining
            │  ├─ If new product launch/discontinuation: Manual refresh
            │  └─ Else: Use existing model
            │
            └─ If retraining:
               ├─ Demand model: Retrain XGBoost (4 segments)
               ├─ Commodity model: Retrain SARIMAX + XGBoost ensemble
               ├─ Backtest: Compare new model vs. old on last 3 months
               ├─ If new model MAPE < old: Adopt new; else keep old
               └─ Log model version change to audit_log.jsonl

Day 6:      Quarterly Forward Guidance (if quarter-end)
            ├─ Run full simulation: Monte Carlo + 7 scenarios
            ├─ Compute earnings ranges
            │  ├─ Bear case (p10): EBIT £X.XX
            │  ├─ Base case (mean): EBIT £X.XX
            │  ├─ Bull case (p90): EBIT £X.XX
            │  └─ Output guidance range (e.g., "£1.2B–£1.6B EBIT")
            ├─ Prepare investor presentation
            │  ├─ Scenario comparison table
            │  ├─ Sensitivity waterfall (volume, commodity, mix, other)
            │  └─ Risk heat map (which commodities = biggest exposure?)
            └─ Finance leadership review (CFO, Controller, FPA)

Day 7:      Publishing & Stakeholder Communication
            ├─ Executive Summary: 1-pager for leadership
            ├─ Detailed Report: Forecast, scenarios, risk metrics
            ├─ Segment Scorecards: Revenue, margin, drivers by vehicle segment
            ├─ Risk Dashboard: VaR, CVaR, concentration risk by commodity
            ├─ Circulate to: CFO, Controller, Segment Heads, Treasury
            └─ Store all reports to shared drive (versioned, dated)
```

---

## **SYSTEM ARCHITECTURE (TECHNICAL)**

### **Technology Stack**

```
Language:           Python 3.10+
Data Layer:         Polars (lazy evaluation), Parquet (compression), SQL (optional)
ML/Forecasting:     XGBoost, StatsModels (SARIMAX), Scikit-Learn
API Framework:      FastAPI + Uvicorn
Visualization:      Streamlit (dashboards), Plotly (interactive charts)
Logging:            Loguru (structured) + JSONL (audit trail)
Job Scheduling:     APScheduler (cron-like scheduling)
Configuration:      YAML (settings.yaml)
Version Control:    Git (GitHub/GitLab)
Testing:            Pytest (unit), Walk-forward backtesting
Package Management: Poetry or Pip + requirements.txt
```

### **Deployment Architecture**

#### **Development Environment**
```
┌─────────────────────────────────────────┐
│ Developer Laptop (Windows/Mac/Linux)     │
│                                          │
│ Python venv                              │
│ ├─ GIC source code (src/)                │
│ ├─ Tests (tests/)                        │
│ ├─ Scripts (scripts/)                    │
│ └─ Data (data/ — local copy)             │
│                                          │
│ IDE: VS Code + Python extension          │
│ Tools: pytest, black, mypy, ruff         │
└─────────────────────────────────────────┘
```

#### **Staging Environment**
```
┌──────────────────────────────────────────────────┐
│ AWS EC2 (or on-prem server) | t3.large instance  │
│                                                   │
│ Docker Container: GIC Platform                    │
│ ├─ Python 3.10 + dependencies                    │
│ ├─ /app/src (GIC code)                           │
│ ├─ /app/data (mounted volume: parquet files)     │
│ ├─ /app/logs (audit trail + debug logs)          │
│ └─ /app/models (saved ML models)                 │
│                                                   │
│ Port 8000: FastAPI (health check, API testing)   │
│ Port 8501: Streamlit (dashboard testing)         │
│ Port 5432: PostgreSQL (optional, for audit log)  │
│                                                   │
│ Cron Job: Daily 06:00 AM forecast run            │
│                                                   │
│ Monitoring:                                      │
│ ├─ CloudWatch (AWS) or Prometheus (on-prem)     │
│ ├─ Alerts: CPU >80%, disk >90%, API errors      │
│ └─ Logs: Aggregated to ELK/Splunk               │
└──────────────────────────────────────────────────┘
```

#### **Production Environment**
```
┌────────────────────────────────────────────────────────┐
│ JLR's On-Prem Data Center (Data Sovereignty)           │
│                                                        │
│ Load Balancer (nginx/HAProxy)                         │
│ ├─ Port 80/443: HTTPS traffic                         │
│ └─ Route to: FastAPI service                         │
│                                                        │
│ ┌─────────────────────────────────────────────────┐  │
│ │ FastAPI Microservice (Docker)                    │  │
│ │                                                   │  │
│ │ Replicas: 2 (for high availability)              │  │
│ │ Memory: 4GB per instance                         │  │
│ │ CPU: 2 cores per instance                        │  │
│ │                                                   │  │
│ │ Endpoints:                                       │  │
│ │ ├─ GET /health                                   │  │
│ │ ├─ GET /models                                   │  │
│ │ ├─ POST /forecast/commodity                      │  │
│ │ ├─ POST /forecast/demand                         │  │
│ │ ├─ POST /simulation/scenario                     │  │
│ │ ├─ GET /pnl/monthly                              │  │
│ │ └─ POST /override (with governance checks)       │  │
│ └─────────────────────────────────────────────────┘  │
│                                                        │
│ ┌─────────────────────────────────────────────────┐  │
│ │ Streamlit Dashboards (Docker)                    │  │
│ │                                                   │  │
│ │ Port 8501 (reverse-proxied behind nginx)         │  │
│ │ Memory: 2GB per instance                         │  │
│ │ CPU: 1 core per instance                         │  │
│ │                                                   │  │
│ │ 7 Pages (all interactive, real-time):            │  │
│ │ ├─ Executive Summary                             │  │
│ │ ├─ Commodity Intelligence                        │  │
│ │ ├─ Financial P&L                                 │  │
│ │ ├─ Scenario Simulation                           │  │
│ │ ├─ Market Monitor                                │  │
│ │ ├─ Backtesting Results                           │  │
│ │ └─ Data Explorer                                 │  │
│ └─────────────────────────────────────────────────┘  │
│                                                        │
│ ┌─────────────────────────────────────────────────┐  │
│ │ Data Storage Layer                               │  │
│ │                                                   │  │
│ │ /data/parquet/  (Compressed Parquet files)       │  │
│ │ ├─ commodity_prices.parquet (7yr history)       │  │
│ │ ├─ sales_data.parquet (7yr history)             │  │
│ │ ├─ macro_indicators.parquet                      │  │
│ │ ├─ production_inventory.parquet                  │  │
│ │ ├─ bom_data.parquet                              │  │
│ │ ├─ output_pnl.parquet (daily forecast)           │  │
│ │ ├─ output_scenarios.parquet (daily)              │  │
│ │ └─ output_simulation.parquet (daily MC)          │  │
│ │                                                   │  │
│ │ /logs/audit/  (Immutable JSONL logs)            │  │
│ │ ├─ audit_log.jsonl (forecast events)            │  │
│ │ ├─ variance_log.jsonl (forecast errors)         │  │
│ │ ├─ override_log.jsonl (manual changes)          │  │
│ │ └─ bias_history.parquet                          │  │
│ │                                                   │  │
│ │ /models/saved/  (Serialized ML models)          │  │
│ │ ├─ demand_xgb_premium_*.joblib                  │  │
│ │ ├─ commodity_sarimax_copper_*.pkl               │  │
│ │ ├─ commodity_xgb_lithium_*.joblib               │  │
│ │ └─ latest.json (pointer to current models)      │  │
│ │                                                   │  │
│ │ Optional: PostgreSQL (for structured audit log) │  │
│ │ ├─ schema: audit_trail, bias_tracking, etc.     │  │
│ │ └─ Backup: Daily snapshots to S3                │  │
│ └─────────────────────────────────────────────────┘  │
│                                                        │
│ ┌─────────────────────────────────────────────────┐  │
│ │ Job Scheduler (APScheduler or cron)              │  │
│ │                                                   │  │
│ │ Daily Job (06:00 AM): Run full pipeline          │  │
│ │ Weekly Job (Friday 4 PM): Model retraining       │  │
│ │ Monthly Job (Day 1 at 9 AM): Guidance refresh    │  │
│ └─────────────────────────────────────────────────┘  │
│                                                        │
│ ┌─────────────────────────────────────────────────┐  │
│ │ Monitoring & Alerting                            │  │
│ │                                                   │  │
│ │ Prometheus (metrics collection)                  │  │
│ │ ├─ API response time                             │  │
│ │ ├─ Data freshness (age of last update)           │  │
│ │ ├─ Model prediction variance                     │  │
│ │ └─ Forecast accuracy (vs. actual)                │  │
│ │                                                   │  │
│ │ PagerDuty/Slack (alerting)                       │  │
│ │ ├─ Daily forecast run status                     │  │
│ │ ├─ Data connector failures                       │  │
│ │ ├─ Forecast quality alerts (bias > threshold)   │  │
│ │ └─ System health (CPU, memory, disk)             │  │
│ └─────────────────────────────────────────────────┘  │
│                                                        │
└────────────────────────────────────────────────────────┘
```
