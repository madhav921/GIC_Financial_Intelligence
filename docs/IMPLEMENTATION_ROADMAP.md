# **IMPLEMENTATION ROADMAP & EXECUTION PLAN**
## **GIC Platform | 12-Month Phased Rollout**

**Document Version:** 1.0  
**Date:** April 20, 2026  
**Status:** Executive Action Plan  

---

## **EXECUTIVE SUMMARY**

This document provides a **detailed, month-by-month action plan** to operationalize GIC Platform from current state (80% built, synthetic data) to full production (real data, integrated into monthly financial close, CFO sign-off required for all guidance).

**Timeline:** 12 months (April 2026 – March 2027)  
**Investment:** £2.5M Year 1 (platform completion + team + data integration)  
**Success Criteria:** 
- ✅ May 2026: Q2 guidance published using GIC Platform
- ✅ July 2026: Monthly close process integrated  
- ✅ Q4 2026: Forecast accuracy validated (MAPE <8% for commodities, <10% for volumes)
- ✅ Mar 2027: Platform embedded in annual budget planning cycle

---

## **PHASE 1: FOUNDATION (APRIL–JUNE 2026) — "OPERATIONALIZE CORE"**

### **Objectives**
1. Stabilize GIC platform on synthetic data (pre-production validation)
2. Complete data audit (identify JLR's real data landscape)
3. Establish governance framework (CFO approval protocols, override policies)
4. Publish Q2 2026 guidance using GIC forecast (internal only, not external)

### **Workstreams**

#### **Workstream 1.1: Platform Stabilization & Validation**
| Task | Owner | Duration | Dependencies | Deliverable |
|------|-------|----------|--------------|-------------|
| Run daily forecast pipeline on synthetic data (1 month test) | Data Eng | 4 weeks | Code complete | Daily forecast runs, log analysis |
| Validate output quality (backtests pass, metrics reasonable) | Analytics | 2 weeks | Daily runs × 20 | Validation report (MAPE targets met?) |
| Dashboard usability testing (finance team) | Product Mgr | 2 weeks | Dashboard pages complete | User feedback log, UX improvements |
| API load testing (simulate 100 concurrent users) | DevOps | 1 week | FastAPI routes complete | Performance profile, scaling limits |
| Stress test Monte Carlo (5000 sims within SLA) | ML Engineer | 1 week | Simulation code | Execution time <5 min, validated |
| Deploy to staging environment (AWS/on-prem) | DevOps | 2 weeks | Code, Docker, infra | Staging environment live, monitoring configured |
| **Month 1 Checkpoint:** Staging environment stable, daily runs successful | | | | **Go/No-Go Decision** |

#### **Workstream 1.2: Data Audit (Identify Real Data Landscape)**
| Task | Owner | Duration | Dependencies | Deliverable |
|------|-------|----------|--------------|-------------|
| Interview SAP owner: Production data availability | SAP Lead | 1 week | Calendar | Data structure doc, table list |
| Interview Finance: COGS, GL structure | Controller | 1 week | Calendar | GL account mapping, hierarchy |
| Interview Sales: Salesforce history, CRM setup | Sales Ops | 1 week | Calendar | Salesforce object diagram |
| Interview Treasury: FX, hedging, debt data | Treasurer | 1 week | Calendar | Treasury systems inventory |
| Interview Supply Chain: Supplier, lead time data | Sourcing Dir | 1 week | Calendar | Supply chain systems inventory |
| Interview HR: Labor costs, headcount | CHRO | 1 week | Calendar | HR/Payroll system structure |
| Compile Data Inventory spreadsheet | Project Mgr | 2 weeks | All interviews | Data_Landscape_Master.xlsx (what exists, quality, access) |
| **Month 2 Checkpoint:** Complete data inventory, identify gaps | | | | **Data Gap Analysis Document** |

#### **Workstream 1.3: Governance Framework**
| Task | Owner | Duration | Dependencies | Deliverable |
|------|-------|----------|--------------|-------------|
| Draft forecast methodology doc (assumptions, limitations) | Chief Analyst | 2 weeks | | Governance_Methodology.pdf |
| Draft override policy (when can models be overridden? who approves?) | CFO/Chief Analyst | 2 weeks | Methodology doc | Override_Policy.pdf (includes approval matrix) |
| Draft data governance (lineage, audit trail, data quality SLAs) | CIO | 2 weeks | | Data_Governance_Framework.pdf |
| Draft model governance (retraining triggers, accuracy thresholds, escalation) | Chief Analyst | 2 weeks | Backtesting results | Model_Governance_Framework.pdf |
| Present to CFO + Controller for sign-off | CFO Office | 1 week | All docs | Signed approvals, policy amendments |
| **Month 3 Checkpoint:** All governance docs signed, team trained | | | | **Governance Package** |

#### **Workstream 1.4: Q2 2026 Guidance Preparation (Internal)**
| Task | Owner | Duration | Dependencies | Deliverable |
|------|-------|----------|--------------|-------------|
| Run full simulation: 7 scenarios, Monte Carlo | Analytics | 1 week | Platform stable, data ready | Scenario outputs (bear/base/bull) |
| Build scenario comparison table | FP&A | 1 week | Simulation outputs | Excel: Scenario Revenue/EBIT/Margin |
| Create sensitivity waterfall (volume, commodity, mix, other impact) | Analytics | 1 week | Simulation outputs | Waterfall chart (visual, PowerPoint-ready) |
| Prepare P&L bridge: Base → GIC Forecast | FP&A | 1 week | Simulation outputs | P&L bridge doc (assumption transparency) |
| Internal review meeting (CFO, Controller, Segment Heads) | CFO Office | 2 days | All prep docs | Feedback, assumption refinements, |
| Revise based on feedback | FP&A | 3 days | Review feedback | Final guidance package |
| **Month 3 Checkpoint:** Internal guidance Q2 consensus (not external yet) | | | | **Internal Guidance Package** |

### **Phase 1 Dependencies & Risk Mitigations**

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Data audit reveals major data gaps | Medium | High | Start connector development in parallel (Workstream 2.1); can still proceed with Phase 1 on synthetic |
| CFO skeptical of probabilistic guidance (wants point forecast) | Medium | High | Comparative analysis: GIC forecast accuracy vs. last year's Excel-based guesses (show MAPE improvement) |
| Platform stability issues (crashes during daily run) | Low | High | Intensive testing in Workstream 1.1; add monitoring + alerts |
| Governance approval delays | Low | Medium | Engage CFO early (April); don't wait until June |

### **Phase 1 Success Criteria**
- [ ] Staging environment running daily forecasts without errors (10+ consecutive successful runs)
- [ ] Platform performance meets SLA (5000-sim MC in <5 min; API response <2 sec)
- [ ] Dashboard pages load & render correctly (all 7 pages, all charts)
- [ ] Data audit complete (master inventory of real data landscape)
- [ ] Governance framework approved by CFO & legal
- [ ] Finance team trained on GIC methodology
- [ ] Internal Q2 guidance finalized (not yet published to market)

### **Phase 1 Budget**
```
Salaries (4 FTE × 3 months):
- Lead Data Engineer: £100K/yr ÷ 4 × 3 = £75K
- Analytics Manager: £90K/yr ÷ 4 × 3 = £67.5K
- FP&A Manager: £85K/yr ÷ 4 × 3 = £63.75K
- Project Manager: £80K/yr ÷ 4 × 3 = £60K
──────────────────────────────────────
Subtotal: £266.25K

External Services:
- SAP Specialist (contractor, 4 weeks): £15K/week = £60K
- Consulting (governance, risk framework): £30K
──────────────────────────────────────
Subtotal: £90K

Infrastructure & Tools:
- AWS (staging servers, compute, storage): £5K
- Monitoring (Prometheus, logs): £2K
─────────────────────────────────────
Subtotal: £7K

──────────────────────────────────────
PHASE 1 TOTAL: £363K
```

---

## **PHASE 2: DATA INTEGRATION (JULY–SEPTEMBER 2026) — "CONNECT TO REALITY"**

### **Objectives**
1. Build SAP, Salesforce, Treasury connectors
2. Load real JLR operational data (production, sales, COGS, inventory)
3. Validate synthetic approximations against actuals (calibrate to within 5%)
4. Publish Q3 2026 guidance using real data baseline

### **Workstreams**

#### **Workstream 2.1: Priority 1 Connectors (Production, Sales, COGS)**
| Task | Owner | Duration | Status | Deliverable |
|------|-------|----------|--------|-------------|
| **SAP Production Connector** | Data Eng + SAP Specialist | | | |
| - Analyze SAP PP module structure (AUFK, RESB, IOOB tables) | SAP Spec | 1 week | | Technical spec |
| - Develop RFC function Z_GIC_PRODUCTION_DATA in SAP | SAP Spec | 2 weeks | | RFC function, tested on sandbox |
| - Build Python connector (src/data/connectors/sap_production.py) | Data Eng | 2 weeks | | Code, unit tests |
| - Load FY2025 + YTD 2026 production data to Parquet | Data Eng | 1 week | | production.parquet (24 months history) |
| - Validate: Production totals reconcile to reported units | Analytics | 1 week | | Validation report |
| **SAP COGS Connector** | Finance + Data Eng | | | |
| - Analyze GL structure (accounts 5000–5999) | Controller | 1 week | | GL mapping doc |
| - Develop RFC Z_GIC_MANUFACTURING_COSTS | SAP Spec | 2 weeks | | RFC function, sandbox-tested |
| - Build Python connector | Data Eng | 2 weeks | | Code, unit tests |
| - Load FY2025 + YTD COGS data | Data Eng | 1 week | | cogs.parquet |
| - Validate: GL COGS reconciles to reported (within 1%) | Controller | 1 week | | Validation report, GL mapping |
| **Salesforce Sales Connector** | Sales Ops + Data Eng | | | |
| - Analyze Salesforce Orders object structure | Sales Ops | 1 week | | Object diagram, field list |
| - Develop API queries (SOQL) for sales orders | Sales Ops | 1 week | | SOQL templates |
| - Build Python connector | Data Eng | 2 weeks | | Code, pagination logic |
| - Load Salesforce Order history (FY2024–present) | Data Eng | 1 week | | sales_orders.parquet |
| - Validate: Sales revenue reconciles to GL (within 1%) | FP&A | 1 week | | Validation report |
| **Month 1 Checkpoint:** Priority 1 connectors live, real data flowing | | | | **Data Integration Report** |

#### **Workstream 2.2: Priority 2 Connectors (Inventory, Warranty, Capital)**
| Task | Owner | Duration | Status | Deliverable |
|------|-------|----------|--------|-------------|
| **SAP Inventory Connector** | | 4 weeks | | inventory.parquet |
| **Warranty Management Connector** | | 3 weeks | | warranty.parquet |
| **SAP Asset Accounting (Depreciation)** | | 2 weeks | | capex_depreciation.parquet |
| **Month 2 Checkpoint:** Priority 2 connectors live | | | | **Complete Data Feed** |

#### **Workstream 2.3: Synthetic Calibration**
| Task | Owner | Duration | Dependencies | Deliverable |
|------|-------|----------|--------------|-------------|
| Reconstruct FY2025 P&L from real GL data | Controller | 2 weeks | SAP connectors live | Actual_PnL_FY2025.xlsx (monthly detail) |
| Compare synthetic P&L vs. actual (by segment, by month) | Analytics | 2 weeks | Real data loaded | Variance_Analysis_Synthetic_vs_Actual.xlsx |
| Identify biggest gaps (where synthetic <> actual) | Analytics | 1 week | Variance analysis | Gap summary (5–10 bullets) |
| Adjust synthetic parameters to match actual | Analytics | 2 weeks | Gap analysis | Updated synthetic_generator.py |
| Re-run synthetic data generation; validate <5% error | Analytics | 1 week | Parameter updates | Quality_Report_Synthetic_FY2025.pdf |
| **Month 2–3 Checkpoint:** Synthetic approximations validated | | | | **Calibration Report** |

#### **Workstream 2.4: Real Data Model Retraining**
| Task | Owner | Duration | Dependencies | Deliverable |
|------|-------|----------|--------------|-------------|
| Retrain demand forecasts on real sales history (FY2024–Apr2026) | ML Engineer | 3 weeks | Salesforce connector live | New demand models (v2026.07) |
| Retrain commodity forecasts on real price data | ML Engineer | 2 weeks | Market data (already live) | New commodity models (v2026.07) |
| Backtest new models vs. old (past 6 months) | Analytics | 2 weeks | Retrained models | Backtest_Comparison_Report.pdf (accuracy gains?) |
| If new models better: Adopt; else keep old | Analytics | 1 week | Backtest results | Model_Adoption_Decision |
| **Month 2–3 Checkpoint:** Models retrained on real data | | | | **Updated Model Registry** |

#### **Workstream 2.5: Q3 Guidance (Real Data Baseline)**
| Task | Owner | Duration | Dependencies | Deliverable |
|------|-------|----------|--------------|-------------|
| Run full pipeline on real data (production, sales, COGS, inventory) | Analytics | 1 week | All connectors live | Real-data-driven forecast |
| Compare GIC forecast vs. prior month actual (bias check) | Analytics | 1 week | Forecast outputs | Bias_Report_Jul2026.pdf |
| Build Q3 deterministic P&L (month-by-month, segment) | FP&A | 1 week | Forecast outputs | Q3_PnL_Forecast.xlsx |
| Run Monte Carlo + 7 scenarios | Analytics | 1 week | Deterministic P&L | Simulation_Results_Q3.parquet |
| Prepare guidance package (waterfall, sensitivity, risk) | FP&A | 1 week | Scenario outputs | GIC_Q3_Guidance_Package.pptx |
| Internal review + CFO sign-off | CFO Office | 1 week | Guidance package | Approved Q3 guidance |
| **Month 3 Checkpoint:** Q3 guidance finalized (internal, not external yet) | | | | **Q3 Guidance Package** |

### **Phase 2 Success Criteria**
- [ ] All Priority 1 + 2 connectors live and feeding real data
- [ ] Synthetic approximations validated to within 5% of actual (by segment, by month)
- [ ] Demand + commodity models retrained on real history; accuracy improved vs. synthetic
- [ ] FY2025 actual P&L fully reconstructed from GL data; reconciles to reported results within <2%
- [ ] Q3 guidance finalized using real data baseline
- [ ] All governance logs populated (audit trail capturing all model changes)

### **Phase 2 Budget**
```
Salaries (team continues from Phase 1):
- Lead Data Engineer: £75K
- Analytics Manager: £67.5K
- FP&A Manager: £63.75K
- Project Manager: £60K
- Add: 1× Data Integration Specialist: £70K (3 months)
──────────────────────────────────────
Subtotal: £336K

External Services:
- SAP Specialist (contractor, 12 weeks): £15K/week = £180K
- Salesforce Admin (contractor, 4 weeks): £12K/week = £48K
- Consulting (data governance, quality): £25K
──────────────────────────────────────
Subtotal: £253K

Infrastructure:
- AWS/on-prem storage (real data: parquet files): £8K
- Database (PostgreSQL for audit): £3K
──────────────────────────────────────
Subtotal: £11K

──────────────────────────────────────
PHASE 2 TOTAL: £600K
```

---

## **PHASE 3: OPERATIONALIZATION (OCTOBER–DECEMBER 2026) — "EMBED IN PROCESSES"**

### **Objectives**
1. Integrate into monthly close cycle (automated P&L variance analysis)
2. Monthly model retraining + governance reviews
3. Publish Q4 earnings guidance externally (first market-facing use)
4. Risk dashboard live (board-ready)

### **Workstreams**

#### **Workstream 3.1: Monthly Close Integration**
| Task | Owner | Duration | Deliverable |
|------|-------|----------|-------------|
| **Create Monthly Close Checklist** | Finance Ops | 2 weeks | Close_Checklist_GIC_Integration.pdf |
| - Day 1–3: GL close, actual P&L posted | Controller | (existing) | |
| - Day 4: Run GIC variance analysis (forecast vs. actual) | Analytics | New | Variance_Report_[Month].xlsx |
| - Day 5: Review bias tracking, model health | Analytics | New | Model_Health_Report_[Month].pdf |
| - Day 6–7: Segment controller reviews → consolidates | Controllers | Existing (enhanced with GIC outputs) | |
| **Deploy "Close Bot" (Automated P&L Variance Calculation)** | Data Eng | 3 weeks | Python script, scheduled daily |
| - Trigger: Day 4 morning, GL close complete | | | |
| - Input: Actual P&L from GL; Forecast from GIC database | | | |
| - Output: Variance report (units, ASP, COGS drivers) | | | |
| - Distribution: Email to segment controllers | | | |
| **Implement Override Review Board** | Finance Ops | 2 weeks | Override_Review_Board_Policy.pdf |
| - Policy: Any forecast override > 5% requires CFO review | | | |
| - Quarterly meeting: Review all overrides from past quarter | | | |
| - Log: All overrides captured in audit_log.jsonl | | | |
| **Month 1 Checkpoint:** Monthly close integrated; Close Bot running daily | | | **Close_Integration_Verification** |

#### **Workstream 3.2: Production Deployment**
| Task | Owner | Duration | Deliverable |
|------|-------|----------|-------------|
| **Infrastructure Hardening** | DevOps | 3 weeks | |
| - Load balancing (failover to backup server) | | 2 weeks | HA_Setup_Verified |
| - Backup strategy (daily snapshots to S3) | | 1 week | Backup_SLA_Documented |
| - Monitoring + alerting (Prometheus, Slack) | | 2 weeks | Monitoring_Dashboard_Live |
| - SSL/TLS certificates (HTTPS) | | 1 week | Security_Audit_Passed |
| **Cutover to Production** | DevOps | 2 weeks | |
| - Dry-run: Full production stack on Saturday (low impact) | | 1 week | Dry_Run_Report |
| - Go-live: Monday 06:00 AM daily forecast on production | | 1 week | Go_Live_Checklist_Completed |
| - Monitor 24/7 for first week (escalation: on-call engineer) | | 1 week | Incident_Log (track any issues) |
| **Month 1–2 Checkpoint:** Production live, stable for 20+ consecutive forecast runs | | | **Prod_Stability_Report** |

#### **Workstream 3.3: Q4 External Earnings Guidance**
| Task | Owner | Duration | Deliverable |
|------|-------|----------|-------------|
| **Prepare Guidance Package** | FP&A + IR | 3 weeks | |
| - Run full simulation (bear/base/bull, Monte Carlo) | Analytics | 1 week | Scenario_Outputs_Q4.parquet |
| - Build investor deck (1 page on GIC methodology, 3 pages on scenarios) | IR | 2 weeks | Investor_Presentation_Q4_2026.pptx |
| - Prepare risk disclosures (commodity exposure, supply chain risk) | IR | 1 week | Risk_Disclosures_Annex.pdf |
| - Run by external auditors (confirm audit trail, methodology sound) | Auditors | 2 weeks | Auditor_Sign_Off_Letter.pdf |
| **Publish Guidance** | IR | | |
| - Press release: "JLR Q4 2026 Earnings Guidance: £X–£Y EBIT (80% CI)" | PR | 1 day | Press_Release |
| - Investor call: Present GIC methodology + scenarios to sell-side | IR | 1 day | Earnings_Call_Script |
| - File 8-K/RNS: Official regulatory filing | Legal | 1 day | Filed_Disclosure |
| **Month 2–3 Checkpoint:** Q4 external guidance published | | | **Market_Announcement_Completed** |

#### **Workstream 3.4: Risk Dashboard (Board-Ready)**
| Task | Owner | Duration | Deliverable |
|------|-------|----------|-------------|
| **Build Risk KPI Dashboard** | Analytics + BI | 4 weeks | |
| - Commodity Exposure Heat Map: % P&L impact by commodity | | 2 weeks | Dashboard_Page_1: Commodity_Risk |
| - Concentration Risk: Top 5 suppliers' concentration % | | 1 week | Dashboard_Page_2: Supply_Chain_Risk |
| - Margin-at-Risk: VaR(95%) under normal conditions | | 1 week | Dashboard_Page_3: Financial_Risk |
| - Forecast Accuracy Trending: MAPE by model (improving?) | | 1 week | Dashboard_Page_4: Model_Health |
| **Publish to Board** | CFO Office | | |
| - Monthly distribution to Board Risk Committee | | | Dashboard_Viewer_App |
| - Annual risk review: Present to full Board with GIC outputs | | | Board_Risk_Presentation |
| **Month 2–3 Checkpoint:** Risk dashboard live, quarterly board review scheduled | | | **Risk_Dashboard_Verified** |

### **Phase 3 Success Criteria**
- [ ] Monthly close cycle includes automated GIC variance analysis (no manual steps)
- [ ] Production environment stable (zero unplanned downtime in Month 1)
- [ ] Override review board established + at least 1 quarterly meeting held
- [ ] Q4 earnings guidance published externally (using GIC forecast + Monte Carlo ranges)
- [ ] External auditors confirmed audit trail integrity + methodology rigor
- [ ] Risk dashboard deployed to Board (monthly distribution confirmed)
- [ ] All governance + escalation procedures documented + tested

### **Phase 3 Budget**
```
Salaries (team continues):
- Lead Data Engineer: £75K
- Analytics Manager: £67.5K
- FP&A Manager: £63.75K
- Project Manager: £60K
- Data Integration Specialist: £70K
- Add: 1× DevOps/Infrastructure Engineer: £80K (full 3 months)
──────────────────────────────────────
Subtotal: £416K

External Services:
- Infrastructure consulting (production hardening): £40K
- External audit (risk/governance review): £25K
──────────────────────────────────────
Subtotal: £65K

Infrastructure:
- Production hardware/cloud: £15K
- Monitoring tools: £5K
──────────────────────────────────────
Subtotal: £20K

──────────────────────────────────────
PHASE 3 TOTAL: £501K
```

---

## **PHASE 4: CONTINUOUS IMPROVEMENT (JANUARY–MARCH 2027) — "OPTIMIZE & EXPAND"**

### **Objectives**
1. Model refinement cycle (re-calibrate elasticity, commodity drivers)
2. Expand platform to supply chain / Treasury integration
3. Establish annual budget planning cycle with GIC
4. Document lessons learned + establish SLAs

### **Key Activities**

#### **Model Refinement**
| Task | Duration | Output |
|------|----------|--------|
| Analyze 2026 full-year forecast accuracy | 2 weeks | Model_Performance_Review_2026.pdf |
| Investigate systematic biases (why was EV demand under-forecast?) | 2 weeks | Bias_Root_Cause_Analysis.pdf |
| Re-estimate demand elasticity by segment (now with 12+ months real data) | 3 weeks | Elasticity_Estimates_FY2026.xlsx |
| A/B test ensemble weights (is XGBoost beating SARIMAX?) | 2 weeks | Ensemble_Comparison_Report.pdf |
| Update commodity forecasting methodology (supply constraints, geopolitics) | 2 weeks | Commodity_Model_Updates.pdf |

#### **Expansion: Supply Chain Integration**
- Extend financial model to supply chain (cost-to-serve by supplier)
- Supplier financial health risk (early warning of default)
- Lead time impact on working capital

#### **Expansion: Treasury Integration**
- FX impact on consolidated P&L
- Interest rate scenarios (impact on debt covenants)
- Hedging effectiveness analysis

#### **Annual Budget Cycle**
- GIC platform used to build FY2027 budget (base case + upside/downside scenarios)
- Zero-based budgeting: Challenge each assumption with data
- Multiple scenario budgets approved (not single-point)

### **Phase 4 Budget**
```
Salaries (team continues at sustainable pace):
- Lead Data Engineer: £75K
- Analytics Manager: £67.5K
- FP&A Manager: £63.75K
- Project Manager: £60K
- Data Integration Specialist: £70K
- DevOps Engineer: £80K
──────────────────────────────────────
Subtotal: £416K (full 3 months)

External Services:
- Supply chain consulting (integration design): £20K
─────────────────────────────────────
Subtotal: £20K

──────────────────────────────────────
PHASE 4 TOTAL: £436K

YEAR 1 TOTAL (All Phases): £363K + £600K + £501K + £436K = £1.9M ✅
(Target was £2.0M, under budget!)
```

---

## **MASTER TIMELINE GANTT CHART**

```
2026                Q2              Q3              Q4              2027
         Apr  May  Jun  Jul  Aug  Sep  Oct  Nov  Dec  Jan  Feb  Mar
Phase 1:
- Platform Stabilization       ████████
- Data Audit                   ████████
- Governance Framework             ████
- Q2 Internal Guidance              ██

Phase 2:
- Priority 1 Connectors           ███████████
- Priority 2 Connectors             ████████
- Synthetic Calibration            ██████████
- Model Retraining                    ████████
- Q3 Guidance                          ██

Phase 3:
- Close Integration                      ██████
- Production Deployment                  ████████
- Q4 External Guidance                         ██████
- Risk Dashboard                         ████████

Phase 4:
- Model Refinement                                     ████████
- Supply Chain / Treasury Expansion                    ████████
- Annual Budget Integration                               ██████


KEY MILESTONES:
✓ May 31: Phase 1 Go/No-Go decision
✓ Jul 15: Priority 1 connectors live
✓ Sep 30: Real data baseline established
✓ Oct 01: Monthly close integrated
✓ Dec 15: Q4 external guidance published
✓ Jan 31: Production environment stable
✓ Mar 31: FY2027 budget built with GIC
```

---

## **SUCCESS METRICS & KPIs**

### **Adoption Metrics**
```
Target: By Q1 2027, 90%+ of financial decisions supported by GIC Platform

Q2 2026: 20% (initial internal use)
Q3 2026: 40% (guidance planning)
Q4 2026: 70% (external guidance + board reporting)
Q1 2027: 90% (embedded in monthly close + budget planning)

Measurement: % of financial reports that cite GIC forecast + scenarios
```

### **Accuracy Metrics**
```
Target: MAPE <8% for commodities, <10% for demand volumes

Commodity Forecast (12-month horizon):
- Copper: Current MAPE 1.7%, Target 1.5% (reduce variance via ensemble)
- Lithium: Current MAPE 8%, Target 6% (capture supply constraints better)
- Natural Gas: Current MAPE 43%, Target 25% (still volatile, but improved)
- Average across 12 commodities: Target <8%

Demand Forecast (6-month horizon):
- Premium SUV: Target MAPE <8%
- Luxury SUV: Target MAPE <9%
- Performance: Target MAPE <10%
- EV: Target MAPE <12% (nascent, more volatile)
- Average: Target <10%

Tracking: Monthly bias reports posted to dashboard
```

### **Governance Metrics**
```
Target: 100% governance compliance by Q1 2027

- Zero unlogged overrides (requirement: all captured in audit_log.jsonl)
- 100% of overrides reviewed by approval authority (CFO sign-off required)
- Zero data quality issues (data freshness SLA: <4 hours old)
- 100% model retraining log entries (document when + why models updated)
- External auditor sign-off on audit trail & governance controls

Tracking: Monthly governance report to CFO + internal audit
```

### **Operational Metrics**
```
Target: Production platform SLA 99.9% uptime

- Daily forecast job completion rate: >99% (must complete by 07:30 AM)
- API response time p95: <2 seconds
- Monte Carlo simulation runtime: <5 minutes for 5000 runs
- Dashboard page load time: <2 seconds

Tracking: Prometheus metrics dashboard, daily SLA report to ops team
```

---

## **RISK MANAGEMENT DURING ROLLOUT**

### **Critical Risks & Mitigations**

| Risk | Probability | Impact | Phase | Mitigation |
|------|-------------|--------|-------|-----------|
| **Data Quality Issues** | Medium | High | 2–3 | Upfront audit (Phase 1), validation layer, data quality monitoring |
| **Adoption Resistance** | Medium | High | 3 | Executive mandate, training, "must-use" for guidance process |
| **Model Overfitting** | Low | Medium | 2 | Walk-forward backtesting, bias tracking, monthly retraining |
| **Integration Delays** | Medium | Medium | 2 | Start connectors in parallel, build fallbacks (synthetic data) |
| **Key Person Dependency** | Low | High | 1–4 | Cross-train 2nd engineer by Month 6, document all processes |
| **Scope Creep** | Medium | Medium | 1–4 | Change control board, freeze features after core launch |
| **Vendor (Market Data) Downtime** | Low | Low | Ongoing | Fallback to synthetic; cache last-known prices; alert system |
| **Performance Issues** | Low | Medium | 3 | Load testing, profiling, optimization before prod |

### **Contingency Plans**

**If Phase 1 Data Audit Reveals Critical Gaps:**
- Extend Phase 1 by 4 weeks to fill gaps
- Start Phase 2 connectors in parallel (don't wait)
- Adjust timeline: Phase 1 ends Aug 15 (not Jun 30)

**If Production Database Performance Poor:**
- Implement read replicas for dashboards (writes → primary, reads → replicas)
- Cache aggregated results (pre-compute monthly summaries)
- Consider switching from Parquet to Columnar DB (ClickHouse, DuckDB)

**If External Auditors Reject Governance:**
- Engage internal audit early (Phase 1); get feedback before formal audit
- Build additional controls as needed (signing authorities, approval matrices)
- Delay Q4 external guidance if necessary (publish Q1 instead)

---

## **STAFFING & ORGANIZATION**

### **Team Structure**

```
CFO Office
├─ Chief Financial Officer (sponsor)
├─ Controller (finance data owner)
└─ Chief Analyst (GIC platform owner) ← New role
   ├─ Lead Data Engineer (1 FTE)
   │  └─ Data Integration Specialist (1 FTE, Phases 2–4)
   ├─ ML/Analytics Manager (1 FTE)
   │  └─ Analytics Analyst (0.5 FTE)
   ├─ FP&A Manager (1 FTE, from existing team)
   └─ DevOps/Infrastructure Engineer (1 FTE, Phases 3+)

External Support:
├─ SAP Specialist (contractor, Phases 1–3)
├─ Salesforce Admin (contractor, Phase 2)
├─ Consulting (governance, risk, data architecture)
└─ External Auditors (Phase 3 onwards)
```

### **Roles & Responsibilities**

| Role | Responsibilities | Reports To |
|------|-----------------|-----------|
| Chief Analyst (GIC Platform Owner) | Oversee platform; approve models & forecasts; governance | CFO |
| Lead Data Engineer | Build & maintain data pipelines, connectors, ML infrastructure | Chief Analyst |
| Analytics Manager | Develop forecasting models; accuracy tracking; backtesting | Chief Analyst |
| FP&A Manager | Build financial driver models; P&L forecasting; scenarios | Chief Analyst / Controller |
| DevOps Engineer | Deployment; monitoring; SLA management; disaster recovery | Chief Analyst / CIO |
| Project Manager | Timeline; communication; stakeholder management; risks | CFO |

---

## **COMMUNICATION PLAN**

### **Stakeholders & Cadence**

| Stakeholder | Frequency | Format | Owner |
|-------------|-----------|--------|-------|
| **CFO + Finance Leadership** | Weekly | Status update email | Project Mgr |
| **Finance Team (Analysts, Controllers)** | Bi-weekly | Lunch-and-learn, training | Chief Analyst |
| **Segment Heads (P&L owners)** | Monthly | Forecast walkthrough | FP&A Mgr |
| **Board Risk Committee** | Quarterly | Risk dashboard review | CFO |
| **External Auditors** | Quarterly | Governance audit | Controller |
| **IT/Infrastructure** | Bi-weekly | Technical sync | DevOps Eng |
| **External Consultants** | Weekly | Project status | Chief Analyst |

### **Key Communication Milestones**

- **May 15:** Kickoff presentation to CFO + leadership (vision, timeline, investment)
- **June 30:** Phase 1 go-live announcement (platform stable, team trained)
- **Aug 31:** Real data feeds live (Phase 2 completion)
- **Oct 1:** Q4 external guidance announcement (using GIC)
- **Dec 31:** Year-end retrospective (learnings, metrics, plans for Year 2)

---

## **EXECUTION CHECKLIST**

### **Pre-Launch (April 1–15, 2026)**
- [ ] CFO approves investment (£2M Year 1)
- [ ] Chief Analyst hired / assigned
- [ ] Team allocated (Lead Engineer, Analytics, FP&A, PM)
- [ ] Contractors identified (SAP specialist, Salesforce admin, consultants)
- [ ] Kickoff meeting held (vision, roles, timeline confirmed)
- [ ] Project management tool set up (Jira, Azure DevOps)
- [ ] Governance docs drafted (methodology, override policy, data governance)

### **Phase 1 (April–June, 2026)**
- [ ] Daily forecast pipeline running on synthetic data (10+ successful runs)
- [ ] Data audit complete (master inventory of real data)
- [ ] Staging environment deployed (AWS/on-prem)
- [ ] Dashboard usability tested (finance team feedback incorporated)
- [ ] Governance framework approved by CFO + legal
- [ ] Finance team trained on GIC methodology
- [ ] Internal Q2 guidance finalized
- [ ] Phase 1 retrospective & decision to proceed to Phase 2

### **Phase 2 (July–September, 2026)**
- [ ] SAP production, COGS, GL connectors live
- [ ] Salesforce sales connector live
- [ ] FY2025 actual P&L reconstructed & validated
- [ ] Synthetic data calibrated to within 5% of actual
- [ ] Demand + commodity models retrained on real data
- [ ] Backtest results show improvement vs. synthetic models
- [ ] Q3 internal guidance finalized
- [ ] Phase 2 retrospective & decision to proceed to Phase 3

### **Phase 3 (October–December, 2026)**
- [ ] Monthly close process integrated (Close Bot running)
- [ ] Production environment live & stable (20+ consecutive runs)
- [ ] Override review board established & meeting held
- [ ] Q4 external earnings guidance published (first market-facing use)
- [ ] External auditors sign-off on governance + audit trail
- [ ] Risk dashboard live & distributed to Board
- [ ] Phase 3 retrospective

### **Phase 4 (January–March, 2027)**
- [ ] Model refinement cycle complete (elasticity, commodity drivers re-estimated)
- [ ] Supply chain financial integration designed (not yet implemented)
- [ ] Treasury integration designed (not yet implemented)
- [ ] FY2027 budget built with GIC (multiple scenarios approved)
- [ ] Annual retrospective & Year 2 planning
- [ ] Operational SLAs documented

---

## **FINANCIAL SUMMARY**

### **Year 1 Investment (2026)**

| Phase | Duration | Salaries | Contractors | Infrastructure | Total |
|-------|----------|----------|-------------|-----------------|-------|
| Phase 1 (Apr–Jun) | 3 mo | £266K | £90K | £7K | £363K |
| Phase 2 (Jul–Sep) | 3 mo | £336K | £253K | £11K | £600K |
| Phase 3 (Oct–Dec) | 3 mo | £416K | £65K | £20K | £501K |
| Phase 4 (Jan–Mar) | 3 mo | £416K | £20K | — | £436K |
| **TOTAL YEAR 1** | 12 mo | **£1,434K** | **£428K** | **£38K** | **£1,900K** |

### **Year 2+ Operating Costs (Ongoing)**

```
Core Team (5–6 FTE):
- Lead Data Engineer: £100K
- Analytics Manager: £90K
- FP&A Manager: £85K
- DevOps Engineer: £80K
- Data Integration Specialist: £70K (part-time Year 2)
- Project Manager: £80K (transitions to product owner)
────────────────────────────────────
Total: £505K/year

Infrastructure:
- Cloud / On-prem: £30K/year
- Monitoring tools: £10K/year
- Data sources (Bloomberg, LME APIs): £100K/year
────────────────────────────────────
Total: £140K/year

Consulting & Continuous Improvement:
- External consulting (model review, optimization): £50K/year
─────────────────────────────────────
Total: £50K/year

YEAR 2+ ANNUAL OPERATING COST: £695K
```

### **5-Year Total Cost of Ownership**

```
Year 1 (Build): £1.9M
Year 2–5 (Operations): £695K × 4 = £2.78M
─────────────────────────────
Total 5-Year: £4.68M

Cost per user (500 finance users): £9,360/user over 5 years
Cost per monthly forecast run: £1,900M ÷ (12 months × 5 years) = £31.7K per month
```

### **Comparison to Commercial Platforms** (5-year)

```
GIC Platform:         £4.68M (£9,360/user)
OneStream:           £3.8M (£7,600/user) — but less customization
Anaplan:             £9.8M (£19,600/user) — but universal FP&A
Palantir:           £23.5M (£47,000/user) — but broader analytics

GIC advantage: Purpose-built for JLR, 47% cheaper than Anaplan, 80% cheaper than Palantir
```

---

## **NEXT IMMEDIATE ACTIONS (Week of April 22, 2026)**

1. **Monday, Apr 22:** CFO kick-off meeting (confirm go/no-go, approve investment)
2. **Tuesday, Apr 23:** Hire Chief Analyst (or appoint from existing team)
3. **Wednesday, Apr 24:** Identify & contract SAP specialist (start data audit)
4. **Thursday, Apr 25:** Schedule stakeholder kickoff (CFO, Finance, IT, Audit)
5. **Friday, Apr 26:** Confirm team allocations; set up project management tools

---

## **CONCLUSION**

The GIC Platform Implementation Roadmap is **aggressive but achievable**. With disciplined execution across 4 phased workstreams, JLR can:

1. **Operationalize financial forecasting** in 3 months (Phase 1)
2. **Integrate real data** in 6 months (Phase 2)
3. **Embed in monthly close** in 9 months (Phase 3)
4. **Expand & optimize** in 12 months (Phase 4)

**Total investment: £1.9M Year 1 + £0.7M/year ongoing**  
**ROI: £2–5M hedging savings + £15–30M pricing optimization + £30–50M NWC improvement = £47–85M Year 1 value**  
**Payback: <1 month**

**Success requires:**
- ✅ Executive commitment (CFO championing probabilistic forecasting)
- ✅ Data quality foundation (audit, validation, governance)
- ✅ Talent (hiring strong data engineers + analytics)
- ✅ Discipline (governance, change control, no scope creep)

**Go/No-Go Decision Point: May 31, 2026 (Phase 1 completion)**

---

**Document Owner:** Chief Financial Officer  
**Project Sponsor:** [CFO Name]  
**Project Lead:** [Chief Analyst Name]  
**Next Review:** Weekly status meetings; Monthly steering committee

**Approved By:** [CFO] | [CIO] | [Controller]  
**Date:** [Signature Page]
