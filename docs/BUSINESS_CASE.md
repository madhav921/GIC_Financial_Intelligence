# Business Case — GIC Financial Intelligence Platform

---

## 1. Problem Statement

Automotive OEMs spend **60–70% of COGS on materials** — making commodity price volatility the single largest controllable risk to EBIT.

**Quantified exposure on a £176bn revenue base (GIC's modelled scale):**

| Commodity | BOM Weight | Revenue × 27% margin × weight | 1% price move = EBIT impact |
|---|---|---|---|
| Steel | 22% | £176bn × 0.27 × 0.22 = £10.5bn COGS | ~£105M |
| Lithium | 18% | £176bn × 0.27 × 0.18 = £8.6bn COGS | ~£86M |
| Aluminum | 12% | £176bn × 0.27 × 0.12 = £5.7bn COGS | ~£57M |
| Copper | 7% | £176bn × 0.27 × 0.07 = £3.3bn COGS | ~£33M |
| Cobalt | 7% | £176bn × 0.27 × 0.07 = £3.3bn COGS | ~£33M |
| Top 5 combined | ~66% | £31.4bn material COGS | ~£314M per 1% uniform move |

**Key facts:**
- Lithium lost 85% of value from Dec 2022 peak to Jan 2024 — a swing of >£7bn in annual material cost exposure for an OEM at this scale
- Natural Gas 12-month MAPE of 31% means the industry forecast error on energy costs alone could exceed £500M annually
- VaR(95%) = £19.3bn on this revenue base; the platform has identified and quantified this risk

**Current state for most OEMs:** Excel-based planning + Bloomberg terminal + quarterly manual reports. Typical lag from price move to treasury action: 4–6 weeks.

---

## 2. Solution Value Proposition

### Benefit 1: Forecast Accuracy → Reduced Safety Stock → Working Capital Freed

- GIC achieves 7.0% MAPE on Copper, 8.9% on Platinum vs industry-typical 15–25% (Excel/ARIMA baseline)
- Every 1pp MAPE improvement on a £10bn commodity exposure reduces required safety-stock buffer by ~£100M
- Conservative estimate: 3pp MAPE improvement × £10bn Steel exposure = **£300M working capital freed**

### Benefit 2: Early-Warning System → 2–4 Week Earlier Reaction

- CUSUM detects mean shifts in ~2–3 observations; BOCPD provides calibrated break probability per step
- vs manual monitoring: analysts review Bloomberg weekly at best → 4–6 week detection lag
- **2–4 weeks of earlier hedging action** on a £86M/1% Lithium exposure = material cost avoidance on trend moves

### Benefit 3: Hedge Optimiser → Expected Savings Quantified

- Portfolio-theory optimal h* = argmin[α·E[cost(h)] + (1−α)·VaR(h)]
- Blends expected procurement cost minimisation with VaR cap (α=0.5 default)
- ARCHITECTURE_GUIDE.md cites: **£1.5M/yr expected savings vs industry-standard static 50% hedge ratio** on the modelled commodity portfolio
- Source: optimizer output for Aluminum at 75% vs 50% static ratio across historical volatility

### Benefit 4: Bias Escalation → Governance Compliance (IFRS 9 / Hedging Relationships)

- IFRS 9 requires documented evidence that hedge relationships are "highly effective" (80–125% band)
- Systematic forecast bias (>5% alert, >10% escalation) directly threatens hedge accounting eligibility
- Immutable JSONL audit trail + bias tracker provides the governance evidence required for IFRS 9 hedge documentation
- **Risk averted:** Loss of hedge accounting = P&L volatility reclassified from OCI to income statement; EBIT impact can exceed hedge notional

### Benefit 5: Scenario Simulation → CFO-Level Risk Visibility in Minutes vs Weeks

- 10,000 Monte Carlo paths run in <2 seconds (vectorised numpy)
- 7 preset scenarios (Bear/Bull/EV boom/Recession/Supply shock/Currency crisis/Base) available on demand
- **Before GIC:** CFO requests bear-case scenario → FP&A team spends 2–3 weeks building in Excel
- **After GIC:** CFO runs bear-case in 3 clicks with probabilistic distribution + VaR/CVaR

---

## 3. ROI Model

| Scenario | Conservative | Base | Optimistic |
|---|---|---|---|
| **Inputs** | | | |
| Annual revenue | £24bn (GIC modelled) | £24bn | £24bn |
| Commodity COGS exposure | 27% | 27% | 27% |
| MAPE improvement (pp) | 2pp | 4pp | 7pp |
| Hedge savings (annual) | £0.8M | £1.5M | £3.0M |
| Early-warning reaction benefit | £5M | £15M | £30M |
| **Outputs** | | | |
| Working capital freed | £180M | £360M | £630M |
| EBIT improvement (direct savings) | £5.8M | £16.5M | £33M |
| Implementation cost (one-time) | £500K | £500K | £500K |
| Annual SaaS licence | £250K | £400K | £600K |
| **Payback period** | 14 months | 5 months | 3 months |
| **3-year NPV (10% WACC)** | £12M | £38M | £80M |

*Working capital benefit counted at 0.1% cost of capital per £ freed (opportunity cost).*

---

## 4. Target Customer Profile

**Primary buyer:** VP Procurement / Head of Treasury / CFO direct report

**Ideal customer profile:**

| Dimension | Criteria |
|---|---|
| Revenue | £5bn+ (meaningful commodity exposure) |
| Sector | Automotive OEM, Tier-1 supplier, EV manufacturer |
| Commodity inputs | >20 distinct materials in BOM |
| Current process | Excel + Bloomberg + quarterly manual reports |
| ERP | SAP S/4HANA or Oracle ERP (roadmap connector targets) |
| Pain point | IFRS 9 hedge documentation + CFO board reporting on commodity risk |
| Budget authority | £100K–£1M annually for risk management tooling |

**Secondary buyers:** Head of FP&A (variance bridge / plan vs actual), Chief Risk Officer (VaR/CVaR), Commodity Manager (hedge recommendations), Sustainability Officer (scope 3 emissions — roadmap).

**Anti-profile:** Pure software companies (no commodity exposure), companies with existing Anaplan/o9 investment (integration cost exceeds switching benefit in year 1), companies with <£1bn revenue (exposure too small for ROI).

---

## 5. Market Opportunity

**TAM:** Global commodity risk management software market ~£3.5bn (2025 estimate), growing ~12% CAGR driven by EV transition (new battery-metal exposures), post-COVID supply chain volatility, IFRS 9 governance requirements.

**SAM:** Automotive OEMs and Tier-1 suppliers with £5bn+ revenue and >20 commodity inputs. ~200 companies globally (JLR, BMW, Stellantis, Ford, GM, Toyota, Volkswagen Group, Magna, Continental, Bosch, etc.). At £300K average ACV = **£60M SAM**.

**SOM (3-year):** Targeting 5–10 enterprise deployments in years 1–2, scaling to 25–30 via partner channel. **£7–10M ARR SOM**.

**Key tailwind:** EV transition requires automotive OEMs to forecast 8–12 battery-metal commodities they historically had no exposure to (Lithium, Cobalt, Nickel, Manganese). This creates a greenfield procurement risk management need — existing Anaplan/SAP configurations don't have these commodity models.

---

## 6. Pricing Model

| Tier | ACV | Target | Features |
|---|---|---|---|
| **Starter** | £80K/yr | £5–15bn OEM / Tier-1 | 3 commodity modules, SARIMAX forecast, 3 scenarios, basic P&L dashboard, JSONL audit |
| **Growth** | £250K/yr | £15–50bn OEM | All 12 commodities, XGBoost ensemble, Monte Carlo, hedge optimiser, conformal intervals, bias tracking, InsightCards, WebSocket feed |
| **Enterprise** | £600K+/yr | £50bn+ OEM (JLR/BMW scale) | Full platform + custom ERP connector + dedicated retraining schedule + IFRS 9 documentation generator + Quantile VaR + CUSUM/BOCPD + SHAP explainability + SLA + on-prem option |

**Add-ons:**
- SAP S/4HANA certified connector: £50K one-time
- Automated model retraining (MLOps): +£75K/yr
- Custom commodity modelling (Rhodium, specialty metals): £30K/commodity

---

## 7. Implementation Timeline

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | Kickoff + data audit | Data inventory, ERP connection assessment, BOM weight validation |
| 2 | Data integration | CSV/API data feeds connected; historical commodity prices validated |
| 3 | Model calibration | SARIMAX + XGBoost trained on customer's commodity history; MAPE baseline |
| 4 | Financial model config | BOM weights, revenue segments, COGS structure mapped to customer GL |
| 5 | Scenario calibration | 7 preset scenarios reviewed with CFO/VP Procurement; custom scenarios added |
| 6 | Governance setup | RBAC users created; audit trail tested; bias thresholds agreed |
| 7 | UAT | 2-week parallel run vs existing Excel process; discrepancies investigated |
| 8 | Go-live | Production deployment; CFO dashboard walkthrough; IFRS 9 documentation pack |

---

## 8. Risk Factors

| Risk | Severity | Mitigation |
|---|---|---|
| **Data quality** | High | O-U synthetic fallback for offline demo; real data validation via Great Expectations (roadmap) |
| **Model drift** | Medium | Bias tracker alerts at >5%; manual retraining documented; automated retraining on roadmap |
| **IFRS 13/17 regulatory** | Medium | Audit trail + bias documentation supports fair-value hierarchy and hedge effectiveness evidence; not a substitute for qualified accountant sign-off |
| **ERP integration complexity** | High | SAP BAPI connectors are roadmap P0; current deployment uses CSV extract until certified connector available |
| **Volatile commodity tail** | Medium | Natural Gas 31% MAPE, Palladium 29% — disclosed to customers; conformal intervals provide honest uncertainty; foundation model (Chronos) on roadmap to close gap |
| **Single-tenant architecture** | Medium | Current deployment is single-tenant; multi-tenancy is P0 roadmap item required for SaaS scale |
