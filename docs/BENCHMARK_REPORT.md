# GIC Benchmark Report — GIC vs State-of-the-Art

**Prepared for:** GIC Plan-to-Perform Engine board / steering committee
**Date:** 2026-06-11
**Purpose:** An honest, board-credible benchmark of GIC against (a) **commercial FP&A / planning platforms** — Anaplan, Pigment, o9 Solutions, Board, Workday Adaptive Planning, Kinaxis — and (b) **academic SOTA** forecasting. Calls out where GIC leads, where it lags, and a prioritized backlog.

> **Honesty principle.** Where a number is *measured* it is cited to GIC's own 2024 hold-out backtest (`docs/TECHNICAL_ASSESSMENT.md` / `docs/FULL_ARCHITECTURE_RUN.md`). Where it is a *judgement* (e.g. comparison to a commercial platform whose internals are proprietary) it is marked **indicative** and reflects analyst assessment of public capability, not a controlled bake-off. See §6 Methodology & Caveats.

---

## 0. What GIC actually is (grounding)

- **Forecasting:** SARIMAX + XGBoost ensemble, 4 methods (SARIMAX / XGBoost-macro / futures-curve / scenario), Hurst-regime adaptive weighting, across 12 commodities.
- **Measured accuracy (2024 hold-out):** best MAPE **7.0% (Copper)**, **8.9% (Platinum)**, **9.8% (Polypropylene)**; worst **31.1% (Natural Gas)**, **29.1% (Palladium)**, **17.2% (ABS Resin)**. Directional accuracy 52–76%.
- **Uncertainty:** parametric 80% CI **calibrated to 79%** (measured), now augmented with **split-conformal + Adaptive Conformal Inference** (`src/models/conformal.py`).
- **Risk:** 10,000-run Monte Carlo, fat-tailed (Student-t df=5) commodity shocks; **VaR(95%) ≈ £705M**, CVaR(95%) ≈ −£452M; EBIT 80% CI **£1,231M–£1,571M** (measured).
- **Financial model:** driver-based P&L (Revenue = vol×price×(1−incentive); COGS = BOM-weighted commodity index; warranty/depr/tax), validated to within ~1% on sensitivity checks.
- **This iteration added:** RBAC auth layer; actionable **Insight Engine** (prescriptive, £-quantified); **Plan-to-Perform Variance Bridge** (Volume→Price/Mix→Commodity→FX→Warranty→Other); **warranty analytics**; **real-time WebSocket market feed**; SOTA modules — **split-conformal + ACI**, **SHAP attribution**, **CUSUM + BOCPD change-point detection**, **gradient-boosted quantile regression**, **regime detection (Hurst)**, anomaly detection (z-score/IQR), backtesting harness, hedge optimizer.

---

## 1. Scorecard (1–5; 5 = best-in-class)

Scores: **GIC** = current state *after this iteration*. **Commercial SOTA** = best-of-breed enterprise planning platforms (indicative). **Academic SOTA** = published research frontier (indicative).

| Dimension | GIC | Comm. SOTA | Acad. SOTA | One-line justification |
|---|:---:|:---:|:---:|---|
| **Forecast accuracy & calibration** | 3.5 | 3.5 | 5 | GIC strong on stable commodities (7% MAPE) but weak tail (31%); no foundation-model forecaster yet. Commercial platforms are planning-first, forecasting-light. Academic = TSFM/PatchTST frontier. |
| **Uncertainty quantification** | 4.5 | 2.5 | 5 | GIC has split-conformal + ACI + Monte Carlo VaR/CVaR — *ahead of* most commercial planning tools, which offer scenario ranges not calibrated intervals. |
| **Explainability** | 4 | 3 | 4.5 | SHAP attribution + driver-based P&L + audit narratives. Commercial tools explain *plan math*, not ML drivers. |
| **Scenario / Monte-Carlo** | 4.5 | 3.5 | 4 | 10k-run fat-tailed MC + 7 presets + what-if builder. Anaplan/Board do scenarios but rarely full probabilistic MC with VaR/CVaR. |
| **Prescriptive / recommendations** | 4 | 3 | 4 | Insight Engine emits £-quantified prescriptive actions + early-warning score. Most commercial tools stop at analytics, not prescription. |
| **Real-time data** | 3.5 | 3 | 3.5 | WebSocket live tape + yfinance/FRED/CCXT. Commercial tools integrate ERP in batch; GIC's feed is market-data live but partly synthetic. |
| **Driver-based planning** | 4 | 5 | 3 | GIC's driver tree is transparent and correct, but commercial platforms (Anaplan, Pigment, o9) are *purpose-built* multi-dimensional planning engines at enterprise scale. |
| **Governance / audit** | 4 | 5 | 2.5 | Immutable JSONL audit trail + RBAC + bias tracking. Enterprise platforms have mature SOC2/SSO/approval-workflow governance GIC has not yet matched. |
| **Data integration** | 2.5 | 5 | 2 | GIC connects market/macro APIs but ERP (SAP) / data-lake connectors are stubs; operational data is synthetic. Commercial = deep, certified ERP/CRM connectors. |
| **UX** | 3.5 | 4.5 | 2 | 8-page Streamlit dashboard is strong for a POC; commercial platforms have polished, multi-tenant, mobile, collaborative UX. |
| **Weighted overall** | **~3.8** | **~3.9** | **~3.6** | GIC is *board-credible and competitive*; leads on uncertainty/prescription, lags on integration/enterprise-scale planning. |

---

## 2. Where GIC Already Leads or Matches

1. **Transparent, correct driver-based P&L** — every £ traces from commodity → BOM weight → COGS → EBIT, validated to ~1% on sensitivity tests. Matches commercial driver-based planning on *transparency*, and beats them on *commodity-to-EBIT causality wiring*.
2. **Conformal-calibrated prediction intervals (LEADS).** Split-conformal + Adaptive Conformal Inference give distribution-free coverage guarantees — a capability essentially **absent** from commercial FP&A platforms, which offer un-calibrated scenario bands. This is GIC's single biggest differentiator.
3. **Integrated prescriptive insights + variance bridge (LEADS/MATCHES).** Quantified, prescriptive Insight Cards + a standards-compliant Volume→Price/Mix→Commodity→FX→Warranty→Other EBIT bridge in one engine. Most tools do *either* analytics *or* planning, not a closed prescribe-and-explain loop.
4. **Full probabilistic Monte-Carlo risk (LEADS).** 10k-run fat-tailed VaR/CVaR with a measured **79% vs 80%** calibration — most planning platforms stop at deterministic scenarios.
5. **Open-source, governed LLM/ML stack with full audit trail (LEADS on transparency).** Immutable JSONL audit, RBAC, bias-tracking, SHAP — auditable end-to-end vs black-box vendor models.
6. **Commodity/automotive domain depth.** BOM-weighted index, futures-curve method, hedge optimizer, warranty analytics — purpose-built for an automotive OEM, vs generic horizontal platforms.

---

## 3. Gap Analysis (where GIC lags — specific)

| # | Gap | Detail | Severity |
|---|---|---|---|
| G1 | **No foundation-model forecaster** | No Chronos/TimesFM/Moirai or PatchTST/N-HiTS; volatile tail (NatGas 31%, Palladium 29% MAPE) is unmitigated; per-commodity training is slow. | High |
| G2 | **Aggregate-only regional demand** | No NA/EU/APAC/China splits; demand leg of P&L + variance bridge is coarse. | High |
| G3 | **Causal elasticity not validated** | Log-log Ridge is correlational; "causality not validated" is documented. No DML/DoWhy. | Med–High |
| G4 | **Operational data is synthetic, not ERP-connected** | P&L uses JLR-calibrated synthetic operations; SAP S/4HANA + data-lake connectors are stubs. No real GL actuals. | High |
| G5 | **No automated retraining / MLOps** | Retraining is manual; no scheduler, model monitoring, drift-triggered refit, or CI/CD for models. | Med–High |
| G6 | **Limited backtest horizon & breadth** | Single 2024 hold-out year; no multi-regime, multi-year rolling backtest across crises; calibration validated on one window. | Med |
| G7 | **Change-point detection** ✅ *largely closed this iteration* | Now ships true **CUSUM + simplified Gaussian BOCPD** (`src/models/change_point.py`) alongside the z-score/IQR anomaly flags. Remaining: auto-wire break events → reforecast trigger in the pipeline. | Med → Low |
| G8 | **3 commodities fully synthetic** | Rhodium, Polypropylene, ABS Resin via O-U process (no free exchange instrument); no factor-model anchor. | Med |
| G9 | **Real-time feed partly synthetic** | WebSocket tape applies a mean-reverting random walk when live data absent; not a hardened market-data subscription. | Med |
| G10 | **Enterprise data integration shallow** | No certified ERP/CRM/data-warehouse connectors, no SSO at enterprise grade, no multi-tenant. | High (for productionization) |
| G11 | **Quantile forecasting** ✅ *largely closed this iteration* | Now ships **gradient-boosted pinball quantile regression** (`src/models/quantile_forecast.py`, monotone post-sort prevents crossing). Remaining: wire quantile fits into the VaR/CVaR path and add full Conformalized Quantile Regression (CQR). | Med → Low |
| G12 | **LLM narrative is template-based** | Explainability narratives are templated, not LLM/RAG-grounded; no conversational what-if. | Low–Med |

---

## 4. Prioritized, Actionable Backlog

Effort: **S** (≤1 wk), **M** (2–4 wk), **L** (1–2 mo). Impact: indicative. Status ties "Now" items to this-iteration additions.

### NOW (next 1–2 sprints — calibration, breaks, recommendations core)

| Gap | Recommended fix | Effort | Expected impact | Status |
|---|---|---|---|---|
| Uncertainty | Split-conformal + Adaptive Conformal Inference for all forecasts | M | 80% CI calibrated *by construction* under fat tails/drift | ✅ Closed-this-iteration (`conformal.py`) |
| Explainability | SHAP per-forecast driver attribution feeding narratives | M | Opaque XGBoost → ranked signed drivers; trust/audit | ✅ Closed-this-iteration (`explainability_shap.py`) |
| Prescription | Insight Engine: £-quantified prescriptive actions + early-warning score | M | Analytics → action; one risk number for leadership | ✅ Closed-this-iteration (`src/insights/`) |
| Variance | Plan→Actual EBIT Variance Bridge (Vol→Price→Commodity→FX→Warranty→Other) | M | Standards-compliant attribution of EBIT gap | ✅ Closed-this-iteration (`variance_bridge.py`) |
| Real-time | WebSocket live market tape + REST snapshot | M | Sub-second live commodity→EBIT context | ✅ Closed-this-iteration (`api/routes/realtime.py`) |
| Governance | RBAC auth layer (Admin/User permission matrix) | M | Access control + audit foundation for prod | ✅ Closed-this-iteration (`auth/`) |
| G7 Change-point | True **CUSUM + BOCPD** module (`change_point.py`); remaining: emit break events → auto-trigger reforecast | S (remaining) | Earlier regime-break action; less stale forecasts | ✅ Module shipped; wiring ⏳ |
| G11 Quantiles | **Quantile-XGBoost** shipped (`quantile_forecast.py`); remaining: feed VaR/CVaR + full CQR | S (remaining) | Sharper, asymmetric, honest tail risk into £705M VaR | ✅ Module shipped; wiring ⏳ |

### NEXT (this quarter — accuracy, causality, conversational UX)

| Gap | Recommended fix | Effort | Expected impact | Status |
|---|---|---|---|---|
| G1 Forecaster | Add **Chronos/TimesFM zero-shot challenger** + N-HiTS/TiDE; benchmark vs ensemble in walk-forward harness; promote per-commodity winner; wrap in conformal | L | Attacks volatile tail + time-to-value; potential 5–20% MAPE cut on weak commodities | ⏳ Open |
| G3 Causality | **Double ML** for price elasticity (causal estimate + CIs); optional DoWhy DAG | M | Closes "causality not validated"; defensible scenario math | ⏳ Open |
| G12 / LLM | **LLM function-calling what-if** over existing `/simulation` + shock APIs; **RAG-grounded narrative** | M | Conversational scenarios; cited "why" narratives | ⏳ Open |
| G8 Synthetics | **Cross-commodity factor model** (PGM/energy/battery factors) anchors Rhodium/PP/ABS + correlated MC shocks | M | More realistic VaR; better illiquid-commodity forecasts | ⏳ Open |
| G6 Backtest | Multi-year, multi-regime rolling backtest (incl. 2020/2022 shocks) | M | Robust, crisis-tested accuracy + calibration evidence | ⏳ Open |
| Hedging | Extend hedge optimizer to **CVaR objective** | M | Tail-aware, coherent-risk hedging | ⏳ Open |
| Warranty | **Weibull/Cox survival** layer over EWMA warranty model | M | Sharper accrual adequacy + failure-mode foresight | ⏳ Open |

### LATER (productionization & enterprise scale)

| Gap | Recommended fix | Effort | Expected impact | Status |
|---|---|---|---|---|
| G4 / G10 | Real **SAP S/4HANA + Snowflake/Databricks** connectors; real GL actuals | L | True actuals vs synthetic; enterprise-grade integration | ⏳ Open |
| G5 MLOps | Automated retraining scheduler + drift monitoring + model CI/CD | L | Self-maintaining models; no manual refit | ⏳ Open |
| G2 Regional | Regional demand splits (NA/EU/APAC/China) + demand sensing | L | Granular demand leg in P&L + bridge | ⏳ Open |
| G9 Feed | Hardened market-data subscription (Bloomberg/LME/ICIS) replacing synthetic fallback | L | Production-grade live data | ⏳ Open |
| G10 UX | Mobile PWA + multi-tenant SSO + collaborative planning UX | L | Enterprise UX parity | ⏳ Open |

---

## 5. Re-Comparison — "After This Iteration"

The additions this iteration moved five dimensions. Deltas vs the pre-iteration baseline (analyst estimate):

| Dimension | Before | After | Δ | Driver of change |
|---|:---:|:---:|:---:|---|
| Uncertainty quantification | 3.0 | **4.5** | +1.5 | Split-conformal + ACI added on top of MC VaR/CVaR |
| Explainability | 3.0 | **4.0** | +1.0 | SHAP attribution + audit narratives |
| Prescriptive / recommendations | 2.5 | **4.0** | +1.5 | Insight Engine (quantified, prescriptive) + early-warning |
| Real-time data | 2.5 | **3.5** | +1.0 | WebSocket live market tape |
| Governance / audit | 3.0 | **4.0** | +1.0 | RBAC layer on top of audit trail + bias tracking |
| Driver-based planning | 3.5 | **4.0** | +0.5 | Variance bridge closes the plan-to-perform loop |
| Forecast accuracy & calibration | 3.5 | **3.5** | 0.0 | Unchanged — no new forecaster yet (G1 open) |
| Data integration | 2.5 | **2.5** | 0.0 | Unchanged — ERP still stubbed (G4 open) |
| **Weighted overall** | **~3.3** | **~3.8** | **+0.5** | Five dimensions up; accuracy & integration unchanged |

### Remaining Top-5 Gaps to Close Next

1. **G1 — Foundation-model forecaster** (Chronos/TimesFM + N-HiTS challenger) → attack the volatile-commodity tail and time-to-value. *Highest accuracy lever.*
2. **G7/G11 — True change-point (CUSUM/BOCPD) + Conformalized Quantile Regression** → break-triggered reforecast and sharper asymmetric VaR tails. *Quickest analytical wins (extend shipped modules).*
3. **G3 — Causal elasticity via Double ML** → convert the documented "causality not validated" caveat into a strength.
4. **G4 — Real ERP/GL integration** → replace synthetic operational data; the biggest credibility gap for production.
5. **G5 — Automated retraining / MLOps** → make the platform self-maintaining (drift-triggered refit, model CI/CD, monitoring).

---

## 6. Methodology & Caveats

**Measured numbers** (from `docs/TECHNICAL_ASSESSMENT.md` + `docs/FULL_ARCHITECTURE_RUN.md`, 2024 hold-out backtest; models trained Jun-2019–Dec-2023, tested Jan–Dec-2024):

- Best MAPE **7.0% (Copper)**, **8.9% (Platinum)**, **9.8% (Polypropylene)**; worst **31.1% (Natural Gas)**, **29.1% (Palladium)**, **17.2% (ABS Resin)**.
- Directional accuracy **52–76%**.
- 80% CI calibration: **79%** realized vs **80%** target (PASS).
- **VaR(95%) ≈ £705M**; CVaR(95%) ≈ **−£452M**; EBIT 80% CI **£1,231M–£1,571M**.
- Financial sensitivity checks within ~1% (e.g. Steel +10% → COGS +2.2%).

**Indicative numbers** (analyst judgement, *not* a controlled bake-off):

- All **scorecard scores** (§1, §5), including the "before"/"after" deltas — commercial platforms' internals are proprietary; scores reflect public capability assessment, not measured head-to-head accuracy.
- All **"expected impact"** figures in the backlog and the MAPE-uplift ranges for foundation models / quantile methods — drawn from adjacent-domain literature (M-competitions, energy, agricultural commodities), not GIC data. Validate on GIC's own `backtesting.py` harness before relying on them.

**Implementation caveats (honesty about what is *not* yet built):**

- **Change-point detection** now ships true **CUSUM + simplified Gaussian BOCPD** (`src/models/change_point.py`) in addition to the z-score/IQR anomaly flags (`src/insights/anomaly_detector.py`). The remaining open item is wiring break events to auto-trigger a reforecast (backlog G7).
- **Quantile regression** now ships as a native **gradient-boosted pinball** module (`src/models/quantile_forecast.py`); the remaining open item is feeding those quantiles into the VaR/CVaR path and adding full Conformalized Quantile Regression (backlog G11). Uncertainty also comes from **split-conformal + ACI** (`conformal.py`) and Monte-Carlo.
- **Regime detection** is **Hurst-exponent (R/S)** based adaptive weighting, not Markov regime-switching.
- The **real-time feed** seeds from synthetic CSV with a mean-reverting random walk when live data is absent — live-capable, but not a hardened market-data subscription.
- **Operational/P&L inputs are JLR-calibrated synthetic data**, not connected to a real SAP general ledger; financial-model *equations* are validated, the *operational inputs* are simulated.
- **Calibration evidence is a single 2024 window** — not yet multi-regime/multi-year (backlog G6).

**Net assessment.** GIC is a board-credible, domain-deep POC that **leads commercial planning platforms on uncertainty quantification, probabilistic risk, and prescriptive+explainable analytics**, and **lags them on enterprise data integration, scale, and UX**. Against academic SOTA it is mid-tier on raw forecast accuracy (no foundation models yet) but ahead on the *applied decision loop* (calibrated intervals → risk → prescription → variance bridge → audit). The prioritized backlog closes the accuracy and integration gaps without disturbing the differentiated calibration/prescription core already shipped.
