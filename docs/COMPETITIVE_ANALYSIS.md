# Competitive Analysis — GIC Financial Intelligence Platform

*Honest comparison. Where numbers are measured from GIC's own pipeline they are cited. Commercial competitor internals are proprietary — comparisons are analyst assessment of public capability, not controlled bake-offs.*

---

## 1. Competitive Landscape Map

| Vendor | Category | Price Point | Key Strength | Key Weakness | Our Advantage |
|---|---|---|---|---|---|
| **GIC** | AI Commodity Risk | £80K–£600K/yr | ML depth, conformal intervals, open-source | ERP connectors, scale, UX polish | — |
| **Anaplan** | Connected Planning | £300K–£2M+/yr | Enterprise UX, 200+ connectors, collaborative | No calibrated ML intervals; weak commodity domain | Uncertainty quant, probabilistic risk, cost |
| **Pigment** | FP&A / Planning | £150K–£500K/yr | Modern UX, fast implementation | Limited ML; no commodity-specific models | ML depth, conformal prediction, open-source |
| **o9 Solutions** | Supply Chain + Finance | £500K–£3M+/yr | Supply chain network optimisation | Financial-driver focus is thin; commodity ML weak | Purpose-built financial risk; IFRS 9 trail |
| **Kinaxis RapidResponse** | Supply Chain Planning | £500K–£2M+/yr | Real-time supply chain, concurrent planning | Finance/risk module is secondary product | Commodity→EBIT causality, VaR/CVaR, hedge opt |
| **SAP IBP** | Integrated Business Planning | Bundled with S/4HANA | Native SAP integration; enterprise proven | Slow to deploy; generic; costly ML add-ons | Speed (8 wk vs 18 mo), ML sophistication, cost |
| **Excel + Bloomberg** | Manual | £20–25K/user/yr Bloomberg | Familiar; flexible | No calibration, no VaR, no audit, 4–6 week lag | Everything |
| **Coupa (Risk)** | Procurement Risk | £200K–£1M/yr | Supplier risk, spend analytics | Not a commodity price forecasting tool | Commodity ML; EBIT impact modelling |
| **Palantir Foundry** | Data/AI Platform | £1M+/yr | General AI platform; heavy customisation | Requires significant integration work; no packaged commodity models | Automotive-specific out-of-the-box, faster ROI |

---

## 2. Feature Matrix

| Feature | GIC | Anaplan | Pigment | o9 | Kinaxis | SAP IBP | Excel+BB |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Commodity price forecasting (ML) | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ❌ |
| SARIMAX seasonal decomposition | ✅ | ❌ | ❌ | ❌ | ❌ | ⚠️ | ❌ |
| XGBoost gradient boosting | ✅ | ❌ | ❌ | ⚠️ | ❌ | ❌ | ❌ |
| Regime-adaptive ensemble (Hurst) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Monte Carlo simulation (10K+ paths) | ✅ | ⚠️ | ❌ | ⚠️ | ⚠️ | ❌ | ❌ |
| Calibrated conformal prediction intervals | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| VaR / CVaR (tail risk) | ✅ | ❌ | ❌ | ⚠️ | ⚠️ | ❌ | ❌ |
| CUSUM + BOCPD change-point detection | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| SHAP feature attribution | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Quantile regression VaR (asymmetric) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Hedge optimiser (portfolio-theory) | ✅ | ⚠️ | ❌ | ⚠️ | ⚠️ | ❌ | ❌ |
| Real-time WebSocket market feed | ✅ | ❌ | ❌ | ⚠️ | ⚠️ | ❌ | ❌ |
| RBAC with audit trail | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Immutable JSONL audit (IFRS 9 ready) | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ❌ |
| Bias tracking + escalation alerts | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| BOM-weighted commodity index | ✅ | ⚠️ | ❌ | ✅ | ✅ | ⚠️ | ⚠️ |
| Plan-to-Perform EBIT waterfall bridge | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ⚠️ |
| Warranty analytics (EV learning curve) | ✅ | ❌ | ❌ | ⚠️ | ⚠️ | ⚠️ | ❌ |
| Open-source (no licence fee) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| API-first (embeddable) | ✅ | ⚠️ | ⚠️ | ⚠️ | ⚠️ | ❌ | ❌ |
| ERP connectors (certified) | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| Multi-tenant SaaS | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A |
| SSO / enterprise identity | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

*✅ = available; ⚠️ = partial/basic; ❌ = not available*

---

## 3. Where We Win

**1. Uncertainty quantification (unique differentiator).**
Split-conformal + ACI prediction intervals provide distribution-free coverage guarantees — a capability essentially absent from every commercial planning platform benchmarked. Competitors offer scenario ranges; GIC offers provably-calibrated intervals (Gibbs & Candès 2021; Angelopoulos & Bates 2021). Source: `src/models/conformal.py`.

**2. Probabilistic risk quantification.**
10,000-run fat-tailed Monte Carlo with VaR(95%)/CVaR(95%) decomposed by commodity/FX/demand is not standard in any FP&A platform. Anaplan/Pigment provide deterministic scenario analysis. Source: `layers/layer4_simulation/controller.py`.

**3. ML explainability.**
SHAP TreeExplainer per-forecast signed driver attribution converts opaque XGBoost output into ranked financial drivers. No commercial planning tool does this. Source: `src/models/explainability_shap.py`.

**4. Automotive commodity domain depth.**
12 automotive-specific commodities with calibrated BOM weights, EV battery-metal modelling (Lithium, Cobalt, Nickel), warranty analytics with EV learning curve, and Rhodium/PGM modelling. Generic planning tools require months of configuration to reach the same starting point.

**5. Open-source + API-first.**
Zero licence lock-in. FastAPI with OpenAPI schema means any ERP or BI tool can consume GIC as a microservice. On-prem deployment for data-sovereign environments. Source: `src/api/app.py`.

**6. IFRS 9-ready governance.**
Immutable UUID-keyed JSONL audit trail + systematic bias tracking at configurable thresholds (5% alert, 10% escalation) provides the documented evidence required for hedge accounting. Competitors have general audit logs; none are specifically designed for hedge effectiveness documentation.

---

## 4. Where We Lag

**1. ERP data connectors (highest severity).**
SAP S/4HANA, Oracle ERP, Workday connectors are stubs in the current build. Competitors have certified, production-hardened connectors with IT-approved security. Without real ERP data, P&L inputs are JLR-calibrated synthetic values — credible for demos, not for production sign-off.

**2. Multi-tenancy / enterprise scale.**
Current architecture is single-tenant. Running 50+ OEM deployments requires tenant isolation, shared infrastructure management, and SOC2 Type II certification. None of these exist yet.

**3. UX maturity.**
React SPA with Recharts is functional and fast. Commercial platforms (Anaplan, Pigment) have polished, collaborative, mobile-friendly UX with commenting, workflow approvals, and embedded training. The gap is significant for enterprise adoption.

**4. SSO / enterprise identity.**
No SAML, OKTA, or Azure AD integration. Enterprise security reviews will block deployment without this.

**5. Automated MLOps / model retraining.**
Retraining is manual. No drift-triggered refit, model monitoring, or CI/CD for models. Production deployments require scheduled automated retraining.

**6. Forecast accuracy on volatile tail.**
Natural Gas 31.1% MAPE, Palladium 29.1% (2024 hold-out). Foundation models (Chronos, TimesFM) could close this gap but are not yet integrated.

**7. Regulatory certification.**
No SOC2, ISO 27001, or financial software certification. A dealbreaker for regulated financial environments.

**8. Support and SLAs.**
No 24/7 support, no professional services organisation, no established SLA framework.

---

## 5. Academic SOTA Comparison

| Method | GIC Status | Academic SOTA | Gap |
|---|---|---|---|
| **Foundation models (Chronos/TimesFM)** | ❌ Not integrated | Zero-shot competitive; Chronos-2 adds covariate support (2025) | High value — addresses volatile-tail gap; roadmap P2 |
| **N-BEATS / N-HiTS** | ❌ Not integrated | 10–25% MAPE reduction on long-horizon benchmarks | Medium; EVALUATE before committing |
| **Temporal Fusion Transformer** | ❌ Not integrated | Multi-horizon quantile + variable selection | High complexity; roadmap P1 |
| **SARIMAX + XGBoost ensemble** | ✅ Shipped | SOTA on supervised stable-series | 7% on Copper — competitive with published ensemble methods |
| **Conformal prediction (split + ACI)** | ✅ Shipped | Current best-practice for calibrated UQ in production | GIC is at academic frontier on UQ |
| **CUSUM + BOCPD** | ✅ Shipped | Standard in sequential analysis literature | Module complete; auto-reforecast wiring pending |
| **SHAP TreeExplainer** | ✅ Shipped | Standard for tree-model XAI | At library frontier |
| **Quantile XGBoost (pinball loss)** | ✅ Shipped | XGBoost 2.x joint objective; Conformalized QR not yet added | CQR integration pending |

**Why SARIMAX + XGBoost over N-BEATS / TFT for this use case:**
- Interpretability requirement: CFOs and auditors need to understand why the model said what it said. N-BEATS is a black-box MLP; TFT has variable selection but is still a deep learning model with limited intuitive explanation.
- Data volume: ~84 months per commodity (7 years monthly). Deep learning methods need 3–5× more data to outperform well-tuned classical methods on this scale.
- Deployment simplicity: XGBoost + statsmodels are battle-tested Python packages with no GPU requirement. TFT requires PyTorch, significant hyperparameter tuning, and GPU for reasonable training time.
- Accuracy on stable commodities: 7.0% MAPE on Copper is competitive with published N-BEATS/TFT results on commodity benchmarks. The volatile tail (NatGas, Palladium) is where foundation models would help most.

---

## 6. Honest Gap Analysis (Priority Order)

| # | Gap | Description | Effort | Priority |
|---|---|---|---|---|
| 1 | **ERP data connectors** | SAP BAPI / REST connector to replace synthetic operational data; deal-blocker for production | L (2 mo) | P0 |
| 2 | **Multi-tenancy** | Tenant isolation, shared infra, SOC2 audit prep | L (3 mo) | P0 |
| 3 | **Volatile-tail forecasting** | Natural Gas 31%, Palladium 29% MAPE; Chronos/TimesFM zero-shot challenger | M (4 wk) | P0 |
| 4 | **Automated MLOps** | Drift-triggered retraining, model monitoring, CI/CD; no production model hygiene yet | L (6 wk) | P0 |
| 5 | **SSO / enterprise identity** | SAML, OKTA, Azure AD integration | M (3 wk) | P0 |
| 6 | **Conformalized Quantile Regression** | CQR over quantile forecasts → asymmetric calibrated VaR tails; extends shipped modules | M (2 wk) | P1 |
| 7 | **BOCPD auto-reforecast wiring** | Change-point module ships; connecting break events → reforecast trigger is remaining work | S (1 wk) | P1 |
| 8 | **UX maturity** | Collaborative workflows, mobile, commenting, approvals | L (ongoing) | P2 |
