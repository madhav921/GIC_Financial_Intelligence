# GIC Research — State-of-the-Art "Wow-Factor" Survey

**Prepared for:** GIC Plan-to-Perform Engine (AI Financial Intelligence, automotive OEM)
**Date:** 2026-06-11
**Scope:** SOTA techniques (2023–2026) across forecasting, uncertainty, regime/structural breaks, causality, LLMs-in-finance, and commodity/automotive-specific methods — with an ADOPT / EVALUATE / SKIP call for GIC.

> **Reading note on figures.** Accuracy/uplift numbers below are *indicative* — drawn from the cited literature on adjacent datasets (M-competitions, electricity/energy, agricultural commodities) and from GIC's own 2024 hold-out backtest where stated. Commodity forecasting is regime-dependent; expect realized uplift to be a fraction of headline benchmark gains. Treat every number as a planning prior, not a guarantee, until validated on GIC's own walk-forward backtest harness (`src/models/backtesting.py`).

---

## How to read the tables

- **Technique** — named method (so the team can find the paper/library).
- **How it works** — one line.
- **Expected uplift** — accuracy *or* decision-impact, indicative.
- **Complexity** — Low / Med / High implementation+maintenance effort for GIC's stack (Python, Polars, XGBoost/SARIMAX, FastAPI, React).
- **Call** — **ADOPT** (do it), **EVALUATE** (spike/benchmark first), **SKIP** (not worth it for GIC now).

GIC's current baseline for context: SARIMAX+XGBoost ensemble with Hurst-regime adaptive weighting; best MAPE ~7% (Copper), worst ~31% (Natural Gas); 80% CI calibrated to 79% (parametric, now augmented with split-conformal); Monte Carlo VaR(95%) ≈ £705M.

---

## 1. Time-Series Forecasting

| Technique | How it works | Expected uplift | Complexity | Call |
|---|---|---|---|---|
| **N-BEATS** | Pure deep MLP with doubly-residual trend/seasonality basis stacks; no feature engineering. | 5–15% MAPE reduction vs ARIMA on univariate M-competition-style series; weaker with exogenous drivers. | Med | **EVALUATE** |
| **N-HiTS** | N-BEATS + multi-rate sampling & hierarchical interpolation; strong on long horizons, ~10× cheaper than transformers. | 10–25% long-horizon error reduction vs N-BEATS/transformers; fast to train. | Med | **EVALUATE** (best classical-DL candidate for GIC's 12–18m horizon) |
| **Temporal Fusion Transformer (TFT)** | LSTM encoder + variable-selection + interpretable multi-head attention; native multi-horizon **and** quantile outputs + built-in feature importance. | Competitive-to-best multi-horizon; gives quantiles + explainability "for free". Mixed evidence vs LSTM in some studies. | High | **EVALUATE** (its quantile + variable-selection story aligns with GIC's explainability goals) |
| **DeepAR** | Autoregressive RNN producing a full probabilistic (e.g. Student-t) predictive distribution per step. | Strong probabilistic forecasts on many related series; pairs well with GIC's 12-commodity panel. | Med | **EVALUATE** |
| **PatchTST** | Patches the series + channel-independent transformer; current SOTA among supervised transformers for long-horizon. | Reported SOTA MAE/RMSE/MAPE on several long-horizon benchmarks incl. commodity prices (e.g. potato-price study). | High | **EVALUATE** |
| **TimesFM / Chronos / Moirai (foundation models)** | Pretrained on 10^9–10^11 time points; **zero-shot** forecasts that often beat tuned statistical models out-of-the-box. Chronos tokenizes values; TimesFM is a decoder-only TS transformer; Moirai is any-variate/cross-frequency. | Zero-shot competitive with tuned baselines; **huge** time-to-value win (no per-commodity training). Chronos-2/TimesFM (Oct-2025) add covariate-informed forecasting. Caveat: TSFMs can be **mis-calibrated** out-of-the-box — wrap in conformal. | Low (inference) / Med (covariate fine-tune) | **ADOPT (EVALUATE→ADOPT)** — highest wow-per-effort; strong fit because GIC has only ~5y history per commodity |
| **TiDE** | Dense-MLP encoder-decoder with covariates; transformer-class accuracy at MLP speed/cost. | Matches transformers on long-horizon at far lower compute; good covariate handling (macro drivers). | Med | **EVALUATE** |
| **Prophet / NeuralProphet** | Additive trend+seasonality+holiday decomposition (NeuralProphet adds AR-net + neural terms). | Rarely SOTA on accuracy; wins on interpretability, fast prototyping, analyst-friendly components. | Low | **SKIP for accuracy / EVALUATE as an interpretable baseline** (GIC already has SARIMAX as the explainable baseline) |

**Forecasting verdict for GIC.** GIC's bottleneck is not the *family* of model on stable commodities (Copper already at 7% MAPE) — it is the **volatile tail** (Natural Gas 31%, Palladium 29%) and **time-to-value**. Foundation models (Chronos/TimesFM) directly attack both: zero-shot coverage of all 12 commodities with covariate support, no per-series training. Add N-HiTS/TiDE as the trainable challenger and benchmark all three against the existing ensemble in the walk-forward harness before promoting anything.

---

## 2. Uncertainty Quantification

| Technique | How it works | Expected uplift | Complexity | Call |
|---|---|---|---|---|
| **Split-conformal prediction** | Calibrate interval half-width on held-out residual quantiles; distribution-free marginal coverage regardless of model misspecification. | Fixes systematic under-coverage of parametric CIs (claimed 80% → realized ~70% under fat tails) → coverage *by construction*. | Low | **ADOPT — already shipped** (`src/models/conformal.py`) |
| **Adaptive Conformal Inference (ACI / online CP)** | Online-update the working α from recent miscoverage so coverage holds under distribution drift / regime shifts. | Defends the 80% target exactly when forecasters fail (regime breaks) — the highest-value moments. Recent families: ACI, EnbPI, Weighted-CP, Block-CP. | Low–Med | **ADOPT** (ACI implemented; wire it into live monthly reforecast loop) |
| **Quantile regression (incl. quantile XGBoost / LightGBM, conformalized quantile regression CQR)** | Directly fit conditional quantiles (pinball loss); CQR conformalizes them for guaranteed coverage. | Sharper, asymmetric intervals than symmetric conformal; honest tail risk for VaR/CVaR feed. | Med | **ADOPT** — natural next step; XGBoost already in stack, just add quantile objectives + CQR wrapper |
| **Bayesian structural time series (BSTS)** | State-space model with spike-and-slab regressor selection; full posterior over trend/seasonality/regressors. | Principled uncertainty + automatic driver selection + good for causal-impact "nowcasting"; interpretable components. | Med–High | **EVALUATE** (strong for driver attribution + scenario nowcasting; heavier to maintain) |

**Uncertainty verdict.** This is GIC's strongest existing differentiator post-iteration. Split-conformal + ACI are shipped; the clean win is to extend to **conformalized quantile regression (CQR)** so the Monte Carlo / VaR layer is fed empirically-calibrated *asymmetric* tails instead of a symmetric band — directly improving the £705M VaR honesty.

---

## 3. Regime Detection & Structural Breaks

| Technique | How it works | Expected uplift | Complexity | Call |
|---|---|---|---|---|
| **Hurst exponent (R/S)** | Long-memory statistic classifies trending vs mean-reverting vs volatile → adaptive ensemble weights. | 15–25% MAPE reduction during regime-shift periods vs fixed weights (per GIC's own module rationale). | Low | **ADOPT — already shipped** (`src/models/regime_detector.py`) |
| **BOCPD (Bayesian Online Change-Point Detection)** | Online posterior over "run length" since last break; flags structural breaks in real time. | Earlier break detection → triggers reforecast/rehedge *before* variance blows out; reduces forecast staleness. | Med | **ADOPT** (not yet a dedicated module — see caveat; high value for the live feed) |
| **CUSUM / Page-Hinkley** | Cumulative-sum of deviations crosses a threshold → drift/break alarm; cheap and robust. | Low-latency drift alarms feeding the early-warning score; complements BOCPD. | Low | **ADOPT** (lightweight; pairs with existing z-score/IQR anomaly detector) |
| **Markov regime-switching (Hamilton)** | Latent discrete states with state-dependent dynamics + transition probabilities (e.g. calm vs crisis vol). | Captures persistence of vol regimes; cleaner scenario probabilities than ad-hoc PMI/DXY weighting. | Med–High | **EVALUATE** (could replace the heuristic scenario-probability shifting in Method 4) |

**Regime verdict.** Hurst is in place. The honest gap is that GIC's "change-point detection" is currently **rolling z-score / IQR anomaly flags**, not true BOCPD/CUSUM. Adding a real **CUSUM (Low effort) + BOCPD (Med effort)** module that emits break events into the early-warning system and triggers automatic reforecast is the single highest-leverage regime upgrade.

---

## 4. Causality & Driver Attribution

| Technique | How it works | Expected uplift | Complexity | Call |
|---|---|---|---|---|
| **SHAP (TreeExplainer)** | Game-theoretic Shapley attribution of each forecast to its drivers; locally accurate & consistent. | Turns opaque XGBoost into ranked signed drivers for the narrative layer; trust + auditability. | Low | **ADOPT — already shipped** (`src/models/explainability_shap.py`) |
| **Double / Debiased ML (DML)** | Orthogonalize treatment & outcome with ML nuisance models → unbiased causal effect (e.g. true price elasticity) with valid CIs. | Converts GIC's *correlational* log-log elasticity into a *causal* estimate with confidence bounds — fixes the documented "causality not validated" gap. | Med | **ADOPT (EVALUATE→ADOPT)** — directly closes a known limitation |
| **Granger causality / Transfer Entropy** | Tests whether driver X improves prediction of Y beyond Y's own past (Transfer Entropy = nonlinear variant). | Validates which macro drivers genuinely lead each commodity; prunes spurious features. | Low–Med | **EVALUATE** |
| **DoWhy / causal-graph (DAG) modelling** | Encode assumptions as a DAG; identify + estimate + refute effects with explicit assumptions. | Makes elasticity & scenario assumptions explicit, testable, and refutable — board-credible causal story. | Med | **EVALUATE** |
| **Shapley driver attribution for P&L variance** | Apply Shapley value to the P&L bridge so variance is fairly allocated across drivers (vs sequential bridge order-dependence). | Removes order-dependence bias in the Volume→Price→Commodity→FX bridge; fairer attribution. | Med | **EVALUATE** (enhances the existing variance bridge) |

**Causality verdict.** SHAP is shipped on the forecast side. The flagship gap is **elasticity causality** — DML is the precise, literature-blessed fix and turns a caveat ("causality not validated") into a strength.

---

## 5. LLMs in Finance & Forecasting

| Technique | How it works | Expected uplift | Complexity | Call |
|---|---|---|---|---|
| **LLM-as-forecaster (Time-LLM, reprogramming)** | Reprogram a frozen LLM (patch→text-prototype) to forecast; or "slow-thinking" reasoning over series. | Promising on context-rich/few-shot series; **not yet** a reliable accuracy win vs PatchTST/TSFMs for pure numeric commodity forecasting. | High | **SKIP for point forecasting / EVALUATE for context-aided forecasting** |
| **Retrieval-augmented financial reasoning (RAG)** | Ground LLM answers in retrieved docs (filings, OEM reports, ICIS/Platts notes, internal memos) → cited, current reasoning. | High decision-impact: cited "why is Palladium moving?" narratives tied to real sources; reduces hallucination. | Med | **ADOPT (EVALUATE→ADOPT)** — upgrades template explainability to grounded narrative |
| **LLM agent for FP&A narrative + recommendation** | Agent reads forecasts/P&L/variance bridge → drafts board narrative + prioritized prescriptive actions. | Turns the 766-line report + insight cards into an interactive analyst; "last-mile" forecasting agents are an active 2025–26 SOTA thread. | Med | **ADOPT** — natural wrapper over the existing Insight Engine + variance bridge |
| **Function-calling for what-if** | LLM maps natural-language asks ("Lithium +25%, EU −8%, hedge 60%?") to the existing `/simulation/scenario` + shock endpoints. | Conversational scenario building → dramatically lowers analyst friction; uses APIs GIC already exposes. | Low–Med | **ADOPT** — low effort, high demo wow; the scenario/shock endpoints already exist |
| **News→signal fusion (agentic news extraction)** | LLM agent extracts economic-news shocks and fuses them with price signals (temporal+semantic fusion). | Improves volatile-commodity nowcasting (the NatGas/Palladium tail) where pure price models fail. | High | **EVALUATE** |

**LLM verdict.** GIC should resist *LLM-as-numeric-forecaster* (not yet worth it) and instead deploy LLMs where they dominate today: **function-calling what-if (Low effort)**, **RAG-grounded narrative (Med)**, and an **FP&A recommendation agent (Med)** sitting on top of the Insight Engine. Keep it open-source/governed to preserve the audit-trail story.

---

## 6. Commodity & Automotive Specifics

| Technique | How it works | Expected uplift | Complexity | Call |
|---|---|---|---|---|
| **Cross-commodity factor models** | Decompose the 12-commodity panel into latent factors (PGM basket, energy, battery-metals, FX) → shared-factor forecasting + correlated shocks. | Better correlated Monte Carlo shocks (today commodities shocked semi-independently) → more honest portfolio VaR; lifts illiquid synthetics (Rhodium ← PGM factor). | Med | **ADOPT (EVALUATE→ADOPT)** — improves both forecasting *and* risk realism |
| **Futures term-structure signals** | Use contango/backwardation slope, roll yield, and basis as predictive features / market-implied path. | Market-implied forward beats pure statistical model on liquid metals; GIC already extracts curves (Method 3) — turn slope into a *feature*, not just a path. | Low–Med | **ADOPT** — cheap, GIC already has the curve module |
| **Demand sensing** | High-frequency signals (orders, dealer inventory, web/search, registrations) nowcast near-term demand below the planning grain. | Sharpens the demand leg of the P&L + variance bridge; enables regional splits (a documented gap). | Med–High | **EVALUATE** (depends on data access; pairs with regional-demand backlog item) |
| **Warranty / reliability survival models (Weibull, Cox PH)** | Model time-to-failure hazard (Weibull shape/scale; Cox for covariates) → forward warranty cost + accrual adequacy. | Upgrades GIC's EWMA+seasonal warranty model to a true reliability curve; better accrual adequacy + failure-mode foresight. | Med | **ADOPT (EVALUATE→ADOPT)** — warranty module exists; add Weibull hazard layer |
| **Hedging via stochastic optimization / CVaR optimization** | Optimize hedge ratios to minimize CVaR (tail risk) under simulated price scenarios, not just variance/VaR. | Tail-aware hedging beats the current variance/VaR-blend optimizer; coherent risk measure (CVaR) is the standard. | Med | **ADOPT (EVALUATE→ADOPT)** — extends `hedge_optimizer.py` from VaR-blend to CVaR objective |

**Commodity/automotive verdict.** Two upgrades compound across the whole platform: a **cross-commodity factor model** (better correlated shocks → better VaR, and a real anchor for the 3 synthetic series) and **CVaR-objective hedging** (tail-aware, coherent). Both leverage modules GIC already has.

---

## Top 8 Wow-Factors to Adopt Next (Prioritized)

Ranked by **(decision-impact × fit) ÷ effort**, biased toward items that extend modules already in the codebase.

| # | Wow-factor | Why it wins for GIC | Effort | Indicative impact | Status |
|---|---|---|---|---|---|
| 1 | **Foundation-model forecaster (Chronos / TimesFM) as zero-shot challenger** | Zero-shot covers all 12 commodities w/ only ~5y history; attacks the volatile tail + time-to-value; wrap in existing conformal layer for calibration. | Low–Med | Time-to-value step-change; potential 5–20% MAPE cut on weak commodities | New |
| 2 | **CUSUM + BOCPD change-point module → auto-reforecast trigger** | Real structural-break detection (today only z-score/IQR) feeding early-warning + live feed; rehedge before variance blows out. | Med | Earlier action on regime breaks; lower forecast staleness | New |
| 3 | **Conformalized Quantile Regression (CQR) feeding VaR/CVaR** | Empirically-calibrated *asymmetric* tails into the £705M VaR engine; sharper than symmetric conformal. | Med | Honest, sharper tail risk; better-calibrated EBIT CI | Extends conformal |
| 4 | **Double ML for causal price elasticity** | Converts correlational log-log elasticity into a *causal* estimate w/ CIs — closes the documented "causality not validated" gap. | Med | Board-credible elasticity; defensible scenario assumptions | Closes known gap |
| 5 | **LLM function-calling what-if + RAG-grounded narrative** | Conversational scenario building over existing `/simulation` + shock APIs; cited narratives upgrade template explainability. | Low–Med | Major UX/decision-speed + trust uplift | Wraps Insight Engine |
| 6 | **Cross-commodity factor model for correlated shocks** | Correlated Monte Carlo shocks → more honest portfolio VaR; gives the 3 synthetic series a real PGM/energy anchor. | Med | More realistic VaR/CVaR; better illiquid-commodity forecasts | New |
| 7 | **CVaR-objective hedge optimization** | Tail-aware, coherent-risk hedging vs current variance/VaR blend; extends existing optimizer. | Med | Better worst-case hedge economics on £-large exposures | Extends hedge optimizer |
| 8 | **Weibull/Cox warranty survival layer** | True reliability hazard vs EWMA+seasonal; sharper accrual adequacy + failure-mode foresight. | Med | More accurate warranty accrual + early failure-mode signal | Extends warranty model |

**Sequencing.** Items 1–3 are the "Now" wave (calibration + accuracy + break-detection are the analytical core). Items 4–6 are "Next" (causality, conversational UX, correlated risk). Items 7–8 are "Later" polish that deepen domain credibility.

---

## Sources

- [The 2026 Time Series Toolkit: 5 Foundation Models for Autonomous Forecasting (MachineLearningMastery)](https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/)
- [Time-Series Foundation Models in Finance: Pretraining Corpora, Architectures, Benchmarks, Risk-Aware Evaluation (ACM, 2025)](https://dl.acm.org/doi/full/10.1145/3785706.3785728)
- [Beyond Accuracy: Are Time Series Foundation Models Well-Calibrated? (arXiv:2510.16060)](https://arxiv.org/pdf/2510.16060)
- [The Promise of Time-Series Foundation Models for Agricultural Forecasting: Evidence from Commodity Prices (arXiv:2601.06371)](https://arxiv.org/pdf/2601.06371)
- [Foundation models for time series forecasting: Application in conformal prediction (arXiv:2507.08858)](https://arxiv.org/html/2507.08858v1)
- [Conformal Prediction Algorithms for Time Series Forecasting: Methods and Benchmarking (arXiv:2601.18509)](https://arxiv.org/html/2601.18509v2)
- [A Gentle Introduction to Conformal Time Series Forecasting (arXiv:2511.13608)](https://arxiv.org/html/2511.13608v1)
- [Online conformal inference for multi-step time series (Monash WP20-2024)](https://www.monash.edu/business/ebs/research/publications/ebs/2024/wp20-2024.pdf)
- [Temporal Fusion Transformers for interpretable multi-horizon forecasting (Lim et al.)](https://www.researchgate.net/publication/352448180_Temporal_Fusion_Transformers_for_interpretable_multi-horizon_time_series_forecasting)
- [PatchTST-DME: Frequency-Aware Transformer for Potato Price Forecasting (Springer, 2025)](https://link.springer.com/article/10.1007/s11540-025-09907-4)
- [Forecasting Commodity Price Shocks Using Temporal and Semantic Fusion of Price Signals and Agentic GenAI News (arXiv:2508.06497)](https://arxiv.org/html/2508.06497v1)
- [Bridging the Last Mile of Time Series Forecasting with LLM Agents (arXiv:2606.02497)](https://arxiv.org/html/2606.02497)
- [Augur: Modeling Covariate Causal Associations in Time Series via LLMs (arXiv:2510.07858)](https://arxiv.org/pdf/2510.07858)
- Angelopoulos & Bates (2021), *A Gentle Introduction to Conformal Prediction* (arXiv:2107.07511); Gibbs & Candès (2021), *Adaptive Conformal Inference Under Distribution Shift* (NeurIPS); Lundberg & Lee (2017), *SHAP* (NeurIPS) — per in-repo module references.
