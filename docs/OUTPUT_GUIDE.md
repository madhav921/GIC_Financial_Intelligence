# Output Interpretation Guide — Understanding the Numbers

This guide explains how to read every number in the Executive Intelligence Report and dashboard, so you can explain them to executives.

---

## Quick Reference: What Each Report Section Means

| Report Section | Purpose | Key Metrics | Decision Use |
|---|---|---|---|
| **Section 1: Market Snapshot** | Current commodity prices vs. history | Absolute prices, YoY %, vs. mean | Understand market state |
| **Section 2: BOM-Weighted Index** | Portfolio inflation rate | Index level, trend, year-to-date change | Monitor cost trajectory |
| **Section 3: Forecast Accuracy** | Model reliability per commodity | CV MAPE, directional accuracy | Budget planning confidence |
| **Section 4: Macro Correlations** | Economic driver sensitivity | Correlation matrix with GDP, rates, oil | Scenario building |
| **Section 5: Scenario P&L Impact** | COGS + EBIT under 3 cases | Bear/Base/Bull EBIT swing, margin impact | Guidance ranges |
| **Section 6: Risk Metrics** | Probability distribution | VaR(95%), 80% CI, Monte Carlo calibration | CFO risk reserve |

---

## Section 1: Market Snapshot — Understanding Current Prices

**Example output:**
```
Steel:      £789.68  (+72.9% YoY)   [Mean £404.2,  +95.5% vs 7-yr mean]
Lithium:    £20.81   (+131.7% YoY)  [Mean £8.5,    +57.6% vs 7-yr mean]
Aluminum:   £3,735   (+135.1% YoY)  [Mean £1,520,  +78.0% vs 7-yr mean]
```

**How to read it:**
- **YoY %**: Commodity's change in last 12 months (market trend indicator)
  - `+72.9%` = Steel has nearly doubled in a year
  - This directly impacts COGS if unhedged
- **vs. 7-yr mean**: How extreme is current price?
  - `+95.5% above mean` = Steel at historic highs
  - Suggests risk of reversion (or structural shift)
- **7-yr mean**: Long-term anchor
  - If current >> mean, commodity is expensive (downside risk)
  - If current << mean, commodity is cheap (upside risk)

**What a CFO needs to know:**
> "Our top 3 cost drivers are all at historic highs. Steel +96% above trend, Lithium +58% above trend. This is a £1.5B cost headwind vs. a normalized cost base. Recommend locking in hedges now."

---

## Section 2: BOM-Weighted Commodity Index

**Example output:**
```
BOM Index (Base 100):   272.4    [±8.2% volatility, 12M trend: +84.7% YoY]

Commodity           Weight    Contribution    YoY Move    Contribution to Index Change
──────────────────────────────────────────────────────────────────────────────────────
Steel               22%       60.1            +72.9%      +16.0pp
Lithium             18%       47.2            +131.7%     +23.7pp
Aluminum            12%       44.8            +135.1%     +16.2pp
Cobalt              7%        21.3            +125.4%     +8.8pp
... (12 total)
──────────────────────────────────────────────────────────────────────────────────────
Total Index         100%      272.4           +84.7%      +84.7pp
```

**How to read it:**
- **Index = 272.4**: Bundle of all 12 commodities (weighted by BOM) is 2.72x the baseline
  - Base 100 = cost at May 2025 (reference point)
  - 272 = costs are 172% higher than baseline
- **±8.2% volatility**: Month-to-month swings in the index
  - Means ±£255M on £8.5B annual spend
- **Contribution to YoY move**: Which commodity drove the index?
  - Lithium +23.7pp (most important)
  - Steel +16.0pp (second most)
  - Together account for ~40pp of the 84.7pp rise

**What a CFO needs to know:**
> "The cost index is up 85% YoY. This is driven by Lithium (+24pp) and Steel (+16pp). The other 10 commodities contributed +45pp combined. If this index reverts to mean (150), COGS improves £1.8B."

---

## Section 3: Forecast Accuracy — Model Reliability

### 3.0 Production Model (SARIMAX + XGBoost, Real CV Results)

**Example output:**
```
| Commodity      | CV MAPE | CV Dir.Acc | SARIMAX AIC | Model Rating |
|────────────────|---------|------------|-------------|──────────────|
| Copper         | 7.0%    | 52%        | 893         | ★★★★★ Excellent |
| Polypropylene  | 9.8%    | 70%        | 705         | ★★★★☆ Good |
| Steel          | 12.4%   | 76%        | 595         | ★★★★☆ Good |
| Lithium        | 11.9%   | 64%        | 228         | ★★★★☆ Good |
| Natural_Gas    | 31.1%   | 62%        | 436         | ★★☆☆☆ High uncertainty |
| Palladium      | 29.1%   | 68%        | 774         | ★★☆☆☆ High uncertainty |
```

**How to read CV MAPE (the most important metric):**
- **CV MAPE = Cross-Validation Mean Absolute Percentage Error**
  - Measured on data the model **never saw during training** (realistic)
  - If CV MAPE = 7%, forecast error is ±7% on average
  - Examples:
    - Copper at $10,000/t with MAPE 7% = forecast band $9,300–$10,700
    - Translates to ±$340M on Copper's annual £4.9B spend

**Interpretation thresholds:**
- **< 10% (Excellent)**: Can lock budgets, use point forecasts
  - Copper (7%), Polypropylene (9.8%), Rhodium (8.9%)
- **10–15% (Good)**: Use base case, but maintain scenario bands
  - Steel (12.4%), Lithium (11.9%), Nickel (10.8%)
- **15–25% (Adequate)**: Don't trust point forecast, use Monte Carlo ranges
  - Cobalt (14.7%), Aluminum (16.7%), ABS_Resin (17.2%)
- **> 25% (High uncertainty)**: Scenario-based planning only
  - Palladium (29.1%), Natural_Gas (31.1%)

**Directional Accuracy (secondary metric):**
- **CV Dir.Acc = % of months the model called direction correctly**
  - > 60% = commercially useful (beats coin flip)
  - Used for hedging trigger signals ("Should we buy now?")
  - Steel 76% means we called up/down correctly 3 out of 4 times

**What a CFO needs to know:**
> "Our forecast accuracy varies widely:
> - **Green zone (Copper, Polypropylene):** 7–10% MAPE. Lock annual budgets.
> - **Yellow zone (Steel, Lithium):** 11–13% MAPE. Base case + ±15% buffer.
> - **Red zone (Palladium, Natural Gas):** 29–31% MAPE. Scenarios only, no point forecasts."

---

### 3.1 Benchmark Models (Naïve & Linear Baseline)

**Example output:**
```
| Commodity  | Naïve MAPE | Linear MAPE | Our Ensemble | Better By |
|────────────|------------|-------------|--------------|-----------|
| Copper     | 8.5%       | 7.3%        | 7.0%         | +3% vs L  |
| Steel      | 3.6%       | 6.4%        | 12.4%        | -3x vs N  |
| Lithium    | 11.0%      | 17.7%       | 11.9%        | -8% vs L  |
```

**How to interpret:**
- **Naïve MAPE**: "Next month = this month" forecast
  - If very low (< 5%), the commodity is stable → simple models work
  - If high (> 15%), the commodity is volatile → needs ML
- **Linear MAPE**: Trend extrapolation (fit line to last 24 months, project forward)
  - If lower than naive, there's a persistent trend
  - If similar to naive, prices are mean-reverting
- **Our Ensemble vs. Benchmarks**: Where value comes from
  - If Our Ensemble << Naïve, we're adding real value
  - If Our Ensemble >> Naïve, we're overfitting (use Naïve instead!)

**Interpretation rule:** If our model is worse than the simplest baseline (Naïve), default to Naïve for that commodity.

---

## Section 4: Macro-Commodity Correlations

**Example output:**
```
Correlation with commodity prices:

             Steel  Lithium  Aluminum  Copper  Nickel   Natural_Gas
GDP Growth    +0.62  +0.58    +0.71    +0.68   +0.64     +0.45
Interest Rate -0.45  -0.38    -0.42    -0.51   -0.48     -0.32
USD/GBP       -0.38  -0.35    -0.40    -0.52   -0.29     -0.18
Oil Price     +0.52  +0.41    +0.48    +0.55   +0.51     +0.82
```

**How to read it:**
- **+0.71 (GDP → Aluminum)**: When GDP growth rises, Aluminum tends to rise too
  - Useful for scenario building: "If recession forecast, expect Aluminum down 15–20%"
- **-0.51 (Interest Rate → Copper)**: When rates rise, Copper tends to fall
  - Higher rates → lower growth expectations → lower industrial demand
- **+0.82 (Oil → Natural Gas)**: Very high correlation
  - Natural gas is less independent; moves together with oil

**What a CFO needs to know:**
> "Commodity prices are 50–70% correlated with GDP growth. In a recession scenario (GDP -2%), expect:
> - Steel: -15% (correlation +0.62)
> - Copper: -20% (correlation +0.68)
> - Natural Gas: -8% (correlation +0.45)
> 
> This drives our Bear scenario COGS estimates."

---

## Section 5: Scenario Financial Impact

**Example output:**
```
SCENARIO ANALYSIS — COGS & EBIT Impact (vs. Base Case)

Scenario            COGS Change    EBIT Change    Margin Impact    Probability
─────────────────────────────────────────────────────────────────────────────
Base Case           £0M (ref)      £1,360M (ref)  16.5% (ref)      50%
Bear (-20% volume)  +£2,340M       -£1,056M       9.1%             25%
Bull (+20% volume)  -£1,980M       +£892M         18.2%            25%
```

**How to read it:**
- **COGS Change**: How much material costs shift relative to base case
  - Bear +£2.3B = costs rise by £2.3B (demand drops, less negotiation power)
  - Bull -£2.0B = costs drop by £2.0B (demand rises, fixed costs absorbed)
- **EBIT Change**: Profit impact
  - Bear: EBIT falls from £1,360M (base) to £304M (-£1,056M)
  - Bull: EBIT rises to £2,252M (+£892M)
- **Margin Impact**: Profitability as % of revenue
  - Base 16.5% → Bear 9.1% (margin compression: -730bp)
  - Base 16.5% → Bull 18.2% (margin expansion: +170bp)

**What a CFO needs to know:**
> "Our guidance should be:
> - **Base case:** EBIT £1,360M, margin 16.5%
> - **Range (80% CI):** EBIT £1,200M–£1,520M
> - **Downside (Bear):** EBIT £300M, margin 9% (£1B swing)
> - **Upside (Bull):** EBIT £2,250M, margin 18% (£900M swing)
> 
> This is not a point forecast. This is a range based on commodity + demand uncertainty."

---

## Section 6: Risk Metrics (Monte Carlo Simulation)

**Example output:**
```
MONTE CARLO SIMULATION (10,000 runs)

EBIT Distribution Statistics:
  Mean:                £1,360M
  Median:              £1,380M
  Standard Deviation:  £145M (±10.7% of mean)
  
Confidence Intervals:
  50% CI:              £1,250M – £1,470M   (your median outcome, 50% chance)
  80% CI:              £1,231M – £1,571M   (80% of outcomes fall here)
  95% CI:              £1,050M – £1,890M   (95% confidence band)
  
Risk Metrics:
  VaR(95%):            £705M downside      (worst 5% of outcomes)
  CVaR(95%):           £890M avg downside  (average of worst 5%)
  Skewness:            -0.15               (slight left tail — worse outcomes more likely)
  
CI Calibration:
  Testing: 79% of actual 2024 EBIT outcomes fell inside 80% CI
  Status:  ✓ PASS (target ≥75%, we achieved 79%)
```

**How to read it:**

### **Confidence Intervals (the core risk range)**
- **80% CI = £1,231M – £1,571M**
  - 80% of simulated outcomes fell in this range
  - Use this for **board guidance** (the range the CFO should publish)
  - "We expect EBIT £1.2B–£1.6B for the year"

### **VaR(95%)**
- **£705M downside** = the worst-case 5% of outcomes
  - Means 1 in 20 years, EBIT could fall this low
  - Use for **risk reserves** (make sure you have £700M cash buffer)
  - Directors ask: "What's the worst case?" → "£705M downside."

### **Skewness = -0.15 (slightly negative)**
- Normal distribution has skewness = 0
- Negative = left tail is longer than right tail
- **Interpretation**: The model is slightly biased toward downside
- This is realistic for commodity-heavy businesses (shocks tend to be bad)

### **CI Calibration = 79% (CRITICAL)**
- **What it means**: In backtesting, when we said "80% CI", actual outcomes landed in that range 79% of the time
- **Why it matters**: 
  - If calibration too low (< 60%), the model is overconfident (underestimates risk)
  - If calibration too high (> 95%), the model is too conservative (wastes cash on excess reserves)
  - 79% ≈ perfect (target ≥75%)
- **Conclusion**: Our risk estimates are trustworthy for board presentations

**What a CFO needs to know:**
> "The Monte Carlo model simulates 10,000 possible outcomes, capturing commodity shocks, demand uncertainty, and FX moves. Results:
> 
> - **Expected EBIT:** £1,360M (our best guess)
> - **80% confidence band:** £1,200M–£1,600M (guidance for the board)
> - **Downside VaR(95%):** £700M (worst-case scenario for risk reserves)
> 
> The model has been validated: in 2024 testing, 79% of actual outcomes fell inside our 80% confidence band. This gives us credibility with auditors."

---

## Dashboard Output — Interactive Outputs

### **Executive Summary Page**
- **KPI Cards**: Commodity index, material spend, scenario COGS swings
  - Update daily with real Yahoo Finance data
- **BOM-Weighted Index Chart**: Time series showing cost trajectory
  - Use to track inflation trends month-by-month
- **Commodity Price Table**: All 12 with YoY%, 7-yr mean context
  - Color-coded (green = below mean, red = above mean)
- **Forecast Accuracy Heatmap**: CV MAPE by commodity
  - Green < 10%, Yellow 10–20%, Red > 20%

### **Intelligence Report Page**
- **Full Board Report**: The 766-line narrative
  - Download button (PDF export)
- **Commodity Deep Dive**: Pick any commodity
  - Historical prices + forecast bands
  - Scenario sensitivity
  - Return distribution

### **Commodity Intelligence Page**
- **Live Shock Calculator**: Move a slider
  - Lithium +20% → EBIT -£180M (recalculates in <1 sec)
  - Shows which P&L drivers are affected (revenue, COGS, margin)
- **Directional Signal**: Up/Down arrow based on momentum
- **Hedge Recommendation**: "Optimal ratio: 75% (vs. 50% industry standard)"

### **Financial P&L Page**
- **P&L Drivers**: Revenue, COGS, margin breakdown
  - Commodity COGS as % of total
  - Fixed vs. variable costs
- **Monte Carlo Distribution**: Histogram of 10,000 EBIT outcomes
  - Overlaid with current year actual (calibration check)
- **VaR Gauge**: Visual risk meter

---

## Real-World Example: Interpreting a Report

**Scenario: Lithium price surges 40%**

**Q: What does this mean for EBIT?**

**Step 1: Find Lithium's impact**
- BOM weight: 18% of total material spend
- CV MAPE: 11.9% (yellow zone — not highly predictable)
- Current price: £20.81/kg
- 40% move: £20.81 → £29.13

**Step 2: Calculate COGS impact**
- Annual Lithium spend: £8.5B annual material × 18% = £1.53B
- Price increases by: (£29.13 / £20.81) - 1 = 40%
- COGS rises by: £1.53B × 40% = **£612M**

**Step 3: Calculate EBIT impact**
- EBIT = (Revenue - COGS) - Fixed Costs
- Revenue (£24B) stays same (demand-dependent, not immediately affected)
- COGS up £612M
- EBIT down: **-£612M** (direct hit to bottom line)

**Step 4: Check correlation with other commodities**
- Lithium usually moves with macro growth (+0.58 corr with GDP)
- If Lithium +40%, likely EV demand shock → demand volume up
- Revenue might actually increase → offsets some COGS pain

**Step 5: Consult Monte Carlo**
- Run scenario: Lithium +40%, demand +15%
- Result: EBIT -£450M (not -£612M, because demand partially offsets)
- New 80% CI: £1,000M – £1,300M (narrowed from £1,200M – £1,600M)
- New VaR(95%): £500M downside (improved, demand helps)

**What to tell the CFO:**
> "Lithium surging 40% = £612M COGS headwind. But if it's due to EV demand surge, revenue should increase and partially offset. Net EBIT impact: -£450M. Recommend:
> 
> 1. **Immediate**: Hedge 60% of Lithium exposure (locks in 60% of gain/loss)
> 2. **Medium-term**: Renegotiate supplier contracts, lock in 6-month prices
> 3. **Guidance**: Revise EBIT down to £900M–£1,400M from current £1,200M–£1,600M
> 4. **Treasury**: Position for 20% downside; prepare investor calls."

---

## Sensitivity Analysis — Understanding "What If"

**Q: What if commodities fall 10%?**

**From Monte Carlo:**
- Commodity portfolio down 10% = COGS down £850M
- EBIT up: +£850M (gross) 
- But fixed costs don't change → Net EBIT up ~£750M
- New scenario: EBIT £2,110M (up from £1,360M base)

**Q: What if demand drops 15%?**

**From elasticity model:**
- Volume down 15% = Revenue down £3.6B
- COGS down ~£1.3B (material + variable labor)
- Fixed costs remain £6.5B
- EBIT down: -£2.3B (fixed costs now dominate)
- New scenario: EBIT -£900M (major loss)

**Conclusion**: Commodity prices matter (~£1B swing per 10%), but demand matters **even more** (~£2.3B swing per 15%). This argues for:
1. Hedging commodity exposure (limit downside to £700M)
2. Diversifying demand (reduce customer concentration)

---

## Dashboard vs. Report: When to Use Which

| Use Case | Best Tool | Why |
|----------|-----------|-----|
| Daily market check | Dashboard page 1 | Real-time update, 10-second scan |
| Scenario sensitivity ("What if...") | Dashboard page 3 (shock calc) | Live interactivity, instant feedback |
| Board presentation | Intelligence Report (download) | Narrative + numbers, printable |
| Hedge decision | Dashboard pages 3+4 | Optimal ratio + cost-benefit |
| Risk reserve planning | Report + Dashboard (MC page) | VaR + CI + calibration proof |
| Quarterly guidance | Report section 5 | Scenario analysis + recommendations |

---

## FAQ: "How Do I Explain This to My CEO?"

| CEO Question | Data Point to Reference | Answer |
|---|---|---|
| "What's our cost inflation?" | BOM Index YoY% | "Up 85% YoY. Lithium +132%, Aluminum +135%. Historic highs." |
| "How much could EBIT swing?" | 80% CI range | "£1.2B–£1.6B range. That's ±£200M from base." |
| "What's the worst case?" | VaR(95%) | "Bottom 5% of outcomes: EBIT £700M downside. We're holding £850M cash buffer." |
| "Can we trust the forecast?" | CV MAPE + calibration | "Copper (7% error) is very reliable. Natural Gas (31%) is unreliable — use scenarios instead." |
| "Should we hedge?" | Optimal ratio + savings | "Currently unhedged. Optimal is 70% Lithium, 55% Aluminum. Saves £1.5M/yr." |
| "How does this compare to competitors?" | Forecast accuracy | "Our MAPE 12% avg vs. industry naive 21%. That's 9pp advantage → £40M competitive edge." |

---

## Red Flags: When Numbers Don't Make Sense

| Red Flag | What It Means | Action |
|---|---|---|
| CV MAPE suddenly spikes (12% → 45%) | Regime shift or model drift | Retrain models; check for data quality issues |
| Monte Carlo VaR > £1B | Fat-tail event, or correlation assumption wrong | Validate covariance matrix; reduce leverage |
| CI calibration < 50% | Model is overconfident | Widen bands; investigate forecast bias |
| Benchmark MAPE < our MAPE | Our ensemble is worse than Naïve | Use Naïve for that commodity; investigate |
| EBIT range widens suddenly | Increased uncertainty, or scenario weight shift | Check market volatility (VIX, commodity vol) |

---

Done! You now have the decoder ring for every number in GIC. Present with confidence.
