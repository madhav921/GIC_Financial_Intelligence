"""
Executive Intelligence Report — Interactive dashboard view.

Renders the full executive intelligence report (docs/EXECUTIVE_INTELLIGENCE_REPORT.md)
with live data refresh, interactive charts, and downloadable export.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]

BOM_WEIGHTS = {
    "Steel": 0.22, "Aluminum": 0.12, "Lithium": 0.18, "Cobalt": 0.07,
    "Copper": 0.06, "Nickel": 0.05, "Platinum": 0.04, "Natural_Gas": 0.04,
    "Palladium": 0.03, "Polypropylene": 0.03, "ABS_Resin": 0.02, "Rhodium": 0.02,
}
SCENARIO_CONFIG = {
    "Steel":         {"bear": 380,   "base": 510,   "bull": 640},
    "Aluminum":      {"bear": 1900,  "base": 2500,  "bull": 3100},
    "Copper":        {"bear": 8200,  "base": 10500, "bull": 12800},
    "Lithium":       {"bear": 7,     "base": 12,    "bull": 20},
    "Cobalt":        {"bear": 18000, "base": 28000, "bull": 40000},
    "Nickel":        {"bear": 13000, "base": 18000, "bull": 24000},
    "Palladium":     {"bear": 700,   "base": 1100,  "bull": 1500},
    "Natural_Gas":   {"bear": 20,    "base": 38,    "bull": 60},
    "Platinum":      {"bear": 700,   "base": 1050,  "bull": 1400},
    "Rhodium":       {"bear": 3000,  "base": 5000,  "bull": 7500},
    "Polypropylene": {"bear": 900,   "base": 1300,  "bull": 1600},
    "ABS_Resin":     {"bear": 1100,  "base": 1600,  "bull": 2000},
}
ANNUAL_MATERIAL_SPEND = 8_500_000_000
ANNUAL_REVENUE = 24_000_000_000


@st.cache_data(ttl=600)
def _load_commodity_data():
    path = _ROOT / "data" / "raw" / "commodity_prices.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    return df


@st.cache_data(ttl=600)
def _load_macro_data():
    path = _ROOT / "data" / "raw" / "macro_indicators.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date")


def _commodity_index(comm: pd.DataFrame) -> pd.Series:
    cols = [c for c in comm.columns if c != "date"]
    weights = np.array([BOM_WEIGHTS.get(c, 0) for c in cols])
    weights = weights / weights.sum()
    prices = comm[cols].values
    base = prices[0]
    norm = prices / (base + 1e-9) * 100
    index = (norm * weights).sum(axis=1)
    return pd.Series(index, index=comm["date"], name="commodity_index")


def _pct(a, b):
    return (a - b) / abs(b) * 100 if b != 0 else 0.0


def render():
    st.title("📋 Executive Intelligence Report")
    st.markdown(
        "**Classification:** CONFIDENTIAL — Board / C-Suite / Supply Chain Leadership  \n"
        "Powered by real market data (Yahoo Finance + FRED). Refreshes every 10 minutes."
    )

    comm = _load_commodity_data()
    macro = _load_macro_data()
    if comm is None:
        st.error("No commodity data found. Run `python scripts/fetch_data.py` first.")
        return

    commodity_cols = [c for c in comm.columns if c != "date"]
    latest_date = comm["date"].max().strftime("%B %Y")

    # ── Refresh / Regenerate controls ─────────────────────────────────
    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        st.info(f"**Data current as of:** {latest_date}  |  **{len(comm)} months** of observations")
    with col_r2:
        if st.button("🔄 Regenerate Full Report"):
            with st.spinner("Regenerating executive report..."):
                result = subprocess.run(
                    [sys.executable, str(_ROOT / "scripts" / "generate_executive_report.py")],
                    capture_output=True, text=True, cwd=str(_ROOT)
                )
                if result.returncode == 0:
                    st.success("Report regenerated!")
                    st.cache_data.clear()
                else:
                    st.error(f"Error: {result.stderr[-500:]}")

    st.divider()

    # ── Section 1: Market Snapshot KPIs ───────────────────────────────
    st.header("1. Commodity Market Snapshot")

    index_series = _commodity_index(comm)
    idx_now  = float(index_series.iloc[-1])
    idx_prev = float(index_series.iloc[-13]) if len(index_series) >= 13 else float(index_series.iloc[0])
    idx_5yr  = float(index_series.iloc[-61]) if len(index_series) >= 61 else float(index_series.iloc[0])
    idx_chg  = _pct(idx_now, idx_prev)

    # Scenario totals
    scen_base_total = sum(
        ANNUAL_MATERIAL_SPEND * BOM_WEIGHTS.get(c, 0) *
        _pct(SCENARIO_CONFIG[c]["base"], float(comm[c].dropna().iloc[-1])) / 100 / 1e6
        for c in commodity_cols if c in SCENARIO_CONFIG
    )
    scen_bear_total = sum(
        ANNUAL_MATERIAL_SPEND * BOM_WEIGHTS.get(c, 0) *
        _pct(SCENARIO_CONFIG[c]["bear"], float(comm[c].dropna().iloc[-1])) / 100 / 1e6
        for c in commodity_cols if c in SCENARIO_CONFIG
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BOM Commodity Index", f"{idx_now:.1f}", f"{idx_chg:+.1f}% YoY",
              delta_color="inverse")
    c2.metric("Annual Material Spend", "$8.5B", "45% of COGS", delta_color="off")
    c3.metric("Base-Case COGS Δ (12M)", f"${scen_base_total:+.0f}M",
              f"{scen_base_total / ANNUAL_REVENUE * 100:+.2f}% EBIT", delta_color="inverse")
    c4.metric("Bear-Case COGS Δ (12M)", f"${scen_bear_total:+.0f}M",
              f"{scen_bear_total / ANNUAL_REVENUE * 100:+.2f}% EBIT", delta_color="inverse")

    # ── Commodity Index Chart ──────────────────────────────────────────
    st.subheader("BOM-Weighted Commodity Cost Index (Base = 100, Jun 2019)")
    idx_df = index_series.reset_index()
    idx_df.columns = ["date", "index"]

    fig_idx = go.Figure()
    fig_idx.add_trace(go.Scatter(
        x=idx_df["date"], y=idx_df["index"],
        mode="lines", name="Commodity Index",
        line=dict(color="#00d4aa", width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.08)"
    ))
    fig_idx.add_hline(y=100, line_dash="dot", line_color="gray", annotation_text="Baseline (Jun 2019)")
    fig_idx.add_hline(y=float(index_series.mean()), line_dash="dash", line_color="orange",
                      annotation_text=f"7yr Mean ({index_series.mean():.1f})")
    fig_idx.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"), xaxis=dict(gridcolor="#2a2a3e"),
        yaxis=dict(gridcolor="#2a2a3e", title="Index Level"),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_idx, use_container_width=True)

    st.divider()

    # ── Section 2: Full Commodity Price Table ──────────────────────────
    st.header("2. Full Commodity Price Dashboard — Real Data")

    rows = []
    for col in commodity_cols:
        series = comm.set_index("date")[col].dropna()
        if series.empty:
            continue
        current = float(series.iloc[-1])
        prev1m  = float(series.iloc[-2]) if len(series) >= 2 else current
        prev3m  = float(series.iloc[-4]) if len(series) >= 4 else current
        prev12m = float(series.iloc[-13]) if len(series) >= 13 else float(series.iloc[0])
        mean7   = float(series.mean())
        std7    = float(series.std())
        vol     = float(series.pct_change().std() * np.sqrt(12) * 100)
        z       = (current - mean7) / (std7 + 1e-9)

        if z > 1.5:   valuation = "Expensive"
        elif z > 0.5: valuation = "Above Avg"
        elif z < -1.5: valuation = "Cheap"
        elif z < -0.5: valuation = "Below Avg"
        else:          valuation = "Fair Value"

        ma6  = float(series.tail(6).mean())
        ma12 = float(series.tail(12).mean())
        if current > ma6 > ma12:   trend = "↑↑ Strong Up"
        elif current > ma12:       trend = "↑ Uptrend"
        elif current < ma6 < ma12: trend = "↓↓ Strong Down"
        elif current < ma12:       trend = "↓ Downtrend"
        else:                      trend = "→ Sideways"

        rows.append({
            "Commodity": col,
            "Current": round(current, 1),
            "BOM %": f"{BOM_WEIGHTS.get(col, 0)*100:.0f}%",
            "1M Chg": f"{_pct(current, prev1m):+.1f}%",
            "3M Chg": f"{_pct(current, prev3m):+.1f}%",
            "12M Chg": f"{_pct(current, prev12m):+.1f}%",
            "7yr Mean": round(mean7, 1),
            "vs Mean": f"{_pct(current, mean7):+.1f}%",
            "Ann Vol": f"{vol:.1f}%",
            "Regime": trend,
            "Valuation": valuation,
        })

    table_df = pd.DataFrame(rows)

    def _color_chg(val):
        if isinstance(val, str) and val.endswith("%"):
            try:
                v = float(val.replace("%", "").replace("+", ""))
                if v > 5:  return "color: #ff6b6b; font-weight: bold"
                if v > 1:  return "color: #ffab40"
                if v < -5: return "color: #00d4aa; font-weight: bold"
                if v < -1: return "color: #80deea"
            except: pass
        return ""

    st.dataframe(
        table_df.style.applymap(_color_chg, subset=["1M Chg", "3M Chg", "12M Chg", "vs Mean"]),
        use_container_width=True, height=460
    )

    # ── Section 3: Forecast Accuracy ──────────────────────────────────
    st.header("3. Forecast Accuracy Assessment")
    st.markdown(
        "Walk-forward validation against the last 12 months of **actual observed prices**. "
        "Measures how well simple benchmark models perform — the SARIMAX+XGBoost ensemble "
        "is designed to beat both benchmarks."
    )

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.linear_model import LinearRegression

    acc_rows = []
    for col in commodity_cols:
        series = comm.set_index("date")[col].dropna()
        n_test = 12
        if len(series) < n_test + 12:
            continue
        train = series.iloc[:-n_test]
        test  = series.iloc[-n_test:]
        preds_naive = test.shift(1).fillna(train.iloc[-1])

        n_fit = min(24, len(train))
        X_tr = np.arange(n_fit).reshape(-1, 1)
        y_tr = train.values[-n_fit:]
        lr = LinearRegression().fit(X_tr, y_tr)
        X_te = np.arange(n_fit, n_fit + n_test).reshape(-1, 1)
        preds_lr = lr.predict(X_te)

        actual_dir = np.sign(test.values[1:] - test.values[:-1])
        naive_dir  = np.sign(preds_naive.values[1:] - preds_naive.values[:-1])
        lr_dir     = np.sign(preds_lr[1:] - preds_lr[:-1])

        naive_mape = float(np.mean(np.abs((test.values - preds_naive.values) / (np.abs(test.values) + 1e-9))) * 100)
        lr_mape    = float(np.mean(np.abs((test.values - preds_lr) / (np.abs(test.values) + 1e-9))) * 100)
        naive_da   = float(np.mean(naive_dir == actual_dir) * 100)
        lr_da      = float(np.mean(lr_dir == actual_dir) * 100)
        error_12m  = float((test.values[-1] - preds_lr[-1]) / (test.values[-1] + 1e-9) * 100)

        if lr_mape < 5:   interp = "Stable"
        elif lr_mape < 10: interp = "Moderate"
        elif lr_mape < 20: interp = "High vol"
        else:              interp = "Very High vol"

        acc_rows.append({
            "Commodity":      col,
            "Naïve MAPE":     f"{naive_mape:.1f}%",
            "Naïve DirAcc":   f"{naive_da:.0f}%",
            "Linear MAPE":    f"{lr_mape:.1f}%",
            "Linear DirAcc":  f"{lr_da:.0f}%",
            "12M Error":      f"{error_12m:+.1f}%",
            "Volatility Tier": interp,
            "_naive_mape_val": naive_mape,
            "_lr_mape_val":    lr_mape,
        })

    acc_df = pd.DataFrame(acc_rows)

    # Heatmap-style table
    def _color_mape(val):
        if isinstance(val, str) and val.endswith("%"):
            try:
                v = float(val.replace("%", "").replace("+", "").replace("-", ""))
                if v > 25:  return "background-color: rgba(255,59,48,0.3)"
                if v > 10:  return "background-color: rgba(255,204,0,0.2)"
                if v < 5:   return "background-color: rgba(0,212,170,0.2)"
            except: pass
        return ""

    disp_cols = ["Commodity", "Naïve MAPE", "Naïve DirAcc", "Linear MAPE", "Linear DirAcc", "12M Error", "Volatility Tier"]
    st.dataframe(
        acc_df[disp_cols].style.applymap(_color_mape, subset=["Naïve MAPE", "Linear MAPE"]),
        use_container_width=True, height=460
    )

    # MAPE bar chart
    fig_mape = go.Figure()
    fig_mape.add_bar(x=acc_df["Commodity"], y=acc_df["_naive_mape_val"],
                     name="Naïve MAPE", marker_color="#4a9eff")
    fig_mape.add_bar(x=acc_df["Commodity"], y=acc_df["_lr_mape_val"],
                     name="Linear MAPE", marker_color="#ff6b6b")
    fig_mape.add_hline(y=10, line_dash="dash", line_color="orange",
                       annotation_text="10% threshold")
    fig_mape.update_layout(
        title="12-Month Forecast MAPE by Commodity (lower = more accurate)",
        barmode="group", height=340,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"), xaxis=dict(gridcolor="#2a2a3e"),
        yaxis=dict(gridcolor="#2a2a3e", title="MAPE (%)"),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_mape, use_container_width=True)

    st.divider()

    # ── Section 4: Scenario Financial Impact ──────────────────────────
    st.header("4. Scenario Financial Impact — 12-Month Forward")

    scen_rows = []
    for col in commodity_cols:
        sc = SCENARIO_CONFIG.get(col)
        if not sc:
            continue
        current = float(comm[col].dropna().iloc[-1])
        bw = BOM_WEIGHTS.get(col, 0)
        spend = ANNUAL_MATERIAL_SPEND * bw
        scen_rows.append({
            "Commodity":    col,
            "BOM Weight":   f"{bw*100:.0f}%",
            "Current":      round(current, 0),
            "Bear":         sc["bear"],
            "Base":         sc["base"],
            "Bull":         sc["bull"],
            "Bear COGS $M": round(spend * _pct(sc["bear"], current) / 100 / 1e6, 1),
            "Base COGS $M": round(spend * _pct(sc["base"], current) / 100 / 1e6, 1),
            "Bull COGS $M": round(spend * _pct(sc["bull"], current) / 100 / 1e6, 1),
        })

    scen_df = pd.DataFrame(scen_rows)

    # Waterfall chart for base scenario
    fig_wf = go.Figure(go.Waterfall(
        name="Base Case COGS Impact",
        orientation="v",
        x=scen_df["Commodity"],
        y=scen_df["Base COGS $M"],
        text=[f"${v:+.0f}M" for v in scen_df["Base COGS $M"]],
        textposition="outside",
        connector={"line": {"color": "rgba(100,100,100,0.3)"}},
        increasing={"marker": {"color": "#ff6b6b"}},
        decreasing={"marker": {"color": "#00d4aa"}},
        totals={"marker": {"color": "#4a9eff"}},
    ))
    fig_wf.update_layout(
        title="Base-Case 12M Forward COGS Impact by Commodity ($M vs current prices)",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"), yaxis=dict(title="COGS Change ($M)", gridcolor="#2a2a3e"),
        xaxis=dict(gridcolor="#2a2a3e")
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Grouped bar: Bear/Base/Bull
    fig_scen = go.Figure()
    fig_scen.add_bar(x=scen_df["Commodity"], y=scen_df["Bear COGS $M"],
                     name="🐻 Bear", marker_color="#ff4444")
    fig_scen.add_bar(x=scen_df["Commodity"], y=scen_df["Base COGS $M"],
                     name="➡ Base", marker_color="#4a9eff")
    fig_scen.add_bar(x=scen_df["Commodity"], y=scen_df["Bull COGS $M"],
                     name="🐂 Bull", marker_color="#00d4aa")
    fig_scen.add_hline(y=0, line_color="gray")
    fig_scen.update_layout(
        title="Bear / Base / Bull Scenario COGS Impact ($M)",
        barmode="group", height=380,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"), yaxis=dict(title="COGS Change ($M)", gridcolor="#2a2a3e"),
        xaxis=dict(gridcolor="#2a2a3e"), legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig_scen, use_container_width=True)

    # Summary table
    def _color_impact(val):
        if isinstance(val, (int, float)):
            if val > 50: return "color: #ff6b6b; font-weight: bold"
            if val > 0:  return "color: #ffab40"
            if val < -50: return "color: #00d4aa; font-weight: bold"
            if val < 0:  return "color: #80deea"
        return ""

    st.dataframe(
        scen_df.style.applymap(_color_impact, subset=["Bear COGS $M", "Base COGS $M", "Bull COGS $M"]),
        use_container_width=True, height=460
    )

    # Totals
    total_bear = scen_df["Bear COGS $M"].sum()
    total_base = scen_df["Base COGS $M"].sum()
    total_bull = scen_df["Bull COGS $M"].sum()
    col_b, col_ba, col_bu = st.columns(3)
    col_b.metric("🐻 Bear COGS Total", f"${total_bear:+.0f}M",
                 f"{total_bear / ANNUAL_REVENUE * 100:+.2f}% EBIT", delta_color="inverse")
    col_ba.metric("➡ Base COGS Total", f"${total_base:+.0f}M",
                  f"{total_base / ANNUAL_REVENUE * 100:+.2f}% EBIT", delta_color="inverse")
    col_bu.metric("🐂 Bull COGS Total", f"${total_bull:+.0f}M",
                  f"{total_bull / ANNUAL_REVENUE * 100:+.2f}% EBIT", delta_color="inverse")

    st.divider()

    # ── Section 5: Macro Correlations ─────────────────────────────────
    st.header("5. Macro Driver Correlations")
    st.markdown("Pearson correlation of commodity **monthly returns** vs. key macro variables (7-year history).")

    if macro is not None:
        macro_cols = ["oil_price_usd", "dxy_index", "manufacturing_pmi", "interest_rate_pct", "cpi_index"]
        macro_cols = [c for c in macro_cols if c in macro.columns]

        comm_idx = comm.set_index("date")[commodity_cols]
        macro_idx = macro.set_index("date")[macro_cols]
        aligned = comm_idx.join(macro_idx, how="inner")

        corr_data = {}
        for mc in macro_cols:
            corr_data[mc] = [float(aligned[c].corr(aligned[mc])) for c in commodity_cols]

        corr_df = pd.DataFrame(corr_data, index=commodity_cols)
        labels = {
            "oil_price_usd": "Oil (WTI)", "dxy_index": "USD (DXY)",
            "manufacturing_pmi": "Mfg PMI", "interest_rate_pct": "Interest Rate",
            "cpi_index": "CPI"
        }
        corr_df.columns = [labels.get(c, c) for c in corr_df.columns]

        fig_corr = px.imshow(
            corr_df.T.round(2),
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto",
            text_auto=".2f",
            title="Commodity vs. Macro Variable Correlation Heatmap"
        )
        fig_corr.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0")
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # ── Section 6: Individual Commodity Deep Dive ──────────────────────
    st.header("6. Commodity Deep Dive")

    selected = st.selectbox("Select commodity:", commodity_cols)
    if selected:
        series = comm.set_index("date")[selected].dropna()
        current  = float(series.iloc[-1])
        mean7    = float(series.mean())
        std7     = float(series.std())
        vol      = float(series.pct_change().std() * np.sqrt(12) * 100)
        sc       = SCENARIO_CONFIG.get(selected, {})
        bom_usd  = ANNUAL_MATERIAL_SPEND * BOM_WEIGHTS.get(selected, 0) / 1e6

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Current Price", f"{current:,.1f}", f"{_pct(current, float(series.iloc[-13]) if len(series)>=13 else float(series.iloc[0])):+.1f}% YoY")
        mc2.metric("7yr Mean", f"{mean7:,.1f}", f"{_pct(current, mean7):+.1f}% vs mean")
        mc3.metric("Ann. Volatility", f"{vol:.1f}%", f"±${bom_usd * vol/100:.0f}M COGS uncertainty")
        mc4.metric("Annual Spend", f"${bom_usd:.0f}M", f"{BOM_WEIGHTS.get(selected,0)*100:.0f}% of BOM")

        # Price history chart with scenarios
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines", name="Actual Price",
            line=dict(color="#00d4aa", width=2)
        ))
        fig_price.add_hline(y=mean7, line_dash="dash", line_color="orange",
                            annotation_text=f"7yr Mean: {mean7:,.0f}")
        fig_price.add_hline(y=mean7 + std7, line_dash="dot", line_color="rgba(255,100,100,0.5)")
        fig_price.add_hline(y=mean7 - std7, line_dash="dot", line_color="rgba(0,212,170,0.5)")

        if sc:
            fig_price.add_hline(y=sc["bear"], line_dash="longdash", line_color="#ff4444",
                                annotation_text=f"Bear {sc['bear']:,}")
            fig_price.add_hline(y=sc["base"], line_dash="longdash", line_color="#4a9eff",
                                annotation_text=f"Base {sc['base']:,}")
            fig_price.add_hline(y=sc["bull"], line_dash="longdash", line_color="#00d4aa",
                                annotation_text=f"Bull {sc['bull']:,}")

        fig_price.update_layout(
            title=f"{selected} — Historical Price + Scenario Targets",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"), yaxis=dict(gridcolor="#2a2a3e"),
            xaxis=dict(gridcolor="#2a2a3e"), legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # Monthly returns distribution
        returns = series.pct_change().dropna() * 100
        fig_hist = px.histogram(
            x=returns, nbins=25,
            title=f"{selected} — Monthly Return Distribution",
            labels={"x": "Monthly Return (%)"},
            color_discrete_sequence=["#4a9eff"]
        )
        fig_hist.add_vline(x=0, line_color="white", line_dash="dash")
        fig_hist.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0e0"), yaxis=dict(gridcolor="#2a2a3e"),
            xaxis=dict(gridcolor="#2a2a3e")
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # ── Section 7: Download Full Report ───────────────────────────────
    st.header("7. Download Full Executive Report")

    report_path = _ROOT / "docs" / "EXECUTIVE_INTELLIGENCE_REPORT.md"
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        st.download_button(
            label="📥 Download Full Executive Report (.md)",
            data=report_text,
            file_name=f"GIC_Executive_Intelligence_Report_{pd.Timestamp.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
        )
        st.caption(f"Report size: {report_path.stat().st_size / 1024:.1f} KB  |  "
                   f"Generated: {pd.Timestamp(report_path.stat().st_mtime, unit='s').strftime('%d %b %Y %H:%M')}")
    else:
        st.warning("Report file not found. Click 'Regenerate Full Report' above.")
