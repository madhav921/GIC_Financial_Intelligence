"""
Financial P&L — revenue, COGS, margins, and financial impact analysis.

Shows:
  - Monthly/Annual P&L waterfall
  - Segment-level revenue & margin analysis
  - Commodity cost impact on COGS
  - Trend analysis and YoY comparison
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src.dashboard.helpers import format_currency, load_parquet


def render():
    st.title("Financial P&L Analysis")
    st.markdown("**Deterministic financial model — Revenue, COGS, Margins & Net Income**")
    st.markdown("---")

    # ── Load Data ──
    sales_df = load_parquet("sales_data")
    commodity_df = load_parquet("commodity_prices")

    if sales_df is None:
        st.error("No sales data available. Run `python scripts/generate_data.py`.")
        return

    pdf = sales_df.to_pandas()
    if "date" not in pdf.columns:
        st.error("Sales data missing 'date' column")
        return

    pdf["date"] = __import__("pandas").to_datetime(pdf["date"])

    tab1, tab2, tab3 = st.tabs(["P&L Overview", "Segment Analysis", "Cost Impact"])

    with tab1:
        _render_pnl_overview(pdf)
    with tab2:
        _render_segment_analysis(pdf)
    with tab3:
        _render_cost_impact(pdf, commodity_df)


def _render_pnl_overview(pdf):
    """P&L waterfall and monthly trends."""
    st.subheader("Profit & Loss Overview")

    from src.config import get_settings
    settings = get_settings()
    cogs_pct = settings["financial"]["base_cogs_pct"]
    warranty_pct = settings["financial"]["warranty_reserve_pct"]
    tax_rate = settings["financial"]["tax_rate"]

    # Compute financials
    pdf["net_revenue"] = pdf["volume"] * pdf["avg_price_usd"] * (1 - pdf["incentive_pct"])
    pdf["cogs"] = pdf["net_revenue"] * cogs_pct
    pdf["gross_margin"] = pdf["net_revenue"] - pdf["cogs"]
    pdf["warranty"] = pdf["net_revenue"] * warranty_pct
    pdf["depreciation"] = 8_000_000  # Monthly approximation
    pdf["operating_income"] = pdf["gross_margin"] - pdf["warranty"] - pdf["depreciation"]
    pdf["tax"] = np.where(pdf["operating_income"] > 0, pdf["operating_income"] * tax_rate, 0)
    pdf["net_income"] = pdf["operating_income"] - pdf["tax"]

    # KPI row
    total_rev = pdf["net_revenue"].sum()
    total_cogs = pdf["cogs"].sum()
    total_gm = pdf["gross_margin"].sum()
    total_ni = pdf["net_income"].sum()

    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    kcol1.metric("Total Revenue", format_currency(total_rev))
    kcol2.metric("COGS", format_currency(total_cogs))
    kcol3.metric("Gross Margin", f"{total_gm / total_rev * 100:.1f}%")
    kcol4.metric("Net Income", format_currency(total_ni))

    st.markdown("---")

    # P&L Waterfall
    st.subheader("P&L Waterfall (Annual)")
    waterfall_data = {
        "Revenue": total_rev,
        "COGS": -total_cogs,
        "Gross Margin": total_gm,
        "Warranty": -pdf["warranty"].sum(),
        "Depreciation": -pdf["depreciation"].sum(),
        "Operating Income": pdf["operating_income"].sum(),
        "Tax": -pdf["tax"].sum(),
        "Net Income": total_ni,
    }

    fig = go.Figure(go.Waterfall(
        x=list(waterfall_data.keys()),
        y=list(waterfall_data.values()),
        measure=["relative", "relative", "total", "relative", "relative", "total", "relative", "total"],
        connector=dict(line=dict(color="rgba(100,100,100,0.3)")),
        increasing=dict(marker=dict(color="#00d4aa")),
        decreasing=dict(marker=dict(color="#ff6b6b")),
        totals=dict(marker=dict(color="#007aff")),
    ))
    fig.update_layout(
        height=400,
        template="plotly_dark",
        yaxis_title="USD",
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Monthly revenue trend
    st.subheader("Monthly Revenue & Margin Trend")
    monthly = pdf.groupby(pdf["date"].dt.to_period("M")).agg({
        "net_revenue": "sum",
        "gross_margin": "sum",
        "net_income": "sum",
    }).reset_index()
    monthly["date"] = monthly["date"].dt.to_timestamp()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["net_revenue"],
        mode="lines+markers", name="Revenue",
        line=dict(color="#00d4aa", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["gross_margin"],
        mode="lines+markers", name="Gross Margin",
        line=dict(color="#007aff", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["net_income"],
        mode="lines+markers", name="Net Income",
        line=dict(color="#ff9500", width=2),
    ))

    fig.update_layout(
        height=400,
        template="plotly_dark",
        yaxis_title="USD",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_segment_analysis(pdf):
    """Revenue and margin analysis by vehicle segment."""
    st.subheader("Segment Revenue Breakdown")

    pdf["net_revenue"] = pdf["volume"] * pdf["avg_price_usd"] * (1 - pdf["incentive_pct"])

    if "segment" not in pdf.columns:
        st.warning("No segment data available")
        return

    seg_data = pdf.groupby("segment").agg({
        "volume": "sum",
        "net_revenue": "sum",
    }).reset_index()
    seg_data["avg_price"] = seg_data["net_revenue"] / seg_data["volume"]

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            seg_data, values="net_revenue", names="segment",
            title="Revenue by Segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            seg_data, x="segment", y="volume",
            title="Volume by Segment",
            color="segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Segment detail table
    st.subheader("Segment Financial Summary")
    seg_data["Revenue"] = seg_data["net_revenue"].apply(format_currency)
    seg_data["Volume"] = seg_data["volume"].apply(lambda x: f"{x:,}")
    seg_data["Avg Price"] = seg_data["avg_price"].apply(lambda x: format_currency(x))
    display_cols = ["segment", "Volume", "Revenue", "Avg Price"]
    st.dataframe(seg_data[display_cols], use_container_width=True, hide_index=True)


def _render_cost_impact(pdf, commodity_df):
    """Commodity cost impact on COGS."""
    st.subheader("Commodity Cost Impact on COGS")
    st.markdown("*How commodity price movements affect Cost of Goods Sold*")

    if commodity_df is None:
        st.warning("No commodity data for cost impact analysis")
        return

    from src.config import get_settings
    settings = get_settings()
    material_fraction = 0.45  # 45% of COGS is raw materials

    # Show BOM impact
    bom_weights = {c["name"]: c["bom_weight"] for c in settings["commodities"]}

    # Simulate COGS sensitivity
    shock_range = list(range(-30, 35, 5))
    base_cogs = float(pdf["volume"].sum() * pdf["avg_price_usd"].mean() * settings["financial"]["base_cogs_pct"])

    sensitivity_data = []
    for shock in shock_range:
        adjusted_cogs = base_cogs * (1 + shock / 100 * material_fraction)
        margin_impact = base_cogs - adjusted_cogs
        sensitivity_data.append({
            "Commodity Shock (%)": shock,
            "COGS Impact ($)": adjusted_cogs,
            "Margin Impact ($)": margin_impact,
        })

    import pandas as pd
    sens_df = pd.DataFrame(sensitivity_data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sens_df["Commodity Shock (%)"],
        y=sens_df["Margin Impact ($)"],
        marker_color=["#00d4aa" if v >= 0 else "#ff6b6b" for v in sens_df["Margin Impact ($)"]],
    ))
    fig.update_layout(
        title="Gross Margin Sensitivity to Commodity Price Changes",
        xaxis_title="Commodity Price Shock (%)",
        yaxis_title="Margin Impact (USD)",
        height=400,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)
