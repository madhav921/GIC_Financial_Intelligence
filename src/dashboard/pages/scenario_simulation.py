"""
Scenario Simulation — Monte Carlo & what-if analysis.

Shows:
  - Monte Carlo simulation distributions
  - Preset scenario comparison
  - Custom what-if parameter controls
  - VaR / CVaR risk metrics
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src.dashboard.helpers import format_currency, load_parquet


@st.cache_data(ttl=300)
def _get_base_revenue() -> float:
    """
    Derive base revenue from actual sales + financial model.
    Falls back to config-based estimate if data not available.
    """
    try:
        import pandas as pd
        from src.drivers.financial_model import FinancialModel

        sales_df = load_parquet("sales_data")
        if sales_df is None:
            raise ValueError("No sales data available")

        pnl = FinancialModel().build_pnl(
            sales_df=sales_df.to_pandas(),
            commodity_index_df=pd.DataFrame({"date": [], "commodity_index": []}),
        )
        return float(pnl["net_revenue"].sum())
    except Exception:
        from src.config import get_settings
        settings = get_settings()
        # Estimate from config defaults
        avg_price = settings["revenue"]["avg_vehicle_price"]
        total_volume = sum(
            seg.get("base_volume", 0) * 12
            for seg in settings["revenue"]["segments"]
        )
        return avg_price * total_volume if total_volume > 0 else 25_000_000_000


def render():
    st.title("Scenario Simulation")
    st.markdown("**Monte Carlo simulation & what-if analysis for strategic planning**")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Monte Carlo", "What-If Analysis", "Scenario Comparison"])

    with tab1:
        _render_monte_carlo()
    with tab2:
        _render_what_if()
    with tab3:
        _render_scenario_comparison()


def _render_monte_carlo():
    """Monte Carlo simulation with interactive parameters."""
    st.subheader("Monte Carlo Simulation Engine")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Simulation Parameters**")
        n_sims = st.slider("Number of Simulations", 1000, 50000, 10000, 1000)
        demand_vol = st.slider("Demand Volatility (%)", 1, 30, 10) / 100
        commodity_vol = st.slider("Commodity Volatility (%)", 5, 50, 20) / 100
        demand_mean = st.slider("Demand Bias (%)", -20, 20, 0) / 100
        commodity_mean = st.slider("Commodity Bias (%)", -30, 30, 0) / 100
        use_fat_tails = st.checkbox("Fat-Tailed Distribution (t-dist)", value=True)

        run_sim = st.button("Run Simulation", type="primary", use_container_width=True)

    with col2:
        if run_sim:
            with st.spinner("Running Monte Carlo simulation..."):
                rng = np.random.default_rng(42)

                # Generate demand & commodity shocks
                demand_shocks = rng.normal(demand_mean, demand_vol, n_sims)
                if use_fat_tails:
                    commodity_shocks = rng.standard_t(df=5, size=n_sims) * commodity_vol + commodity_mean
                else:
                    commodity_shocks = rng.normal(commodity_mean, commodity_vol, n_sims)

                # Base financials — derived from actual financial model if data available
                base_revenue = _get_base_revenue()
                from src.config import get_settings
                _settings = get_settings()
                base_cogs_pct = _settings["financial"]["base_cogs_pct"]
                material_fraction = 0.45

                revenue_dist = base_revenue * (1 + demand_shocks)
                base_cogs = base_revenue * base_cogs_pct
                cogs_dist = base_cogs * (1 + demand_shocks) + base_cogs * material_fraction * commodity_shocks
                margin_dist = revenue_dist - cogs_dist
                margin_pct_dist = margin_dist / revenue_dist * 100

                # Display results
                st.markdown("### Simulation Results")

                # KPIs
                kcol1, kcol2, kcol3, kcol4 = st.columns(4)
                kcol1.metric("Mean Margin", format_currency(np.mean(margin_dist)))
                kcol2.metric("Median Margin", format_currency(np.median(margin_dist)))
                kcol3.metric("VaR (95%)", format_currency(np.percentile(margin_dist, 5)))
                kcol4.metric("CVaR (95%)", format_currency(np.mean(margin_dist[margin_dist <= np.percentile(margin_dist, 5)])))

                # Distribution chart
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=margin_dist / 1e9,
                    nbinsx=100,
                    marker_color="#00d4aa",
                    opacity=0.7,
                    name="Gross Margin",
                ))

                # Add VaR line
                var_95 = np.percentile(margin_dist, 5) / 1e9
                fig.add_vline(
                    x=var_95, line_dash="dash", line_color="#ff3b30",
                    annotation_text=f"VaR(95%) = ${var_95:.2f}B",
                )
                fig.add_vline(
                    x=np.mean(margin_dist) / 1e9, line_dash="dash", line_color="#007aff",
                    annotation_text=f"Mean = ${np.mean(margin_dist) / 1e9:.2f}B",
                )

                fig.update_layout(
                    title=f"Gross Margin Distribution ({n_sims:,} simulations)",
                    xaxis_title="Gross Margin ($B)",
                    yaxis_title="Frequency",
                    height=450,
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Percentile table
                st.markdown("### Percentile Analysis")
                percentiles = [5, 10, 25, 50, 75, 90, 95]
                perc_data = []
                for p in percentiles:
                    val = np.percentile(margin_dist, p)
                    perc_data.append({
                        "Percentile": f"P{p}",
                        "Gross Margin": format_currency(val),
                        "Margin %": f"{np.percentile(margin_pct_dist, p):.1f}%",
                        "Revenue": format_currency(np.percentile(revenue_dist, p)),
                    })
                st.dataframe(
                    __import__("pandas").DataFrame(perc_data),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("Configure parameters and click **Run Simulation** to generate results")


def _render_what_if():
    """Custom what-if scenario builder."""
    st.subheader("What-If Scenario Builder")
    st.markdown("*Adjust key drivers and see real-time P&L impact*")

    from src.config import get_settings
    settings = get_settings()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Demand Drivers**")
        demand_change = st.slider("Volume Change (%)", -30, 30, 0, key="wif_demand")
        price_change = st.slider("Price Change (%)", -20, 20, 0, key="wif_price")
        incentive_change = st.slider("Incentive Change (pp)", -3, 5, 0, key="wif_incentive")

    with col2:
        st.markdown("**Cost Drivers**")
        commodity_shock = st.slider("Commodity Price Shock (%)", -40, 60, 0, key="wif_commodity")
        labor_change = st.slider("Labor Cost Change (%)", -10, 20, 0, key="wif_labor")
        fx_change = st.slider("FX Rate Change (%)", -15, 15, 0, key="wif_fx")

    # Compute scenario — base revenue from actual financial model
    base_revenue = _get_base_revenue()
    base_cogs_pct = settings["financial"]["base_cogs_pct"]
    warranty_pct = settings["financial"]["warranty_reserve_pct"]
    tax_rate = settings["financial"]["tax_rate"]
    material_fraction = 0.45
    labor_fraction = 0.30

    adj_revenue = base_revenue * (1 + demand_change / 100) * (1 + price_change / 100)
    adj_revenue *= (1 - incentive_change / 100)

    adj_cogs = adj_revenue * base_cogs_pct
    adj_cogs *= (1 + commodity_shock / 100 * material_fraction)
    adj_cogs *= (1 + labor_change / 100 * labor_fraction)
    adj_cogs *= (1 + fx_change / 100 * 0.15)  # 15% FX exposure

    gross_margin = adj_revenue - adj_cogs
    warranty = adj_revenue * warranty_pct
    operating_income = gross_margin - warranty - 96_000_000
    tax = max(0, operating_income * tax_rate)
    net_income = operating_income - tax

    base_margin = base_revenue * (1 - base_cogs_pct)

    st.markdown("---")
    st.markdown("### Scenario Impact")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    rev_delta = (adj_revenue - base_revenue) / base_revenue * 100
    margin_delta = (gross_margin - base_margin) / base_margin * 100

    mcol1.metric("Revenue", format_currency(adj_revenue), f"{rev_delta:+.1f}%")
    mcol2.metric("COGS", format_currency(adj_cogs))
    mcol3.metric("Gross Margin", f"{gross_margin / adj_revenue * 100:.1f}%", f"{margin_delta:+.1f}%")
    mcol4.metric("Net Income", format_currency(net_income))

    # Waterfall
    fig = go.Figure(go.Waterfall(
        x=["Base Revenue", "Volume", "Price", "Incentive", "COGS Adj", "FX & Labor", "Net Margin"],
        y=[
            base_revenue,
            base_revenue * demand_change / 100,
            base_revenue * price_change / 100,
            -base_revenue * incentive_change / 100,
            -(adj_cogs - base_revenue * base_cogs_pct),
            -(adj_cogs * 0.15 * fx_change / 100 + adj_cogs * labor_fraction * labor_change / 100),
            gross_margin,
        ],
        measure=["absolute", "relative", "relative", "relative", "relative", "relative", "total"],
        increasing=dict(marker=dict(color="#00d4aa")),
        decreasing=dict(marker=dict(color="#ff6b6b")),
        totals=dict(marker=dict(color="#007aff")),
    ))
    fig.update_layout(height=400, template="plotly_dark", title="Scenario Waterfall")
    st.plotly_chart(fig, use_container_width=True)


def _render_scenario_comparison():
    """Compare preset scenarios."""
    st.subheader("Preset Scenario Comparison")

    from src.config import get_settings
    settings = get_settings()
    presets = settings["simulation"]["scenario_presets"]

    base_revenue = _get_base_revenue()
    base_cogs_pct = settings["financial"]["base_cogs_pct"]
    material_fraction = 0.45

    records = []
    for name, params in presets.items():
        demand_s = params.get("demand_shock", 0)
        commodity_s = params.get("commodity_shock", 0)
        fx_s = params.get("fx_shock", 0)

        rev = base_revenue * (1 + demand_s)
        cogs = rev * base_cogs_pct * (1 + commodity_s * material_fraction)
        margin = rev - cogs
        margin_pct = margin / rev * 100 if rev else 0

        records.append({
            "Scenario": name.replace("_", " ").title(),
            "Demand Shock": f"{demand_s:+.0%}",
            "Commodity Shock": f"{commodity_s:+.0%}",
            "FX Shock": f"{fx_s:+.0%}",
            "Revenue": format_currency(rev),
            "Margin %": f"{margin_pct:.1f}%",
            "Margin $": format_currency(margin),
        })

    import pandas as pd
    st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

    # Chart
    margin_values = []
    scenario_names = []
    for name, params in presets.items():
        demand_s = params.get("demand_shock", 0)
        commodity_s = params.get("commodity_shock", 0)
        rev = base_revenue * (1 + demand_s)
        cogs = rev * base_cogs_pct * (1 + commodity_s * material_fraction)
        margin_values.append((rev - cogs) / 1e9)
        scenario_names.append(name.replace("_", " ").title())

    fig = go.Figure(go.Bar(
        x=scenario_names,
        y=margin_values,
        marker_color=["#007aff" if v > 0 else "#ff6b6b" for v in margin_values],
    ))
    fig.update_layout(
        title="Gross Margin by Scenario ($B)",
        yaxis_title="Gross Margin ($B)",
        height=400,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)
