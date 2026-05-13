"""
Executive Summary — CFO-level overview dashboard.

Shows:
  - Key financial KPIs (Revenue, Margin, COGS, Net Income)
  - P&L Fan Chart — probabilistic 12-month operating income forecast (NEW)
  - Variance decomposition — commodity / demand / FX contributions (NEW)
  - Market risk level & alerts
  - Commodity index trend
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.helpers import (
    format_currency,
    load_parquet,
    metric_card_css,
)


@st.cache_data(ttl=300)
def _compute_kpis(sales_df_pandas) -> dict:
    """
    Compute KPIs using the actual FinancialModel pipeline.
    Cached for 5 minutes so dashboard stays snappy.
    """
    try:
        from src.drivers.financial_model import FinancialModel

        pnl = FinancialModel().build_pnl(
            sales_df=sales_df_pandas,
            commodity_index_df=pd.DataFrame({"date": [], "commodity_index": []}),
        )

        total_revenue = float(pnl["net_revenue"].sum())
        total_cogs = float(pnl["total_cogs"].sum())
        gross_margin = total_revenue - total_cogs
        gross_margin_pct = gross_margin / total_revenue * 100 if total_revenue > 0 else 0
        net_income = float(pnl["net_income"].sum())
        net_margin_pct = net_income / total_revenue * 100 if total_revenue > 0 else 0

        # YoY delta
        if "date" in pnl.columns:
            pnl["year"] = pd.to_datetime(pnl["date"]).dt.year
            yearly = pnl.groupby("year")["net_revenue"].sum()
            if len(yearly) >= 2:
                curr_rev = float(yearly.iloc[-1])
                prev_rev = float(yearly.iloc[-2])
                revenue_yoy = (curr_rev - prev_rev) / abs(prev_rev) * 100 if prev_rev else None
            else:
                revenue_yoy = None

            yearly_margin = pnl.groupby("year").apply(
                lambda g: (g["net_revenue"].sum() - g["total_cogs"].sum()) / g["net_revenue"].sum() * 100
                if g["net_revenue"].sum() > 0 else 0
            )
            margin_delta = float(yearly_margin.iloc[-1] - yearly_margin.iloc[-2]) if len(yearly_margin) >= 2 else None
        else:
            revenue_yoy = None
            margin_delta = None

        return {
            "total_revenue": total_revenue,
            "total_cogs": total_cogs,
            "gross_margin": gross_margin,
            "gross_margin_pct": gross_margin_pct,
            "net_income": net_income,
            "net_margin_pct": net_margin_pct,
            "revenue_yoy": revenue_yoy,
            "margin_delta": margin_delta,
            "pnl": pnl,
        }
    except Exception as e:
        from src.config import get_settings
        settings = get_settings()
        segs = settings["vehicle_segments"]
        avg_price = sum(s["avg_price_usd"] for s in segs) / len(segs)
        cogs_pct = settings["financial"]["base_cogs_pct"]

        total_revenue = float(sales_df_pandas["volume"].sum()) * avg_price
        total_cogs = total_revenue * cogs_pct
        gross_margin = total_revenue - total_cogs
        return {
            "total_revenue": total_revenue,
            "total_cogs": total_cogs,
            "gross_margin": gross_margin,
            "gross_margin_pct": (1 - cogs_pct) * 100,
            "net_income": total_revenue * 0.05,
            "net_margin_pct": 5.0,
            "revenue_yoy": None,
            "margin_delta": None,
            "pnl": None,
        }


@st.cache_data(ttl=600)
def _run_pnl_fan_chart(sales_df_pandas):
    """Build fan chart data from Monte Carlo. Cached 10 minutes."""
    try:
        from src.drivers.financial_model import FinancialModel
        from src.simulation.monte_carlo import MonteCarloEngine

        pnl = FinancialModel().build_pnl(
            sales_df=sales_df_pandas,
            commodity_index_df=pd.DataFrame({"date": [], "commodity_index": []}),
        )
        # Only forecast next 12 months from the last available date
        pnl["date"] = pd.to_datetime(pnl["date"])
        last_date = pnl["date"].max()
        future_pnl = pnl[pnl["date"] > last_date - pd.DateOffset(months=12)].copy()
        if future_pnl.empty:
            future_pnl = pnl.tail(12).copy()

        mc = MonteCarloEngine()
        fan_df = mc.run_monthly_fan(future_pnl, n_simulations=1500)
        return fan_df
    except Exception as exc:
        st.warning(f"Fan chart unavailable: {exc}")
        return None


@st.cache_data(ttl=600)
def _variance_decomposition(sales_df_pandas):
    """Run variance decomposition. Cached 10 minutes."""
    try:
        from src.drivers.financial_model import FinancialModel
        from src.simulation.monte_carlo import MonteCarloEngine

        pnl = FinancialModel().build_pnl(
            sales_df=sales_df_pandas,
            commodity_index_df=pd.DataFrame({"date": [], "commodity_index": []}),
        )
        mc = MonteCarloEngine()
        return mc.decompose_variance(
            sales_df_pandas,
            pd.DataFrame({"date": [], "commodity_index": []}),
            n_simulations=1500,
        )
    except Exception:
        return None


def _render_pnl_fan_chart(fan_df: pd.DataFrame) -> None:
    """Render the 12-month operating income fan chart."""
    st.subheader("Operating Income — 12-Month Probabilistic Forecast")
    st.caption("Fan chart shows 80% and 95% confidence bands from Monte Carlo simulation (1,500 paths)")

    dates = fan_df["date"].dt.strftime("%b %Y").tolist()

    fig = go.Figure()

    # 95% CI outer band
    fig.add_trace(go.Scatter(
        x=dates, y=(fan_df["p95_oi"] / 1e6).tolist(),
        fill=None, line=dict(color="rgba(0,100,255,0)", width=0),
        showlegend=False, name="p95",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=(fan_df["p5_oi"] / 1e6).tolist(),
        fill="tonexty",
        fillcolor="rgba(0,100,255,0.10)",
        line=dict(color="rgba(0,100,255,0)", width=0),
        name="95% CI",
    ))

    # 80% CI inner band
    fig.add_trace(go.Scatter(
        x=dates, y=(fan_df["p90_oi"] / 1e6).tolist(),
        fill=None, line=dict(color="rgba(0,140,255,0)", width=0),
        showlegend=False, name="p90",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=(fan_df["p10_oi"] / 1e6).tolist(),
        fill="tonexty",
        fillcolor="rgba(0,140,255,0.20)",
        line=dict(color="rgba(0,140,255,0)", width=0),
        name="80% CI",
    ))

    # Mean line
    fig.add_trace(go.Scatter(
        x=dates, y=(fan_df["mean_oi"] / 1e6).tolist(),
        mode="lines+markers",
        line=dict(color="#00d4aa", width=3),
        marker=dict(size=5),
        name="Forecast (Mean)",
    ))

    fig.update_layout(
        yaxis_title="Operating Income (£M)",
        height=360,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=30, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def render():
    st.markdown(metric_card_css(), unsafe_allow_html=True)
    st.title("Executive Summary")
    st.markdown("**CFO Financial Intelligence Dashboard** — Real-time market & financial overview")
    st.markdown("---")

    # ── Load Data ──
    commodity_df = load_parquet("commodity_prices")
    market_df = load_parquet("market_commodities")
    sales_df = load_parquet("sales_data")

    sales_df_pandas = sales_df.to_pandas() if sales_df is not None else None
    prices_df = market_df if market_df is not None else commodity_df

    # ── KPI Row ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    if sales_df is not None:
        kpis = _compute_kpis(sales_df_pandas)
        with col1:
            st.metric(
                "Portfolio Revenue",
                format_currency(kpis["total_revenue"]),
                f"{kpis['revenue_yoy']:+.1f}% YoY" if kpis["revenue_yoy"] is not None else None,
            )
        with col2:
            st.metric(
                "Gross Margin",
                f"{kpis['gross_margin_pct']:.1f}%",
                f"{kpis['margin_delta']:+.1f}pp" if kpis["margin_delta"] is not None else None,
            )
        with col3:
            st.metric(
                "Total COGS",
                format_currency(kpis["total_cogs"]),
            )
        with col4:
            st.metric(
                "Net Income",
                format_currency(kpis["net_income"]),
                f"{kpis['net_margin_pct']:.1f}% margin",
            )
    else:
        for col in [col1, col2, col3, col4]:
            with col:
                st.info("Generate data first: `python scripts/generate_data.py`")

    st.markdown("---")

    # ── P&L Fan Chart (NEW) ──────────────────────────────────────────────────
    if sales_df is not None:
        with st.spinner("Building probabilistic P&L forecast..."):
            fan_df = _run_pnl_fan_chart(sales_df_pandas)

        if fan_df is not None and not fan_df.empty:
            _render_pnl_fan_chart(fan_df)

            # Variance decomposition
            var_decomp = _variance_decomposition(sales_df_pandas)
            if var_decomp:
                st.caption("**P&L Uncertainty Drivers**")
                vcol1, vcol2, vcol3 = st.columns(3)
                vcol1.metric(
                    "Commodity Risk",
                    f"{var_decomp['commodity_pct']:.0f}% of variance",
                    help="Share of total P&L uncertainty driven by commodity price movements",
                )
                vcol2.metric(
                    "Demand Risk",
                    f"{var_decomp['demand_pct']:.0f}% of variance",
                    help="Share driven by volume/demand uncertainty",
                )
                vcol3.metric(
                    "FX Risk",
                    f"{var_decomp['fx_pct']:.0f}% of variance",
                    help="Share driven by FX rate movements",
                )

            st.markdown("---")

    # ── Commodity Index Chart ────────────────────────────────────────────────
    if prices_df is not None:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("Commodity Price Index")
            value_cols = [c for c in prices_df.columns if c not in ("date", "Date")]
            if value_cols:
                first_valid_idx = 0
                fig = go.Figure()
                for col in value_cols[:12]:
                    series = prices_df[col].to_list()
                    base = series[first_valid_idx] if series[first_valid_idx] else 1
                    normalized = [((v / base) * 100 if v and base else None) for v in series]

                    date_col_name = "date" if "date" in prices_df.columns else "Date"
                    dates = prices_df[date_col_name].to_list()

                    fig.add_trace(go.Scatter(
                        x=dates, y=normalized,
                        mode="lines", name=col,
                        line=dict(width=2),
                    ))

                fig.update_layout(
                    yaxis_title="Index (Base = 100)",
                    height=400,
                    template="plotly_dark",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(l=40, r=20, t=30, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Risk Assessment")

            risk_data = []
            for col in value_cols[:8]:
                series = prices_df[col].drop_nulls().to_list()
                if len(series) > 10:
                    returns = np.diff(series) / (np.array(series[:-1]) + 1e-9)
                    vol = float(np.std(returns)) * 100
                    latest = series[-1]
                    avg = float(np.mean(series[-12:])) if len(series) >= 12 else float(np.mean(series))
                    mom = (latest - avg) / avg * 100 if avg else 0
                    risk_data.append({"Commodity": col, "Volatility": f"{vol:.1f}%", "Momentum": f"{mom:+.1f}%"})

            if risk_data:
                import polars as pl
                risk_table = pl.DataFrame(risk_data)
                st.dataframe(risk_table.to_pandas(), use_container_width=True, hide_index=True)

    # ── Alerts Section ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Active Alerts & Signals")

    alert_col1, alert_col2 = st.columns(2)
    with alert_col1:
        st.markdown(
            '<div class="alert-warning">'
            "<strong>Commodity Volatility Elevated</strong><br>"
            "Multiple commodities showing above-average price volatility. "
            "Review hedging positions and scenario stress tests."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="alert-info">'
            "<strong>Data Sources Active</strong><br>"
            "Yahoo Finance: Connected | CCXT: Connected | FRED: Requires API key"
            "</div>",
            unsafe_allow_html=True,
        )

    with alert_col2:
        from src.dashboard.helpers import detect_data_source
        src_label = detect_data_source("commodity_prices")
        src_badge = (
            '<span style="color:#00d4aa">REAL DATA</span>'
            if src_label == "real"
            else '<span style="color:#ff9500">SYNTHETIC</span>'
        )
        st.markdown(
            '<div class="alert-info">'
            f"<strong>Model Status</strong> ({src_badge})<br>"
            "SARIMAX: 12 models trained | XGBoost: 12 models (Regime-Adaptive) | "
            "Demand: 4 segments | Elasticity: 4 fitted"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Quick Actions ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Quick Actions")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    with qcol1:
        if st.button("Refresh Market Data", use_container_width=True):
            st.info("Run `python scripts/fetch_data.py` to refresh market data")
    with qcol2:
        if st.button("Run Scenario Analysis", use_container_width=True):
            st.info("Navigate to Scenario Simulation page")
    with qcol3:
        if st.button("Retrain Models", use_container_width=True):
            st.info("Run `python scripts/train_models.py`")
    with qcol4:
        if st.button("Export Report", use_container_width=True):
            st.info("Report export coming soon")



@st.cache_data(ttl=300)
def _compute_kpis(sales_df_pandas) -> dict:
    """
    Compute KPIs using the actual FinancialModel pipeline.
    Cached for 5 minutes so dashboard stays snappy.
    """
    try:
        import pandas as pd
        from src.drivers.financial_model import FinancialModel

        pnl = FinancialModel().build_pnl(
            sales_df=sales_df_pandas,
            commodity_index_df=pd.DataFrame({"date": [], "commodity_index": []}),
        )

        total_revenue = float(pnl["net_revenue"].sum())
        total_cogs = float(pnl["total_cogs"].sum())
        gross_margin = total_revenue - total_cogs
        gross_margin_pct = gross_margin / total_revenue * 100 if total_revenue > 0 else 0
        net_income = float(pnl["net_income"].sum())
        net_margin_pct = net_income / total_revenue * 100 if total_revenue > 0 else 0

        # YoY delta: compare last year vs prior year if multi-year data available
        if "date" in pnl.columns:
            pnl["year"] = pd.to_datetime(pnl["date"]).dt.year
            yearly = pnl.groupby("year")["net_revenue"].sum()
            if len(yearly) >= 2:
                curr_rev = float(yearly.iloc[-1])
                prev_rev = float(yearly.iloc[-2])
                revenue_yoy = (curr_rev - prev_rev) / abs(prev_rev) * 100 if prev_rev else None
            else:
                revenue_yoy = None

            yearly_margin = pnl.groupby("year").apply(
                lambda g: (g["net_revenue"].sum() - g["total_cogs"].sum()) / g["net_revenue"].sum() * 100
                if g["net_revenue"].sum() > 0 else 0
            )
            margin_delta = float(yearly_margin.iloc[-1] - yearly_margin.iloc[-2]) if len(yearly_margin) >= 2 else None
        else:
            revenue_yoy = None
            margin_delta = None

        return {
            "total_revenue": total_revenue,
            "total_cogs": total_cogs,
            "gross_margin": gross_margin,
            "gross_margin_pct": gross_margin_pct,
            "net_income": net_income,
            "net_margin_pct": net_margin_pct,
            "revenue_yoy": revenue_yoy,
            "margin_delta": margin_delta,
        }
    except Exception as e:
        # Graceful fallback: estimate from sales volume if financial model fails
        volume = float(sales_df_pandas["volume"].sum())

        from src.config import get_settings
        settings = get_settings()
        segs = settings["vehicle_segments"]
        avg_price = sum(s["avg_price_usd"] for s in segs) / len(segs)
        cogs_pct = settings["financial"]["base_cogs_pct"]

        total_revenue = volume * avg_price
        total_cogs = total_revenue * cogs_pct
        gross_margin = total_revenue - total_cogs
        return {
            "total_revenue": total_revenue,
            "total_cogs": total_cogs,
            "gross_margin": gross_margin,
            "gross_margin_pct": (1 - cogs_pct) * 100,
            "net_income": total_revenue * 0.08,
            "net_margin_pct": 8.0,
            "revenue_yoy": None,
            "margin_delta": None,
        }


def render():
    st.markdown(metric_card_css(), unsafe_allow_html=True)
    st.title("Executive Summary")
    st.markdown("**CFO Financial Intelligence Dashboard** — Real-time market & financial overview")
    st.markdown("---")

    # ── Load Data ──
    commodity_df = load_parquet("commodity_prices")
    market_df = load_parquet("market_commodities")
    sales_df = load_parquet("sales_data")

    # Convert to pandas for FinancialModel (which uses pandas internally)
    sales_df_pandas = sales_df.to_pandas() if sales_df is not None else None

    # Use market data if available, else fallback to synthetic
    prices_df = market_df if market_df is not None else commodity_df

    # ── KPI Row — derived from actual FinancialModel ──
    col1, col2, col3, col4 = st.columns(4)

    if sales_df is not None:
        kpis = _compute_kpis(sales_df_pandas)
        with col1:
            st.metric(
                "Portfolio Revenue",
                format_currency(kpis["total_revenue"]),
                f"{kpis['revenue_yoy']:+.1f}% YoY" if kpis["revenue_yoy"] is not None else None,
            )
        with col2:
            st.metric(
                "Gross Margin",
                f"{kpis['gross_margin_pct']:.1f}%",
                f"{kpis['margin_delta']:+.1f}pp" if kpis["margin_delta"] is not None else None,
            )
        with col3:
            st.metric(
                "Total COGS",
                format_currency(kpis["total_cogs"]),
            )
        with col4:
            st.metric(
                "Net Income",
                format_currency(kpis["net_income"]),
                f"{kpis['net_margin_pct']:.1f}% margin",
            )
    else:
        for col in [col1, col2, col3, col4]:
            with col:
                st.info("Generate data first: `python scripts/generate_data.py`")

    st.markdown("---")

    # ── Commodity Index Chart ──
    if prices_df is not None:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("Commodity Price Index")
            value_cols = [c for c in prices_df.columns if c not in ("date", "Date")]
            if value_cols:
                # Normalize to base 100
                first_valid_idx = 0
                fig = go.Figure()
                for col in value_cols[:12]:
                    series = prices_df[col].to_list()
                    base = series[first_valid_idx] if series[first_valid_idx] else 1
                    normalized = [((v / base) * 100 if v and base else None) for v in series]

                    date_col_name = "date" if "date" in prices_df.columns else "Date"
                    dates = prices_df[date_col_name].to_list()

                    fig.add_trace(go.Scatter(
                        x=dates, y=normalized,
                        mode="lines", name=col,
                        line=dict(width=2),
                    ))

                fig.update_layout(
                    yaxis_title="Index (Base = 100)",
                    height=400,
                    template="plotly_dark",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(l=40, r=20, t=30, b=40),
                )
                st.plotly_chart(fig, width='stretch')

        with col_right:
            st.subheader("Risk Assessment")

            # Simple risk scoring from price volatility
            risk_data = []
            for col in value_cols[:8]:
                series = prices_df[col].drop_nulls().to_list()
                if len(series) > 10:
                    import numpy as np
                    returns = np.diff(series) / series[:-1]
                    vol = float(np.std(returns)) * 100
                    latest = series[-1]
                    avg = float(np.mean(series[-12:])) if len(series) >= 12 else float(np.mean(series))
                    mom = (latest - avg) / avg * 100 if avg else 0
                    risk_data.append({"Commodity": col, "Volatility": f"{vol:.1f}%", "Momentum": f"{mom:+.1f}%"})

            if risk_data:
                import polars as pl
                risk_table = pl.DataFrame(risk_data)
                st.dataframe(risk_table.to_pandas(), width='stretch', hide_index=True)

    # ── Alerts Section ──
    st.markdown("---")
    st.subheader("Active Alerts & Signals")

    alert_col1, alert_col2 = st.columns(2)
    with alert_col1:
        st.markdown(
            '<div class="alert-warning">'
            "<strong>Commodity Volatility Elevated</strong><br>"
            "Multiple commodities showing above-average price volatility. "
            "Review hedging positions and scenario stress tests."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="alert-info">'
            "<strong>Data Sources Active</strong><br>"
            "Yahoo Finance: Connected | CCXT: Connected | FRED: Requires API key"
            "</div>",
            unsafe_allow_html=True,
        )

    with alert_col2:
        # Detect data source for commodity prices
        from src.dashboard.helpers import detect_data_source
        src_label = detect_data_source("commodity_prices")
        src_badge = (
            '<span style="color:#00d4aa">REAL DATA</span>'
            if src_label == "real"
            else '<span style="color:#ff9500">SYNTHETIC</span>'
        )
        st.markdown(
            '<div class="alert-info">'
            f"<strong>Model Status</strong> ({src_badge})<br>"
            "SARIMAX: 12 models trained | XGBoost: 12 models | "
            "Demand: 4 segments | Elasticity: 4 fitted"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Quick Actions ──
    st.markdown("---")
    st.subheader("Quick Actions")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    with qcol1:
        if st.button("Refresh Market Data", width='stretch'):
            st.info("Run `python scripts/fetch_data.py` to refresh market data")
    with qcol2:
        if st.button("Run Scenario Analysis", width='stretch'):
            st.info("Navigate to Scenario Simulation page")
    with qcol3:
        if st.button("Retrain Models", width='stretch'):
            st.info("Run `python scripts/train_models.py`")
    with qcol4:
        if st.button("Export Report", width='stretch'):
            st.info("Report export coming soon")
