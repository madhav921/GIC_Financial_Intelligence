

"""
Commodity Intelligence — deep-dive into commodity price analytics.

Shows:
  - Individual commodity price charts (real-world data)
  - SARIMAX & XGBoost forecast comparison
  - Correlation heatmap across commodity basket
  - FFN performance analytics (Sharpe, drawdown, CAGR)
  - BOM-weighted commodity index
  - [NEW] Live Shock Calculator — commodity sliders → animated P&L waterfall
  - [NEW] Hedge Optimizer — optimal hedge ratio per commodity
  - [NEW] Regime Classification — Hurst exponent market state detection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
import streamlit as st

from src.config import get_settings
from src.dashboard.helpers import format_currency, load_parquet


@st.cache_data(ttl=300)
def _get_base_revenue() -> float:
    """Derive base annual revenue from financial model."""
    try:
        from src.drivers.financial_model import FinancialModel
        sales_df = load_parquet("sales_data")
        if sales_df is None:
            raise ValueError("No sales data")
        pnl = FinancialModel().build_pnl(
            sales_df=sales_df.to_pandas(),
            commodity_index_df=pd.DataFrame({"date": [], "commodity_index": []}),
        )
        return float(pnl["net_revenue"].sum())
    except Exception:
        settings = get_settings()
        return sum(s["avg_price_usd"] * s["annual_volume"] for s in settings["vehicle_segments"])


def render():
    st.title("Commodity Intelligence")
    st.markdown("**Real-time commodity price tracking, forecasts, regime detection & risk analytics**")
    st.markdown("---")

    # ── Load Data ──
    market_df = load_parquet("market_commodities")
    synthetic_df = load_parquet("commodity_prices")
    prices_df = market_df if market_df is not None else synthetic_df

    if prices_df is None:
        st.error("No commodity data available. Run `python scripts/fetch_data.py` or `python scripts/generate_data.py`.")
        return

    from src.dashboard.helpers import detect_data_source
    src = detect_data_source("commodity_prices")
    if src == "real":
        st.success("Data Source: **Yahoo Finance** (real-world prices, some ETF proxies)")
    else:
        st.info("Data Source: **Synthetic** — run `python scripts/fetch_data.py` for real data")

    date_col = "date" if "date" in prices_df.columns else "Date"
    value_cols = [c for c in prices_df.columns if c not in ("date", "Date")]

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Price Charts", "Correlation Matrix", "Performance Analytics",
        "Commodity Index", "⚡ Live Shock Calculator", "🎯 Regime & Hedge"
    ])

    with tab1:
        _render_price_charts(prices_df, date_col, value_cols)

    with tab2:
        _render_correlation(prices_df, date_col, value_cols)

    with tab3:
        _render_performance(prices_df, date_col, value_cols)

    with tab4:
        _render_commodity_index(prices_df, date_col, value_cols)

    with tab5:
        _render_shock_calculator(prices_df, date_col, value_cols)

    with tab6:
        _render_regime_and_hedge(prices_df, date_col, value_cols)


# ─── NEW: Live Shock Calculator ──────────────────────────────────────────────


def _render_shock_calculator(
    prices_df: pl.DataFrame, date_col: str, value_cols: list[str]
) -> None:
    """Commodity price sliders → animated P&L waterfall. The core demo feature."""
    from src.models.commodity_shock import CommodityShockCalculator
    from src.models.hedge_optimizer import HedgeOptimizer

    st.subheader("Commodity → P&L Impact Calculator")
    st.markdown(
        "Drag sliders to simulate commodity price shocks. "
        "The waterfall shows the **exact EBIT impact** in real time."
    )

    settings = get_settings()
    commodities = settings["commodities"]
    base_revenue = _get_base_revenue()
    calc = CommodityShockCalculator()

    # ── Sliders ──────────────────────────────────────────────────────────────
    st.markdown("**Set commodity price shocks (%)**")
    shocks: dict[str, float] = {}
    cols = st.columns(4)
    for i, commodity in enumerate(commodities):
        name = commodity["name"]
        with cols[i % 4]:
            shock = st.slider(
                f"{name}",
                min_value=-50,
                max_value=50,
                value=0,
                step=1,
                key=f"shock_{name}",
                format="%d%%",
            )
            shocks[name] = shock / 100.0

    # ── Compute waterfall ────────────────────────────────────────────────────
    waterfall_data = calc.waterfall(shocks, base_revenue)

    if waterfall_data:
        total_ebit_impact = sum(d["ebit_impact"] for d in waterfall_data)
        total_cogs_impact = sum(d["cogs_impact"] for d in waterfall_data)

        # KPI metrics
        k1, k2, k3 = st.columns(3)
        k1.metric(
            "Total EBIT Impact",
            f"£{total_ebit_impact / 1e6:+.1f}M",
            delta_color="inverse",
        )
        k2.metric(
            "Total COGS Impact",
            f"£{total_cogs_impact / 1e6:+.1f}M",
            delta_color="inverse",
        )
        k3.metric(
            "As % of EBIT",
            f"{sum(d['pct_of_base_ebit'] for d in waterfall_data):+.1f}%",
            delta_color="inverse",
        )

        # ── Waterfall chart ───────────────────────────────────────────────────
        labels = [d["commodity"] for d in waterfall_data] + ["Total EBIT Impact"]
        values = [d["ebit_impact"] / 1e6 for d in waterfall_data] + [total_ebit_impact / 1e6]
        measures = ["relative"] * len(waterfall_data) + ["total"]

        fig = go.Figure(go.Waterfall(
            name="EBIT Impact (£M)",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            texttemplate="%{y:+.1f}M",
            textposition="outside",
            connector={"line": {"color": "rgba(150,150,150,0.4)"}},
            decreasing={"marker": {"color": "#ff3b30"}},
            increasing={"marker": {"color": "#30d158"}},
            totals={"marker": {"color": "#00d4aa"}},
        ))
        fig.update_layout(
            title="P&L Impact Waterfall (£M) — After-Tax EBIT",
            yaxis_title="EBIT Impact (£M)",
            height=420,
            template="plotly_dark",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Hedge recommendations for top 3 impacted commodities ────────────
        st.subheader("Hedge Recommendations (Top 3 Commodities by Impact)")
        optimizer = HedgeOptimizer()

        for d in waterfall_data[:3]:
            commodity_name = d["commodity"]
            bom_weight = calc.bom_weights.get(calc._resolve_key(commodity_name), 0.01)
            material_exposure = (
                base_revenue
                * settings["financial"]["base_cogs_pct"]
                * settings["financial"]["material_cogs_fraction"]
                * bom_weight
            )

            forecast_std = abs(d["shock_pct"]) * 0.5 if abs(d["shock_pct"]) > 0 else 0.05
            hedge = optimizer.optimize(
                forecast_mean=1.0,
                forecast_std=forecast_std,
                futures_price=1.0 * (1 + d["shock_pct"] * 0.5),
                exposure_units=material_exposure,
            )
            col_a, col_b, col_c = st.columns(3)
            col_a.metric(
                f"{commodity_name} — Hedge Ratio",
                f"{hedge['optimal_hedge_ratio'] * 100:.0f}%",
            )
            col_b.metric(
                "Expected Savings",
                f"£{hedge['expected_savings'] / 1e6:.1f}M",
            )
            col_c.metric(
                "VaR Reduction",
                f"£{hedge['var_reduction'] / 1e6:.1f}M",
            )
    else:
        st.info("Move at least one slider to see the P&L impact waterfall")


# ─── NEW: Regime Classification & Hedge Schedule ─────────────────────────────


def _render_regime_and_hedge(
    prices_df: pl.DataFrame, date_col: str, value_cols: list[str]
) -> None:
    """Market regime classification table + hedge schedule."""
    from src.models.regime_detector import RegimeDetector

    st.subheader("Market Regime Classification")
    st.markdown(
        "Hurst exponent analysis classifies each commodity's current price dynamic. "
        "The ensemble model weights are adjusted automatically based on regime."
    )

    detector = RegimeDetector()
    settings = get_settings()
    pdf = prices_df.to_pandas()

    # ── Regime table ─────────────────────────────────────────────────────────
    regime_rows = []
    for col in value_cols:
        series_raw = pdf[col].dropna().values
        if len(series_raw) < 12:
            continue
        result = detector.detect(series_raw)
        regime_label = result["regime"].value.replace("_", " ").title()
        dominant_model = max(
            result["ensemble_weights"], key=result["ensemble_weights"].get
        ).title()
        regime_rows.append({
            "Commodity": col,
            "Regime": regime_label,
            "Hurst (H)": f"{result['hurst']:.3f}",
            "Vol (12M)": f"{result['rolling_vol_pct']:.1f}%",
            "Confidence": result["confidence"].title(),
            "Dominant Model": dominant_model,
            "SARIMAX w": f"{result['ensemble_weights']['sarimax']:.0%}",
            "XGBoost w": f"{result['ensemble_weights']['xgboost']:.0%}",
        })

    if regime_rows:
        regime_df = pd.DataFrame(regime_rows)

        def _color_regime(row: pd.Series):
            if "Mean" in row["Regime"]:
                return ["background-color: rgba(0,200,100,0.15)"] * len(row)
            elif "Trending" in row["Regime"]:
                return ["background-color: rgba(255,200,0,0.15)"] * len(row)
            else:
                return ["background-color: rgba(255,60,60,0.15)"] * len(row)

        st.dataframe(
            regime_df.style.apply(_color_regime, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # ── Regime distribution pie ───────────────────────────────────────────
        st.subheader("Regime Distribution")
        regime_counts = regime_df["Regime"].value_counts()
        fig = go.Figure(go.Pie(
            labels=regime_counts.index.tolist(),
            values=regime_counts.values.tolist(),
            marker=dict(colors=["#30d158", "#ffd60a", "#ff453a"]),
            hole=0.4,
        ))
        fig.update_layout(height=280, template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough price history to classify regimes (need ≥ 12 periods)")

    # ── Hurst Exponent Bar Chart ──────────────────────────────────────────────
    if regime_rows:
        st.subheader("Hurst Exponent by Commodity")
        hurst_vals = [float(r["Hurst (H)"]) for r in regime_rows]
        commodities_list = [r["Commodity"] for r in regime_rows]
        colors = [
            "#30d158" if h < 0.45 else "#ffd60a" if h > 0.55 else "#ff453a"
            for h in hurst_vals
        ]
        fig2 = go.Figure(go.Bar(
            x=commodities_list,
            y=hurst_vals,
            marker_color=colors,
            text=[f"{h:.3f}" for h in hurst_vals],
            textposition="outside",
        ))
        fig2.add_hline(y=0.45, line_dash="dash", line_color="rgba(100,200,100,0.5)",
                       annotation_text="Mean-reverting threshold (H=0.45)")
        fig2.add_hline(y=0.55, line_dash="dash", line_color="rgba(255,200,0,0.5)",
                       annotation_text="Trending threshold (H=0.55)")
        fig2.update_layout(
            yaxis_title="Hurst Exponent",
            yaxis_range=[0, 1],
            height=350,
            template="plotly_dark",
        )
        st.plotly_chart(fig2, use_container_width=True)


# ─── Existing chart functions (unchanged) ────────────────────────────────────


def _render_price_charts(df: pl.DataFrame, date_col: str, value_cols: list[str]):
    """Individual commodity price charts with technical indicators."""
    st.subheader("Commodity Price Tracker")

    col1, col2 = st.columns([1, 3])
    with col1:
        selected = st.selectbox("Select Commodity", value_cols, index=0)
        show_ma = st.checkbox("Show Moving Averages", value=True)
        show_bands = st.checkbox("Show Bollinger Bands", value=False)

    with col2:
        series = df.select(pl.col(date_col), pl.col(selected)).drop_nulls()
        dates = series[date_col].to_list()
        prices = series[selected].to_list()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode="lines", name=selected,
            line=dict(color="#00d4aa", width=2),
        ))

        if show_ma and len(prices) > 20:
            ma20 = _moving_average(prices, 20)
            ma50 = _moving_average(prices, 50) if len(prices) > 50 else None

            fig.add_trace(go.Scatter(
                x=dates, y=ma20,
                mode="lines", name="MA(20)",
                line=dict(color="#ff9500", width=1, dash="dash"),
            ))
            if ma50 is not None:
                fig.add_trace(go.Scatter(
                    x=dates, y=ma50,
                    mode="lines", name="MA(50)",
                    line=dict(color="#ff3b30", width=1, dash="dash"),
                ))

        if show_bands and len(prices) > 20:
            upper, lower = _bollinger_bands(prices, 20)
            fig.add_trace(go.Scatter(
                x=dates, y=upper,
                mode="lines", name="Upper Band",
                line=dict(color="rgba(255,255,255,0.2)", width=1),
            ))
            fig.add_trace(go.Scatter(
                x=dates, y=lower,
                mode="lines", name="Lower Band",
                line=dict(color="rgba(255,255,255,0.2)", width=1),
                fill="tonexty", fillcolor="rgba(100,100,255,0.05)",
            ))

        fig.update_layout(
            title=f"{selected} — Price History",
            yaxis_title="Price",
            height=450,
            template="plotly_dark",
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    if len(prices) > 0:
        scol1, scol2, scol3, scol4, scol5 = st.columns(5)
        current = prices[-1]
        high = max(prices)
        low = min(prices)
        avg = sum(prices) / len(prices)
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1]]
        vol = float(np.std(returns)) * 100 if returns else 0

        scol1.metric("Current", format_currency(current))
        scol2.metric("52W High", format_currency(high))
        scol3.metric("52W Low", format_currency(low))
        scol4.metric("Average", format_currency(avg))
        scol5.metric("Volatility", f"{vol:.1f}%")


def _render_correlation(df: pl.DataFrame, date_col: str, value_cols: list[str]):
    """Correlation heatmap across commodities."""
    st.subheader("Commodity Correlation Matrix")
    st.markdown("*Returns-based correlation — identifies diversification opportunities and co-movement risks*")

    pdf = df.to_pandas()
    numeric_cols = [c for c in value_cols if c in pdf.columns]
    returns = pdf[numeric_cols].pct_change().dropna()

    if returns.empty or len(returns) < 5:
        st.warning("Not enough data to compute correlations")
        return

    corr = returns.corr()

    fig = px.imshow(
        corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
    )
    fig.update_layout(height=500, template="plotly_dark", title="Return Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Key Correlations:**")
    high_corr = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            val = corr.iloc[i, j]
            if abs(val) > 0.6:
                high_corr.append((corr.index[i], corr.columns[j], val))

    if high_corr:
        for a, b, v in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:5]:
            direction = "positively" if v > 0 else "negatively"
            st.markdown(f"- **{a}** & **{b}**: {v:.2f} ({direction} correlated)")
    else:
        st.info("No strong correlations detected (all < 0.6)")


def _render_performance(df: pl.DataFrame, date_col: str, value_cols: list[str]):
    """FFN-style performance analytics."""
    st.subheader("Performance Analytics")

    pdf = df.to_pandas()
    numeric_cols = [c for c in value_cols if c in pdf.columns]

    if not numeric_cols:
        st.warning("No numeric columns for analysis")
        return

    records = []
    for col in numeric_cols:
        series = pdf[col].dropna()
        if len(series) < 10:
            continue

        total_return = (series.iloc[-1] / series.iloc[0] - 1) * 100
        returns = series.pct_change().dropna()
        ann_vol = float(returns.std() * np.sqrt(12) * 100)
        sharpe = float(returns.mean() / returns.std() * np.sqrt(12)) if returns.std() > 0 else 0

        cummax = series.cummax()
        drawdown = ((series - cummax) / cummax * 100)
        max_dd = float(drawdown.min())

        records.append({
            "Commodity": col,
            "Total Return": f"{total_return:.1f}%",
            "Ann. Volatility": f"{ann_vol:.1f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.1f}%",
            "Current Price": f"${series.iloc[-1]:,.2f}",
        })

    if records:
        perf_df = pl.DataFrame(records)
        st.dataframe(perf_df.to_pandas(), use_container_width=True, hide_index=True)

    st.subheader("Drawdown Analysis")
    selected_dd = st.multiselect("Select commodities", numeric_cols, default=numeric_cols[:3])

    if selected_dd:
        fig = go.Figure()
        for col in selected_dd:
            series = pdf[col].dropna()
            cummax = series.cummax()
            dd = (series - cummax) / cummax * 100
            fig.add_trace(go.Scatter(x=pdf[date_col], y=dd, mode="lines", name=col, fill="tozeroy"))

        fig.update_layout(
            yaxis_title="Drawdown (%)", height=350, template="plotly_dark",
            margin=dict(l=40, r=20, t=30, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_commodity_index(df: pl.DataFrame, date_col: str, value_cols: list[str]):
    """BOM-weighted commodity index."""
    st.subheader("BOM-Weighted Commodity Index")
    st.markdown("*Composite index weighted by Bill of Materials cost impact*")

    settings = get_settings()
    weights = {}
    for c in settings["commodities"]:
        weights[c["name"]] = c["bom_weight"]

    pdf = df.to_pandas()
    available_commodities = [c for c in value_cols if c in weights]

    if not available_commodities:
        st.warning("No matching commodities found in BOM configuration")
        return

    normalized = {}
    for col in available_commodities:
        series = pdf[col].dropna()
        if len(series) > 0:
            base = series.iloc[0]
            normalized[col] = (series / base) * 100

    total_weight = sum(weights.get(c, 0) for c in available_commodities)
    if total_weight <= 0:
        st.warning("Total BOM weight is zero — check settings.yaml commodities")
        return

    index_series = sum(
        normalized[c] * weights.get(c, 0) / total_weight
        for c in available_commodities if c in normalized
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pdf[date_col], y=index_series,
        mode="lines", name="Commodity Index",
        line=dict(color="#00d4aa", width=3),
        fill="tozeroy", fillcolor="rgba(0, 212, 170, 0.1)",
    ))
    fig.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Base = 100")
    fig.update_layout(yaxis_title="Index Value", height=400, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    weight_data = [{"Commodity": c, "BOM Weight": f"{weights[c]:.0%}"} for c in available_commodities]
    st.dataframe(pd.DataFrame(weight_data), use_container_width=True, hide_index=True)


def _moving_average(values: list[float], window: int) -> list[float | None]:
    result = [None] * len(values)
    for i in range(window - 1, len(values)):
        result[i] = sum(values[i - window + 1:i + 1]) / window
    return result


def _bollinger_bands(values: list[float], window: int, num_std: float = 2.0):
    upper = [None] * len(values)
    lower = [None] * len(values)
    for i in range(window - 1, len(values)):
        segment = values[i - window + 1:i + 1]
        mean = sum(segment) / window
        std = (sum((x - mean) ** 2 for x in segment) / window) ** 0.5
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std
    return upper, lower
